"""The single file that turns a bare Colab runtime into a machine that can
train a cell of this project's grid.

`colab exec -f` ships one file's *source* to a remote kernel. This project is
not one file -- it is a package (src/data, src/models, src/training) plus a
processed dataset that git does not carry, since data/raw, data/processed and
runs/ are all gitignored. So the one file that gets sent has to be a bootstrap
that assembles everything else on the far side before it trains anything:
fetch the code, install the dependencies without breaking the accelerator,
stage the data, run the cell, and leave the results somewhere the driver can
pull them from.

Everything here is idempotent and safe to re-run after a teardown. That is not
tidiness, it is the operating condition: a free-tier runtime caps at 12h and is
reclaimed after 90 minutes idle, so a long cell *will* be interrupted, and the
recovery path is "run the same command again". Training is invoked with
--resume and an optional --max-seconds budget so a cell can be run as a
sequence of bounded segments, each one ending with its artefacts on disk and a
resumable state behind it (see src/training/checkpoint.py).

Two traps this file exists to avoid
-----------------------------------
The first is the jax clobber. requirements.txt lists bare `jax` and `jaxlib`,
which is right for a CPU laptop and actively destructive on a Colab GPU or TPU
VM: pip would replace the preinstalled CUDA/TPU build with CPU wheels, and the
run would complete, report plausible numbers, and have measured nothing. The
whole point of moving to Colab is the accelerator, so those two lines are
filtered out and the backend is re-checked afterwards.

The second is that a run whose code version is unknown is not reproducible.
The resolved commit SHA is printed and recorded in STATUS.json.

Invocation
----------
Under `colab run`, which forwards argv and sets __name__ == "__main__":

    colab run --gpu T4 --timeout 3600 scripts/colab_bootstrap.py \
        --sets FIN --arm bdh --width 64 --steps 3000

Under `colab exec -f`, where the driver prepends a `sys.argv = [...]` line to a
copy of this file. Note the CLI's --timeout defaults to **30 seconds** on both
`run` and `exec`; anything that trains needs it raised explicitly or the call
returns long before the work does.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/roro2006/MTG-bdh-architecture-and-scaling"
DEFAULT_REF = "origin/main"
S3_TEMPLATE = (
    "https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/"
    "draft_data_public.{set_code}.PremierDraft.csv.gz"
)

# src/training/run.py returns this when --max-seconds cut the run short and a
# resume state was left behind. It is distinct from 1 so the driver never
# mistakes a genuine crash for "there is more to do"; propagated verbatim.
EXIT_INCOMPLETE = 75

# What a staged set has to contain before staging can be skipped. PickData.load
# reads the first two and src/training/run.py reads the third, so a directory
# missing any of them is not usable even though it looks populated.
PROCESSED_MARKERS = ("picks.npz", "vocab.json", "card_features.npz")

# Dropped from requirements.txt on the VM -- see the module docstring.
ACCELERATOR_OWNED = {"jax", "jaxlib"}

# Mirrors RESUME_PARAMS_FILE / RESUME_STATE_FILE in src/training/checkpoint.py.
# Duplicated rather than imported on purpose: importing that module would pull
# jax and flax into this process, which is meant to stay stdlib-only and must
# work before anything is installed. If those names ever change, this list is
# the thing to update -- the only cost of drift is mirroring two extra files.
RESUME_FILES = ("resume.msgpack", "resume.json")


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------

_PROBE = """
import json
try:
    import jax, jaxlib
    print(json.dumps({
        "ok": True,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "device_kind": getattr(jax.devices()[0], "device_kind", "?"),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
    }))
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
"""


def probe_backend() -> dict:
    """Reports what jax can see, from a *fresh* interpreter.

    Deliberately a subprocess rather than an import. This process may run the
    probe twice -- once before installing dependencies and once after -- and a
    process that has already imported jax would keep reporting the version it
    imported first, which is exactly the clobber the second probe is there to
    detect. It also keeps jax out of this module entirely, so the file stays
    stdlib-only and the training subprocess gets a clean import.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except (ValueError, IndexError):
        return {
            "ok": False,
            "error": f"probe failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}",
        }


def report_backend(label: str, info: dict) -> None:
    if not info.get("ok"):
        print(f"[{label}] jax unavailable: {info.get('error')}", flush=True)
        return
    print(
        f"[{label}] backend={info['backend']} devices={info['devices']} "
        f"kind={info['device_kind']} jax={info['jax']} jaxlib={info['jaxlib']}",
        flush=True,
    )


def require_accelerator(info: dict, allow_cpu: bool, when: str) -> None:
    """Fails loudly on CPU, because a CPU run here is silently pointless.

    Everything in this project has run CPU-only so far; that is the blocker
    the Colab work exists to remove. A run that quietly falls back to CPU
    would still produce a metrics.json and a loss curve, and the only symptom
    would be a wall-clock number nobody checked against the reference.
    """
    if not info.get("ok"):
        raise SystemExit(f"jax is not importable {when}: {info.get('error')}")
    if info["backend"] in ("gpu", "tpu"):
        return
    if allow_cpu:
        print(
            f"  WARNING: backend is {info['backend']} {when}; continuing only "
            "because --allow-cpu was passed. Timings from this run are not "
            "comparable with accelerator runs.",
            flush=True,
        )
        return
    raise SystemExit(
        f"backend is {info['backend']} {when}, not gpu/tpu.\n"
        "  This runtime has no accelerator, so the run would measure nothing.\n"
        "  Provision with `colab new --gpu T4` or `--tpu v5e1` (note: the TPU\n"
        "  flag values are v5e1 and v6e1, not v5e), or pass --allow-cpu to\n"
        "  override deliberately."
    )


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------


def sync_repo(root: Path, ref: str) -> tuple[Path, str]:
    """Clones, or fast-forwards an existing checkout, to `ref`.

    Hard reset rather than pull: the checkout on a reused runtime is
    disposable, and a merge conflict here would strand the session in a state
    no one is watching. Anything not pushed does not exist on this VM.
    """
    repo = root / "MTG-bdh-architecture-and-scaling"
    if not (repo / ".git").is_dir():
        print(f"cloning {REPO_URL} -> {repo}", flush=True)
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", REPO_URL, str(repo)], check=True
        )
    else:
        print(f"updating existing checkout at {repo}", flush=True)

    # Full clone and a plain fetch, deliberately. A --depth 1 clone would be
    # faster, but it pins the checkout to one branch tip and then cannot resolve
    # --ref when it names a tag or a bare SHA -- which is exactly what anyone
    # reproducing an earlier run will pass. The repository carries code only
    # (data/ and runs/ are gitignored), so the whole history is small.
    _git(repo, "fetch", "--quiet", "origin")
    _git(repo, "reset", "--hard", "--quiet", ref)

    commit = _git(repo, "rev-parse", "HEAD").strip()
    subject = _git(repo, "log", "-1", "--pretty=%s").strip()
    print(f"  at {commit[:12]}  {subject}", flush=True)
    return repo, commit


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}"
        )
    return proc.stdout


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------


def install_requirements(repo: Path) -> list[str]:
    """Installs requirements.txt minus anything the runtime already owns.

    Returns the list of skipped lines so STATUS.json records what was left
    alone. See the module docstring for why this filtering is load-bearing
    rather than an optimisation.
    """
    path = repo / "requirements.txt"
    wanted, skipped = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("[")[0]
        for sep in ("==", ">=", "<=", "~=", ">", "<", "!="):
            name = name.split(sep)[0]
        (skipped if name.strip().lower() in ACCELERATOR_OWNED else wanted).append(line)

    if skipped:
        print(
            f"  keeping the runtime's own {', '.join(skipped)} -- installing the "
            "generic wheels would replace the accelerated build with CPU ones",
            flush=True,
        )
    print(f"  installing: {', '.join(wanted)}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *wanted], check=True
    )
    return skipped


# --------------------------------------------------------------------------
# Data staging
# --------------------------------------------------------------------------
#
# Cache backend: Drive, not GCS.
#
# Ingesting one set takes ~283s and 215MB of download, and a runtime is
# ephemeral, so paying that on every session is the single largest avoidable
# cost in this loop. Both candidate caches are Google-side, and the deciding
# difference is setup rather than speed:
#
#   Drive is the same Google identity the Colab CLI already authenticated as.
#   It needs no GCP project, no billing account and no bucket, and the free
#   15GB comfortably holds all ten processed sets (~565MB total).
#
#   GCS needs a project with billing enabled, a bucket, and `colab auth` to
#   put credentials on the VM -- and `colab auth`, like `colab drivemount`,
#   requires a TTY and cannot be driven by a script.
#
# So GCS costs strictly more to set up and buys nothing here. Drive still costs
# one interactive `colab drivemount` per session, which is why caching is
# opt-in via --cache-dir and the default path is download-and-ingest: the
# default keeps the acceptance criterion of no manual browser steps beyond the
# initial OAuth, and --cache-dir is the accelerator for anyone willing to mount
# once per session.


def processed_dir_for(repo: Path, set_code: str) -> Path:
    return repo / "data" / "processed" / f"{set_code}.PremierDraft"


def is_staged(processed: Path) -> bool:
    return all((processed / name).exists() for name in PROCESSED_MARKERS)


def stage_set(repo: Path, set_code: str, cache_dir: Path | None) -> Path:
    """Ensures data/processed/<SET>.PremierDraft is populated. Idempotent."""
    processed = processed_dir_for(repo, set_code)
    if is_staged(processed):
        print(f"[data] {set_code}: already staged at {processed}", flush=True)
        return processed

    cached = (cache_dir / f"{set_code}.PremierDraft") if cache_dir else None
    if cached is not None and is_staged(cached):
        print(f"[data] {set_code}: restoring from cache {cached}", flush=True)
        processed.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cached, processed, dirs_exist_ok=True)
        return processed

    raw = repo / "data" / "raw" / f"draft_data_public.{set_code}.PremierDraft.csv.gz"
    if not raw.exists():
        _download(S3_TEMPLATE.format(set_code=set_code), raw)

    print(f"[data] {set_code}: ingesting (this is the ~283s step)", flush=True)
    _stream(
        [sys.executable, "-m", "src.data.ingest",
         "--csv", str(raw), "--out", str(processed)],
        cwd=repo,
    )
    print(f"[data] {set_code}: fetching Scryfall card features", flush=True)
    _stream(
        [sys.executable, "-m", "src.data.card_features",
         "--processed-dir", str(processed)],
        cwd=repo,
    )

    if not is_staged(processed):
        missing = [n for n in PROCESSED_MARKERS if not (processed / n).exists()]
        raise SystemExit(f"staging {set_code} left {missing} missing in {processed}")

    if cached is not None:
        print(f"[data] {set_code}: populating cache at {cached}", flush=True)
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(processed, cached, dirs_exist_ok=True)

    # The raw export is ~206MB for FIN and is not needed again once the
    # processed arrays exist. A Colab VM's disk is not large enough to keep
    # several sets' raw exports around alongside everything else.
    raw.unlink(missing_ok=True)
    return processed


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[data] downloading {url}", flush=True)
    started = time.monotonic()
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = next_mark = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            done += len(chunk)
            if total and done * 100 // total >= next_mark:
                print(
                    f"    {done / 1e6:,.0f}MB / {total / 1e6:,.0f}MB "
                    f"({done * 100 // total}%)",
                    flush=True,
                )
                next_mark += 10
    tmp.replace(dest)
    print(
        f"    done: {dest.stat().st_size / 1e6:,.0f}MB in "
        f"{time.monotonic() - started:,.0f}s",
        flush=True,
    )


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def build_training_argv(args: argparse.Namespace, processed: Path, out_dir: Path) -> list[str]:
    """Maps this script's flags onto src/training/run.py's.

    Every flag run.py accepts is reachable from here. A flag that could only
    be set by editing the bootstrap would mean grid cells that cannot be
    reproduced from a single command, which is the property run.py's own
    docstring is protecting.
    """
    argv = [
        sys.executable, "-m", "src.training.run",
        "--processed-dir", str(processed),
        "--out-dir", str(out_dir),
        "--arm", args.arm,
        "--width", str(args.width),
        "--arm-layers", str(args.arm_layers),
        "--pool-layers", str(args.pool_layers),
        "--pack-layers", str(args.pack_layers),
        "--num-heads", str(args.num_heads),
        "--neuron-multiplier", str(args.neuron_multiplier),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--eval-every", str(args.eval_every),
        "--seed", str(args.seed),
        "--data-fraction", str(args.data_fraction),
        "--eval-batch-size", str(args.eval_batch_size),
    ]
    if args.resume:
        # Default on: a no-op with no state present, and with state present the
        # difference between continuing a segmented run and silently
        # restarting it from step 1.
        argv.append("--resume")
    if args.epochs is not None:
        argv += ["--epochs", str(args.epochs)]
    if args.max_seconds is not None:
        argv += ["--max-seconds", str(args.max_seconds)]
    if args.fused_kernels:
        argv.append("--fused-kernels")
    if args.skip_full_eval:
        argv.append("--skip-full-eval")
    return argv


def _stream(argv: list[str], cwd: Path) -> int:
    """Runs a subprocess with its output forwarded line by line, live.

    Capturing and printing at the end would be simpler and wrong: a training
    segment runs for tens of minutes and the per-step progress is the only
    signal that it is alive. PYTHONUNBUFFERED stops the child buffering its
    stdout into a pipe and delivering it all at exit.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        argv, cwd=str(cwd), env=env, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip("\n"), flush=True)
    return proc.wait()


# --------------------------------------------------------------------------
# Artefacts
# --------------------------------------------------------------------------


def mirror_artefacts(
    out_dir: Path, artefact_dir: Path, include_resume: bool
) -> list[str]:
    """Copies the run's output where the driver knows to look for it.

    Copied rather than written there directly so that the training code keeps
    its ordinary runs/<cell>/ layout and nothing downstream has to know a
    Colab-specific path existed.

    The resume state is excluded by default. It is the largest thing in the
    directory -- parameters, both Adam moments and the best-so-far copy, so
    roughly four times params.msgpack -- and pulling it after every segment
    would dominate the transfer for something only useful if the VM is lost
    mid-run. --mirror-resume turns it on for a cell long enough to be worth
    insuring.
    """
    artefact_dir.mkdir(parents=True, exist_ok=True)
    resume_files = set(RESUME_FILES)

    written: list[str] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".tmp"):
            continue  # a half-written atomic replace; never a valid artefact
        if path.name in resume_files and not include_resume:
            continue
        relative = path.relative_to(out_dir)
        target = artefact_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        written.append(str(relative))
    return written


def read_progress(out_dir: Path) -> tuple[int | None, int | None]:
    """Step reached and total, from whichever of the two run.py writes."""
    for name, step_key in (("progress.json", "stopped_at_step"), ("metrics.json", "steps")):
        path = out_dir / name
        if not path.exists():
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if name == "metrics.json":
            return blob.get("steps"), blob.get("steps")
        return blob.get(step_key), blob.get("total_steps")
    return None, None


# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a Colab runtime and train one cell.",
    )
    parser.add_argument("--sets", nargs="+", default=["FIN"],
                        help="17lands set codes to stage. Staging more than one "
                             "is only useful for warming a --cache-dir.")
    parser.add_argument("--train-set", default=None,
                        help="which staged set to train on (default: the first "
                             "of --sets)")
    parser.add_argument("--ref", default=DEFAULT_REF,
                        help="git ref to check out (default: %(default)s)")
    parser.add_argument("--work-root", default="/content",
                        help="where the checkout lives on the VM")
    parser.add_argument("--out-root", default="runs",
                        help="run directory root, relative to the checkout")
    parser.add_argument("--artefact-root", default="/content/artifacts",
                        help="where the driver looks for results to download")
    parser.add_argument("--run-name", default=None,
                        help="artefact directory name. Defaults to a name built "
                             "from arm/width/set/steps; pass it explicitly when "
                             "using --epochs, since the default reads --steps.")
    parser.add_argument("--cache-dir", default=None,
                        help="optional processed-data cache, e.g. a mounted "
                             "/content/drive/MyDrive/mtg-cache. Off by default; "
                             "see the note above stage_set().")
    parser.add_argument("--mirror-resume", action="store_true",
                        help="also copy the resume state into the artefact dir")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="proceed on a CPU runtime instead of failing")
    parser.add_argument("--skip-install", action="store_true",
                        help="assume dependencies are present (a re-invocation "
                             "on a warm session)")
    parser.add_argument("--stage-only", action="store_true",
                        help="stage data and exit without training")

    # Passed through to src/training/run.py; defaults mirror its own.
    parser.add_argument("--arm", default="attention", choices=["attention", "bdh"])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--arm-layers", type=int, default=2)
    parser.add_argument("--pool-layers", type=int, default=2)
    parser.add_argument("--pack-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--neuron-multiplier", type=int, default=4)
    parser.add_argument("--fused-kernels", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--skip-full-eval", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="segment budget handed to run.py. On hitting it the "
                             f"run exits {EXIT_INCOMPLETE} with a resumable state, "
                             "and this script exits the same way.")
    # Resuming is the default because it is a no-op with no state present and
    # the difference between continuing and silently restarting when there is.
    # --resume is still accepted explicitly: the driver passes it on every
    # segment after the first, and an argparse error there would break the
    # segment loop on its second iteration -- after a provision has been spent
    # and the first segment's work is already on the VM.
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="continue an interrupted run (the default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="ignore any resume state and train from step 1. "
                             "Note this does not delete the state; run.py will "
                             "refuse to overwrite a run configured differently.")
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def default_run_name(args: argparse.Namespace, set_code: str) -> str:
    short = {"attention": "attn", "bdh": "bdh"}[args.arm]
    name = f"{short}_d{args.width}_{set_code}_s{args.steps}"
    if args.fused_kernels:
        name += "_fused"
    if args.data_fraction < 1.0:
        name += f"_f{args.data_fraction:g}"
    if args.seed:
        name += f"_seed{args.seed}"
    return name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()

    print("=" * 72, flush=True)
    print("colab_bootstrap: preparing runtime", flush=True)
    print("=" * 72, flush=True)

    # 1. Hardware, before anything can have disturbed it.
    before = probe_backend()
    report_backend("hw", before)
    require_accelerator(before, args.allow_cpu, "on arrival")

    # 2. Code.
    repo, commit = sync_repo(Path(args.work_root), args.ref)

    # 3. Dependencies.
    skipped: list[str] = []
    if args.skip_install:
        print("[deps] skipped (--skip-install)", flush=True)
    else:
        print("[deps] installing requirements.txt", flush=True)
        skipped = install_requirements(repo)

    after = probe_backend()
    report_backend("hw", after)
    require_accelerator(after, args.allow_cpu, "after installing dependencies")
    if before.get("ok") and after.get("ok") and before["backend"] != after["backend"]:
        raise SystemExit(
            f"backend regressed from {before['backend']} to {after['backend']} "
            "while installing dependencies -- something pulled in generic jax "
            "wheels over the runtime's accelerated build"
        )

    # 4. Data.
    for set_code in args.sets:
        stage_set(repo, set_code, Path(args.cache_dir) if args.cache_dir else None)

    train_set = args.train_set or args.sets[0]
    processed = processed_dir_for(repo, train_set)
    if not is_staged(processed):
        raise SystemExit(
            f"--train-set {train_set} is not among the staged sets {args.sets}"
        )

    if args.stage_only:
        print(f"\nstaged {args.sets} in {time.monotonic() - started:,.0f}s; "
              "--stage-only, not training", flush=True)
        return 0

    # 5. Train.
    run_name = args.run_name or default_run_name(args, train_set)
    out_dir = repo / args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    artefact_dir = Path(args.artefact_root) / run_name

    print(f"\n[train] {run_name} -> {out_dir}", flush=True)
    argv_train = build_training_argv(args, processed, out_dir)
    print(f"  {' '.join(argv_train[2:])}\n", flush=True)
    code = _stream(argv_train, cwd=repo)

    # 6. Artefacts, whatever the outcome -- a failed or truncated segment still
    # has a learning curve worth pulling off a VM that is about to vanish.
    files = mirror_artefacts(out_dir, artefact_dir, args.mirror_resume)
    stopped_at, total_steps = read_progress(out_dir)

    status = {
        "run_name": run_name,
        "completed": code == 0,
        "exit_code": code,
        "stopped_at_step": stopped_at,
        "total_steps": total_steps,
        "commit": commit,
        "backend": after.get("backend"),
        "devices": after.get("devices", []),
        "device_kind": after.get("device_kind"),
        "jax": after.get("jax"),
        "set": train_set,
        "arm": args.arm,
        "width": args.width,
        "fused_kernels": args.fused_kernels,
        "skipped_requirements": skipped,
        "elapsed_s": round(time.monotonic() - started, 1),
        "files": files,
    }
    (artefact_dir / "STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )

    print(f"\n[artefacts] {len(files)} file(s) in {artefact_dir}", flush=True)
    for name in files:
        print(f"    {name}", flush=True)

    if code == EXIT_INCOMPLETE:
        print(
            f"\nsegment ended at step {stopped_at} of {total_steps}; resumable. "
            "Re-invoke with the same arguments to continue.",
            flush=True,
        )
    elif code != 0:
        print(f"\ntraining failed with exit code {code}", flush=True)
    else:
        print(f"\ndone in {status['elapsed_s']:,.0f}s", flush=True)
    return code


# Called unconditionally rather than under `if __name__ == "__main__"`. Only
# `colab run` documents that it sets __name__ to "__main__"; `colab exec -f`
# reads this file and hands its source to a remote kernel, where that is not
# guaranteed. This file is never imported, so there is nothing to protect.
sys.exit(main())
