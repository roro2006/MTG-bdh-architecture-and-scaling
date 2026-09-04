# Running a cell on a Colab accelerator

CPU-only, at 25.6 GFLOP/s, the grid in `docs/PROJECT_PLAN.md` §4 is not a
workload that exists. These two scripts are how a cell runs on a T4 instead --
and, by design though not yet in practice, on a TPU.

    ./scripts/colab_run.sh --gpu T4 --arm bdh --width 64 --steps 3000

That provisions a runtime, stages the data, trains, pulls the artefacts back
into `runs/<run-name>/`, and tears the session down. One command, no browser
step after the one-time authentication below.

## What has actually been run on hardware

Until 2026-09-03 neither script had ever been exercised against a real
accelerator. The path described here was designed, reviewed and unproven. It
has now been run end to end, and the numbers below are measured rather than
predicted.

One T4 session, `--gpu T4 --arm bdh --width 64 --steps 3000`, on FIN, at commit
`c169f60`:

| | |
|---|---|
| throughput, steady state | ~18,400 ex/s |
| throughput, whole run including JIT warmup | 15,291 ex/s |
| CPU baseline it replaces | 561 ex/s |
| training, 3,000 steps | 100s (CPU reference for this cell: 3,363s) |
| staging: 216MB download, then ingest | 16s, then 199s |
| whole driver invocation, provision to teardown | 8m 16s |
| segments used | 1 of 24 |

So roughly **33x** the CPU throughput once the JIT warmup is amortised. The
2.3 h/epoch that `docs/PROJECT_PLAN.md` section 9 names as the binding
constraint on everything downstream becomes about five minutes.

Confirmed working in that run: provisioning, the pushed-HEAD preflight, the
generated remote invocation, the backend probe both before and after installing
dependencies, staging from the 17lands export, training, `STATUS.json`,
artefact download into `runs/<name>/`, session-log export, and teardown from the
trap.

Still unexercised, and so still only designed: `--cache-dir`, TPU runtimes, and
`--fused-kernels`. Re-running the kernel tests with `KERNEL_INTERPRET=0` is also
still pending -- the bootstrap has no pytest path, and one was deliberately not
improvised.

### The multi-segment path, and the bug it was hiding

The first cell long enough to cross a segment boundary was run on 2026-09-03:
`--arm bdh --width 64 --steps 92000`, ten passes over FIN. Segment 1 trained to
step 60,750 in 1,806s, saved state, and exited 75; segment 2 restarted with
`resuming at step 60,750 of 92,000 (best val 0.8246 at step 60,250)`. So the
part that was only designed -- state off the VM, step counter and best-val
tracker carried across a Colab round trip -- does work.

What it exposed is that the budget was being charged the *wrong clock*.
`elapsed_s` is cumulative by design, so the history stays monotonic across an
interruption; the budget check in `src/training/train.py` read that same
figure. Once the cumulative total passed `--max-seconds`, every later segment
tripped the budget at its first evaluation boundary: segments 2 and 3 advanced
250 steps each, reporting 1,831s and 1,856s against an 1,800s budget. A run
long enough to need segments therefore could not finish -- it would burn all 24
segments to cover ~6,000 of the 31,250 steps remaining, one eval interval per
round trip, while the driver's same-step guard stayed quiet because the step
counter *was* advancing.

The budget now measures segment-local time. `tests/test_checkpoint.py::
test_segment_budget_is_per_segment_not_cumulative` backdates a resume state by
ten hours and asserts the next segment still runs to completion. The existing
resume tests could not have caught this: they all use `max_seconds=0.0`, which
fires whichever clock it reads.

One limitation this turned up and did not fix: `--mirror-resume` copies the
resume state *off* the VM, but nothing copies it back. It is a hedge that
currently buys an artefact, not an automated recovery.

## Authenticate once

    colab sessions

It prints a URL. Open it in any browser on the Windows side, approve, and paste
the code back at the prompt. The token is cached under `~/.config/colab-cli/`
and reused, so this is the only interactive step in the workflow. It works from
WSL precisely because it is a paste-the-code flow and needs no local browser and
no `gcloud`; the scopes it requests already include `colaboratory`, so the
keep-alive daemon will not 403 later.

Pin the CLI's own dependency when you install it:

    uv tool install --force google-colab-cli --with "jupyter-kernel-client==0.15.0"

`google-colab-cli` 0.6.0 requires `jupyter-kernel-client` with no upper bound,
and that package renamed `KernelClient` to `JupyterKernelClient` in 1.0.0. Left
to resolve freely it picks up 1.0.2, and then every `colab exec` dies with
`AttributeError: module 'jupyter_kernel_client' has no attribute
'KernelClient'` -- *after* provisioning a session, so it costs a runtime to
discover. 0.15.0 is the last release the CLI can drive.

## Why two files rather than one

`colab exec -f` reads a file locally and ships its **source** to the remote
kernel. It does not upload a project. This repo is a package plus a processed
dataset that git does not carry -- `data/raw/`, `data/processed/` and `runs/`
are all gitignored -- so the one file that gets sent has to be a bootstrap that
assembles everything else on the far side.

- **`colab_bootstrap.py`** runs *on the VM*. Stdlib-only at import time, because
  it runs before anything is installed. It checks the backend, clones the repo,
  installs dependencies, stages the data, trains, and writes a `STATUS.json`.
- **`colab_run.sh`** runs *in WSL*. It provisions, preflights, drives the
  bootstrap in segments, retrieves artefacts, and guarantees teardown.

## The segment loop

A free runtime caps at 12h and is reclaimed after 90 minutes idle, and
`colab exec` blocks for the whole cell -- so a long cell cannot be one call, and
results have to come off the VM as they are produced rather than at the end.

The bootstrap therefore trains under a wall-clock budget (`--max-seconds`,
default 30 minutes). When the budget expires it saves resume state and exits
**75**. The driver pulls whatever artefacts exist, then execs again; the
bootstrap always passes `--resume`, so the next segment continues rather than
restarting. This repeats until exit 0.

Resume is exact, not approximate: optimiser moments, step counter and
batch-stream position are all restored, and `tests/test_checkpoint.py` asserts
that a segmented run lands on bit-for-bit identical parameters to an
uninterrupted one. See `src/training/README.md` for why that matters and why
resume state is a separate artefact from the best-val checkpoint.

**`STATUS.json` is the authority, not the exit status.** Only `colab run`
documents that it propagates a script's exit code; `colab exec` does not, and a
`SystemExit` inside a Jupyter kernel is caught rather than fatal. So the driver
reads `completed` and `exit_code` out of the file the bootstrap wrote, and
treats the CLI's own status as a hint.

## Data staging, and why Drive over GCS

Cloning gets code and nothing else. By default the bootstrap downloads the raw
export from 17lands on the VM and ingests it there -- measured at 216MB in 16s
and a 199s ingest for FIN, paid once per session, on a link far faster than a
home connection.

`--cache-dir` opts into caching the processed set so later sessions restore it
instead of re-ingesting. Point it at a Drive mount. Drive rather than GCS
because it is the same Google identity already used for Colab auth and needs no
GCP project, billing account or bucket, and 15GB free covers all ten processed
sets (~565MB). Both would need one interactive mount -- `colab drivemount` and
`colab auth` each require a TTY and are not automatable -- which is exactly why
caching is opt-in and the default path stays fully automated.

Artefact retrieval needs neither: `colab download` moves files directly.

## Traps these scripts exist to avoid

**The CLI's `--timeout` defaults to 30 seconds** on both `exec` and `run`. A
training cell left on the default dies almost immediately, having done nothing.
The driver always sets it explicitly, to `--max-seconds + --setup-allowance`.

**`requirements.txt` lists bare `jax` and `jaxlib`.** That is right for a CPU
laptop and destructive on an accelerator VM: pip would replace Colab's CUDA/TPU
build with CPU wheels, and the run would finish, report plausible numbers, and
have measured nothing. The bootstrap filters both out and re-probes the backend
afterwards, in a fresh subprocess, since a process that already imported jax
would keep reporting the version it imported first.

**Colab's own flax can be older than Colab's own jax.** The image that ran the
measurement above shipped jax 0.11.1 beside a flax predating 0.12.7, whose
`flax/core/tracers.py` calls `jax.core.get_opaque_trace_state` unconditionally
-- a function removed in jax 0.11.0. The cell reaches `model.init()` and dies
there, having already paid for the download and the ingest. A bare
`pip install flax` does not save you: with any flax present pip answers
"already satisfied" and leaves the stale one exactly where it is. So the
bootstrap installs flax and optax with `--upgrade`, under a constraint file
pinning jax and jaxlib to the versions the arrival probe found on the VM --
otherwise the upgrade could satisfy itself by pulling the accelerated build out
from under the run, which is the same clobber arriving down a dependency edge.

**The runtime clones from GitHub**, so anything unpushed does not exist there.
The driver refuses to run with a dirty tree or unpushed commits, and pins
`--ref` to the exact SHA it verified rather than to `origin/main`, so a push
landing mid-run cannot silently change what is training.

**TPU flag values are `v5e1` and `v6e1`**, not `v5e`. An unrecognised `--gpu`
value silently falls back to A100, so the driver validates locally first.

**An unstopped session bills indefinitely.** Teardown runs from a `trap`, so it
happens on error and interrupt too, not only on success.

**Pack geometry is measured, not assumed** -- which is what makes `--set`
meaningful. Arena's usual shape is 3x14, but four of the ten sets ingested so
far do not use it: BLB and EOE draft 13-card packs, LCI and SIR draft 15.
Ingest records the geometry it saw in `ingest_stats.json`, `PickData` reads it
back, and `ModelConfig` is sized from the corpus. Staging a set whose export
predates that record still works -- the geometry is inferred from the arrays.

## Checking a run is real

The uniform baseline is a property of the pack geometry alone, so it is the one
number a correct run has to reproduce exactly:

| | |
|---|---|
| uniform baseline, FIN | 1.7994 |
| pick-rate prior, FIN | 1.5662 |

Loss numbers are **not** comparable across the representation rebuild. Every
result predating it used a 65-column card feature table; the table is 119
columns now (15 fixed keywords + 73 mechanics). A run today therefore starts a
new series rather than continuing the old one, and the figures in
`docs/RESULTS.md` should not be read against it.

Post-rebuild, from the T4 run above (BDH, d=64, 3,000 steps -- which is 0.326
of an epoch, not a converged model):

| | |
|---|---|
| all picks | 0.8814 (acc 0.6638, 589,008 rows) |
| picks 0-8, the headline slice | 1.0806 (acc 0.5997, 378,648 rows) |
| best sampled val | 0.8846 at step 2,750 |

Pick 13 is forced -- one card left, loss identically 0 -- and is 7.1% of the
val split, which is why the all-picks figure sits below every individual
decision pick.

A run landing far from the uniform baseline means something is wrong in the
port or the staging. It is not a thing to average away.
