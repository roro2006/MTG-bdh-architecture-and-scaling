# Running a cell on a Colab accelerator

Everything in this project has run CPU-only, and at 25.6 GFLOP/s the grid in
`docs/PROJECT_PLAN.md` §4 is not a workload that exists. These two scripts are
how a cell runs on a T4 or a TPU instead.

    ./scripts/colab_run.sh --gpu T4 --arm bdh --width 64 --steps 3000

That provisions a runtime, stages the data, trains, pulls the artefacts back
into `runs/<run-name>/`, and tears the session down. One command, no browser
step after the one-time authentication below.

## Authenticate once

    colab sessions

It prints a URL. Open it in any browser on the Windows side, approve, and paste
the code back at the prompt. The token is cached under `~/.config/colab-cli/`
and reused, so this is the only interactive step in the workflow. It works from
WSL precisely because it is a paste-the-code flow and needs no local browser and
no `gcloud`; the scopes it requests already include `colaboratory`, so the
keep-alive daemon will not 403 later.

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
export from 17lands on the VM and ingests it there -- about 206MB and ~283s for
FIN, paid once per session, on a link far faster than a home connection.

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

Sanity numbers on the FIN val split, which any correct run reproduces:

| | |
|---|---|
| uniform baseline | 1.7994 |
| pick-rate prior | 1.5683 |
| attention, d=64, 3000 steps | ~0.9323 all-picks |
| BDH, same config | ~0.9080 |

Reference CPU wall clock for that config: attention 2745s, BDH 3363s. The
speedup on an accelerator should be obvious immediately; if it is not,
something has fallen back to CPU and the backend probe in the log will say so.

A Colab run landing far from these numbers means something is wrong in the port
or the staging. It is not a thing to average away.
