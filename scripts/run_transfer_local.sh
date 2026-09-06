#!/usr/bin/env bash
#
# Launch one zero-shot cross-set cell on this machine, detached.
#
#   scripts/run_transfer_local.sh attention 0-10
#   scripts/run_transfer_local.sh bdh 11-21
#
# The Colab driver (scripts/colab_run.sh) is the right tool when a GPU is
# available; this is the CPU fallback for a machine that has cores and no
# accelerator. A 92,000-step cell is ~10h for the attention arm and ~14h for
# BDH at this machine's ~1,400 and ~1,000 examples/second, so it has to
# survive the shell that started it: setsid detaches it, and train.py writes
# resume state at every evaluation boundary, so a kill costs at most one
# 250-step interval.
#
# The core mask is the second argument. Both arms are meant to run at once
# and JAX's CPU backend sizes its thread pool from the whole machine, so
# without disjoint masks the two processes oversubscribe every core and both
# get slower.

set -euo pipefail

ARM="${1:?usage: run_transfer_local.sh <attention|bdh> <core-mask> [steps]}"
CORES="${2:?usage: run_transfer_local.sh <attention|bdh> <core-mask> [steps]}"
STEPS="${3:-92000}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Artefacts go to the main checkout, not to this worktree: a worktree is
# deleted with the session that made it, and a 14h run should outlive that.
ARTEFACTS="${TRANSFER_ARTEFACT_ROOT:-/home/rores/scaling/MTG-bdh-architecture-and-scaling/runs}"
NAME="transfer_${ARM}_d64_s${STEPS}"
OUT="${ARTEFACTS}/${NAME}"
LOG="${ARTEFACTS}/${NAME}.log"

mkdir -p "${OUT}"
RESUME=()
if [ -f "${OUT}/resume.json" ]; then
    echo "[launch] resume state present in ${OUT}; continuing that run"
    RESUME=(--resume)
fi

cd "${REPO}"
setsid nohup taskset -c "${CORES}" ./.venv/bin/python -m src.training.transfer \
    --held-out FIN \
    --arm "${ARM}" \
    --width 64 \
    --steps "${STEPS}" \
    --seed 0 \
    --out-dir "${OUT}" \
    "${RESUME[@]}" \
    >>"${LOG}" 2>&1 </dev/null &

echo "[launch] ${ARM} arm on cores ${CORES}, pid $!"
echo "[launch] log ${LOG}"
echo "[launch] out ${OUT}"
