#!/usr/bin/env bash
#
# Emit one line whenever the local zero-shot runs need attention.
#
# Written as a file rather than passed inline: this has to reach WSL through
# a Windows shell, and every $(...) and $N in an inline command gets expanded
# on the way. A watch that silently evaluates `[ "" -lt 2 ]` looks exactly
# like a watch that has nothing to report.
#
# Silence here means "both alive, memory fine". Everything else is a line.

set -u

RUNS="${1:-/home/rores/scaling/MTG-bdh-architecture-and-scaling/runs}"
EXPECTED="${2:-2}"
LOW_MB="${3:-700}"

last_report=0

while true; do
    alive=$(pgrep -fc "src.training.transfer" || true)
    alive=${alive:-0}
    avail=$(free -m | awk '/^Mem:/ {print $7}')

    if [ "${alive}" -lt "${EXPECTED}" ]; then
        echo "RUN DIED: ${alive} of ${EXPECTED} transfer processes alive, ${avail}MB free"
        for log in "${RUNS}"/transfer_*_d64_s92000.log; do
            [ -f "${log}" ] || continue
            echo "  $(basename "${log}"): $(grep -c 'step ' "${log}" || echo 0) evals, last: $(grep 'step ' "${log}" | tail -1)"
        done
        grep -i -m2 'Traceback\|MemoryError\|Killed\|Error' "${RUNS}"/transfer_*.log 2>/dev/null | head -4
    fi

    if [ "${avail}" -lt "${LOW_MB}" ]; then
        echo "MEMORY LOW: ${avail}MB free, OOM kill likely"
    fi

    if [ "${alive}" -eq 0 ]; then
        echo "ALL TRANSFER RUNS GONE"
        exit 0
    fi

    # A heartbeat every ~2h so a stalled run is distinguishable from a
    # healthy quiet one. Progress is read off the logs, not assumed.
    now=$(date +%s)
    if [ $((now - last_report)) -ge 7200 ]; then
        last_report=${now}
        for log in "${RUNS}"/transfer_*_d64_s92000.log; do
            [ -f "${log}" ] || continue
            echo "PROGRESS $(basename "${log}" .log): $(grep 'step ' "${log}" | tail -1)"
        done
    fi

    sleep 120
done
