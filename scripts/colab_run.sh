#!/usr/bin/env bash
#
# Drive one training cell on a Colab accelerator, from a clean WSL shell.
#
#   scripts/colab_run.sh --gpu T4 --arm bdh --width 64 --steps 3000
#
# Everything in this project has run CPU-only until now, and the scaling grid
# in docs/PROJECT_PLAN.md section 4 is unplannable without a measured per-step
# time on real hardware. This script is the thing that gets a cell onto that
# hardware and the artefacts back off it, with no manual browser step beyond
# the one-time OAuth.
#
# The three failure modes it exists to prevent, all of which cost real money or
# real hours when they bite:
#
#   1. `colab exec --timeout` defaults to THIRTY SECONDS. A training cell left
#      on the default is killed before JAX has finished tracing, and the
#      failure looks like a hang rather than a timeout. Every exec here passes
#      an explicit ceiling derived from the segment budget.
#
#   2. The runtime clones from GitHub. A commit sitting unpushed on this
#      machine does not exist on the VM, so the run silently trains the
#      *previous* revision of the code and reports plausible numbers for it.
#      Preflight refuses to provision until HEAD is pushed.
#
#   3. An unstopped session bills until the 24h keep-alive cap. Teardown runs
#      from a trap, so an error or a Ctrl-C releases the VM too -- not just the
#      happy path.
#
# Runtimes are ephemeral: 12h hard cap, 90 minutes idle. A cell that outlives
# one session therefore cannot be a single blocking exec. It is run as a
# sequence of bounded segments -- train for --max-seconds, save resume state,
# exit 75, hand back to this script, which pulls the artefacts produced so far
# and execs again with --resume. Results come off the VM every segment rather
# than only at the end, and an interruption costs at most one segment.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly BOOTSTRAP="${REPO_ROOT}/scripts/colab_bootstrap.py"
readonly REMOTE_ARTIFACTS="/content/artifacts"

# The CLI silently falls back to A100 when --gpu is a value it does not know,
# which then fails to allocate on a free account and wastes the round trip.
# Validate locally against what `colab new --help` actually lists.
readonly VALID_GPUS="T4 L4 G4 H100 A100"
readonly VALID_TPUS="v5e1 v6e1"

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

GPU=""
TPU=""
SET_NAME="FIN"
ARM="attention"
WIDTH="64"
STEPS="3000"
# Empty means "not given". A grid cell must be sized in epochs rather than
# steps -- PROJECT_PLAN.md section 6 requires the D axis to be data scale, and
# at fixed steps a small --data-fraction silently becomes many passes over
# little data, which turns beta into a repetition exponent. run.py gives
# --epochs precedence over --steps, so setting this makes STEPS inert.
EPOCHS=""
DATA_FRACTION=""
NEURON_MULTIPLIER=""
SEED="0"
FUSED_KERNELS=0
RUN_NAME=""
SESSION=""
# 30 minutes. Comfortably inside the 90-minute idle timeout, and short enough
# that an interruption loses at most half an hour of compute.
MAX_SECONDS="1800"
# Wall clock the bootstrap may spend on everything that is not training:
# clone, pip install, the 206MB raw download and the ~283s ingest. Only the
# first segment pays the full cost; later ones restore from cache. This is a
# ceiling, not a wait, so it is set generously.
SETUP_ALLOWANCE="1800"
KEEP=0
DRY_RUN=0
MAX_SEGMENTS="24"   # 24 * 30min = 12h, the session cap
EXTRA_ARGS=()

# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------

log()  { printf '[driver] %s\n' "$*" >&2; }
warn() { printf '[driver] WARNING: %s\n' "$*" >&2; }
die()  { printf '[driver] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Run one training cell on a Colab accelerator and retrieve its artefacts.

  scripts/colab_run.sh [--gpu T4 | --tpu v5e1] [options] [-- extra bootstrap args]

Accelerator (pick at most one; omitting both gets a CPU runtime):
  --gpu VARIANT          T4, L4, G4, H100, A100
  --tpu VARIANT          v5e1, v6e1

Cell definition (forwarded to scripts/colab_bootstrap.py):
  --set SET              17lands set code to train on          (default: FIN)
  --arm {attention,bdh}  interaction arm                       (default: attention)
  --width N              hidden dim                            (default: 64)
  --steps N              training steps                        (default: 3000)
  --epochs F             passes over the (subsampled) train split; overrides
                         --steps. Use this for grid cells, not --steps.
  --data-fraction F      subsample drafts to this fraction       (default: 1.0)
  --neuron-multiplier N  BDH neuron width multiplier             (default: 4)
  --seed N               seed                                  (default: 0)
  --fused-kernels        run the arm through its Pallas kernel

Session control:
  --run-name NAME        artefact directory name    (default: <arm>_d<width>_s<steps>)
  --session NAME         Colab session name         (default: derived from run name)
  --max-seconds N        per-segment training budget            (default: 1800)
  --setup-allowance N    extra exec headroom for staging        (default: 1800)
  --max-segments N       loop guard                             (default: 24)
  --keep                 leave the session running when done
  --dry-run              print the plan and exit without provisioning
  -h, --help             this message

Anything after `--` is passed through to the bootstrap verbatim.

The exec timeout is (--max-seconds + --setup-allowance). The CLI's own default
is 30 seconds, which kills a training cell instantly; never rely on it.
USAGE
}

# Every colab call routes through here. Two things matter:
#
#   - stdin is closed. An unauthenticated CLI prompts for a pasted OAuth code
#     and would otherwise block this script forever; with no stdin it exits 1
#     immediately and we can report the remediation.
#   - --config points at a per-run state file, so a driver run cannot clobber
#     the session state of anything else using the CLI. Global flags must come
#     before the subcommand.
colab_cmd() {
    "$COLAB_BIN" --config "$STATE_FILE" "$@" < /dev/null
}

cleanup() {
    local rc=$?
    # Session teardown happens BEFORE the temp directory is removed: the
    # session state file lives inside it, and `colab stop` needs it to know
    # which server assignment the name refers to.
    #
    # STOPPED guards against tearing down twice: the trap fires on normal exit
    # as well as on error, and stopping an already-stopped session prints a
    # confusing 404.
    if [[ "${PROVISIONED:-0}" == "1" && "${STOPPED:-0}" == "0" ]]; then
        if [[ "$KEEP" == "1" ]]; then
            # The state file has to outlive this run, or the command printed
            # here would point at a path the next line deletes -- and a kept
            # session bills until the 24h cap.
            local kept_state="${LOCAL_ARTIFACTS}/sessions.json"
            mkdir -p "$LOCAL_ARTIFACTS" 2>/dev/null || true
            if cp -f "$STATE_FILE" "$kept_state" 2>/dev/null; then
                log "leaving session '${SESSION}' running (--keep)."
                log "it bills until stopped: $COLAB_BIN --config $kept_state stop -s ${SESSION}"
            else
                warn "leaving session '${SESSION}' running (--keep), but its state file"
                warn "could not be preserved. Stop it via: $COLAB_BIN stop -s ${SESSION}"
                warn "or find it with: $COLAB_BIN sessions"
            fi
        else
            log "stopping session '${SESSION}'..."
            STOPPED=1
            colab_cmd stop -s "$SESSION" || warn "could not stop '${SESSION}'; stop it by hand or it will bill until the 24h cap"
        fi
    fi

    if [[ -n "${TMPDIR_RUN:-}" && -d "$TMPDIR_RUN" ]]; then
        rm -rf "$TMPDIR_RUN"
    fi
    exit "$rc"
}

# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)              GPU="${2:?--gpu needs a value}"; shift 2 ;;
        --tpu)              TPU="${2:?--tpu needs a value}"; shift 2 ;;
        --set)              SET_NAME="${2:?--set needs a value}"; shift 2 ;;
        --arm)              ARM="${2:?--arm needs a value}"; shift 2 ;;
        --width)            WIDTH="${2:?--width needs a value}"; shift 2 ;;
        --steps)            STEPS="${2:?--steps needs a value}"; shift 2 ;;
        --epochs)           EPOCHS="${2:?--epochs needs a value}"; shift 2 ;;
        --data-fraction)    DATA_FRACTION="${2:?--data-fraction needs a value}"; shift 2 ;;
        --neuron-multiplier) NEURON_MULTIPLIER="${2:?--neuron-multiplier needs a value}"; shift 2 ;;
        --seed)             SEED="${2:?--seed needs a value}"; shift 2 ;;
        --fused-kernels)    FUSED_KERNELS=1; shift ;;
        --run-name)         RUN_NAME="${2:?--run-name needs a value}"; shift 2 ;;
        --session)          SESSION="${2:?--session needs a value}"; shift 2 ;;
        --max-seconds)      MAX_SECONDS="${2:?--max-seconds needs a value}"; shift 2 ;;
        --setup-allowance)  SETUP_ALLOWANCE="${2:?--setup-allowance needs a value}"; shift 2 ;;
        --max-segments)     MAX_SEGMENTS="${2:?--max-segments needs a value}"; shift 2 ;;
        --keep)             KEEP=1; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        --)                 shift; EXTRA_ARGS=("$@"); break ;;
        *)                  usage >&2; die "unknown argument: $1" ;;
    esac
done

[[ -n "$GPU" && -n "$TPU" ]] && die "pass --gpu or --tpu, not both"

if [[ -n "$GPU" ]] && ! grep -qw -- "$GPU" <<<"$VALID_GPUS"; then
    die "--gpu '$GPU' is not one of: $VALID_GPUS
An unrecognised value does not error -- the CLI quietly provisions an A100
instead, which then fails to allocate on a free-tier account."
fi
if [[ -n "$TPU" ]] && ! grep -qw -- "$TPU" <<<"$VALID_TPUS"; then
    die "--tpu '$TPU' is not one of: $VALID_TPUS
Note the trailing 1: the variants are v5e1 and v6e1, not v5e."
fi

case "$ARM" in
    attention|bdh) ;;
    *) die "--arm must be 'attention' or 'bdh', got '$ARM'" ;;
esac

for pair in "MAX_SECONDS:$MAX_SECONDS" "SETUP_ALLOWANCE:$SETUP_ALLOWANCE" \
            "MAX_SEGMENTS:$MAX_SEGMENTS" "STEPS:$STEPS" "WIDTH:$WIDTH" "SEED:$SEED"; do
    name="${pair%%:*}"; value="${pair#*:}"
    [[ "$value" =~ ^[0-9]+$ ]] || die "--${name,,} must be a non-negative integer, got '$value'"
done
unset name value pair

# The default name has to separate any two cells that are different
# experiments. It keyed on --steps alone, which was fine while --steps was the
# only size knob; with --epochs and --data-fraction it is not, and two grid
# cells sharing a name means the second silently overwrites the first's
# artefacts and then *resumes* from them.
if [[ -z "$RUN_NAME" ]]; then
    if [[ -n "$EPOCHS" ]]; then
        RUN_NAME="${ARM}_d${WIDTH}_e${EPOCHS}"
    else
        RUN_NAME="${ARM}_d${WIDTH}_s${STEPS}"
    fi
    [[ -n "$DATA_FRACTION" ]] && RUN_NAME="${RUN_NAME}_f${DATA_FRACTION}"
    [[ -n "$NEURON_MULTIPLIER" ]] && RUN_NAME="${RUN_NAME}_n${NEURON_MULTIPLIER}"
    [[ "$SEED" != "0" ]] && RUN_NAME="${RUN_NAME}_seed${SEED}"
fi

# The run name becomes a path segment on both sides -- runs/<name> here and
# /content/artifacts/<name> on the VM -- and `colab download` takes those
# paths as single arguments. A name containing a space or a quote produces
# remote paths that fail to resolve well after the compute has been spent, so
# normalise it up front rather than discovering it at retrieval time.
RUN_NAME_SAFE="$(tr -c 'A-Za-z0-9._-' '-' <<<"$RUN_NAME" | tr -s '-' | sed 's/^-//; s/-$//')"
if [[ "$RUN_NAME_SAFE" != "$RUN_NAME" ]]; then
    warn "run name '${RUN_NAME}' normalised to '${RUN_NAME_SAFE}' for use as a path"
    RUN_NAME="$RUN_NAME_SAFE"
fi
[[ -n "$RUN_NAME" ]] || die "--run-name normalised to nothing usable"

# Session names address the VM in every later call, so keep them to characters
# that survive a shell round trip. The run name is free-form; the session name
# derived from it is not.
if [[ -z "$SESSION" ]]; then
    SESSION="mtg-$(tr -c 'A-Za-z0-9_-' '-' <<<"$RUN_NAME" | tr -s '-' | sed 's/-$//')"
fi

readonly EXEC_TIMEOUT=$(( MAX_SECONDS + SETUP_ALLOWANCE ))
readonly LOCAL_ARTIFACTS="${REPO_ROOT}/runs/${RUN_NAME}"
readonly REMOTE_RUN_DIR="${REMOTE_ARTIFACTS}/${RUN_NAME}"

# --------------------------------------------------------------------------
# Preflight -- all cheap, all catching a failure that is expensive later
# --------------------------------------------------------------------------

COLAB_BIN="$(command -v colab || true)"
if [[ -z "$COLAB_BIN" ]]; then
    # uv tool install puts it here, which is not on a non-login shell's PATH.
    if [[ -x "$HOME/.local/bin/colab" ]]; then
        COLAB_BIN="$HOME/.local/bin/colab"
    else
        die "the colab CLI is not on PATH.
Install it with:  uv tool install google-colab-cli
It lands in ~/.local/bin, which you may need to add to PATH."
    fi
fi

[[ -f "$BOOTSTRAP" ]] || die "missing $BOOTSTRAP -- the remote entry point this script drives"

command -v python3 >/dev/null || die "python3 is required (to build the remote argv and parse STATUS.json)"

TMPDIR_RUN="$(mktemp -d "${TMPDIR:-/tmp}/colab-run-XXXXXX")"
STATE_FILE="${TMPDIR_RUN}/sessions.json"
PROVISIONED=0
STOPPED=0
trap cleanup EXIT INT TERM

# Auth. A read-only call is enough to tell authenticated from not, and costs
# nothing. With stdin closed the CLI aborts instead of waiting for a pasted
# code, so this cannot hang.
auth_ok=1
auth_output="$(colab_cmd sessions 2>&1)" || auth_ok=0
if [[ "$auth_ok" == "0" ]]; then
    msg="not authenticated to Colab.

Authenticate once, interactively, then re-run this script:

    colab sessions

It prints a URL; open it in any browser on the Windows side, approve, and
paste the code back at the prompt. That is the only manual browser step in
this workflow -- everything after it is automated. The token is cached in
~/.config/colab-cli/ and reused.

The CLI reported:
${auth_output}"
    if [[ "$DRY_RUN" == "1" ]]; then
        warn "${msg}"
    else
        die "${msg}"
    fi
fi

# The VM clones from GitHub. Anything not pushed does not exist there, and the
# run would train an older revision while reporting as if it were this one.
if ! git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    die "$REPO_ROOT is not a git repository"
fi

git -C "$REPO_ROOT" update-index --refresh >/dev/null 2>&1 || true
if ! git -C "$REPO_ROOT" diff --quiet HEAD 2>/dev/null; then
    die "the working tree has uncommitted changes.
The runtime clones from GitHub, so uncommitted work will not be there.
Commit and push, or stash, before running.

$(git -C "$REPO_ROOT" status --short)"
fi

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
REPO_URL="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
[[ -n "$REPO_URL" ]] || die "no 'origin' remote; the runtime has nothing to clone from"

git -C "$REPO_ROOT" fetch --quiet origin "$GIT_BRANCH" 2>/dev/null || \
    warn "could not fetch origin/$GIT_BRANCH; the pushed-state check may be stale"

if ! git -C "$REPO_ROOT" merge-base --is-ancestor "$GIT_COMMIT" "origin/${GIT_BRANCH}" 2>/dev/null; then
    die "HEAD (${GIT_COMMIT:0:8}) is not on origin/${GIT_BRANCH}.
The runtime clones from GitHub and would train an older revision.
Push first:  git push origin ${GIT_BRANCH}"
fi

# --------------------------------------------------------------------------
# The remote invocation
# --------------------------------------------------------------------------

# `colab exec -f` ships a file's *source* to the kernel and does not forward
# argv the way `colab run` does. So the invocation is materialised here: a
# prelude setting sys.argv, followed by the bootstrap verbatim.
#
# The argv literal is built by python's repr rather than by string
# concatenation, so a set code or run name containing a space or a quote
# cannot break out into the generated source.
build_remote_script() {
    local resume_flag="$1" out="$2"
    # Flag names here are the bootstrap's, not this script's: it takes a list
    # of sets to stage (--sets) and separately which one to train on
    # (--train-set), and it spells the artefact root the British way. Both are
    # passed explicitly rather than leaning on its defaults, so a change to
    # either side shows up as an argparse error rather than as a run that
    # quietly trained the wrong set.
    #
    # --ref is pinned to the exact SHA preflight verified as pushed, not left
    # at its origin/main default. Otherwise a push landing between preflight
    # and clone would train a commit this script never checked.
    local -a bootstrap_argv=(
        --sets "$SET_NAME"
        --train-set "$SET_NAME"
        --ref "$GIT_COMMIT"
        --arm "$ARM"
        --width "$WIDTH"
        --steps "$STEPS"
        --seed "$SEED"
        --run-name "$RUN_NAME"
        --artefact-root "$REMOTE_ARTIFACTS"
        --max-seconds "$MAX_SECONDS"
    )
    [[ -n "$EPOCHS" ]] && bootstrap_argv+=(--epochs "$EPOCHS")
    [[ -n "$DATA_FRACTION" ]] && bootstrap_argv+=(--data-fraction "$DATA_FRACTION")
    [[ -n "$NEURON_MULTIPLIER" ]] && bootstrap_argv+=(--neuron-multiplier "$NEURON_MULTIPLIER")
    [[ "$FUSED_KERNELS" == "1" ]] && bootstrap_argv+=(--fused-kernels)
    [[ "$resume_flag" == "1" ]] && bootstrap_argv+=(--resume)
    [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && bootstrap_argv+=("${EXTRA_ARGS[@]}")

    BOOT_SRC="$BOOTSTRAP" BOOT_OUT="$out" python3 - "${bootstrap_argv[@]}" <<'PY'
import ast
import os
import sys

source = open(os.environ["BOOT_SRC"], encoding="utf-8").read()
argv = ["colab_bootstrap.py"] + sys.argv[1:]

prelude = [
    "# --- generated by scripts/colab_run.sh -- do not edit ---",
    "import sys",
    "sys.argv = " + repr(argv),
    # `colab run` documents that it sets __name__ == "__main__" like native
    # python; `colab exec` promises nothing. Cheap insurance either way.
    '__name__ = "__main__"',
    "# --- end generated prelude ---",
    "",
]

# The prelude cannot simply go at the top. `from __future__ import
# annotations` is required by the language to be the first statement after
# the module docstring, so prepending anything ahead of it makes the whole
# file a SyntaxError -- and every segment would die instantly, before
# argparse ever ran. Splice in after the docstring and any future imports
# instead, located properly rather than by guessing a line number.
tree = ast.parse(source)
insert_after = 0
for node in tree.body:
    is_docstring = (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
    is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
    if is_docstring or is_future:
        insert_after = max(insert_after, node.end_lineno)
    else:
        break

lines = source.splitlines(keepends=True)
composed = "".join(lines[:insert_after]) + "\n".join(prelude) + "\n" + "".join(lines[insert_after:])

# Fail here rather than on the VM: a composition error is free to catch
# locally and costs a whole provision to discover remotely.
ast.parse(composed)
open(os.environ["BOOT_OUT"], "w", encoding="utf-8").write(composed)
PY
}

# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------

accel_desc="CPU (no accelerator requested)"
[[ -n "$GPU" ]] && accel_desc="GPU ${GPU}"
[[ -n "$TPU" ]] && accel_desc="TPU ${TPU}"

cat >&2 <<PLAN
[driver] plan
  accelerator     ${accel_desc}
  session         ${SESSION}
  run name        ${RUN_NAME}
  cell            arm=${ARM} width=${WIDTH} $([[ -n "$EPOCHS" ]] && printf 'epochs=%s (--steps inert)' "$EPOCHS" || printf 'steps=%s' "$STEPS") seed=${SEED} set=${SET_NAME}$([[ -n "$DATA_FRACTION" ]] && printf ' data-fraction=%s' "$DATA_FRACTION")$([[ -n "$NEURON_MULTIPLIER" ]] && printf ' neuron-multiplier=%s' "$NEURON_MULTIPLIER")$([[ "$FUSED_KERNELS" == "1" ]] && printf ' fused-kernels')
  commit          ${GIT_COMMIT:0:8} on ${GIT_BRANCH} (pushed)
  segment budget  ${MAX_SECONDS}s training, up to ${MAX_SEGMENTS} segments
  exec timeout    ${EXEC_TIMEOUT}s  (${MAX_SECONDS} budget + ${SETUP_ALLOWANCE} staging headroom)
  remote dir      ${REMOTE_RUN_DIR}
  local dir       ${LOCAL_ARTIFACTS}
PLAN

if [[ "$DRY_RUN" == "1" ]]; then
    preview="${TMPDIR_RUN}/preview.py"
    build_remote_script 0 "$preview"
    # Show the spliced prelude, not the head of the file: the prelude sits
    # after the bootstrap's docstring and future import, so a fixed line count
    # would just print the docstring.
    log "generated remote invocation:"
    sed -n '/--- generated by scripts\/colab_run.sh/,/--- end generated prelude ---/p' \
        "$preview" >&2
    log "dry run: nothing provisioned."
    exit 0
fi

# --------------------------------------------------------------------------
# Artefact retrieval
# --------------------------------------------------------------------------

STATUS_LOCAL="${TMPDIR_RUN}/STATUS.json"

# Returns 0 if a STATUS.json came back. The bootstrap writes it last, so its
# absence means the segment died before finishing -- which is a normal thing
# to survive, not a reason to wedge the loop.
fetch_status() {
    rm -f "$STATUS_LOCAL"
    colab_cmd download -s "$SESSION" "${REMOTE_RUN_DIR}/STATUS.json" "$STATUS_LOCAL" \
        >/dev/null 2>&1 || return 1
    [[ -s "$STATUS_LOCAL" ]] || return 1
    python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$STATUS_LOCAL" 2>/dev/null || return 1
}

status_field() {
    python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
v = d.get(sys.argv[2])
print("" if v is None else v)
' "$STATUS_LOCAL" "$1" 2>/dev/null || true
}

# colab download moves one file per call, so the manifest in STATUS.json is
# what makes retrieval possible at all -- there is no recursive get.
pull_artifacts() {
    mkdir -p "$LOCAL_ARTIFACTS"
    cp -f "$STATUS_LOCAL" "${LOCAL_ARTIFACTS}/STATUS.json" 2>/dev/null || true

    local files pulled=0 failed=0
    files="$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
for f in d.get("files", []):
    if isinstance(f, str) and f.strip():
        print(f)
' "$STATUS_LOCAL" 2>/dev/null || true)"

    [[ -n "$files" ]] || { log "STATUS.json lists no artefacts yet"; return 0; }

    while IFS= read -r rel; do
        [[ -n "$rel" ]] || continue
        mkdir -p "${LOCAL_ARTIFACTS}/$(dirname "$rel")"
        if colab_cmd download -s "$SESSION" "${REMOTE_RUN_DIR}/${rel}" \
               "${LOCAL_ARTIFACTS}/${rel}" >/dev/null 2>&1; then
            pulled=$(( pulled + 1 ))
        else
            failed=$(( failed + 1 ))
            warn "could not download ${rel}"
        fi
    done <<< "$files"

    log "pulled ${pulled} artefact(s) into ${LOCAL_ARTIFACTS}$([[ $failed -gt 0 ]] && printf ' (%d failed)' "$failed")"
}

# --------------------------------------------------------------------------
# Provision and run
# --------------------------------------------------------------------------

log "provisioning ${accel_desc} as session '${SESSION}'..."
new_args=(new -s "$SESSION")
[[ -n "$GPU" ]] && new_args+=(--gpu "$GPU")
[[ -n "$TPU" ]] && new_args+=(--tpu "$TPU")
colab_cmd "${new_args[@]}" || die "could not provision ${accel_desc}.
A 400 here usually means the account has no entitlement for this accelerator
on its current tier. Fall back to --gpu T4, or omit the flag for CPU."
PROVISIONED=1

colab_cmd status -s "$SESSION" || true

segment=0
resume=0
completed=0
last_step=""
final_rc=0

while (( segment < MAX_SEGMENTS )); do
    segment=$(( segment + 1 ))
    remote_script="${TMPDIR_RUN}/segment.py"
    build_remote_script "$resume" "$remote_script"

    log "segment ${segment}/${MAX_SEGMENTS}$([[ "$resume" == "1" ]] && printf ' (resuming)') -- exec timeout ${EXEC_TIMEOUT}s"

    rc=0
    colab_cmd exec -s "$SESSION" -f "$remote_script" --timeout "$EXEC_TIMEOUT" || rc=$?

    # Whether `colab exec` propagates the script's exit code is not documented
    # (only `colab run` promises it), so the exec status is treated as a hint
    # and STATUS.json as the authority. When the two disagree, the file wins:
    # it is written by the bootstrap itself and says what actually happened.
    if fetch_status; then
        pull_artifacts
        status_completed="$(status_field completed)"
        status_rc="$(status_field exit_code)"
        step="$(status_field stopped_at_step)"
        total="$(status_field total_steps)"

        [[ -n "$step" ]] && log "progress: step ${step}${total:+ of ${total}}"

        if [[ "$status_completed" == "True" || "$status_completed" == "true" ]]; then
            completed=1
            final_rc=0
            break
        fi

        # A segment that ends on the same step it started on is not making
        # progress; looping would burn the whole 12h cap achieving nothing.
        if [[ -n "$step" && "$step" == "$last_step" ]]; then
            final_rc="${status_rc:-1}"
            die "segment ${segment} ended at step ${step}, the same step as the previous one.
The run is not progressing. Artefacts so far are in ${LOCAL_ARTIFACTS}."
        fi
        last_step="$step"

        if [[ -n "$status_rc" && "$status_rc" != "75" && "$status_rc" != "0" ]]; then
            final_rc="$status_rc"
            die "the bootstrap failed with exit code ${status_rc}.
Session log: $COLAB_BIN --config $STATE_FILE log -s ${SESSION}
Artefacts so far are in ${LOCAL_ARTIFACTS}."
        fi
    else
        # No STATUS.json. If exec itself also failed, this is a real error;
        # there is nothing to resume from and nothing to retrieve.
        if [[ "$rc" != "0" ]]; then
            final_rc="$rc"
            die "segment ${segment} failed (exec exit ${rc}) and wrote no STATUS.json.
Inspect the session: $COLAB_BIN --config $STATE_FILE log -s ${SESSION}"
        fi
        warn "segment ${segment} produced no STATUS.json but exec succeeded; retrying with --resume"
    fi

    resume=1
done

if (( completed == 0 )); then
    warn "hit the ${MAX_SEGMENTS}-segment guard without finishing."
    warn "re-run the same command to continue: the bootstrap resumes from the state on the VM,"
    warn "but note that state does not survive teardown unless it was cached off-VM."
    final_rc=75
fi

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

log "session history -> ${LOCAL_ARTIFACTS}/session_log.md"
mkdir -p "$LOCAL_ARTIFACTS"
colab_cmd log -s "$SESSION" -o "${LOCAL_ARTIFACTS}/session_log.md" >/dev/null 2>&1 \
    || warn "could not export the session log"

backend="$(status_field backend)"
devices="$(status_field devices)"
ran_commit="$(status_field commit)"

cat >&2 <<SUMMARY

[driver] ================ summary ================
  run             ${RUN_NAME} ($([[ "$completed" == "1" ]] && printf 'completed' || printf 'incomplete -- resumable'))
  cell            arm=${ARM} width=${WIDTH} steps=${STEPS} seed=${SEED} set=${SET_NAME}
  accelerator     ${accel_desc}
  backend         ${backend:-<not reported>}
  devices         ${devices:-<not reported>}
  commit on VM    ${ran_commit:-<not reported>}
  segments        ${segment}
  artefacts       ${LOCAL_ARTIFACTS}
SUMMARY

if [[ -d "$LOCAL_ARTIFACTS" ]]; then
    find "$LOCAL_ARTIFACTS" -type f -printf '    %P (%s bytes)\n' 2>/dev/null | sort >&2 || true
fi

exit "$final_rc"
