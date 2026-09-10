#!/usr/bin/env bash
# Supervise a long training run so it survives this host's unclean reboots.
#
# spark-cefe goes down uncleanly every few hours and has silently killed two
# long runs mid-flight (run 18 at 1727/1800, run 19 at 1208/3000) -- the console
# log just ends, with no traceback. nohup does not help, because the host itself
# reboots. This script relaunches from the newest checkpoint until the run
# reaches MAX_ITERS, and is also wired to cron's @reboot so it comes back after
# the machine does.
#
#   ./scripts/supervise_run.sh              # start (or pick up) the run
#   crontab -l                              # see the @reboot hook
#   touch <repo>/logs/console/STOP_SUPERVISOR   # ask it to stop after this segment
#
# Delete the crontab line to disarm it permanently.

set -uo pipefail

REPO="/home/afthabshiraz/MujocoRL-Internship/leapXelaMjLab"
UV="/home/afthabshiraz/.local/bin/uv"

TASK="${TASK:-Mjlab-LeapXELA-Cube-Reorient-Reference}"
EXPERIMENT="${EXPERIMENT:-leap_xela_cube_reorient_reference}"
CONSOLE_DIR="$REPO/logs/console"

# Where the in-flight run's settings live. run_queue.sh writes this before each
# run, so the @reboot hook -- which passes no environment at all -- resumes the
# run that was actually going rather than whatever the defaults happened to
# name. Hardcoded defaults were a live hazard: after run 21 finished they still
# said RUN_NAME=entropy-1e-3, so any bare invocation would have relaunched a
# finished run under a stale config. There are now no run defaults; an
# unspecified RUN_NAME is a hard error.
STATE_FILE="$CONSOLE_DIR/CURRENT_RUN.env"
if [ -f "$STATE_FILE" ]; then
  while IFS= read -r line; do
    case "$line" in ''|'#'*) continue;; esac
    key=${line%%=*}; val=${line#*=}
    case "$key" in
      RUN_NAME|MAX_ITERS|NUM_ENVS|SEED|SAVE_INTERVAL|EXTRA_ARGS|TASK|EXPERIMENT) ;;
      *) continue;;
    esac
    # An explicit environment variable always wins over the state file.
    [ -n "${!key:-}" ] || printf -v "$key" '%s' "$val"
  done <"$STATE_FILE"
fi

# EXTRA_ARGS is applied to the fresh launch AND every resume: an override that
# is dropped on resume silently trains a different config in the same run
# directory, which is the same class of bug as the --num-envs note below.
RUN_NAME="${RUN_NAME:-}"
MAX_ITERS="${MAX_ITERS:-1500}"
NUM_ENVS="${NUM_ENVS:-8192}"
SEED="${SEED:-42}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
read -r -a EXTRA_ARGS <<<"${EXTRA_ARGS:-}"

if [ -z "$RUN_NAME" ]; then
  echo "RUN_NAME is unset and $STATE_FILE names no run. Refusing to guess." >&2
  echo "Start runs through scripts/run_queue.sh, or set RUN_NAME explicitly." >&2
  exit 2
fi
RUN_GLOB="$REPO/logs/rsl_rl/$EXPERIMENT/*_${RUN_NAME}"
STOP_FILE="$CONSOLE_DIR/STOP_SUPERVISOR"
LOCK="$CONSOLE_DIR/.${RUN_NAME}.lock"
SUP_LOG="$CONSOLE_DIR/${RUN_NAME}_supervisor.log"

mkdir -p "$CONSOLE_DIR"
cd "$REPO" || exit 1

# Only one supervisor at a time (the @reboot hook and a manual start can race).
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Is)] another supervisor holds the lock; exiting" >>"$SUP_LOG"
  exit 0
fi

log() { echo "[$(date -Is)] $*" >>"$SUP_LOG"; }

# Newest checkpoint across every segment of this run, by iteration number.
# Zero-length files are skipped: a host reboot that lands mid-write leaves a
# truncated checkpoint behind (model_650.pt of orientation-fine, 2026-09-02),
# and resuming from one dies with EOFError inside runner.load before the first
# iteration. Without this filter the supervisor picks the same broken file on
# every relaunch and burns its whole failure budget on it -- 65 dead segments
# across 13 reboots before it was noticed.
latest_ckpt() {
  find $RUN_GLOB -maxdepth 1 -name 'model_*.pt' -size +0c 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' \
    | sort -k1,1n | tail -1 | cut -d' ' -f2
}
latest_iter() {
  local c; c=$(latest_ckpt)
  [ -n "$c" ] && basename "$c" | sed 's/model_\([0-9]*\)\.pt/\1/' || echo 0
}

log "supervisor up (pid $$) — target ${MAX_ITERS} iters, run '${RUN_NAME}'"

fails=0
while :; do
  if [ -f "$STOP_FILE" ]; then
    log "STOP_SUPERVISOR present; standing down"; exit 0
  fi

  it=$(latest_iter)
  # rsl_rl numbers iterations from 0, so a completed --max-iterations N run
  # leaves model_$((N-1)).pt as its last checkpoint. Comparing -ge N against
  # that never fires, and the supervisor relaunches a finished run until the
  # no-progress guard trips (5 wasted segments at the end of run 20).
  if [ "$it" -ge "$((MAX_ITERS - 1))" ]; then
    log "reached ${it}/${MAX_ITERS} — done"; exit 0
  fi

  # Wait for the GPU to be ready (matters on the @reboot path).
  for _ in $(seq 1 30); do
    /usr/bin/nvidia-smi >/dev/null 2>&1 && break
    sleep 10
  done

  ckpt=$(latest_ckpt)
  stamp=$(date +%Y%m%d_%H%M%S)

  # --num-envs and --seed MUST be repeated on resume. The task registers
  # num_envs=1 (it is expected to come from the CLI), so a resume that omits
  # the flag silently trains a 1-env batch in the same run directory -- which
  # is exactly what happened on the first attempt at this run: iterations
  # 100-204 ran at 40 steps/iteration and 61 steps/s before it was caught.
  # Keep every shape-determining flag in COMMON so both branches share them.
  COMMON=(--num-envs "$NUM_ENVS" --seed "$SEED" --max-iterations "$MAX_ITERS"
          --save-interval "$SAVE_INTERVAL" --logger tensorboard
          "${EXTRA_ARGS[@]}")

  if [ -n "$ckpt" ]; then
    log "resuming from $(basename "$ckpt") (iter ${it})"
    out="$CONSOLE_DIR/${RUN_NAME}_resume${it}_${stamp}.log"
    "$UV" run python scripts/train.py "$TASK" \
      "${COMMON[@]}" --resume-from "$ckpt" >>"$out" 2>&1
  else
    log "no checkpoint yet — starting fresh"
    out="$CONSOLE_DIR/${RUN_NAME}_${stamp}.log"
    "$UV" run python scripts/train.py "$TASK" \
      "${COMMON[@]}" --run-name "$RUN_NAME" >>"$out" 2>&1
  fi
  rc=$?

  # The config actually used is dumped to params/env.yaml; verify the segment
  # ran the batch shape we asked for rather than trusting the flags.
  used=$(grep -m1 -E "^  num_envs:" $RUN_GLOB/params/env.yaml 2>/dev/null | tr -dc '0-9')
  if [ -n "$used" ] && [ "$used" != "$NUM_ENVS" ]; then
    log "FATAL: segment ran num_envs=${used}, expected ${NUM_ENVS} — stopping"
    exit 1
  fi
  new_it=$(latest_iter)
  log "train.py exited rc=${rc} at iter ${new_it} (log: $(basename "$out"))"

  # A non-empty checkpoint can still be unreadable (a write that got far enough
  # to have bytes but not to finish). The signature is unmistakable: the segment
  # died inside runner.load, so it trained nothing and never reached iteration
  # one. Quarantine that file and the next pass falls back to the checkpoint
  # before it, costing one save interval instead of the whole retry budget.
  if [ "$rc" -ne 0 ] && [ "$new_it" -le "$it" ] && [ -n "$ckpt" ] \
     && grep -qE 'runner\.load|_legacy_load|UnpicklingError|EOFError' "$out"; then
    log "checkpoint $(basename "$ckpt") failed to load — quarantining as .corrupt"
    mv -- "$ckpt" "${ckpt}.corrupt"
    fails=0
    sleep 5
    continue
  fi

  # Progress resets the failure budget; a segment that trains nothing does not.
  if [ "$new_it" -gt "$it" ]; then fails=0; else fails=$((fails + 1)); fi
  if [ "$fails" -ge 5 ]; then
    log "5 consecutive segments made no progress — giving up, see $(basename "$out")"
    exit 1
  fi
  sleep 20
done
