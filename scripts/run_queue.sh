#!/usr/bin/env bash
# Run the queued reorient experiments back to back, scoring each one as it
# finishes.
#
# Why a queue script rather than two manual launches: this host reboots
# uncleanly every few hours (TRAINING_NOTES.md), and the @reboot cron hook has
# to be able to work out on its own which run was in flight. It does that by
# reading logs/console/CURRENT_RUN.env, which this script writes before each
# run. Everything here is idempotent -- a completed run is skipped, an
# interrupted one is resumed from its newest checkpoint by supervise_run.sh --
# so re-running this script at any point does the right thing.
#
#   ./scripts/run_queue.sh                       # start / resume the queue
#   tail -f logs/console/QUEUE.log               # what it is doing
#   touch logs/console/STOP_SUPERVISOR           # stand down after this segment
#
set -uo pipefail

REPO="/home/afthabshiraz/MujocoRL-Internship/leapXelaMjLab"
UV="/home/afthabshiraz/.local/bin/uv"
EXPERIMENT="leap_xela_cube_reorient_reference"
NUM_ENVS=8192
SAVE_INTERVAL=50

CONSOLE_DIR="$REPO/logs/console"
STATE_FILE="$CONSOLE_DIR/CURRENT_RUN.env"
STOP_FILE="$CONSOLE_DIR/STOP_SUPERVISOR"
QUEUE_LOG="$CONSOLE_DIR/QUEUE.log"
QUEUE_LOCK="$CONSOLE_DIR/.queue.lock"
EVAL_DIR="$REPO/eval"

# Run 14 at the SAME iteration as these runs. The 3000-iteration checkpoint is
# the one every earlier comparison used, which left run 21 scored against twice
# the training budget -- see TRAINING_NOTES.md run 21.
RUN14_MATCHED="$REPO/logs/rsl_rl/$EXPERIMENT/2026-08-29_16-11-45_reference-v3-eulerdamp/model_1500.pt"
# Run 14 trained on to 2999 under the unchanged reward: the matched control for
# any run warm-started from RUN14_MATCHED and trained to 3000.
RUN14_FINAL="$REPO/logs/rsl_rl/$EXPERIMENT/2026-08-29_16-11-45_reference-v3-eulerdamp/model_2999.pt"

# Episodes per eval. One 32-env eval is noisy on its own: run 14's model_2999
# scored 27.7 / 30.7 / 26.0 / 28.3 deg median best error at eval seeds 7-10, so
# decisions need more episodes than the ~3 deg floor assumed. Results at any
# count other than 32 carry an _n<count> tag so they never mix with the
# historical 32-env files.
EVAL_NUM_ENVS=128

# name | max_iters | seed | train overrides | eval overrides (must rebuild the SAME env) | training-env eval overrides
#
# thresh-04 (EXPERIMENT 1) is done and REGRESSED: 72 deg vs run 14's 26 deg,
# tip-over closure 20%. Its training error never left ~90 deg -- the step at
# iteration 300-500 where every working run learns to tip the cube never came.
# Runs 22 (fixed goal), 23 (orientation-fine) and 20 (success x20) fail at that
# same step. Every from-scratch reward/goal edit has died there, so none of them
# ever tested what they were for: whether a reward that pays for precision
# moves the ~28 deg stall of a policy that already tips.
#
# warm-inv-pin tests that directly. It warm-starts run 14's model_1500 (past
# the step, tipping learned) and trains to 3000 with two changes:
#   --orientation-kernel inverse   1/(err+0.1) x8.93: same pull as the linear
#                                  term at 130 deg, 18x its pull at 26 deg, and
#                                  still rising at the goal instead of flat below
#                                  11.5 deg.
#   --goal-drift/resample False    goal pinned for the episode. Under the drift
#                                  a success kicks the goal ~160 deg away; with
#                                  the inverse kernel that costs ~190 discounted
#                                  reward against a +5 bonus (-94 even at a 2000
#                                  bonus), so the policy would be trained never
#                                  to cross the threshold. Pinned, the optimum is
#                                  to arrive and stay.
# Control: run 14's own model_2999 -- same start, same 1500 extra iterations,
# old reward. Scored in the REFERENCE env (drift on, the real task) and in the
# pinned env (HELD at 0.1 is only possible without the kick).
#
# Works if median best error in the reference env falls below ~20 deg (control
# 27.7) AND at least one episode gets under 0.1 rad -- a first. Null if best
# error stays within a few deg of the control with 0 successes: then an 18x
# stronger pull with no kick does not move the stall, it is not a reward
# equilibrium, and the next step is structural (tip-axis waypoint / skills).
# Abort if training error climbs past ~70 deg for 200 iterations, action std
# passes ~4, or the critic loss goes NaN.
#
# warm-inv-pin was STOPPED at iteration 1650 on that std criterion: action std
# rose in a straight line, 2.81 -> 4.03 by 1637 (~0.009/iter), drops/ep 0.15 ->
# ~0.5 -- the runaway runs 20 and 22 showed. rsl_rl's Gaussian is unbounded, so
# the entropy bonus pushes log(std) up at a constant rate; run 14's reward held
# it at 2.81 and the new reward does not. Its 1650 checkpoint was unchanged from
# the control on the median (25.2 vs 25.4 deg), so it is left out of the queue
# rather than resumed.
#
# warm-inv-pin-ent1e3 is the same cell with --entropy-coef 0.001, which takes
# away most of that constant push. Run 21 showed 1e-3 on its own does not move
# the stall (std 0.39, same ~30 deg), so it should remove the runaway without
# being the explanation for any improvement. Check std at iteration ~1600: if it
# still climbs, the new reward itself pays for noise and the kernel needs
# capping inside the threshold.
#
# RESULT (2026-09-11 21:28): warm-inv-pin-ent1e3 WORKED. Reference env, 3 eval
# seeds x ~128 episodes, against run 14's model_2999: 31% of episodes reach the
# goal vs 1.8%, median best error 11-13 vs 28-30 deg, tip-over closure 90% vs
# 72%, total 91% -- outside the 77.6-81.4% band every earlier cell stayed in.
# (Only visible after the 2026-09-11 eval_policy.py fix: before it, drift-on
# evals measured the already-kicked goal and could not record a success.)
# Goals per episode were still rising at 2999 (0.25/0.30/0.34/0.40 at
# 2500/2700/2900/2999), so warm-inv-pin-ent1e3-ext continues the SAME config
# from its model_2999 to 4500. Same flags; --init-from gives it its own
# directory and eval labels, and iteration numbering carries on from 2999.
RUN_ENT1E3_FINAL="$REPO/logs/rsl_rl/$EXPERIMENT/2026-09-11_19-42-36_warm-inv-pin-ent1e3/model_2999.pt"
QUEUE=(
  "warm-inv-pin-ent1e3-ext|4500|42|--cube-priority 0 --orientation-kernel inverse --goal-drift False --goal-resample-on-success False --entropy-coef 0.001 --init-from $RUN_ENT1E3_FINAL|--cube-priority 0|--cube-priority 0 --goal-drift False --goal-resample-on-success False"
)

mkdir -p "$CONSOLE_DIR" "$EVAL_DIR"
cd "$REPO" || exit 1

exec 8>"$QUEUE_LOCK"
if ! flock -n 8; then
  echo "[$(date -Is)] another queue holds the lock; exiting" >>"$QUEUE_LOG"
  exit 0
fi

log() { echo "[$(date -Is)] $*" | tee -a "$QUEUE_LOG"; }

latest_iter() {  # $1 = run name
  ls -1 "$REPO/logs/rsl_rl/$EXPERIMENT"/*_"$1"/model_*.pt 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1
}
final_ckpt() {   # $1 = run name, $2 = max iters
  ls -1 "$REPO/logs/rsl_rl/$EXPERIMENT"/*_"$1"/model_$(($2 - 1)).pt 2>/dev/null | head -1
}

# $1 = label, $2 = checkpoint, $3 = REPORTING threshold (rad), $4.. = env
# overrides. Threshold 0.1 keeps the historical ${label}_det.json name so every
# earlier comparison still lines up; any other threshold gets a suffix.
score() {
  local label=$1 ckpt=$2 thr=$3; shift 3
  [ "$EVAL_NUM_ENVS" = "32" ] || label="${label}_n${EVAL_NUM_ENVS}"
  [ "$thr" = "0.1" ] || label="${label}_at$(printf '%03d' "$(awk "BEGIN{print int($thr*100+0.5)}")")"
  local json="$EVAL_DIR/${label}_det.json"
  if [ -f "$json" ]; then log "eval $label already scored — skipping"; return 0; fi
  log "evaluating $label — $(basename "$(dirname "$ckpt")")/$(basename "$ckpt")"
  "$UV" run python scripts/eval_policy.py "$ckpt" \
      --num-envs "$EVAL_NUM_ENVS" --num-steps 700 --seed 7 --success-threshold "$thr" \
      --json-out "$json" "$@" >>"$EVAL_DIR/${label}_eval.txt" 2>&1
  log "eval $label finished rc=$?"
}
REPORT_THRESHOLDS=(0.1 0.4)

log "=== queue start (pid $$) — ${#QUEUE[@]} runs ==="

# Baselines, once each. Cheap, and every comparison below needs them.
PINNED=(--goal-drift False --goal-resample-on-success False)
if [ -f "$RUN14_FINAL" ]; then
  for thr in "${REPORT_THRESHOLDS[@]}"; do
    score "run14_iter2999" "$RUN14_FINAL" "$thr" --cube-priority 0
    score "run14_iter2999_pinned" "$RUN14_FINAL" "$thr" --cube-priority 0 "${PINNED[@]}"
  done
else
  log "WARNING: $RUN14_FINAL missing — no control for warm-started runs"
fi
if [ -f "$RUN14_MATCHED" ]; then
  # --cube-priority 0 is REQUIRED, not cosmetic: run 14 predates the flag and
  # trained under element-wise-max mixing, while get_cube_spec now defaults to
  # priority 1. The cached eval/run14_iter1500_det.json happens to be correct
  # because it was scored on 2026-09-02 19:42, hours before the default
  # changed -- delete that file and rescore without this flag and the baseline
  # silently moves. See TRAINING_NOTES.md runs 22-24.
  for thr in "${REPORT_THRESHOLDS[@]}"; do
    score "run14_iter1500" "$RUN14_MATCHED" "$thr" --cube-priority 0
  done
else
  log "WARNING: $RUN14_MATCHED missing — no iteration-matched baseline"
fi

for entry in "${QUEUE[@]}"; do
  IFS='|' read -r name iters seed train_args eval_args trainenv_args <<<"$entry"

  if [ -f "$STOP_FILE" ]; then log "STOP_SUPERVISOR present; standing down"; exit 0; fi

  it=$(latest_iter "$name"); it=${it:-none}
  log "--- run '$name' (target $iters, newest checkpoint: $it) ---"

  printf 'RUN_NAME=%s\nMAX_ITERS=%s\nNUM_ENVS=%s\nSEED=%s\nSAVE_INTERVAL=%s\nEXTRA_ARGS=%s\n' \
    "$name" "$iters" "$NUM_ENVS" "$seed" "$SAVE_INTERVAL" "$train_args" >"$STATE_FILE"

  # supervise_run.sh reads STATE_FILE, exits 0 immediately if already complete,
  # and otherwise trains/resumes until it is.
  "$REPO/scripts/supervise_run.sh"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "supervisor for '$name' exited rc=$rc — stopping the queue"
    exit "$rc"
  fi

  ckpt=$(final_ckpt "$name" "$iters")
  if [ -z "$ckpt" ]; then
    log "'$name' finished but model_$((iters - 1)).pt is missing — stopping"
    exit 1
  fi
  # shellcheck disable=SC2086
  for thr in "${REPORT_THRESHOLDS[@]}"; do
    score "$name" "$ckpt" "$thr" $eval_args
  done
  if [ -n "${trainenv_args:-}" ]; then
    for thr in "${REPORT_THRESHOLDS[@]}"; do
      # shellcheck disable=SC2086
      score "${name}_trainenv" "$ckpt" "$thr" $trainenv_args
    done
  fi
done

rm -f "$STATE_FILE"
log "=== queue done — eval/*_det.json hold the numbers ==="
