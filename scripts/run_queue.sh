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
SAVE_INTERVAL=25   # ~1.7 min of training per save on the DGX; this host reboots uncleanly

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
# ATTRIBUTION. Run 33 (warm-inv-pin-ent1e3) changed THREE things at once against
# a warm start from run 14's model_1500 -- the inverse orientation kernel, the
# pinned goal, and entropy_coef 1e-3 -- and took episodes reaching 5.7 deg from
# 1.8% to 31.2%. Which of the three is load-bearing is untested.
#
# Leave-one-out, not one-at-a-time: the question is which change is NECESSARY,
# and run 32 already supplies the third cell. Every cell is a warm start from
# run 14 model_1500 to iteration 3000, so all four are directly comparable to
# run 33 @2999 and to the control (run 14's own model_2999, 1.8%).
#
#   full recipe   = inverse kernel + pinned goal + entropy 1e-3   -> run 33, 31.2%
#   minus entropy = inverse kernel + pinned goal + entropy 1e-2   -> run 32, std
#                   runaway 2.81 -> 4.03, stopped at 1650; 25.2 deg, 0/129
#   minus kernel  = LINEAR kernel + pinned goal + entropy 1e-3    -> abl-nokernel
#   minus pin     = inverse kernel + goal DRIFT ON + entropy 1e-3 -> abl-nopin
#
# Pre-registered readings, written before the runs:
#
# abl-nopin is the one with an arithmetic prediction attached. Under the drift a
# success kicks the goal ~160 deg away; discounted at gamma=0.99 that kick costs
# ~190 reward against a +5 bonus with the inverse kernel (-11 with the linear
# one), so the policy should be trained to approach the threshold and NOT cross
# it. Expect best error to improve -- the near-goal pull is real -- while
# successes stay far below run 33's 31.2%. If instead it matches run 33, the
# kick arithmetic is wrong and pinning is unnecessary, which would be the
# cheaper recipe and should become the default.
#
# abl-nokernel is warm-started run 22 with the entropy fix. Pinning removes the
# kick penalty but adds no gradient below 11.5 deg, where the linear term is
# flat, so expect an improvement over the 1.8% control that falls well short of
# 31.2%. If it matches run 33, the kernel is not what did the work and the
# result is about the goal machinery instead.
#
# The likeliest outcome is that BOTH fall short, i.e. the kernel and the pinning
# are necessary together: the kernel supplies the near-goal gradient and the
# pinned goal stops the kick from taxing the policy for using it. That is a
# conclusion about an interaction, and it needs both cells to be legible.
QUEUE=(
  "abl-nokernel|3000|42|--cube-priority 0 --goal-drift False --goal-resample-on-success False --entropy-coef 0.001 --init-from $RUN14_MATCHED|--cube-priority 0|--cube-priority 0 --goal-drift False --goal-resample-on-success False"
  "abl-nopin|3000|42|--cube-priority 0 --orientation-kernel inverse --entropy-coef 0.001 --init-from $RUN14_MATCHED|--cube-priority 0|"
  # The bare LEAP hand -- mujoco_playground's model, no tactile pads, run under
  # the run-14 recipe (linear kernel, drift on, entropy 0.01) so the ONLY
  # difference from run 14 is the hand. This is the control for every "the
  # difference is the skin" claim in TRAINING_NOTES.md, none of which has been
  # tested directly. It reached 1238/3000 by hand on 2026-09-12 23:00 and died
  # in the 00:06 reboot; queued here so the @reboot hook resumes IT rather than
  # the main line. Newest checkpoint model_1225.pt.
  "leap-control|3000|42|--cube-priority 0|--task Mjlab-LEAP-Cube-Reorient-Control --cube-priority 0||Mjlab-LEAP-Cube-Reorient-Control|leap_plain_cube_reorient_control"
  # Overnight, after the ablations: the main line continues on the DGX from where
  # the rented A100 left it at iteration 8999. Same recipe as runs 33/34 and the
  # A100 segment, so the whole curve from 1500 stays one comparable series.
  # Target 14000 is what fits the night at ~4.06 s/iteration; it does not have to
  # finish -- every 25th checkpoint is kept and the run resumes from the newest.
  "night-ext|14000|42|--cube-priority 0 --orientation-kernel inverse --goal-drift False --goal-resample-on-success False --entropy-coef 0.001 --init-from $REPO/logs/rsl_rl/$EXPERIMENT/remote_brev-ext/model_8999.pt|--cube-priority 0|--cube-priority 0 --goal-drift False --goal-resample-on-success False"
  # THE BARE HAND UNDER THE RECIPE THAT WORKS. Run 42 showed the bare LEAP hand
  # beats LeapXELA under the run-14 recipe (8.5% vs 1.6%, 17.0 vs 28.6 deg), but
  # the run-14 recipe is superseded: the main line reaches 82.5% / 5.3 deg with
  # the inverse kernel, the pinned goal and entropy 1e-3 on the TACTILE hand.
  # The open question is whether the hand moves THAT ceiling, and it cannot be
  # answered from run 42.
  #
  # Design mirrors run 33 exactly, with the hand swapped. Run 33 warm-started
  # run 14's model_1500 -- past the iteration-300 tip-over breakthrough that
  # every from-scratch reward edit dies at -- and applied the three changes.
  # leap-control's model_1500 is the same thing for the bare hand: same recipe,
  # same iteration, and it demonstrably tips (84.6% tip closure at 2999).
  #
  # Target 8000 rather than 3000 because checkpoints land every 25 iterations
  # and section 37 already scored the main line retrospectively at 2999 / 4499 /
  # 5500 / 6750 / 8000. The same series on the bare hand gives a curve against a
  # curve, not two points, for one launch of ~6500 iterations (~7.2 h at 4 s).
  #
  # Readings, written before the run:
  #   @2999 vs run 33's 31.2% pooled / 10.78 deg median best
  #   @8000 vs run 37's 82.5% pooled / 5.3 deg median best
  # A win at 8000 says the pads (or the splay cap -- run 41 leaves them
  # confounded) cost the converged policy real performance, and the main line
  # should move to the bare hand for Task 1. A null says run 42's advantage was
  # an artefact of a weak recipe: the bare hand learns the OLD reward faster and
  # both hands saturate the same ceiling once the reward is right, which would
  # make the ceiling a property of the task, not the skin.
  # Abort on the usual criteria: training error above ~70 deg for 200
  # iterations, action std past ~4, or NaN critic loss.
  "bare-inv-pin|8000|42|--cube-priority 0 --orientation-kernel inverse --goal-drift False --goal-resample-on-success False --entropy-coef 0.001 --init-from $REPO/logs/rsl_rl/leap_plain_cube_reorient_control/2026-09-12_23-00-33_leap-control/model_1500.pt|--task Mjlab-LEAP-Cube-Reorient-Control --cube-priority 0|--task Mjlab-LEAP-Cube-Reorient-Control --cube-priority 0 --goal-drift False --goal-resample-on-success False|Mjlab-LEAP-Cube-Reorient-Control|leap_plain_cube_reorient_control"
)

mkdir -p "$CONSOLE_DIR" "$EVAL_DIR"
cd "$REPO" || exit 1

exec 8>"$QUEUE_LOCK"
if ! flock -n 8; then
  echo "[$(date -Is)] another queue holds the lock; exiting" >>"$QUEUE_LOG"
  exit 0
fi

log() { echo "[$(date -Is)] $*" | tee -a "$QUEUE_LOG"; }

# $2 is the experiment tree, which is NOT always $EXPERIMENT: the bare-hand
# control logs under leap_plain_cube_reorient_control so its checkpoints never
# interleave with the XELA main line. Defaulting it here rather than reading the
# global is what lets a non-reference run be queued at all.
latest_iter() {  # $1 = run name, $2 = experiment (default $EXPERIMENT)
  ls -1 "$REPO/logs/rsl_rl/${2:-$EXPERIMENT}"/*_"$1"/model_*.pt 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1
}
final_ckpt() {   # $1 = run name, $2 = max iters, $3 = experiment
  ls -1 "$REPO/logs/rsl_rl/${3:-$EXPERIMENT}"/*_"$1"/model_$(($2 - 1)).pt 2>/dev/null | head -1
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
  IFS='|' read -r name iters seed train_args eval_args trainenv_args task experiment <<<"$entry"
  # Fields 7-8 are optional; an entry that omits them is a reference-task run.
  task="${task:-Mjlab-LeapXELA-Cube-Reorient-Reference}"
  experiment="${experiment:-$EXPERIMENT}"

  if [ -f "$STOP_FILE" ]; then log "STOP_SUPERVISOR present; standing down"; exit 0; fi

  it=$(latest_iter "$name" "$experiment"); it=${it:-none}
  log "--- run '$name' (target $iters, newest checkpoint: $it) ---"

  # TASK and EXPERIMENT belong in the state file for the same reason the rest of
  # it does: the @reboot hook passes no environment. Omitting them cost the
  # bare-hand control 1238 iterations on 2026-09-13 -- the 00:06 reboot fired
  # this queue, which rewrote the state file for its own last entry, and nothing
  # remembered that a plain-LEAP run had been in flight.
  printf 'RUN_NAME=%s\nMAX_ITERS=%s\nNUM_ENVS=%s\nSEED=%s\nSAVE_INTERVAL=%s\nTASK=%s\nEXPERIMENT=%s\nEXTRA_ARGS=%s\n' \
    "$name" "$iters" "$NUM_ENVS" "$seed" "$SAVE_INTERVAL" "$task" "$experiment" "$train_args" >"$STATE_FILE"

  # supervise_run.sh reads STATE_FILE, exits 0 immediately if already complete,
  # and otherwise trains/resumes until it is.
  "$REPO/scripts/supervise_run.sh"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "supervisor for '$name' exited rc=$rc — stopping the queue"
    exit "$rc"
  fi

  ckpt=$(final_ckpt "$name" "$iters" "$experiment")
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
