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

# name | max_iters | seed | train overrides | eval overrides (must rebuild the SAME env)
#
# reference-seed7 (run 24) ANSWERED the variance question it was queued for:
# it reproduces run 14 to within 1-2 deg of deterministic best error, so the
# n=1 A/Bs in TRAINING_NOTES.md are readable after all -- but only at a
# resolution of ~3 deg. orientation-fine (run 23) is abandoned at 650/1500: it
# is 27 deg behind run 14 at matched iteration 600, so finishing it buys
# nothing. Both are left out of the queue rather than deleted from the logs.
#
# OBSERVATION NOISE. Cube size (runs 29-30) is the fourth null, and the shape of
# the nulls is now the finding: condim 6, friction and size each IMPROVED
# tip-over closure and DEGRADED spin closure by comparable amounts, leaving the
# total at 26-27 deg every time.
#
#                    spin down   tip down   total down   best
#     run 14           93.5%      66.9%       80.1%      26.3 deg
#     friction (26)    76.7%      73.1%       81.4%      25.1 deg
#     cube-0325 (29)   91.9%      71.9%       77.6%      27.1 deg
#
# A capability limit does not rebalance. An equilibrium does, and there is a
# mechanism for one: cube_orientation_tolerance is
# tolerance(err, bounds=(0,0.2), margin=pi, sigmoid="linear"), i.e. LINEAR in
# error from 180 deg down to 11.5 deg and flat below. Linear means the marginal
# reward for closing one more degree is the same at 130 deg as at 30 deg, while
# the marginal cost of closing it rises steeply near the goal. The policy stops
# where those meet, and that point belongs to the reward, not the hand -- change
# the contact model and it re-allocates between spin and tip without moving the
# total. The bare hand works on this identical reward because a lower cost of
# precision puts the same equilibrium inside 0.1 rad, where the 100-point
# success bonus fires and bootstraps.
#
# These cells attack the other half: whether the policy can PERCEIVE the target
# it is scored on. cube_ori observation noise is +/-0.1 on rotation-matrix
# entries -- per component std 0.058, tilting a unit column by ~0.082 rad
# ~= 4.7 deg -- against a success threshold of 0.1 rad = 5.7 deg. The
# measurement noise is the same order as the target, so the gradient the policy
# would need to servo inside the threshold is buried in its own observation
# noise.
#
# The corroborating detail: eval already runs with corruption OFF (play=True
# sets enable_corruption=False, env_cfg.py) and the policy STILL stops at
# 26-27 deg with clean observations. That is the signature of a policy trained
# blind, not one blinded at test time.
#
# --obs-noise-scale is playground's own obs_noise.level (default_config has
# level=1.0 over scales joint_pos 0.05 / cube_pos 0.02 / cube_ori 0.1), which
# this port hard-coded rather than exposed. 1.0 is exact parity, so this is
# turning a knob the reference already has, not inventing a deviation.
#
# --cube-priority 0 restores run 14's contact model, so noise is the ONLY
# difference from run 14 and each comparison is single-variable.
#
# Expect from a fix: best error falls below 26.3 deg by more than the ~3 deg
# resolution floor, and held-success rises above 0 for the first time. If
# zero noise still scores 0/32, perception is excluded, the equilibrium story
# stands alone, and the reward SHAPE is the only lever left -- switch the
# orientation term's sigmoid from "linear" to the convex _long_tail_tolerance
# already sitting unused at rewards.py:49.
QUEUE=(
  "obsnoise-0|1500|42|--cube-priority 0 --obs-noise-scale 0.0|--cube-priority 0 --obs-noise-scale 0.0"
  "obsnoise-half|1500|42|--cube-priority 0 --obs-noise-scale 0.5|--cube-priority 0 --obs-noise-scale 0.5"
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

score() {        # $1 = label, $2 = checkpoint, $3.. = env overrides
  local label=$1 ckpt=$2; shift 2
  local json="$EVAL_DIR/${label}_det.json"
  if [ -f "$json" ]; then log "eval $label already scored — skipping"; return 0; fi
  log "evaluating $label — $(basename "$(dirname "$ckpt")")/$(basename "$ckpt")"
  "$UV" run python scripts/eval_policy.py "$ckpt" \
      --num-envs 32 --num-steps 700 --seed 7 \
      --json-out "$json" "$@" >>"$EVAL_DIR/${label}_eval.txt" 2>&1
  log "eval $label finished rc=$?"
}

log "=== queue start (pid $$) — ${#QUEUE[@]} runs ==="

# Iteration-matched baseline, once. Cheap, and every comparison below needs it.
if [ -f "$RUN14_MATCHED" ]; then
  # --cube-priority 0 is REQUIRED, not cosmetic: run 14 predates the flag and
  # trained under element-wise-max mixing, while get_cube_spec now defaults to
  # priority 1. The cached eval/run14_iter1500_det.json happens to be correct
  # because it was scored on 2026-09-02 19:42, hours before the default
  # changed -- delete that file and rescore without this flag and the baseline
  # silently moves. See TRAINING_NOTES.md runs 22-24.
  score "run14_iter1500" "$RUN14_MATCHED" --cube-priority 0
else
  log "WARNING: $RUN14_MATCHED missing — no iteration-matched baseline"
fi

for entry in "${QUEUE[@]}"; do
  IFS='|' read -r name iters seed train_args eval_args <<<"$entry"

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
  score "$name" "$ckpt" $eval_args
done

rm -f "$STATE_FILE"
log "=== queue done — eval/*_det.json hold the numbers ==="
