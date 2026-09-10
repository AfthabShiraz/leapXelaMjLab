#!/usr/bin/env bash
# Wait for run 21 to finish, then score it and run 14 deterministically with
# IDENTICAL eval settings so the two numbers are directly comparable.
set -uo pipefail
REPO=/home/afthabshiraz/MujocoRL-Internship/leapXelaMjLab
UV=/home/afthabshiraz/.local/bin/uv
cd "$REPO" || exit 1
mkdir -p eval
OUT=eval/AUTO_EVAL_REPORT.txt
R21_DIR=$(ls -d logs/rsl_rl/leap_xela_cube_reorient_reference/*_entropy-1e-3 2>/dev/null | head -1)
R14=logs/rsl_rl/leap_xela_cube_reorient_reference/2026-08-29_16-11-45_reference-v3-eulerdamp/model_2999.pt
log(){ echo "[$(date -Is)] $*" >>"$OUT"; }

# A reboot re-fires the @reboot hook. If the eval already completed, do not
# truncate the report and re-run it -- the results are the artifact.
if [ -f "$OUT" ] && grep -q "ALL DONE" "$OUT"; then exit 0; fi

: >"$OUT"
log "waiting for run 21 (entropy-1e-3) to reach iteration 1499"
# Wait up to 8h; survives a reboot-driven resume because it only watches for the file.
for _ in $(seq 1 2880); do
  R21_DIR=$(ls -d logs/rsl_rl/leap_xela_cube_reorient_reference/*_entropy-1e-3 2>/dev/null | head -1)
  [ -n "$R21_DIR" ] && [ -f "$R21_DIR/model_1499.pt" ] && break
  sleep 10
done
if [ ! -f "$R21_DIR/model_1499.pt" ]; then log "TIMED OUT — model_1499.pt never appeared"; exit 1; fi
log "found $R21_DIR/model_1499.pt"
# Let the trainer exit and free the GPU before evaluating.
for _ in $(seq 1 60); do pgrep -f "train.py .*entropy-1e-3" >/dev/null || break; sleep 10; done
sleep 20

for pair in "run21:$R21_DIR/model_1499.pt" "run14:$R14"; do
  name=${pair%%:*}; ckpt=${pair#*:}
  log "evaluating $name — $ckpt"
  { echo; echo "################ $name — $ckpt ################"; } >>"$OUT"
  "$UV" run python scripts/eval_policy.py "$ckpt" \
      --num-envs 32 --num-steps 700 --seed 7 \
      --json-out "eval/${name}_det.json" >>"$OUT" 2>&1
  log "$name done (rc=$?)"
done
log "ALL DONE — compare the two blocks above; eval/run21_det.json and eval/run14_det.json hold the raw numbers"
