#!/usr/bin/env bash
# Install scripts/run_queue.staged.sh over scripts/run_queue.sh once the running
# queue has exited, then make sure a queue is running. Bash reads a script
# incrementally from a byte offset, so overwriting run_queue.sh while it is
# mid-flight can make it resume at the wrong place -- hence the wait.
#
# Safe to run repeatedly and from @reboot: if the staged file is already
# installed it skips the swap, and the queue's own flock stops a second copy.
REPO=/home/afthabshiraz/MujocoRL-Internship/leapXelaMjLab
LIVE=$REPO/scripts/run_queue.sh
STAGED=$REPO/scripts/run_queue.staged.sh
LOG=$REPO/logs/console/CHAIN.log
say() { echo "[$(date -Is)] $*" >>"$LOG"; }

exec 7>"$REPO/logs/console/.chain.lock"
flock -n 7 || { say "another chain holds the lock; exiting"; exit 0; }

if [ ! -s "$STAGED" ]; then say "no staged queue; nothing to do"; exit 0; fi

if ! cmp -s "$LIVE" "$STAGED"; then
  say "waiting for the running queue to exit before swapping"
  # Test the queue's OWN flock rather than pgrep: the shell wrappers that
  # launched it linger in the process table with run_queue.sh in their command
  # line, and a pgrep would match those and wait forever.
  while ! flock -n "$REPO/logs/console/.queue.lock" true 2>/dev/null; do sleep 30; done
  cp "$LIVE" "$LIVE.bak-$(date +%Y%m%d_%H%M%S)"
  cp "$STAGED" "$LIVE"; chmod +x "$LIVE"
  if ! bash -n "$LIVE"; then
    say "FATAL: staged queue fails syntax check; restoring"
    cp "$(ls -t "$LIVE".bak-* | head -1)" "$LIVE"; exit 1
  fi
  say "staged queue installed"
else
  say "staged queue already installed"
fi

CK=$REPO/logs/rsl_rl/leap_xela_cube_reorient_reference/remote_brev-ext/model_8999.pt
[ -s "$CK" ] || { say "FATAL: $CK missing; night-ext cannot warm-start"; exit 1; }

if ! flock -n "$REPO/logs/console/.queue.lock" true 2>/dev/null; then
  say "a queue is already running; leaving it alone"
else
  say "starting queue"
  setsid nohup "$LIVE" >/dev/null 2>&1 < /dev/null &
  say "queue started, pid $!"
fi
