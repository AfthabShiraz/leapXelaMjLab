"""Score an eval JSON on GOAL ACQUISITION rather than on terminal precision.

Why a second scorer. ``eval_policy.py`` answers "how close does it get and does
it stay there", which was the right question while the policy stalled at 25-30
deg. The converged policy's precision is no longer the binding constraint --
pinned, it reaches 0.73 deg and holds -- and what limits the reference-env score
is how LONG a goal takes to acquire. Measured 2026-09-12 on the mainline:

  * time to acquisition is uncorrelated with the rotation demanded
    (Pearson r = -0.13 overall, +0.01 against tip demanded)
  * the cube travels ~7x the geodesic before the goal is caught
  * the hazard of first acquisition is flat from step 100 to step 700, so the
    episodes that never acquire are the tail of a constant-rate process, not a
    stuck mode

so the quantities that move the headline are the ones below. ``PATH EFFICIENCY``
is approximate: travelled distance is estimated from the episode's two logged
mean angular speeds, not integrated from a full trace, so read it as a ratio
between runs and not as an absolute.

Usage:
  python scripts/acq_metrics.py eval/a_det.json [eval/b_det.json ...] [--label NAME]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import numpy as np

DEG = 180.0 / math.pi


def _q(v, p):
  return float(np.percentile(v, p)) if len(v) else float("nan")


def load(paths: list[str]) -> tuple[list[dict], dict, dict]:
  """Return (full-length episodes, config, whole-rollout totals).

  The episode filter is right for anything measured *per episode* (acquisition
  time, path efficiency), which needs a full episode to be meaningful. It is
  wrong for drops, so those totals are carried separately and unfiltered.
  """
  eps, cfg = [], {}
  tot = {"entries": 0, "drops": 0, "episodes": 0}
  for p in paths:
    d = json.load(open(p))
    cfg = d["config"]
    eps += [e for e in d["episodes"] if e["length"] >= 0.85 * d["config"]["num_steps"]]
    tot["entries"] += d["threshold_entries_total"]
    tot["drops"] += d["drops"]["total"]
    tot["episodes"] += d["n_episodes"]
  return eps, cfg, tot


def report(eps: list[dict], cfg: dict, tot: dict, label: str) -> None:
  entries_total, drops_total, ep_total = tot["entries"], tot["drops"], tot["episodes"]
  n = len(eps)
  steps = cfg["num_steps"]
  dt = 0.05
  if n == 0:
    print(f"{label}: no full-length episodes"); return
  legacy = "first_entry_step" not in eps[0]
  if legacy:
    print("  [WARN] this JSON predates first_entry_step; falling back to best_step on")
    print("         single-entry episodes only, which conditions on the entry count.")

  fail = [e for e in eps if not e["reached_threshold"]]
  succ = [e for e in eps if e["reached_threshold"]]
  if legacy:
    t = np.array([e["best_step"] for e in succ if e["n_threshold_entries"] == 1], float)
  else:
    t = np.array([e["first_entry_step"] for e in succ], float)

  entries = sum(e["n_threshold_entries"] for e in eps)
  ep_s = steps * dt
  print(f"\n=== {label} ===")
  print(f"  episodes {n}   rollout {steps} steps ({ep_s:.0f} s)   seeds/files pooled")
  print(f"  never acquired            {len(fail)}/{n} = {len(fail)/n:6.1%}")
  print(f"  goals per episode         {entries/n:6.2f}      goals per minute  {entries/n/ep_s*60:5.2f}")
  print(f"  time to first goal        p10 {_q(t,10):5.0f}  median {_q(t,50):5.0f}  p90 {_q(t,90):5.0f} steps"
        f"   (median {_q(t,50)*dt:.1f} s)")

  # Path efficiency: geodesic / distance actually travelled getting there.
  eff = []
  for e, tt in zip([e for e in succ if legacy is False or e["n_threshold_entries"] == 1], t):
    spd = (e["ang_speed_early"] + e["ang_speed_late"]) / 2
    if spd > 0 and tt > 0:
      eff.append(e["start_err"] / (spd * tt * dt))
  eff = np.array(eff)
  if len(eff):
    print(f"  path efficiency           p10 {_q(eff,10):5.3f}  median {_q(eff,50):5.3f}  p90 {_q(eff,90):5.3f}"
          f"   (~{1/_q(eff,50):.0f}x the geodesic)")

  mn = np.array([e["min_err"] for e in eps]) * DEG
  print(f"  median best error         {np.median(mn):6.2f} deg")
  # Drops must come from the eval's own whole-rollout accounting, NOT from the
  # full-length episode list: a drop ENDS an episode early, so filtering to
  # full-length episodes systematically discards the very episodes that dropped
  # and inflates goals-per-drop (it read ~60 against the true 10.0 before this
  # was caught, 2026-09-12). Same reason `entries` here is the filtered count and
  # the ratio below uses the unfiltered totals from the JSON.
  print(
    f"  drops {drops_total} over {ep_total} episodes (all lengths)"
    f"   goals per drop {entries_total / max(drops_total, 1):5.1f}"
  )

  # Normalised closure, which is what says whether tip is really the weak axis:
  # goals are drawn Rx(a)*Ry(b) and the error has 2 tip DOF against 1 spin, so
  # absolute magnitudes are tip-heavy before the policy acts.
  for nm, g in (("all", eps), ("failures", fail)):
    if not g:
      continue
    cz = np.median([1 - e["best_r_z"] / max(e["start_r_z"], 1e-9) for e in g]) * 100
    cxy = np.median([1 - e["best_r_xy"] / max(e["start_r_xy"], 1e-9) for e in g]) * 100
    print(f"  closure ({nm:8s})        spin {cz:5.1f}%   tip {cxy:5.1f}%")

  print(f"  entries histogram         {dict(sorted(Counter(e['n_threshold_entries'] for e in eps).items()))}")
  print("  hazard of first acquisition per 100 steps (flat => memoryless):")
  nf = len(fail)
  for a in range(0, steps, 100):
    at_risk = int(np.sum(t >= a)) + nf
    ev = int(np.sum((t >= a) & (t < a + 100)))
    if at_risk:
      print(f"    [{a:4d},{a+100:4d})  at_risk {at_risk:4d}  acquired {ev:4d}  hazard {ev/at_risk:6.1%}")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("json", nargs="+")
  ap.add_argument("--label", default=None)
  a = ap.parse_args()
  eps, cfg, tot = load(a.json)
  report(eps, cfg, tot, a.label or ", ".join(p.split("/")[-1] for p in a.json))


if __name__ == "__main__":
  main()
