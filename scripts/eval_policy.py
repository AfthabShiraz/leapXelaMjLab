"""Measure the DETERMINISTIC terminal precision of a trained reorient policy.

Training logs `orientation_error` under the *stochastic* policy. rsl_rl's
GaussianDistribution is unbounded, so the entropy bonus drives its std up
without limit; runs 12-19 converge at std 2.79, which at action scale 0.5 is a
commanded joint delta of std 1.40 rad per 50 ms control step. The logged error
therefore measures a policy that is being shaken hard, not the policy that would
be deployed. This script rolls out the MEAN action instead and reports what the
policy can actually hold.

Three things this measures that a naive "final error" report would get wrong:

1. The goal RUNS AWAY after a success. ``InHandReorientationCommand`` with
   ``use_mjx_goal_drift=True`` kicks the goal with 1-5 rad/s per axis whenever
   the error drops under the threshold, decaying 0.8 per step. So the minimum
   error reached and the error at the end of the episode are genuinely different
   quantities and both are reported.
2. The error is decomposed in the WORLD frame. ``quat_box_minus(goal, cube)``
   is ``log(goal * cube^-1)``, a left-multiplied (world-frame) rotation vector r
   whose norm is exactly the logged ``orientation_error``. World +z is the palm
   normal, so |r_z| is the "spin the cube in the fingertips" component and
   |r_xy| the "tip it over" component, with |r|^2 = r_z^2 + |r_xy|^2 exactly.
   The notes record that the policy closes the spin component and stalls on the
   tip-over one; this makes that split reproducible.
3. Episodes reset on termination, so every statistic is computed per
   episode-segment rather than over a whole env's 700-step trace.

Usage:
  uv run python scripts/eval_policy.py logs/rsl_rl/.../model_2999.pt --num-envs 64
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

from load_env import load_project_dotenv

load_project_dotenv()

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_box_minus
from mjlab.utils.torch import configure_torch_backends

# The one LeapXELA configuration that learns to reorient (palm 1.92, cube 1.0);
# runs 12-20 in TRAINING_NOTES.md all use it.
_DEFAULT_TASK = "Mjlab-LeapXELA-Cube-Reorient-Reference"

_RAD2DEG = 180.0 / np.pi


@dataclass(frozen=True)
class EvalConfig:
  checkpoint: tyro.conf.Positional[str]
  """Path to an rsl_rl checkpoint (.pt)."""
  task: str = _DEFAULT_TASK
  num_envs: int = 64
  num_steps: int = 700
  """Control steps to roll out. The env steps at 20 Hz, episodes cap at 1000."""
  seed: int = 42
  success_threshold: float = 0.1
  """Reporting threshold only (rad). The env's own goal-drift trigger is
  whatever the task was configured with and is deliberately left alone, so the
  dynamics stay identical to training."""
  min_episode_steps: int = 50
  """Segments shorter than this are dropped from the precision statistics. A
  two-step segment (cube already falling at reset) has no meaningful "minimum
  error reached" and would otherwise dominate the median."""
  play_env: bool = False
  """Use the task's play config (no observation noise, no domain
  randomization). Default is False so the measurement is taken under the same
  env the checkpoint was trained in, which is what the logged metrics reflect."""
  device: str | None = None
  json_out: str | None = None
  # Cube physics overrides -- these must reconstruct the env the checkpoint was
  # TRAINED with. --cube-condim especially: run 19's checkpoints are condim 6,
  # and scoring them under the condim-3 default measures a different physical
  # system (condim 3 makes torsional and rolling friction inert entirely).
  cube_half_size: float | None = None
  cube_friction_sliding: float | None = None
  cube_friction_torsional: float | None = None
  cube_condim: int | None = None


def _build_env_cfg(cfg: EvalConfig):
  """Load the task env config and apply any cube overrides.

  Reuses ``train.py``'s ``_apply_cube_overrides`` verbatim so an eval env built
  with ``--cube-condim 6`` is byte-for-byte the env that flag produced at
  training time -- including the njmax bump and the deliberate choice not to
  disable cube-friction DR for a condim-only override.
  """
  from train import TrainConfig, _apply_cube_overrides

  env_cfg = load_env_cfg(cfg.task, play=cfg.play_env)
  overrides = TrainConfig(
    cube_half_size=cfg.cube_half_size,
    cube_friction_sliding=cfg.cube_friction_sliding,
    cube_friction_torsional=cfg.cube_friction_torsional,
    cube_condim=cfg.cube_condim,
  )
  overridden = _apply_cube_overrides(env_cfg, overrides, cfg.task)
  return overridden if overridden is not None else env_cfg


def _segment(done: np.ndarray) -> list[tuple[int, int]]:
  """Split one env's trace into [start, end] index pairs, one per episode.

  ``done[t]`` means the env terminated *during* step t and was reset before the
  sample at index t was taken (mjlab resets inside ``step()``, then calls
  ``sim.forward()`` and ``command_manager.compute()``). So index t is the first
  sample of a NEW episode and the preceding episode ends at t-1. The terminal
  state itself is never recorded, which is what we want -- a fallen cube's pose
  is not part of the precision the policy achieved.
  """
  starts = [0] + [int(t) for t in np.nonzero(done)[0] if t > 0]
  return [(s, e - 1) for s, e in zip(starts, starts[1:] + [len(done)])]


def _spread(values: np.ndarray) -> dict[str, float]:
  if values.size == 0:
    return {"median": float("nan"), "p10": float("nan"), "p90": float("nan"),
            "min": float("nan"), "max": float("nan"), "mean": float("nan"), "n": 0}
  return {
    "median": float(np.median(values)),
    "p10": float(np.percentile(values, 10)),
    "p90": float(np.percentile(values, 90)),
    "min": float(values.min()),
    "max": float(values.max()),
    "mean": float(values.mean()),
    "n": int(values.size),
  }


def _rollout(cfg: EvalConfig) -> dict:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = _build_env_cfg(cfg)
  agent_cfg = load_rl_cfg(cfg.task)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(
    env, clip_actions=getattr(agent_cfg, "clip_actions", None)
  )

  # Same construction as train.py / render.py: MjlabOnPolicyRunner strips the
  # None optional model fields that plain rsl_rl chokes on. Building the full
  # runner (rather than hand-loading the state dict) is what guarantees the
  # EmpiricalNormalization buffers are restored and applied exactly as at
  # training time -- getting that wrong produces silent garbage.
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  with tempfile.TemporaryDirectory() as tmp:
    runner = runner_cls(wrapped, asdict(agent_cfg), tmp, device=device)
    runner.load(cfg.checkpoint, map_location=device)
  # get_inference_policy() calls alg.eval_mode() and hands back the MLPModel.
  # MLPModel.forward defaults to stochastic_output=False, i.e. it returns
  # distribution.deterministic_output(...) == the Gaussian MEAN, never a sample.
  # eval_mode() also stops EmpiricalNormalization.update() (it early-returns on
  # `not self.training`), so the normalizer stays frozen at the trained stats.
  policy = runner.get_inference_policy(device=device)

  actor = runner.alg.get_policy()
  assert actor.obs_normalization, "checkpoint was trained without obs normalization?"
  std = actor.distribution.std_param.detach()
  print(
    f"[INFO] policy loaded; learned action std = {std.mean().item():.3f} "
    "(NOT used -- rolling out the mean)"
  )

  cube = env.scene["cube"]
  command = env.command_manager.get_term("goal_orientation")

  n_rec = cfg.num_steps + 1
  err = torch.zeros(n_rec, cfg.num_envs, device=device)
  r_z = torch.zeros(n_rec, cfg.num_envs, device=device)
  r_xy = torch.zeros(n_rec, cfg.num_envs, device=device)
  ang_speed = torch.zeros(n_rec, cfg.num_envs, device=device)
  done_buf = torch.zeros(n_rec, cfg.num_envs, dtype=torch.bool, device=device)
  fell_buf = torch.zeros(n_rec, cfg.num_envs, dtype=torch.bool, device=device)

  def record(t: int) -> None:
    # World-frame rotation vector taking the cube to the goal. quat_box_minus is
    # log(q1 * q2^-1) -- left multiplication, so r lives in the world frame and
    # r_z is the palm-normal (spin) component by construction.
    r = quat_box_minus(command.command, cube.data.root_link_quat_w)
    err[t] = torch.linalg.vector_norm(r, dim=-1)
    r_z[t] = r[:, 2].abs()
    r_xy[t] = torch.linalg.vector_norm(r[:, :2], dim=-1)
    ang_speed[t] = torch.linalg.vector_norm(cube.data.root_link_ang_vel_w, dim=-1)

  obs, _ = wrapped.reset()
  # env.reset() ends with command_manager.compute(), exactly as step() does, so
  # this sample is the same kind of object as a post-reset mid-rollout sample.
  record(0)
  print(f"[INFO] rolling out {cfg.num_steps} steps x {cfg.num_envs} envs on {device}")
  with torch.inference_mode():
    for i in range(cfg.num_steps):
      actions = policy(obs)
      obs, _, dones, _ = wrapped.step(actions)
      t = i + 1
      record(t)
      done_buf[t] = dones.bool()
      fell_buf[t] = env.termination_manager.get_term("cube_fell")
      if (i + 1) % 200 == 0:
        print(f"       {i + 1}/{cfg.num_steps} steps")

  env.close()
  return {
    "err": err.cpu().numpy(),
    "r_z": r_z.cpu().numpy(),
    "r_xy": r_xy.cpu().numpy(),
    "ang_speed": ang_speed.cpu().numpy(),
    "done": done_buf.cpu().numpy(),
    "fell": fell_buf.cpu().numpy(),
  }


def _analyze(trace: dict, cfg: EvalConfig) -> dict:
  err, r_z, r_xy = trace["err"], trace["r_z"], trace["r_xy"]
  ang, done, fell = trace["ang_speed"], trace["done"], trace["fell"]
  n_rec, num_envs = err.shape

  episodes: list[dict] = []
  n_short = 0
  total_drops = 0
  for e in range(num_envs):
    segs = _segment(done[:, e])
    for k, (s, t_end) in enumerate(segs):
      # A segment is closed by whatever fired on the step that started the NEXT
      # segment; the trailing segment is simply cut off by the rollout window.
      closed_by_fall = k + 1 < len(segs) and bool(fell[segs[k + 1][0], e])
      total_drops += int(closed_by_fall)
      length = t_end - s + 1
      if length < cfg.min_episode_steps:
        n_short += 1
        continue
      window = slice(s, t_end + 1)
      seg_err = err[window, e]
      i_best = int(np.argmin(seg_err))
      n_early = max(1, int(np.ceil(0.2 * length)))
      i_late = int(np.floor(0.5 * length))
      episodes.append(
        {
          "env": e,
          "start_step": s,
          "length": length,
          "terminated_by_fall": closed_by_fall,
          "start_err": float(seg_err[0]),
          "min_err": float(seg_err[i_best]),
          "final_err": float(seg_err[-1]),
          "best_step": i_best,
          "reached_threshold": bool((seg_err < cfg.success_threshold).any()),
          "start_r_z": float(r_z[s, e]),
          "start_r_xy": float(r_xy[s, e]),
          "best_r_z": float(r_z[s + i_best, e]),
          "best_r_xy": float(r_xy[s + i_best, e]),
          "ang_speed_early": float(ang[s : s + n_early, e].mean()),
          "ang_speed_late": float(ang[s + i_late : t_end + 1, e].mean()),
        }
      )

  def col(key: str, scale: float = 1.0) -> np.ndarray:
    return np.array([ep[key] for ep in episodes], dtype=np.float64) * scale

  def reduction(start_key: str, best_key: str) -> dict[str, float]:
    """Median of the PER-EPISODE percent reduction.

    Not 1 - median(best)/median(start): medians of a ratio and a ratio of
    medians differ, and the per-episode form is what the 2026-08-31 probe
    reported (74.2% total / 88.1% r_z / 64.3% r_xy for run 14).
    """
    a, b = col(start_key), col(best_key)
    if a.size == 0:
      return {"median_pct": float("nan"), "ratio_of_medians_pct": float("nan")}
    per_ep = 100.0 * (1.0 - b / np.maximum(a, 1e-9))
    return {
      "median_pct": float(np.median(per_ep)),
      "ratio_of_medians_pct": float(100.0 * (1.0 - np.median(b) / np.median(a))),
    }

  n_ep = len(episodes)
  return {
    "config": {
      "checkpoint": cfg.checkpoint,
      "task": cfg.task,
      "num_envs": cfg.num_envs,
      "num_steps": cfg.num_steps,
      "seed": cfg.seed,
      "success_threshold_rad": cfg.success_threshold,
      "min_episode_steps": cfg.min_episode_steps,
      "play_env": cfg.play_env,
      "cube_condim": cfg.cube_condim,
      "cube_half_size": cfg.cube_half_size,
      "cube_friction_sliding": cfg.cube_friction_sliding,
      "cube_friction_torsional": cfg.cube_friction_torsional,
    },
    "n_episodes": n_ep,
    "n_short_episodes_excluded": n_short,
    "error_deg": {
      "start": _spread(col("start_err", _RAD2DEG)),
      "min": _spread(col("min_err", _RAD2DEG)),
      "final": _spread(col("final_err", _RAD2DEG)),
      "drift_back_up": _spread(col("final_err", _RAD2DEG) - col("min_err", _RAD2DEG)),
    },
    "best_step": _spread(col("best_step")),
    "success_fraction": float(col("reached_threshold").mean()) if n_ep else float("nan"),
    "n_success": int(col("reached_threshold").sum()) if n_ep else 0,
    "decomposition_deg": {
      "total": {"start": _spread(col("start_err", _RAD2DEG)),
                "at_best": _spread(col("min_err", _RAD2DEG)),
                "reduction": reduction("start_err", "min_err")},
      "r_z": {"start": _spread(col("start_r_z", _RAD2DEG)),
              "at_best": _spread(col("best_r_z", _RAD2DEG)),
              "reduction": reduction("start_r_z", "best_r_z")},
      "r_xy": {"start": _spread(col("start_r_xy", _RAD2DEG)),
               "at_best": _spread(col("best_r_xy", _RAD2DEG)),
               "reduction": reduction("start_r_xy", "best_r_xy")},
    },
    "cube_ang_speed": {
      "first_20pct": _spread(col("ang_speed_early")),
      "last_50pct": _spread(col("ang_speed_late")),
    },
    "drops": {
      "total": total_drops,
      "per_env_rollout": total_drops / cfg.num_envs,
      "episodes_ended_by_fall": int(col("terminated_by_fall").sum()) if n_ep else 0,
    },
    "episodes": episodes,
  }


def _fmt(s: dict, unit: str = "") -> str:
  return (
    f"{s['median']:8.2f}{unit}  [p10 {s['p10']:7.2f}, p90 {s['p90']:7.2f}]  "
    f"min {s['min']:7.2f}  max {s['max']:7.2f}"
  )


def _report(res: dict) -> None:
  c = res["config"]
  thr_deg = c["success_threshold_rad"] * _RAD2DEG
  bar = "=" * 78
  print(f"\n{bar}\nDETERMINISTIC (mean-action) EVALUATION\n{bar}")
  print(f"  checkpoint     {c['checkpoint']}")
  print(f"  task           {c['task']}")
  print(
    f"  rollout        {c['num_envs']} envs x {c['num_steps']} steps, seed {c['seed']}"
    f"{', play env' if c['play_env'] else ''}"
  )
  if c["cube_condim"] is not None:
    print(f"  cube_condim    {c['cube_condim']} (override)")
  print(
    f"  episodes       {res['n_episodes']} scored"
    f" ({res['n_short_episodes_excluded']} shorter than"
    f" {c['min_episode_steps']} steps excluded)"
  )

  print(f"\n-- orientation error (deg), per episode ------------------------------")
  print("                       median      spread                    range")
  for label, key in (("start", "start"), ("minimum reached", "min"), ("final", "final")):
    print(f"  {label:<20s} {_fmt(res['error_deg'][key])}")
  print(f"  {'drift after best':<20s} {_fmt(res['error_deg']['drift_back_up'])}")
  print(
    f"  best reached at step {res['best_step']['median']:.0f} of the episode "
    "(median) -- the goal drifts away after a success, so 'final' is not 'best'"
  )

  print(f"\n-- success (error < {thr_deg:.1f} deg at any point) -------------------------")
  print(
    f"  {res['n_success']}/{res['n_episodes']} episodes "
    f"= {100.0 * res['success_fraction']:.1f}%"
  )

  print("\n-- world-frame error decomposition (deg): |r|^2 = r_z^2 + |r_xy|^2 -----")
  print(f"  {'component':<10s} {'start':>10s} {'at best':>10s} {'reduction':>12s}")
  names = {"total": "total |r|", "r_z": "spin  r_z", "r_xy": "tip  r_xy"}
  for key, name in names.items():
    d = res["decomposition_deg"][key]
    print(
      f"  {name:<10s} {d['start']['median']:10.1f} {d['at_best']['median']:10.1f} "
      f"{d['reduction']['median_pct']:11.1f}%"
    )
  print(
    "  (reduction is the median of per-episode reductions; medians of start and "
    "at-best\n   are reported separately and do not combine in quadrature)"
  )

  print("\n-- cube angular speed (rad/s): approach phase vs stall -----------------")
  print(f"  {'first 20% of episode':<22s} {_fmt(res['cube_ang_speed']['first_20pct'])}")
  print(f"  {'last 50% of episode':<22s} {_fmt(res['cube_ang_speed']['last_50pct'])}")

  print("\n-- drops (cube_fell) ---------------------------------------------------")
  print(
    f"  {res['drops']['total']} total = "
    f"{res['drops']['per_env_rollout']:.2f} per env-rollout "
    f"({c['num_steps']} steps)"
  )
  print(bar + "\n")


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import leap_xela_mjlab.tasks  # noqa: F401

  cfg = tyro.cli(EvalConfig, prog=sys.argv[0])
  known = list_tasks()
  if cfg.task not in known:
    raise SystemExit(f"Unknown task {cfg.task!r}. Registered: {sorted(known)}")
  if not Path(cfg.checkpoint).is_file():
    raise SystemExit(f"No such checkpoint: {cfg.checkpoint}")

  trace = _rollout(cfg)
  res = _analyze(trace, cfg)
  _report(res)

  if cfg.json_out is not None:
    out = Path(cfg.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[INFO] wrote {out}")


if __name__ == "__main__":
  main()
