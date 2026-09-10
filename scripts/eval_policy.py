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
  hold_steps: int = 10
  """Consecutive steps under the threshold required to count as a HELD success.

  The stock criterion is instantaneous -- error below threshold at any single
  sample -- and that turns out to count tumble-throughs. Measured on run 14 at
  2999: 8 of 257 episodes dip under 5.7 deg, but the cube's median angular speed
  AT that instant is 1.21 rad/s, i.e. 3.5 deg of rotation per control step. It
  is passing through the goal region, not arriving in it. 10 steps is 0.5 s.
  """
  hold_ang_speed: float = 0.2
  """Cube angular speed (rad/s) below which it counts as settled rather than
  tumbling. Only 1 of those 257 episodes is under threshold AND under this."""
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
  obs_noise_scale: float | None = None
  cube_half_size: float | None = None
  cube_friction_sliding: float | None = None
  cube_friction_torsional: float | None = None
  cube_condim: int | None = None
  # --cube-priority is the same trap as --cube-condim, and it bites every
  # checkpoint trained before 2026-09-02 21:26. `get_cube_spec` now defaults to
  # priority 1, so the cube outranks the hand and its 0.3 sliding friction is
  # what the grasp sees. Runs 14, 22 and 23 predate the flag entirely: they
  # trained under the element-wise max, where the fingertips' 0.5-1.0 won.
  # Scoring them at the current default silently gives them a weaker grip than
  # they ever trained with. Pass 0 for any checkpoint whose params/env.yaml
  # build_kwargs block has no cube_priority key.
  cube_priority: int | None = None
  # xyz Euler radians of the hand base. All four reorient runs to date use the
  # Box_palm192 fingertip variant with the angle baked in and palm_euler unset,
  # so this is here to keep the reconstruction honest once a run does set it.
  palm_euler: tuple[float, float, float] | None = None
  # Goal-update / reward-shaping overrides -- same requirement as the cube
  # flags: a run 22 checkpoint was trained with the goal pinned for the whole
  # episode, and scoring it under the drift-on default measures a different
  # task, not a worse policy.
  goal_drift: bool | None = None
  goal_resample_on_success: bool | None = None
  orientation_fine: bool | None = None


def _build_env_cfg(cfg: EvalConfig):
  """Load the task env config and apply any cube overrides.

  Reuses ``train.py``'s ``_apply_env_overrides`` verbatim so an eval env built
  with ``--cube-condim 6`` is byte-for-byte the env that flag produced at
  training time -- including the njmax bump and the deliberate choice not to
  disable cube-friction DR for a condim-only override.
  """
  from train import TrainConfig, _apply_env_overrides

  env_cfg = load_env_cfg(cfg.task, play=cfg.play_env)
  overrides = TrainConfig(
    obs_noise_scale=cfg.obs_noise_scale,
    cube_half_size=cfg.cube_half_size,
    cube_friction_sliding=cfg.cube_friction_sliding,
    cube_friction_torsional=cfg.cube_friction_torsional,
    cube_condim=cfg.cube_condim,
    cube_priority=cfg.cube_priority,
    palm_euler=cfg.palm_euler,
    goal_drift=cfg.goal_drift,
    goal_resample_on_success=cfg.goal_resample_on_success,
    orientation_fine=cfg.orientation_fine,
  )
  overridden = _apply_env_overrides(env_cfg, overrides, cfg.task)
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


def _longest_run(mask: np.ndarray) -> int:
  """Longest run of consecutive True values."""
  best = cur = 0
  for v in mask:
    cur = cur + 1 if v else 0
    if cur > best:
      best = cur
  return best


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
  robot = env.scene["robot"]
  command = env.command_manager.get_term("goal_orientation")

  n_rec = cfg.num_steps + 1
  err = torch.zeros(n_rec, cfg.num_envs, device=device)
  r_z = torch.zeros(n_rec, cfg.num_envs, device=device)
  r_xy = torch.zeros(n_rec, cfg.num_envs, device=device)
  ang_speed = torch.zeros(n_rec, cfg.num_envs, device=device)
  done_buf = torch.zeros(n_rec, cfg.num_envs, dtype=torch.bool, device=device)
  fell_buf = torch.zeros(n_rec, cfg.num_envs, dtype=torch.bool, device=device)
  # Grasp state, for asking what actually separates a good episode from a bad
  # one. Nothing about the GOAL predicts the outcome (corr of min-error with
  # start error +0.06, with spin demanded +0.11, with tip-over demanded -0.01),
  # so the variance has to come from the state the episode starts in. The hand
  # base is fixed in world, so the cube's world position is already its position
  # relative to the hand -- but ONLY once the per-env grid offset is removed.
  # mjlab lays multiple envs out on a grid, so root_link_pos_w carries the env
  # origin; without subtracting it, "cube x" is really "which column of the grid
  # this env sits in" and correlates with nothing.
  env_origins = env.scene.env_origins
  n_j = robot.data.joint_pos.shape[-1]
  cube_pos = torch.zeros(n_rec, cfg.num_envs, 3, device=device)
  joint_pos = torch.zeros(n_rec, cfg.num_envs, n_j, device=device)

  def record(t: int) -> None:
    # World-frame rotation vector taking the cube to the goal. quat_box_minus is
    # log(q1 * q2^-1) -- left multiplication, so r lives in the world frame and
    # r_z is the palm-normal (spin) component by construction.
    r = quat_box_minus(command.command, cube.data.root_link_quat_w)
    err[t] = torch.linalg.vector_norm(r, dim=-1)
    r_z[t] = r[:, 2].abs()
    r_xy[t] = torch.linalg.vector_norm(r[:, :2], dim=-1)
    ang_speed[t] = torch.linalg.vector_norm(cube.data.root_link_ang_vel_w, dim=-1)
    cube_pos[t] = cube.data.root_link_pos_w - env_origins
    joint_pos[t] = robot.data.joint_pos

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
    "cube_pos": cube_pos.cpu().numpy(),
    "joint_pos": joint_pos.cpu().numpy(),
  }


def _analyze(trace: dict, cfg: EvalConfig) -> dict:
  err, r_z, r_xy = trace["err"], trace["r_z"], trace["r_xy"]
  ang, done, fell = trace["ang_speed"], trace["done"], trace["fell"]
  cube_pos, joint_pos = trace["cube_pos"], trace["joint_pos"]
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
      seg_ang = ang[window, e]
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
          # Hold-based criteria. `reached_threshold` above is instantaneous and
          # counts a cube tumbling through the goal; these ask whether it was
          # actually brought to rest there.
          "longest_hold": int(_longest_run(seg_err < cfg.success_threshold)),
          "held_threshold": bool(
            _longest_run(seg_err < cfg.success_threshold) >= cfg.hold_steps
          ),
          "still_threshold": bool(
            (
              (seg_err < cfg.success_threshold)
              & (seg_ang < cfg.hold_ang_speed)
            ).any()
          ),
          # The best error the policy actually HOLDS: the minimum over samples
          # where the cube is settled. nan when it is never settled at all.
          "min_err_still": float(
            seg_err[seg_ang < cfg.hold_ang_speed].min()
            if bool((seg_ang < cfg.hold_ang_speed).any())
            else np.nan
          ),
          "ang_at_best": float(seg_ang[i_best]),
          "start_r_z": float(r_z[s, e]),
          "start_r_xy": float(r_xy[s, e]),
          "best_r_z": float(r_z[s + i_best, e]),
          "best_r_xy": float(r_xy[s + i_best, e]),
          "ang_speed_early": float(ang[s : s + n_early, e].mean()),
          "ang_speed_late": float(ang[s + i_late : t_end + 1, e].mean()),
          # Grasp state at the instant the episode starts.
          "start_cube_x": float(cube_pos[s, e, 0]),
          "start_cube_y": float(cube_pos[s, e, 1]),
          "start_cube_z": float(cube_pos[s, e, 2]),
          "start_joint_mean": float(joint_pos[s, e].mean()),
          "start_joint_std": float(joint_pos[s, e].std()),
          # Where the cube ends up sitting once the policy has settled: the
          # mean over the second half of the episode, which is the pose it
          # actually holds rather than the one it was handed.
          "held_cube_z": float(cube_pos[s + i_late : t_end + 1, e, 2].mean()),
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
  # Derived from asdict(cfg) rather than hand-listed. This block is the only
  # record of WHICH env a result was measured in, and the hand-listed version
  # silently omitted every field added after it was written -- the goal-drift
  # pair and orientation_fine -- so a JSON produced with the goal pinned looked
  # identical to one produced with it drifting. Anything added to EvalConfig is
  # now recorded automatically; device and json_out are dropped as non-physical.
  cfg_record = {k: v for k, v in asdict(cfg).items() if k not in ("json_out", "device")}
  cfg_record["success_threshold_rad"] = cfg_record.pop("success_threshold")
  return {
    "config": cfg_record,
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
    # Hold-based success. The instantaneous number above counts a cube that
    # tumbles through the goal region; these two require it to stay there.
    "held_fraction": float(col("held_threshold").mean()) if n_ep else float("nan"),
    "n_held": int(col("held_threshold").sum()) if n_ep else 0,
    "still_fraction": float(col("still_threshold").mean()) if n_ep else float("nan"),
    "n_still": int(col("still_threshold").sum()) if n_ep else 0,
    "longest_hold_steps": _spread(col("longest_hold")) if n_ep else {},
    "ang_at_best": _spread(col("ang_at_best")) if n_ep else {},
    # Settled precision: the best error reached while the cube is NOT tumbling.
    "settled_error_deg": (
      _spread(np.array([v for v in col("min_err_still") * _RAD2DEG if np.isfinite(v)]))
      if n_ep else {}
    ),
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
  # Print every override that is actually in force, not just cube_condim. Two
  # results in this table can differ only by --goal-drift, and a header that
  # does not say so makes them look like the same measurement.
  _overrides = [
    (k, c[k])
    for k in (
      "obs_noise_scale",
      "cube_half_size", "cube_friction_sliding", "cube_friction_torsional",
      "cube_condim", "cube_priority", "palm_euler",
      "goal_drift", "goal_resample_on_success", "orientation_fine",
    )
    if c.get(k) is not None
  ]
  if _overrides:
    print("  overrides      " + ", ".join(f"{k}={v}" for k, v in _overrides))
  else:
    print("  overrides      none (task defaults)")
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

  print(f"\n-- success (error < {thr_deg:.1f} deg) ------------------------------------")
  print(
    f"  instantaneous (any single step)   {res['n_success']}/{res['n_episodes']}"
    f" = {100.0 * res['success_fraction']:.1f}%"
  )
  print(
    f"  HELD ({c['hold_steps']} consecutive steps)        {res['n_held']}/{res['n_episodes']}"
    f" = {100.0 * res['held_fraction']:.1f}%"
  )
  print(
    f"  SETTLED (|w| < {c['hold_ang_speed']} rad/s)         {res['n_still']}/{res['n_episodes']}"
    f" = {100.0 * res['still_fraction']:.1f}%"
  )
  print(
    f"  cube angular speed at best error : median"
    f" {res['ang_at_best']['median']:.2f} rad/s"
    "   (a genuine arrival would be near 0)"
  )
  if res["settled_error_deg"].get("n"):
    print(
      f"  settled error (best while |w| < {c['hold_ang_speed']}): "
      f"{_fmt(res['settled_error_deg'])}"
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
