"""Reward terms for cube reorientation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_box_minus, quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT = SceneEntityCfg("robot", joint_names=(".*",))


def _linear_tolerance(
  x: torch.Tensor,
  bounds: tuple[float, float],
  margin: float,
) -> torch.Tensor:
  """dm_control-style linear tolerance used by the MJX leap reorient task."""
  lower, upper = bounds
  in_bounds = (x >= lower) & (x <= upper)
  if margin <= 0.0:
    return in_bounds.float()
  dist = torch.where(x < lower, lower - x, torch.where(x > upper, x - upper, 0.0))
  return torch.where(in_bounds, torch.ones_like(x), torch.clamp(1.0 - dist / margin, min=0.0))


def cube_orientation_tolerance(
  env: ManagerBasedRlEnv,
  command_name: str = "goal_orientation",
  object_name: str = "cube",
  bounds: tuple[float, float] = (0.0, 0.2),
  margin: float = 3.141592653589793,
) -> torch.Tensor:
  cube: Entity = env.scene[object_name]
  goal = env.command_manager.get_command(command_name)
  assert goal is not None
  err = quat_error_magnitude(cube.data.root_link_quat_w, goal)
  return _linear_tolerance(err, bounds=bounds, margin=margin)


def cube_orientation_inverse(
  env: ManagerBasedRlEnv,
  command_name: str = "goal_orientation",
  object_name: str = "cube",
  eps: float = 0.1,
) -> torch.Tensor:
  """Inverse-distance orientation reward ``1 / (err + eps)``.

  The shape DeXtreme, Chen et al. (CoRL 2021) and DexReMoE use. Unlike the
  linear tolerance -- constant marginal reward from 180 deg down to 11.5 deg,
  then flat -- its marginal reward rises monotonically all the way to zero
  error, so there is no dead band and the last degree is worth the most. It is
  bounded by construction at ``1 / eps`` (10 at the default), so it needs no
  separate cap. See research/verdicts/00-INTERIM-reward-kernel.md.
  """
  cube: Entity = env.scene[object_name]
  goal = env.command_manager.get_command(command_name)
  assert goal is not None
  err = quat_error_magnitude(cube.data.root_link_quat_w, goal)
  return 1.0 / (err + eps)


def cube_angvel_toward_goal(
  env: ManagerBasedRlEnv,
  command_name: str = "goal_orientation",
  object_name: str = "cube",
  eps: float = 0.05,
  speed_floor: float = 0.5,
) -> torch.Tensor:
  """How well the cube's rotation is AIMED at the goal, in [-1, 1].

  ``cos = (omega . r_hat) / (|omega| + speed_floor)`` where ``r_hat`` is the unit
  rotation vector carrying the cube to the goal. +1 is rotating straight at the
  goal, -1 straight away, 0 orthogonal.

  Why alignment and not the raw projection ``omega . r_hat`` (tried first,
  2026-09-12, and withdrawn 450 iterations in):

  1. *The projection telescopes.* ``omega . r_hat`` is approximately
     ``-d(err)/dt``, so its episode sum is just the total error closed -- bounded
     by ``err_0`` ~ 2.2 rad no matter how long the episode runs -- while the
     orientation kernel accrues every step without bound. Measured at weight 1.0:
     ``Episode_Reward/angvel_align`` = 0.008 against ``orientation`` = 36.7 and
     ``success`` = 46.9, i.e. **0.01% of the reward**. No weight fixes this: one
     large enough to matter in total is explosive per step whenever the policy
     does align, which buys a spin-forever policy that never lands.
  2. *It rewards the wrong thing.* Traces show the hand already produces 47-55
     deg/s of diffusion and converts only 1.3-13 deg/s into progress. The deficit
     is aim, not speed, and the projection pays for both.
  3. *It sidesteps the potential-shaping objection.* ``verdicts/05`` predicts a
     null for progress shaping because it is potential-based and so leaves the
     optimal policy unchanged. A cosine is ``-d(err)/dt`` divided by ``|omega|``,
     which is not a potential difference, so that argument does not apply here.

  Bounded by construction, so it cannot dominate: at most 1.0 per step, 2.0 per
  40-step rollout window after dt scaling, against the orientation term's ~37.

  ``speed_floor`` (rad/s) attenuates the term when the cube is barely moving,
  where the direction of a near-zero ``omega`` is noise -- and stops a policy
  collecting full alignment credit for creeping. At the measured 1.3 rad/s churn
  a perfectly aimed rotation scores 1.3/1.8 = 0.72. ``eps`` does the same job at
  the other end, fading the term out inside ~3 deg where ``r`` vanishes and its
  direction is meaningless, leaving the inverse kernel to land the cube.
  """
  cube: Entity = env.scene[object_name]
  goal = env.command_manager.get_command(command_name)
  assert goal is not None
  omega = cube.data.root_link_ang_vel_w
  # World-frame rotation vector to the goal; same convention as eval_policy.py.
  r = quat_box_minus(goal, cube.data.root_link_quat_w)
  r_hat = r / (torch.linalg.vector_norm(r, dim=-1, keepdim=True) + eps)
  speed = torch.linalg.vector_norm(omega, dim=-1)
  return (omega * r_hat).sum(dim=-1) / (speed + speed_floor)


def _long_tail_tolerance(
  x: torch.Tensor,
  margin: float,
  value_at_margin: float = 0.1,
) -> torch.Tensor:
  """dm_control-style ``long_tail`` sigmoid: 1.0 at x=0, ``value_at_margin`` at x=margin."""
  scale = math.sqrt(1.0 / value_at_margin - 1.0)
  scaled = x / margin * scale
  return 1.0 / (scaled * scaled + 1.0)


def cube_orientation_fine(
  env: ManagerBasedRlEnv,
  command_name: str = "goal_orientation",
  object_name: str = "cube",
  margin: float = 0.4,
) -> torch.Tensor:
  """Peaked orientation reward that only pays out near the goal.

  The linear term in :func:`cube_orientation_tolerance` gives global direction but
  is far too shallow (5/pi per rad) to drive the last radian, so the policy parks
  at ~1.5 rad and farms episode length instead. This concentrates the gradient
  near zero error.
  """
  cube: Entity = env.scene[object_name]
  goal = env.command_manager.get_command(command_name)
  assert goal is not None
  err = quat_error_magnitude(cube.data.root_link_quat_w, goal)
  return _long_tail_tolerance(err, margin=margin)


def cube_position_tolerance(
  env: ManagerBasedRlEnv,
  object_name: str = "cube",
  hand_name: str = "robot",
  bounds: tuple[float, float] = (0.0, 0.02),
  margin: float = 0.05,
) -> torch.Tensor:
  from leap_xela_mjlab.tasks.reorient.mdp.observations import cube_pos_error_from_palm

  del hand_name
  err = torch.linalg.vector_norm(
    cube_pos_error_from_palm(env, object_name=object_name), dim=-1
  )
  return _linear_tolerance(err, bounds=bounds, margin=margin)


def hand_pose_l2_from_default(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(q - q0), dim=-1)


class action_rate_l2:
  """Penalize first and second differences of the policy action."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.prev = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    self.prev_prev = torch.zeros_like(self.prev)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.prev[env_ids] = 0.0
    self.prev_prev[env_ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    act = env.action_manager.action
    c1 = torch.sum(torch.square(act - self.prev), dim=-1)
    c2 = torch.sum(torch.square(act - 2.0 * self.prev + self.prev_prev), dim=-1)
    self.prev_prev = self.prev.clone()
    self.prev = act.clone()
    return c1 + c2


class orientation_progress:
  """Dense reward for *reducing* orientation error, paid at any error magnitude.

  The tolerance terms only carry useful gradient near the goal, so a policy that
  starts ~2.25 rad away gets almost no signal connecting its actions to the goal
  and settles on a fixed compromise pose. This pays ``prev_err - err`` every step,
  which telescopes over an episode to the total progress made.

  ``clip`` bounds the per-step delta so the discontinuous error jump when the goal
  resamples on success cannot swamp the success bonus with a large negative. Normal
  per-step changes are well below the clip, so it effectively only binds on resample.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.prev_err = torch.zeros(env.num_envs, device=env.device)
    self.started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.started[env_ids] = False
    self.prev_err[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "goal_orientation",
    object_name: str = "cube",
    clip: float = 0.1,
  ) -> torch.Tensor:
    cube: Entity = env.scene[object_name]
    goal = env.command_manager.get_command(command_name)
    assert goal is not None
    err = quat_error_magnitude(cube.data.root_link_quat_w, goal)
    # No reward on the first step after a reset: prev_err is not meaningful yet.
    progress = torch.where(self.started, self.prev_err - err, torch.zeros_like(err))
    progress = torch.clamp(progress, -clip, clip)
    self.prev_err = err.clone()
    self.started[:] = True
    return progress


def joint_vel_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT,
  max_velocity: float = 5.0,
  vel_tolerance: float = 1.0,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  scale = max(max_velocity - vel_tolerance, 1e-6)
  return torch.sum(torch.square(vel / scale), dim=-1)


def energy_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  tau = asset.data.actuator_force[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(vel) * torch.abs(tau), dim=-1)


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the MAGNITUDE of the raw policy output.

  ``action_rate_l2`` penalizes only the *change* in action, and the action term
  clips to the actuator ctrl range, so above the clip the reward gradient with
  respect to the action is exactly zero and nothing pulls its magnitude back.
  The optimum of a rate-only penalty is therefore a large CONSTANT action -- and
  that is what the reference line learned. Measured on run 14's model_2999,
  deterministic, between the approach phase and the stall:

      steps 0-80     mean |a| 8.3   step-to-step |da| 2.16
      steps 150-400  mean |a| 13.4  step-to-step |da| 0.93

  Magnitude grows while change collapses. At ``scale=0.5`` a mean |a| of 13.4 is
  a commanded joint delta of 6.7 rad against a median ctrl range of 2.27 rad --
  3x the entire range, p95 15x. Every action is fully saturated, so the policy
  is a bang-bang controller with no fine authority, which is the standing
  explanation for the ~35 deg steady-state shell it converges to from any
  starting offset. Same space as ``action_rate_l2``: raw output, pre-scale.
  """
  return torch.sum(torch.square(env.action_manager.action), dim=1)


def termination_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1.0 on environments that terminated for a non-timeout reason this step."""
  # Prefer termination manager's terminated flag when available.
  term_manager = getattr(env, "termination_manager", None)
  if term_manager is not None and hasattr(term_manager, "terminated"):
    return term_manager.terminated.float()
  return env.reset_buf.float()


def success_bonus(
  env: ManagerBasedRlEnv,
  command_name: str = "goal_orientation",
  object_name: str = "cube",
  success_threshold: float = 0.1,
) -> torch.Tensor:
  """Bonus when cube orientation error is below the success threshold.

  Computed here (not via command metrics) because rewards run before
  ``command_manager.compute()`` in the env step.
  """
  cube: Entity = env.scene[object_name]
  goal = env.command_manager.get_command(command_name)
  assert goal is not None
  err = quat_error_magnitude(cube.data.root_link_quat_w, goal)
  return (err < success_threshold).float()
