"""Reward terms for cube reorientation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

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
