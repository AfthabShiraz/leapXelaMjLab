"""Observations for cube reorientation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from leap_xela_mjlab.robots.leap_xela import FINGERTIP_SITE_NAMES
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_conjugate, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT = SceneEntityCfg("robot", joint_names=(".*",))


def joint_pos_abs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel_abs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids]


def joint_pos_error_from_command(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT,
) -> torch.Tensor:
  """Measured joint pos minus commanded motor targets (MJX qpos_error)."""
  asset: Entity = env.scene[asset_cfg.name]
  measured = asset.data.joint_pos[:, asset_cfg.joint_ids]
  term = env.action_manager.get_term(action_name)
  target = getattr(term, "target", None)
  if target is None:
    target = asset.data.joint_pos_target[:, asset_cfg.joint_ids]
  return measured - target


def _palm_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World-frame grasp-site position (tracks the raised hand base)."""
  hand: Entity = env.scene["robot"]
  cache_key = "_leap_xela_grasp_site_id"
  site_id = getattr(env, cache_key, None)
  if site_id is None:
    found, _ = hand.find_sites("grasp_site", preserve_order=True)
    if len(found) != 1:
      raise ValueError(f"Expected one grasp_site, found {len(found)}.")
    site_id = int(found[0])
    setattr(env, cache_key, site_id)
  return hand.data.site_pos_w[:, site_id]


def cube_pos_error_from_palm(
  env: ManagerBasedRlEnv,
  object_name: str = "cube",
) -> torch.Tensor:
  cube: Entity = env.scene[object_name]
  return _palm_pos_w(env) - cube.data.root_link_pos_w


def cube_ori_error_mat(
  env: ManagerBasedRlEnv,
  command_name: str = "goal_orientation",
  object_name: str = "cube",
) -> torch.Tensor:
  """Flattened rotation-matrix rows 1:3 of quat_mul(cube, inv(goal)) (6D)."""
  cube: Entity = env.scene[object_name]
  goal = env.command_manager.get_command(command_name)
  assert goal is not None
  quat_diff = quat_mul(cube.data.root_link_quat_w, quat_conjugate(goal))
  mat = matrix_from_quat(quat_diff)
  return mat[:, 1:3, :].reshape(env.num_envs, 6)


def cube_lin_vel(
  env: ManagerBasedRlEnv, object_name: str = "cube"
) -> torch.Tensor:
  cube: Entity = env.scene[object_name]
  return cube.data.root_link_lin_vel_w


def cube_ang_vel(
  env: ManagerBasedRlEnv, object_name: str = "cube"
) -> torch.Tensor:
  cube: Entity = env.scene[object_name]
  return cube.data.root_link_ang_vel_w


def fingertip_positions_rel_palm(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  site_names: tuple[str, ...] = FINGERTIP_SITE_NAMES,
) -> torch.Tensor:
  """Fingertip site positions relative to the grasp site (12D)."""
  hand: Entity = env.scene[asset_cfg.name]
  cache_key = f"_leap_xela_tip_site_ids::{asset_cfg.name}"
  tip_ids = getattr(env, cache_key, None)
  if tip_ids is None:
    ids: list[int] = []
    for name in site_names:
      found, _ = hand.find_sites(name, preserve_order=True)
      if len(found) != 1:
        raise ValueError(f"Expected one site '{name}', found {len(found)}.")
      ids.append(int(found[0]))
    tip_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    setattr(env, cache_key, tip_ids)

  tip_pos_w = hand.data.site_pos_w[:, tip_ids]
  palm = _palm_pos_w(env).unsqueeze(1)
  return (tip_pos_w - palm).reshape(env.num_envs, -1)
