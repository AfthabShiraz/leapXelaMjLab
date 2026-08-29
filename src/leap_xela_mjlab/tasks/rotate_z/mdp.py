"""MDP terms specific to the z-axis rotation task.

Everything else is shared with the reorient task; only the reward and two
critic observations differ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT = SceneEntityCfg("robot", joint_names=(".*",))


def cube_ang_vel_axis(
  env: ManagerBasedRlEnv,
  object_name: str = "cube",
  axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
  """Cube angular velocity projected onto ``axis`` (world frame).

  This is the whole task reward: the MJX ``LeapCubeRotateZAxis`` env uses
  ``cube_angvel @ [0, 0, 1]`` unconditionally. World +z is the palm normal in
  this scene, so it is the same axis the MJX task uses.
  """
  cube: Entity = env.scene[object_name]
  axis_t = torch.tensor(axis, device=env.device, dtype=torch.float32)
  return cube.data.root_link_ang_vel_w @ axis_t


def cube_quat(env: ManagerBasedRlEnv, object_name: str = "cube") -> torch.Tensor:
  cube: Entity = env.scene[object_name]
  return cube.data.root_link_quat_w


def joint_torque_abs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.actuator_force[:, asset_cfg.joint_ids]
