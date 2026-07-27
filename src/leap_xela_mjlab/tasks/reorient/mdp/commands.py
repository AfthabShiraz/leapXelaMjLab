"""In-hand orientation goal command (MJX / Isaac Lab style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_error_magnitude,
  quat_from_angle_axis,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
  return q / torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=1e-8)


def _quat_integrate(q: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
  """Integrate quaternion by angular velocity (MJX quat_integrate style)."""
  angle = torch.linalg.vector_norm(omega, dim=-1)
  axis = omega / torch.clamp(angle.unsqueeze(-1), min=1e-8)
  dq = quat_from_angle_axis(angle * dt, axis)
  return _normalize_quat(quat_mul(q, dq))


class InHandReorientationCommand(CommandTerm):
  """Sample SO(3) goals and drive a visual mocap goal cube."""

  cfg: InHandReorientationCommandCfg

  def __init__(self, cfg: InHandReorientationCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.asset_name]
    self.goal_entity: Entity | None = None
    if cfg.goal_asset_name is not None and cfg.goal_asset_name in env.scene.entities:
      self.goal_entity = env.scene[cfg.goal_asset_name]

    self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.quat_command_w[:, 0] = 1.0
    self.goal_dquat = torch.zeros(self.num_envs, 3, device=self.device)
    self._x_unit = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
      self.num_envs, 1
    )
    self._y_unit = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(
      self.num_envs, 1
    )
    self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["consecutive_success"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self._sync_goal_visual(slice(None))

  @property
  def command(self) -> torch.Tensor:
    """Goal orientation quaternion (wxyz). Shape: [num_envs, 4]."""
    return self.quat_command_w

  def _goal_pos_w(self) -> torch.Tensor:
    """Fixed world position for the goal cube (per-env, with env origins)."""
    offset = torch.tensor(self.cfg.goal_pos_offset, device=self.device)
    if self.goal_entity is not None:
      # Prefer the entity's authored default position (already includes floor lift).
      default = self.goal_entity.data.default_root_state[:, :3]
      return default + self._env.scene.env_origins
    return self.object.data.default_root_state[:, :3] + self._env.scene.env_origins + offset

  def _sync_goal_visual(self, env_ids: torch.Tensor | slice) -> None:
    if self.goal_entity is None:
      return
    if isinstance(env_ids, slice):
      ids = torch.arange(self.num_envs, device=self.device)
    else:
      ids = env_ids
    if len(ids) == 0:
      return
    pos = self._goal_pos_w()[ids]
    quat = self.quat_command_w[ids]
    pose = torch.cat([pos, quat], dim=-1)
    if self.goal_entity.is_mocap:
      self.goal_entity.write_mocap_pose_to_sim(pose, env_ids=ids)
    else:
      self.goal_entity.write_root_link_pose_to_sim(pose, env_ids=ids)

  def _update_metrics(self) -> None:
    err = quat_error_magnitude(
      self.object.data.root_link_quat_w, self.quat_command_w
    )
    self.metrics["orientation_error"] = err
    success = (err < self.cfg.orientation_success_threshold).float()
    self.metrics["success"] = success
    self.metrics["consecutive_success"] += success

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    rand = 2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
    quat = quat_mul(
      quat_from_angle_axis(rand[:, 0] * torch.pi, self._x_unit[env_ids]),
      quat_from_angle_axis(rand[:, 1] * torch.pi, self._y_unit[env_ids]),
    )
    self.quat_command_w[env_ids] = _normalize_quat(quat)
    self.goal_dquat[env_ids] = 0.0
    self.metrics["consecutive_success"][env_ids] = 0.0
    self._sync_goal_visual(env_ids)

  def _update_command(self) -> None:
    success = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold

    if self.cfg.use_mjx_goal_drift:
      kick = sample_uniform(1.0, 5.0, (self.num_envs, 3), device=self.device)
      self.goal_dquat = torch.where(
        success.unsqueeze(-1),
        kick,
        self.goal_dquat * 0.8,
      )
      self.quat_command_w = _quat_integrate(
        self.quat_command_w, self.goal_dquat, 2.0 * float(self._env.step_dt)
      )
      self._sync_goal_visual(slice(None))
    elif self.cfg.update_goal_on_success:
      ids = success.nonzero(as_tuple=False).squeeze(-1)
      self._resample_command(ids)

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Small axes on the goal cube (orientation is mainly shown by the mesh)."""
    if not self.cfg.debug_vis_axes:
      return
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    if self.goal_entity is not None:
      pos = self.goal_entity.data.root_link_pos_w.detach().cpu().numpy()
    else:
      offset = torch.tensor(self.cfg.goal_pos_offset, device=self.device)
      pos = (self.object.data.root_link_pos_w + offset).detach().cpu().numpy()
    rotm = matrix_from_quat(self.quat_command_w).detach().cpu().numpy()
    for env_id in env_indices:
      visualizer.add_frame(
        position=pos[env_id],
        rotation_matrix=rotm[env_id],
        scale=0.04,
        axis_radius=0.003,
        label=f"goal_{env_id}",
      )


@dataclass(kw_only=True)
class InHandReorientationCommandCfg(CommandTermCfg):
  asset_name: str = "cube"
  goal_asset_name: str | None = "goal"
  orientation_success_threshold: float = 0.1
  update_goal_on_success: bool = True
  use_mjx_goal_drift: bool = True
  # Used only if no goal entity is present.
  goal_pos_offset: tuple[float, float, float] = (0.225, 0.17, 0.0)
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  debug_vis: bool = True
  debug_vis_axes: bool = False

  def build(self, env: ManagerBasedRlEnv) -> InHandReorientationCommand:
    return InHandReorientationCommand(self, env)
