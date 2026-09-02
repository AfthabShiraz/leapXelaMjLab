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
    # Playground's second success gauge (``steps_since_last_success``). Metrics
    # are logged at episode end and zeroed there, so a value near the episode
    # length reads as "never reached the goal", while a small value means the
    # policy was on target recently. Complements consecutive_success, which
    # counts hits but says nothing about when they happened.
    self.metrics["steps_since_last_success"] = torch.zeros(
      self.num_envs, device=self.device
    )
    # Diagnostic only: distinguishes "cannot rotate the cube" from
    # "rotates it, but not toward the goal".
    self.metrics["cube_ang_speed"] = torch.zeros(self.num_envs, device=self.device)
    # Goals actually reached this episode. Unlike consecutive_success this is not
    # cleared by the success-triggered goal resample, so it is a true per-episode
    # count and the signal the curriculum promotes on.
    self.success_count = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goals_reached"] = torch.zeros(self.num_envs, device=self.device)
    self.difficulty = float(cfg.initial_difficulty)
    # Relative goals cannot be composed at reset: the cube pose written by the
    # reset event only reaches root_link_quat_w after sim.forward(), so reading it
    # there yields a stale orientation. Stash the offset and compose on the first
    # step, when the pose is current.
    self.pending_delta = torch.zeros(self.num_envs, 4, device=self.device)
    self.pending_delta[:, 0] = 1.0
    self.pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.metrics["difficulty"] = torch.zeros(self.num_envs, device=self.device)
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

  def _apply_pending_goals(self) -> None:
    """Compose deferred relative goals against the now-current cube pose."""
    if not self.cfg.goal_relative_to_object:
      return
    ids = self.pending.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
      return
    self.quat_command_w[ids] = _normalize_quat(
      quat_mul(self.object.data.root_link_quat_w[ids], self.pending_delta[ids])
    )
    self.pending[ids] = False
    self._sync_goal_visual(ids)

  def _update_metrics(self) -> None:
    self._apply_pending_goals()
    err = quat_error_magnitude(
      self.object.data.root_link_quat_w, self.quat_command_w
    )
    self.metrics["orientation_error"] = err
    success = (err < self.cfg.orientation_success_threshold).float()
    self.metrics["success"] = success
    self.metrics["consecutive_success"] += success
    self.metrics["steps_since_last_success"] = torch.where(
      success > 0.0,
      torch.zeros_like(self.metrics["steps_since_last_success"]),
      self.metrics["steps_since_last_success"] + 1.0,
    )
    # Counted here, on the success flag itself, exactly as playground does --
    # NOT in _update_command's `elif update_goal_on_success` branch, where this
    # used to live. `use_mjx_goal_drift=True` makes that branch unreachable, so
    # success_count never incremented and `goals_reached` was pinned at exactly
    # 0.0 for every run that used the drift (all of runs 12-19).
    #
    # Runs 9-11 are NOT affected: they set use_mjx_goal_drift=false, so the
    # branch was live and their goals_reached is real. But the curriculum's
    # promotion criterion reads this same counter, so the trap is live for any
    # future curriculum run on the reference config, which does have drift on --
    # difficulty would sit at its initial value forever with no error raised.
    # How the goal is updated on success is independent of counting it.
    self.success_count += success
    self.metrics["cube_ang_speed"] = torch.linalg.vector_norm(
      self.object.data.root_link_ang_vel_w, dim=-1
    )
    self.metrics["goals_reached"] = self.success_count.clone()
    self.metrics["difficulty"][:] = self.difficulty

  def _sample_goal(self, env_ids: torch.Tensor) -> None:
    """Draw new goal orientations for ``env_ids``, scaled by the curriculum."""
    if len(env_ids) == 0:
      return
    span = torch.pi * self.difficulty
    rand = 2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
    delta = quat_mul(
      quat_from_angle_axis(rand[:, 0] * span, self._x_unit[env_ids]),
      quat_from_angle_axis(rand[:, 1] * span, self._y_unit[env_ids]),
    )
    if self.cfg.goal_relative_to_object:
      # Offset from where the cube is now, so difficulty controls the actual
      # distance the policy has to cover. Composed on the next step (see below).
      self.pending_delta[env_ids] = delta
      self.pending[env_ids] = True
      delta = quat_mul(self.object.data.root_link_quat_w[env_ids], delta)
    self.quat_command_w[env_ids] = _normalize_quat(delta)
    self.goal_dquat[env_ids] = 0.0
    self.metrics["consecutive_success"][env_ids] = 0.0
    self._sync_goal_visual(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Episode reset: new goal *and* clear the per-episode success count."""
    if len(env_ids) == 0:
      return
    self._sample_goal(env_ids)
    self.success_count[env_ids] = 0.0

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
      self._sample_goal(ids)

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
  # Curriculum. Goals are sampled within +-(difficulty * pi) on each of two axes.
  # With goal_relative_to_object, that offset is applied to the cube's *current*
  # orientation, so early goals are reachable instead of a near-arbitrary pose.
  goal_relative_to_object: bool = False
  initial_difficulty: float = 0.1
  # Used only if no goal entity is present.
  goal_pos_offset: tuple[float, float, float] = (0.225, 0.17, 0.0)
  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  debug_vis: bool = True
  debug_vis_axes: bool = False

  def build(self, env: ManagerBasedRlEnv) -> InHandReorientationCommand:
    return InHandReorientationCommand(self, env)
