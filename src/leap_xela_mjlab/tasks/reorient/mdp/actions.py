"""Delta joint-position action matching the MJX leapXELA controller.

Policy outputs are in [-1, 1]. Each env step:
  target = clip(prev_target + action * scale)
  target = ema_alpha * target + (1 - ema_alpha) * prev_target
and that target is held across physics substeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DeltaJointPositionActionCfg(BaseActionCfg):
  """Integrate normalized actions into absolute joint position targets."""

  use_default_offset: bool = True
  """If True, initialize targets from default joint positions on reset."""

  clip_to_ctrl_limits: bool = True
  """Clamp targets to actuator ctrlrange / soft joint limits."""

  ema_alpha: float = 1.0
  """EMA blend with previous targets. 1.0 disables smoothing."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT
    if self.offset != 0.0:
      raise ValueError(
        "DeltaJointPositionActionCfg does not support a fixed offset. "
        "Use scale for the per-step delta magnitude."
      )

  def build(self, env: ManagerBasedRlEnv) -> DeltaJointPositionAction:
    return DeltaJointPositionAction(self, env)


class DeltaJointPositionAction(BaseAction):
  cfg: DeltaJointPositionActionCfg

  def __init__(self, cfg: DeltaJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._target = torch.zeros_like(self._raw_actions)
    self._initialize_target(slice(None))

  def _limits(
    self, env_ids: torch.Tensor | slice
  ) -> tuple[torch.Tensor, torch.Tensor]:
    limits = self._entity.data.soft_joint_pos_limits
    lower = limits[env_ids][:, self._target_ids, 0]
    upper = limits[env_ids][:, self._target_ids, 1]
    return lower, upper

  def _initialize_target(self, env_ids: torch.Tensor | slice) -> None:
    if self.cfg.use_default_offset:
      source = self._entity.data.default_joint_pos
    else:
      source = self._entity.data.joint_pos
    self._target[env_ids] = source[env_ids][:, self._target_ids]
    if self.cfg.clip_to_ctrl_limits:
      lower, upper = self._limits(env_ids)
      self._target[env_ids] = torch.clamp(self._target[env_ids], min=lower, max=upper)
    self._processed_actions[env_ids] = self._target[env_ids]

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    delta = self._raw_actions * self._scale
    candidate = self._target + delta
    if self.cfg.clip_to_ctrl_limits:
      lower, upper = self._limits(slice(None))
      candidate = torch.clamp(candidate, min=lower, max=upper)
    alpha = float(self.cfg.ema_alpha)
    self._target = alpha * candidate + (1.0 - alpha) * self._target
    self._processed_actions = self._target

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(
      self._processed_actions, joint_ids=self._target_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._initialize_target(env_ids)

  @property
  def target(self) -> torch.Tensor:
    """Current absolute joint position command."""
    return self._target
