"""Termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def cube_fell_below(
  env: ManagerBasedRlEnv,
  object_name: str = "cube",
  minimum_height: float = -0.05,
) -> torch.Tensor:
  """Terminate when cube height relative to env origin drops too low."""
  cube: Entity = env.scene[object_name]
  height = cube.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return height < minimum_height
