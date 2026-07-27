"""Reset and domain-randomization events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _resolve_env_ids(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if isinstance(env_ids, slice):
    start, stop, step = env_ids.indices(env.num_envs)
    return torch.arange(start, stop, step, device=env.device, dtype=torch.long)
  return env_ids.to(device=env.device, dtype=torch.long)


def _uniform_quat(n: int, device: str | torch.device) -> torch.Tensor:
  """Uniform random unit quaternions (wxyz)."""
  u = sample_uniform(0.0, 1.0, (n, 3), device=device)
  return torch.stack(
    [
      torch.sqrt(1.0 - u[:, 0]) * torch.sin(2.0 * torch.pi * u[:, 1]),
      torch.sqrt(1.0 - u[:, 0]) * torch.cos(2.0 * torch.pi * u[:, 1]),
      torch.sqrt(u[:, 0]) * torch.sin(2.0 * torch.pi * u[:, 2]),
      torch.sqrt(u[:, 0]) * torch.cos(2.0 * torch.pi * u[:, 2]),
    ],
    dim=-1,
  )


def reset_cube_pose_uniform_quat(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  pose_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> None:
  """Reset cube pose with uniform position noise and a fully random quaternion."""
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  if pose_range is None:
    pose_range = {}

  cube: Entity = env.scene[asset_cfg.name]
  default_pose = cube.data.default_root_state[env_ids].clone()
  # default_root_state is typically (pos[3], quat[4], lin_vel[3], ang_vel[3]).
  pos = default_pose[:, 0:3] + env.scene.env_origins[env_ids]
  pos[:, 0] += sample_uniform(
    *pose_range.get("x", (0.0, 0.0)), len(env_ids), device=env.device
  )
  pos[:, 1] += sample_uniform(
    *pose_range.get("y", (0.0, 0.0)), len(env_ids), device=env.device
  )
  pos[:, 2] += sample_uniform(
    *pose_range.get("z", (0.0, 0.0)), len(env_ids), device=env.device
  )
  quat = _uniform_quat(len(env_ids), env.device)
  pose = torch.cat([pos, quat], dim=-1)
  cube.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  cube.write_root_link_velocity_to_sim(
    torch.zeros((len(env_ids), 6), device=env.device),
    env_ids=env_ids,
  )


@requires_model_fields("geom_friction")
def randomize_geom_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  friction_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  axes: tuple[int, ...] = (0,),
) -> None:
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  asset: Entity = env.scene[asset_cfg.name]
  geom_ids = asset_cfg.geom_ids
  if isinstance(geom_ids, slice):
    geom_ids = asset.indexing.geom_ids[geom_ids]
  elif isinstance(geom_ids, list):
    geom_ids = torch.tensor(geom_ids, device=env.device, dtype=torch.long)
  else:
    geom_ids = geom_ids.to(device=env.device, dtype=torch.long)
  if geom_ids.numel() == 0:
    return
  friction = sample_uniform(
    friction_range[0],
    friction_range[1],
    (len(env_ids), len(geom_ids)),
    device=env.device,
  )
  for axis in axes:
    env.sim.model.geom_friction[env_ids[:, None], geom_ids[None, :], axis] = friction


@requires_model_fields("body_mass")
def randomize_body_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  mass_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  operation: str = "scale",
) -> None:
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice):
    body_ids = asset.indexing.body_ids[body_ids]
  elif isinstance(body_ids, list):
    body_ids = torch.tensor(body_ids, device=env.device, dtype=torch.long)
  else:
    body_ids = body_ids.to(device=env.device, dtype=torch.long)
  if body_ids.numel() == 0:
    return
  default = env.sim.get_default_field("body_mass")[body_ids].unsqueeze(0)
  scale = sample_uniform(
    mass_range[0],
    mass_range[1],
    (len(env_ids), len(body_ids)),
    device=env.device,
  )
  if operation == "scale":
    out = default.expand_as(scale) * scale
  elif operation == "abs":
    out = scale
  else:
    raise ValueError(f"Unsupported operation '{operation}'.")
  env.sim.model.body_mass[env_ids[:, None], body_ids[None, :]] = out


def apply_cube_velocity_perturbation(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
  linear_velocity_range: tuple[float, float] = (0.0, 3.0),
  angular_velocity_range: tuple[float, float] = (0.0, 0.5),
  probability: float = 0.02,
) -> None:
  """Occasionally impart a random velocity impulse to the cube (MJX pert)."""
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0 or probability <= 0.0:
    return
  mask = torch.rand(len(env_ids), device=env.device) < probability
  env_ids = env_ids[mask]
  if len(env_ids) == 0:
    return

  cube: Entity = env.scene[asset_cfg.name]
  direction = torch.randn((len(env_ids), 6), device=env.device)
  direction = direction / torch.clamp(
    torch.linalg.vector_norm(direction, dim=-1, keepdim=True), min=1e-6
  )
  lin = sample_uniform(
    *linear_velocity_range, (len(env_ids), 1), device=env.device
  )
  ang = sample_uniform(
    *angular_velocity_range, (len(env_ids), 1), device=env.device
  )
  vel = direction * torch.cat([lin.expand(-1, 3), ang.expand(-1, 3)], dim=-1)
  current = cube.data.root_link_vel_w[env_ids]
  cube.write_root_link_velocity_to_sim(current + vel, env_ids=env_ids)
