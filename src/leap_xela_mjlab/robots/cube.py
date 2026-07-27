"""Textured dex-cube and goal-cube entities for the reorient task."""

from __future__ import annotations

import mujoco

from leap_xela_mjlab import LEAPXELA_MODEL_DIR
from leap_xela_mjlab.robots.leap_xela import (
  CUBE_SPAWN_POS_LOCAL,
  FLOOR_Z_OFFSET,
)
from mjlab.entity import EntityCfg

# Match MJX reorientation_cube.xml with cube_scale_factor≈1.1.
_CUBE_HALF_SIZE = 0.0385
_CUBE_MASS = 0.108

# Goal display offset from scene_mjx_cube_*_mjx.xml (+ floor lift).
GOAL_POS_LOCAL = (0.325, 0.17, 0.0475)

_CUBE_XML = """
<mujoco model="reorient_cube">
  <asset>
    <texture name="dexcube" type="2d" file="reorientation_cube_textures/dex_cube.png"/>
    <material name="dexcube" texture="dexcube"/>
    <mesh name="cube_mesh" file="meshes/dex_cube.obj" scale="{s} {s} {s}"/>
  </asset>
  <worldbody>
    <body name="cube">
      <freejoint name="cube_freejoint"/>
      <geom type="mesh" mesh="cube_mesh" material="dexcube"
            contype="0" conaffinity="0" density="0" group="2"/>
      <geom name="cube" type="box" size="{s} {s} {s}" mass="{mass}"
            friction="{f0} {f1} {f2}" condim="3" group="3"/>
      <site name="cube_center" pos="0 0 0" group="4"/>
    </body>
  </worldbody>
</mujoco>
"""

_GOAL_XML = """
<mujoco model="goal_cube">
  <asset>
    <texture name="dexcube" type="2d" file="reorientation_cube_textures/dex_cube.png"/>
    <material name="dexcube" texture="dexcube"/>
    <mesh name="cube_mesh" file="meshes/dex_cube.obj" scale="{s} {s} {s}"/>
  </asset>
  <worldbody>
    <body name="goal">
      <geom type="mesh" mesh="cube_mesh" material="dexcube"
            contype="0" conaffinity="0" density="0" group="2"/>
      <geom name="goal_box" type="box" size="{s} {s} {s}"
            contype="0" conaffinity="0" density="0" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def _load_cube_assets() -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  tex_dir = LEAPXELA_MODEL_DIR / "reorientation_cube_textures"
  for path in sorted(tex_dir.glob("*")):
    if path.is_file():
      assets[f"reorientation_cube_textures/{path.name}"] = path.read_bytes()

  mesh_path = LEAPXELA_MODEL_DIR / "meshes" / "dex_cube.obj"
  if not mesh_path.exists():
    mesh_path = LEAPXELA_MODEL_DIR / "assets" / "meshes" / "dex_cube.obj"
  assets["meshes/dex_cube.obj"] = mesh_path.read_bytes()
  return assets


def get_cube_spec(
  half_size: float = _CUBE_HALF_SIZE,
  mass: float = _CUBE_MASS,
  friction: tuple[float, float, float] = (0.3, 0.05, 0.001),
) -> mujoco.MjSpec:
  xml = _CUBE_XML.format(
    s=half_size, mass=mass, f0=friction[0], f1=friction[1], f2=friction[2]
  )
  spec = mujoco.MjSpec.from_string(xml)
  spec.assets = _load_cube_assets()
  return spec


def get_goal_cube_spec(half_size: float = _CUBE_HALF_SIZE) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_string(_GOAL_XML.format(s=half_size))
  spec.assets = _load_cube_assets()
  return spec


def get_cube_cfg(
  spawn_pos: tuple[float, float, float] | None = None,
  half_size: float = _CUBE_HALF_SIZE,
  mass: float = _CUBE_MASS,
  friction: float = 0.3,
) -> EntityCfg:
  if spawn_pos is None:
    spawn_pos = (
      CUBE_SPAWN_POS_LOCAL[0],
      CUBE_SPAWN_POS_LOCAL[1],
      CUBE_SPAWN_POS_LOCAL[2] + FLOOR_Z_OFFSET,
    )
  friction_tuple = (friction, 0.05, 0.001)

  def _spec_fn() -> mujoco.MjSpec:
    return get_cube_spec(
      half_size=half_size, mass=mass, friction=friction_tuple
    )

  return EntityCfg(
    spec_fn=_spec_fn,
    init_state=EntityCfg.InitialStateCfg(
      pos=spawn_pos,
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
    ),
  )


def get_goal_cube_cfg(
  pos: tuple[float, float, float] | None = None,
  half_size: float = _CUBE_HALF_SIZE,
) -> EntityCfg:
  if pos is None:
    pos = (
      GOAL_POS_LOCAL[0],
      GOAL_POS_LOCAL[1],
      GOAL_POS_LOCAL[2] + FLOOR_Z_OFFSET,
    )

  def _spec_fn() -> mujoco.MjSpec:
    return get_goal_cube_spec(half_size=half_size)

  return EntityCfg(
    spec_fn=_spec_fn,
    init_state=EntityCfg.InitialStateCfg(
      pos=pos,
      rot=(1.0, 0.0, 0.0, 0.0),
    ),
  )
