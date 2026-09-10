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
            friction="{f0} {f1} {f2}" condim="{condim}" priority="{priority}"
            solref="{solref0} {solref1}" solimp="{i0} {i1} {i2} {i3} {i4}"
            group="3"/>
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


# The cube's own solver parameters. These are MuJoCo's defaults, which is what
# the cube geom already resolved to before ``priority`` existed here -- keeping
# them explicit matters because priority overrides solref/solimp too, so they
# stop being a don't-care the moment the cube outranks the hand. See
# ``get_cube_spec`` for which contacts that actually moves.
_DEFAULT_SOLREF = (0.02, 1.0)
_DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


def get_cube_spec(
  half_size: float = _CUBE_HALF_SIZE,
  mass: float = _CUBE_MASS,
  friction: tuple[float, float, float] = (0.3, 0.05, 0.0001),
  condim: int = 3,
  priority: int = 1,
  solref: tuple[float, float] = _DEFAULT_SOLREF,
  solimp: tuple[float, float, float, float, float] = _DEFAULT_SOLIMP,
) -> mujoco.MjSpec:
  """``condim`` 3 is the MJX default and makes friction[1]/[2] inert.

  MuJoCo takes a dynamically generated contact's condim as the max of the two
  geoms', and the hand geoms carry no condim (so 3). Raising it here is therefore
  enough to make torsional (condim>=4) and rolling (condim=6) friction live on
  every fingertip/cube contact.

  ``priority`` is why this cube's ``friction`` means anything at all. With equal
  priority MuJoCo mixes a dynamic contact's friction as the **element-wise max**
  of the two geoms, and the measured hand is:

      hand pads (32 geoms, palm_collision_* / *_uspa4*)  friction[0] = 0.2
      fingertips (th/if/mf/rf_tip)                       friction[0] = 0.5
      terrain                                            friction[0] = 1.0
      cube (this geom)                                   friction[0] = 0.3

  so at priority 0 a cube-fingertip contact resolves to max(0.3, 0.5) = **0.5**
  and the cube's sliding friction is simply discarded at the four geoms that do
  the grasping. Sweeping ``friction_sliding`` anywhere below 0.5 changed nothing
  measurable, and ``dr_cube_friction`` (range 0.1-0.5) was likewise inert
  against ``dr_fingertip_friction`` (range 0.5-1.0), which always won the max.

  At ``priority >= 1`` the cube outranks every hand geom (all priority 0) and
  dictates condim *and* friction on every contact it takes part in, which is
  what makes sliding friction a real hyperparameter. Two consequences worth
  knowing:

  * ``dr_fingertip_friction`` no longer affects cube-fingertip contacts.
  * priority also decides solref/solimp, which at equal priority were a
    solmix-weighted average. Cube-fingertip contacts are unaffected (the tips
    carry MuJoCo's defaults, same as the cube, so 0.02 either way), but
    cube-pad contacts go from the average 0.01005 to the cube's 0.02, i.e.
    roughly 2x softer. Pin ``solref``/``solimp`` here to control that.
  """
  if condim not in (1, 3, 4, 6):
    raise ValueError(f"condim must be one of 1, 3, 4, 6; got {condim}")
  if priority < 0:
    raise ValueError(f"priority must be >= 0; got {priority}")
  xml = _CUBE_XML.format(
    s=half_size, mass=mass, f0=friction[0], f1=friction[1], f2=friction[2],
    condim=condim, priority=priority,
    solref0=solref[0], solref1=solref[1],
    i0=solimp[0], i1=solimp[1], i2=solimp[2], i3=solimp[3], i4=solimp[4],
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
  friction_sliding: float = 0.3,
  friction_torsional: float = 0.05,
  # MuJoCo's default rolling friction, matching the MJX cube, which leaves it
  # unset. Note both this and friction_torsional are *inert* at condim=3: a 3D
  # contact has a normal and two tangential directions only, so friction[1] and
  # friction[2] are never read. The fric-tors ablation in TRAINING_NOTES.md
  # (0.05 / 0.3 / 1.0) therefore varied nothing.
  friction_rolling: float = 0.0001,
  condim: int = 3,
  # Default 1 so the cube outranks every hand geom (all priority 0) and its
  # friction is the one MuJoCo uses. At 0 the old element-wise-max mixing comes
  # back and ``friction_sliding`` is discarded at the fingertips -- see
  # ``get_cube_spec`` for the measured numbers. Kept settable so the pre-2026-09
  # runs stay reproducible.
  priority: int = 1,
  solref: tuple[float, float] = _DEFAULT_SOLREF,
  solimp: tuple[float, float, float, float, float] = _DEFAULT_SOLIMP,
) -> EntityCfg:
  if spawn_pos is None:
    spawn_pos = (
      CUBE_SPAWN_POS_LOCAL[0],
      CUBE_SPAWN_POS_LOCAL[1],
      CUBE_SPAWN_POS_LOCAL[2] + FLOOR_Z_OFFSET,
    )
  friction_tuple = (
    friction_sliding,
    friction_torsional,
    friction_rolling,
  )

  def _spec_fn() -> mujoco.MjSpec:
    return get_cube_spec(
      half_size=half_size, mass=mass, friction=friction_tuple, condim=condim,
      priority=priority, solref=solref, solimp=solimp,
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
