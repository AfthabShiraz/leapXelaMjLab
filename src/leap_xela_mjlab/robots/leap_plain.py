"""Bare LEAP hand EntityCfg -- the control for "what do the XELA pads cost us".

This is mujoco_playground's `leap_rh_mjx.xml` (meshes from mujoco_menagerie),
vendored under `assets/leap_plain/`. It is the same hand as LeapXELA without the
tactile skin, and the comparison is unusually clean -- verified 2026-09-12:

    joint axes identical      16/16
    joint positions identical 16/16
    total mass                746 g (XELA) vs 749 g (bare)
    geoms                     66 vs 56      <- the +10 are the pads
    grasp_site world position (0.11, 0, 0.03) in BOTH

so the cube spawns in the same place relative to the hand and only the fingertip
collision geometry and the joint ranges differ.

Two things it does NOT share, deliberately:

1. **Joint limits.** The bare model ships LEAP's own ranges -- lateral splay
   +/-1.047 rad against the LeapXELA MJCF's +/-0.349. That difference is real and
   is itself under test (see `leap_xela.py:_LEAP_JOINT_RANGE` and the
   `leap-limits` run), so it is left as the bare model has it rather than forced
   to match. A bare-hand result therefore differs from LeapXELA in BOTH pads and
   limits; attribute accordingly.
2. **Palm transform.** The two MJCFs express the mount differently -- XELA
   carries a `leap_mount` offset plus a rotated palm, the bare model folds it
   into the palm body -- but the grasp site lands in the same world position, so
   the scene geometry is equivalent where it matters.

The recipe this is trained under must match whatever it is being compared to, or
the comparison confounds the hand with the reward. See
`research/NEXT_EXPERIMENTS.md`, "PENDING DECISION -- the bare-hand control".
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from leap_xela_mjlab.robots.leap_xela import (
  FLOOR_Z_OFFSET,
  HOME_JOINT_POS,
  JOINT_NAMES,
  _load_meshdir_assets,
)
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

LEAP_PLAIN_DIR = (
  Path(__file__).resolve().parent.parent / "assets" / "leap_plain"
)


def get_hand_xml() -> Path:
  xml = LEAP_PLAIN_DIR / "leap_rh_mjx.xml"
  if not xml.exists():
    raise FileNotFoundError(
      f"Missing bare-LEAP MJCF at {xml}. It is gitignored because it is ~15 MB "
      "of third-party Apache-2.0 assets; see the regeneration commands in "
      "leapXelaMjLab/.gitignore."
    )
  return xml


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(get_hand_xml()))
  # The vendored XML references meshes by relative path (assets/*.obj and
  # meshes/*.obj) which MjSpec resolves from disk, but embedding them by
  # basename as well matches how leap_xela.py loads its own and keeps the spec
  # self-contained if it is ever serialised.
  spec.assets = {
    **_load_meshdir_assets(LEAP_PLAIN_DIR / "assets"),
    **_load_meshdir_assets(LEAP_PLAIN_DIR / "meshes"),
  }
  return spec


def get_leap_plain_cfg() -> EntityCfg:
  """Fixed-base bare LEAP hand, XML position actuators.

  Joint names, home pose and floor offset are imported from ``leap_xela`` rather
  than restated: they are verified identical across the two models, and a second
  copy would be a place for them to silently drift apart.
  """

  def _spec_fn() -> mujoco.MjSpec:
    return get_spec()

  return EntityCfg(
    spec_fn=_spec_fn,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=JOINT_NAMES),),
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, FLOOR_Z_OFFSET),
      rot=(1.0, 0.0, 0.0, 0.0),
      joint_pos=dict(HOME_JOINT_POS),
      joint_vel={".*": 0.0},
    ),
  )
