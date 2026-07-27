"""LEAP-XELA hand EntityCfg for mjlab."""

from __future__ import annotations

from pathlib import Path

import mujoco

from leap_xela_mjlab import LEAPXELA_MODEL_DIR
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


JOINT_NAMES = (
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
)

ACTUATOR_NAMES = tuple(f"{name}_act" for name in JOINT_NAMES)

FINGERTIP_SITE_NAMES = ("th_tip", "if_tip", "mf_tip", "rf_tip")

# Original MJX scene placed the floor at z=-0.25. mjlab's plane terrain sits at
# z=0, so the whole hand+cube assembly is lifted by this amount.
FLOOR_Z_OFFSET = 0.25

# Home keyframe from scene_mjx_cube_*_mjx.xml
HOME_JOINT_POS = {
    "if_mcp": 0.8,
    "if_rot": 0.0,
    "if_pip": 0.8,
    "if_dip": 0.8,
    "mf_mcp": 0.8,
    "mf_rot": 0.0,
    "mf_pip": 0.8,
    "mf_dip": 0.8,
    "rf_mcp": 0.8,
    "rf_rot": 0.0,
    "rf_pip": 0.8,
    "rf_dip": 0.8,
    "th_cmc": 0.8,
    "th_axl": 0.8,
    "th_mcp": 0.8,
    "th_ipl": 0.0,
}

# Grasp site in the MJCF worldbody (before FLOOR_Z_OFFSET is applied).
GRASP_SITE_POS_LOCAL = (0.11, 0.0, 0.03)
CUBE_SPAWN_POS_LOCAL = (0.1, 0.0, 0.05)


def _load_meshdir_assets(asset_dir: Path) -> dict[str, bytes]:
  """Embed mesh files keyed by basename.

  The hand MJCF sets ``meshdir="./assets/"`` and references files by basename
  (e.g. ``file="palm_face.stl"``). MuJoCo rejects duplicate logical paths in
  ``spec.assets``, so we must not also insert ``assets/...`` / ``./assets/...``
  aliases for the same file.
  """
  assets: dict[str, bytes] = {}
  if not asset_dir.exists():
    return assets
  for path in sorted(asset_dir.rglob("*")):
    if not path.is_file():
      continue
    # Prefer the first path if the same basename appears in a subdirectory.
    assets.setdefault(path.name, path.read_bytes())
  return assets


def get_hand_xml(finger_tip_type: str = "Box") -> Path:
  xml = LEAPXELA_MODEL_DIR / f"leapXela_generated_mjx_{finger_tip_type}.xml"
  if not xml.exists():
    raise FileNotFoundError(
      f"Missing LEAP-XELA MJCF at {xml}. "
      "Expected leapXELA_model next to this package (see README)."
    )
  return xml


def get_spec(finger_tip_type: str = "Box") -> mujoco.MjSpec:
  xml_path = get_hand_xml(finger_tip_type)
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.assets = _load_meshdir_assets(LEAPXELA_MODEL_DIR / "assets")
  return spec


def get_leap_xela_cfg(finger_tip_type: str = "Box") -> EntityCfg:
  """Fixed-base LEAP-XELA hand using XML position actuators."""

  def _spec_fn() -> mujoco.MjSpec:
    return get_spec(finger_tip_type)

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
