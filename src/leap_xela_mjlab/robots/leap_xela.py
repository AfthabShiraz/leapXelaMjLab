"""LEAP-XELA hand EntityCfg for mjlab."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

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


# Palm orientation, as the xyz Euler triple that
# ``simplify_model_for_mjx.py --palm-euler`` bakes into ``<body name="palm">``'s
# quat. It is the single most consequential geometric knob on this task -- see
# the Notion reference in TRAINING_NOTES.md, where 1.88 plateaus at reward ~150
# for every cube scale tried and 1.92 is the only angle that takes off -- but it
# lived in the MJCF, so varying it meant regenerating a model and adding a new
# ``finger_tip_type``. ``PALM_EULER_BAKED`` records what each shipped file
# already contains, so ``palm_euler`` can default to "leave the file alone" and
# a sweep can override it without touching disk.
PALM_EULER_BAKED: dict[str, tuple[float, float, float]] = {
    "Box": (0.0, 1.88, -1.57),
    "Box_palm192": (0.0, 1.92, -1.57),
}

# Reproduced from ``overwrite_pose_of_the_hand``; the generator sets pos as well
# as quat, and both shipped XMLs carry this exact value.
PALM_POS = (0.0, 0.011, -0.01)


def palm_euler_to_quat(euler: tuple[float, float, float]) -> list[float]:
  """xyz Euler -> wxyz quat, via the same helper the model generator uses.

  Verified to reproduce the baked attributes bit for bit: 1.88 -> (0.417209,
  -0.570802, 0.571257, -0.416877) and 1.92 -> (0.405701, -0.579025, 0.579487,
  -0.405378), matching ``leapXela_generated_mjx_Box{,_palm192}.xml``.
  """
  quat = np.zeros((4, 1), dtype=np.float64)
  mujoco.mju_euler2Quat(
    quat, np.array(euler, dtype=np.float64).reshape(3, 1), "xyz"
  )
  return quat.flatten().tolist()


def get_hand_xml(finger_tip_type: str = "Box") -> Path:
  xml = LEAPXELA_MODEL_DIR / f"leapXela_generated_mjx_{finger_tip_type}.xml"
  if not xml.exists():
    raise FileNotFoundError(
      f"Missing LEAP-XELA MJCF at {xml}. "
      "Expected leapXELA_model next to this package (see README)."
    )
  return xml


# Joint limits of the BARE LEAP hand, read per joint name from the vendored
# ``assets/leap_plain/leap_rh_mjx.xml`` (mujoco_playground's model, whose meshes
# come from mujoco_menagerie). ``ctrlrange == jnt_range`` throughout that model,
# so one table drives both.
#
# Why this is here (2026-09-12). The generated LeapXELA MJCF bakes in the
# ``leapXela.sim`` section of ``joint_config.json`` instead, and those ranges are
# materially tighter -- most sharply the lateral-splay ``rot`` joints at +/-0.349
# rad (+/-20 deg) against LEAP's +/-1.047 (+/-60 deg), a 3x reduction. Lateral
# splay is the motion that rotates a held cube about a HORIZONTAL axis, which is
# the component this task has never been able to steer, so the restriction is a
# live suspect rather than a detail. It has never been varied.
#
# Taken from the MJCF and NOT from ``joint_config.json``, deliberately: the
# config's thumb rows do not map onto the model's thumb joints one-for-one
# (its ``thumb.axl`` row is the model's ``th_mcp``, and its ``thumb.mcp`` row
# appears nowhere), so keying off the config silently mis-assigns two joints.
# The config is also partly malformed -- ``leapXela.sim.fingers.rot`` is written
# ``{"lower": 0.349, "upper": -0.349}``, lower ABOVE upper, as are
# ``fingers.mcp`` and ``thumb.mcp``. The magnitudes still agree with the
# ``leapXela.hardware`` section (pi +/- 0.349 for rot), so +/-20 deg is the
# intended figure -- the slip is in the order, not the value.
_LEAP_JOINT_RANGE: dict[str, tuple[float, float]] = {
  "if_mcp": (-0.314, 2.23),
  "if_rot": (-1.047, 1.047),
  "if_pip": (-0.506, 1.885),
  "if_dip": (-0.366, 2.042),
  "mf_mcp": (-0.314, 2.23),
  "mf_rot": (-1.047, 1.047),
  "mf_pip": (-0.506, 1.885),
  "mf_dip": (-0.366, 2.042),
  "rf_mcp": (-0.314, 2.23),
  "rf_rot": (-1.047, 1.047),
  "rf_pip": (-0.506, 1.885),
  "rf_dip": (-0.366, 2.042),
  "th_cmc": (-0.349, 2.094),
  "th_axl": (-0.349, 2.094),
  "th_mcp": (-0.47, 2.443),
  "th_ipl": (-1.34, 1.88),
}


def _apply_leap_joint_limits(spec: mujoco.MjSpec) -> None:
  """Widen the hand's joint (and actuator ctrl) ranges to the bare-LEAP values.

  Both are set: the actuators are XML position actuators whose ``ctrlrange``
  mirrors ``jnt_range`` in the generated file, and leaving ctrlrange alone would
  clamp the command before the joint limit ever mattered -- the same class of
  miss as the 2026-08-28 bug, where limits were written to a ``<default>`` block
  that MjSpec had already resolved and ``ctrlrange`` was never touched at all.
  """
  seen = set()
  for joint in spec.joints:
    rng = _LEAP_JOINT_RANGE.get(joint.name)
    if rng is None:
      raise ValueError(f"no bare-LEAP range for joint {joint.name!r}")
    joint.range = list(rng)
    seen.add(joint.name)
  missing = set(_LEAP_JOINT_RANGE) - seen
  if missing:
    raise ValueError(f"bare-LEAP ranges never applied to: {sorted(missing)}")
  for act in spec.actuators:
    rng = _LEAP_JOINT_RANGE.get(act.target)
    if rng is not None:
      act.ctrlrange = list(rng)


def get_spec(
  finger_tip_type: str = "Box",
  palm_euler: tuple[float, float, float] | None = None,
  leap_joint_limits: bool = False,
) -> mujoco.MjSpec:
  """``palm_euler=None`` keeps whatever the generated MJCF already has baked in.

  ``leap_joint_limits=True`` replaces the baked LeapXELA ranges with the bare
  LEAP ones -- see ``_LEAP_JOINT_RANGE``. Pads, meshes, inertia and actuator
  gains are untouched, so it isolates the joint limits from the tactile skin.
  """
  xml_path = get_hand_xml(finger_tip_type)
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.assets = _load_meshdir_assets(LEAPXELA_MODEL_DIR / "assets")
  if palm_euler is not None:
    palm = spec.body("palm")
    palm.pos = list(PALM_POS)
    palm.quat = palm_euler_to_quat(palm_euler)
  if leap_joint_limits:
    _apply_leap_joint_limits(spec)
  return spec


def get_leap_xela_cfg(
  finger_tip_type: str = "Box",
  palm_euler: tuple[float, float, float] | None = None,
  leap_joint_limits: bool = False,
) -> EntityCfg:
  """Fixed-base LEAP-XELA hand using XML position actuators."""

  def _spec_fn() -> mujoco.MjSpec:
    return get_spec(
      finger_tip_type, palm_euler=palm_euler, leap_joint_limits=leap_joint_limits
    )

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
