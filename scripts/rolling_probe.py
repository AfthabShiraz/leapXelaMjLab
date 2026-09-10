"""Measure how much the sensor-pad shell resists ROLLING, not sliding.

Hamid's observation on the Notion page, and the one claim behind the whole XELA
task that has never been measured: "the cube stops rolling when it hits the 4x4
finger sensor pads." His plain-LeapHand run reaches reward 375 and genuinely
reorients on video; the same script on LeapXELA plateaus at 131-197 and does not.

Why this is a different question from ``friction_probe.py`` / ``slope_test.py``
------------------------------------------------------------------------------
Those two measure **sliding**: slope_test reads the critical angle back as
``atan(mu)``, friction_probe reads tangential slip at the contacts. Both are
about a cube that translates across a surface.

Rolling is not that. A cube rolling over a surface built from 32 discrete boxes
has to climb the ridge at every box boundary, and that costs work regardless of
what ``friction[0]`` says. So the corrugation of the shell is a separate physical
quantity from its friction, and it is invisible to both existing probes -- which
is the most likely reason every friction/condim/priority intervention so far came
back null (runs 19, 26): they were all tuning sliding on a task whose residual
error is tip-over, and tipping the cube means *rolling* it against the palm.

Run 25 already answered the neighbouring question and answered it correctly: the
pads do not leave **gaps** for the cube to wedge into. Measured half-extents are
~15 x 11 x 14 mm at 16.0 mm median centre spacing, so they overlap by about 6 mm
and the shell is continuous. But "continuous" is not "smooth" -- 36 overlapping
boxes at assorted orientations form a faceted surface, and gap width says nothing
about facet height. That is what this script measures.

What is measured
----------------
The shell is frozen: every collision geom of the compiled hand is re-emitted as a
STATIC worldbody box at its home-pose world transform, so there is no hand
dynamics, no actuator, and no policy in the loop -- only geometry and contact.
The cube is settled onto the surface under gravity, then a torque about a
horizontal axis is ramped linearly and the cube's angular speed about that axis
is watched.

``tau_roll`` is the applied torque at breakaway. Compare it across surfaces:

  smooth   one flat plate, friction copied from the pads it replaces. The control.
           Without it the pad number is uninterpretable.
  pads     the real LeapXELA shell (36 boxes: 6 palm, 26 finger pads, 4 tips).
  bare     the plain LEAP hand shell, if ``--bare-xml`` points at
           ``leap_rh_mjx.xml``. Optional; skipped when absent.

Breakaway is detected on angular SPEED, not on accumulated angle. This is the
trap slope_test.py documents and it bites harder here: a cube that is stuck
against a ridge still creeps, and creep integrates, so an angle threshold trips
earlier the slower you ramp and reports a rolling resistance that depends on the
ramp rate rather than on the surface.

Usage:
  uv run python scripts/rolling_probe.py
  uv run python scripts/rolling_probe.py --surfaces smooth pads \\
      --torque-rate 0.02 --json-out eval/rolling/probe.json
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import tyro

from leap_xela_mjlab.robots.leap_xela import get_spec

# Matches the training scene (see friction_probe.py / the reference env cfg).
_CUBE_HALF = 0.035
_CUBE_MASS = 0.108
_CUBE_FRICTION = (0.3, 0.05, 0.0001)
# Density that lands the free box on _CUBE_MASS without an explicit inertial.
_CUBE_DENSITY = _CUBE_MASS / (2.0 * _CUBE_HALF) ** 3

_TIMESTEP = 0.002
# Angular speed (rad/s) about the torque axis at which the cube counts as having
# broken away. Well above settle jitter, well below a real roll.
_RELEASE_OMEGA = 0.5
# |1 - travelled/ideal| above this means the cube is not rolling and the torque
# reading is not a rolling resistance.
_SLIP_VALID = 0.6


@dataclass
class ProbeConfig:
  surfaces: tuple[str, ...] = ("smooth", "pads")
  """Which shells to test. 'bare' additionally needs --bare-xml."""
  finger_tip_type: str = "Box_palm192"
  palm_euler: tuple[float, float, float] = (0.0, 1.92, -1.57)
  """The hardware-correct palm angle: 1.92 rad = 110 deg = 90 + the 20 deg
  downward tilt of the bracket in the MuJoCo Playground paper."""
  bare_xml: Path | None = None
  """Path to playground's ``leap_rh_mjx.xml`` for the 'bare' surface."""
  region: str = "palm"
  """Which part of the shell to rest the cube on: palm | fingers | tips | all.

  'palm' is the default because the residual error on reorient is tip-over
  (96% horizontal-axis), and tipping the cube means rolling it against the palm.
  """
  normal_force: float = 5.0
  """N pressing the cube into the surface, on top of its 1.06 N of weight.

  Load-bearing, not cosmetic. Under gravity alone the cube rests on whichever
  single box protrudes furthest and touches ONE geom, which measures a balancing
  act rather than a surface. In the grasp ``friction_probe`` measures 3.4-4.9
  distinct hand geoms in contact at once, so the cube has to be pressed in for
  the test to engage the array the way a grasp does. Rolling resistance scales
  with normal load, so this is held fixed across surfaces and the comparison is
  between shells at matched load.
  """
  omegas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
  """Driven roll rates (rad/s) at which to measure the torque required.

  This is the metric that matches the claim. ``tau_roll`` below is a breakaway
  number and it answers a different question: a cube perched on a corrugated
  surface pivots about a ridge with a smaller support patch, so it starts to
  tip at LOWER torque than on a flat plate. Starting is not the problem Hamid
  described -- "the cube stops rolling" is about whether rotation continues once
  begun, which means climbing the next ridge, and the next.
  """
  tile: int = 3
  """Copies of the patch on each side along the travel axis. 0 disables."""
  servo_kv: float = 5.0
  """Gain of the velocity servo driving the roll joint."""
  hold_s: float = 2.0
  torque_rate: float = 0.02
  """N*m per second. Slow enough that the ramp is quasi-static."""
  max_torque: float = 2.0
  settle_s: float = 1.5
  release_omega: float = _RELEASE_OMEGA
  drop_axis: str = "auto"
  """Palm-frame direction to drop the cube along: auto | +x|-x|+y|-y|+z|-z."""
  json_out: Path | None = None
  seed: int = 0
  extra_frictions: tuple[float, ...] = field(default_factory=tuple)
  """Optionally re-run each surface with the pad friction overridden, to show
  that rolling resistance does NOT track friction the way sliding does."""


def _quat_from_mat(mat: np.ndarray) -> np.ndarray:
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, mat.reshape(9))
  return q


@dataclass
class Shell:
  """A frozen collision shell, expressed in the palm frame."""

  sizes: np.ndarray  # (n, 3) box half-extents
  poss: np.ndarray  # (n, 3)
  quats: np.ndarray  # (n, 4)
  frictions: np.ndarray  # (n, 3)
  condims: np.ndarray  # (n,)
  priorities: np.ndarray  # (n,)
  names: list[str]

  def __len__(self) -> int:
    return len(self.names)


def _extent_along(shell: "Shell", axis: int) -> np.ndarray:
  """Half-extent of each box along a world axis, exactly.

  For a box with half-sizes s and rotation R, the extent along axis e is
  sum_j |R[e, j]| * s[j]. Using ``sizes.max()`` instead reads a wide thin plate
  as if it were 0.56 m thick, which is what put the cube two thirds of a metre
  above the smooth control.
  """
  out = np.empty(len(shell))
  for i in range(len(shell)):
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, shell.quats[i])
    out[i] = float(np.abs(mat.reshape(3, 3)[axis]) @ shell.sizes[i])
  return out


def _extract_shell(model: mujoco.MjModel, data: mujoco.MjData) -> Shell:
  """Every collision geom, re-expressed in the palm body frame.

  Working in the palm frame rather than the world frame is what lets the drop
  direction be chosen by axis instead of by reasoning about where the palm
  normal points after ``palm_euler`` has been applied.
  """
  palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
  assert palm >= 0, "no body named 'palm' in the compiled hand"
  p_palm = data.xpos[palm].copy()
  r_palm = data.xmat[palm].reshape(3, 3).copy()

  sizes, poss, quats, frics, condims, prios, names = [], [], [], [], [], [], []
  for g in range(model.ngeom):
    if not (model.geom_contype[g] or model.geom_conaffinity[g]):
      continue
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
      # The LeapXELA shell is all boxes; anything else would need its own
      # emit path, so fail loudly rather than silently dropping contact area.
      raise ValueError(
        f"geom {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)} is type "
        f"{model.geom_type[g]}, not a box; extend _emit_shell before using it"
      )
    r_world = data.geom_xmat[g].reshape(3, 3)
    poss.append(r_palm.T @ (data.geom_xpos[g] - p_palm))
    quats.append(_quat_from_mat(np.ascontiguousarray(r_palm.T @ r_world)))
    sizes.append(model.geom_size[g].copy())
    frics.append(model.geom_friction[g].copy())
    condims.append(int(model.geom_condim[g]))
    prios.append(int(model.geom_priority[g]))
    names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}")

  return Shell(
    sizes=np.array(sizes),
    poss=np.array(poss),
    quats=np.array(quats),
    frictions=np.array(frics),
    condims=np.array(condims),
    priorities=np.array(prios),
    names=names,
  )


def _hand_shell(cfg: ProbeConfig, xml: Path | None) -> Shell:
  if xml is None:
    spec = get_spec(cfg.finger_tip_type, palm_euler=cfg.palm_euler)
  else:
    spec = mujoco.MjSpec.from_file(str(xml))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return _extract_shell(model, data)


# The compiled shell names geoms per phalanx: ``<finger>_<segment>_uspa<N>`` for
# a XELA sensor pad, ``<finger>_<segment>_collision_<N>`` for structural
# collision, ``palm_collision_<N>`` and bare ``uspa<N>_<M>`` on the palm, plus
# four ``*_tip``. 36 in total. Matching "uspa46" alone caught 3 of them.
_FINGERS = ("if_", "mf_", "rf_", "th_")
_REGIONS = {
  "palm": lambda n: n.startswith("palm_collision") or n.startswith("uspa"),
  "fingers": lambda n: n.startswith(_FINGERS),
  # The XELA pads themselves, wherever they sit -- this is the "4x4 finger
  # sensor pads" of Hamid's observation.
  "sensors": lambda n: "uspa" in n,
  "tips": lambda n: n.endswith("_tip"),
  "all": lambda n: True,
}


def _select(shell: Shell, region: str) -> Shell:
  keep = [i for i, n in enumerate(shell.names) if _REGIONS[region](n)]
  if not keep:
    raise ValueError(f"region {region!r} matched no geoms in {shell.names}")
  return Shell(
    sizes=shell.sizes[keep],
    poss=shell.poss[keep],
    quats=shell.quats[keep],
    frictions=shell.frictions[keep],
    condims=shell.condims[keep],
    priorities=shell.priorities[keep],
    names=[shell.names[i] for i in keep],
  )


def _tile(shell: Shell, axis: int, n: int) -> Shell:
  """Repeat the patch along the travel axis to make a track.

  A 70 mm cube cannot complete a roll on a ~50 mm palm patch: it reaches the
  edge within a fraction of a turn and falls off, which reads as enormous
  negative slip and a torque that means nothing. Tiling the same boxes at the
  patch period gives the cube somewhere to roll TO, so the measurement is of the
  surface rather than of its edge. Macro curvature of the palm is deliberately
  not reproduced -- corrugation is the quantity under test.
  """
  travel = (axis + 2) % 3
  reach = shell.sizes.max()
  lo = (shell.poss[:, travel] - reach).min()
  hi = (shell.poss[:, travel] + reach).max()
  period = float(hi - lo)

  poss, sizes, quats, frics, condims, prios, names = [], [], [], [], [], [], []
  for k in range(-n, n + 1):
    for i in range(len(shell)):
      p = shell.poss[i].copy()
      p[travel] += k * period
      poss.append(p)
      sizes.append(shell.sizes[i])
      quats.append(shell.quats[i])
      frics.append(shell.frictions[i])
      condims.append(shell.condims[i])
      prios.append(shell.priorities[i])
      names.append(f"{shell.names[i]}_t{k}")
  return Shell(
    sizes=np.array(sizes), poss=np.array(poss), quats=np.array(quats),
    frictions=np.array(frics), condims=np.array(condims),
    priorities=np.array(prios), names=names,
  )


def _smooth_shell(pads: Shell, axis: int, sign: float) -> Shell:
  """One flat plate standing in for the pad shell, same friction.

  Placed so its contact face sits at the outermost extent of the pads along the
  drop axis, i.e. the cube rests at the same height it would on the real shell.
  """
  face = float((pads.poss[:, axis] * sign + _extent_along(pads, axis)).max())
  # The plate must match the FOOTPRINT of the patch it replaces, not just its
  # friction. A 160 mm plate against a 50 mm patch compares "large flat surface"
  # with "small bumpy surface": the cube topples off the edge of the small one
  # early and the bumps never get a say. Match the in-plane extent so the only
  # difference left is the corrugation.
  reach = pads.sizes.max()
  lo = (pads.poss - reach).min(axis=0)
  hi = (pads.poss + reach).max(axis=0)
  size = np.maximum((hi - lo) / 2.0, 1e-3)
  # Widen across the roll axis so the cube cannot walk off the side.
  size[(axis + 1) % 3] = max(size[(axis + 1) % 3], 4 * _CUBE_HALF)
  size[axis] = 0.01
  # Centred on the pads it replaces, so the cube rests over the same spot.
  pos = (hi + lo) / 2.0
  pos[axis] = sign * (face + 0.01)
  # Friction copied from the pads, so the only difference is the geometry.
  fric = pads.frictions.mean(axis=0)
  return Shell(
    sizes=size[None, :],
    poss=pos[None, :],
    quats=np.array([[1.0, 0.0, 0.0, 0.0]]),
    frictions=fric[None, :],
    condims=np.array([int(np.max(pads.condims))]),
    priorities=np.array([0]),
    names=["smooth_plate"],
  )


def _build_driven_scene(
  shell: Shell, gravity: np.ndarray, cube_pos: np.ndarray, axis: int, kv: float
) -> mujoco.MjModel:
  """The cube on a rolling rig, driven at constant angular velocity.

  Why a rig and not a free cube with a torque applied: a free rigid body under a
  pure torque is unconstrained once it breaks static friction, so it spins up as
  ``omega = tau*t/I`` and the surface underneath stops mattering entirely. The
  first version of this script did that and measured 866 rev/s on both surfaces,
  which is the free-body answer and says nothing about rolling.

  Instead the cube keeps three degrees of freedom -- roll about the horizontal
  axis, travel along the surface, and lift along the normal so it can ride over
  a ridge -- and a velocity servo drives the roll. The torque the servo has to
  supply IS the rolling resistance, and it stays bounded and comparable.
  """
  spec = mujoco.MjSpec()
  spec.option.timestep = _TIMESTEP
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.gravity = gravity.tolist()

  for i in range(len(shell)):
    spec.worldbody.add_geom(
      name=f"shell_{i}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=shell.sizes[i].tolist(),
      pos=shell.poss[i].tolist(),
      quat=shell.quats[i].tolist(),
      friction=shell.frictions[i].tolist(),
      condim=int(shell.condims[i]),
      priority=int(shell.priorities[i]),
    )

  torque_axis = np.zeros(3)
  torque_axis[(axis + 1) % 3] = 1.0
  travel_axis = np.zeros(3)
  travel_axis[(axis + 2) % 3] = 1.0
  normal_axis = np.zeros(3)
  normal_axis[axis] = 1.0

  body = spec.worldbody.add_body(name="cube", pos=cube_pos.tolist())
  body.add_joint(
    name="travel", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=travel_axis.tolist()
  )
  body.add_joint(
    name="lift", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=normal_axis.tolist()
  )
  body.add_joint(
    name="roll", type=mujoco.mjtJoint.mjJNT_HINGE, axis=torque_axis.tolist()
  )
  body.add_geom(
    name="cube",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[_CUBE_HALF] * 3,
    density=_CUBE_DENSITY,
    friction=list(_CUBE_FRICTION),
    condim=3,
    priority=1,
  )
  # Velocity servo: force = kv * (ctrl - qvel).
  spec.add_actuator(
    name="roll_drive",
    target="roll",
    trntype=mujoco.mjtTrn.mjTRN_JOINT,
    gaintype=mujoco.mjtGain.mjGAIN_FIXED,
    biastype=mujoco.mjtBias.mjBIAS_AFFINE,
    gainprm=[kv] + [0.0] * 9,
    biasprm=[0.0, 0.0, -kv] + [0.0] * 7,
  )
  return spec.compile()


def _build_scene(
  shell: Shell, gravity: np.ndarray, cube_pos: np.ndarray
) -> mujoco.MjModel:
  spec = mujoco.MjSpec()
  spec.option.timestep = _TIMESTEP
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.gravity = gravity.tolist()

  for i in range(len(shell)):
    spec.worldbody.add_geom(
      name=f"shell_{i}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=shell.sizes[i].tolist(),
      pos=shell.poss[i].tolist(),
      quat=shell.quats[i].tolist(),
      friction=shell.frictions[i].tolist(),
      condim=int(shell.condims[i]),
      priority=int(shell.priorities[i]),
    )

  body = spec.worldbody.add_body(name="cube", pos=cube_pos.tolist())
  body.add_freejoint()
  body.add_geom(
    name="cube",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[_CUBE_HALF] * 3,
    density=_CUBE_DENSITY,
    friction=list(_CUBE_FRICTION),
    condim=3,
    # Priority 1 so the cube dictates contact friction, matching the current
    # training default. The shell's own friction still sets the SMOOTH control's
    # value, which is why _smooth_shell copies it rather than inventing one.
    priority=1,
  )
  return spec.compile()


_AXES = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0),
         "+z": (2, 1.0), "-z": (2, -1.0)}


def _start_pos(shell: Shell, axis: int, sign: float) -> np.ndarray:
  """Above the centroid of the surface, clear of it along the drop axis.

  Over the centroid rather than over the palm-frame origin: dropping on the
  origin lands the cube on whichever geom happens to sit there, which for the
  finger arrays is nowhere near the patch being tested.
  """
  start = shell.poss.mean(axis=0)
  # Clearance is measured from the surface's extent along the DROP axis only.
  # Using sizes.max() across all dimensions put the cube 0.67 m above the wide
  # smooth plate, and it then tunnelled through the 20 mm plate at 8.7 m/s
  # (17 mm of travel per 2 ms step). Drop it from just clear of the surface.
  face = float((shell.poss[:, axis] * sign + _extent_along(shell, axis)).max())
  start[axis] = sign * (face + 2 * _CUBE_HALF + 0.005)
  return start


def _press(
  data: mujoco.MjData, cube_body: int, axis: int, sign: float, force: float
) -> None:
  """Hold the cube against the surface. Cleared torque, normal force only."""
  data.xfrc_applied[cube_body, :3] = 0.0
  data.xfrc_applied[cube_body, axis] = -sign * force


def _settle(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  steps: int,
  cube_body: int,
  axis: int,
  sign: float,
  force: float,
) -> tuple[int, int, float]:
  """Step until the cube stops moving under gravity plus the normal load.

  Returns (n_contacts, n_distinct_geoms, final speed).
  """
  cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
  for _ in range(steps):
    _press(data, cube_body, axis, sign, force)
    mujoco.mj_step(model, data)
  touching, distinct = 0, set()
  for i in range(data.ncon):
    c = data.contact[i]
    if cube_geom not in (c.geom1, c.geom2):
      continue
    touching += 1
    distinct.add(int(c.geom2 if c.geom1 == cube_geom else c.geom1))
  return touching, len(distinct), float(np.linalg.norm(data.qvel[:3]))


def _choose_drop_axis(cfg: ProbeConfig, shell: Shell) -> tuple[int, float]:
  """Pick the palm-frame direction the cube can actually rest against.

  Tried by simulation rather than by geometry: drop along each candidate and
  keep the one where the cube ends up in contact and at rest. This is what makes
  the script independent of how ``palm_euler`` orients the hand.
  """
  if cfg.drop_axis != "auto":
    return _AXES[cfg.drop_axis]

  best, best_score = None, None
  for label, (axis, sign) in _AXES.items():
    gravity = np.zeros(3)
    gravity[axis] = -sign * 9.81
    try:
      model = _build_scene(shell, gravity, _start_pos(shell, axis, sign))
    except Exception:
      continue
    data = mujoco.MjData(model)
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    contacts, distinct, speed = _settle(
      model, data, int(2.0 / _TIMESTEP), cube_body, axis, sign, cfg.normal_force
    )
    if contacts == 0:
      continue
    score = (distinct, contacts, -speed)
    if best_score is None or score > best_score:
      best, best_score = (axis, sign), score
      print(
        f"  [drop] {label}: {contacts} contacts on {distinct} geoms, "
        f"|v| {speed * 1e3:.2f} mm/s"
      )
  if best is None:
    raise RuntimeError(
      "the cube never came to rest on this shell in any direction; "
      "pass --drop-axis explicitly"
    )
  return best


def _drive(
  cfg: ProbeConfig, shell: Shell, axis: int, sign: float, omega: float
) -> dict:
  """Roll the cube at constant angular velocity; report the torque it takes.

  Two numbers come out. ``torque_mean`` is the rolling resistance. ``slip`` is
  how far short of ``omega * r`` the cube travelled: a cube that is climbing
  ridges rather than rolling over them spins against the surface without going
  anywhere, and that shows up here as slip approaching 1.0 while the torque
  climbs.
  """
  gravity = np.zeros(3)
  gravity[axis] = -sign * 9.81
  model = _build_driven_scene(
    shell, gravity, _start_pos(shell, axis, sign), axis, cfg.servo_kv
  )
  data = mujoco.MjData(model)
  cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
  cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube")

  # Settle with the servo held at zero so the cube beds into the surface.
  for _ in range(int(cfg.settle_s / _TIMESTEP)):
    _press(data, cube_body, axis, sign, cfg.normal_force)
    data.ctrl[0] = 0.0
    mujoco.mj_step(model, data)
  if not any(
    cube_geom in (data.contact[i].geom1, data.contact[i].geom2)
    for i in range(data.ncon)
  ):
    return {"error": "no contact"}

  travel_qpos = 0  # joint order: travel, lift, roll
  start_travel = float(data.qpos[travel_qpos])
  torques, touched = [], set()
  for _ in range(int(cfg.hold_s / _TIMESTEP)):
    _press(data, cube_body, axis, sign, cfg.normal_force)
    data.ctrl[0] = omega
    mujoco.mj_step(model, data)
    torques.append(abs(float(data.actuator_force[0])))
    for i in range(data.ncon):
      c = data.contact[i]
      if cube_geom in (c.geom1, c.geom2):
        touched.add(int(c.geom2 if c.geom1 == cube_geom else c.geom1))

  travelled = abs(float(data.qpos[travel_qpos]) - start_travel)
  # A cube advances one full edge (2*half) per 90 deg, so the effective rolling
  # radius is (2*half)/(pi/2), not `half`.
  r_eff = (2.0 * _CUBE_HALF) / (np.pi / 2.0)
  ideal = abs(omega) * r_eff * cfg.hold_s
  return {
    "omega_rad_s": omega,
    "torque_mean_N_m": float(np.mean(torques)),
    "torque_p95_N_m": float(np.percentile(torques, 95)),
    "travelled_mm": travelled * 1e3,
    "ideal_mm": ideal * 1e3,
    "slip": float(1.0 - travelled / ideal) if ideal > 0 else float("nan"),
    "distinct_geoms": len(touched),
  }


def _measure(cfg: ProbeConfig, shell: Shell, axis: int, sign: float) -> dict:
  gravity = np.zeros(3)
  gravity[axis] = -sign * 9.81

  model = _build_scene(shell, gravity, _start_pos(shell, axis, sign))
  data = mujoco.MjData(model)
  cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
  cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube")

  contacts, distinct, speed = _settle(
    model, data, int(cfg.settle_s / _TIMESTEP), cube_body, axis, sign,
    cfg.normal_force,
  )
  if contacts == 0:
    return {"error": "cube did not settle in contact with the shell"}
  rest_pos = data.qpos[:3].copy()

  # Torque about a horizontal axis, i.e. any axis perpendicular to gravity.
  torque_axis = np.zeros(3)
  torque_axis[(axis + 1) % 3] = 1.0

  tau = 0.0
  n_max = int(cfg.max_torque / (cfg.torque_rate * _TIMESTEP))
  tau_roll = None
  depths: list[float] = []
  pads_touched: set[int] = set()

  for _ in range(n_max):
    tau += cfg.torque_rate * _TIMESTEP
    # Normal load is held through the ramp: lift it and the cube would simply
    # be levered off the surface rather than rolled along it.
    _press(data, cube_body, axis, sign, cfg.normal_force)
    data.xfrc_applied[cube_body, 3:6] = tau * torque_axis
    mujoco.mj_step(model, data)

    for i in range(data.ncon):
      c = data.contact[i]
      if cube_geom not in (c.geom1, c.geom2):
        continue
      depths.append(-float(c.dist))
      pads_touched.add(int(c.geom2 if c.geom1 == cube_geom else c.geom1))

    omega = float(abs(np.dot(data.qvel[3:6], torque_axis)))
    if omega > cfg.release_omega:
      tau_roll = tau
      break

  travel = float(np.linalg.norm(data.qpos[:3] - rest_pos))
  return {
    "n_geoms": len(shell),
    "settle_contacts": contacts,
    "settle_speed_mm_s": speed * 1e3,
    "tau_roll_N_m": tau_roll,
    "rolled": tau_roll is not None,
    "max_torque_N_m": cfg.max_torque,
    "pen_p95_mm": float(np.percentile(depths, 95) * 1e3) if depths else 0.0,
    "distinct_geoms_touched": len(pads_touched),
    "translation_at_breakaway_mm": travel * 1e3,
    "shell_friction": float(shell.frictions[:, 0].mean()),
  }


def _fmt(v) -> str:
  if v is None:
    return "  --  "
  return f"{v:.4f}" if isinstance(v, float) else str(v)


def main(cfg: ProbeConfig) -> None:
  np.random.seed(cfg.seed)
  results: dict[str, dict] = {}

  print("building shells...")
  full = _hand_shell(cfg, None)
  pads = _select(full, cfg.region)
  print(f"  shell: {len(full)} collision geoms, region {cfg.region!r} keeps "
        f"{len(pads)} (friction[0] mean {pads.frictions[:, 0].mean():.3f})")

  axis, sign = _choose_drop_axis(cfg, pads)
  label = next(k for k, v in _AXES.items() if v == (axis, sign))
  print(f"  drop axis (palm frame): {label}")

  shells: dict[str, Shell] = {}
  if "pads" in cfg.surfaces:
    shells["pads"] = _tile(pads, axis, cfg.tile) if cfg.tile else pads
  if "smooth" in cfg.surfaces:
    shells["smooth"] = _smooth_shell(
      _tile(pads, axis, cfg.tile) if cfg.tile else pads, axis, sign
    )
  if "bare" in cfg.surfaces:
    if cfg.bare_xml is None or not cfg.bare_xml.exists():
      print("  [skip] 'bare' needs --bare-xml pointing at leap_rh_mjx.xml")
    else:
      _b = _select(_hand_shell(cfg, cfg.bare_xml), cfg.region)
      shells["bare"] = _tile(_b, axis, cfg.tile) if cfg.tile else _b
      print(f"  bare: {len(shells['bare'])} collision geoms in region")

  for name, shell in shells.items():
    print(f"\nmeasuring {name}...")
    results[name] = _measure(cfg, shell, axis, sign)
    results[name]["driven"] = [
      _drive(cfg, shell, axis, sign, w) for w in cfg.omegas
    ]

  bar = "=" * 78
  print(f"\n{bar}\nROLLING RESISTANCE OF THE COLLISION SHELL\n{bar}")
  print(f"  region         {cfg.region}")
  print(f"  normal load    {cfg.normal_force} N pressed in, "
        f"+ {_CUBE_MASS * 9.81:.2f} N of weight")
  print(f"  torque ramp    {cfg.torque_rate} N*m/s to {cfg.max_torque} N*m")
  print(f"  breakaway      |omega| > {cfg.release_omega} rad/s about the torque axis")
  print(f"  cube           {2 * _CUBE_HALF * 1e3:.0f} mm, {_CUBE_MASS} kg\n")
  head = (f"  {'surface':10} {'geoms':>6} {'mu':>6} {'tau_roll':>10} "
          f"{'pen_p95':>9} {'touched':>8} {'slid':>8}")
  print(head)
  print(f"  {'-' * (len(head) - 2)}")
  for name, r in results.items():
    if "error" in r:
      print(f"  {name:10} {r['error']}")
      continue
    print(
      f"  {name:10} {r['n_geoms']:>6} {r['shell_friction']:>6.2f} "
      f"{_fmt(r['tau_roll_N_m']):>10} {r['pen_p95_mm']:>8.2f}mm "
      f"{r['distinct_geoms_touched']:>8} "
      f"{r['translation_at_breakaway_mm']:>7.1f}mm"
    )

  print(f"\n  DRIVEN ROLL: torque required to hold each rate for {cfg.hold_s} s\n")
  cols = "".join(f"{w:>13.1f}" for w in cfg.omegas)
  print(f"  {'surface':10}{cols}      <- omega, rad/s")
  print(f"  {'-' * (10 + 13 * len(cfg.omegas))}")
  # A torque number is only meaningful where the cube is actually ROLLING.
  # |slip| > _SLIP_VALID means it is sliding or tumbling down the surface
  # instead, which happens on the curved finger regions (they do not tile into a
  # track) and at high omega. Those cells are blanked rather than printed, so a
  # later reader cannot mistake them for measurements.
  for name, r in results.items():
    row = ""
    for d in r.get("driven", []):
      slip, tq = d.get("slip", float("nan")), d.get("torque_mean_N_m")
      row += "         --  " if (tq is None or not abs(slip) <= _SLIP_VALID) \
        else f"{tq:>13.4f}"
    print(f"  {name:10}{row}  N*m")
  for name, r in results.items():
    row = "".join(
      f"{d.get('slip', float('nan')):>13.3f}" for d in r.get("driven", [])
    )
    print(f"  {name:10}{row}  slip")
  print(f"\n  cells blanked where |slip| > {_SLIP_VALID}: the cube was sliding "
        f"or tumbling, not rolling, so the torque is not a rolling resistance.")

  if "smooth" in results and "pads" in results:
    s, p = results["smooth"].get("tau_roll_N_m"), results["pads"].get("tau_roll_N_m")
    if s and p:
      print(f"\n  pads / smooth = {p / s:.2f}x the torque to start rolling")
    elif s and not p:
      print(f"\n  the cube NEVER rolled on the pads up to {cfg.max_torque} N*m, "
            f"against {s:.4f} N*m on the smooth plate")
  print(bar)

  if cfg.json_out:
    cfg.json_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.json_out.write_text(
      json.dumps({"config": vars(cfg) | {"json_out": str(cfg.json_out),
                                         "bare_xml": str(cfg.bare_xml)},
                  "results": results}, indent=2, default=str)
    )
    print(f"\n[INFO] wrote {cfg.json_out}")


if __name__ == "__main__":
  main(tyro.cli(ProbeConfig))
