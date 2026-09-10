"""Validate the cube's sliding friction physically, on an inclined plane.

Run this BEFORE committing a friction value to a training run. A rigid block on
an incline is the one case where Coulomb friction has a closed-form answer: it
stays put iff

    tan(theta) <= mu_sliding

so the angle at which the cube starts to slide *is* a direct readout of the
friction the solver is actually applying. If the measured critical angle does
not match ``atan(mu)``, the number in the config is not the number in the
contact, and no amount of training will fix that.

That failure is not hypothetical here. The ramp in this scene is deliberately
given friction 1.0 -- the same value the training terrain has -- while the cube
is given whatever ``--frictions`` asks for. With the pre-2026-09 model, where
every geom had ``priority`` 0, MuJoCo mixed contact friction as the element-wise
**max**, so the ramp's 1.0 won every time and *every* cube stuck at 45 degrees
regardless of its own friction. Run this with ``--cube-priority 0`` to see that
directly: the measured critical angle is flat at ~45 deg across the whole sweep.
At ``--cube-priority 1`` (the new default) the cube outranks the ramp and each
cube breaks away at its own ``atan(mu)``.

Outputs
-------
* a table of predicted vs measured critical angle per friction value, found by
  bisection on the ramp angle (headless, no rendering)
* an MP4 of every cube released simultaneously on one ramp, so the sweep is
  visible rather than just tabulated

Usage:
  uv run python scripts/slope_test.py --frictions 0.05 0.1 0.2 0.3 0.5 0.8 1.2 \\
      --ramp-angle-deg 20 --out eval/friction/slope.mp4
"""

from __future__ import annotations

import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from leap_xela_mjlab.robots.cube import get_cube_spec

# Matches the training scene: cube half-size 0.035 (scale 1.0), mass 0.108, and
# the terrain's sliding friction, which is what the ramp stands in for.
_CUBE_HALF = 0.035
_CUBE_MASS = 0.108
_RAMP_FRICTION = 1.0
# Ramp-frame sliding speed (m/s) at which a cube counts as having broken away.
_RELEASE_SPEED = 0.03

_SCENE_XML = """
<mujoco model="slope_test">
  <!-- MuJoCo's default angle unit is DEGREES; the ramp angle is passed in
       radians, so this directive is load-bearing, not boilerplate. -->
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offheight="720" offwidth="1280"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.24 0.26 0.30"
             rgb2="0.30 0.32 0.36" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
    <material name="ramp" rgba="0.45 0.48 0.55 1"/>
  </asset>
  <worldbody>
    <light pos="0 -1 2.5" dir="0 0.3 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"
          friction="1 0.005 0.0001"/>
    <!-- mocap so the tilt sweep can drive the angle frame by frame. A mocap
         body is static as far as the solver is concerned, so the cubes rest on
         it exactly as they would on a welded ramp. -->
    <body name="ramp" mocap="true" pos="0 0 {rz}" euler="0 {ramp_rad} 0">
      <geom name="ramp" type="box" size="{rl} {rw} 0.01" pos="0 0 -0.01"
            material="ramp" friction="{rf} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


# The ramp is lifted clear of the floor plane so a sliding cube never reaches it
# and the only contact under test is cube-vs-ramp.
_RAMP_Z = 0.5


def _build(frictions: tuple[float, ...], angle_rad: float, priority: int,
           spacing: float = 0.11) -> mujoco.MjSpec:
  """One ramp, one cube per friction value, laid out across the ramp's width."""
  n = len(frictions)
  ramp_len = 1.2
  ramp_width = max(0.25, spacing * n * 0.5 + 0.1)
  spec = mujoco.MjSpec.from_string(
    _SCENE_XML.format(ramp_rad=angle_rad, rl=ramp_len, rw=ramp_width,
                      rf=_RAMP_FRICTION, rz=_RAMP_Z)
  )
  # A free joint has to live on a top-level body, so the cubes attach to the
  # worldbody and carry the ramp's rotation themselves rather than being nested
  # inside it. R_y(theta) maps ramp-local coordinates to world.
  c, sn = math.cos(angle_rad), math.sin(angle_rad)
  rot = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])
  quat = [math.cos(angle_rad / 2.0), 0.0, math.sin(angle_rad / 2.0), 0.0]
  world = spec.worldbody
  for i, mu in enumerate(frictions):
    cube = get_cube_spec(
      half_size=_CUBE_HALF, mass=_CUBE_MASS,
      friction=(mu, 0.05, 0.0001), condim=3, priority=priority,
    )
    # Sit the cubes ON the ramp's pivot (local x = 0). The tilt sweep rotates
    # the ramp about its own origin, so anywhere else the surface would sweep
    # out from under them and drag them loose well before their true critical
    # angle -- at x = 0.35 m that spurious slip is ~37 mm/s at 6 deg/s. At the
    # pivot the contact point is stationary and the release angle is honest.
    # There is still a full ramp half-length of runway down-slope.
    local = np.array([0.0, (i - (n - 1) / 2.0) * spacing, _CUBE_HALF + 1e-4])
    pos = rot @ local + np.array([0.0, 0.0, _RAMP_Z])
    frame = world.add_frame(pos=pos.tolist(), quat=quat)
    spec.attach(cube, prefix=f"c{i}_", frame=frame)
  return spec


def _slide_distance(frictions: tuple[float, ...], angle_rad: float, priority: int,
                    seconds: float = 2.0) -> np.ndarray:
  """Tangential displacement of each cube after settling + sliding, in metres."""
  spec = _build(frictions, angle_rad, priority)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  bodies = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"c{i}_cube")
    for i in range(len(frictions))
  ]
  # Let the cubes settle onto the ramp before the displacement clock starts, so
  # the initial 0.1 mm drop is not counted as sliding.
  for _ in range(int(0.5 / model.opt.timestep)):
    mujoco.mj_step(model, data)
  start = np.array([data.xpos[b].copy() for b in bodies])
  for _ in range(int(seconds / model.opt.timestep)):
    mujoco.mj_step(model, data)
  end = np.array([data.xpos[b].copy() for b in bodies])
  return np.linalg.norm(end - start, axis=1)


def _critical_angle(mu: float, priority: int, lo_deg: float = 1.0,
                    hi_deg: float = 60.0, iters: int = 12,
                    slide_threshold: float = 0.02) -> float:
  """Bisect the ramp angle for the smallest angle at which this cube slides."""
  lo, hi = lo_deg, hi_deg
  # Guard: if it already slides at lo, or never slides by hi, report the bound.
  if _slide_distance((mu,), math.radians(lo), priority)[0] > slide_threshold:
    return lo
  if _slide_distance((mu,), math.radians(hi), priority)[0] <= slide_threshold:
    return hi
  for _ in range(iters):
    mid = 0.5 * (lo + hi)
    if _slide_distance((mu,), math.radians(mid), priority)[0] > slide_threshold:
      hi = mid
    else:
      lo = mid
  return 0.5 * (lo + hi)


# Low friction -> warm, high friction -> cool. Distinct enough to read off a
# legend at video resolution, and ordered so the release sequence is obvious.
_PALETTE = [
  (0.90, 0.24, 0.22),  # red
  (0.95, 0.52, 0.15),  # orange
  (0.93, 0.78, 0.20),  # yellow
  (0.45, 0.75, 0.30),  # green
  (0.20, 0.66, 0.60),  # teal
  (0.25, 0.50, 0.85),  # blue
  (0.55, 0.36, 0.78),  # purple
  (0.85, 0.40, 0.65),  # pink
]


def _tint(model: mujoco.MjModel, n: int) -> list[tuple[int, int, int]]:
  """Colour each cube's collision box and return the matching legend colours."""
  out = []
  for i in range(n):
    r, g, b = _PALETTE[i % len(_PALETTE)]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"c{i}_cube")
    if gid >= 0:
      model.geom_rgba[gid] = [r, g, b, 1.0]
      model.geom_matid[gid] = -1  # drop the shared material so rgba is used
    out.append((int(r * 255), int(g * 255), int(b * 255)))
  return out


def _font(size: int):
  for path in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
  ):
    if Path(path).exists():
      return ImageFont.truetype(path, size)
  return ImageFont.load_default()


def _collision_only_option() -> mujoco.MjvOption:
  """Draw the ramp/floor (group 0) and the cubes' collision boxes (group 3).

  The cube's group-2 textured mesh is hidden so the per-cube tint is visible --
  and so what you are watching is the geometry the solver actually uses.
  """
  opt = mujoco.MjvOption()
  opt.geomgroup[:] = 0
  opt.geomgroup[0] = 1
  opt.geomgroup[3] = 1
  return opt


def _annotate(frame, angle_deg: float, frictions, colours, released: dict,
              width: int) -> np.ndarray:
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img, "RGBA")
  big, small = _font(40), _font(23)

  draw.rectangle([16, 14, 16 + 330, 14 + 62], fill=(0, 0, 0, 165))
  draw.text((30, 22), f"ramp angle  {angle_deg:5.1f}\u00b0", font=big,
            fill=(255, 255, 255))

  rows = len(frictions)
  box_h = 30 * rows + 52
  draw.rectangle([16, 92, 16 + 430, 92 + box_h], fill=(0, 0, 0, 165))
  draw.text((30, 100), "  mu    atan(mu)   released", font=small,
            fill=(210, 214, 220))
  for i, mu in enumerate(frictions):
    y = 130 + 30 * i
    draw.rectangle([30, y + 5, 48, y + 21], fill=colours[i])
    rel = released.get(i)
    rel_s = f"{rel:5.1f}\u00b0" if rel is not None else "  --"
    hit = rel is not None
    draw.text(
      (58, y), f"{mu:5.2f}   {math.degrees(math.atan(mu)):5.1f}\u00b0    {rel_s}",
      font=small, fill=(120, 240, 140) if hit else (235, 235, 235),
    )
  return np.asarray(img)


@dataclass(frozen=True)
class SlopeConfig:
  frictions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)
  cube_priority: int = 1
  """0 restores element-wise-max mixing, where the ramp's 1.0 masks every cube."""
  mode: str = "tilt"
  """``tilt`` slowly raises the ramp and shows each cube breaking away at its own
  atan(mu) -- the clearest single check that the configured friction is the
  friction the solver applies. ``fixed`` holds one angle and races the cubes."""
  ramp_angle_deg: float = 20.0
  """Angle for ``mode=fixed``. 20 deg sits between atan(0.3)=16.7 and
  atan(0.5)=26.6, so the sweep visibly splits into sliders and stickers."""
  tilt_max_deg: float = 55.0
  tilt_rate_deg_s: float = 4.0
  """Slow enough that the release angle is quasi-static, i.e. a real reading."""
  seconds: float = 3.0
  measure_critical_angles: bool = True
  out: str = "eval/friction/slope.mp4"
  width: int = 1280
  height: int = 720
  fps: int = 60
  json_out: str | None = None


def _run_tilt(cfg: "SlopeConfig") -> dict:
  """Raise the ramp from flat and record the angle each cube breaks away at."""
  spec = _build(cfg.frictions, 0.0, cfg.cube_priority)
  model = spec.compile()
  model.vis.global_.offwidth = cfg.width
  model.vis.global_.offheight = cfg.height
  colours = _tint(model, len(cfg.frictions))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  bodies = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"c{i}_cube")
    for i in range(len(cfg.frictions))
  ]

  renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
  opt = _collision_only_option()
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)
  cam.lookat[:] = [-0.15, 0.0, _RAMP_Z - 0.02]
  cam.distance = 1.9
  cam.elevation = -8.0
  cam.azimuth = 104.0

  dt = model.opt.timestep
  # Settle flat first so the release angles are not contaminated by the initial
  # 0.1 mm drop onto the ramp.
  for _ in range(int(0.4 / dt)):
    mujoco.mj_step(model, data)
  # Displacement has to be measured in the RAMP's frame. As the ramp tilts, a
  # cube that is perfectly stuck to it still sweeps through the world frame --
  # at x = 0.35 m and 6 deg/s that is ~37 mm/s of pure rotation, which would
  # trip any world-frame threshold long before the cube actually slips.
  ramp_origin = np.array([0.0, 0.0, _RAMP_Z])
  origin_local = np.array(
    [data.xpos[b].copy() - ramp_origin for b in bodies]
  )
  prev_local = origin_local.copy()
  slide_speed = np.zeros(len(bodies))

  total_s = cfg.tilt_max_deg / cfg.tilt_rate_deg_s
  n_steps = int(total_s / dt)
  every = max(1, int(round(1.0 / (cfg.fps * dt))))
  released: dict[int, float] = {}
  frames = []

  for k in range(n_steps):
    angle_deg = min(cfg.tilt_max_deg, cfg.tilt_rate_deg_s * (k * dt))
    a = math.radians(angle_deg)
    data.mocap_quat[0] = [math.cos(a / 2), 0.0, math.sin(a / 2), 0.0]
    mujoco.mj_step(model, data)

    # Release is detected on SPEED along the ramp, not displacement. A stuck
    # cube still creeps a fraction of a mm per second as the contact reseats,
    # and that creep integrates: a fixed displacement threshold therefore trips
    # earlier the slower you tilt, which is exactly backwards. Measured here,
    # mu=0.6 read 27.2 deg at 4 deg/s and 21.9 deg at 2 deg/s against a true
    # 31.0 deg. Creep stays under ~1 mm/s; a cube that has actually broken away
    # passes 30 mm/s almost immediately.
    c_a, s_a = math.cos(a), math.sin(a)
    r_inv = np.array([[c_a, 0.0, -s_a], [0.0, 1.0, 0.0], [s_a, 0.0, c_a]])
    pos_local = (np.array([data.xpos[b] for b in bodies]) - ramp_origin) @ r_inv.T
    inst = np.linalg.norm(pos_local - prev_local, axis=1) / dt
    slide_speed += 0.05 * (inst - slide_speed)  # EMA, ~40 ms window
    prev_local = pos_local
    moved = np.linalg.norm(pos_local - origin_local, axis=1)
    for i in range(len(bodies)):
      if i not in released and slide_speed[i] > _RELEASE_SPEED and moved[i] > 2e-3:
        released[i] = angle_deg

    if k % every == 0:
      renderer.update_scene(data, camera=cam, scene_option=opt)
      frames.append(
        _annotate(renderer.render(), angle_deg, cfg.frictions, colours,
                  released, cfg.width)
      )
    if len(released) == len(bodies):
      # Hold the final frame briefly so the completed table is readable.
      for _ in range(int(1.2 * cfg.fps)):
        frames.append(frames[-1])
      break

  out = Path(cfg.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(out, frames, fps=cfg.fps, quality=8, macro_block_size=1)
  print(f"[INFO] Wrote {out} — {len(frames)} frames, "
        f"{len(frames) / cfg.fps:.1f}s, {cfg.width}x{cfg.height}")

  print(f"\n[INFO] Release angles (cube_priority={cfg.cube_priority}, "
        f"ramp friction {_RAMP_FRICTION}):")
  print(f"{'mu':>6} {'atan(mu)':>10} {'released':>10} {'delta':>8}")
  print("-" * 38)
  rows = []
  for i, mu in enumerate(cfg.frictions):
    pred = math.degrees(math.atan(mu))
    rel = released.get(i)
    rows.append({"mu": mu, "predicted_deg": pred, "released_deg": rel})
    rel_s = f"{rel:9.1f}°" if rel is not None else "     never"
    dl = f"{rel - pred:+7.1f}°" if rel is not None else "       --"
    print(f"{mu:>6.2f} {pred:>9.1f}° {rel_s} {dl}")
  return {"release_angles": rows}


def main() -> None:
  cfg = tyro.cli(SlopeConfig)

  if cfg.mode == "tilt":
    payload = _run_tilt(cfg)
    if cfg.json_out:
      payload["cube_priority"] = cfg.cube_priority
      payload["ramp_friction"] = _RAMP_FRICTION
      Path(cfg.json_out).parent.mkdir(parents=True, exist_ok=True)
      Path(cfg.json_out).write_text(json.dumps(payload, indent=2))
      print(f"[INFO] wrote {cfg.json_out}")
    return

  ang = math.radians(cfg.ramp_angle_deg)

  rows = []
  if cfg.measure_critical_angles:
    print(f"[INFO] Measuring critical angles (cube_priority={cfg.cube_priority}, "
          f"ramp friction {_RAMP_FRICTION})...")
    print(f"{'mu':>6} {'predicted':>11} {'measured':>10} {'delta':>8}")
    print("-" * 39)
    for mu in cfg.frictions:
      predicted = math.degrees(math.atan(mu))
      measured = _critical_angle(mu, cfg.cube_priority)
      rows.append({"mu": mu, "predicted_deg": predicted, "measured_deg": measured})
      print(f"{mu:>6.2f} {predicted:>10.2f}° {measured:>9.2f}° "
            f"{measured - predicted:>+7.2f}°")
    print()

  # Slide distances at the video's angle, so the table explains the footage.
  dist = _slide_distance(cfg.frictions, ang, cfg.cube_priority, cfg.seconds)
  print(f"[INFO] Slide distance after {cfg.seconds:g}s at {cfg.ramp_angle_deg:g}°:")
  for mu, d in zip(cfg.frictions, dist):
    verdict = "SLIDES" if d > 0.02 else "sticks"
    print(f"       mu={mu:<5.2f} {d * 100:7.2f} cm  {verdict}")

  # Render.
  spec = _build(cfg.frictions, ang, cfg.cube_priority)
  model = spec.compile()
  model.vis.global_.offwidth = cfg.width
  model.vis.global_.offheight = cfg.height
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)
  cam.lookat[:] = [-0.15, 0.0, _RAMP_Z - 0.02]
  cam.distance = 1.5
  cam.elevation = -12.0
  cam.azimuth = 118.0

  n_steps = int(cfg.seconds / model.opt.timestep)
  every = max(1, int(round(1.0 / (cfg.fps * model.opt.timestep))))
  frames = []
  for i in range(n_steps):
    mujoco.mj_step(model, data)
    if i % every == 0:
      renderer.update_scene(data, camera=cam)
      frames.append(renderer.render())

  out = Path(cfg.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(out, frames, fps=cfg.fps, quality=8, macro_block_size=1)
  print(f"[INFO] Wrote {out} — {len(frames)} frames, "
        f"{len(frames) / cfg.fps:.1f}s, {cfg.width}x{cfg.height}")
  print(f"[INFO] Cubes left-to-right: mu = "
        f"{', '.join(f'{m:g}' for m in cfg.frictions)}")

  if cfg.json_out:
    payload = {
      "cube_priority": cfg.cube_priority,
      "ramp_friction": _RAMP_FRICTION,
      "ramp_angle_deg": cfg.ramp_angle_deg,
      "critical_angles": rows,
      "slide_distance_m": {str(m): float(d) for m, d in zip(cfg.frictions, dist)},
    }
    Path(cfg.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.json_out).write_text(json.dumps(payload, indent=2))
    print(f"[INFO] wrote {cfg.json_out}")


if __name__ == "__main__":
  main()
