"""Render a LeapXELA / mjlab policy rollout to an MP4.

``play.py`` only opens an interactive viewer, which is useless on a headless box.
This drives the same env with ``render_mode="rgb_array"`` and writes frames to a
video file instead.

Offscreen rendering needs an EGL context. Device 0 is the NVIDIA one on this
machine; the Mesa device fails on /dev/dri permissions, so pin it unless the
caller has already chosen.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class RenderConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  out: str = "rollout.mp4"
  num_steps: int = 600
  """Control steps to record. The env steps at 1/step_dt Hz (20 Hz here)."""
  width: int = 960
  height: int = 720
  fps: float | None = None
  """Defaults to 1/step_dt, i.e. real-time playback."""
  camera_distance: float | None = None
  camera_elevation: float | None = None
  camera_azimuth: float | None = None
  device: str | None = None
  seed: int | None = None
  view: Literal["visual", "collision", "both"] = "visual"
  """Which geom groups to draw.

  The hand MJCF splits every link into a ``visual`` mesh (group 2, contype 0)
  and one or more ``collision`` boxes (group 3), and the cube does the same: a
  textured mesh in group 2 and the actual box in group 3. What the physics sees
  is group 3 ONLY -- the group-2 meshes are decorative and collide with nothing.

  ``collision`` hides group 2 and shows group 3, which is the view that answers
  "is the cube getting stuck between the gaps": the pads are discrete boxes with
  real spaces between them, and a cube wedged into one is invisible in the
  default render because the visual mesh covers it.
  """
  show_contacts: bool = False
  """Overlay contact points and contact force arrows (MuJoCo's own viz)."""
  transparent: bool = False
  """Draw everything see-through, so cube-inside-hand geometry is visible."""
  cube_priority: int | None = None
  """Rebuild the env with this cube contact priority before replaying.

  This matters when replaying an OLD checkpoint. Every run up to and including
  run 21 was trained with all geoms at priority 0, where a cube-fingertip
  contact resolved to the fingertip's friction (0.5) rather than the cube's
  (0.3). The task now defaults to priority 1, so replaying such a checkpoint
  through the stock config quietly gives it a more slippery grasp than it ever
  trained against. Pass ``--cube-priority 0`` for a faithful replay, or 1 to see
  what the change does to a policy that never saw it.
  """
  cube_friction_sliding: float | None = None
  """Rebuild the env with this cube sliding friction. Only bites at priority 1."""
  palm_euler: tuple[float, float, float] | None = None
  """Hand-base orientation, xyz Euler radians, e.g. ``--palm-euler 0 1.92 -1.57``.

  ``None`` keeps the angle baked into the model file (Box = 1.88,
  Box_palm192 = 1.92). Passing the reference value explicitly is a no-op that
  makes the angle visible in the log and the video's provenance rather than
  hidden inside a ``finger_tip_type`` string.
  """


# Geom groups, as the LEAP-XELA MJCF assigns them.
_GROUP_VISUAL = 2
_GROUP_COLLISION = 3

# Group 0 is mjlab's terrain plane -- kept in every view so the scene still has
# a floor and the lighting has something to fall on. Everything in the hand and
# cube models carries an explicit group (2 visual / 3 collision), so including 0
# adds the ground and nothing else.
_GROUP_TERRAIN = 0

_VIEW_GROUPS: dict[str, tuple[int, ...]] = {
  "visual": (_GROUP_TERRAIN, _GROUP_VISUAL),
  "collision": (_GROUP_TERRAIN, _GROUP_COLLISION),
  "both": (_GROUP_TERRAIN, _GROUP_VISUAL, _GROUP_COLLISION),
}


def _scene_options(env) -> list:
  """Every MjvOption the offscreen renderer actually draws through.

  mjlab keeps two: ``Renderer._scene_option`` is what ``update_scene`` uses for
  the tracked env (it is not passed explicitly, so the Renderer's own default
  applies), and ``OffscreenRenderer._opt`` is used by ``mjv_addGeoms`` for the
  neighbouring context envs. Both have to be set or the two halves of the frame
  disagree.
  """
  opts = []
  renderer = getattr(env, "_offline_renderer", None)
  if renderer is None:
    return opts
  own = getattr(renderer, "_opt", None)
  if own is not None:
    opts.append(own)
  try:
    inner = renderer.renderer
  except ValueError:
    inner = None  # not initialized yet; caller renders a frame first
  if inner is not None and getattr(inner, "_scene_option", None) is not None:
    opts.append(inner._scene_option)
  return opts


def _apply_scene_options(env, cfg: "RenderConfig") -> None:
  if cfg.view == "visual" and not cfg.show_contacts and not cfg.transparent:
    return  # stock rendering, nothing to override

  opts = _scene_options(env)
  if not opts:
    # The Renderer is built lazily on the first render; force it, then retry.
    env.render()
    opts = _scene_options(env)
  if not opts:
    raise SystemExit(
      "Could not reach the renderer's MjvOption to set geom groups. "
      "mjlab's OffscreenRenderer internals may have changed."
    )

  groups = _VIEW_GROUPS[cfg.view]
  for opt in opts:
    opt.geomgroup[:] = 0
    for g in groups:
      opt.geomgroup[g] = 1
    if cfg.show_contacts:
      opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
      opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    if cfg.transparent:
      opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

  print(
    f"[INFO] view={cfg.view} (geom groups {list(groups)})"
    f"{' +contacts' if cfg.show_contacts else ''}"
    f"{' +transparent' if cfg.transparent else ''}"
  )


def _apply_cube_overrides(env_cfg, cfg: "RenderConfig", task_id: str):
  """Rebuild the env cfg when a cube contact override is requested.

  Mirrors ``train.py``'s ``_apply_env_overrides``: rebuild from the arguments the
  task was registered with, so overriding one knob does not silently revert
  preset / finger_tip_type to their defaults.
  """
  if (
    cfg.cube_priority is None
    and cfg.cube_friction_sliding is None
    and cfg.palm_euler is None
  ):
    return env_cfg

  if "Rotate" in task_id:
    from leap_xela_mjlab.tasks.rotate_z.config.env_cfg import (
      make_rotate_z_env_cfg as make_env_cfg,
    )
  else:
    from leap_xela_mjlab.tasks.reorient.config.env_cfg import (
      make_reorient_env_cfg as make_env_cfg,
    )

  kwargs = dict(getattr(env_cfg, "build_kwargs", {}))
  if not kwargs:
    raise SystemExit(
      f"{task_id} was registered without build_kwargs, so a cube override would "
      "rebuild it with default arguments."
    )
  if cfg.cube_priority is not None:
    kwargs["cube_priority"] = cfg.cube_priority
  if cfg.cube_friction_sliding is not None:
    kwargs["cube_friction_sliding"] = cfg.cube_friction_sliding
  if cfg.palm_euler is not None:
    kwargs["palm_euler"] = tuple(cfg.palm_euler)
  rebuilt = make_env_cfg(play=True, **{
    k: v for k, v in kwargs.items() if k != "play"
  })
  print(
    f"[INFO] cube override: priority={kwargs.get('cube_priority')} "
    f"friction_sliding={kwargs.get('cube_friction_sliding')} "
    f"palm_euler={kwargs.get('palm_euler')}"
  )
  return rebuilt


def run_render(task_id: str, cfg: RenderConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg = _apply_cube_overrides(env_cfg, cfg, task_id)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1
  if cfg.seed is not None:
    env_cfg.seed = cfg.seed

  # The task ships a sensible camera (tracks the palm); only override what was asked
  # for, and raise the resolution off the 320x240 training default.
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  env_cfg.viewer.max_extra_envs = 0
  if cfg.camera_distance is not None:
    env_cfg.viewer.distance = cfg.camera_distance
  if cfg.camera_elevation is not None:
    env_cfg.viewer.elevation = cfg.camera_elevation
  if cfg.camera_azimuth is not None:
    env_cfg.viewer.azimuth = cfg.camera_azimuth

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  step_dt = env.step_dt
  wrapped = RslRlVecEnvWrapper(
    env, clip_actions=getattr(agent_cfg, "clip_actions", None)
  )

  if cfg.agent in {"zero", "random"}:
    action_shape: tuple[int, ...] = env.action_space.shape  # type: ignore
    if cfg.agent == "zero":
      def policy(obs):
        del obs
        return torch.zeros(action_shape, device=device)
    else:
      def policy(obs):
        del obs
        return 2 * torch.rand(action_shape, device=device) - 1
  else:
    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
    elif cfg.wandb_run_path is not None:
      resume_path, _ = get_wandb_checkpoint_path(log_root, Path(cfg.wandb_run_path))
    else:
      resume_path = get_checkpoint_path(log_root)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    # Same construction as train.py: MjlabOnPolicyRunner strips the None optional
    # fields (cnn_cfg, rnn_type, ...) that plain rsl_rl chokes on.
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    with tempfile.TemporaryDirectory() as tmp:
      runner = runner_cls(wrapped, asdict(agent_cfg), tmp, device=device)
      runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

  _apply_scene_options(env, cfg)

  fps = cfg.fps if cfg.fps is not None else 1.0 / step_dt
  out_path = Path(cfg.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  obs, _ = wrapped.reset()
  frames: list[np.ndarray] = []
  resets = 0
  print(f"[INFO] Recording {cfg.num_steps} steps at {fps:g} fps -> {out_path}")
  with torch.inference_mode():
    for i in range(cfg.num_steps):
      actions = policy(obs)
      obs, _, dones, _ = wrapped.step(actions)
      resets += int(dones.sum().item())
      frame = env.render()
      if frame is not None:
        frames.append(np.asarray(frame))
      if (i + 1) % 100 == 0:
        print(f"       {i + 1}/{cfg.num_steps} frames, {resets} resets")

  env.close()

  if not frames:
    raise SystemExit("No frames were rendered.")
  imageio.mimwrite(out_path, frames, fps=fps, quality=8, macro_block_size=1)
  secs = len(frames) / fps
  print(
    f"[INFO] Wrote {out_path} — {len(frames)} frames, {secs:.1f}s, "
    f"{frames[0].shape[1]}x{frames[0].shape[0]}, {resets} episode resets"
  )


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import leap_xela_mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )
  args = tyro.cli(
    RenderConfig,
    args=remaining_args,
    default=RenderConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
  )
  run_render(chosen_task, args)


if __name__ == "__main__":
  main()
