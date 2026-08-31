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


def run_render(task_id: str, cfg: RenderConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
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
