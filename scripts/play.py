"""Play LeapXELA / mjlab tasks with zero, random, or trained policies."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "random"
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  num_envs: int | None = 1
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"


def _prepare_agent_cfg(agent_cfg) -> dict:
  cfg_dict = asdict(agent_cfg)
  for model_key in ("actor", "critic"):
    model_cfg = cfg_dict.get(model_key)
    if isinstance(model_cfg, dict):
      if model_cfg.get("class_name", "MLPModel") != "CNNModel":
        model_cfg.pop("cnn_cfg", None)
  return cfg_dict


def run_play(task_id: str, cfg: PlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(
    env, clip_actions=getattr(agent_cfg, "clip_actions", None)
  )

  dummy = cfg.agent in {"zero", "random"}
  if dummy:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore

    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
    elif cfg.wandb_run_path is not None:
      resume_path, _ = get_wandb_checkpoint_path(log_root, Path(cfg.wandb_run_path))
    else:
      resume_path = get_checkpoint_path(log_root)

    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, _prepare_agent_cfg(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

  if cfg.viewer == "auto":
    import os

    resolved = (
      "native"
      if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      else "viser"
    )
  else:
    resolved = cfg.viewer

  if resolved == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()
  env.close()


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
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
