"""Train LeapXELA / mjlab tasks with RSL-RL."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


@dataclass
class TrainConfig:
  num_envs: int | None = None
  max_iterations: int | None = None
  seed: int = 42
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import leap_xela_mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )
  args = tyro.cli(TrainConfig, args=remaining_args)

  configure_torch_backends()
  env_cfg = load_env_cfg(chosen_task)
  agent_cfg = load_rl_cfg(chosen_task)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
  if args.max_iterations is not None:
    agent_cfg.max_iterations = args.max_iterations
  agent_cfg.seed = args.seed

  if args.gpu_ids == [] or args.gpu_ids is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"
  else:
    if args.gpu_ids != "all":
      os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
    device = "cuda:0"

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

  log_dir = (
    Path("logs")
    / "rsl_rl"
    / agent_cfg.experiment_name
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  log_dir.mkdir(parents=True, exist_ok=True)
  dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner = OnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device=device)
  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )
  env.close()


if __name__ == "__main__":
  # Usage: uv run python scripts/train.py Mjlab-LeapXELA-Cube-Reorient --num-envs 4096
  main()
