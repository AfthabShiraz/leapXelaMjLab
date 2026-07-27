"""Train LeapXELA / mjlab tasks with RSL-RL (single- or multi-GPU)."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


@dataclass
class TrainConfig:
  num_envs: int | None = None
  max_iterations: int | None = None
  seed: int = 42
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  torchrunx_log_dir: str | None = None


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.seed + rank

  configure_torch_backends()

  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.max_iterations is not None:
    agent_cfg.max_iterations = cfg.max_iterations
  agent_cfg.seed = seed
  env_cfg.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")
  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

  if rank == 0:
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner = OnPolicyRunner(env, asdict(agent_cfg), str(log_dir), device=device)
  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )
  env.close()


def launch_training(task_id: str, args: TrainConfig) -> None:
  agent_cfg = load_rl_cfg(task_id)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  log_dir = (
    Path("logs")
    / "rsl_rl"
    / agent_cfg.experiment_name
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

  if num_gpus <= 1:
    run_train(task_id, args, log_dir)
    return

  import torchrunx

  logging.basicConfig(level=logging.INFO)
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    if args.torchrunx_log_dir is not None:
      os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
    else:
      os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

  print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,  # Let rsl_rl handle process group initialization.
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_train, task_id, args, log_dir)


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
  launch_training(chosen_task, args)


if __name__ == "__main__":
  # Usage:
  #   uv run python scripts/train.py Mjlab-LeapXELA-Cube-Reorient --num-envs 4096
  #   uv run python scripts/train.py Mjlab-LeapXELA-Cube-Reorient --gpu-ids "[0, 1]"
  main()
