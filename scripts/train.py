"""Train LeapXELA / mjlab tasks with RSL-RL (single- or multi-GPU)."""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from load_env import load_project_dotenv

load_project_dotenv()

import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags


@dataclass
class TrainConfig:
  num_envs: int | None = None
  max_iterations: int | None = None
  seed: int = 42
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  torchrunx_log_dir: str | None = None
  # Cube physics (optional overrides for hyperparameter search).
  # Global multiplier on the actor's observation noise -- playground's own
  # obs_noise.level, which this port hard-coded. 1.0 is exact parity. The
  # cube_ori term is +/-0.1 on rotation-matrix entries, ~4.7 deg of orientation
  # noise against a 5.7 deg success threshold, so the policy cannot perceive
  # whether it is inside the threshold it is being scored on.
  obs_noise_scale: float | None = None
  cube_half_size: float | None = None
  cube_friction_sliding: float | None = None
  cube_friction_torsional: float | None = None
  # Contact dimensionality of the cube geom. 3 (default) makes friction[1] and
  # friction[2] inert; 4 enables torsional friction, 6 also enables rolling.
  cube_condim: int | None = None
  # Contact priority of the cube geom (default 1 in the env cfg). The cube
  # outranks every hand geom at >=1, so its condim and friction are the ones
  # MuJoCo uses. Pass 0 to restore the pre-2026-09 element-wise-max mixing, in
  # which the fingertips' 0.5 beat the cube's 0.3 and --cube-friction-sliding
  # did nothing at the grasp. See robots/cube.get_cube_spec.
  cube_priority: int | None = None
  # Hand-base orientation, xyz Euler radians, e.g. --palm-euler "(0, 1.92, -1.57)".
  # Previously only reachable by regenerating the MJCF; 1.88 vs 1.92 is the
  # difference between Hamid's four flat runs and his one that took off.
  palm_euler: tuple[float, float, float] | None = None
  # Override the dr_fingertip_friction sampling range (sliding friction only).
  # NOTE: at cube_priority >= 1 this no longer affects cube-fingertip contacts
  # at all -- the cube dictates their friction -- so it now only moves
  # fingertip-vs-terrain contacts. Use --cube-friction-sliding for the grasp.
  # At cube_priority 0 the old behaviour returns: friction is the elementwise
  # max, the fingertips (0.5-1.0) dominate the cube (0.1-0.5), and this is the
  # only knob that changes how well the hand grips.
  fingertip_friction: tuple[float, float] | None = None
  # Exploration overrides. rsl_rl's GaussianDistribution is UNBOUNDED, so its
  # entropy is `const + log(std)` and the entropy bonus pushes log(std) up by a
  # constant forever -- there is no finite optimum, unlike the tanh-squashed
  # policy brax/playground uses. Runs 12-19 converge at std 2.79, which at
  # action scale 0.5 is a commanded joint delta of 1.40 rad per 50 ms step
  # against ~1-2 rad joint ranges: ~47% of sampled actions saturate
  # clip_to_ctrl_limits. See TRAINING_NOTES.md, "the exploration noise floor".
  entropy_coef: float | None = None
  init_std: float | None = None
  # Goal-update / terminal-shaping overrides (reorient only). None = preset
  # default. `--goal-drift False --goal-resample-on-success False` pins the goal
  # for the whole episode, which turns the success bonus into a dense dwell
  # reward; `--orientation-fine True` adds the long-tail term that is the only
  # reward gradient below 0.2 rad. See TRAINING_NOTES.md runs 22-23.
  goal_drift: bool | None = None
  goal_resample_on_success: bool | None = None
  orientation_fine: bool | None = None
  # L2 penalty on raw action MAGNITUDE, e.g. --action-l2 0.001. None = the
  # preset default (0.0, i.e. off, which is what runs 1-26 had). Only the action
  # *rate* has ever been penalized, and since the action term clips to the ctrl
  # range there is no gradient above the clip pulling the magnitude back: run
  # 14's deterministic mean action sits at |a| = 13.4, a commanded joint delta
  # 3x the full range, so it is permanently saturated. See mdp.rewards.action_l2
  # and TRAINING_NOTES.md.
  action_l2: float | None = None
  # Logging / checkpointing.
  logger: Literal["wandb", "tensorboard"] | None = None
  save_interval: int | None = None
  # Path to a checkpoint (.pt) to resume from. Training continues in that
  # checkpoint's own run directory, and max_iterations is read as the *total*
  # iteration target rather than a count of additional iterations.
  resume_from: str | None = None
  # WandB / run labeling.
  run_name: str | None = None
  wandb_project: str | None = None
  wandb_tags: tuple[str, ...] = ()
  upload_model: bool | None = None


def _apply_env_overrides(
  env_cfg, cfg: TrainConfig, task_id: str
) -> object | None:
  """Rebuild env config when cube or goal/reward hyperparameters are set.

  The factory has to be chosen from ``task_id``: rebuilding a rotate_* task with
  ``make_reorient_env_cfg`` would silently train a *different task* than the one
  named on the command line.
  """
  if (
    cfg.obs_noise_scale is None
    and cfg.cube_half_size is None
    and cfg.cube_friction_sliding is None
    and cfg.cube_friction_torsional is None
    and cfg.cube_condim is None
    and cfg.cube_priority is None
    and cfg.palm_euler is None
    and cfg.goal_drift is None
    and cfg.goal_resample_on_success is None
    and cfg.orientation_fine is None
    and cfg.action_l2 is None
  ):
    return

  if "Rotate" in task_id:
    from leap_xela_mjlab.tasks.rotate_z.config.env_cfg import (
      make_rotate_z_env_cfg as make_env_cfg,
    )
  else:
    from leap_xela_mjlab.tasks.reorient.config.env_cfg import (
      make_reorient_env_cfg as make_env_cfg,
    )

  # Rebuild from the arguments the task was registered with, so overriding the
  # cube does not silently revert preset / finger_tip_type to their defaults.
  kwargs = dict(getattr(env_cfg, "build_kwargs", {}))
  if not kwargs:
    raise SystemExit(
      f"{task_id} was registered without build_kwargs, so cube overrides would "
      "silently rebuild it with default arguments. Add build_kwargs to its "
      "env-cfg factory before using --cube-* flags."
    )
  if cfg.obs_noise_scale is not None:
    kwargs["obs_noise_scale"] = cfg.obs_noise_scale
    print(f"[INFO] obs_noise_scale -> {cfg.obs_noise_scale}")
  if cfg.cube_half_size is not None:
    kwargs["cube_half_size"] = cfg.cube_half_size
  if cfg.cube_friction_sliding is not None:
    kwargs["cube_friction_sliding"] = cfg.cube_friction_sliding
  if cfg.cube_friction_torsional is not None:
    kwargs["cube_friction_torsional"] = cfg.cube_friction_torsional
  if cfg.cube_condim is not None:
    kwargs["cube_condim"] = cfg.cube_condim
  if cfg.cube_priority is not None:
    kwargs["cube_priority"] = cfg.cube_priority
  if cfg.palm_euler is not None:
    kwargs["palm_euler"] = tuple(cfg.palm_euler)
  # reorient-only; the rotate_* factories do not take these.
  _params = inspect.signature(make_env_cfg).parameters
  for _name, _val in (
    ("goal_drift", cfg.goal_drift),
    ("goal_resample_on_success", cfg.goal_resample_on_success),
    ("orientation_fine", cfg.orientation_fine),
    ("action_l2_weight", cfg.action_l2),
  ):
    if _val is None:
      continue
    if _name not in _params:
      raise SystemExit(f"{task_id} does not support --{_name.replace('_', '-')}.")
    kwargs[_name] = _val
    print(f"[INFO] {_name} -> {_val}")
  # Only when a friction value is being pinned: dr_cube_friction resamples the
  # cube's sliding friction every reset and would overwrite it. condim is not
  # touched by that event, so a condim-only override must leave DR alone --
  # otherwise it silently changes a second variable against the baseline run.
  if (
    cfg.cube_friction_sliding is not None or cfg.cube_friction_torsional is not None
  ) and "disable_cube_friction_dr" in inspect.signature(make_env_cfg).parameters:
    kwargs["disable_cube_friction_dr"] = True
  overridden = make_env_cfg(**kwargs)
  overridden.seed = env_cfg.seed
  overridden.scene.num_envs = env_cfg.scene.num_envs
  overridden.episode_length_s = env_cfg.episode_length_s
  overridden.decimation = env_cfg.decimation
  overridden.scale_rewards_by_dt = env_cfg.scale_rewards_by_dt
  overridden.viewer = env_cfg.viewer
  return overridden


def _collect_run_metrics(rank: int) -> dict[str, Any]:
  """Read final WandB summary metrics after training (rank 0 only)."""
  if rank != 0:
    return {}
  try:
    import wandb
  except ImportError:
    return {}

  if wandb.run is None:
    return {}

  summary: dict[str, Any] = {}
  for key, value in wandb.run.summary.items():
    if not str(key).startswith("_"):
      summary[key] = value
  summary["wandb_run_id"] = wandb.run.id
  summary["wandb_run_name"] = wandb.run.name or ""
  summary["wandb_project"] = wandb.run.project
  return summary


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> dict[str, Any]:
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

  overridden = _apply_env_overrides(env_cfg, cfg, task_id)
  if overridden is not None:
    env_cfg = overridden

  if cfg.fingertip_friction is not None:
    term = env_cfg.events.get("dr_fingertip_friction")
    if term is None:
      raise SystemExit(f"{task_id} has no dr_fingertip_friction event to override.")
    term.params["friction_range"] = tuple(cfg.fingertip_friction)
    print(f"[INFO] fingertip friction range -> {term.params['friction_range']}")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.max_iterations is not None:
    agent_cfg.max_iterations = cfg.max_iterations
  agent_cfg.seed = seed
  env_cfg.seed = seed
  if cfg.logger is not None:
    agent_cfg.logger = cfg.logger
  if cfg.save_interval is not None:
    agent_cfg.save_interval = cfg.save_interval
  if cfg.run_name is not None:
    agent_cfg.run_name = cfg.run_name
  if cfg.wandb_project is not None:
    agent_cfg.wandb_project = cfg.wandb_project
  if cfg.wandb_tags:
    agent_cfg.wandb_tags = cfg.wandb_tags
  if cfg.upload_model is not None:
    agent_cfg.upload_model = cfg.upload_model
  if cfg.entropy_coef is not None:
    agent_cfg.algorithm.entropy_coef = cfg.entropy_coef
    print(f"[INFO] entropy_coef -> {cfg.entropy_coef}")
  if cfg.init_std is not None:
    agent_cfg.actor.distribution_cfg["init_std"] = cfg.init_std
    print(f"[INFO] init_std -> {cfg.init_std}")

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")
  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

  agent_dict = asdict(agent_cfg)
  # mjlab's runner strips None optional fields (cnn_cfg, etc.) before rsl_rl builds MLPModel.
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner

  if rank == 0:
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_dict)

  runner = runner_cls(env, agent_dict, str(log_dir), device=device)
  if cfg.resume_from is not None:
    print(f"[INFO] Resuming from checkpoint: {cfg.resume_from}")
    runner.load(cfg.resume_from)
    print(f"[INFO] Resumed at iteration {runner.current_learning_iteration}")
  add_wandb_tags(agent_cfg.wandb_tags)
  if rank == 0 and (
    cfg.obs_noise_scale is not None
    or cfg.cube_half_size is not None
    or cfg.cube_friction_sliding is not None
    or cfg.cube_friction_torsional is not None
    or cfg.cube_condim is not None
    or cfg.cube_priority is not None
    or cfg.palm_euler is not None
    or cfg.goal_drift is not None
    or cfg.goal_resample_on_success is not None
    or cfg.orientation_fine is not None
    or cfg.action_l2 is not None
  ):
    try:
      import wandb

      if wandb.run is not None:
        wandb.config.update(
          {
            "obs_noise_scale": cfg.obs_noise_scale,
            "cube_half_size": cfg.cube_half_size,
            "cube_friction_sliding": cfg.cube_friction_sliding,
            "cube_friction_torsional": cfg.cube_friction_torsional,
            "cube_condim": cfg.cube_condim,
            "cube_priority": cfg.cube_priority,
            "palm_euler": cfg.palm_euler,
            "goal_drift": cfg.goal_drift,
            "goal_resample_on_success": cfg.goal_resample_on_success,
            "orientation_fine": cfg.orientation_fine,
            "action_l2": cfg.action_l2,
          },
          allow_val_change=True,
        )
    except ImportError:
      pass
  # rsl_rl's learn() runs num_learning_iterations *beyond* the iteration the
  # checkpoint was saved at, so subtract what has already been done.
  start_iteration = getattr(runner, "current_learning_iteration", 0)
  remaining = agent_cfg.max_iterations - start_iteration
  if remaining <= 0:
    raise SystemExit(
      f"Nothing to do: checkpoint is at iteration {start_iteration}, "
      f"max_iterations is {agent_cfg.max_iterations}."
    )
  runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
  metrics = _collect_run_metrics(rank)
  env.close()
  return metrics


def launch_training(task_id: str, args: TrainConfig) -> tuple[Path, dict[str, Any]]:
  agent_cfg = load_rl_cfg(task_id)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  if args.resume_from is not None:
    # Keep the resumed run in one directory so its curve stays contiguous.
    log_dir = Path(args.resume_from).resolve().parent
  else:
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
      log_dir_name += f"_{args.run_name}"
    log_dir = (
      Path("logs")
      / "rsl_rl"
      / agent_cfg.experiment_name
      / log_dir_name
    )

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

  if num_gpus <= 1:
    metrics = run_train(task_id, args, log_dir)
    return log_dir, metrics

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
  return log_dir, {}


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
