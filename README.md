# LEAP-XELA → mjlab

Port of the MJX `LeapXELACubeReorient` environment (`leapXELA/reorient.py`) to
[mjlab](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)'s
manager-based API (MuJoCo Warp + RSL-RL).

## Clone (with submodule)

The MuJoCo hand model lives in the `leapXELA_model` submodule under
`src/leap_xela_mjlab/assets/leapXELA_model`.

```bash
# Clone repo and submodule in one step
git clone --recurse-submodules git@github.com:mohammad200h/leapXelaMjLab.git
cd leapXelaMjLab
```

If you already cloned without submodules:

```bash
cd leapXelaMjLab
git submodule update --init --recursive
```

Confirm the model is present:

```bash
ls src/leap_xela_mjlab/assets/leapXELA_model/leapXela_generated_mjx_Box.xml
```

## Setup

Requires Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## List tasks

```bash
uv run python scripts/list_envs.py leap
```

Registered task ID: `Mjlab-LeapXELA-Cube-Reorient`.

## Play

```bash
# Uniform random actions in [-1, 1] (default).
# Opens the native MuJoCo viewer if DISPLAY/WAYLAND is set, otherwise Viser.
uv run python scripts/play.py Mjlab-LeapXELA-Cube-Reorient --agent random

# Hold zero actions (home pose targets)
uv run python scripts/play.py Mjlab-LeapXELA-Cube-Reorient --agent zero --viewer viser

# Play the latest local checkpoint under logs/rsl_rl/<experiment>/
uv run python scripts/play.py Mjlab-LeapXELA-Cube-Reorient --agent trained

# Or point at a specific checkpoint
uv run python scripts/play.py Mjlab-LeapXELA-Cube-Reorient --agent trained \
  --checkpoint-file logs/rsl_rl/<experiment>/<run>/model_XXXX.pt
```

Useful flags: `--num-envs N`, `--device cuda:0|cpu`, `--viewer auto|native|viser`.

## Train

```bash
# Single GPU (default)
uv run python scripts/train.py Mjlab-LeapXELA-Cube-Reorient --num-envs 4096

# Two GPUs (data-parallel via torchrunx)
uv run python scripts/train.py Mjlab-LeapXELA-Cube-Reorient --num-envs 4096 --gpu-ids "[0, 1]"

# All available GPUs
uv run python scripts/train.py Mjlab-LeapXELA-Cube-Reorient --num-envs 4096 --gpu-ids all

# CPU
uv run python scripts/train.py Mjlab-LeapXELA-Cube-Reorient --gpu-ids None
```

Multi-GPU notes:

- Each GPU runs the full `--num-envs` count independently; experience per iteration scales with GPU count.
- GPU indices are relative to `CUDA_VISIBLE_DEVICES` if set, e.g. `CUDA_VISIBLE_DEVICES=2,3 ... --gpu-ids "[0, 1]"` uses physical GPUs 2 and 3.
- `max-iterations` is not scaled automatically. For similar wall-clock length, reduce it by the GPU count (e.g. halve when going from 1 → 2 GPUs).
- See [mjlab distributed training](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html).

Optional flags:

| Flag | Description |
| --- | --- |
| `--num-envs N` | Parallel envs **per GPU** |
| `--max-iterations N` | PPO learning iterations |
| `--seed N` | RNG seed (default `42`; each rank uses `seed + rank`) |
| `--gpu-ids "[0]"` | GPUs to use (`"[0, 1]"`, `all`, or `None` for CPU) |

Checkpoints and configs are written under:

```text
logs/rsl_rl/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/
```

## Layout

```text
leapXelaMjLab/
  src/leap_xela_mjlab/
    robots/                      # EntityCfg for hand + cube
    assets/leapXELA_model/       # git submodule (MuJoCo XML + meshes)
    tasks/reorient/
      mdp/                       # observations, rewards, commands, actions, events
      config/env_cfg.py          # ManagerBasedRlEnvCfg
      config/rl_cfg.py           # RSL-RL PPO config
  scripts/train.py
  scripts/play.py
  scripts/list_envs.py
```

## Mapping from MJX

| MJX (`reorient.py`) | mjlab |
| --- | --- |
| `LeapHandEnv` + `MjSpec` scene | `EntityCfg` hand + cube in `SceneCfg` |
| `reset` / goal quat / DR | `EventManager` + `InHandReorientationCommand` |
| `step` action EMA deltas | `DeltaJointPositionAction` |
| reward / obs / done helpers | manager terms under `tasks/reorient/mdp/` |
| Brax / RSL training | `RslRlOnPolicyRunnerCfg` |

Control frequency matches the MJX defaults: `sim_dt=0.01`, `decimation=5` → 20 Hz,
`episode_length_s=50`.

## Notes

- Goal orientation uses continuous MJX-style drift (`use_mjx_goal_drift=True`). Set
  it to `False` in `InHandReorientationCommandCfg` for hard resample-on-success.
- Actor observations include joint positions, tracking error, cube pose/orientation
  error, and last action (asymmetric critic adds velocities / fingertips / object twist).
- Domain randomization covers cube/fingertip friction and body masses.
