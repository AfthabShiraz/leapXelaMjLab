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
| `--cube-half-size F` | Cube collision half-extent (m); disables cube friction DR |
| `--cube-friction-sliding F` | Cube sliding friction (axis 0) |
| `--cube-friction-torsional F` | Cube torsional friction (axis 1) |
| `--run-name NAME` | WandB run name and log subdirectory label |
| `--wandb-project NAME` | WandB project (default from RL config: `mjlab`) |

Checkpoints and configs are written under:

```text
logs/rsl_rl/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/
```

Training metrics are logged to [WandB](https://wandb.ai) by default (`logger=wandb` in
`rl_cfg.py`). Set `WANDB_API_KEY` in `.env` at the project root (loaded automatically by
`train.py` and `run_sweep.py`), or log in with `wandb login`.

## Hyperparameter search

Grid search over cube size and friction (plus seeds), with each run trained via
`scripts/train.py` and logged to WandB.

```bash
# Run the sweep defined in hyperparameter_search/config.yaml
uv run python hyperparameter_search/run_sweep.py

# Custom config path
uv run python hyperparameter_search/run_sweep.py --config path/to/config.yaml
```

Edit `hyperparameter_search/config.yaml` to define the grid. List values are expanded
in a Cartesian product; scalars are fixed for every run:

```yaml
task_id: Mjlab-LeapXELA-Cube-Reorient
num_envs: 4096
max_iterations: 10000          # screening budget; full training default is 100_000
wandb_project: leap_xela_hparam_search
upload_model: false            # metrics only (no checkpoint uploads to WandB)
gpu_ids: [0]

cube_half_size: 0.035          # base half-extent (m)
cube_scale: [1.0]               # effective half-size = cube_half_size × scale
cube_friction:
  sliding: [0.049, 0.3]         # swept sliding friction
  torsional: 0.05               # fixed torsional friction
seed: [1, 2, 3]
best_metric: Train/mean_reward   # metric maximized to pick the winner
```

After all runs finish, results are written to `hyperparameter_search/best_run.yaml`
(same directory as the config file). It contains the winning hyperparameters,
`best_metric_value`, log/WandB run ids, and a summary row per grid point.

Each grid point:

- Sets cube size/friction at build time (visual mesh and collision stay matched).
- Turns off cube friction domain randomization so the configured friction is fixed.
- Scales cube mass with volume when size changes.
- Names the WandB run like `hs0.0350_mu0.049_tz0.050_sc1.00_s1`.

**Choosing `max_iterations`:** there is no universal cutoff. Use short runs to screen
obviously bad settings, then train top candidates to the full `100_000` iterations
from `rl_cfg.py`. Compare learning curves in WandB; if reward is still rising at the
sweep budget, rankings may change if you train longer. Use multiple seeds (as above)
to reduce noise.

Single trial without the sweep script:

```bash
uv run python scripts/train.py Mjlab-LeapXELA-Cube-Reorient \
  --num-envs 4096 \
  --cube-half-size 0.035 \
  --cube-friction-sliding 0.3 \
  --cube-friction-torsional 0.05 \
  --run-name my_trial \
  --wandb-project leap_xela_hparam_search
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
  hyperparameter_search/
    config.yaml                # sweep grid + training defaults
    run_sweep.py               # grid runner → train.py + WandB
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
