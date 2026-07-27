# LEAP-XELA → mjlab

Port of the MJX `LeapXELACubeReorient` environment (`leapXELA/reorient.py`) to
[mjlab](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)'s
manager-based API (MuJoCo Warp + RSL-RL).

## Layout

```
leapXelaMjLab/
  src/leap_xela_mjlab/
    robots/leap_xela.py          # EntityCfg for hand + cube
    assets/leapXELA_model -> …   # symlink to sibling leapXELA model
    tasks/reorient/
      mdp/                       # observations, rewards, commands, actions, events
      config/env_cfg.py          # ManagerBasedRlEnvCfg
      config/rl_cfg.py           # RSL-RL PPO config
  scripts/train.py
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

## Setup

```bash
cd leapXelaMjLab
uv sync   # or: pip install -e .
```

Confirm the model symlink resolves:

```bash
ls src/leap_xela_mjlab/assets/leapXELA_model/leapXela_generated_mjx_Box.xml
```

## Sanity-check (random / zero policy)

After `uv sync` finishes:

```bash
# Uniform random actions in [-1, 1] (default). Opens native viewer if DISPLAY is set,
# otherwise Viser at http://localhost:8080.
uv run python scripts/play.py Mjlab-LeapXELA-Cube-Reorient --agent random

# Hold zero actions (home pose targets).
uv run python scripts/play.py Mjlab-LeapXELA-Cube-Reorient --agent zero --viewer viser
```

## Train

```bash
uv run python scripts/list_envs.py leap
uv run python scripts/train.py Mjlab-LeapXELA-Cube-Reorient --num-envs 4096
```

Or with the stock mjlab CLI after importing the task package:

```python
import leap_xela_mjlab.tasks  # registers Mjlab-LeapXELA-Cube-Reorient
```

## Notes

- Goal orientation uses continuous MJX-style drift (`use_mjx_goal_drift=True`). Set
  it to `False` in `InHandReorientationCommandCfg` for hard resample-on-success.
- Actor observations include joint positions, tracking error, cube pose/orientation
  error, and last action (asymmetric critic adds velocities / fingertips / object twist).
- Domain randomization covers cube/fingertip friction and body masses.
