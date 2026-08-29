from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def leap_xela_cube_reorient_ppo_cfg(preset: str = "curriculum") -> RslRlOnPolicyRunnerCfg:
  """RSL-RL config aligned with the MJX LeapXELACubeReorient brax/rsl settings.

  ``preset="baseline"`` restores the exploration settings used by run #1 in
  TRAINING_NOTES.md (init_std 1.0, entropy_coef 0.01). ``preset="reference"``
  uses those same settings under its own experiment name, so the palm-1.92 /
  cube-1.0 runs log separately from the palm-1.88 baseline they are compared to.
  """
  if preset not in ("baseline", "curriculum", "reference"):
    raise ValueError(
      f"Unknown preset {preset!r}; expected 'baseline', 'curriculum' or 'reference'."
    )
  baseline = preset in ("baseline", "reference")
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0 if baseline else 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01 if baseline else 0.002,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name={
      "baseline": "leap_xela_cube_reorient_baseline",
      "reference": "leap_xela_cube_reorient_reference",
      "curriculum": "leap_xela_cube_reorient",
    }[preset],
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=100_000,
  )
