"""RSL-RL config for the z-axis rotation task.

Unlike the reorient task -- whose config mixes values from mujoco_playground's
*brax* config into an RSL-RL runner -- this uses playground's own
``rsl_rl_config()`` verbatim, since that is the config that applies when
``train_rsl_rl.py`` is the runner. It is constant across environments there.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def leap_xela_cube_rotate_z_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
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
      entropy_coef=0.01,
      num_learning_epochs=4,
      num_mini_batches=8,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.97,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="leap_xela_cube_rotate_z",
    save_interval=100,
    num_steps_per_env=40,
    max_iterations=100_000,
  )
