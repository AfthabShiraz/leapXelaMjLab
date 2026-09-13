from mjlab.tasks.registry import register_mjlab_task

from leap_xela_mjlab.tasks.reorient.config.env_cfg import (
  leap_plain_cube_reorient_env_cfg,
  leap_xela_cube_reorient_baseline_env_cfg,
  leap_xela_cube_reorient_env_cfg,
  leap_xela_cube_reorient_reference_env_cfg,
)
from leap_xela_mjlab.tasks.reorient.config.rl_cfg import leap_xela_cube_reorient_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-LeapXELA-Cube-Reorient",
  env_cfg=leap_xela_cube_reorient_env_cfg(),
  play_env_cfg=leap_xela_cube_reorient_env_cfg(play=True),
  rl_cfg=leap_xela_cube_reorient_ppo_cfg(),
)

# Run #1 from TRAINING_NOTES.md, kept registered so the pre-curriculum experiments
# can be repeated against the corrected joint limits.
register_mjlab_task(
  task_id="Mjlab-LeapXELA-Cube-Reorient-Baseline",
  env_cfg=leap_xela_cube_reorient_baseline_env_cfg(),
  play_env_cfg=leap_xela_cube_reorient_baseline_env_cfg(play=True),
  rl_cfg=leap_xela_cube_reorient_ppo_cfg(preset="baseline"),
)

# Palm 1.92 + cube scale 1.0 -- the only LeapXELA configuration on the Notion page
# that learned to reorient. Same RL settings as the baseline task, so the only
# difference from `-Baseline` is the palm angle and the cube size.
register_mjlab_task(
  task_id="Mjlab-LeapXELA-Cube-Reorient-Reference",
  env_cfg=leap_xela_cube_reorient_reference_env_cfg(),
  play_env_cfg=leap_xela_cube_reorient_reference_env_cfg(play=True),
  rl_cfg=leap_xela_cube_reorient_ppo_cfg(preset="reference"),
)

# The bare LEAP hand -- mujoco_playground's model, no tactile pads. This is the
# control for every "the difference is the hand" claim in TRAINING_NOTES.md,
# none of which has ever been tested directly. Same reference preset, same cube,
# same goal machinery; only the hand differs.
_control_rl_cfg = leap_xela_cube_reorient_ppo_cfg(preset="reference")
# Own log directory: the reference preset names itself
# `leap_xela_cube_reorient_reference`, and letting the control write there would
# interleave bare-hand checkpoints with the XELA main line under one tree.
_control_rl_cfg.experiment_name = "leap_plain_cube_reorient_control"

register_mjlab_task(
  task_id="Mjlab-LEAP-Cube-Reorient-Control",
  env_cfg=leap_plain_cube_reorient_env_cfg(),
  play_env_cfg=leap_plain_cube_reorient_env_cfg(play=True),
  rl_cfg=_control_rl_cfg,
)
