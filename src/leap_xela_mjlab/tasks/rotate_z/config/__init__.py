from mjlab.tasks.registry import register_mjlab_task

from leap_xela_mjlab.tasks.rotate_z.config.env_cfg import (
  leap_xela_cube_rotate_z_env_cfg,
)
from leap_xela_mjlab.tasks.rotate_z.config.rl_cfg import (
  leap_xela_cube_rotate_z_ppo_cfg,
)

register_mjlab_task(
  task_id="Mjlab-LeapXELA-Cube-RotateZ",
  env_cfg=leap_xela_cube_rotate_z_env_cfg(),
  play_env_cfg=leap_xela_cube_rotate_z_env_cfg(play=True),
  rl_cfg=leap_xela_cube_rotate_z_ppo_cfg(),
)
