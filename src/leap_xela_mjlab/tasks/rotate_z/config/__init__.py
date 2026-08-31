from mjlab.tasks.registry import register_mjlab_task

from leap_xela_mjlab.tasks.rotate_z.config.env_cfg import (
  leap_xela_cube_rotate_env_cfg,
)
from leap_xela_mjlab.tasks.rotate_z.config.rl_cfg import (
  leap_xela_cube_rotate_z_ppo_cfg,
)

# Same task, same reward, only the world axis the angular-velocity reward
# projects onto differs. z is MJX's LeapCubeRotateZAxis (the twist the hand is
# known to manage, 2.20 rad/s in run 15); x and y are the two tip-over motions
# the reorient policy stalls on.
for _axis in ("x", "y", "z"):
  register_mjlab_task(
    task_id=f"Mjlab-LeapXELA-Cube-Rotate{_axis.upper()}",
    env_cfg=leap_xela_cube_rotate_env_cfg(_axis),
    play_env_cfg=leap_xela_cube_rotate_env_cfg(_axis, play=True),
    rl_cfg=leap_xela_cube_rotate_z_ppo_cfg(_axis),
  )
