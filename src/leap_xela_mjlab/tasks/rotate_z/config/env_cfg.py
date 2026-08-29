"""LEAP-XELA z-axis cube rotation environment configuration.

Ports the MJX ``LeapCubeRotateZAxis`` task. Hamid's guide starts here rather
than with reorient -- "rotate_z must be easier so I am going to start with that
one" -- and we skipped it. It is also the most direct test of the open question
from TRAINING_NOTES.md: every converged reorient run holds the cube still
(``cube_ang_speed`` ~1 rad/s) instead of turning it, and here angular velocity
about z *is* the reward, with no goal to farm.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from leap_xela_mjlab.robots.cube import get_cube_cfg
from leap_xela_mjlab.robots.leap_xela import JOINT_NAMES, get_leap_xela_cfg
from leap_xela_mjlab.tasks.reorient import mdp as reorient_mdp
from leap_xela_mjlab.tasks.rotate_z import mdp as rotate_mdp


def make_rotate_z_env_cfg(
  *,
  finger_tip_type: str = "Box_palm192",
  play: bool = False,
  cube_half_size: float = 0.035,
  cube_mass: float = 0.108,
  cube_friction_sliding: float = 0.3,
  cube_friction_torsional: float = 0.05,
) -> ManagerBasedRlEnvCfg:
  robot_cfg = SceneEntityCfg("robot", joint_names=(".*",))

  # MJX state: noisy joint angles (16) + last action (16).
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=reorient_mdp.joint_pos_abs,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      params={"asset_cfg": robot_cfg},
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms = {
    "joint_pos": ObservationTermCfg(
      func=reorient_mdp.joint_pos_abs, params={"asset_cfg": robot_cfg}
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
    "joint_vel": ObservationTermCfg(
      func=reorient_mdp.joint_vel_abs, params={"asset_cfg": robot_cfg}
    ),
    "joint_torque": ObservationTermCfg(
      func=rotate_mdp.joint_torque_abs, params={"asset_cfg": robot_cfg}
    ),
    "fingertip_pos": ObservationTermCfg(func=reorient_mdp.fingertip_positions_rel_palm),
    "cube_pos_error": ObservationTermCfg(
      func=reorient_mdp.cube_pos_error_from_palm, params={"object_name": "cube"}
    ),
    "cube_quat": ObservationTermCfg(
      func=rotate_mdp.cube_quat, params={"object_name": "cube"}
    ),
    "cube_ang_vel": ObservationTermCfg(
      func=reorient_mdp.cube_ang_vel, params={"object_name": "cube"}
    ),
    "cube_lin_vel": ObservationTermCfg(
      func=reorient_mdp.cube_lin_vel, params={"object_name": "cube"}
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": reorient_mdp.DeltaJointPositionActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      scale=0.5,
      offset=0.0,
      use_default_offset=True,
      clip_to_ctrl_limits=True,
      ema_alpha=1.0,
    )
  }

  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_cube_pose": EventTermCfg(
      func=reorient_mdp.reset_cube_pose_uniform_quat,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube"),
        "pose_range": {
          "x": (-0.01, 0.01),
          "y": (-0.01, 0.01),
          "z": (-0.01, 0.01),
        },
      },
    ),
    "dr_cube_friction": EventTermCfg(
      func=reorient_mdp.randomize_geom_friction,
      mode="reset",
      params={
        "friction_range": (0.1, 0.5),
        "asset_cfg": SceneEntityCfg("cube", geom_names=("cube",)),
        "axes": (0,),
      },
    ),
    "dr_fingertip_friction": EventTermCfg(
      func=reorient_mdp.randomize_geom_friction,
      mode="reset",
      params={
        "friction_range": (0.5, 1.0),
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=("th_tip", "if_tip", "mf_tip", "rf_tip")
        ),
        "axes": (0,),
      },
    ),
    "dr_cube_mass": EventTermCfg(
      func=reorient_mdp.randomize_body_mass,
      mode="reset",
      params={
        "mass_range": (0.8, 1.2),
        "operation": "scale",
        "asset_cfg": SceneEntityCfg("cube", body_names=("cube",)),
      },
    ),
    "dr_robot_mass": EventTermCfg(
      func=reorient_mdp.randomize_body_mass,
      mode="reset",
      params={
        "mass_range": (0.9, 1.1),
        "operation": "scale",
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      },
    ),
  }

  # MJX reward_config.scales for LeapCubeRotateZAxis: angvel 1.0, termination
  # -100.0, everything else 0.0. The zero-weight terms are kept so their values
  # still show up in the logs.
  rewards = {
    "angvel": RewardTermCfg(
      func=rotate_mdp.cube_ang_vel_axis,
      weight=1.0,
      params={"object_name": "cube", "axis": (0.0, 0.0, 1.0)},
    ),
    "termination": RewardTermCfg(func=reorient_mdp.termination_penalty, weight=-100.0),
    "linvel": RewardTermCfg(
      func=reorient_mdp.cube_position_tolerance,
      weight=0.0,
      params={"object_name": "cube"},
    ),
    "hand_pose": RewardTermCfg(
      func=reorient_mdp.hand_pose_l2_from_default,
      weight=0.0,
      params={"asset_cfg": robot_cfg},
    ),
    "action_rate": RewardTermCfg(func=reorient_mdp.action_rate_l2, weight=0.0),
    "energy": RewardTermCfg(
      func=reorient_mdp.energy_l1, weight=0.0, params={"asset_cfg": robot_cfg}
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "cube_fell": TerminationTermCfg(
      func=reorient_mdp.cube_fell_below,
      params={"object_name": "cube", "minimum_height": 0.20},
    ),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": get_leap_xela_cfg(finger_tip_type=finger_tip_type),
        "cube": get_cube_cfg(
          half_size=cube_half_size,
          mass=cube_mass,
          friction_sliding=cube_friction_sliding,
          friction_torsional=cube_friction_torsional,
        ),
      },
      num_envs=1,
      env_spacing=0.6,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="palm",
      distance=0.55,
      elevation=-20.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      # Hamid had to raise njmax from 128 to 170 for this task in playground.
      # A CPU rollout of our model peaks at nefc 108 / ncon 24, so the reorient
      # task's njmax=120 has almost no headroom and overflow is silent.
      nconmax=64,
      njmax=200,
      mujoco=MujocoCfg(
        timestep=0.01,
        iterations=5,
        ls_iterations=8,
        integrator="euler",
        # The generated MJCF carries <flag eulerdamp="disable"/>, but mjlab
        # attaches the hand spec into its own scene and the parent's option
        # block wins, so the flag was silently dropped and we were running with
        # MuJoCo's default implicit damping while Hamid runs explicit. Not
        # cosmetic: in a scripted grasp the two differ by 10x in peak joint
        # velocity (21.9 vs 2.0 rad/s). Restore it here -- MujocoCfg.apply()
        # only ORs disableflags in, so this is the only way to get it back.
        disableflags=("eulerdamp",),
      ),
    ),
    decimation=5,
    episode_length_s=50.0,
    scale_rewards_by_dt=True,
  )

  if play:
    cfg.episode_length_s = 1e9
    cfg.observations["actor"].enable_corruption = False
    for key in list(cfg.events):
      if key.startswith("dr_"):
        del cfg.events[key]

  return cfg


def leap_xela_cube_rotate_z_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_rotate_z_env_cfg(play=play)
