"""LEAP-XELA cube reorientation environment configuration.

Ports the MJX ``CubeReorient`` task (mujoco_playground leapXELA) to mjlab's
manager-based API. See:
https://mujocolab.github.io/mjlab/main/source/architecture_overview.html
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
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

from leap_xela_mjlab.robots.cube import get_cube_cfg, get_goal_cube_cfg
from leap_xela_mjlab.robots.leap_xela import JOINT_NAMES, get_leap_xela_cfg
from leap_xela_mjlab.tasks.reorient import mdp as reorient_mdp


def make_reorient_env_cfg(
  *,
  finger_tip_type: str = "Box",
  play: bool = False,
  enable_perturbations: bool = False,
) -> ManagerBasedRlEnvCfg:
  robot_cfg = SceneEntityCfg("robot", joint_names=(".*",))

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=reorient_mdp.joint_pos_abs,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      params={"asset_cfg": robot_cfg},
    ),
    "joint_pos_error": ObservationTermCfg(
      func=reorient_mdp.joint_pos_error_from_command,
      params={"action_name": "joint_pos", "asset_cfg": robot_cfg},
    ),
    "cube_pos_error": ObservationTermCfg(
      func=reorient_mdp.cube_pos_error_from_palm,
      noise=Unoise(n_min=-0.02, n_max=0.02),
      params={"object_name": "cube"},
    ),
    "cube_ori_error": ObservationTermCfg(
      func=reorient_mdp.cube_ori_error_mat,
      noise=Unoise(n_min=-0.1, n_max=0.1),
      params={"command_name": "goal_orientation", "object_name": "cube"},
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms = {
    "joint_pos": ObservationTermCfg(
      func=reorient_mdp.joint_pos_abs,
      params={"asset_cfg": robot_cfg},
    ),
    "joint_pos_error": ObservationTermCfg(
      func=reorient_mdp.joint_pos_error_from_command,
      params={"action_name": "joint_pos", "asset_cfg": robot_cfg},
    ),
    "cube_pos_error": ObservationTermCfg(
      func=reorient_mdp.cube_pos_error_from_palm,
      params={"object_name": "cube"},
    ),
    "cube_ori_error": ObservationTermCfg(
      func=reorient_mdp.cube_ori_error_mat,
      params={"command_name": "goal_orientation", "object_name": "cube"},
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
    "joint_vel": ObservationTermCfg(
      func=reorient_mdp.joint_vel_abs,
      params={"asset_cfg": robot_cfg},
    ),
    "fingertip_pos": ObservationTermCfg(
      func=reorient_mdp.fingertip_positions_rel_palm,
    ),
    "cube_lin_vel": ObservationTermCfg(
      func=reorient_mdp.cube_lin_vel,
      params={"object_name": "cube"},
    ),
    "cube_ang_vel": ObservationTermCfg(
      func=reorient_mdp.cube_ang_vel,
      params={"object_name": "cube"},
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

  commands: dict[str, CommandTermCfg] = {
    "goal_orientation": reorient_mdp.InHandReorientationCommandCfg(
      asset_name="cube",
      goal_asset_name="goal",
      orientation_success_threshold=0.1,
      update_goal_on_success=True,
      use_mjx_goal_drift=True,
      debug_vis=True,
      debug_vis_axes=False,
    ),
  }

  events = {
    # Fixed-base mocap placement for vectorized envs.
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
    # Domain randomization (MJX domain_randomize port).
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

  if enable_perturbations:
    events["cube_velocity_pert"] = EventTermCfg(
      func=reorient_mdp.apply_cube_velocity_perturbation,
      mode="interval",
      interval_range_s=(0.05, 0.05),
      params={
        "asset_cfg": SceneEntityCfg("cube"),
        "linear_velocity_range": (0.0, 3.0),
        "angular_velocity_range": (0.0, 0.5),
        "probability": 0.02,
      },
    )

  # MJX reward_config.scales (+ success bonus as its own term).
  rewards = {
    "orientation": RewardTermCfg(
      func=reorient_mdp.cube_orientation_tolerance,
      weight=5.0,
      params={"command_name": "goal_orientation", "object_name": "cube"},
    ),
    "position": RewardTermCfg(
      func=reorient_mdp.cube_position_tolerance,
      weight=0.5,
      params={"object_name": "cube"},
    ),
    "termination": RewardTermCfg(
      func=reorient_mdp.termination_penalty,
      weight=-100.0,
    ),
    "hand_pose": RewardTermCfg(
      func=reorient_mdp.hand_pose_l2_from_default,
      weight=-0.5,
      params={"asset_cfg": robot_cfg},
    ),
    "action_rate": RewardTermCfg(
      func=reorient_mdp.action_rate_l2,
      weight=-0.001,
    ),
    "joint_vel": RewardTermCfg(
      func=reorient_mdp.joint_vel_l2,
      weight=0.0,
      params={"asset_cfg": robot_cfg},
    ),
    "energy": RewardTermCfg(
      func=reorient_mdp.energy_l1,
      weight=-1e-3,
      params={"asset_cfg": robot_cfg},
    ),
    "success": RewardTermCfg(
      func=reorient_mdp.success_bonus,
      weight=100.0,
      params={
        "command_name": "goal_orientation",
        "object_name": "cube",
        "success_threshold": 0.1,
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "cube_fell": TerminationTermCfg(
      func=reorient_mdp.cube_fell_below,
      # MJX used z < -0.05 with floor at -0.25; with floor at 0 this is 0.20.
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
          half_size=0.0385,
          mass=0.108,
          friction=0.3,
        ),
        "goal": get_goal_cube_cfg(),
      },
      num_envs=1,
      env_spacing=0.6,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="palm",
      distance=0.55,
      elevation=-20.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=48,
      njmax=None,
      mujoco=MujocoCfg(
        timestep=0.01,
        iterations=5,
        ls_iterations=8,
        integrator="euler",
      ),
    ),
    # ctrl_dt=0.05, sim_dt=0.01  -> decimation=5
    decimation=5,
    # episode_length=1000 steps at 20 Hz
    episode_length_s=50.0,
    scale_rewards_by_dt=True,
  )

  if play:
    cfg.episode_length_s = 1e9
    cfg.observations["actor"].enable_corruption = False
    # Keep resets, drop DR / pert for cleaner playback.
    for key in list(cfg.events):
      if key.startswith("dr_") or key == "cube_velocity_pert":
        del cfg.events[key]

  return cfg


def leap_xela_cube_reorient_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_reorient_env_cfg(finger_tip_type="Box", play=play)
