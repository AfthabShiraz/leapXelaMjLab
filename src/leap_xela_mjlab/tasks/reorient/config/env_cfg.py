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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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


_DEFAULT_CUBE_HALF_SIZE = 0.0385
_DEFAULT_CUBE_MASS = 0.108


def _cube_mass_for_half_size(half_size: float) -> float:
  scale = half_size / _DEFAULT_CUBE_HALF_SIZE
  return _DEFAULT_CUBE_MASS * scale**3


def make_reorient_env_cfg(
  *,
  finger_tip_type: str = "Box",
  play: bool = False,
  preset: str = "curriculum",
  enable_perturbations: bool = False,
  cube_half_size: float = _DEFAULT_CUBE_HALF_SIZE,
  cube_mass: float | None = None,
  cube_friction_sliding: float = 0.3,
  cube_friction_torsional: float = 0.05,
  disable_cube_friction_dr: bool = False,
) -> ManagerBasedRlEnvCfg:
  # "baseline" reproduces the pre-curriculum config used by run #1
  # (`baseline-no-touch-10k`) in TRAINING_NOTES.md, so those runs can be repeated
  # against the corrected joint limits. "curriculum" is the current best config.
  if preset not in ("baseline", "curriculum"):
    raise ValueError(f"Unknown preset {preset!r}; expected 'baseline' or 'curriculum'.")
  baseline = preset == "baseline"

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
      orientation_success_threshold=0.1 if baseline else 0.4,
      update_goal_on_success=True,
      use_mjx_goal_drift=baseline,
      goal_relative_to_object=not baseline,
      # Absolute goals over the full +-pi span, matching the original
      # `rand * torch.pi` sampling, since span = pi * difficulty.
      initial_difficulty=1.0 if baseline else 0.1,
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

  if disable_cube_friction_dr:
    events.pop("dr_cube_friction", None)

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
      # Lowered from 5.0: at the parked error of ~1.5 rad this paid 2.6/step for
      # 930 steps, so holding the cube still outearned any attempt to reorient.
      weight=5.0 if baseline else 1.0,
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
        "success_threshold": 0.1 if baseline else 0.4,
      },
    ),
  }

  if not baseline:
    rewards["orientation_fine"] = RewardTermCfg(
      func=reorient_mdp.cube_orientation_fine,
      weight=5.0,
      params={
        "command_name": "goal_orientation",
        "object_name": "cube",
        "margin": 0.4,
      },
    )

  curriculum = {
    "goal_difficulty": CurriculumTermCfg(
      func=reorient_mdp.goal_difficulty,
      params={
        "command_name": "goal_orientation",
        "promote_at": 1.0,
        "demote_at": 0.2,
        # Slow: compute() fires ~24x per training iteration, so update_every=500
        # is ~20 iterations per adjustment. At step 0.02 a full 0.1 -> 1.0 ramp
        # takes ~900 iterations. Faster than this and difficulty outruns
        # competence, which oscillates instead of ramping.
        "step": 0.02,
        "min_difficulty": 0.05,
        "max_difficulty": 1.0,
        "update_every": 500,
        "ema_alpha": 0.01,
      },
    ),
  }

  if baseline:
    curriculum = {}

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "cube_fell": TerminationTermCfg(
      func=reorient_mdp.cube_fell_below,
      # MJX used z < -0.05 with floor at -0.25; with floor at 0 this is 0.20.
      params={"object_name": "cube", "minimum_height": 0.20},
    ),
    "nan": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  if cube_mass is None:
    cube_mass = _cube_mass_for_half_size(cube_half_size)

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
        "goal": get_goal_cube_cfg(half_size=cube_half_size),
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
    curriculum=curriculum,
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
      # Hand+cube contacts exceed mjwarp's default heuristic (~64 nefc); seen overflows at ~80.
      njmax=120,
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

  # train.py's cube overrides rebuild the cfg; keep the arguments around so they
  # can do that without silently reverting preset / finger_tip_type to defaults.
  cfg.build_kwargs = {
    "finger_tip_type": finger_tip_type,
    "preset": preset,
    "enable_perturbations": enable_perturbations,
    "cube_half_size": cube_half_size,
    "cube_mass": cube_mass,
    "cube_friction_sliding": cube_friction_sliding,
    "cube_friction_torsional": cube_friction_torsional,
    "disable_cube_friction_dr": disable_cube_friction_dr,
  }

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


def leap_xela_cube_reorient_baseline_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_reorient_env_cfg(finger_tip_type="Box", play=play, preset="baseline")


def leap_xela_cube_reorient_reference_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The one LeapXELA configuration Hamid found that learns to reorient.

  From the "Training LeapXELA" Notion page: at palm euler 1.88 every cube scale
  he tried (1.0, 1.05, 1.08, 1.09, 1.1, 1.2) plateaus at reward ~150, as does
  palm 1.92 with scale 1.1. Palm 1.92 with the cube at its *original* size took
  off instead (141 -> 284 and still climbing at 200M steps). Our generated model
  is baked at palm 1.88 and the env defaults to half-size 0.0385 (scale 1.1),
  i.e. exactly the combination he has four flat runs on.

  Palm angle lives in the MJCF, so this variant loads a separately generated
  model (``--palm-euler 0 1.92 -1.57``). Cube mass is pinned to 0.108 because his
  XML keeps it constant across scales, whereas _cube_mass_for_half_size scales it
  with volume and would give 0.081 here.
  """
  return make_reorient_env_cfg(
    finger_tip_type="Box_palm192",
    play=play,
    preset="baseline",
    cube_half_size=0.035,
    cube_mass=0.108,
  )
