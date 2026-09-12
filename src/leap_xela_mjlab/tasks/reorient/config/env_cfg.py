"""LEAP-XELA cube reorientation environment configuration.

Ports the MJX ``CubeReorient`` task (mujoco_playground leapXELA) to mjlab's
manager-based API. See:
https://mujocolab.github.io/mjlab/main/source/architecture_overview.html
"""

from __future__ import annotations

import math

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

# ctrl_dt = sim_dt * decimation, matching playground's ctrl_dt=0.05 / sim_dt=0.01.
_SIM_TIMESTEP = 0.01
_DECIMATION = 5

# ``orientation_kernel="inverse"``: 1/(err + eps), weighted so its marginal
# reward d/d(err) = w/(err+eps)^2 equals the linear term's w_lin/pi at 130 deg.
# The approach phase keeps the pull it was learned under and only the near-goal
# shape changes: x1.79 the linear weight, i.e. 8.93 at the reference weight 5.
_INVERSE_EPS = 0.1
_INVERSE_WEIGHT_PER_LINEAR = (math.radians(130.0) + _INVERSE_EPS) ** 2 / math.pi


def _cube_mass_for_half_size(half_size: float) -> float:
  scale = half_size / _DEFAULT_CUBE_HALF_SIZE
  return _DEFAULT_CUBE_MASS * scale**3


def _unoise(magnitude: float) -> Unoise | None:
  """Symmetric uniform noise, or None when scaled to zero.

  ``UniformNoiseCfg`` rejects n_min == n_max, so ``obs_noise_scale=0`` has to
  drop the term's noise rather than pass a zero-width range.
  """
  return Unoise(n_min=-magnitude, n_max=magnitude) if magnitude > 0.0 else None


def make_reorient_env_cfg(
  *,
  finger_tip_type: str = "Box",
  play: bool = False,
  preset: str = "curriculum",
  enable_perturbations: bool = False,
  # Global multiplier on the actor's observation noise. This is playground's own
  # ``obs_noise.level`` (default_config: level=1.0 over scales joint_pos 0.05 /
  # cube_pos 0.02 / cube_ori 0.1), which this port hard-coded away rather than
  # exposed -- so 1.0 is exact parity and this flag only makes the knob
  # reachable.
  #
  # Why it is worth turning. ``cube_ori`` noise is +/-0.1 on rotation-matrix
  # entries: per component std 0.058, which tilts a unit column by ~0.082 rad
  # ~= 4.7 deg. The success threshold is 0.1 rad = 5.7 deg. The measurement
  # noise is the same order as the target the policy is being asked to hit, so
  # it cannot perceive whether it is inside the threshold and cannot learn to
  # servo there. Note the eval already runs with corruption OFF (``play`` sets
  # ``enable_corruption = False`` below) and the policy still stops at 26-27
  # deg, which is the signature of a policy trained blind rather than one being
  # blinded at test time.
  obs_noise_scale: float = 1.0,
  cube_half_size: float = _DEFAULT_CUBE_HALF_SIZE,
  cube_mass: float | None = None,
  cube_friction_sliding: float = 0.3,
  cube_friction_torsional: float = 0.05,
  disable_cube_friction_dr: bool = False,
  cube_condim: int = 3,
  # Contact-parameter priority for the cube geom. At the default 1 the cube
  # outranks every hand geom (all priority 0), so MuJoCo takes condim/friction
  # from the cube instead of element-wise-maxing the pair. Without this,
  # ``cube_friction_sliding`` is silently discarded at the four fingertips
  # (friction 0.5 > the cube's 0.3) -- i.e. exactly where the grasp is. Set 0
  # to reproduce runs 1-21. See ``robots/cube.get_cube_spec`` for the details.
  cube_priority: int = 1,
  # Hand-base orientation as xyz Euler, formerly reachable only by regenerating
  # the MJCF and registering a new ``finger_tip_type``. ``None`` keeps the angle
  # baked into the chosen model file (Box = 1.88, Box_palm192 = 1.92).
  palm_euler: tuple[float, float, float] | None = None,
  # Goal-update and terminal-shaping overrides. ``None`` means "whatever the
  # preset says"; these exist because the two knobs have never been varied
  # independently -- every drift-off run in TRAINING_NOTES.md is also a
  # fine-term + curriculum run, so the Aug-22 line cannot separate them.
  goal_drift: bool | None = None,
  goal_resample_on_success: bool | None = None,
  orientation_fine: bool | None = None,
  actor_cube_ang_vel: bool = False,
  angvel_align_weight: float = 0.0,
  leap_joint_limits: bool = False,
  # L2 penalty on raw action MAGNITUDE. 0.0 keeps the historical behaviour
  # (runs 1-26), where only the action *rate* was penalized and the policy's
  # mean action ran away to ~13 -- 3x the full joint range once scaled, i.e.
  # permanently clipped. See ``mdp.rewards.action_l2``.
  action_l2_weight: float = 0.0,
  # Orientation-error threshold (rad) for success, applied to BOTH the goal
  # command (goal drift / resample trigger, consecutive_success) and the
  # success_bonus reward -- they must move together or the goal machinery and
  # the reward disagree about what a success is. ``None`` keeps the preset's
  # value (baseline 0.1, curriculum 0.4). Exists so the threshold can be varied
  # without the ``preset`` switch, which also moves the orientation weight
  # 5.0 -> 1.0. See research/NEXT_EXPERIMENTS.md, Experiment 1.
  success_threshold: float | None = None,
  # Shape of the dense orientation reward. "linear" is playground's
  # tolerance(err, (0, 0.2), margin=pi): constant marginal reward from 180 deg to
  # 11.5 deg and flat below, so nothing pays for the last 11.5 deg. "inverse" is
  # 1/(err + 0.1), whose marginal reward keeps rising to zero error. Under the
  # goal drift a success kicks the goal ~160 deg away, and with the inverse
  # kernel that kick costs ~190 discounted reward against a +5 bonus -- pair it
  # with a pinned goal (goal_drift=False, goal_resample_on_success=False) or the
  # policy is trained never to cross the threshold.
  orientation_kernel: str = "linear",
) -> ManagerBasedRlEnvCfg:
  # "baseline" reproduces the pre-curriculum config used by run #1
  # (`baseline-no-touch-10k`) in TRAINING_NOTES.md, so those runs can be repeated
  # against the corrected joint limits. "curriculum" is the current best config.
  if preset not in ("baseline", "curriculum"):
    raise ValueError(f"Unknown preset {preset!r}; expected 'baseline' or 'curriculum'.")
  baseline = preset == "baseline"
  if orientation_kernel not in ("linear", "inverse"):
    raise ValueError(
      f"Unknown orientation_kernel {orientation_kernel!r}; expected 'linear' or 'inverse'."
    )
  if success_threshold is None:
    success_threshold = 0.1 if baseline else 0.4

  # With both goal-update paths off the goal is fixed for the whole episode, so
  # the success bonus becomes a dense dwell reward instead of a one-shot spike
  # that throws the goal ~2.6 rad away (see InHandReorientationCommand's drift
  # kick). ``consecutive_success`` then reads as steps-inside-threshold rather
  # than a near-binary hit count.
  use_drift = baseline if goal_drift is None else goal_drift
  resample_on_success = (
    True if goal_resample_on_success is None else goal_resample_on_success
  )
  # ``cube_orientation_tolerance`` has bounds (0, 0.2), so it is exactly flat
  # below 0.2 rad: nothing in the baseline preset pays for the last 11 degrees.
  # The fine term is the only thing in this file with gradient there.
  use_fine = (not baseline) if orientation_fine is None else orientation_fine

  robot_cfg = SceneEntityCfg("robot", joint_names=(".*",))

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=reorient_mdp.joint_pos_abs,
      noise=_unoise(0.05 * obs_noise_scale),
      params={"asset_cfg": robot_cfg},
    ),
    "joint_pos_error": ObservationTermCfg(
      func=reorient_mdp.joint_pos_error_from_command,
      params={"action_name": "joint_pos", "asset_cfg": robot_cfg},
    ),
    "cube_pos_error": ObservationTermCfg(
      func=reorient_mdp.cube_pos_error_from_palm,
      noise=_unoise(0.02 * obs_noise_scale),
      params={"object_name": "cube"},
    ),
    "cube_ori_error": ObservationTermCfg(
      func=reorient_mdp.cube_ori_error_mat,
      noise=_unoise(0.1 * obs_noise_scale),
      params={"command_name": "goal_orientation", "object_name": "cube"},
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }

  # CAPABILITY PROBE (2026-09-12). The actor is velocity-blind and
  # ``history_length=1``, so it cannot finite-difference either -- it is a pure
  # vector field on SO(3) with no damping term, which limit-cycles and settles
  # only where a static cage exists. That is exactly the measured signature:
  # acquisition time is uncorrelated with the rotation demanded (r=+0.01 against
  # tip demanded), the cube travels ~7x the geodesic, 68% of acquisitions happen
  # above 0.5 rad/s, and failures orbit at ~1.0 rad/s while pinned successes slow
  # to 0.33 and hold. The 2026-09-06 parity audit dismissed velocity blindness
  # because playground's reference is blind too -- a parity argument, never a
  # measurement.
  #
  # Deliberately NOT noised, unlike the other actor terms: this asks whether
  # knowing omega helps AT ALL. True cube angular velocity is privileged
  # relative to the reference task, so treat a result here the way the pinned
  # eval is treated -- a capability measurement, not a task score. The
  # deployable form is an observation history (ObservationTermCfg exposes
  # history_length), which also has to average down the 0.1 rad cube_ori_error
  # noise: a 2-frame difference at that noise level gives 2 rad/s of velocity
  # noise against a 1.3 rad/s signal.
  if actor_cube_ang_vel:
    actor_terms["cube_ang_vel"] = ObservationTermCfg(
      func=reorient_mdp.cube_ang_vel,
      params={"object_name": "cube"},
    )

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
      orientation_success_threshold=success_threshold,
      update_goal_on_success=resample_on_success,
      use_mjx_goal_drift=use_drift,
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
    )
    if orientation_kernel == "linear"
    else RewardTermCfg(
      func=reorient_mdp.cube_orientation_inverse,
      weight=(5.0 if baseline else 1.0) * _INVERSE_WEIGHT_PER_LINEAR,
      params={
        "command_name": "goal_orientation",
        "object_name": "cube",
        "eps": _INVERSE_EPS,
      },
    ),
    # Goal-aligned angular velocity: rotate_z's reward with a goal-dependent
    # axis. 0.0 = off, which is the default and leaves every existing run
    # byte-identical. 1.0 matches rotate_z's own weight, the value known to
    # produce 2.19 rad/s of directed rotation on this hand.
    **(
      {
        "angvel_align": RewardTermCfg(
          func=reorient_mdp.cube_angvel_toward_goal,
          weight=angvel_align_weight,
          params={"command_name": "goal_orientation", "object_name": "cube"},
        )
      }
      if angvel_align_weight
      else {}
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
    "action_l2": RewardTermCfg(
      func=reorient_mdp.action_l2,
      weight=-abs(action_l2_weight),
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
      # There IS a real dt discrepancy against playground here: it adds the
      # success bonus outside its dt-scaled sum,
      #     reward  = sum(scales[k] * term[k]) * dt
      #     reward += success * success_reward          # 100.0, unscaled
      # while mjlab's RewardManager.compute() applies one global `scale = dt` to
      # every term, so a weight of 100 lands as 100 * 0.05 = 5 -- 20x weaker.
      #
      # `weight=100.0 / _STEP_DT` (i.e. 2000) cancels that, and run 20 tested it
      # against run 14 as a clean single-variable comparison. It made everything
      # worse: mean reward 150 -> 75, orientation_error 0.74 -> 1.10, and the
      # policy's action std diverged 2.06 -> 5.95 over 3000 iterations where run
      # 14 held flat at 2.79. The success rate did not move (consecutive_success
      # 0.03-0.05 in both). A ~10^-5-per-step event worth 100 reward is a spike
      # the critic cannot fit; the advantage estimate goes to noise and the
      # entropy bonus then owns the std. See TRAINING_NOTES.md run 20.
      #
      # So: left at 100, matching every run 12-19, and knowingly 20x below
      # playground. The discrepancy is real but it is not what is limiting this
      # task -- fixing it is strictly harmful until the exploration noise floor
      # is dealt with.
      weight=100.0,
      params={
        "command_name": "goal_orientation",
        "object_name": "cube",
        "success_threshold": success_threshold,
      },
    ),
  }

  if use_fine:
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
        "robot": get_leap_xela_cfg(
          finger_tip_type=finger_tip_type,
          palm_euler=palm_euler,
          leap_joint_limits=leap_joint_limits
        ),
        "cube": get_cube_cfg(
          half_size=cube_half_size,
          mass=cube_mass,
          friction_sliding=cube_friction_sliding,
          friction_torsional=cube_friction_torsional,
          condim=cube_condim,
          priority=cube_priority,
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
      nconmax=64,
      # Every run logged before 2026-08-29 hit "nefc overflow - please increase
      # njmax" against njmax=120, peaking at a request of 168 (curriculum-smoke).
      # Overflow silently drops constraint rows -- contacts and joint limits --
      # in exactly the high-contact states where the hand is gripping hard, so
      # every run so far trained against occasionally-dropped contacts. Hamid
      # raised this to 220 for the same task in playground; match that.
      #
      # condim 6 costs 6 constraint rows per contact instead of 3, so the same
      # grasp needs roughly double the buffer (the rotate_x condim-6 run
      # overflowed njmax=200 within 5 iterations, peak request 234). Kept at
      # exactly 220 for condim 3 so runs 12-14 stay comparable; njmax has no
      # effect on the dynamics unless it overflows.
      njmax=220 if cube_condim <= 3 else 500,
      mujoco=MujocoCfg(
        timestep=_SIM_TIMESTEP,
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
    # ctrl_dt=0.05, sim_dt=0.01  -> decimation=5
    decimation=_DECIMATION,
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
    "obs_noise_scale": obs_noise_scale,
    "cube_half_size": cube_half_size,
    "cube_mass": cube_mass,
    "cube_friction_sliding": cube_friction_sliding,
    "cube_friction_torsional": cube_friction_torsional,
    "disable_cube_friction_dr": disable_cube_friction_dr,
    "cube_condim": cube_condim,
    "cube_priority": cube_priority,
    "palm_euler": palm_euler,
    "goal_drift": goal_drift,
    "goal_resample_on_success": goal_resample_on_success,
    "orientation_fine": orientation_fine,
    "actor_cube_ang_vel": actor_cube_ang_vel,
    "angvel_align_weight": angvel_align_weight,
    "leap_joint_limits": leap_joint_limits,
    "action_l2_weight": action_l2_weight,
    "success_threshold": success_threshold,
    "orientation_kernel": orientation_kernel,
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
