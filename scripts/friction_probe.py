"""Measure how much the cube SLIDES in the grasp, as a function of sliding friction.

Hamid's ask: "use the policy you trained, only change sliding friction to see if
the sliding goes away." This holds the policy fixed and sweeps
``cube_friction_sliding``, reporting the slip that actually happens at the
cube-hand contacts.

Why this needed a model change first
------------------------------------
MuJoCo mixes a dynamically generated contact's friction as the **element-wise
max** of the two geoms' -- unless one geom has higher ``priority``, in which case
that geom dictates it outright. Measured on this model, before any change:

    hand pads (32 geoms)  0.2      fingertips (4)  0.5
    terrain               1.0      cube            0.3   all priority 0

so a cube-fingertip contact resolved to max(0.3, 0.5) = 0.5. Sweeping the cube's
sliding friction anywhere below 0.5 changed **nothing at the four geoms doing the
grasping**, and ``dr_cube_friction`` (0.1-0.5) was likewise fully masked by
``dr_fingertip_friction`` (0.5-1.0). The cube now carries ``priority=1``
(``robots/cube.get_cube_spec``), so its friction is the one MuJoCo uses --
"cube has the highest priority". ``--cube-priority 0`` reproduces the old
behaviour, and the printed ``contact_fric`` column shows which regime you are in:
it tracks the requested friction at priority 1 and pins to 0.5 at priority 0.

What "sliding" is measured as
-----------------------------
For every cube-hand contact, the relative velocity of the two material points in
contact, projected onto the contact tangent plane:

    slip = ||(I - n n^T) (v_cube_point - v_hand_point)||     [m/s]

computed from body Jacobians at the contact position, so it is the true
tangential relative speed, not a proxy. ``slip_p95`` is the number to watch: mean
slip is dominated by the many near-static contacts, while the tail is the actual
slipping. ``stick_frac`` is the fraction of contacts below 1 mm/s.

Also reported, for the "is the cube getting stuck between the gaps" question:
``penetration`` (max contact depth, mm) and ``pads_per_step`` (how many distinct
hand geoms touch the cube at once -- a cube wedged into the gaps between the
sensor pads touches many at once and penetrates deeply).

Usage:
  uv run python scripts/friction_probe.py \\
      logs/rsl_rl/leap_xela_cube_reorient_reference/2026-08-29_16-11-45_reference-v3-eulerdamp/model_2999.pt \\
      --frictions "(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)"
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

from load_env import load_project_dotenv

load_project_dotenv()

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

_DEFAULT_TASK = "Mjlab-LeapXELA-Cube-Reorient-Reference"
_STICK_THRESHOLD = 1e-3  # m/s; below this a contact is "stuck", not sliding.


@dataclass(frozen=True)
class ProbeConfig:
  checkpoint: tyro.conf.Positional[str]
  task: str = _DEFAULT_TASK
  frictions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)
  """Cube sliding-friction values to sweep. The policy is NOT retrained."""
  cube_priority: int = 1
  """0 reproduces the pre-priority mixing, where these values are mostly inert."""
  num_envs: int = 16
  num_steps: int = 400
  """Control steps per friction value, at 20 Hz."""
  contact_envs: int = 4
  """How many envs to mirror into a host MjData for contact analysis each step."""
  seed: int = 42
  device: str | None = None
  out: str | None = None
  """Optional path to write the results as JSON."""


def _load_policy(task: str, checkpoint: str, wrapped, device: str):
  agent_cfg = load_rl_cfg(task)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  with tempfile.TemporaryDirectory() as tmp:
    runner = runner_cls(wrapped, asdict(agent_cfg), tmp, device=device)
    runner.load(str(checkpoint), map_location=device)
  return runner.get_inference_policy(device=device)


def _contact_stats(model, data, cube_geom: int, hand_geoms: set[int]) -> dict:
  """Slip speed, penetration and contact spread for cube-hand contacts."""
  slips: list[float] = []
  depths: list[float] = []
  frics: list[float] = []
  tip_frics: list[float] = []
  tip_slips: list[float] = []
  pad_frics: list[float] = []
  touched: set[int] = set()

  jacp1 = np.zeros((3, model.nv))
  jacp2 = np.zeros((3, model.nv))
  for i in range(data.ncon):
    c = data.contact[i]
    if c.geom1 != cube_geom and c.geom2 != cube_geom:
      continue
    other = c.geom2 if c.geom1 == cube_geom else c.geom1
    if other not in hand_geoms:
      continue  # terrain contacts are a drop, not a grasp
    touched.add(other)

    b_cube = model.geom_bodyid[cube_geom]
    b_hand = model.geom_bodyid[other]
    mujoco.mj_jac(model, data, jacp1, None, c.pos, b_cube)
    mujoco.mj_jac(model, data, jacp2, None, c.pos, b_hand)
    v_rel = (jacp1 - jacp2) @ data.qvel
    n = c.frame[0:3]  # contact normal, world frame
    v_tan = v_rel - np.dot(v_rel, n) * n
    slip = float(np.linalg.norm(v_tan))
    slips.append(slip)
    depths.append(float(-c.dist))
    frics.append(float(c.friction[0]))
    # The four fingertips carry friction 0.5 against the 32 pads' 0.2, so at
    # priority 0 they mask a different range of requested values than the pads
    # do. Keeping them separate is the only way to see that directly.
    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or "").endswith(
      "_tip"
    ):
      tip_frics.append(float(c.friction[0]))
      tip_slips.append(slip)
    else:
      pad_frics.append(float(c.friction[0]))

  return {
    "slips": slips,
    "depths": depths,
    "frics": frics,
    "tip_frics": tip_frics,
    "tip_slips": tip_slips,
    "pad_frics": pad_frics,
    "n_contacts": len(slips),
    "n_pads": len(touched),
  }


def run_probe(cfg: ProbeConfig) -> list[dict]:
  from leap_xela_mjlab.tasks.reorient.config.env_cfg import make_reorient_env_cfg

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  results: list[dict] = []

  for fric in cfg.frictions:
    # play=True already strips the dr_* events, so the pinned friction survives
    # every reset instead of being resampled by dr_cube_friction.
    env_cfg = make_reorient_env_cfg(
      finger_tip_type="Box_palm192",
      play=True,
      preset="baseline",
      cube_half_size=0.035,
      cube_mass=0.108,
      cube_friction_sliding=fric,
      cube_priority=cfg.cube_priority,
    )
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    # play=True sets an effectively infinite episode; restore a real horizon so
    # a dropped cube resets and the run keeps measuring a grasp.
    env_cfg.episode_length_s = 50.0

    agent_cfg = load_rl_cfg(cfg.task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(
      env, clip_actions=getattr(agent_cfg, "clip_actions", None)
    )
    policy = _load_policy(cfg.task, cfg.checkpoint, wrapped, device)

    model = env.sim.mj_model
    host = mujoco.MjData(model)
    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube/cube")
    hand_geoms = {
      g
      for g in range(model.ngeom)
      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
        "robot/"
      )
      and (model.geom_contype[g] or model.geom_conaffinity[g])
    }

    all_slips: list[float] = []
    all_depths: list[float] = []
    all_frics: list[float] = []
    all_tip_frics: list[float] = []
    all_tip_slips: list[float] = []
    all_pad_frics: list[float] = []
    pads_per_step: list[int] = []
    contacts_per_step: list[int] = []
    drops = 0

    n_mirror = min(cfg.contact_envs, cfg.num_envs)
    obs, _ = wrapped.reset()
    with torch.inference_mode():
      for _ in range(cfg.num_steps):
        actions = policy(obs)
        obs, _, dones, _ = wrapped.step(actions)
        drops += int(dones.sum().item())
        d = env.sim.data
        for e in range(n_mirror):
          host.qpos[:] = d.qpos[e].cpu().numpy()
          host.qvel[:] = d.qvel[e].cpu().numpy()
          # The hand is fixed-base and positioned by mocap, so without this the
          # host model keeps the hand at the origin while the cube's freejoint
          # qpos carries env ``e``'s origin offset -- and the two never touch.
          if model.nmocap > 0:
            host.mocap_pos[:] = d.mocap_pos[e].cpu().numpy()
            host.mocap_quat[:] = d.mocap_quat[e].cpu().numpy()
          mujoco.mj_forward(model, host)
          st = _contact_stats(model, host, cube_geom, hand_geoms)
          all_slips += st["slips"]
          all_depths += st["depths"]
          all_frics += st["frics"]
          all_tip_frics += st["tip_frics"]
          all_tip_slips += st["tip_slips"]
          all_pad_frics += st["pad_frics"]
          pads_per_step.append(st["n_pads"])
          contacts_per_step.append(st["n_contacts"])

    env.close()

    slips = np.asarray(all_slips) if all_slips else np.zeros(1)
    depths = np.asarray(all_depths) if all_depths else np.zeros(1)
    row = {
      "friction_requested": fric,
      "cube_priority": cfg.cube_priority,
      "contact_fric": float(np.median(all_frics)) if all_frics else float("nan"),
      "contact_fric_tip": (
        float(np.median(all_tip_frics)) if all_tip_frics else float("nan")
      ),
      "contact_fric_pad": (
        float(np.median(all_pad_frics)) if all_pad_frics else float("nan")
      ),
      "slip_mean_tip": (
        float(np.mean(all_tip_slips)) if all_tip_slips else float("nan")
      ),
      "n_tip_contacts": len(all_tip_frics),
      "slip_mean": float(slips.mean()),
      "slip_p95": float(np.percentile(slips, 95)),
      "slip_max": float(slips.max()),
      "stick_frac": float((slips < _STICK_THRESHOLD).mean()),
      "penetration_mm_p95": float(np.percentile(depths, 95) * 1e3),
      "pads_per_step": float(np.mean(pads_per_step)),
      "contacts_per_step": float(np.mean(contacts_per_step)),
      "drops": drops,
      "n_samples": int(len(all_slips)),
    }
    results.append(row)
    print(
      f"fric={fric:<5.2f} tip={row['contact_fric_tip']:<5.2f} "
      f"pad={row['contact_fric_pad']:<5.2f} "
      f"slip_mean={row['slip_mean']*1e3:7.2f}mm/s "
      f"slip_p95={row['slip_p95']*1e3:8.2f}mm/s "
      f"stick={row['stick_frac']*100:5.1f}% "
      f"pen_p95={row['penetration_mm_p95']:5.2f}mm "
      f"pads={row['pads_per_step']:4.1f} drops={row['drops']}",
      flush=True,
    )

  return results


def main() -> None:
  import leap_xela_mjlab.tasks  # noqa: F401
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(ProbeConfig)
  results = run_probe(cfg)

  print("\n" + "=" * 108)
  print(
    f"{'fric':>6} {'@tip':>6} {'@pad':>6} {'slip_mean':>11} {'slip_tip':>11} "
    f"{'slip_p95':>11} {'stick%':>7} {'pen_p95':>8} {'pads':>6} {'drops':>6}"
  )
  print("-" * 108)
  for r in results:
    print(
      f"{r['friction_requested']:>6.2f} {r['contact_fric_tip']:>6.2f} "
      f"{r['contact_fric_pad']:>6.2f} "
      f"{r['slip_mean'] * 1e3:>9.2f}mm/s {r['slip_mean_tip'] * 1e3:>9.2f}mm/s "
      f"{r['slip_p95'] * 1e3:>9.2f}mm/s {r['stick_frac'] * 100:>6.1f}% "
      f"{r['penetration_mm_p95']:>6.2f}mm {r['pads_per_step']:>6.1f} {r['drops']:>6}"
    )
  print("=" * 108)

  if cfg.out:
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out).write_text(json.dumps(results, indent=2))
    print(f"[INFO] wrote {cfg.out}")


if __name__ == "__main__":
  main()
