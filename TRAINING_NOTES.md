# Task 1 — PPO cube reorientation (no touch): what we tried and what happened

Environment: `Mjlab-LeapXELA-Cube-Reorient` (mjlab manager-based API, MuJoCo Warp, RSL-RL PPO).
All runs on a single GPU, `--num-envs 4096`, logs under
`logs/rsl_rl/leap_xela_cube_reorient/<timestamp>_<run-name>/`.

Status as of 2026-09-02: **the leading candidate is an exploration noise floor in the policy
distribution.** The per-episode success rate has now been measured at **0.03-0.05 goals per
episode, flat**, across two contact models, a 20x success weight, fine/progress reward shaping
and 3000-10000 iterations (see run 19's counter section, which dates this measurement back to
run 14). Converged `Mean action std` is 2.79, which at action scale 0.5 is a commanded joint
delta of **1.40 rad per 50 ms control step** against joint ranges of 1-2 rad, and rsl_rl's
Gaussian policy is unbounded so nothing pulls it back down. PPO is optimising the return of a
policy whose noise is large enough to erase the difference between a fine terminal manoeuvre
and a coarse hold. Run 21 is the direct test.

Two earlier framings, kept because runs 1-19 were read through them:

> **Retracted 2026-09-02.** *"The bottleneck was the contact model"* (status as of 2026-08-31),
> from run 18's 7.8x on rotate_x at `condim=6`. Run 19 ran that exact change on the reorient
> task and it landed on run 14's curve. The physics finding stands for the rotate tasks; it
> **does not transfer to reorient**, which needs terminal precision rather than gross angular
> velocity. See run 19.

> **Corrected 2026-08-31.** *"The policy reliably learns to hold the cube but not to reorient
> it; every configuration converges to a park-the-cube-and-farm-episode-length solution"*
> (runs 1-14). The run 14 rollouts show it turns the cube ~100 deg and *then* stalls 20-50 deg
> short of the goal — an approach phase followed by a stall, not immobility.

> ### Validity warning — read before trusting runs 1–11
>
> Three model-level bugs were found on 2026-08-28/29, **after** runs 1–11 were recorded.
> All three were present for every one of those runs:
>
> 1. **Wrong joint limits.** `set_joint_limits_from_joint_config` wrote only the `<default>`
>    block; MjSpec resolves defaults at parse time, so each `<joint>` kept its CAD range and
>    the edited default was overridden. Actuator `ctrlrange` was never touched at all.
> 2. **njmax overflow.** `njmax=120` was too small — every run printed
>    `nefc overflow - please increase njmax` (110× in `baseline10k`, 217× in `curriculum-10k`,
>    peak request 168). Overflow **silently drops constraint rows**, contacts and joint limits
>    alike, precisely in the high-contact states where manipulation would happen.
> 3. **Dropped `eulerdamp`.** The MJCF sets `<flag eulerdamp="disable"/>`, but mjlab attaches
>    the hand into its own scene and the parent's option block wins. Joint damping was
>    integrated implicitly instead of explicitly: peak joint velocity 21.9 rad/s vs 2.0 in the
>    source model, and the hand dropped the cube from a grasp the source model held.
>
> Runs 1–11 were therefore trained against a misconfigured model. Their *qualitative* failure
> mode (holds, does not rotate) reproduces on the fixed model — see runs 12–14 — but their
> **quantitative results and reward-shaping conclusions should not be relied on.**
> See the "Interlude" section for the fixes and what they did and did not change.

---

## Fixed setup (common to all runs)

| Item | Value |
| --- | --- |
| Envs / GPU | 4096 |
| Sim timestep / decimation | 0.01 s, decimation 5 → 50 ms control step |
| Integrator / solver | Euler, Newton, 5 iterations, 8 ls-iterations |
| `nconmax` / `njmax` | 48 / 120 — **too small, see validity warning**; 64 / 220 from run 13 on; 64 / 500 at `condim` 6 (run 18+) |
| Cube contact | `condim=3` (MJX default) through run 17 — friction[1]/[2] inert; `condim=6` from run 18 |
| Episode length | 50 s (1000 control steps) |
| Actor / critic MLP | 512-256-128, ELU, obs normalization |
| PPO | 24 steps/env, 5 epochs, 4 minibatches, lr 3e-4 adaptive, γ 0.99, λ 0.95, desired KL 0.01 (runs 13+ use playground's brax shape: 8192 envs × 40 steps, 32 minibatches, 4 epochs) |
| Policy distribution | rsl_rl `GaussianDistribution` (**unbounded**), `init_std` 1.0, `entropy_coef` 0.01 — see the noise-floor diagnosis after run 20 |
| Actor obs | joint pos, joint pos error from command, cube pos error from palm, cube orientation error matrix, last action |
| Critic obs | actor obs + joint vel, fingertip positions rel. palm, cube lin/ang vel |
| Action | delta joint position, scale 0.5, 16 hand joints |
| Terminations | time out, cube below 0.2 m, NaN |
| Throughput | ~85–95 k env-steps/s |

No touch/tactile observations are involved yet — this is the pre-flex baseline half of task 1.

---

## Run index

| # | Run | Iters (of budget) | Wall | What changed |
| --- | --- | --- | --- | --- |
| 0 | `2026-08-21_23-58-37` … `2026-08-22_00-01-15` (7 runs) | 5 each | <2 min each | Env-count scaling sweep, 1024 → 65536 |
| 1 | `baseline-no-touch-10k` | 9999 / 10000 | 221 min | Stock config from the MJX port |
| 2 | `shaped-probe-1k` | 999 / 1000 | 18.5 min | `orientation_fine` reward, success threshold 0.1 → 0.4, orientation weight 5 → 1, `init_std` 1.0 → 0.5, entropy 0.01 → 0.002 |
| 3 | `nodrift-probe-1k` | 999 / 1000 | 18.6 min | + `use_mjx_goal_drift: false` |
| 4 | `progress-probe-1k` | 999 / 1000 | 18.7 min | + `orientation_progress` potential-style reward (weight 20, clip 0.1) |
| 5 | `fric-tors-0.05` | 399 / 400 | 7.3 min | Progress reward removed; cube friction DR off; fixed torsional friction 0.05 |
| 6 | `fric-tors-0.3` | 399 / 400 | 7.2 min | torsional friction 0.3 |
| 7 | `fric-tors-1.0` | 399 / 400 | 7.3 min | torsional friction 1.0 |
| 8 | `long-fixed-10k` | 3321 / 10000 (stopped early) | 63 min | Best fixed-goal config, long run |
| 9 | `curriculum-smoke` | 39 | 0.7 min | Goal curriculum wired up: relative goals, `initial_difficulty` 0.1, step 0.05, `update_every` 50 |
| 10 | `curriculum-pace-600` | 599 | 11.3 min | Same, 600-iteration pacing check |
| 11 | `curriculum-10k` | 9999 / 10000 | 201 min | Curriculum slowed: step 0.02, `update_every` 500, `ema_alpha` 0.01 |
| — | *three model bugs found and fixed* | — | — | Joint limits, njmax, eulerdamp — see Interlude |
| 12 | `reference-palm192-cube1.0` | 3403 / 10000 (stopped) | 72 min | Hamid's exact successful config, absolute goals, no curriculum. Joint limits fixed; njmax + eulerdamp bugs still present |
| 13 | `reference-v2-batch-njmax` | 2999 / 3000 | 215 min | + `njmax` 220 / `nconmax` 64, + playground brax batch shape (8192 × 40, 32 minibatches, 4 epochs) |
| 14 | `reference-v3-eulerdamp` | 2999 / 3000 | 213 min | + `MujocoCfg(disableflags=("eulerdamp",))`. **All three bugs fixed.** Run in two parts (0–618, resumed 600→2999) |
| 15 | `rotatez-3000` | 1789 / 3000 (machine rebooted) | 81 min | `Mjlab-LeapXELA-Cube-RotateZ` — first ever run of the z-spin task. **2.19 rad/s** |
| — | *run 14 rollout analysis* | — | — | The policy is not static: it turns ~100°, then stalls. 96% of the residual error is tip-over |
| 16 | `rotatex-1800` | 1799 / 1800 | 85 min | Same task, reward axis (1,0,0) — tip-over. **0.284 rad/s, 3.93 drops/ep** |
| 17 | `rotatey-1800` | 1799 / 1800 | 82 min | Reward axis (0,1,0) — the other tip-over. **0.575 rad/s, 2.31 drops/ep**, not converged |
| 18 | `rotatex-condim6` | 1727 / 1800 (stopped early) | 94 min | Run 16 + `cube_condim=6`. **2.217 rad/s, 0.52 drops/ep** — 7.8× run 16 |
| 19 | `reference-condim6` | 1208 / 3000 (host reboot) | 104 min | Run 14 + `cube_condim=6`. The payoff test. **No effect on reorient** — matches run 14 at matched iteration |
| — | *`goals_reached` counter bug found* | — | — | Pinned at 0.0 in runs 12–19; `consecutive_success` was always correct and dates the real rate to run 14 |
| 20 | `reference-successfix` | 2999 / 3000 | 220 min | Counter fix + success weight 100 → 2000 (`100/dt`). **Harmful** — error 0.74 → 1.10, std 2.06 → 5.95. Weight reverted |
| 21 | `entropy-1e-3` | running (of 1500) | — | Run 14 + `entropy_coef` 0.01 → 0.001, sole change. The exploration-noise-floor probe |

---

## 0. Env-count scaling (7 × 5 iterations)

Throughput saturates at 4096 envs; there is no point paying the memory for more.

| Envs | 1024 | 1024 | 4096 | 8192 | 16384 | 32768 | 65536 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Best FPS | 55 k | 62 k | **97 k** | 95 k | 94 k | 94 k | 93 k |

**Decision:** 4096 envs for everything afterwards.

---

## 1. `baseline-no-touch-10k` — the failure mode we spent the rest of the day on

Stock config: `orientation` weight 5.0, success threshold 0.1 rad, MJX goal drift on,
`init_std` 1.0, `entropy_coef` 0.01. Full 10 000 iterations.

| Metric | Iter 0 | Iter 250 | Iter 9990 |
| --- | --- | --- | --- |
| `Train/mean_reward` | -3.24 | 129.2 | 122.6 |
| `Train/mean_episode_length` | 22.7 | 980 | 932 |
| `orientation_error` (rad) | 2.26 | 1.79 | 1.62 |
| `success` | 0 | 0 | 0 (peak 0.009 @4621) |
| `Episode_Termination/cube_fell` | 34.5 | — | 0.26 |

What it learned: **don't drop the cube.** `cube_fell` collapses from 34.5 to 0.26 within a
few hundred iterations, episode length pins near the 1000-step cap, and reward plateaus by
iteration ~250 and never moves again for the remaining 9750 iterations.

What it did not learn: reorientation. Orientation error parks at ~1.5–1.6 rad and stays
there; `success` is essentially zero for the whole run.

Final per-step reward breakdown makes the incentive explicit:

| Term | Per-step contribution |
| --- | --- |
| `orientation` | **+2.56** |
| `position` | +0.47 |
| `action_rate` | -0.44 |
| `hand_pose` | -0.11 |
| `success` | **0.00** |

The linear tolerance term paid 2.56/step for ~930 steps just for holding the cube near a
compromise pose. That is ~2400 reward per episode from doing nothing, versus a success bonus
that requires clearing a 0.1 rad threshold the policy never gets close to. `Policy/mean_std`
also drifted *up* to 2.64 — the policy became noisier over time rather than sharpening.

**Conclusion:** the reward is farmable and the task, as configured, has no solvable
instances early in training. Goals were drawn up to ±π from step one.

---

## 2–4. Reward-shaping probes (1000 iterations each)

Three 1 k-iteration probes, each adding one change on top of the last.

**#2 `shaped-probe-1k`** — attacked the farming incentive directly:
- new `orientation_fine` reward: dm_control-style `long_tail` sigmoid, margin 0.4, weight 5.0 — gradient concentrated near zero error
- `orientation` weight 5.0 → 1.0 (kill the flat payout)
- success threshold 0.1 → 0.4 rad (make success reachable at all)
- `init_std` 1.0 → 0.5, `entropy_coef` 0.01 → 0.002 (less thrash)

Result: reward scale drops as intended (42.9 vs 123), episode length still ~886, orientation
error still ~1.81. `success` peaked at 0.067 around iteration 570 then went back to 0.
`consecutive_success` reached 0.5 early and decayed. **No real improvement.**

**#3 `nodrift-probe-1k`** — disabled the MJX goal drift (`use_mjx_goal_drift: false`), on the
theory that a moving target was preventing convergence. Result: statistically identical to
#2 (reward 43.1, error 1.83, success 0.0). **No effect** — but the config is simpler, so it
stayed off.

**#4 `progress-probe-1k`** — added `orientation_progress`, a potential-style dense reward
paying `prev_err - err` every step (weight 20, per-step clip 0.1 so the goal-resample jump
can't swamp the success bonus). Also added a `cube_ang_speed` diagnostic metric to separate
"can't rotate the cube" from "rotates it, but not toward the goal".

Result: reward 46.3 (best of the three probes), error 1.70. `cube_ang_speed` fell from 7.95
to 1.83 rad/s. **The diagnostic was the useful output:** the policy is barely rotating the
cube at all, so the problem is not misdirected rotation — it is that the policy has stopped
moving the cube. The progress reward was dropped afterwards.

---

## 5–7. Friction ablation (400 iterations × 3)

Hypothesis: the cube can't be turned in-hand because torsional friction is too low.
Disabled cube friction DR (`randomize_geom_friction`, range 0.1–0.5) and fixed torsional
friction at 0.05 / 0.3 / 1.0, keeping fingertip friction DR at 0.5–1.0.

| Torsional μ | mean_reward | episode_len | orientation_error | cube_ang_speed | peak success |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 43.4 | 898 | 1.75 | 1.77 | 0.111 |
| 0.3 | 42.6 | 883 | 1.72 | 1.70 | 0.100 |
| 1.0 | 42.9 | 875 | 1.75 | 1.97 | 0.077 |

~~**Conclusion: friction is not the bottleneck.** A 20× change in torsional friction moves
nothing measurable.~~ Cube friction DR was re-enabled for subsequent runs.

> **Retracted 2026-08-29.** This ablation varied `friction[2]` (rolling) and `friction[1]`
> (torsional), and **at `condim=3` MuJoCo reads only `friction[0]`**. The three runs were
> bit-identical configurations as far as the solver was concerned, which is exactly why the
> table shows no differences. `cube_friction_torsional` in the env config does nothing at
> condim 3. **Sliding friction has never been ablated** — this hypothesis is still open.
>
> **Follow-up 2026-08-31:** turning those parameters *on* (`condim=6`) turned out to be the
> single largest effect measured on this project — see run 18. The ablation was not wrong
> about friction being unimportant at condim 3; it was measuring a solver that ignored it.

---

## 8. `long-fixed-10k` — the best fixed-goal config, run long

Best-known fixed-goal configuration (shaped rewards, no drift, no progress term, friction DR
back on), budget 10 000 iterations. **Stopped at 3321** once it was clear it had plateaued.

| Metric | Iter 250 | Iter 1000 | Iter 3000 |
| --- | --- | --- | --- |
| mean_reward | 47.9 | 45.2 | 43.5 |
| orientation_error | 1.53 | 1.38 | 2.01 |
| cube_ang_speed | 1.02 | 0.43 | 0.45 |
| episode_length | 952 | 903 | 874 |

`cube_ang_speed` of 0.43 rad/s at iteration 3000 is the clearest statement of the failure:
the hand is holding the cube nearly motionless. `consecutive_success` stayed at exactly 0
for the entire run. Reward is flat-to-declining across 3300 iterations.

**Conclusion:** more iterations do not fix this. The problem is task structure, not budget.

---

## 9–11. Goal curriculum

Diagnosis behind it: with goals drawn uniformly up to ±π and no relation to where the cube
currently is, there are effectively *no solvable instances* to learn from, so PPO settles on
the pose that minimises average error. Two changes:

1. **Relative goals** (`goal_relative_to_object: true`) — the sampled offset is composed
   against the cube's *current* orientation, so difficulty controls the actual distance the
   policy must cover. (Implementation note: the offset has to be deferred to the first step,
   because at reset the cube pose written by the reset event hasn't reached
   `root_link_quat_w` until after `sim.forward()`.)
2. **`goal_difficulty` curriculum term** — difficulty = fraction of π used for goal sampling,
   starting at 0.1. Promoted while an EMA of *goals reached per episode* is ≥ `promote_at`,
   demoted at half rate below `demote_at`. A new `success_count` per-env counter was added,
   because `consecutive_success` gets cleared by the success-triggered goal resample and so
   can't be used as the promotion signal.

**#9 `curriculum-smoke` (39 iters)** — plumbing check. Difficulty moved (0.10 → 0.15 → back
to 0.095), `goals_reached` peaked at 1.6. Wiring works.

**#10 `curriculum-pace-600` (599 iters)** — pacing check with step 0.05, `update_every` 50.
Difficulty **saturated at 1.0 by iteration 46**. Goals reached collapsed from 10.1/episode to
0.18, orientation error climbed from 0.40 to 1.26. The curriculum ran far ahead of
competence: `compute()` fires ~24× per training iteration, so `update_every=50` is only ~2
iterations per adjustment.

**#11 `curriculum-10k` (9999 iters)** — retuned: step 0.02, `update_every` 500 (≈20 iterations
per adjustment), `ema_alpha` 0.01. Full budget, 201 min.

| Metric | Iter 0 | 250 | 1000 | 3000 | 5000 | 9990 |
| --- | --- | --- | --- | --- | --- | --- |
| difficulty | 0.10 | 0.32 | 0.36 | 0.48 | 0.50 | **0.90** |
| goals_reached / episode | 10.13 | 0.67 | 0.62 | 0.97 | 1.00 | 0.80 |
| orientation_error | 0.39 | 1.16 | 1.16 | 1.39 | 1.30 | 1.18 |
| cube_ang_speed | 8.08 | 1.85 | 1.34 | 1.06 | 1.88 | 1.10 |
| mean_reward | 83.1 | 46.7 | 60.5 | 54.3 | 52.7 | 53.3 |
| episode_length | 23 | 863 | 932 | 946 | 931 | 931 |

Reading this:
- At difficulty 0.1 with relative goals the task is **trivially satisfied** — 10 goals per
  episode at iteration 0, before any learning. That confirms the diagnosis: the original task
  had no reachable instances, not that the hand is incapable.
- The curriculum ramped smoothly to 0.9 over 10 k iterations without oscillating, so the
  slower pacing was the right fix for #10.
- But `goals_reached` hovers around **0.8–1.0 per episode** the whole way, and orientation
  error settles at ~1.2–1.4 rad. The policy is riding the promotion threshold
  (`promote_at: 1.0`) rather than getting genuinely better — difficulty rises because the EMA
  brushes 1.0, not because competence grew.
- `cube_ang_speed` ~1.1 rad/s: still barely turning the cube.
- Final reward breakdown: `orientation` +0.62, `position` +0.46, `orientation_fine` +0.10,
  `success` +0.07 per step. Success is now nonzero — a first — but it is a rounding error
  next to the holding terms.

Note on the `success` metric in TensorBoard: it reads ~0 in all curriculum runs because a
success immediately resamples the goal, so the instantaneous "within threshold" fraction is
near zero by construction. Use `goals_reached` (per-episode count) instead.

---

## Interlude — three model bugs, found 2026-08-28/29

Runs 1–11 all pointed at the same wall, so the next step was to stop tuning the learner and
audit the model itself against the source MJX scene. Three defects came out of it.

### Bug 1 — joint limits never reached the model (2026-08-28)

`simplify_model_for_mjx.py`'s `set_joint_limits_from_joint_config` set
`default.joint.range` on the class defaults only. **MjSpec resolves class defaults into each
element at parse time**, so every `<joint>` already carried the base model's CAD range, and
that inline attribute overrode the edited default. Actuator `ctrlrange` was never written at
all. The generated XML *looked* correct in its `<default>` block while the compiled model
used CAD limits for all 16 joints.

Largest discrepancies: `th_mcp` upper 2.443 → 1.396 (much tighter), `th_cmc` lower
−0.349 → −0.785 (wider), `rot` ±0.262 → ±0.349.

Fixed by writing limits onto `spec.joints` and `spec.actuators` elements directly. Verified
by compiling both `--mode Box` and `--mode CoACD` and diffing `MjModel.jnt_range` /
`actuator_ctrlrange` against `joint_config.json` through mjlab's own `get_spec()` path:
**0/16 mismatches**.

Note: `joint_config.json` lists `rot` with `lower` 0.349 > `upper` −0.349 (inverted); the
script now swaps them with a warning.

### Bug 2 — njmax overflow silently dropped constraints (2026-08-29)

`njmax=120` against a peak request of **168**. Every training run logged so far printed
`nefc overflow - please increase njmax to N`:

| Run | overflow warnings |
| --- | --- |
| `baseline10k` | 110 |
| `curriculum-10k` | 217 |
| `long-fixed-10k` | 173 |
| `progress1k` | 75 |
| `reference-palm192-cube1.0` (run 12) | 56 |

Overflow drops constraint rows — contacts *and* joint limits — and it happens in exactly the
high-contact states where in-hand manipulation would occur. Raised to `njmax=220` /
`nconmax=64` (Hamid's playground values; measured peak `ncon` was 24). Runs 13, 14 and 15
report **0 warnings**.

Why the earlier CPU check missed it: a scripted CPU rollout of the rigid Box model peaks at
`nefc` 108, comfortably under 120. The GPU runs spike higher than a scripted rollout does.
**Grep every run log with `grep -c overflow` before trusting it.**

### Bug 3 — mjlab's attach dropped `eulerdamp` (2026-08-29)

The generated MJCF carries `<flag eulerdamp="disable"/>`, but mjlab attaches the hand spec
into its own scene and the **parent's** option block wins — visible as
`Attach conflict ... keeping parent value` in `MUJOCO_LOG.TXT`. `MujocoCfg.apply()` only ORs
`disableflags` in, so nothing restored it. His model integrates joint damping explicitly,
ours was doing it implicitly.

Measured on a scripted grasp: **peak hand joint velocity 2.0 rad/s (his) vs 21.9 (ours)**,
and ours dropped the cube from a grasp his model held. Fixed with
`MujocoCfg(disableflags=("eulerdamp",))`.

**Generalise this one:** any `<option>`-block setting in a robot MJCF is at risk of being
silently discarded by attach. Check `env.sim.mj_model.opt` against the source XML.

### What the parity audit found otherwise

Compiling our `ManagerBasedRlEnv` model and diffing `MjModel` against
`scene_mjx_cube_Box_mjx.xml` — everything else **matched exactly**: timestep 0.01, Euler,
Newton, pyramidal cone, impratio 1, 5 iterations, 8 ls-iterations, joint ranges, ctrlranges,
`jnt_actfrcrange ±0.2196`, dof damping 0.2, armature 0.00149376, frictionloss 0.02, actuator
kp 3 / kv 0, fingertip friction and size, cube mass 0.108, reward weights, action scale 0.5,
`ema_alpha` 1.0, decimation 5.

Two known, deliberate differences left alone: his floor is `contype`/`conaffinity` 2/2 so the
hand cannot collide with it, ours is 1/1 (the hand sits 25 cm up, so it is unreachable
either way).

### Two dead config values, and one ruled-out hypothesis

`joint_config.json` specifies `effort: 0.95` and `velocity: 8.48`. **Neither is read by
anything.** The compiled torque limit comes from a hardcoded
`actuatorfrcrange="-0.2196 0.2196"` in `replace_dynamics_options`; there is no velocity limit
at all. Also `kv="0.01"` on the `<position>` default is dropped by `MjSpec.to_xml`, so the
servo is pure-P.

The 4.3× gap between 0.2196 and 0.95 looked like the best remaining explanation for
"holds but never rotates" — a ceiling comfortable for holding and too low for breaking
stiction. **Checked 2026-08-29 and ruled out.** Playground's `leap_rh_mjx.xml`, the model
behind the reference **LeapHand** run that reaches reward 375 on this task, carries the
identical line:

```xml
<joint axis="0 0 -1" damping="0.2" armature="0.00149376"
       actuatorfrcrange="-0.2196 0.2196" frictionloss="0.02"/>
```

Hamid's `replace_dynamics_options` is copying playground verbatim. A hand with the same
torque ceiling, damping, armature and frictionloss learns this task to 375, so 0.2196 is not
what blocks rotation. Not worth a training run.

---

## 12–14. Reference runs — Hamid's config on the fixed model

Purpose: stop iterating on our own reward variants and reproduce the supervisor's exact
successful configuration on a model we trust, as a clean baseline.

Config (his `reorient.py`, which is also his post-thumb-fix repeat config):
`palm_euler=[0, 1.57+0.35, -1.57]`, `cube_friction=0.3`, `cube_scale_factor=1`,
`cube_pos=[0.11, 0.0, 0.1]`, `cube_euler=[0,0,0]`, `finger_tip_type='Box'`.
Absolute goals, no curriculum, success threshold 0.1 — i.e. **not** the curriculum task from
runs 9–11.

| Run | Bugs still present | Final 250 iters, mean reward | Peak | ang_speed | orient_err | cube_fell | overflows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 `reference-palm192-cube1.0` | njmax, eulerdamp | 166.6 (@2800) | 183 | 0.73 | 0.91 | 0.11 | 56 |
| 13 `reference-v2-batch-njmax` | eulerdamp | **189.0** | 204.8 | 0.65 | 0.76 | 0.16 | 0 |
| 14 `reference-v3-eulerdamp` | none | **187.5** | 197.0 | 0.69 | 0.75 | 0.15 | 0 |

Reading this:

- **The njmax fix is worth ~22 reward** (12 → 13). This is the only change in the whole
  campaign that moved the curve materially. It is also confounded with the batch-shape
  change, so attribute it to "run 13's two changes together" rather than to njmax alone.
- **The eulerdamp fix is worth nothing in reward** (13 → 14, 189.0 vs 187.5, one seed each —
  within noise). Its value is physics fidelity, not score. Keep it: a model that runs joint
  velocities 10× too high is not a model worth collecting tactile data from in task 2.
- **The failure mode is completely unchanged.** Across all three, `cube_ang_speed` sits at
  0.65–0.73 rad/s and `goals_reached` is **0.0**. Run 14's reward decomposes as `orientation`
  +3.99 and `position` +0.48 per step against `success` +0.003 — the policy is being paid to
  hold, and it holds. `cube_fell` 0.15/episode means the grip itself is excellent.
- Run 14 was flat over its last 1250 iterations (184.7 → 187.4 → 187.5 in 500-iteration
  blocks). Longer budgets are not the answer here either.

> **Corrected 2026-09-02.** The `goals_reached` **0.0** above is a counter bug, not a
> measurement — `success_count` was incremented on an unreachable branch for all of runs 12–19.
> `consecutive_success`, which was never affected, reads **0.0395 goals/episode** at run 14's
> iteration 2999. The reward decomposition and the "it is being paid to hold, and it holds"
> reading are unchanged; the success rate was small but never zero. See run 19.

### Against the supervisor's numbers

Step-matched at 205M steps (iteration 627 at 8192 × 40), run 13 read **176**; his three
post-thumb-fix MJX/brax repeats of this exact config read **157.7 / 166.3 / 171.2**. Ours
then keeps climbing to ~198 by 983M steps — **4.8× further than he ever ran**. Caveat: brax
PPO vs rsl_rl, so indicative rather than a controlled comparison.

**The 284 result is not a target.** His celebrated palm-1.92 run (141 → 284) predates
`leapXELA_model` commit `d60d4bd "fixed the collision model for thumb"`, where `th_px`'s
collision box was defined in `th_bs` and `th_ds`'s in `th_px`, so both moved with the wrong
parent frame. After the fix the same config produced 166/158/171 across three seeds. We are
on the post-fix model (our pin has `d60d4bd` as an ancestor). Do not read our plateau as a
failure to match 284.

---

## 15. `rotatez-3000` — the hand *can* turn the cube

`Mjlab-LeapXELA-Cube-RotateZ` was written on 2026-08-29 and had never been trained. Launched
after the torque hypothesis was ruled out.

Why it is a different experiment, not another variant: the entire reward is
`cube_ang_vel_w · [0,0,1]` at weight 1.0 plus `termination` at −100, with `linvel`,
`hand_pose`, `action_rate` and `energy` all at weight 0.0. There is **no goal, no success
threshold and no curriculum**. The reorient reward pays for orientation error being small,
which a well-placed motionless cube already achieves. rotate_z has no such fixed point:
**a hand that holds still scores exactly zero.**

**Result: `Episode_Reward/angvel` = 2.19 rad/s** (mean of the last 250 iterations), mean
reward 109.4, `cube_fell` 0.17/episode, 0 njmax overflows, 123 k steps/s. mjlab reports
`Episode_Reward/*` per second and the weight is 1.0, so that is **2.19 rad/s of sustained,
signed z-spin** — about 0.35 rev/s. Every converged reorient run reports `cube_ang_speed`
0.65–0.73, and that is an *unsigned norm*, i.e. mostly jitter.

Curve: −5.1 (iter 0) → 77 (150) → 99 (450) → 106 (900) → 110 (1650), still creeping up.

**Conclusion: the physics permits rotation.** The XELA fingertip pads are not a hard blocker,
and reorient's plateau is not a "the hand cannot turn this cube" problem. Per the
pre-registered reading above, this deprioritised fingertip geometry and promoted reward
structure, curriculum and control resolution.

The run did not finish: the machine rebooted overnight and the log ends mid-iteration
1789/3000 with no traceback. Last checkpoint `model_1700.pt`. `--resume-from` can pick it up.

---

## Interlude — what run 14 actually does at play time (2026-08-31)

Before spending runs on reward structure, the reference policy was rolled out and measured:
`reference-v3-eulerdamp/model_2999.pt`, 16 parallel play episodes × 700 steps (35 s),
absolute goals, success threshold 0.1 rad (5.7°).

| | value |
| --- | --- |
| start orientation error | 51–179° (median ~125) |
| **min error reached** | median **27.6°**, best 10.3, worst 79.3 |
| final error | median 30.1° (drifts back up after the minimum) |
| episodes under the 5.7° threshold | **0 / 16** |
| `cube_ang_speed`, first 20% of episode | **1.32 rad/s** |
| `cube_ang_speed`, last 50% | **0.39 rad/s** |

**This corrects the "holds the cube but never turns it" framing used throughout runs 1–14.**
The real failure mode is an **approach phase followed by a stall**: ~100° of purposeful
reorientation in the first ~15 s, then an indefinite hold at 20–50° of error. The
training-average `cube_ang_speed` of 0.65–0.73 is the mean of a fast approach and a long
static hold, which is what made the cube look immobile.

So "nothing in the reward pays for *moving* the cube" is wrong as stated — the orientation
reward demonstrably drives the approach. The open problem is **terminal precision**.

### Which part of the goal does it fail to close?

Decomposing the cube→goal error into a world-frame rotation vector `r` (axis × angle) over
64 play episodes × 700 steps. World +z is the palm normal, so `|r_z|` is the "spin it in the
fingertips" component and `|r_xy|` the "tip it over" component; `|r|² = r_z² + |r_xy|²`, so
the split is exact.

| | start | at best | median reduction |
| --- | --- | --- | --- |
| total | 123.8° | 34.0° | 74.2% |
| vertical (spin, `r_z`) | 56.2° | **8.7°** | **88.1%** |
| horizontal (tip, `r_xy`) | 94.9° | **31.0°** | **64.3%** |

**96% of the leftover error at the best moment is tip-over.** Not a dimensionality artifact:
a uniform 74.2% reduction would leave 15.4 / 26.1, and the actual 8.7 / 31.0 is 1.8× better
than that vertically and 1.2× worse horizontally. The dimension-independent numbers are the
reduction percentages, 88% vs 64%.

This lines up exactly with run 15: rotate_z rewarded spin about world z and got 2.19 rad/s —
the one axis proven available is the one axis the policy closes.

Incidental: `InHandReorientationCommand._sample_goal` composes goals as `Rx(a)·Ry(b)` only,
never a z rotation, so the goal *set* is a 2-parameter subset of SO(3). Matches MJX, and the
cube starts at a uniformly random quaternion so the required error still carries 56° of z
content at start. Not the cause.

> **Caveat:** both probes were throwaway scripts run against `scripts/render.py` and were not
> kept. The numbers above are the record; re-deriving them needs the probe rewritten.

---

## 16–17. `rotate_x` / `rotate_y` — the tip-over axes, at condim 3

`axis` became a parameter of `make_rotate_z_env_cfg` (default z unchanged, so run 15 and the
existing task are untouched), and all three axes are registered from
`tasks/rotate_z/config/__init__.py`. Matched to run 15 exactly: 8192 envs, seed 42, 1800
iterations. 0 njmax overflows, 0 NaN in both.

Mean of the last 250 iterations:

| task | axis | ang vel | reward | drops/episode |
| --- | --- | --- | --- | --- |
| 15 `rotate_z` | twist about palm normal | **2.19 rad/s** | 109.4 | **0.17** |
| 17 `rotate_y` | tip | 0.575 | 27.5 | 2.31 |
| 16 `rotate_x` | tip | 0.284 | 12.3 | 3.93 |

**Read at the time as: the hand does not fail to tip the cube — it can only do it by dropping
it.** Twist is 4–8× faster at 14–23× fewer drops. rotate_z had `cube_fell` down to 0.2 by
iteration 100 and then spun freely; x and y never got below ~2 drops per episode, so PPO
traded drops for rotation for all 1800 iterations without finding a strategy that keeps hold.

Two real caveats: **rotate_y had not converged** (0.15/0.23/0.38/0.49/0.55/0.60 at
300/600/900/1200/1500/1799, still climbing), so 0.575 is a lower bound; and **x and y differ
by 2×**, so the two tip-over axes are not equivalent — consistent with the palm geometry
(thumb on one side).

**Run 18 then showed this whole table is a property of `condim=3`, not of the hand.**

---

## 18. `rotatex-condim6` — the contact model was the bottleneck *for tipping*

`condim` became a parameter of `get_cube_spec`/`get_cube_cfg` (validated to 1/3/4/6) and of
`scripts/train.py` (`--cube-condim`). MuJoCo takes a dynamically generated contact's condim
as the **max** of the two geoms', and the hand geoms declare none (so 3) — setting it on the
cube alone is therefore enough to make torsional (`condim≥4`) and rolling (`condim=6`)
friction live on every fingertip/cube contact.

Identical to run 16 in every other respect: same task, axis (1,0,0), 8192 envs, seed 42.

| rotate_x | ang vel | reward | drops/episode |
| --- | --- | --- | --- |
| 16, condim 3 | 0.284 | 12.3 | 3.93 |
| **18, condim 6** | **2.217** | **110.6** | **0.52** |

**7.8× faster rotation at 7.5× fewer drops**, and 2.217 rad/s about a *tip-over* axis matches
run 15's twist rate about the palm normal (2.19). Curve: 0.13 (iter 100) → 1.54 (300) → 1.94
(600) → 2.16 (1200) → 2.24 (1727), converged. 0 NaN, 0 njmax overflows, 102 k steps/s.

**This overturns the reading of runs 16–17.** "The hand can only tip the cube by dropping it"
is true *at condim 3 only*. With torsional and rolling friction live, the hand tips about as
well as it twists. The bottleneck was neither the fingertip pads nor the reward — it was that
the contact model discarded exactly the friction components fine in-hand regrasping needs.

Why this was not visible earlier: `condim=3` is the MJX/playground default and was inherited
from Hamid's scene, and the runs 5–7 friction ablation appeared to clear friction as a
suspect while in fact varying two parameters the solver never read.

Sizing: condim 6 costs 6 constraint rows per contact instead of 3, so the same grasp needs
roughly double the buffer. At `njmax=200` this run overflowed within 5 iterations (peak
request 234, and it climbs as the grip tightens) → 500. condim 3 keeps its old value in both
tasks so runs 12–17 stay comparable; njmax has no effect on the dynamics unless it overflows.

Stopped at iteration 1727/1800; the cause was not captured because today's runs were launched
without a console log. The numbers were converged and flat over the last 500 iterations, so
this does not affect the result.

> **Scope narrowed 2026-09-02.** Everything above stands as measured. What did *not* survive is
> the extrapolation to the real task: run 19 puts `condim=6` on run 14's reorient config and
> changes nothing. The heading originally read "the contact model was the bottleneck" — it is
> the bottleneck for gross tip-over angular velocity, which is what this task rewards, and not
> for reorient's terminal precision.

---

## 19. `reference-condim6` — the payoff test, answered: no

Run 14's exact configuration plus `--cube-condim 6`. `diff` of the two `params/env.yaml`
files is **3 lines**: `condim: 6` and `njmax: 220 → 500`. Same task, seed, batch shape,
rewards, DR and 3000-iteration budget, so it is directly comparable to run 14's 187.5.

Two things had to be fixed first, both of which would have quietly invalidated the run:

1. **`njmax` in the reorient task was unconditional at 220.** Now `220 if cube_condim <= 3
   else 500`, mirroring the rotate task. Without this the run would have reproduced bug 2.
2. **`_apply_cube_overrides` forced `disable_cube_friction_dr=True` for *any* `--cube-*`
   flag.** That is right when pinning a friction value (`dr_cube_friction` resamples sliding
   friction every reset and would overwrite it) but wrong for condim, which the event does not
   touch — it would have removed sliding-friction DR and changed a second variable against the
   baseline. Now gated on the friction flags only.

Smoke test (3 iterations, 8192 envs): no NaN, **0 njmax overflows**, `dr_cube_friction`
still present.

Pre-registered reading, written before the run (kept verbatim; see the counter caveat below):

> What to look for against run 14 (187.5 mean reward, `orientation_error` 0.76,
> `goals_reached` 0.0, `cube_ang_speed` 0.70):
> - `goals_reached` above zero at all would be the first success this project has produced on
>   absolute goals.
> - Failing that, `orientation_error` dropping below ~0.5 would say the stall moved even if the
>   5.7° threshold is still out of reach.
> - Reward alone is a weak signal — run 14 farms 187 while reaching no goals.

**The run died at iteration 1208/3000**, 104 min in (started 20:25, last checkpoint
`model_1200.pt` at 22:09). The host rebooted; this machine goes down uncleanly every few hours.
Resuming it is no longer worth the GPU time, because the 1208 iterations it did run already
answer the question — through a counter that had been working the whole time while the one
being watched was broken.

### The counter that was structurally zero

`goals_reached` reads **exactly 0.0 in every run from 12 to 19**. That was never a plateau, it
was a bug: `success_count` was incremented inside the `elif self.cfg.update_goal_on_success:`
branch of `_update_command`, and `use_mjx_goal_drift=True` takes the branch before it, so the
increment was **unreachable** for the entire reference series. It is now incremented in
`_update_metrics` off the success flag itself, exactly as playground does — how the goal is
updated on success is independent of counting it.

`consecutive_success` was **never** affected by this. It is accumulated in `_update_metrics`
and was logged in every run. It is cleared on a goal resample, so it is not a per-episode goal
count in the strict sense; but at these rates — order 0.03 goals per episode, i.e. two
successes in one episode essentially never happens — it is numerically the same quantity. Run
20 confirms that directly (`goals_reached` 0.0287 vs `consecutive_success` 0.0287, identical).
So **the true success rate can be read retroactively, back to run 14.**

Note for the curriculum: `goal_difficulty` promotes on `success_count`, the same counter, so the
bug would have frozen promotion outright. But runs 9–11 ran with `use_mjx_goal_drift: false`
(confirmed in `curriculum-10k/params/env.yaml`), which takes the working branch, and their
logged `goals_reached` is nonzero throughout. **Runs 9–11 are not invalidated.** The trap is
live for any future curriculum run on the reference config, which does have drift on.

### Run 19 vs run 14

Run 19 (condim 6), the 1208 iterations it completed:

| iter | 149 | 449 | 749 | 1049 | 1199 |
| --- | --- | --- | --- | --- | --- |
| `consecutive_success` | 0.0000 | 0.0158 | 0.0188 | 0.0258 | 0.0374 |
| `orientation_error` | 1.6622 | 1.3874 | 0.8925 | 0.9440 | 0.8960 |
| `Mean action std` | 1.39 | 2.08 | 2.54 | 2.76 | 2.81 |

Run 14 (condim 3), the full run:

| iter | 749 | 1199 | 1649 | 2099 | 2549 | 2999 |
| --- | --- | --- | --- | --- | --- | --- |
| `consecutive_success` | 0.0433 | 0.0349 | 0.0474 | 0.0379 | 0.0483 | 0.0395 |
| `orientation_error` | 0.9164 | 0.8703 | 0.7686 | 0.7735 | 0.7712 | 0.7372 |
| `Mean action std` | 2.66 | 2.80 | 2.78 | 2.79 | 2.77 | 2.79 |

Matched at iteration 1199, the last point run 19 reached:

| | `consecutive_success` | `orientation_error` | `Mean action std` |
| --- | --- | --- | --- |
| 14, condim 3 | 0.0349 | 0.8703 | 2.80 |
| 19, condim 6 | 0.0374 | 0.8960 | 2.81 |

**`condim=6` does nothing for reorient.** All three quantities are statistically identical at
the matched iteration, and run 19's trajectory over 1200 iterations lies on top of run 14's.
Resuming it to 3000 would buy a confirmation, not an answer — the pre-registered question
("does `goals_reached` leave zero?") is answered by a counter that shows both runs sitting at
the same nonzero-but-tiny rate.

**This does not retract run 18.** The physics finding there is real and was measured on a
converged run: at `condim=3` the torsional and rolling friction coefficients are inert, and
turning them on makes the tip-over axis 7.8× faster at 7.5× fewer drops. What run 19 shows is
that **the finding does not transfer to reorient**, and the reason is visible in what each task
rewards. rotate_x pays for gross angular velocity, which is exactly what extra friction buys.
Reorient needs *terminal precision* — parking the cube inside 5.7° and staying there — and
nothing about the contact model was preventing that.

> **Retracted 2026-09-02.** The status paragraph of 2026-08-31 said *"the bottleneck was the
> contact model"*, on the strength of run 18. Run 19 is the controlled test of that claim on
> the task that matters and it is negative. The bottleneck for reorient is elsewhere.

---

## 20. `reference-successfix` — the counter fix, and a 20× success weight

Two changes on top of run 14, run to the full 3000 iterations (220 min, condim 3):

1. The `success_count` fix above.
2. `success` reward weight **100 → 2000**. Reason: mjlab multiplies every reward term by `dt`
   (0.05 s) before summing, and playground does not, so a weight of 100 copied from Hamid's
   config was being paid as 5 — the success bonus had been **20× too weak in every run to
   date**. `100 / dt = 2000` restores playground's effective magnitude.

Because change 1 turned out to reveal nothing new (below), this is effectively a clean
single-variable test of change 2.

| iter | 349 | 1149 | 1549 | 2999 |
| --- | --- | --- | --- | --- |
| mean reward | 143.60 | 118.31 | 109.74 | **75.05** |
| `Mean action std` | 2.06 | 4.49 | 5.17 | **5.95** |
| `consecutive_success` | — | — | — | 0.0287 |
| `goals_reached` | — | — | — | 0.0287 |

(`orientation_error` and the success counters were not read at the intermediate iterations;
the final values are `orientation_error` 1.0998, episode length 953.69, `cube_fell`
0.425/episode, difficulty pinned at 1.0.)

**The 20× success weight is harmful.** Three independent readings, all pointing the same way:

- **Reward falls monotonically within the run**, 143.6 → 75.05. Note the cross-run comparison
  against run 14's 187.5 is confounded — the success term's weight changed, so total reward is
  not on the same scale — but the *within-run* decline is not confounded, and run 14 was flat
  over its final 1250 iterations where this run is still dropping.
- **`orientation_error` is worse**: 1.0998 vs run 14's 0.7372. The policy holds the cube
  further from the goal than the baseline does.
- **`Mean action std` diverges**: 2.06 → 5.95, where run 14 sits flat at 2.77–2.80 for 3000
  iterations. A rare, high-value bonus behind a threshold the policy cannot reliably clear
  reads to PPO as an argument for more exploration, and there is nothing bounding that (see the
  diagnosis below).
- **The success rate does not move**: 0.0287, inside the 0.03–0.05 band every other run sits in.

**Success weight reverted to 100** (commit `1132681`). The `dt` discrepancy against playground
is real, is now documented, and is **deliberately left in place** — run 20 is the reason.

**The counter fix revealed no new information.** Run 20's `goals_reached` of 0.0287 *is* run
14's `consecutive_success` of 0.0395, the same quantity measured the same way. Across the run
`goals_reached` oscillated between 0.02 and 0.06 with no trend from iteration 299 to 2899. The
fix is worth keeping — the metric now means what its name says, and the curriculum's promotion
criterion no longer reads a dead counter — but it **did not unlock anything**, and the honest
statement is that the project's success rate was always this and was never zero.

### What that leaves

Per-episode goals reached has now been **0.03–0.05, flat**, across:

| varied | runs | result |
| --- | --- | --- |
| contact model (`condim` 3 vs 6) | 14 vs 19 | 0.0349 vs 0.0374 at matched iteration |
| success weight (100 vs 2000, i.e. 20×) | 14 vs 20 | 0.0395 vs 0.0287 |
| reward shaping (`orientation_fine`, `orientation_progress`) | 2–4, 8 | no movement |
| budget (3000 vs 10000 iterations) | 1, 8, 11, 14 | flat after ~250 |

Four different families of intervention, one number. That pattern — everything changes, the
outcome does not — is what motivated looking at the learner instead of the task.

---

## Diagnosis — an exploration noise floor (2026-09-02)

The candidate that survives the table above is that **the policy's own action noise is the
binding constraint**, and that it is not converging downward because nothing in this PPO
implementation makes it.

**1. The converged noise is enormous relative to the action space.**
`Mean action std` converges to **2.79** (run 14, flat for 3000 iterations). The action term is
delta joint position with `scale=0.5` and `clip_to_ctrl_limits=True`
(`tasks/reorient/config/env_cfg.py`), `ema_alpha=1.0`, i.e. no smoothing. So the *commanded
joint delta* has standard deviation **2.79 × 0.5 = 1.40 rad per 50 ms control step**, against
joint ranges of roughly 1–2 rad.

| | action std | commanded delta std | P(\|delta\| > 1.0 rad) |
| --- | --- | --- | --- |
| run 14 (converged) | 2.79 | 1.40 rad | **47%** |
| run 20 (final) | 5.95 | 2.98 rad | **74%** |

Roughly half of all sampled actions saturate the ctrl limits. Training-time behaviour is close
to **random bang-bang joint targets**, low-pass filtered by the `kp=3` P-servo.

**2. Nothing pulls it back down.** `entropy_coef: 0.01` (confirmed in the run's
`params/agent.yaml`). Entropy of a 16-dimensional Gaussian at std 2.79 is 39.1, so the entropy
bonus contributes `0.01 × 39.1 = 0.391` to the loss — against a **mean surrogate loss of
0.004**. Two orders of magnitude apart. The entropy term is not a regulariser here; it is the
objective.

**3. And there is no finite optimum for it to reach.** rsl_rl's `GaussianDistribution` is
**unbounded**: entropy is `const + log(std)`, so the entropy bonus pushes `log(std)` upward by
a constant gradient forever, with no std at which the pressure stops.

> **Verified against source, 2026-09-02** (brax 0.14.2 wheel + a `mujoco_playground` clone,
> both read outside the project venv so the live run was untouched).
>
> - `brax/training/agents/ppo/networks.py:100` — `distribution_type: Literal['normal',
>   'tanh_normal'] = **'tanh_normal'**`. The default is the squashed distribution, and
>   `mujoco_playground` never overrides it (no `distribution_type` / `NormalTanh` reference
>   anywhere in the repo).
> - `brax/training/distribution.py:83-92` — entropy is computed **after** the transform:
>   `dist.entropy() + postprocessor.forward_log_det_jacobian(sample)`. So it is the entropy of
>   the *post-tanh* action distribution, which is supported on (-1, 1) and therefore capped at
>   `log 2 = 0.693` per dimension.
> - `mujoco_playground/config/manipulation_params.py:140-157` — `LeapCubeReorient` uses
>   `entropy_cost = 1e-2`, 8192 envs, unroll 40, 32 minibatches, 4 updates/batch, lr 3e-4,
>   (512, 256, 128). **Identical to our reference preset, including the entropy coefficient.**
>
> Numerically, entropy per dimension as a function of std:
>
> | std | brax (tanh-normal) | rsl_rl (unbounded Gaussian) |
> | --- | --- | --- |
> | 0.4 | 0.353 | 0.503 |
> | 0.8 | **0.677** (peak) | 1.196 |
> | 1.0 | 0.669 | 1.419 |
> | 2.0 | -0.004 | 2.112 |
> | **2.79** (our converged value) | **-0.853** | **2.445** |
> | 5.95 (run 20) | -5.027 | 3.202 |
>
> **The same coefficient does opposite things.** brax's entropy bonus peaks at std ~0.8 and
> then falls steeply, so at our converged std of 2.79 playground's own objective would be
> pushing the std *down*. rsl_rl's grows as `+log(std)` forever, so it pushes *up* without
> limit. `entropy_coef = 0.01` is not a shared setting that we happened to tune badly — it is
> the same number attached to two objectives with opposite gradients above std ~0.8.

**4. This is a parity gap the 2026-08-29 audit could not have found.** That audit diffed
`MjModel` exhaustively and matched everything — timestep, solver, joint ranges, ctrlranges,
damping, armature, frictionloss, actuator gains, masses, reward weights, action scale,
decimation. It never compared the **policy distribution**. Same model, structurally different
policy class, and no amount of model-level diffing would have surfaced it.

**5. It was visible on day one and never followed up.** Run 1's notes record:
*"`Policy/mean_std` also drifted **up** to 2.64 — the policy became noisier over time rather
than sharpening."* That is this diagnosis, written on 2026-08-22 and read at the time as a
curiosity.

**Why this explains what nothing else does.** The rollout probe (interlude after run 15)
measured the *mean* action reaching a median 27.6° of error, while the training-average error
is ~42°. PPO optimises the return of the **noisy** policy, not of its mean. Under 1.4 rad of
per-step joint noise, a fine terminal manoeuvre and a coarse hold near the goal score the
same — the manoeuvre's advantage is inside the noise. Grasping is robust to that noise
(`cube_fell` 0.15/episode); closing the last 30° is not. So every intervention on the reward
side and every intervention on the contact side lands on the same plateau, because none of them
change the thing that is destroying the terminal signal.

**Run 2 is not evidence either way.** It lowered `init_std` 1.0 → 0.5 and `entropy_coef`
0.01 → 0.002, which looks like this experiment — but it changed four things at once, on the
pre-fix broken model, for 1000 iterations, with the shaped rewards. It cannot be read as a test
of exploration noise.

---

## 21. `entropy-1e-3` — the noise-floor probe (running)

Launched 2026-09-02 11:49. Run 14's exact configuration with **`--entropy-coef 0.001` as the
only change** (0.01 → 0.001, confirmed as the sole diff in `params/agent.yaml`): 8192 envs,
seed 42, condim 3, success weight back to 100, 1500 iterations. Shorter budget than run 14
because run 14's own curve is flat from ~1200 on, so 1500 is enough to see the effect or its
absence.

**Pre-registered reading:**

- If `Mean action std` falls below ~1.0 **and** `orientation_error` goes under 0.7 with
  `consecutive_success` rising above the 0.03–0.05 band, the exploration noise floor was the
  binding constraint, and the whole next phase of work is the policy distribution rather than
  the task.
- If `Mean action std` falls and `orientation_error` does **not** improve, the hypothesis is
  eliminated — cleanly, and for about 70 minutes of GPU. That is the cheapest disconfirmation
  available and is the reason this run is first in the queue.
- Ambiguous outcome to watch for: std falls but the policy also stops exploring early and
  `cube_fell` rises. That would mean 0.001 is too low rather than that noise was not the
  problem, and the follow-up is a sweep rather than an abandonment.

---

## Where this leaves us

**Established (on the fixed model):**
- The hand learns to hold the cube within ~250 iterations in every configuration, and
  `cube_fell` drops below 0.2 per episode. Grip is not the problem.
- Longer training is not the limiting factor — run 14 was flat over its final 1250 iterations,
  and run 8 was flat over 3300.
- Constraint-buffer overflow *was* costing real reward (~22 points) and is now fixed.
- **Torque is not the explanation** — the LeapHand that reaches 375 uses the identical
  `actuatorfrcrange ±0.2196`.
- **The hand is not immobile.** Run 14's policy performs ~100° of reorientation in the first
  ~15 s and then stalls at 20–50° of error, 0/16 episodes reaching the 5.7° threshold.
- **The stall is the tip-over component**: 88% of the spin error is closed, 64% of the tip
  error, and 96% of what is left at the best moment is tip-over.
- **`condim=3` does make torsional and rolling friction inert**, and turning them on is the
  largest effect measured on this project *on the rotate tasks* — rotate_x goes 0.28 → 2.22
  rad/s at 7.5× fewer drops.
- **But the contact model is not the reorient bottleneck.** Run 19 puts `condim=6` on run 14's
  exact config and lands on run 14's curve: 0.0374 vs 0.0349 goals/episode, 0.896 vs 0.870 rad
  of error, at the matched iteration.
- **The success rate was never zero and never moved.** `goals_reached` was pinned at 0.0 by a
  counter bug in runs 12–19, but `consecutive_success` was always correct and reads
  **0.03–0.05 goals per episode, flat**, across two contact models, a 20× success weight,
  fine/progress reward shaping, and 3000–10000 iterations.
- **A 20× success weight makes things worse**, not better (run 20): error 0.74 → 1.10, action
  std diverging to 5.95, reward falling through the run, success rate unchanged.
- We are at or slightly above the supervisor's own post-fix results for this config, and have
  run ~5× longer than he did.

**The current diagnosis:** an **exploration noise floor**. Converged action std 2.79 × scale
0.5 is 1.40 rad of commanded joint delta per 50 ms step, so ~47% of sampled actions saturate
the ctrl limits; `entropy_coef × H` is 0.391 against a surrogate loss of 0.004; and rsl_rl's
Gaussian is unbounded, so there is no std at which the entropy pressure stops. PPO optimises
the noisy policy's return, under which the terminal manoeuvre and a coarse hold are
indistinguishable. Run 21 tests it directly. **The tanh-squashed-Normal comparison against brax
is unverified and must be confirmed before it is presented.**

**Retracted along the way:** "friction is not the bottleneck" (runs 5–7 varied parameters
`condim=3` ignores); "the policy holds the cube and never turns it" (it turns it, then
stalls); "the hand can only tip the cube by dropping it" (true at condim 3 only); **"the
bottleneck was the contact model"** (true for the rotate tasks, false for reorient — run 19);
**"`goals_reached` is 0"** (a counter bug in runs 12–19; the real rate was always 0.03–0.05).

**Candidate next steps, in order:**

1. **Run 21 (`entropy-1e-3`).** Running. Pre-registered reading above. Either it moves
   `orientation_error` and the success rate together, or the noise-floor hypothesis is dead for
   ~70 minutes of GPU.
2. **If it works: the exploration / policy-distribution line.** In rough order of expected
   value — a **tanh-squashed (bounded) action distribution** so the entropy bonus has a finite
   optimum, which is also the parity fix against playground if the unverified claim above holds;
   **`init_std`** (currently 1.0, and the policy climbs *away* from it); **action scale**
   (0.5 with `clip_to_ctrl_limits` is what turns std 2.79 into bang-bang); and **control rate**
   (50 ms with `ema_alpha=1.0`, no smoothing at all). These are all the same knob seen from
   different sides and should be varied one at a time.
3. **A kept deterministic-eval script.** Being written now. Terminal precision — min error
   reached, fraction of episodes inside 5.7°, mean-action rollouts — is the quantity that
   actually distinguishes these runs, and it has twice been measured with throwaway probes that
   were not kept (the interlude after run 15). It should be a per-run artifact, not an
   archaeology exercise.
4. **Deprioritised, explicitly:** `condim=6` on reorient (run 19 answered it); resuming run 19
   to 3000 (confirmation, not information); the goal curriculum (predates every fix, and its
   promotion criterion needs re-checking against the fixed counter before it is trusted); the
   friction sweep and `condim=4` ablation (they tune a knob that run 19 shows does not bind on
   this task); extending rotate_y to convergence.

**Fingertip geometry is deprioritised.** It was the leading suspect after run 14 — flat 4×4
sensor pads where the plain LeapHand has rounded tips, plus Hamid's note that *the cube stops
rolling when it hits the pads*. Runs 15 and 18 both reach 2.2 rad/s on those same pads, so
the pads permit both twisting and tipping.

**Open question for the supervisor:** `condim=6` is a deliberate deviation from his MJX scene
and from playground, both of which use 3. Soft sensor pads arguably justify torsional
friction physically, but it changes the contact model that everything else was matched to.
**Now moot for reorient** — run 19 shows it changes nothing there, and reorient runs stay at
condim 3. It still matters for the rotate tasks, where the effect is 7.8×, and for task 2 if
tactile data is collected under it.

**Infrastructure notes:**
- `scripts/train.py` takes `--resume-from <checkpoint.pt>` (continues in the checkpoint's own
  directory, `--max-iterations` read as a *total*), `--cube-condim`, `--fingertip-friction` and
  `--entropy-coef`.
- **`scripts/supervise_run.sh` had an off-by-one.** rsl_rl's last checkpoint of an
  N-iteration run is `model_(N-1).pt`, so the `-ge MAX_ITERS` completion check never fired and
  the supervisor relaunched the *finished* run 20 five times before its no-progress guard
  tripped. Fixed. `RUN_NAME`, `MAX_ITERS` and `EXTRA_ARGS` now come from the environment, and
  `EXTRA_ARGS` is applied on resume as well as on the initial launch — previously a resumed run
  silently dropped its flags.
- The host reboots uncleanly every few hours; it killed run 15 at 1789/3000 and run 19 at
  1208/3000. The supervisor script plus a `@reboot` cron hook auto-resume from the newest
  checkpoint.
- **Redirect training runs to `logs/console/<name>_<timestamp>.log`.** Runs 16–18 were
  launched without it, which is why run 18 stopping 73 iterations early has no explanation.
- The two rollout probes behind the interlude were throwaway scripts and were not kept — see
  next step 3, which exists to stop this recurring.
- The hyperparameter sweep (`hyperparameter_search/run_sweep.py`, grid over cube size /
  friction / seed) is written but **has never been run** — no `best_run.yaml` exists. Given
  that the friction ablation turned out to be measuring nothing, re-scope it before using it.

**Not started:** the flex/touch half of task 1 — submodule bump to the flex model, touch
observations behind a flag. The flex model is a large step up in cost (nv 1120 vs 16,
neq 386) and needs `njmax`/`nconmax` re-sized again — and now `condim` decided for the
taxel contacts too.
