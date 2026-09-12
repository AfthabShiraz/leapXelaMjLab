# Task 1 — PPO cube reorientation (no touch): what we tried and what happened

Environment: `Mjlab-LeapXELA-Cube-Reorient` (mjlab manager-based API, MuJoCo Warp, RSL-RL PPO).
All runs on a single GPU, `--num-envs 4096`, logs under
`logs/rsl_rl/leap_xela_cube_reorient/<timestamp>_<run-name>/`.

Status as of 2026-09-02 (evening): **no hypothesis is currently in the lead.** The
per-episode success rate has now been measured at **0.03-0.05 goals per episode, flat**,
across two contact models, a 20x success weight, fine/progress reward shaping, an order of
magnitude of exploration noise, and 3000-10000 iterations (see run 19's counter section,
which dates this measurement back to run 14). Run 21 cut `entropy_coef` 10x, which collapsed
`Mean action std` from 2.79 to 0.39 — the intervention did exactly what it was designed to
do — and the task did not move. The exploration-noise-floor diagnosis this file led with
earlier in the day is **eliminated**; see run 21.

Earlier framings, kept because runs 1-21 were read through them:

> **Retracted 2026-09-02 (evening).** *"The leading candidate is an exploration noise floor in
> the policy distribution"* (status as of 2026-09-02, morning), from run 14's converged
> `Mean action std` of 2.79 and run 20's divergence to 5.95. Run 21 took the std to 0.39 by
> the most direct route available and the task was unchanged — slightly worse on error,
> identical on success rate, same 30 deg tip-over stall in the deterministic eval. See run 21
> and the diagnosis section, which is kept as history.

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
| 21 | `entropy-1e-3` | 1499 / 1500 | 99 min | Run 14 + `entropy_coef` 0.01 → 0.001, sole change. **Hypothesis eliminated** — std 2.79 → 0.39, error and success rate unmoved |
| 22 | `fixed-goal` | 1499 / 1500 | 103 min | Run 14 + `--goal-drift False --goal-resample-on-success False`, goal pinned all episode. **Much worse** — deterministic best error 77° vs run 14's 33° under the identical pinned-goal measurement |
| 23 | `orientation-fine` | 650 / 1500 (host reboot, never resumed) | 44 min | Run 14 + `--orientation-fine True`, the long-tail reward term. **Worse** — 64° vs 37° against run 14 *at matched iteration 600* |
| 24 | `reference-seed7` | 1499 / 1500 | 103 min | Run 14, `seed` 42 → 7, sole change. **Reproduces run 14** — 34° vs 33°. Numbered after 22–23 to keep the "runs 22–23" pairing that `train.py` cites; it ran chronologically between them |
| 25 | *contact priority* (no training) | — | — | Cube geom given `priority=1`. The cube's sliding friction was previously discarded at the fingertips (max mixing, 0.5 > 0.3), so runs 5–7 and `dr_cube_friction` varied nothing. **The grasp was not thereby stuck low** — `dr_fingertip_friction` held it at U(0.5, 1.0); see the 2026-09-06 correction. Contact friction is now the cube's, and palm angle and sliding friction are hyperparameters |
| 26 | `friction08-prio1` | 1499 / 1500 | 103 min | Run 14 + `--cube-priority 1 --cube-friction-sliding 0.8`, the friction payoff test. **Null on the headline** — best error 26.3° → 25.1°, inside run 24's seed noise. But the error *reallocated*: tip closure 66.9 → 73.1%, spin closure 93.5 → 76.7% |
| — | *eval measurement bug found* | — | — | `eval_policy.py` scored the post-drift-kick goal instead of the goal a step was judged by; `reached_threshold` was 0 by construction in every reference-env eval. Fixed; runs 14–30 and `obsnoise` need re-scoring — see the Interlude |
| 31 | `thresh-04` | 1499 / 1500 | ~1h45m | Run 14 + `--success-threshold 0.4` (moves both the command and success-bonus thresholds), `--cube-priority 0`. **Strong regression** — best error 72.2° vs run 14's 26.3°, 0/32 at 0.1 rad; never reaches the iteration-300 breakthrough |
| — | *breakthrough-curve analysis* | — | — | Every working run drops ~90° → 50–60° error between iteration 300–600; every from-scratch reward/goal edit (runs 20, 22, 23, 31) dies before reaching it. Motivates warm-starting past the breakthrough before changing the reward |
| 32 | `warm-inv-pin` | 1650 / 3000 (stopped, std runaway) | — | First warm start: `--init-from` run 14 `model_1500` + `cube_orientation_inverse` kernel (weight 8.93) + goal pinned + `--cube-priority 0`, default entropy. Training error held, but action std rose 2.81 → 4.03 with no plateau; stopped on a pre-registered std > 4 rule |
| 33 | `warm-inv-pin-ent1e3` | 2999 / 3000 | 1h46m | Run 32 + `--entropy-coef 0.001`, warm start from run 14 `model_1500`. **First policy to reach goals under drift** — 31.2% of episodes < 5.7°, 0.36 goals/episode vs control's 1.8% / 0.02; total closure 91.1%, clear of the 77.6–81.4% equilibrium band |
| 34 | `warm-inv-pin-ent1e3-ext` | 4499 / 4500 | 1h43m | `--init-from` run 33 `model_2999`, same config. Still climbing — 57.3% of episodes < 5.7° and 0.77 goals/episode by @4499, median best error down to 5.6–5.7°, no plateau. Continuing to iteration 9000 remotely |

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

## Diagnosis — an exploration noise floor (2026-09-02, since eliminated)

> **Eliminated by run 21, later the same day.** This section is kept because it is what
> motivated run 21, and because its *measurements* are correct — the converged action std
> really is 2.79, the entropy bonus really does dominate the surrogate loss by two orders of
> magnitude, and the Gaussian really is unbounded. What is wrong is the conclusion that this
> was **the** binding constraint. Cutting `entropy_coef` 10× took the std to 0.39 and changed
> nothing about the task. Read what follows as history.

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

## 21. `entropy-1e-3` — the noise-floor probe, answered: no

Launched 2026-09-02 11:49, **completed 1499/1500, rc=0, 1 h 39 m at 78.7 k steps/s.** It ran
as a single uninterrupted segment — `supervise_run.sh` started it fresh and never had to
resume, the first long run on this host in some time that did not have to be pieced together
across a reboot.

Run 14's exact configuration with **`--entropy-coef 0.001` as the only change** — verified
after the fact rather than assumed, by diffing the two runs' saved configs. `params/agent.yaml`
differs only in `entropy_coef` (0.01 → 0.001) and the three bookkeeping fields
`max_iterations`, `save_interval` and `run_name`; `params/env.yaml` differs only by the new
explicit `cube_condim: 3` key, which is run 14's implicit value. 8192 envs, seed 42, condim 3,
success weight back to 100, 1500 iterations — a shorter budget than run 14 because run 14's own
curve is flat from ~1200 on.

**This is the second branch of the pre-registered reading, and the hypothesis is eliminated.**
`Mean action std` fell **2.79 → 0.39**, and fell *monotonically* — 1.00, 0.64, 0.50, 0.43,
0.41, 0.39 at iterations 0/250/500/750/1000/1499 — where run 14 climbs *away* from its
`init_std` of 1.0 (1.01, 1.56, 2.26, 2.67, 2.77, then flat at 2.79 for 2000 iterations). The
knob worked, at the first setting tried, and in the intended direction. `orientation_error` did
not follow it.

Iteration-matched, mean ± sd over the last 100 iterations of each window:

| | run 21 @ 1400–1499 | run 14 @ 1400–1499 | run 14 @ 2900–2999 |
| --- | --- | --- | --- |
| `orientation_error` (rad) | **0.874 ± 0.045** | 0.819 ± 0.032 | 0.755 ± 0.028 |
| `goals_reached` / `consecutive_success` | **0.0312 ± 0.0127** | 0.0298 ± 0.0096 | 0.0330 ± 0.0103 |
| `cube_fell` (per episode) | 0.224 ± 0.136 | 0.167 ± 0.070 | 0.149 ± 0.058 |

At the matched iteration run 21 is *slightly worse* on error, indistinguishable on success
rate, and sits in the same 0.03–0.05 band as every other run in this file. The third,
ambiguous branch — std falls but `cube_fell` rises, meaning 0.001 was simply too low — is not
what happened either: drops are up by a fifth of a drop per episode, within the run-to-run
scatter of runs 14, 19 and 20, and nowhere near a policy that has stopped exploring and started
failing.

**Do not quote the final iteration's `goals_reached` of 0.0754.** It is the last line of the
console log and therefore the number that catches the eye, but the band it sits in has a
standard deviation of 0.013, so 0.0754 is a **>3σ single-iteration spike**, not a doubling.
Run 21's honest success rate is **0.031**. Single-iteration values of this metric have already
been over-read once on this project; the 100-iteration mean is the number to quote.

**Mean reward rose while the task metric did not** — 204.9 ± 5.1 against run 14's 180.9 ± 4.4
at the matched iteration. This is not an improvement. A policy at std 0.39 thrashes far less
than one at 2.79, so it pays much smaller `action_rate` and `energy` penalties and holds the
cube more steadily for the `position` term, all without getting closer to the goal. **Total
reward is not comparable across entropy settings** — the same confound that made run 20's
cross-run reward comparison unreadable, arriving from the opposite direction.

### Deterministic evaluation

Both checkpoints scored with `scripts/eval_policy.py` under identical settings — 32 envs ×
700 steps, seed 7, **mean actions** (the learned std is loaded and deliberately not sampled,
so this measures the policy rather than its noise). Raw numbers in `eval/run21_det.json` and
`eval/run14_det.json`, console output in `eval/AUTO_EVAL_REPORT.txt`.

| median over 32 episodes | run 21 (`model_1499.pt`) | run 14 (`model_2999.pt`) |
| --- | --- | --- |
| minimum error reached | 30.1° | **25.6°** |
| final error | 35.5° | **30.2°** |
| success (< 5.7° at any point) | **0/32** | **0/32** |
| spin `r_z` reduction | 90.5% | 94.2% |
| tip `r_xy` reduction | 62.2% | 66.0% |
| drops in 700 steps | 2 | 0 |

The eval is **not iteration-matched** — run 21's checkpoint is 1500 iterations against run
14's 3000 — so run 21 is not strictly *behind* here. But it is nowhere *ahead*, at one seventh
the action noise, on the one metric chosen precisely because it isolates terminal precision.
And the failure has the same shape: spin ~90% closed, tip-over ~62% closed, residual error
dominated by tip-over. `model_1499.pt` exists in run 14's directory if the properly matched
comparison is wanted later.

**What this rules out**, precisely. Not "exploration does not matter" in general, but the
specific mechanism this file argued for: that a converged std of 2.79 was drowning the fine
terminal manoeuvre, and that a policy able to act precisely would find the goal. Run 21 **is**
that policy — std 0.39, a commanded joint delta of 0.20 rad per 50 ms step instead of 1.40 —
and it reaches the same ~30° and stops. Whatever prevents the last 30° of tip-over, the
policy's own action noise is not it.

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
- **Exploration noise is not the bottleneck either.** Run 21 cuts `entropy_coef` 10×, which
  collapses `Mean action std` 2.79 → 0.39 — a 7× cut in commanded joint delta, 1.40 → 0.20 rad
  per step — and the task does not move: error 0.874 vs 0.819 at the matched iteration, success
  rate 0.031 vs 0.030, and the deterministic eval still stops at 30° with 0/32 successes.
- **The tip-over stall survives every intervention tried so far**, the low-noise policy
  included. Run 21 closes 90% of the spin error and 62% of the tip error — run 14's split, at
  one seventh the action noise.
- We are at or slightly above the supervisor's own post-fix results for this config, and have
  run ~5× longer than he did.

**There is currently no diagnosis.** The exploration-noise-floor account was the leading
candidate for one day and run 21 killed it. The measurements behind it stand — converged action
std 2.79 × scale 0.5 is 1.40 rad of commanded joint delta per 50 ms step, ~47% of sampled
actions saturate the ctrl limits, `entropy_coef × H` is 0.391 against a surrogate loss of 0.004,
and rsl_rl's Gaussian is genuinely unbounded — but at std 0.39 none of that is true any more
and the policy behaves the same. **The tanh-squashed-Normal comparison against brax remains
unverified**, and is now a question about port parity rather than a candidate explanation.

**Retracted along the way:** "friction is not the bottleneck" (runs 5–7 varied parameters
`condim=3` ignores — and, per run 25, a *sliding* friction the fingertips were overriding
anyway, so those three runs were doubly empty); "the policy holds the cube and never turns it" (it turns it, then
stalls); "the hand can only tip the cube by dropping it" (true at condim 3 only); **"the
bottleneck was the contact model"** (true for the rotate tasks, false for reorient — run 19);
**"`goals_reached` is 0"** (a counter bug in runs 12–19; the real rate was always 0.03–0.05);
**"the bottleneck is an exploration noise floor"** (the leading hypothesis for a day — run 21
took the action std from 2.79 to 0.39 and nothing about the task changed).

**Where the previous next-step list stands after run 21:**

1. **Run 21 (`entropy-1e-3`) — done, and negative.** The pre-registered disconfirmation fired,
   for 99 minutes of GPU as budgeted. It is the cheapest thing this project has learned in a
   fortnight, and it closes the line it was testing.
2. **The exploration / policy-distribution line — motivation gone.** It was listed as "if it
   works", explicitly conditional on run 21, and run 21 did not work. The headline proposal was
   a **tanh-squashed (bounded) action distribution** so the entropy bonus has a finite optimum
   — but run 21 has already *delivered the low-std regime a bounded distribution would produce*,
   by a cruder route, and it reproduced run 14's behaviour instead of beating it. Implementing
   tanh now should be expected to land on run 21, not on a fix. The brax parity question is
   still worth settling as a **fact about the port** (and the claim is still unverified), but it
   is no longer a candidate explanation. **`init_std`**, **action scale** and **control rate**
   are the same knob seen from other sides and inherit the same downgrade.
3. **A kept deterministic-eval script — done.** `scripts/eval_policy.py` (mean-action rollouts,
   per-episode min/final error, world-frame spin/tip decomposition, drop counts, JSON out) plus
   `scripts/auto_eval_run21.sh`, which waited for the checkpoint, waited for the trainer to free
   the GPU, and scored both runs unattended under identical settings. Artifacts in `eval/`.
   Terminal precision is now a per-run artifact rather than an archaeology exercise — run 21 is
   the first run scored this way at launch time rather than in hindsight.
4. **Deprioritised, explicitly:** `condim=6` on reorient (run 19 answered it); resuming run 19
   to 3000 (confirmation, not information); the goal curriculum (predates every fix, and its
   promotion criterion needs re-checking against the fixed counter before it is trusted); the
   friction sweep and `condim=4` ablation (they tune a knob that run 19 shows does not bind on
   this task); extending rotate_y to convergence.

**What is actually open.** Four families of intervention — contact model, reward shaping and
weighting, training budget, and now exploration noise — have each been varied by a large factor,
and the per-episode success rate has not left 0.03–0.05. No replacement hypothesis is being
put forward here, and this file should not pretend one is ready. Whatever the next candidate
turns out to be, the evidence constrains it to explain a specific thing: the policy closes ~90%
of the *spin* error and stalls with ~30° of *tip-over* left, at std 2.79 and at std 0.39 alike,
without dropping the cube.

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
- **`scripts/supervise_run.sh` still defaults to run 21** — `RUN_NAME=entropy-1e-3`,
  `MAX_ITERS=1500`, `EXTRA_ARGS=--entropy-coef 0.001`. These are environment-overridable, but
  the `@reboot` cron hook passes nothing, so **the defaults must be edited before the next run
  is launched**, not after. Both `@reboot` hooks (`supervise_run.sh` and `auto_eval_run21.sh`)
  are still armed and both are idempotent — the supervisor's completion check now fires
  correctly after the off-by-one fix, and the eval script returns early on `ALL DONE` — so a
  reboot today restarts nothing.
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

---

## 22–24. The two reward interventions, and a seed replicate (2026-09-02, scored 2026-09-05)

Three runs finished on 2026-09-02 and sat unwritten for three days. All three are run 14 with
exactly one change, 8192 envs, 1500-iteration budget.

| # | Run | Change | Iters |
| --- | --- | --- | --- |
| 22 | `fixed-goal` | `--goal-drift False --goal-resample-on-success False` | 1499 / 1500 |
| 23 | `orientation-fine` | `--orientation-fine True` | 650 / 1500 — host reboot |
| 24 | `reference-seed7` | `seed` 42 → 7 | 1499 / 1500 |

### The training logs could not answer the question

| run | reward | `orientation_error` | `consecutive_success` |
| --- | --- | --- | --- |
| 14 `reference-v3-eulerdamp` (2999 it) | 187 | 0.76 | 0.033 |
| 22 `fixed-goal` (1499 it) | 97 | 1.61 | 0.625 |
| 23 `orientation-fine` (650 it) | 145 | 1.46 | 0.010 |
| 24 `reference-seed7` (1499 it) | 182 | 0.85 | 0.032 |

Run 22's `consecutive_success` of 0.625 is **not** a 19× improvement over run 14, and the
reward column is not comparable either. With the goal pinned, nothing resamples on success,
so the counter stops counting *goals hit* and starts counting *steps spent inside the
threshold* — a different quantity with the same name. Run 23's reward is lower partly because
it has an extra reward term, and its budget is 650 iterations against run 14's 2999.

Every number in that table is a different unit. That is what `eval_policy.py` exists for.

### Deterministic scoring, and the flag that had to be added first

`eval_policy.py` reconstructs the env a checkpoint trained in and rolls out the **mean**
action. Before it could score these runs it needed `--cube-priority`, for the same reason it
already needed `--cube-condim`:

`get_cube_spec` now defaults to `priority=1` (run 25, added 2026-09-02 21:26). Runs 14, 22 and
24 have **no `cube_priority` key** in their `params/env.yaml` `build_kwargs` — they predate the
flag and trained under MuJoCo's element-wise-max mixing, where the fingertips' 0.5–1.0 beat the
cube's 0.3. Scoring them at today's default would hand them a *weaker grip than they ever
trained with* and flatter run 23 for free. They are scored at `--cube-priority 0`; run 23 at 1.

Rule going forward: **if `build_kwargs` has no `cube_priority` key, pass `--cube-priority 0`.**

64 envs × 700 steps, seed 42. `best` is the median per-episode minimum error reached; the goal
runs away after a success, so `best` and `final` are genuinely different quantities.

**Native — each checkpoint in its own training env:**

| eval | start° | best° | final° | succ% | total ↓ | spin ↓ | tip ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14 @2999 | 134.1 | **31.3** | 38.0 | 0.0 | 74.3% | 88.6% | 70.1% |
| 24 seed7 @1499 | 134.1 | 37.4 | 41.3 | 0.0 | 67.3% | 85.6% | 59.1% |
| 22 fixed-goal @1499 | 132.6 | 77.1 | 78.8 | 0.0 | 31.6% | 67.2% | 12.0% |
| 23 orient-fine @600 | 134.1 | 77.3 | 86.1 | 0.0 | 37.8% | 71.7% | 23.4% |

**Pinned goal — drift off for everyone, native physics.** This is the comparison that matters:
the drift kick is the thing that made `consecutive_success` mean two different things, so
removing it for all four makes "how much error does it close" one quantity.

| eval | start° | best° | final° | succ% | total ↓ | spin ↓ | tip ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14 @2999 | 132.3 | **32.9** | 37.7 | 1.6 | 72.7% | 87.9% | 62.0% |
| 24 seed7 @1499 | 132.3 | 34.2 | 36.0 | 1.6 | 70.6% | 84.8% | 58.4% |
| 22 fixed-goal @1499 | 132.6 | 77.1 | 78.8 | 0.0 | 31.6% | 67.2% | 12.0% |
| 23 orient-fine @600 | 132.3 | 64.2 | 75.8 | 0.0 | 38.4% | 70.3% | 24.3% |

**Equal budget — everything at iteration 600, pinned.** Run 23 only ever reached 650, so the
rows above partly measure training budget rather than the reward change:

| eval | start° | best° | final° | succ% | total ↓ | spin ↓ | tip ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14 @600 | 132.6 | **37.3** | 41.4 | 1.5 | 69.9% | 85.7% | 51.9% |
| 24 seed7 @600 | 132.3 | 38.5 | 41.1 | 3.1 | 70.0% | 84.7% | 52.4% |
| 23 orient-fine @600 | 132.3 | 64.2 | 75.8 | 0.0 | 38.4% | 70.3% | 24.3% |

### What this says

**Run 22 is much worse, not better.** 77° against run 14's 33° under the identical
measurement. The 0.625 was an artifact of the renamed counter, exactly as suspected. Pinning
the goal does not free the policy to be precise — it removes a curriculum. With drift on, a
success moves the goal *somewhere nearby*, which is a reachable next target; with the goal
pinned the policy gets one hard random target per episode and never sees the near-goal regime
at all. Its tip-over reduction collapses to 12%, the worst number in the table.

**Run 23 is worse too, and it is not a budget artifact.** At matched iteration 600 it closes
64° against run 14's 37°. The long-tail term is supposed to be the only reward gradient below
0.2 rad, but the policy never gets below 0.2 rad, so in practice it is a reweighting of the
region the policy actually occupies — and it reweights it badly.

**Run 24 reproduces run 14.** 34.2° vs 32.9° pinned, 37.4° vs 37.3° at matched iteration 600.
The run-14 result is not a seed fluke, and seed noise on this metric is ~1–2°, which sets the
resolution of every comparison here: **a config that moves `best` by less than ~3° has not
been shown to do anything.**

**Success is nonzero for the first time, barely.** 1.5–3.1% of episodes touch 5.7° once the
goal is pinned at evaluation time, against a flat 0.0% native. That is 1–2 episodes out of 64,
so it is a hint and not a result — but it is consistent with the drift kick being what erases
successes rather than the policy being unable to reach them.

### The `cube_priority` confound, measured rather than assumed

Run 23 differs from run 14 in *two* ways — the reward term and `cube_priority` 0 → 1 — so its
deficit was not attributable on the tables above. Scoring both policies at both priorities
separates them (all @600, pinned):

| policy | priority 0 | priority 1 |
| --- | --- | --- |
| run 14 baseline | **37.3°** (trained here) | 35.7° |
| run 23 orientation-fine | 61.3° | **64.2°** (trained here) |

`cube_priority` moves best-error by 2–3° in either direction — inside seed noise. Policy
identity moves it by ~26°. Run 23's deficit is the reward change, not the contact change.

The useful side-result: **run 14's policy is indifferent to the priority switch** (37.3 → 35.7,
if anything slightly better). Turning on priority 1 — which is what makes
`--cube-friction-sliding` reach the grasp at all — does not damage the existing policy. The
friction sweep is not starting from a broken baseline.

### What it means for the reward-bottleneck hypothesis

The standing hypothesis was that the ~30° stall is the reward's optimum, and no contact model
will move it. Run 22 was the direct test of that — and it lost, badly. Run 23, the other reward
intervention, also lost. Two independent edits to the reward both made the policy worse than
leaving it alone.

That does not prove the reward is fine. It does mean the two specific edits available were
both wrong, and that "fix the reward" is not a cheap next step with a known direction. The
contact/geometry sweep is no longer the thing to argue against.

### Housekeeping found while scoring these

- Run 23's `model_650.pt` is **0 bytes** — the 2026-09-02 22:11 reboot landed mid-write.
  `supervise_run.sh` picked the newest checkpoint by iteration number without checking it was
  readable, so every relaunch resumed from the truncated file and died in `runner.load` before
  iteration one: **65 dead segments between 2026-09-03 08:30 and 2026-09-05 23:24**, across 13
  reboots (the 5-failure guard fires, the `@reboot` hook restarts it, repeat). The box trained
  nothing for three days. `latest_ckpt` now skips zero-length files, and a segment that dies
  inside `runner.load` without advancing quarantines its checkpoint as `.corrupt` and falls
  back one save interval instead of burning the whole retry budget.
- `eval_policy.py`'s JSON `config` block and its printed header were both hand-maintained lists
  that had already silently dropped the goal-drift flags. A result measured with the goal
  pinned was indistinguishable from one measured with it drifting. Both are now derived from
  `EvalConfig` itself.

---

## 25. Contact priority — the cube's friction was never reaching the grasp (2026-09-05)

Hamid, by email: *"Cube has the highest priority. Use the policy u trained. Only change
sliding friction to see if the sliding goes away. Make sliding friction a hyperparameter.
Make palm angle a hyperparameter."* He is talking about MuJoCo's contact-parameter
priority — the solver snaps a contact's parameters to one geom or the other, and he
wanted that to be the cube.

He was right, and the consequence is larger than a tuning detail: **runs 5–7's "friction
ablation" was the second thing on this project to vary nothing, and `dr_cube_friction` has
been inert since day one.**

### The mechanism

MuJoCo mixes a *dynamically generated* contact's friction as the **element-wise max** of
the two geoms' — unless one geom has a higher `priority`, in which case that geom dictates
condim, friction, solref and solimp outright. Measured on the compiled reference model,
every geom had `priority = 0`:

| geoms | count | friction[0] | solref[0] |
| --- | --- | --- | --- |
| hand pads (`palm_collision_*`, `*_uspa4*`, `*_collision_*`) | 32 | 0.200 | 0.0001 |
| fingertips (`th/if/mf/rf_tip`) | 4 | 0.500 | 0.02 |
| cube (`cube/cube`) | 1 | 0.300 | 0.02 |
| terrain | 1 | 1.000 | 0.02 |

So a cube-fingertip contact resolved to `max(0.3, 0.5) = 0.5` and a cube-pad contact to
`max(0.3, 0.2) = 0.3`. **The cube's sliding friction was discarded at the four geoms that
do the grasping**, and anything set below 0.5 there was a no-op. Worse, the two DR events
were sampling into each other's shadow: `dr_cube_friction` draws 0.1–0.5 while
`dr_fingertip_friction` draws 0.5–1.0, so the fingertip range won the max on essentially
every draw and the cube's randomization did nothing at all.

### Correction (2026-09-06) — the grasp was never stuck *low*

Everything above is about which geom's number reaches the contact, and it is right. The
inference quietly drawn from it — that the grasp was therefore running at some low, wrong
friction that priority 1 would raise — is **wrong**, and it made the friction axis look far
more promising than it is. Found while designing run 26.

**Unchanged:** `dr_cube_friction` was inert, the runs 5–7 ablation varied nothing, and the
cube's own sliding friction never reached a fingertip contact.

**Corrected:** the term that *did* control the grasp is `dr_fingertip_friction` — range
(0.5, 1.0), `axes=(0,)`, `mode="reset"`, geoms `th_tip`/`if_tip`/`mf_tip`/`rf_tip`. It
resamples every reset, and at priority 0 the contact takes `max(cube, tip)` where the tip
draw is always ≥ 0.5 ≥ the cube draw. So run 14's **grasp friction was U(0.5, 1.0), mean
≈ 0.75** — randomized and high, dictated by the fingertip term rather than the cube one.

That reverses the sign of the opportunity. Going to `priority 1` at μ=0.8:

| contact | run 14 (effective) | at μ=0.8 | change |
| --- | --- | --- | --- |
| cube–fingertip | U(0.5, 1.0), mean 0.75 | 0.80 pinned | +7% on the mean, and the randomization is **removed** |
| cube–pad | U(0.2, 0.5), mean ≈ 0.35 | 0.80 pinned | more than double |

So the intervention is nearly a no-op at the fingertips and large at the palm pads — the
opposite of where it was expected to matter. It also silently converts a randomized
parameter into a fixed one, which is a robustness change, not only a friction change.

**The probe table below overstates the available headroom, and here is why.** Its
`priority = 0` rows read `@tip 0.50` for every request because the probe holds fingertip
friction at its nominal 0.5 with DR off — that is the **bottom** of run 14's training
distribution, not its mean. And the headline "83.8 → 26.4 mm/s, 3.2×" spans μ 0.05 → 1.2, a
range run 14 never occupied. Measured against run 14's actual distribution the honest delta
is **34–39 → 28.7 mm/s**, roughly 25%, with maybe another 10% left before wedging starts.
Read the sweep below as a characterisation of the contact model, not as headroom.

### Measured, not argued

`scripts/friction_probe.py` replays run 14's policy (`reference-v3-eulerdamp/model_2999.pt`,
unchanged, not retrained) across a sweep of `cube_friction_sliding`, mirroring the sim state
into a host `MjData` each control step and reading the contacts directly. `@tip` / `@pad` are
the median resolved contact friction; `slip` is the true tangential relative speed at the
contact, `||(I - nn^T)(v_cube - v_hand)||`, from body Jacobians.

**`cube_priority = 0` (every run 1–21):**

| requested | @tip | @pad | slip_mean | slip_tip |
| --- | --- | --- | --- | --- |
| 0.05 | **0.50** | 0.20 | 35.7 mm/s | 39.2 mm/s |
| 0.10 | **0.50** | 0.20 | 35.7 | 39.4 |
| 0.20 | **0.50** | 0.20 | 36.5 | 38.9 |
| 0.30 | **0.50** | 0.30 | 25.8 | 34.3 |
| 0.50 | **0.50** | 0.50 | 17.1 | 36.1 |

Five different requested values, one fingertip friction. The fingertip slip is flat at
34–39 mm/s across all of them.

**`cube_priority = 1` (new default):**

| requested | @tip | @pad | slip_mean | slip_tip | pen_p95 |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.05 | 0.05 | 68.2 mm/s | 83.8 mm/s | 0.87 mm |
| 0.10 | 0.10 | 0.10 | 63.8 | 75.2 | 0.87 |
| 0.20 | 0.20 | 0.20 | 51.9 | 64.5 | 0.94 |
| 0.30 | 0.30 | 0.30 | 36.9 | 52.9 | 0.89 |
| 0.50 | 0.50 | 0.50 | 18.5 | 38.8 | 0.67 |
| 0.80 | 0.80 | 0.80 | 15.6 | 29.0 | 0.74 |
| 1.20 | 1.20 | 1.20 | 13.4 | 26.4 | 1.80 |
| 2.00 | 2.00 | 2.00 | 11.6 | 27.7 | 3.70 |

The contact now tracks the request exactly, and **the sliding does go away**: fingertip slip
falls 83.8 → 26.4 mm/s, a 3.2× reduction, monotone to about μ=1.2. Raw data in
`eval/friction/sweep_priority{0,1}.json`.

Answering Hamid's question directly: yes, raising sliding friction removes most of the slip
— but it saturates around μ ≈ 1.0–1.2, and past that `pen_p95` climbs (0.7 → 3.9 mm) and
`pads_per_step` rises, i.e. the cube stops sliding and starts **wedging into the pads**
instead. μ ≈ 0.6–1.0 is the usable band. Note this is a *replay* result: the policy never
trained against these contacts, so it is a measurement of the contact model, not a
prediction of what retraining will score.

### Slope test — the friction is now the friction

`scripts/slope_test.py` tilts a ramp from flat and records the angle each cube breaks away
at. A rigid block slides iff `tan(theta) > mu`, so the release angle *is* a readout of the
friction the solver applies. The ramp is deliberately given friction 1.0 (the terrain's
value), so it doubles as a priority test. Release detection is on ramp-frame **speed**, not
displacement: a stuck cube creeps sub-mm/s and that creep integrates, so a displacement
threshold trips earlier the slower you tilt (μ=0.6 read 27.2° at 4°/s and 21.9° at 2°/s
against a true 31.0°).

| μ | atan(μ) | released, priority 1 | released, priority 0 |
| --- | --- | --- | --- |
| 0.05 | 2.9° | 4.1° | 45.5° |
| 0.10 | 5.7° | 6.8° | 45.5° |
| 0.20 | 11.3° | 12.1° | 45.5° |
| 0.30 | 16.7° | 17.2° | 45.5° |
| 0.50 | 26.6° | 27.3° | 45.5° |
| 0.80 | 38.7° | 39.5° | 45.5° |
| 1.20 | 50.2° | 45.3°* | 45.3°* |

At priority 1 every value lands within +0.5–1.2° of `atan(μ)`. At priority 0 **every cube
sticks to 45.5°** — the ramp's 1.0 masks all of them. (*A cube topples rather than slides
once `tan(theta) > 1`, so 45° is a hard ceiling and μ ≥ 1.0 cannot be read this way. That
also makes μ ≈ 1.0 the largest sliding friction with any physical meaning for this shape.)

Videos: `eval/videos/slope_tilt_priority{0,1}.mp4`.

### What changed in the code

- `robots/cube.py` — cube geom gains `priority` (default **1**), plus explicit `solref` /
  `solimp`. The solver params are pinned to MuJoCo's defaults, which is what the cube already
  resolved to, so cube-fingertip contacts are unchanged except for friction. Cube-pad
  contacts do change: at equal priority solref was a solmix average of 0.0001 and 0.02
  (= 0.01005) and is now the cube's 0.02, about 2× softer. That is the one side effect.
- `robots/leap_xela.py` — `palm_euler` applies the hand-base angle at spec-load time, so it
  is a normal kwarg instead of a regenerated MJCF. Verified bit-identical to the baked
  quats for both 1.88 and 1.92, which means `Box_palm192` is now just
  `Box` + `palm_euler=(0, 1.92, -1.57)`.
- `tasks/*/config/env_cfg.py` — `cube_priority` and `palm_euler` threaded through both
  factories and into `build_kwargs`.
- `scripts/train.py` — `--cube-priority`, `--palm-euler`; both logged to WandB. The
  `fingertip_friction` docstring was inverted by this change and has been corrected: at
  priority ≥ 1 that flag no longer touches cube-fingertip contacts at all.
- `scripts/render.py` — `--view visual|collision|both`, `--show-contacts`, `--transparent`,
  and `--cube-priority` / `--cube-friction-sliding` / `--palm-euler` overrides. Palm angle
  was the one of the four asks still missing here (2026-09-06): the renderer could vary the
  contact model but not the hand pose, so a replay could not be given the palm angle
  explicitly. With it added, all four are reachable from `train.py`, `eval_policy.py`,
  `friction_probe.py` and `render.py` alike. The reference value is
  `--palm-euler 0 1.92 -1.57`: 1.92 is the only palm angle with evidence behind it (1.88
  plateaus, 1.92 takes off) and is what runs 12–26 trained at, so passing it explicitly is a
  no-op that puts the angle in the log instead of hiding it inside a `finger_tip_type` string.
- `scripts/friction_probe.py`, `scripts/slope_test.py` — new.
- `hyperparameter_search/` — `cube_priority` and `palm_pitch` are grid axes; the config is
  re-scoped to an 8-run, ~13 h ranking grid (4 frictions × 2 palm angles), replacing a grid
  that would have swept a parameter the solver ignores.

### Reproducibility warning

**`cube_priority` defaults to 1, which changes the physics of every existing task.** Runs
1–21 were all trained at the equivalent of 0. Replaying an old checkpoint through the stock
config now gives it a *more slippery* grasp than it ever trained against (fingertip friction
0.5 → 0.3). Pass `--cube-priority 0` to `render.py` / `friction_probe.py`, or
`cube_priority: 0` in a sweep config, for a faithful replay.

### Collision-model replay

`--view collision` hides group 2 (the decorative meshes, `contype=0`) and draws group 3, the
geometry the solver actually uses, with `--show-contacts` overlaying contact points and
force arrows. This is the view Hamid asked for, to check whether the cube wedges into the
gaps between the discrete sensor pads.

- `eval/videos/policy_collision_astrained.mp4` — run 14's policy at priority 0, i.e. the
  physics it trained in. 600 steps, 0 drops.
- `eval/videos/policy_visual_astrained.mp4` — same rollout, normal rendering.
- `eval/videos/policy_collision_priority1_fric{0.3,1.2}.mp4` — the same policy under the new
  contact model at low and high sliding friction.
- `eval/videos/best_policy_collision_prio1_fric0.8_palm192.mp4`,
  `best_policy_visual_prio1_fric0.8_palm192.mp4` (2026-09-06) — run 14's `model_2999` at the
  settled parameters: `--cube-priority 1 --cube-friction-sliding 0.8 --palm-euler 0 1.92
  -1.57`, collision and visual views of the same seed-42 rollout.
- `eval/videos/best_policy_collision_closeup.mp4` — the same, 1280×960 at camera distance
  0.22, close enough to see individual pad boxes.

What the numbers say about wedging: `pads_per_step` sits at 3.4–4.9 distinct hand geoms in
contact and `pen_p95` at 0.7–0.9 mm through the usable friction band, rising to 3.7 mm only
at μ=2.0. So at sane friction the cube rides on the pads rather than sinking between them;
the wedging regime is something high friction *creates*, not something it fixes.

**There are no gaps to wedge into.** Measured on the compiled `Box_palm192` hand: 36
collision geoms, 4 fingertips and 32 pads, pad half-extents ~15 × 11 × 14 mm at a median
nearest-neighbour centre spacing of **16.0 mm**. Boxes that size at that spacing overlap —
the implied surface gap is about **−6 mm**. The collision shell is continuous, not a row of
separated islands, so the premise of the question does not hold for this model. Against
that, `pen_p95` at μ=0.8 is 0.68 mm, **1.9% of the cube's 35 mm half-width**: ordinary
soft-contact compliance. The gap figure is approximate — it takes each pad's smallest
half-extent and ignores per-box orientation — but a 6 mm overlap is far too large for
orientation to flip the sign.

### Still open (updated 2026-09-12, revised after runs 35–37: the recipe is converged)

- **The contact and actuation model is closed. Five nulls.** `condim 6` (run 19), friction
  (run 26), action saturation (runs 27–28), cube size (runs 29–30), and the rolling-resistance
  probe. Each was tested single-variable, several had their mechanism demonstrably fire
  (`|a|` fell 6×, slip fell 3.2×, clearance was restored), and none moved best error outside
  the ~3° floor or produced a single success. **Stop looking for the answer in the physics.**
- **Cube size: tested, and null across a 14% span.** Runs 29–30 at half-size 0.0325 and
  0.0300 give best error 27.1° and 24.8° against run 14's 26.3°, all 0/32. The clearance
  measurement that motivated them is still correct and still worth having (the pads cost
  ~5.7 mm of cavity, 7.9 mm on the tip axis) — it just does not explain the stall.
- **Palm angle: CLOSED, not open.** The MuJoCo Playground paper says the bracket "tilts the
  palm downward by 20°"; `1.92 rad = 110.0° = 90° + 20°` and `1.88 rad = 107.7°`. 1.92 is the
  hardware-correct value and 1.88 was simply wrong. There is nothing to sweep, and the
  previous revision of this list was wrong to call it "the only cheap structural axis left".
- **The port is faithful — do not go looking for a config bug.** The 2026-09-06 audit checked
  obs (57/91, `history_len=1` both sides), all seven reward scales, the orientation kernel,
  actuators, cube geom and goal machinery against playground's GitHub source. The only real
  deviation left is the 20×-weak success bonus, which run 20 tested and made worse. Velocity
  blindness and contact-buffer overflow both died in that audit.
- **The leading account is now an equilibrium in the reward, not a limit in the hand.**
  Four interventions each raised tip closure and lowered spin closure by comparable amounts
  while total stayed at 77.6–81.4%. A capability limit does not rebalance. The orientation
  reward is *linear* from 180° to 11.5° and flat below, so the marginal reward for one more
  degree is the same at 130° and 30° while the marginal cost rises steeply near the goal —
  the policy stops where those meet, and that point belongs to the reward. It also explains
  why the bare hand succeeds on the identical reward: a lower cost of precision puts the same
  equilibrium inside 0.1 rad, where the success bonus fires and bootstraps.
- **Confirmed, not just predicted (2026-09-11).** Run 33 paired a reward that pays for
  precision (`cube_orientation_inverse`, warm-started past the iteration-300 breakthrough) with
  the same task, and total closure moved to 91.1%, clear of the 77.6–81.4% band every
  cost-of-precision intervention stayed inside. That is the falsifiable prediction this account
  made, and it held. See "## 33. `warm-inv-pin-ent1e3` — the first policy that reaches goals".
- **Cold-start reward edits die before they can test anything.** Lining up training curves,
  every run that works breaks through from ~90° to 50–60° error between iteration 300 and 600;
  runs 20, 22, 23 and 31 all die before that point and are not evidence about terminal
  precision — they never got that far. See "## The iteration-300 breakthrough, and why
  cold-start reward edits keep missing it (2026-09-11)". Any future reward-shape or
  goal-machinery change should warm-start from a post-breakthrough checkpoint (`--init-from`)
  rather than train from scratch.
- **In flight: can the policy PERCEIVE the threshold it is scored on?** `obsnoise-0` and
  `obsnoise-half` (`--obs-noise-scale 0.0 / 0.5`, `--cube-priority 0`), queued 2026-09-06
  15:03. `cube_ori` observation noise is ±0.1 on rotation-matrix entries — per component
  std 0.058, tilting a unit column by ~0.082 rad ≈ **4.7°** — against a success threshold of
  0.1 rad = **5.7°**. The measurement noise is the same order as the target, so the gradient
  needed to servo inside the threshold is buried in the policy's own observation noise. The
  corroborating detail: **eval already runs with corruption off** (`play=True` sets
  `enable_corruption=False`) and the policy still stops at 26–27° with clean observations,
  which is the signature of a policy trained blind rather than one blinded at test time.
  `obs_noise_scale` is playground's own `obs_noise.level` (its `default_config` has
  `level=1.0` over the three scales), which this port hard-coded rather than exposed — so
  1.0 is exact parity and this only turns a knob the reference already has.
- **Update 2026-09-11: this eval has since completed.** Like every other historical drift-on
  success count in this file, its result predates the eval measurement-bug fix (see
  "## Interlude — the eval measurement bug, found 2026-09-11") and has not been re-scored
  under it.
- **Pre-registered follow-up if noise is null: the reward SHAPE.** Switch the orientation
  term's sigmoid from `"linear"` to the convex `_long_tail_tolerance` already sitting unused
  at `rewards.py:49`, so marginal reward *increases* as the goal is approached. This attacks
  the equilibrium directly, unlike run 23's `orientation_fine`, which bolted a second term
  beside the linear one and left it setting the equilibrium.
- **Executed, 2026-09-11 — and it needed more than the kernel swap.** Runs 32–34 tried exactly
  this move (a different convex kernel, `cube_orientation_inverse`, not `_long_tail_tolerance`,
  but the same idea: marginal reward rising toward the goal). The kernel swap alone (run 32)
  produced an action-std runaway with no verdict; it only worked once paired with a lower
  entropy coefficient *and* a warm start past the iteration-300 breakthrough (run 33). A
  cold-start attempt at this would most likely have died the same way runs 22, 23 and 31 did.
- **The finger-pad rolling claim is still untested.** The probe answered the palm (null) but
  cannot reach the curved phalanges. Needs a rig that keeps the curvature and varies only the
  surface — the real finger against the same finger with its pads replaced by a capsule.
- **A second seed at `--action-l2 1e-4` is still the cheapest open question.** That cell moved
  best error 26.3 → 22.4°, just past the ~3° floor, while its 1e-3 sibling moved 3.5° the
  other way. One seed cannot tell those apart. Runs 29–30 produced the same non-monotone
  ±2° straddle, which reinforces the point.
- **ANSWERED (2026-09-12): attribution inside runs 32–34.** The reward kernel, the pinned goal
  and the entropy cut all changed at once in a single seed. Runs 35–36 resolved it leave-one-out:
  **the kernel closes the distance and the pinning converts it**, neither alone is enough, and
  the entropy setting rests on run 32 (std runaway without it) plus run 21 (1e-3 alone moves
  nothing). See "## 35–36. `abl-nokernel` / `abl-nopin`". Still true and worth remembering: the
  training task in runs 32–37 is goal-pinned, not the reference task, though every number quoted
  for them is measured in the reference env with drift on.
- **ANSWERED (2026-09-12): run 34's continuation plateaus at ~iteration 8000.** It ran to 8999
  on the rented A100 and to 11625 locally. Scored at 3 seeds, iteration 8000 gives 82.5% of
  episodes reaching < 5.7° and 11625 gives 80.8% — indistinguishable. 4499 → 8000 is where the
  entire gain lives; the 2,626 iterations after it bought nothing measurable. **The recipe is no
  longer the open question — it is converged, and more iterations of it are not an experiment.**
  See "## 37. `remote_brev-ext` / `night-ext` — the main line, and where it converges".
- **THE open question is now the tip-over residual, at a fifth of episodes.** At the converged
  checkpoint, 30 of 139 reference-env episodes never reach 5.7°, and only 2 of those drop the
  cube — the rest survive the full episode and stall at a median 18.1°, split 14.5° tip against
  8.4° spin. Same signature at iteration 8000. This is the horizontal-axis component that has run
  through this file from the start; it is now confined rather than changed. It will not yield to
  more iterations of the run-33 recipe — that is measured, not assumed (run 37). Whatever comes
  next has to attack the tip-over directly: a tip-axis waypoint or skill decomposition, a
  curriculum on the horizontal component, or an initial-state distribution that stops over-
  sampling the episodes it already wins.
- **Open, and a question about the task rather than the policy: what the kick costs at
  convergence.** Pinned, the converged policy holds the goal for half a second in 73.1% of
  episodes at a median best error of 0.73°. Under the drift it reaches 5.29° and holds nothing.
  The precision is there and the reference task does not pay for it. **The drift is playground's
  own `InHandReorientationCommand` behaviour, so the pinned number is a capability measurement
  and not a task score** — quoting it as a task result would be a departure from the reference
  dressed up as a win. What is genuinely open is whether the kick should be replaced by a fresh
  goal *resample* (new goal, no 160° discounted penalty) for the deployed task, and whether that
  is still the same benchmark. Decide that before quoting pinned numbers anywhere outside this
  file.
- **Open, practical: the DGX has been idle since 06:20 on 2026-09-12** and the queue is stood
  down on `STOP_SUPERVISOR`. There is no next run queued, because the obvious one — continue the
  main line — is the one run 37 rules out.
- **Use ≥3 eval seeds from here on.** The same checkpoint at 128 envs, re-scored three times,
  spans median best error 25.4–28.6° from GPU non-determinism alone — see the Interlude. A
  single-seed eval at n=128 is not enough to resolve a <3° effect.
- **Any sweep needs a resolution floor.** Run 24 puts run-to-run spread at ~1–2° of
  deterministic best error, so **a cell that moves best-error by less than ~3° has shown
  nothing at n=1.**
- **Drop training-time `consecutive_success` as a leading indicator.** It misled twice in
  runs 29–30: 0.030 early on `cube-0325`, and 0.042 on `cube-0300` — above run 14's 0.0395 at
  iteration 2999 — both converting to 0/32. It counts stochastic threshold crossings under
  exploration noise, which accumulate without the policy ever arriving deliberately.
- **There is no action-magnitude readout in `scripts/eval_policy.py`.** The `|a|` numbers in
  runs 27–28 are derived from `Episode_Reward/action_l2`, not measured deterministically.
- **PROPOSED, not started: train the bare LEAP hand in this stack as a true control.** Every
  comparison this project has made against "the reference" is our LeapXELA numbers against
  *someone else's bare-hand numbers measured differently* — Hamid's reward 375, or the
  playground paper's ten real-hardware trials (median 3.5 / mean 7.1 / best 27 consecutive
  rotations). Neither is like-for-like, and the reward comparison is now void anyway: runs
  33–34 changed the orientation kernel, so our reward is no longer on the same scale as the
  375. Meanwhile the whole premise of task 1 — that the XELA skin is what makes this the hard
  case — has never been measured on one bench.
  The test: register playground's `leap_rh_mjx.xml` (public, not in this repo) as a second
  `finger_tip_type`, and run the run-33 recipe on it with the same eval. That gives bare-hand
  vs XELA under identical physics, reward, episode length and metrics, and the difference
  between the two curves is what the tactile skin costs — which is the number the internship
  exists to produce. Cost: model registration plus one 3000-iteration run, ~2 h on the DGX.
  Deferred 2026-09-12 by the supervisor's call ("not yet but maybe later"); the warm-start
  recipe should be settled first so the control is run against a stable recipe rather than a
  moving one.

---

## 26. `friction08-prio1` — the friction payoff test, answered: no

Run 14 + `--cube-priority 1 --cube-friction-sliding 0.8`, seed 42, 8192 envs, 1499/1500
iterations, 103 min, started 2026-09-06 00:01. The run Hamid's question actually needed:
everything before it was replay evidence, and he asked what *happens*, which needs training.

Two things move against run 14, not one, and they cannot be separated in a single run —
`priority 1` is the precondition for the friction number to reach the contact at all.
`--cube-friction-sliding` also disables `dr_cube_friction`, so μ is pinned rather than
resampled.

Deterministic eval, 32 envs, seed 7, each checkpoint scored in its **own** training env:

| | start | best | final | succ | drops | total ↓ | spin ↓ | tip ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 @1500, priority 0 | 134.7° | **26.3°** | 29.7° | 0.0% | 0.03 | 80.1% | **93.5%** | 66.9% |
| 26 @1499, priority 1 μ=0.8 | 139.2° | **25.1°** | 36.1° | 0.0% | 0.06 | 81.4% | 76.7% | **73.1%** |

**The headline is null.** 26.3 → 25.1° is 1.2°, inside the ~1–2° seed spread run 24
established, so it is not a result. Zero successes either side, as in every reorient run.

**The finding is that the error moved rather than shrank.** Tip-over closure improved
66.9 → 73.1% while spin closure fell 93.5 → 76.7%. Run 14 vs run 24 at matched iteration 600
differ by ~1 point on spin and ~0.5 on tip, so a −16.8 / +6.2 swing is well outside noise.
The policy got better at the half it had been failing and worse at the half it had already
solved, and the two cancelled. `final` also degrades 29.7 → 36.1°, i.e. it holds the pose
less well after reaching it.

**Why that is the expected shape, given the correction in run 25.** The intervention is weak
where it was expected to matter and strong where it was not. At the fingertips run 14 already
sampled μ ∈ (0.5, 1.0) every reset via `dr_fingertip_friction`, so pinning 0.8 moves the mean
about 7%; the palm pads go from ≈ 0.35 to 0.8, more than double. Tipping the cube means
rolling it against the palm, so a large pad-friction increase improving tip-over while
leaving spin — a fingertip skill — unhelped or worse is exactly what that asymmetry predicts.
The spin regression is consistent with the other half of the change: the grasp friction
stopped being randomized, and a policy trained on one fixed value has less to hold onto.

Caveats. The two rows are scored in different physics, each in the env it trained in, so this
compares *configurations*, not one policy under one contact model. n = 32 episodes, and the
drops difference (0.03 vs 0.06) is one or two events either way — do not read it.

Answering the email end to end: the cube now has priority and dictates the contact (verified,
`@tip` and `@pad` both resolve to 0.80); raising sliding friction does remove most of the
slip (34–39 → 28.7 mm/s at the fingertips); sliding friction and palm angle are both
hyperparameters; and the cube is not wedging into the pad gaps because the pads overlap and
there are no gaps. What it does not do is move reorient. **This is the second contact-model
fix to fail on this task**, after `condim 6` in run 19 — and run 19 is the precedent that
predicted it.

---

## 27–28. `action-l2-1e3` / `action-l2-1e4` — the saturation fix, answered: the clip was real, the stall is not the clip

Two cells bracketing one weight, both run 14 + `--action-l2 <w>` at seed 42, 8192 envs,
1499/1500 iterations, ~100 min each, started 2026-09-06 02:47 and 04:27. `--cube-priority 0`
restores run 14's contact model, so `--action-l2` is the **only** difference from run 14 and
the comparison is genuinely single-variable — unlike run 26, where two things moved at once.

The term is new: `mdp.rewards.action_l2` (`rewards.py:193`) returns
`torch.sum(torch.square(env.action_manager.action), dim=1)`, the raw pre-scale policy output,
the same space `action_rate_l2` lives in. The motivation is that **only the action *rate* was
ever penalized, and the action term clips to the actuator ctrl range**, so above the clip the
reward gradient with respect to the action is exactly zero and nothing pulls its magnitude
back. The optimum of a rate-only penalty is therefore a large *constant* action, and that is
what the reference line learned — measured on run 14's `model_2999`, deterministic, between
the approach phase and the stall:

| phase | mean \|a\| | step-to-step \|da\| |
| --- | --- | --- |
| steps 0–80 (approach) | 8.3 | 2.16 |
| steps 150–400 (stall) | **13.4** | 0.93 |

Magnitude grows while change collapses. At `scale=0.5` a mean \|a\| of 13.4 is a commanded
joint delta of 6.7 rad against a median ctrl range of 2.27 rad — 3× the entire range, p95 15×.
That made "the policy is a saturated bang-bang controller with no fine authority" the
standing explanation for the ~30° steady-state shell it converges to from any starting offset.

Deterministic eval, 32 envs × 700 steps, seed 7, `cube_priority=0` — all three rows scored in
the same physics:

| | start | best | final | succ | held | drops | total ↓ | spin ↓ | tip ↓ | best @ step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 @1500 (baseline) | 134.7° | **26.3°** | 29.7° | 0.0% | — | 0.03 | 80.1% | **93.5%** | 66.9% | 104 |
| 27 @1499, `l2 = 1e-3` | 127.3° | **29.8°** | 34.8° | 0.0% | 0.0% | 0.03 | 77.0% | 89.5% | 63.6% | 249 |
| 28 @1499, `l2 = 1e-4` | 134.7° | **22.4°** | 29.9° | 0.0% | 0.0% | 0.03 | 80.2% | 91.0% | **69.3%** | 122 |

**The mechanism fired and the shell did not move.** Backing action magnitude out of the
training logs gives a per-dim \|a\| of ~5.2 at `1e-4` and ~2.2 at `1e-3`, against run 14's
13.4 — a 2.6× and 6× collapse. The penalty did exactly what it was designed to do. Best error
still sits at 22–30°, and successes are still **0/32 on all three criteria** — instantaneous,
held, and settled. Saturation was real, and it is not what is holding this task at 30°.

**This is the pre-registered negative branch, and the queue comment called it in advance:**
"If `|a|` falls but the shell does not move, saturation was real but not the binding
constraint, and the palm-angle axis is next" (`run_queue.sh:61–63`). The other half of that
prediction — held-success rising above 0 — did not happen either. **This is now the third
actuation- or contact-model fix to fail on reorient**, after `condim 6` (run 19) and friction
(run 26). Three different physical explanations for the stall have each been tested
single-variable and each returned null.

**Between the two cells, `1e-3` over-penalizes.** It is worse than baseline on every axis —
total 80.1 → 77.0%, spin 93.5 → 89.5%, tip 66.9 → 63.6% — and it takes 249 steps to reach
best against the baseline's 104. Clamping the action that hard costs authority: the policy
gets a calmer hand and a slower, weaker one. Its `action_rate` penalty also falls (−0.487 →
−0.358), i.e. the magnitude penalty quiets the rate term for free, which is consistent with
a smaller action rather than a better-controlled one.

**`1e-4` is the usable weight.** It matches baseline on spin (93.5 → 91.0%), gains on
tip-over (66.9 → **69.3%** — the half that has always been the failure), and holds the pose
better afterward (final 29.9° vs `1e-3`'s 34.8°, and drift-after-best p90 8.4° vs run 14's
19.0°). If the action penalty is kept in the reward at all, this is the weight to keep.

**On whether `1e-4`'s 22.4° is a real improvement: do not call it one.** 26.3 → 22.4° is
3.9°, which just clears the ~3° resolution floor run 24 established, and it is the first
movement in several runs that is even arguably above noise. But it is n=1 at one seed, and
the `1e-3` cell moved 3.5° *the other way* from the same intervention. A pair of cells
straddling the baseline by ±4° in opposite directions is what a noise band looks like, not
what a trend looks like. A second seed at `1e-4` is the cheapest thing that would settle it.

Caveats — **the \|a\| figures above are derived, not measured.** There is no
action-magnitude readout in `scripts/eval_policy.py`, so a direct deterministic measurement
of these two checkpoints is still unwritten. What is available is `Episode_Reward/action_l2`,
divided back out. mjlab's `RewardManager.reset()` logs `episodic_sum_avg /
max_episode_length_s` (`.venv/.../mjlab/managers/reward_manager.py:108`), and the per-step
reward is `w · ‖a‖² · dt`, so with `max_episode_length_s = 50` and `ctrl_dt = 0.05`:

    mean_t ‖a‖²  =  Episode_Reward · 50 / (w · 0.05 · T)          T = mean episode length in steps

then subtracting the exploration-noise floor `16 · std²`, since these are *sampled* training
actions and not the mean:

| run | `Episode_Reward/action_l2` | T | std | mean ‖a‖² | noise floor | per-dim \|a\| |
| --- | --- | --- | --- | --- | --- | --- |
| 27 `l2 = 1e-3` | −0.1418 | 975 | 2.014 | 145.5 | 64.9 | **2.24** |
| 28 `l2 = 1e-4` | −0.0524 | 981 | 2.587 | 534.6 | 107.1 | **5.17** |

Two things this is not. These are episode-averaged over the whole rollout, while run 14's
13.4 was isolated to the stall phase (steps 150–400) where magnitude is highest — so the
comparison is biased *against* the new runs, and the true collapse is if anything larger than
2.6×/6×. And they are RMS per dimension where run 14's 13.4 is a mean absolute value, which
biases the other way. The direction and rough scale are solid; the exact ratios are not.
Nothing in the conclusion turns on the difference — a 2× error either way still leaves the
clip broken and the shell intact.

---

## 29–30. `cube-0325` / `cube-0300` — the cavity test, answered: no

Two cells shrinking the cube, both run 14 + `--cube-half-size <h>` at seed 42, 8192 envs,
1500/1500 iterations, ~100 min each, started 2026-09-06 11:37 and 13:21. `--cube-priority 0`
restores run 14's contact model, so **size is the only difference from run 14** and each
comparison is single-variable.

The motivation is measured, not guessed, and it is in the next section: the XELA taxel pads
sit proud of the structural collision surface by a median 2.84 mm, so a grasp loses ~5.7 mm
of cavity and a 70 mm cube in this hand has the clearance a 75.7 mm cube would have in the
bare LEAP hand — **scale 1.081 before anyone changes anything**. Restoring the bare hand's
absolute clearance means 70 − 5.7 ≈ 64 mm, or 70 − 7.9 ≈ 62 mm on the tip-over axis:
half-size 0.031–0.032 against the reference 0.035. The two cells bracket that.

It is also an axis nobody had ever explored in this direction. Hamid's cube-scale sweep was
1.0 / 1.05 / 1.08 / 1.09 / 1.1 / 1.2 — **every cell at or above reference**, on a hand
already effectively at 1.08 — and his note on dropping to 1.0 was "it seems the hand is
struggling to rotate the cube". He never went below 1.0.

Mass is **not** swept with size. `build_kwargs` carries `cube_mass: 0.108` and
`_apply_env_overrides` rebuilds from `build_kwargs`, so overriding only `--cube-half-size`
leaves mass pinned — which is Hamid's own convention (his XML holds mass constant across
scales) and keeps these cells on the same axis as his 1.0–1.2 cells. The consequence is that
the smaller cubes are **denser than physical**: at half-size 0.0300 a pinned 0.108 kg is
~59% above the volume-scaled mass. Confirmed harmless here only because the result is null;
if size ever does move this task, that confound has to be broken with a third cell at
volume-scaled mass before "it was the size" can be claimed.

Deterministic eval, 32 envs × 700 steps, seed 7 — each row scored in its own training env:

| | start | best | final | succ | held | drops | total ↓ | spin ↓ | tip ↓ | best @ step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 @1500, 70 mm | 134.7° | **26.3°** | 29.7° | 0.0% | — | 0.03 | 80.1% | **93.5%** | 66.9% | 104 |
| 29 @1499, 65 mm (`0.0325`) | 134.7° | **27.1°** | 34.3° | 0.0% | 0.0% | 0.00 | 77.6% | 91.9% | 71.9% | 153 |
| 30 @1499, 60 mm (`0.0300`) | 134.7° | **24.8°** | 33.3° | 0.0% | 0.0% | 0.00 | 77.8% | 88.0% | **73.3%** | 138 |

**The headline is null, and it is null across a 14% span of cube size.** Best error moves
26.3 → 27.1 → 24.8°, a total range of 2.3° against the ~1–2° seed spread run 24 established
and well under the ~3° resolution floor. Successes are **0/32 on all three criteria** —
instantaneous, held and settled — at every size. That is not a shallow gradient being climbed
too slowly; a real effect over a 14% span of the supposed binding constraint would not hide
inside seed noise. Note also that the two cells move in *opposite* directions from baseline
(65 mm worse by 0.8°, 60 mm better by 1.5°), which is the same non-monotone signature runs
27–28 produced and is what noise looks like rather than a trend.

Drops fall 0.03 → 0.00 at both sizes. The grip is if anything better with a smaller cube, so
nothing here is a grasp-stability problem.

**Training-time `consecutive_success` misled, twice, and should be dropped as a leading
indicator.** On `cube-0325` it reached ~0.030 by iteration 150 and sat flat there for 1350
more iterations; on `cube-0300` it crept to **0.042**, *above* run 14's 0.0395 at iteration
**2999** and reached in a third of the budget. Both converted to 0/32 in the deterministic
eval. The metric counts stochastic threshold crossings under exploration noise, which a
policy can accumulate without ever being able to arrive deliberately — run 14 already
demonstrated the same disconnect. Read the deterministic best error instead.

### The conservation law — the shape of the nulls is now the finding

Four interventions have now been tested single-variable against run 14, and each one
**improved tip-over closure while degrading spin closure by a comparable amount, leaving the
total pinned**:

| | spin ↓ | tip ↓ | total ↓ | best |
| --- | --- | --- | --- | --- |
| 14 @1500 (baseline) | **93.5%** | 66.9% | 80.1% | 26.3° |
| 26 `friction08-prio1` | 76.7% | 73.1% | 81.4% | 25.1° |
| 29 `cube-0325` | 91.9% | 71.9% | 77.6% | 27.1° |
| 30 `cube-0300` | 88.0% | **73.3%** | 77.8% | 24.8° |

(`condim 6` — run 19 — belongs in this list as the first of the four, but it predates the
`eval_policy.py` decomposition and has no spin/tip numbers to quote. Its total was null on
the same task while transforming `rotate_x`, which is the same pattern one level up.)

Spin spans 76.7–93.5% and tip spans 66.9–73.3% across these rows, while **total stays inside
77.6–81.4% and best error inside 24.8–27.1°**. Every intervention buys tip-over and pays for
it in spin, at par.

**A capability limit does not rebalance. An equilibrium does.** If the hand simply could not
tip the cube, giving it more grip or more room would raise tip closure and leave spin alone;
the total would move. Instead the policy re-allocates and the total does not move, which is
what an optimum looks like when you change the relative cost of two ways of reaching it
without changing what either is worth.

There is a mechanism for exactly that, and it is in the reward.
`mdp.rewards.cube_orientation_tolerance` is
`tolerance(err, bounds=(0, 0.2), margin=π, sigmoid="linear")` — **linear** in orientation
error from 180° down to 11.5°, and flat below. Linear means the marginal reward for closing
one more degree is *identical at 130° and at 30°*, while the marginal cost of closing that
degree rises steeply near the goal: finer manipulation, more action rate and energy, and more
exposure to the −100 termination. The policy settles where marginal reward meets marginal
cost, and **that point is a property of the reward, not of the hand**. Change the contact
model or the cube and you change the relative cost of spin versus tip, so the policy
re-allocates between them — but the total sits where the reward puts it. That is precisely
the table above.

It also explains the thing that has been most puzzling: the bare LEAP hand succeeds on this
*identical* reward. A lower cost of precision moves the same equilibrium inward past 0.1 rad,
where the 100-point success bonus starts firing and bootstraps itself. The padded hand pays
more for every degree and its equilibrium lands outside the threshold, where the bonus never
fires and the only gradient is the flat-below-0.2 linear term.

This is a hypothesis, not a measurement. What makes it worth acting on is that it is the only
account so far that predicts the *shape* of five nulls rather than explaining each one after
the fact, and it makes a falsifiable prediction: interventions that change the cost of
precision will keep returning null, and only interventions that change what precision is
*worth* — the reward shape — or that change whether the policy can *perceive* the threshold
will move it. Both are testable and neither has been tried.

---

## Parity audit against playground source (2026-09-06)

Every previous parity claim in this file was checked against Hamid's Notion page or against
our own notes. This one was checked against the **actual `google-deepmind/mujoco_playground`
source on GitHub**, file by file. The result is that the port is faithful, and the search for
a config-level bug is over.

| | playground | ours | |
| --- | --- | --- | --- |
| actor obs | 57 dims | 57 dims | ✓ |
| privileged obs | 91 dims | 91 dims | ✓ |
| `history_len` | **1** | `history_length=1` | ✓ |
| obs noise scales | 0.05 / 0.02 / 0.1 | 0.05 / 0.02 / 0.1 | ✓ |
| reward scales | orientation 5.0, position 0.5, termination −100, hand_pose −0.5, action_rate −0.001, joint_vel 0.0, energy −1e-3 | identical | ✓ |
| orientation kernel | `tolerance(err,(0,0.2),margin=π,"linear")` | identical | ✓ |
| actuators | `kp 3.0`, `damping 0.2`, `armature 0.00149376`, `frictionloss 0.02` | identical | ✓ |
| cube | `size=".035"`, `mass=".108"`, `condim="3"`, `friction=".3 0.05"`, **no `priority`** | identical at `--cube-priority 0` | ✓ |
| goal | uniform quat, `goal_quat_dquat` kick on success, ×0.8 per-step decay, threshold 0.1, no curriculum | identical | ✓ |
| success bonus | 100, added **outside** the dt-scaled sum | 100 **inside** it → 20× weaker | ✗ known |
| critic extras | + `pert_dir`, `xfrc_applied` | absent | ✗ inert |

Actuator and cube figures are from `xmls/leap_rh_mjx.xml` and `xmls/reorientation_cube.xml`;
obs, rewards and goal machinery from `_src/manipulation/leap_hand/reorient.py`.

**Two hypotheses died in this audit.**

*Velocity blindness is not a bug.* Our actor sees joint pos, joint pos error, cube pos error,
cube ori error and last action — no `joint_vel`, no `cube_ang_vel`, no `cube_lin_vel`, all of
which are critic-only, and `history_length=1` so it cannot finite-difference either. That
looked like a strong candidate for why the policy closes at 1.5–2.0 rad/s instead of
arriving. But playground's own `default_config()` sets **`history_len=1`**, and its state
vector is the same five terms: the reference is *also* velocity-blind and single-frame, and
it works. Adding velocity or a history stack would be a deviation *away* from the reference,
not a fix toward it.

*Contact-buffer overflow is not happening.* `nconmax=64` looked alarmingly small next to
playground's `naconmax=30*8192`, and this project has already been bitten once by silent
constraint dropping (the njmax overflow, runs before 2026-08-29). But mjwarp's `nconmax` is
**per world** — `naconmax` is the all-worlds override — so 64 × 8192 = 524k against
playground's 246k. We are more generous, and measured peak `ncon` was 24. Dead.

### Palm angle is settled — 1.92 is correct and there is nothing to sweep

The MuJoCo Playground paper states the hardware bracket **"tilts the palm downward by 20°"**.
`1.92 rad = 110.0° = 90° + 20°`. `1.88 rad = 107.7°`. So **1.92 is the hardware-correct palm
angle and 1.88 was simply wrong** — which is the whole of the "1.88 plateaus, 1.92 takes off"
evidence, now explained rather than merely observed.

This closes the axis that the run 27–28 list had promoted to "the only cheap structural axis
left" and was about to become the next experiment. 1.84 / 1.96 do not need running: there is
no reason to expect anything at an angle the hardware does not use. **Removed from the open
list, for free.**

### What the reference actually achieves

Nobody in this project had ever established this. Every reference number in this file up to
now — Hamid's 284, the 157.7 / 166.3 / 171.2 post-fix repeats, our own ~198 — is **reward**,
not success, and reward is not comparable across implementations. So "does the reference
even reach the threshold?" was genuinely open, and if the answer had been no, our 0/32 would
have been parity rather than a gap.

It is not. MuJoCo Playground paper, Table I, ten real-world hardware trials measuring
consecutive successful rotations before failure:

| metric | value |
| --- | --- |
| median rotations | 3.5 |
| mean rotations | 7.1 |
| best trial | 27 |

and the policy "trains within 30 min on two RTX 4090 GPUs". Corroborated on our own side of
the fence by Hamid's bare-LeapHand run on the same script — reward **375**, climbing steadily
from 130, and the video at the top of his Notion page is the one where the cube is genuinely
reoriented — against LeapXELA's 131–197 on the same script.

**The reference definitively works, our 0/32 is a real gap, and the difference is the hand.**

---

## Pad geometry and rolling resistance (2026-09-06)

Two measurements on the compiled `Box_palm192` hand, both aimed at Hamid's one-line diagnosis
of the whole XELA problem: *"the cube stops rolling when it hits the 4x4 finger sensor pads."*

### Pad protrusion — what the taxels cost in usable cube size

Each phalanx carries both a structural collision box (`<finger>_<segment>_collision_<N>`) and
a XELA pad (`<finger>_<segment>_uspa<N>`). Measuring how far the pad's outer surface lies
beyond the structural surface it mounts on, along the struct→pad centroid direction:

| body | struct surface | pad surface | protrusion |
| --- | --- | --- | --- |
| `palm` | 38.2 mm | 39.3 mm | **+1.10 mm** |
| `if/mf/rf_px` | 33.4 mm | 36.2 mm | **+2.84 mm** |
| `if/mf/rf_md` | 9.0 mm | 15.7 mm | **+6.75 mm** |
| `th_px` | 12.6 mm | 17.0 mm | +4.42 mm |
| `th_ds` | 12.1 mm | 8.7 mm | −3.35 mm (recessed) |

Median **2.84 mm per surface**, so a grasp loses ~5.7 mm of cavity: **a 70 mm cube in this
hand has the clearance a 75.7 mm cube would have in the bare LEAP hand, i.e. scale 1.081
before anyone changes anything.** On the tip-over axis it is worse — tipping rolls the cube
against the palm using the middle phalanges, and `palm + md` is 1.10 + 6.75 = **7.9 mm** —
and the reorient residual is 96% horizontal-axis error, i.e. exactly that axis. The middle
phalanges are the single most padded segment on the hand.

**This is the day's one durable positive result, and it stands independently of runs 29–30.**
Those runs show that restoring the clearance does not fix reorient; this measurement still
quantifies what tactile skin costs in usable cube size, which is a real number about the
hardware and is the kind of thing the internship exists to produce.

Caveat: protrusion is measured along the struct→pad centroid direction as a proxy for the
outward normal, so treat it as ±1 mm. The `th_ds` sign is worth a second look given that the
thumb is exactly where the old collision-model bug lived (commit `d60d4bd`).

### `scripts/rolling_probe.py` — corrugation costs nothing on the palm

Run 25 established that the pads do not leave **gaps** to wedge into: half-extents ~15 × 11 ×
14 mm at 16.0 mm centre spacing, so they overlap by ~6 mm and the shell is continuous. But
continuous is not smooth — 36 overlapping boxes at assorted orientations form a *faceted*
surface, and gap width says nothing about facet height. Rolling over ridges is also a
different physical quantity from sliding, which is all `slope_test.py` and `friction_probe.py`
measure, and it would explain why every friction/condim/priority intervention returned null
on a task whose residual is tip-over.

The probe freezes every collision geom as a static worldbody box at its home-pose transform,
tiles the patch into a track, presses the cube in with a 5 N normal load, and drives the roll
with a velocity servo, reading back the torque required. Palm region, matched friction 0.20,
matched footprint:

| surface | geoms | μ | breakaway τ | τ @ ω=0.5 | ω=1.0 | ω=2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| pads | 63 (9 tiled) | 0.20 | 0.2114 N·m | 0.1000 | 0.1095 | 0.1182 |
| smooth | 1 | 0.20 | 0.2140 N·m | 0.1000 | 0.1262 | 0.1199 |

**Null.** The real pad shell is 0.99× the smooth plate to start rolling and equal or
marginally *lower* to keep rolling. Corrugation costs nothing there.

**This does not refute Hamid.** His claim is about the *finger* pads, and that region is not
measurable by this rig: the finger pads are distributed along curved phalanges, so tiling
them produces a jumble rather than a track and the cube falls down it — slip −18 to −2,
against −0.13 to −0.23 on the palm. Testing the fingers needs a different design that keeps
the curvature and varies only the surface: the real finger against the same finger with its
pad boxes replaced by a capsule of matching radius. **The finger claim remains open.**

Three rig formulations failed before one worked, each of which looked fine until the numbers
came back:

1. **Free cube + applied torque measures free spin, not rolling.** Once static friction
   breaks, an unconstrained rigid body under a pure torque obeys `ω = τt/I` and the surface
   underneath stops mattering. It read **866 rev/s on both surfaces** — the rigid-body answer.
   Fixed by driving the roll with a velocity servo on a hinge and reading `actuator_force`.
2. **Gravity alone rests the cube on one protruding box.** 1.06 N of weight balances the cube
   on whichever geom sticks out furthest, touching a single geom — a balancing act, not a
   surface. The grasp has 3.4–4.9 pads in contact per `friction_probe`, so the load has to be
   imposed.
3. **A single patch is too small.** A 70 mm cube reaches the edge of a ~50 mm palm patch
   within a fraction of a turn and falls off (slip −20). Tile the patch into a track.

And one bug worth recording because it did not present as one: a box's half-extent along the
drop axis must be computed as `sum_j |R[axis,j]| · size[j]`, **not** `sizes.max()`. The latter
read the wide thin control plate as 0.56 m thick, started the cube 0.67 m above it, and
tunnelled it straight through the 20 mm plate at 8.7 m/s (17 mm of travel per 2 ms step).
The symptom was "cube did not settle in contact with the shell", not an obvious error.

The script blanks any driven cell with `|slip| > 0.6`, printing `--` instead of a torque, so
a later reader cannot mistake a sliding or tumbling cell for a rolling resistance. Raw data
in `eval/rolling/{palm,fingers,sensors}.json`.

---

## Interlude — the eval measurement bug, found 2026-09-11

mjlab's `CommandTerm.compute()` calls `_update_metrics()` (which tests success against the
current goal) and then `_update_command()` (which applies the MJX goal-drift kick) in the
*same* call. `scripts/eval_policy.py` read `command.command` *after* `step()` returned — i.e.
the goal already kicked to its post-success value. So in every drift-on (reference-env) eval, a
step that crossed the 0.1 rad threshold was scored against the goal it had just been kicked to,
not the one it actually closed on, and was recorded as 10–50° of error instead of a success.

**Consequence: `reached_threshold` was 0 by construction in every reference-env eval ever run,
and the recorded minimum error piled up just above the 5.7° threshold.** This is the exact
signature `research/NEXT_EXPERIMENTS.md` reported as its headline finding: "across seven
independent runs... the best episode lands at 5.94–6.39°, and the success threshold is 5.73°" —
that table is an artifact of this bug, not a property of the policy. Pinned-goal evals were
never affected; there is no kick to mistime when the goal never moves.

**Fixed** by scoring against the goal the step was actually judged by — the previous step's
post-update goal, or the freshly sampled goal on a reset step — cross-checked against the
command's own `orientation_error` metric (max discrepancy 9.5e-7 rad once time-aligned). The
old, mistimed quantity is kept per episode as `min_err_post_update` /
`reached_threshold_post_update`, so old-style numbers stay reproducible. A new
`threshold_entries_total` / `threshold_entries_per_episode` counts *separate* entries into the
threshold — goals reached per episode, the quantity runs 32–34 below report as "goals/episode".

Corrected baselines, 128 envs × 700 steps, seed 7, `--cube-priority 0`, reference env: run 14
`model_2999` reads **2/128** real successes where the old, mistimed measure said 0; run 14
`model_1500` reads **0/129**.

**Every historical drift-on success count in this file is censored and must be re-scored before
being quoted** — runs 14 through 30, `obsnoise-0` / `obsnoise-half`, and every "0/32 everywhere"
line written above this section.

One more thing this surfaced: **eval noise is larger than previously assumed.** The same
checkpoint at 128 envs, re-scored three times, gave median best error 25.4° / 26.8° / 28.6° —
GPU non-determinism alone, no config difference. Treat ±2° as noise even at n=128, and prefer 3
eval seeds (~390 episodes) over 1 — every eval below this point uses 3 seeds for that reason.

---

## 31. `thresh-04` — Experiment 1 from research/NEXT_EXPERIMENTS.md, answered: no (strong regression)

Run 14 + `--success-threshold 0.4` + `--cube-priority 0`, seed 42, 8192 envs, 1499/1500
iterations, ~1h45m, started 2026-09-11 15:59. The new `--success-threshold` flag moves *both*
the command's `orientation_success_threshold` and the success-bonus reward threshold together
(verified by config diff: only those two values move, orientation weight stays 5.0).

Deterministic eval, 32 envs, seed 7, reference env:

| | best error | succ @ 0.1 rad | held @ 0.4 rad | spin ↓ | tip ↓ |
| --- | --- | --- | --- | --- | --- |
| 14 @1500 | 26.3° | 0/32 | 11/32 | **93.5%** | 66.9% |
| 31 `thresh-04` @1499 | **72.2°** | 0/32 | 5/32 | 82.3% | 20.1% |

**Training curve identical to run 14 until iteration ~300, then they split.** Run 14 breaks
through to ~56° by iteration 600; `thresh-04` sits at ~90–95° for all 1500 iterations. It never
gets the breakthrough at all.

Mechanism (inferred, not directly tested): the threshold also sets where the goal-drift kick
fires. At 0.4 rad the kick fires at 22.9° — exactly the region run 14 passes *through* while
learning to tip the cube — so the policy learns to stay away from it instead. The handoff's
experiment therefore tested "move the kick to 22.9°", not "pay the policy for getting close".
And the handoff's own success criterion could not have discriminated the two anyway: run 14
*already* scores HELD 11/32 and SETTLED 11/32 at 0.4 rad, before this run changes anything.

---

## The iteration-300 breakthrough, and why cold-start reward edits keep missing it (2026-09-11)

Lining up training-time `orientation_error` (logged every 100 iterations) across every run that
has been scored shows the same shape wherever the policy works at all: **flat around 90° until
iteration ~300, then down to 50–60° by iteration 500–600** — the point at which tip-over gets
learned. Run 14, `reference-seed7`, `obsnoise-0` and `cube-0300` all do it. Physical
interventions (friction, condim, cube size, action penalty) never block it — they change where
the policy settles *after* the breakthrough, not whether the breakthrough happens.

Every from-scratch reward or goal edit has instead failed to reach it:

| run | curve |
| --- | --- |
| 20 `success-x20` | gets halfway, ~65°, then action std diverges to 5.95 |
| 22 `fixed-goal` | ~90° for the whole run; action std runs away to 4.3 |
| 23 `orientation-fine` | 84° at iteration 600 |
| 31 `thresh-04` | ~90–95° for all 1500 iterations |

The common factor: all four change the reward *near the goal* — a region a pre-breakthrough
policy only visits by luck. So runs 20, 22, 23 and 31 died upstream of the thing each was built
to test, and **none of them is evidence about terminal precision.** Nothing in this project had
been warm-started from a trained checkpoint before 2026-09-11 — every one of the above trained
its new reward from scratch, fighting the same 300-iteration climb the edit was supposed to be
tested past.

This is what motivates runs 32–34 below: start from a checkpoint that has already made the
breakthrough, and only then change the reward.

One more calculation worth recording here, since it bears directly on
`research/NEXT_EXPERIMENTS.md` Experiment 2. Using a discounted (γ=0.99) value model, calibrated
against the previously measured ≈−12 net reward per success under the drift kick, the net value
of *one* success works out to:

| kernel | bonus weight 100 | bonus weight 2000 |
| --- | --- | --- |
| linear | −11.4 | +83.6 |
| inverse (weight 8.93) | −189 | −94 |

Pairing the inverse kernel with the drift kick *on* — which is what Experiment 2 specifies —
would train the policy to approach the goal and never cross it, worse than the linear kernel at
the same bonus weight. Experiment 2's `cap=10.0` is also a no-op: `1/(d+0.1)` never exceeds 10
over the valid error range. Runs 32–34 pin the goal during training instead, which sidesteps
this trap entirely.

---

## 32. `warm-inv-pin` — stopped at 1650/3000, action-std runaway

First warm start run in this project. `--init-from` run 14 `model_1500`, target 3000
iterations, with `--orientation-kernel inverse` (new reward `cube_orientation_inverse` =
1/(err+eps), eps 0.1, weight 8.93 = 1.79× the linear weight, chosen so marginal reward matches
the linear kernel exactly at 130°), goal pinned (`--goal-drift False
--goal-resample-on-success False`), `--cube-priority 0`, `entropy_coef` left at its default
0.01.

The warm start itself worked — no collapse, training error held at run 14's ~45–48° level. But
action std rose linearly at ~0.009/iteration, 2.81 → 4.03 by iteration 1637, with no sign of
levelling off, and drops rose 0.15 → ~0.5 per episode. Stopped on a pre-registered std > 4
criterion. (For scale: run 22 rose at ~0.002/iteration; run 20 reached 5.95 before it was judged
harmful.)

At iteration 1650, 128 envs: reference env median best error 25.2° vs control's 25.4°, 0/129
successes; pinned env 11/129 instantaneous successes vs control's 6/128. Spin 87.5% / tip 75.5%
/ total 80.3% — another rebalance inside the old 77.6–81.4% band, not outside it.

**Not a verdict on the kernel** — only 150 iterations ran, with std actively running away for
all of them. Cause: rsl_rl's `GaussianDistribution` is unbounded (see the noise-floor diagnosis
earlier in this file), so the entropy bonus pushes `log(std)` up at a roughly constant rate
unless the reward's own gradient opposes it. Run 14's reward held std at 2.81; this reward does
not.

## 33. `warm-inv-pin-ent1e3` — the first policy that reaches goals

Run 32's config + `--entropy-coef 0.001`. Warm start from run 14 `model_1500`, iterations
1500→2999, 8192 envs, seed 42, 1h46m, 2026-09-11 19:42–21:28.

The entropy fix worked: action std drifted *down*, 2.80 → 1.91, instead of up. Drops stayed
0.10–0.35 per episode throughout. Run 21 already showed `entropy_coef` 1e-3 alone does not move
the stall (error and success rate unmoved there), so entropy alone is an unlikely explanation
for what follows.

Final @2999, reference env, 3 eval seeds × ~128 episodes each, against control run 14
`model_2999` (same start, same 1500 extra iterations, old linear reward):

| | run 33 @2999 | control (run 14 @2999) |
| --- | --- | --- |
| episodes reaching < 5.7° | **121/388 = 31.2%** | 7/385 = 1.8% |
| goals per 35 s episode | **0.36** | 0.02 |
| median best error (3 seeds) | 10.8° / 12.3° / 12.6° | 28.6° / 29.8° / 28.2° |
| spin / tip / total closure, seed 7 | 92.8% / 90.0% / **91.1%** | 83.9% / 72.4% / 76.5% |
| pinned-goal: reach / HELD @ 0.1 rad | 44/128 / 36/128 | — / 1/128 |
| pinned-goal p10 best error | 0.6° | — |

**Total closure leaves the 77.6–81.4% band that every one of the seven nulls stayed
inside.** That is the pre-registered, worth-of-precision-vs-cost-of-precision
prediction in `research/verdicts/00-INTERIM-reward-kernel.md`, and it held. This is a
worth-of-precision intervention, and it behaved unlike every cost-of-precision intervention
tried so far.

Trajectory, reference env, seed 7 — not plateaued:

| iteration | 1500 | 2200 | 2300 | 2500 | 2700 | 2900 | 2999 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| successes | 0/129 | 11 | 17 | 31 | 35 | 39 | 43/130 |
| best error | 29.2° | 19.4° | 19.0° | 15.9° | 14.2° | 14.0° | **10.8°** |

Caveat: three things changed at once — kernel, goal pinning, entropy — in a
single training seed, so attribution among them is untested (see Still open, below). Also, the
training task is no longer the reference task (the goal is pinned during training), though
every number above is measured in the reference env with drift on.

## 34. `warm-inv-pin-ent1e3-ext` — continuation, still climbing

Same config, `--init-from` run 33's `model_2999`, iterations 2999→4499, 2026-09-11 21:34–23:17.

Reference env, 3 eval seeds × ~130 episodes each:

| checkpoint | episodes < 5.7° | goals/episode | median best error |
| --- | --- | --- | --- |
| run 14 @2999 (control) | 1.8% | 0.02 | 28–30° |
| run 33 @2999 | 31.2% | 0.36 | 11–13° |
| run 34 @3450 | 42.5% | 0.51 | 7–10° |
| run 34 @4499 | **57.3%** | **0.77** | **5.6–5.7°** |

The median episode now reaches the success threshold. Still no plateau.

Training-side at iteration 4499: training error ~22.6°, action std 1.55, ~263 steps per episode
spent inside 5.7° (up from ~10 at the start of run 33), drops 0.25/episode.

A continuation to iteration 9000 was run on a rented A100 and then on past it locally. It
plateaus at ~iteration 8000 — see "## 37. `remote_brev-ext` / `night-ext` — the main line, and
where it converges", and the infrastructure note below for the rental.

---

## What changed in the code (2026-09-11)

- `mdp/rewards.py` — new `cube_orientation_inverse(env, command_name, object_name, eps=0.1)` =
  1/(err+eps). Bounded by construction at 1/eps.
- `config/env_cfg.py` — new `orientation_kernel` argument (`"linear"` default / `"inverse"`),
  new `success_threshold` argument (overrides *both* threshold sites together; `None` keeps the
  preset value). Both recorded in `build_kwargs`. Module constant `_INVERSE_WEIGHT_PER_LINEAR`
  derives the 1.79× weight so marginal reward matches the linear kernel at 130°.
- `scripts/train.py` — new `--orientation-kernel`, `--success-threshold`, and `--init-from`.
  `--init-from` loads a checkpoint's weights/optimizer/iteration count but trains in a *new* run
  directory, unlike `--resume-from`, which continues in the checkpoint's own directory and would
  overwrite that run's later checkpoints. `--resume-from` wins if both are given, which is what
  `supervise_run.sh` does after a reboot. The source checkpoint is recorded in
  `params/init_from.txt`.
- `scripts/eval_policy.py` — the judged-goal fix described in the Interlude above, plus
  `--env-success-threshold` (rebuilds the env's threshold; the pre-existing `--success-threshold`
  is reporting-only and does *not* change the env — the handoff's scoring commands assumed
  otherwise).
- `scripts/render.py` — `--overlay` (default on) stamps orientation error, goals reached and
  elapsed time on each frame; the console also logs the time of each goal reached.
- `scripts/run_queue.sh` — evals now run at 128 envs (files tagged `_n128`) and at both 0.1 and
  0.4 rad; added pinned-goal baselines.

## Videos (2026-09-11)

`eval/videos/warm-inv-pin-ent1e3_it2999_seed{1..7}.mp4`, 60 s each, reference env,
`--cube-priority 0`. Seed 4 reaches two goals (12.1 s and 28.5 s), seed 6 reaches one (5.1 s),
the other five reach none — consistent with the ~31% episode success rate at that checkpoint.
Renders run with `play=True`, so no observation noise or domain randomization.

## Infrastructure note (2026-09-11)

An A100 80GB PCIe was rented via Brev (org `AfthabS`; the other org is out of credits) to
continue run 34 to iteration 9000. Measured 2.23–2.33 s/iteration against the local GB10's
4.06 s median at 8192 envs — only 1.8× faster despite the class difference. Remote torch had to
be downgraded to 2.11.0+cu128 because the instance driver is 570.195 (CUDA 12.8) and the locked
torch 2.13 is a cu130 build that cannot see the GPU; warp and mjlab stay at the locked versions,
and every remote checkpoint is scored locally so the numbers stay comparable. Remote user is
`shadeform`. Use `uv run --no-sync` remotely — a plain `uv sync` reverts the torch downgrade.

---

## 35–36. `abl-nokernel` / `abl-nopin` — which of run 33's three changes did the work

Run 33 changed three things at once against a warm start from run 14's `model_1500` — the
inverse orientation kernel, the pinned goal, and `entropy_coef` 1e-3 — so the result was
unattributed. Leave-one-out rather than one-at-a-time, because the question is which change is
*necessary*, and run 32 already supplies the third cell. Every cell is a warm start from
`model_1500` to iteration 3000, so all four are directly comparable.

Deterministic eval, 3 seeds × ~130 episodes each, reference env (drift on), `--cube-priority 0`:

| cell | kernel | goal | entropy | reached < 5.7° | goals/ep | median best error |
| --- | --- | --- | --- | --- | --- | --- |
| control, run 14 @2999 | linear | drift | 1e-2 | 7/385 = 1.8% | 0.02 | 28.6 / 29.8 / 28.2° |
| 35 `abl-nokernel` | linear | **pinned** | 1e-3 | 34/392 = 8.7% | 0.10 | 25.5 / 28.0 / 23.5° |
| 36 `abl-nopin` | **inverse** | drift | 1e-3 | 48/390 = 12.3% | 0.14 | **12.1 / 12.5 / 15.3°** |
| 33 full recipe | **inverse** | **pinned** | 1e-3 | **121/388 = 31.2%** | 0.36 | 10.8 / 12.3 / 12.6° |
| 32 `warm-inv-pin` | inverse | pinned | 1e-2 | — | — | std runaway, stopped at 1650 |

**The kernel and the pinning do different jobs, and neither alone is enough.**

*The kernel closes the distance.* `abl-nopin` has no pinning and still drags median best error
from 28° to 12–15°, which is essentially the full recipe's 11–13°. Nearly all of the approach
improvement is the kernel's near-goal gradient.

*The pinned goal converts proximity into crossings.* At the same error, `abl-nopin` converts
12.3% of episodes against the full recipe's 31.2%. Getting close and getting there are separate
problems, and only the goal machinery fixes the second.

This confirms the kick arithmetic recorded above: under the drift a success kicks the goal
~160° away, costing ~190 discounted reward against a +5 bonus with the inverse kernel, so the
policy is trained to approach the threshold and stop short. "Low error, few crossings" is
exactly the signature that predicts. **The prediction was pre-registered in `run_queue.sh`
before either run started, and held on both cells** — had `abl-nopin` matched run 33, the
arithmetic would have been wrong and the pinning droppable.

`abl-nokernel`'s training curve says the same thing from the other side: it stops descending
around iteration 1700 and sits at 43–50° for 1200 iterations, where run 33 was through 30° and
still falling.

**Consequences.** Do not drop the pinning as a simplification. Do not run the inverse kernel
with the drift on — which is what `research/NEXT_EXPERIMENTS.md` Experiment 2 specifies, and
run 36 is now the measured reason not to. Attribution of the entropy setting rests on run 32
(std runaway without it) plus run 21 (1e-3 alone moves nothing), not on a dedicated cell.

---

## 37. `remote_brev-ext` / `night-ext` — the main line, and where it converges

Run 34 continued under an unchanged recipe — inverse orientation kernel, goal pinned during
training, `entropy_coef` 1e-3, seed 42, 8192 envs — in two segments that are one run:

- **4499 → 8999** on the rented A100 (`logs/rsl_rl/.../remote_brev-ext`). Checkpoints only; no
  console log was brought back from the instance, so there is no training-side trace for this
  stretch.
- **8999 → 11625** locally as `night-ext`, 2026-09-12 03:23–06:20, 2h56m, `--init-from` the
  A100's `model_8999`. Target was 14000; it took SIGTERM at 11625.

Nothing past 4499 had ever been scored. `night-ext` was stopped before its final checkpoint
existed, so the queue's own guard fired (`'night-ext' finished but model_13999.pt is missing —
stopping`) and the eval step that normally closes a queue entry never ran. The whole segment sat
dark until 2026-09-12, when it was scored retrospectively from the kept checkpoints.

### The curve, and where it flattens

Reference env (drift on), deterministic, `--cube-priority 0`, `--num-envs 128 --num-steps 700`,
seed 7, ~140 episodes per cell. "Goals per drop" is threshold entries ÷ `cube_fell`
terminations — the same quantity the playground paper reports as consecutive rotations before
failure.

| iteration | reach < 5.7° | goals/ep | median best | closure | spin | tip | goals per drop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2999 (run 33) | 33.1% | 0.40 | 10.78° | 91.1% | 92.8% | 90.0% | 7.4 |
| 4499 (run 34) | 50.7% | 0.70 | 5.72° | 94.6% | 93.0% | 95.1% | 8.5 |
| 5500 | 67.4% | 0.95 | 5.57° | 95.4% | 93.4% | 95.3% | 9.8 |
| 6750 | 75.5% | 1.01 | 5.36° | 95.7% | 94.4% | 95.7% | 8.3 |
| **8000** | **83.9%** | **1.14** | **5.30°** | 95.8% | 95.2% | 95.7% | 9.1 |
| 9000 | 83.0% | 1.15 | 5.39° | 95.9% | 93.3% | 96.0% | 9.5 |
| 10250 | 80.1% | 1.17 | 5.29° | 96.0% | 94.1% | 96.0% | 9.2 |
| 11625 | 78.4% | 1.14 | 5.29° | 95.9% | 94.2% | 95.6% | 9.3 |

**The 2999 and 4499 rows are single-seed (seed 7) and are not the pooled figures quoted in
sections 33 and 34** (31.2% and 57.3%, each pooled over 3 seeds × ~130 episodes). Read down this
column only against other rows in this table; the 50.7% at 4499 is not a regression from 57.3%,
it is the same checkpoint measured on one seed instead of three.

**The recipe works, and it converges at ~iteration 8000.** Everything after that is flat.

### The plateau is measured, not read off one seed

The single-seed curve appears to *decline* after 8000 — 83.9 → 78.4%. It does not. Both ends
re-scored at seeds 7/8/9:

| checkpoint | seed 7 | seed 8 | seed 9 | pooled | median best | goals/ep |
| --- | --- | --- | --- | --- | --- | --- |
| 8000 | 83.9% | 79.1% | 84.3% | **82.5%** | 5.3° at every seed | 1.14 / 1.14 / 1.28 |
| 11625 | 78.4% | 85.0% | 79.1% | **80.8%** | 5.3° at every seed | 1.14 / 1.29 / 1.09 |

82.5% against 80.8%, with a ±3% per-seed spread at each and an identical 5.3° median best error.
The two checkpoints are indistinguishable. The honest statement is **no further improvement**,
not a decline — and the apparent 6-point drop on seed 7 alone is exactly the artefact the
"use ≥3 eval seeds from here on" entry in Still open was written to prevent.

**The overnight run bought nothing measurable.** 2,626 iterations, 2h56m of GB10, from a
checkpoint that was already converged to one that scores the same. So did the last ~1000
iterations of the A100 segment. The rental paid for 4499 → ~8000, which is where the whole gain
lives; everything past 8000 was spent.

### What the ceiling is made of — the pinned-goal eval

`model_11625` re-scored with `--goal-drift False --goal-resample-on-success False`, against run
33 @2999 under the same pinning:

| | run 33 @2999 pinned | run 37 @11625 pinned |
| --- | --- | --- |
| HELD @ 0.1 rad (10 consecutive steps) | 28.1% | **73.1%** (98/134) |
| SETTLED (\|w\| < 0.2 rad/s) | 28.9% | **73.9%** |
| goals/episode | 3.16 | **5.61** |
| median best error | 11.79° | **0.73°** |
| p10 best error | 0.62° | **0.15°** |

A median best error of 0.73° with 73% of episodes *holding* the goal for half a second is not a
policy that lacks precision. Under the drift the same checkpoint reaches 5.29° and holds
nothing — **the drift-on ceiling is the kick, not the policy.** That is the kick arithmetic from
runs 35–36 showing up in the converged policy: a success kicks the goal ~160° away, which under
the inverse kernel costs ~190 discounted reward against a +5 bonus, so the optimum remains
"arrive, then leave". The pinned number is what the hand can do; the drift-on number is what the
task pays it to do.

### The training metric is not a convergence signal

Training-side across 9000 → 11625: `orientation_error` flat at 12–18°, `success` flat at
0.60–0.69, `goals_reached` ~500/episode, action std drifting 1.18 → 1.29. Flat — and the
deterministic reference score is flat over the same span, so here the two agree.

They did not agree earlier. The training metric was already sitting at this level at iteration
9000, while the reference score had been climbing 50.7% → 83.9% across 4499 → 8000 and was only
just finishing. A flat training curve therefore does not mean a converged policy, and did not for
~3500 iterations of this run. This is the same lesson as "drop training-time
`consecutive_success` as a leading indicator" (Still open, runs 29–30), now from the other
direction: it under-reported real progress instead of over-reporting none. **Convergence has to
be called on the deterministic eval.**

### The residual is the tip-over, again

At 11625, seed 7: 30 of 139 episodes never reach 5.7°. **Only 2 of those 30 end in a drop** — the
rest survive the full episode and stall. Their best error is a median 18.1° (range 6.7–117.4°),
split **14.5° tip against 8.4° spin**, down from 80° tip / 55° spin at episode start. 10 of the
30 get inside 12° and stop there. At 8000 the picture is the same: 23 failures, 2 drops, 21.0°
median, 18.9° tip against 9.1° spin.

This is the horizontal-axis component that has run through this file since the residual-error
decomposition — the policy closes the spin and stalls on the tip-over. It has not changed
character; it has been confined. It used to describe every episode and now describes a fifth of
them, and within those it is still 1.7× the spin component.

### Against the reference benchmark

~9.3 successes per drop, against the MuJoCo Playground paper's ten hardware trials quoted under
"What the reference actually achieves": median 3.5, mean 7.1, best 27 consecutive rotations
before failure.

This is a scale comparison and not a parity claim. Ours is sim against their hardware; it is a
deterministic mean-action rollout (observation noise and domain randomization are on — `play` is
False — but the exploration noise is not); and the 700-step cap truncates long runs, which
censors the upper tail their "best 27" comes from.

With that said: **the 0/32 that opened this investigation is closed.** Worth recording what
closed it. The 2026-09-06 audit concluded "the reference definitively works, our 0/32 is a real
gap, and the difference is the hand." The gap was closed without touching the hand — same
LeapXELA model, same pads, same contact model, same palm angle — by changing the reward kernel
and the goal machinery. The bare-hand control has still never been run, so "the difference is the
hand" was never directly tested; this run is evidence against it.

### Artifacts

`eval/mainline_it{5500,6750,8000,9000,10250,11625}_n128_det.json` (+ `_eval.txt`), the seed
replicates `eval/mainline_it{8000,11625}_n128_s{8,9}_det.json`, and the pinned
`eval/mainline_it11625_trainenv_n128_det.json`. All at `--cube-priority 0`; the `mainline_`
prefix is used because these score checkpoints from two run directories as one series.

### Infrastructure

- `SAVE_INTERVAL` went 50 → 25 in `run_queue.sh` (~1.7 min of training per save at 8192 envs).
  This host reboots uncleanly, and at 50 a reboot cost up to ~3.4 min of work; it also made the
  retrospective scoring above finer-grained than it would otherwise have been.
- The stop at 11625 was **not a crash**. `train.py` exited rc=143 (SIGTERM) and the queue's
  missing-final-checkpoint guard then wrote `STOP_SUPERVISOR` rather than looping on a target it
  could not reach.
- The `@reboot` cron hook fired at 10:21 and correctly stood down on `STOP_SUPERVISOR`
  (`a queue is already running; leaving it alone` → `STOP_SUPERVISOR present; standing down`).
  The guard behaved as designed in both places; the cost was ~4 h of idle GPU, which is the
  price of a guard that does not restart a run nobody has looked at yet.
