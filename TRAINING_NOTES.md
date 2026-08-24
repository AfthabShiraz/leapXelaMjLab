# Task 1 — PPO cube reorientation (no touch): what we tried and what happened

Environment: `Mjlab-LeapXELA-Cube-Reorient` (mjlab manager-based API, MuJoCo Warp, RSL-RL PPO).
All runs on a single GPU, `--num-envs 4096`, logs under
`logs/rsl_rl/leap_xela_cube_reorient/<timestamp>_<run-name>/`.

Status as of 2026-08-22: **the policy reliably learns to hold the cube, but not to reorient it.**
Every configuration converges to a "park the cube at a compromise pose and farm episode
length" solution. The curriculum run is the first one that reaches goals at all, and only
while difficulty is small.

---

## Fixed setup (common to all runs)

| Item | Value |
| --- | --- |
| Envs / GPU | 4096 |
| Sim timestep / decimation | 0.01 s, decimation 5 → 50 ms control step |
| Integrator / solver | Euler, Newton, 5 iterations, 8 ls-iterations |
| `nconmax` / `njmax` | 48 / 120 |
| Episode length | 50 s (1000 control steps) |
| Actor / critic MLP | 512-256-128, ELU, obs normalization |
| PPO | 24 steps/env, 5 epochs, 4 minibatches, lr 3e-4 adaptive, γ 0.99, λ 0.95, desired KL 0.01 |
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

**Conclusion: friction is not the bottleneck.** A 20× change in torsional friction moves
nothing measurable. Cube friction DR was re-enabled for subsequent runs.

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

## Where this leaves us

Established:
- 4096 envs is the throughput sweet spot (~90 k FPS); more envs buys nothing.
- The hand learns to hold the cube within ~250 iterations in every configuration, and
  `cube_fell` drops to <0.5%.
- Friction (0.05–1.0 torsional) is not the limiting factor.
- MJX goal drift is not the limiting factor.
- Longer training is not the limiting factor — everything plateaus by ~250–1000 iterations.
- The original reward is farmable: holding the cube at ~1.5 rad error out-earned any attempt
  to reorient. Reweighting (`orientation` 5→1, `orientation_fine` added, threshold 0.1→0.4)
  removed the farming payout but did not by itself produce reorientation.
- Relative goals + a difficulty curriculum are what finally produced *any* successes, and
  they prove the task is reachable when goals are near the current pose.

Still open — the policy does not actually rotate the cube. `cube_ang_speed` sits near 1 rad/s
in every converged run, i.e. the hand holds still rather than manipulating.

Candidate next steps:
- Promotion criterion is too loose: `promote_at: 1.0` goal/episode lets difficulty ratchet on
  marginal competence. Try `promote_at` 3–5, and/or gate promotion on orientation error
  rather than goal count.
- Nothing in the reward pays for *moving* the cube. The `orientation_progress` term was the
  right idea but was tested against absolute goals (run #4), where it had no reachable
  target; worth re-testing on top of the curriculum.
- Action scale 0.5 with `ema_alpha` 1.0 and a 50 ms control step may simply be too coarse for
  in-hand rotation — worth an ablation.
- `init_std` 0.5 / `entropy_coef` 0.002 may be under-exploring; the baseline's std drifted
  *up* to 2.6, which suggests the opposite problem existed before. Untested in between.
- The hyperparameter sweep infrastructure (`hyperparameter_search/run_sweep.py`, grid over
  cube size / friction / seed) is written but **has not been run** — no `best_run.yaml` exists.
  Given the friction ablation result, a size/friction sweep is probably not the best use of it.

Not started: the flex/touch half of task 1 (submodule bump to the flex model, touch
observations behind a flag).
