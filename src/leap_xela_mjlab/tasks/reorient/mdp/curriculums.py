"""Curriculum terms for cube reorientation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class goal_difficulty:
  """Widen the goal sampling range as the policy actually reaches goals.

  Difficulty is the fraction of pi used to sample goal offsets. It starts small
  so goals are reachable, and grows only while the policy is clearing at least
  ``promote_at`` goals per episode. Without this the task has no solvable
  instances to learn from: goals are drawn up to +-pi from step one, the policy
  never reaches one, and it settles on the pose that minimises average error.

  Promotion is driven by an EMA of goals reached per episode so a lucky batch of
  resets cannot ratchet difficulty up faster than competence.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    del cfg, env
    self.ema: float | None = None
    self.calls: int = 0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids  # EMA is global and intentionally survives resets.

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    command_name: str = "goal_orientation",
    promote_at: float = 1.0,
    demote_at: float = 0.2,
    step: float = 0.05,
    min_difficulty: float = 0.05,
    max_difficulty: float = 1.0,
    ema_alpha: float = 0.05,
    update_every: int = 50,
  ) -> dict[str, float]:
    term = env.command_manager.get_term(command_name)
    ids = slice(None) if env_ids is None else env_ids
    reached = term.success_count[ids]
    if reached.numel() > 0:
      mean_reached = float(reached.mean())
      self.ema = (
        mean_reached
        if self.ema is None
        else (1.0 - ema_alpha) * self.ema + ema_alpha * mean_reached
      )
      # This runs on every _reset_idx -- roughly once per env-step with staggered
      # resets -- so difficulty must only move on a slow cadence, or it saturates
      # long before the EMA reflects real competence.
      self.calls += 1
      if self.calls % update_every != 0:
        return {
          "difficulty": term.difficulty,
          "goals_per_episode": float(self.ema),
        }
      if self.ema >= promote_at:
        term.difficulty = min(term.difficulty + step, max_difficulty)
      elif self.ema < demote_at:
        # Back off at half rate: recovering competence is slower than losing it.
        term.difficulty = max(term.difficulty - 0.5 * step, min_difficulty)
    return {
      "difficulty": term.difficulty,
      "goals_per_episode": float(self.ema) if self.ema is not None else 0.0,
    }
