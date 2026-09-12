"""Widen a checkpoint's ACTOR input layer so it can warm-start into a larger obs.

Why this exists. Every from-scratch reward/obs edit this project has tried is
0 for 4 at surviving the iteration-300 tip-over step (runs 20, 22, 23, 31), so a
cell that changes the task has to warm-start from a post-breakthrough checkpoint
or it tests nothing. But ``runner.load()`` (scripts/train.py) is a strict
``load_state_dict``, and adding an observation term changes the actor's first
Linear from (512, 57) to (512, 60) -- the load simply fails.

The graft pads the new input columns with ZEROS and the normalizer's new entries
with mean 0 / var 1 / std 1, so at iteration 0 the grafted policy produces
bit-identical actions to the original: the new observations are multiplied by
zero weights. The run is therefore a true continuation of the source checkpoint,
and anything that moves afterwards is the new observation being learned rather
than a restart artifact.

The Adam moments for that weight are padded with zeros too. rsl_rl's
``PPO.load`` restores the optimizer unconditionally, and torch raises on a shape
mismatch, so skipping this would make the checkpoint unloadable.

New dims are appended at the END, which is only correct if the new observation
term is the LAST entry of the actor group's dict -- mjlab concatenates terms in
insertion order. ``actor_cube_ang_vel`` in tasks/reorient/config/env_cfg.py is
appended after ``last_action`` for exactly this reason.

Usage:
  python scripts/graft_obs.py --src <in.pt> --dst <out.pt> --new-dims 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _pad_cols(t: torch.Tensor, n: int, fill: float) -> torch.Tensor:
  pad = torch.full((t.shape[0], n), fill, dtype=t.dtype, device=t.device)
  return torch.cat([t, pad], dim=1)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", required=True)
  ap.add_argument("--dst", required=True)
  ap.add_argument("--new-dims", type=int, default=0)
  ap.add_argument(
    "--scale-critic",
    type=float,
    default=None,
    help=(
      "Multiply the critic's output layer (and its Adam moments) by this factor. "
      "Use when warm-starting across a DISCOUNT change: the value of a reward "
      "stream scales as 1/(1-gamma), so moving gamma 0.99 -> 0.997 needs "
      "(1-0.99)/(1-0.997) = 3.33 or the critic starts 3.3x too small and the run "
      "spends its first few hundred iterations recalibrating instead of testing "
      "the hypothesis."
    ),
  )
  args = ap.parse_args()

  ck = torch.load(args.src, map_location="cpu", weights_only=False)
  actor = ck["actor_state_dict"]
  n = args.new_dims

  if args.scale_critic is not None:
    k = float(args.scale_critic)
    critic = ck["critic_state_dict"]
    # Last Linear of the critic MLP: the only parameter whose output is V.
    last = max(
      int(key.split(".")[1])
      for key in critic
      if key.startswith("mlp.") and key.endswith(".weight")
    )
    for suffix in ("weight", "bias"):
      critic[f"mlp.{last}.{suffix}"] = critic[f"mlp.{last}.{suffix}"] * k
    # Adam moments for those two tensors: first moment is linear in the
    # gradient, second is quadratic, so they scale by k and k**2.
    st = ck["optimizer_state_dict"]["state"]
    shapes = {
      tuple(critic[f"mlp.{last}.weight"].shape),
      tuple(critic[f"mlp.{last}.bias"].shape),
    }
    hits = [i for i, v in st.items() if "exp_avg" in v and tuple(v["exp_avg"].shape) in shapes]
    assert len(hits) == 2, f"expected 2 optimizer entries for the critic head, got {hits}"
    for i in hits:
      st[i]["exp_avg"] = st[i]["exp_avg"] * k
      st[i]["exp_avg_sq"] = st[i]["exp_avg_sq"] * (k * k)
    print(f"[OK] critic head mlp.{last} scaled by {k} (Adam moments by {k} / {k * k})")
    if n == 0:
      Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
      torch.save(ck, args.dst)
      print(f"[OK] {args.src}\n  -> {args.dst}\n  critic rescaled only, iter={ck['iter']}")
      return

  old_dim = actor["mlp.0.weight"].shape[1]
  new_dim = old_dim + n
  assert actor["obs_normalizer._mean"].shape == (1, old_dim), "unexpected normalizer width"

  # Normalizer: new dims pass through unchanged (mean 0, unit variance).
  actor["obs_normalizer._mean"] = _pad_cols(actor["obs_normalizer._mean"], n, 0.0)
  actor["obs_normalizer._var"] = _pad_cols(actor["obs_normalizer._var"], n, 1.0)
  actor["obs_normalizer._std"] = _pad_cols(actor["obs_normalizer._std"], n, 1.0)

  # First Linear: new columns contribute nothing.
  actor["mlp.0.weight"] = _pad_cols(actor["mlp.0.weight"], n, 0.0)

  # Adam moments for that same weight, found by shape -- the critic's first
  # layer is (512, 91) and every other entry is 1-D or a different rank, so
  # (512, old_dim) is unambiguous. Assert that, rather than trusting it.
  hits = [
    i for i, st in ck["optimizer_state_dict"]["state"].items()
    if "exp_avg" in st and tuple(st["exp_avg"].shape) == (actor["mlp.0.weight"].shape[0], old_dim)
  ]
  assert len(hits) == 1, f"expected exactly one optimizer entry of shape (*, {old_dim}), got {hits}"
  st = ck["optimizer_state_dict"]["state"][hits[0]]
  for key in ("exp_avg", "exp_avg_sq"):
    st[key] = _pad_cols(st[key], n, 0.0)

  # The grafted columns must be exactly zero, or the run is not a continuation.
  assert torch.count_nonzero(actor["mlp.0.weight"][:, old_dim:]) == 0

  Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
  torch.save(ck, args.dst)
  print(
    f"[OK] {args.src}\n  -> {args.dst}\n"
    f"  actor obs {old_dim} -> {new_dim} (+{n} zero-init), optimizer entry {hits[0]} padded, "
    f"iter={ck['iter']}"
  )


if __name__ == "__main__":
  main()
