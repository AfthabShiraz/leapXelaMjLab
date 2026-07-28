"""Grid-search cube hyperparameters and train with WandB logging.

Usage (from leapXelaMjLab root):

  uv run python hyperparameter_search/run_sweep.py
  uv run python hyperparameter_search/run_sweep.py --config hyperparameter_search/config.yaml
"""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from load_env import load_project_dotenv  # noqa: E402

load_project_dotenv()

import yaml

import mjlab.tasks  # noqa: F401, E402
import leap_xela_mjlab.tasks  # noqa: F401, E402
from train import TrainConfig, launch_training  # noqa: E402


def _as_list(value: Any) -> list[Any]:
  if isinstance(value, list):
    return value
  return [value]


def _load_config(path: Path) -> dict[str, Any]:
  with path.open() as f:
    return yaml.safe_load(f) or {}


def _build_param_grid(raw: dict[str, Any]) -> list[dict[str, Any]]:
  """Cartesian product of swept cube physics and seeds."""
  base_half = float(raw.get("cube_half_size", 0.035))
  scales = _as_list(raw.get("cube_scale", [1.0]))
  friction_cfg = raw.get("cube_friction", {}) or {}
  sliding_vals = _as_list(friction_cfg.get("sliding", [0.3]))
  torsional_vals = _as_list(friction_cfg.get("torsional", 0.05))
  seeds = _as_list(raw.get("seed", [42]))

  combos: list[dict[str, Any]] = []
  for scale, sliding, torsional, seed in itertools.product(
    scales, sliding_vals, torsional_vals, seeds
  ):
    half_size = base_half * float(scale)
    combos.append(
      {
        "cube_half_size": half_size,
        "cube_friction_sliding": float(sliding),
        "cube_friction_torsional": float(torsional),
        "cube_scale": float(scale),
        "seed": int(seed),
      }
    )
  return combos


def _run_name(params: dict[str, Any]) -> str:
  return (
    f"hs{params['cube_half_size']:.4f}"
    f"_mu{params['cube_friction_sliding']:.3f}"
    f"_tz{params['cube_friction_torsional']:.3f}"
    f"_sc{params['cube_scale']:.2f}"
    f"_s{params['seed']}"
  )


def _metric_value(metrics: dict[str, Any], metric_key: str) -> float | None:
  value = metrics.get(metric_key)
  if value is None:
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _write_best_run(
  output_path: Path,
  best: dict[str, Any],
  config_path: Path,
) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w") as f:
    yaml.safe_dump(best, f, sort_keys=False, default_flow_style=False)
  print(f"[sweep] wrote best run to {output_path}")


def run_sweep(config_path: Path) -> None:
  raw = _load_config(config_path)
  task_id = raw.get("task_id", "Mjlab-LeapXELA-Cube-Reorient")
  num_envs = raw.get("num_envs")
  max_iterations = raw.get("max_iterations")
  wandb_project = raw.get("wandb_project", "leap_xela_hparam_search")
  wandb_tags = tuple(raw.get("wandb_tags", ["hyperparameter_search"]))
  upload_model = raw.get("upload_model", False)
  gpu_ids = raw.get("gpu_ids", [0])
  best_metric = raw.get("best_metric", "Train/mean_reward")

  combos = _build_param_grid(raw)
  print(f"[sweep] {len(combos)} runs from {config_path}")

  best_score: float | None = None
  best_record: dict[str, Any] | None = None
  all_runs: list[dict[str, Any]] = []

  for i, params in enumerate(combos, start=1):
    run_name = _run_name(params)
    print(f"[sweep] run {i}/{len(combos)}: {run_name}")

    train_cfg = TrainConfig(
      num_envs=num_envs,
      max_iterations=max_iterations,
      seed=params["seed"],
      gpu_ids=gpu_ids,
      cube_half_size=params["cube_half_size"],
      cube_friction_sliding=params["cube_friction_sliding"],
      cube_friction_torsional=params["cube_friction_torsional"],
      run_name=run_name,
      wandb_project=wandb_project,
      wandb_tags=wandb_tags,
      upload_model=upload_model,
    )
    log_dir, metrics = launch_training(task_id, train_cfg)
    score = _metric_value(metrics, best_metric)

    run_record = {
      "run_name": run_name,
      "log_dir": str(log_dir.resolve()),
      "parameters": params,
      "best_metric": best_metric,
      "best_metric_value": score,
      "wandb_run_id": metrics.get("wandb_run_id"),
    }
    all_runs.append(run_record)

    if score is None:
      print(f"[sweep] warning: metric '{best_metric}' missing for {run_name}")
      continue

    if best_score is None or score > best_score:
      best_score = score
      best_record = {
        **run_record,
        "metrics": metrics,
      }

  if best_record is None:
    print("[sweep] no run produced a valid best_metric; skipping best_run.yaml")
    return

  best_output = {
    "selected_at": datetime.now(timezone.utc).isoformat(),
    "sweep_config": str(config_path.resolve()),
    "best_metric": best_metric,
    "best_metric_value": best_score,
    "run_name": best_record["run_name"],
    "log_dir": best_record["log_dir"],
    "wandb_run_id": best_record["metrics"].get("wandb_run_id"),
    "wandb_run_name": best_record["metrics"].get("wandb_run_name"),
    "wandb_project": best_record["metrics"].get("wandb_project"),
    "parameters": best_record["parameters"],
    "total_runs": len(combos),
    "all_runs": all_runs,
  }

  best_run_path = config_path.parent / "best_run.yaml"
  _write_best_run(best_run_path, best_output, config_path)
  print(
    f"[sweep] best run: {best_record['run_name']} "
    f"({best_metric}={best_score})"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Leap-XELA hyperparameter grid search")
  parser.add_argument(
    "--config",
    type=Path,
    default=PROJECT_ROOT / "hyperparameter_search" / "config.yaml",
    help="YAML file describing the search grid",
  )
  args = parser.parse_args()
  run_sweep(args.config.resolve())


if __name__ == "__main__":
  main()
