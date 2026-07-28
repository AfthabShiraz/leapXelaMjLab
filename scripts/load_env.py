"""Load leapXelaMjLab/.env into os.environ (e.g. WANDB_API_KEY)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project_dotenv() -> None:
  """Load ``<project_root>/.env`` if present. Existing env vars are not overridden."""
  load_dotenv(_PROJECT_ROOT / ".env", override=False)
