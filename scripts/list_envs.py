"""List registered mjlab tasks (including LeapXELA)."""

import tyro
from prettytable import PrettyTable

import mjlab.tasks  # noqa: F401
import leap_xela_mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks


def list_environments(keyword: str | None = None):
  table = PrettyTable(["#", "Task ID"])
  table.title = "Available Environments"
  table.align["Task ID"] = "l"

  idx = 0
  for task_id in list_tasks():
    if keyword and keyword.lower() not in task_id.lower():
      continue
    table.add_row([idx + 1, task_id])
    idx += 1

  print(table)
  return idx


if __name__ == "__main__":
  tyro.cli(list_environments)
