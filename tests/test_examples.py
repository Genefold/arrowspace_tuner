"""
test_examples.py — execute the maintained examples.

examples/quickstart.py and examples/power_user.py are the executable
source of truth for the README snippets. Running them as subprocesses
proves the documented import/build workflow works on the installed
package (including the built wheel — see the release smoke test).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _run_example(name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / name)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{name} failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_quickstart_example_runs() -> None:
    """examples/quickstart.py — README Quickstart flow end-to-end."""
    _run_example("quickstart.py")


def test_power_user_example_runs() -> None:
    """examples/power_user.py — README Power-user flow end-to-end."""
    _run_example("power_user.py")
