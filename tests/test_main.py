import subprocess
import sys
from pathlib import Path

from tamis import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_version_flag_prints_version_and_exits_without_opening_a_window():
    result = subprocess.run(
        [sys.executable, "main.py", "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"Tamis {__version__}"
