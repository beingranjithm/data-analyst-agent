"""The planner prompt must be identical on every run, or the reply cache can never hit."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_planner_prompt_is_identical_across_processes():
    if not (ROOT / "sample_data" / "sales_sample.csv").exists():
        pytest.skip("sample data not found")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_prompt_stability.py")],
                       capture_output=True, text=True, timeout=240, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
