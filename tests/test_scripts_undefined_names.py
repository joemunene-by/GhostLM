"""scripts/ is outside the ruff gate, but an undefined name there is a crash, not style."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("ruff") is None and not (Path(sys.executable).parent / "ruff").exists(),
                    reason="ruff not installed")
def test_no_undefined_names_in_scripts():
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--isolated", "--select", "F821,F822,F823",
         "scripts/", "data/"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout
