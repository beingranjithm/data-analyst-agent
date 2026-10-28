"""Day 23a: the requirements.txt writer reads the project's real imports and pins installed versions."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import make_requirements as mr  # noqa: E402


def test_it_finds_outside_imports_only(tmp_path):
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "a.py").write_text("import os\nimport pandas as pd\nfrom agents import b\nfrom google.adk import X\n"
                                              "from . import sibling\nfrom .helper import z\nimport tools.charts\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "t.py").write_text("import plotly.express as px\nimport json, numpy\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("import pytest\n")
    assert mr.imported_names(("agents", "tools"), tmp_path) == {"pandas", "google", "plotly", "numpy"}


def test_a_broken_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "bad.py").write_text("def (:\n")
    (tmp_path / "app" / "ok.py").write_text("import streamlit\n")
    assert mr.imported_names(("app",), tmp_path) == {"streamlit"}


def test_pinning_maps_modules_to_the_installed_distributions():
    lines, missing = mr.pinned({"pandas", "google"}, extra=())
    names = [l.split("==")[0].lower() for l in lines]
    assert "pandas" in names and any(n.startswith("google") for n in names) and missing == []
    assert all("==" in l and l.split("==")[1] for l in lines)
    assert lines == sorted(lines, key=lambda x: x.lower())


def test_a_module_that_is_not_installed_is_reported():
    lines, missing = mr.pinned({"definitely_not_installed_pkg"}, extra=())
    assert lines == [] and missing == ["definitely_not_installed_pkg"]


def test_the_optional_extra_is_skipped_quietly_when_absent():
    assert mr.pinned(set(), extra=("definitely_not_installed_pkg",)) == ([], [])


def test_the_real_project_needs_only_known_packages():
    names = mr.imported_names()
    assert {"streamlit", "pandas", "plotly", "numpy"} <= names and "pytest" not in names
    assert not (names & {"agents", "tools", "ui", "app"})
