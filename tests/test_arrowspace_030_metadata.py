"""Guard the ArrowSpace 0.30 packaging and diagnostics contract."""

import json
import tomllib
from pathlib import Path

from click.testing import CliRunner
from packaging.requirements import Requirement

from arrowspace_tuner.cli import main
from arrowspace_tuner.mcp_server import _tool_get_tuner_info


def test_manifest_accepts_arrowspace_030() -> None:
    manifest = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    requirement = next(
        Requirement(dep)
        for dep in manifest["project"]["dependencies"]
        if Requirement(dep).name == "arrowspace"
    )
    assert "0.29.0" not in requirement.specifier
    assert "0.30.0" in requirement.specifier
    assert "0.30.1" in requirement.specifier
    assert "0.31.0" not in requirement.specifier


def test_cli_reports_arrowspace_030_requirement() -> None:
    result = CliRunner().invoke(main, ["version", "--format", "json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["arrowspace_requirement"] == ">=0.30.0,<0.31"


def test_mcp_reports_arrowspace_030_requirement() -> None:
    assert _tool_get_tuner_info()["arrowspace_requirement"] == ">=0.30.0,<0.31"
