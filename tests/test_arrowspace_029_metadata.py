"""Guard the ArrowSpace 0.29 packaging compatibility contract."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_manifest_accepts_arrowspace_029() -> None:
    manifest = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    requirement = next(
        Requirement(dep)
        for dep in manifest["project"]["dependencies"]
        if Requirement(dep).name == "arrowspace"
    )
    assert "0.29.0" in requirement.specifier
    assert "0.30.0" not in requirement.specifier
