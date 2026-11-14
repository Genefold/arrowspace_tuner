"""Keep tuner releases aligned to the tested ArrowSpace compatibility line."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_and_lockfile_match() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    project = next(p for p in lock["package"] if p["name"] == "arrowspace-tuner")
    assert manifest["project"]["version"] == project["version"] == "0.30.1"
    requirement = next(
        Requirement(dep)
        for dep in manifest["project"]["dependencies"]
        if Requirement(dep).name == "arrowspace"
    )
    assert str(requirement.specifier) == "<0.31,>=0.30.0"


def test_release_docs_describe_eps_k_only_and_breaking_migration() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "and `tau` for your corpus" not in readme
    assert "tuned automatically" not in readme
    assert "auto-discover eps, k, tau" not in readme
    assert "auto-discover eps and k" in readme
    assert "0.30.x compatibility line" in readme
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert "## [0.30.1]" in changelog
    release_section = changelog.split("## [0.30.1]", 1)[1].split("## [0.30.0]", 1)[0]
    assert "**Breaking:**" in release_section
    assert "Versioning follows [Semantic Versioning]" not in changelog
    assert "Versioning follows the ArrowSpace compatibility line" in changelog
    notes = (ROOT / "docs/releases/0.30.1.md").read_text()
    assert "breaking tuner API changes" in notes
    assert "tau_low" in notes and "best_tau" in notes
