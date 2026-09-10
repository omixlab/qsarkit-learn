"""Check that the README's claims about the package are true.

A README is the first thing anyone reads and the last thing anyone updates.
These tests do not execute its snippets — they are deliberately elided
fragments, not doctests — but they do verify that every API the README names
actually exists, and that the package layout table matches the package.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import qsarkit

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_exists(readme: str) -> None:
    assert readme.startswith("# qsarkit")


def test_layout_table_matches_the_package(readme: str) -> None:
    """Every subpackage is documented, and nothing removed is still listed."""
    section = readme.split("## Package layout", 1)[1].split("## ", 1)[0]
    listed = set()
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        first_column = line.split("|")[1]
        # A row may name several modules, e.g. "`neighbors`, `cluster`".
        listed.update(re.findall(r"`([\w_]+)`", first_column))

    actual = set(qsarkit._SUBPACKAGES)
    assert not listed - actual, f"README documents modules that do not exist: {listed - actual}"
    assert not actual - listed, f"README is missing modules: {actual - listed}"


def test_named_imports_resolve(readme: str) -> None:
    """Every `from qsarkit.x import Y` in the README actually imports."""
    pattern = re.compile(r"^from (qsarkit[\w.]*) import ([^\n]+)$", flags=re.M)
    checked = 0
    for module_name, names in pattern.findall(readme):
        module = importlib.import_module(module_name)
        for name in (n.strip() for n in names.split(",")):
            if not name or name == "*":
                continue
            assert hasattr(module, name), f"{module_name} has no {name!r}"
            checked += 1
    assert checked > 10, "README stopped showing imports; the check is now vacuous"


def test_extras_table_matches_pyproject(readme: str) -> None:
    """The advertised extras are the ones that exist."""
    section = readme.split("Feature-specific extras:", 1)[1].split("## ", 1)[0]
    listed = set(re.findall(r"^\| `([\w_]+)` \|", section, flags=re.M))

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = pyproject.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    defined = set(re.findall(r"^([\w_]+) = \[", block, flags=re.M))

    # `dev`, `docs` and `all` are not feature extras and are documented elsewhere.
    defined -= {"dev", "docs", "all"}
    assert listed == defined, f"README extras {listed} != pyproject extras {defined}"


def test_notebook_links_resolve(readme: str) -> None:
    """Every notebook the README links to is present."""
    links = re.findall(r"\]\((notebooks/[\w./-]+)\)", readme)
    assert links, "README no longer links to the notebooks"
    for link in links:
        assert (ROOT / link).exists(), f"README links to a missing file: {link}"
