"""Check that the README's factual claims about the package hold.

The README is hand-maintained prose, so these tests deliberately do not
police its structure -- headings and tables may come and go. They check
only the things that silently become false as the code changes:

* every ``from qsarkit... import X`` it shows actually imports;
* every module it names exists;
* every extra it advertises is defined in ``pyproject.toml``;
* every relative link it makes resolves to a real file.

Each check skips when the README does not contain that kind of content,
so rewriting a section never fails the suite for the wrong reason.
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
    if not README.is_file():
        pytest.skip("no README.md")
    return README.read_text(encoding="utf-8")


def test_readme_names_the_package(readme: str) -> None:
    assert "qsarkit" in readme[:200].lower()


def test_named_imports_resolve(readme: str) -> None:
    """Every ``from qsarkit.x import Y`` in the README actually imports."""
    pattern = re.compile(r"^\s*from (qsarkit[\w.]*) import ([^\n#]+)$", flags=re.M)
    matches = pattern.findall(readme)
    if not matches:
        pytest.skip("README shows no qsarkit imports")

    broken = []
    for module_name, names in matches:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            broken.append(f"{module_name} (module does not exist)")
            continue
        for name in (n.strip().rstrip(",") for n in names.split(",")):
            if not name or name == "*" or name.startswith("("):
                continue
            if not hasattr(module, name):
                broken.append(f"{module_name}.{name}")
    assert not broken, f"README references API that no longer exists: {broken}"


def test_modules_it_names_exist(readme: str) -> None:
    """Every ``qsarkit.<module>`` mentioned is a real subpackage."""
    mentioned = set(re.findall(r"qsarkit\.(\w+)", readme))
    # Names that are classes or functions reached through a module path,
    # not subpackages themselves.
    mentioned -= {"__version__"}
    actual = set(qsarkit._SUBPACKAGES)
    unknown = {name for name in mentioned if name not in actual}
    # A mention may be a class (qsarkit.models.QSARRegressor -> "models"),
    # so only flag first-segment names that are not subpackages at all.
    assert not unknown, f"README names modules that do not exist: {sorted(unknown)}"


def test_advertised_extras_exist(readme: str) -> None:
    """Every extra shown in a pip install line is defined in pyproject."""
    shown: set[str] = set()
    for group in re.findall(r"pip install [\"']?qsarkit\[([^\]]+)\]", readme):
        shown.update(part.strip() for part in group.split(","))
    if not shown:
        pytest.skip("README advertises no extras")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = pyproject.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    defined = set(re.findall(r"^([\w_]+) = \[", block, flags=re.M))
    assert shown <= defined, (
        f"README advertises undefined extras: {sorted(shown - defined)}. "
        f"Defined: {sorted(defined)}"
    )


def test_relative_links_resolve(readme: str) -> None:
    """Every relative Markdown link points at a file that exists."""
    links = [
        target
        for target in re.findall(r"\]\(([^)#]+)\)", readme)
        if not target.startswith(("http://", "https://", "mailto:", "#"))
    ]
    if not links:
        pytest.skip("README has no relative links")
    missing = [link for link in links if not (ROOT / link.lstrip("./")).exists()]
    assert not missing, f"README links to missing files: {missing}"
