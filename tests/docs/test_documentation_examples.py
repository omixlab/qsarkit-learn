"""Execute every ``>>>`` example in ``docs-sphinx/source`` and in the package itself.

Two separate guarantees:

* :func:`test_documentation_example` runs the examples written in the
  reStructuredText pages under ``docs-sphinx/source``.
* :func:`test_docstring_example` runs the ``Examples`` sections of the
  package's own docstrings, which is what autodoc renders into the API
  reference.

Both are collected as individual test cases so a failure names the exact
file that went stale.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
import re
import sys
from pathlib import Path
from typing import List

import pytest

import qsarkit
from qsarkit.base import OptionalDependencyError

ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = ROOT / "docs-sphinx" / "source"

# Output-formatting leniency. NORMALIZE_WHITESPACE lets an expected array span
# several lines for readability; ELLIPSIS lets an example show the shape of a
# long repr without pinning every character of it.
OPTIONFLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL


class _Runner(doctest.DocTestRunner):
    """A runner that distinguishes "wrong" from "not installed here".

    An example demonstrating ChemBERTa or Mol2Vec is still a correct
    example on a machine without ``transformers`` or ``gensim``; it just
    cannot be checked there. Marking those `+SKIP` would leave them
    unverified everywhere, including on machines that *can* run them, so
    instead the failure is classified at run time: a genuine assertion
    mismatch fails, a missing optional dependency skips.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.missing_dependency: List[str] = []
        self.output: List[str] = []

    def report_unexpected_exception(self, out, test, example, exc_info) -> None:  # noqa: ANN001
        if issubclass(exc_info[0], OptionalDependencyError):
            self.missing_dependency.append(exc_info[1].package)
            # Abort this DocTest: later examples depend on the one that
            # could not run, and their cascading failures say nothing.
            raise doctest.UnexpectedException(test, example, exc_info)
        super().report_unexpected_exception(out, test, example, exc_info)

    def _run(self, test: doctest.DocTest) -> None:
        try:
            self.run(test, out=self.output.append, clear_globs=False)
        except doctest.UnexpectedException:
            pass


def _rst_files() -> List[Path]:
    return sorted(p for p in DOCS_SOURCE.rglob("*.rst"))


def _package_modules() -> List[str]:
    """Every importable module in qsarkit, including private submodules."""
    names: List[str] = []
    for sub in qsarkit._SUBPACKAGES:
        package = importlib.import_module(f"qsarkit.{sub}")
        names.append(package.__name__)
        if not hasattr(package, "__path__"):
            continue
        for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
            names.append(info.name)
    return sorted(set(names))


def _globs() -> dict:
    """The namespace documentation examples are evaluated in."""
    if str(DOCS_SOURCE) not in sys.path:
        sys.path.insert(0, str(DOCS_SOURCE))
    import numpy as np

    import demo_data

    np.set_printoptions(precision=3, suppress=True)
    ns = {"np": np}
    ns.update({name: getattr(demo_data, name) for name in demo_data.__all__})
    return ns


@pytest.mark.parametrize(
    "path", _rst_files(), ids=lambda p: str(p.relative_to(DOCS_SOURCE))
)
def test_documentation_example(path: Path) -> None:
    """Every example in the documentation runs and prints what it claims."""
    parser = doctest.DocTestParser()
    text = path.read_text(encoding="utf-8")
    test = parser.get_doctest(text, _globs(), str(path), str(path), 0)
    if not test.examples:
        pytest.skip("no examples in this page")

    runner = _Runner(optionflags=OPTIONFLAGS, verbose=False)
    runner._run(test)
    result = runner.summarize(verbose=False)
    if runner.missing_dependency and not result.failed:
        pytest.skip(f"needs optional dependency: {runner.missing_dependency[0]}")
    if result.failed:
        pytest.fail(
            f"{result.failed} of {result.attempted} examples failed in "
            f"{path.relative_to(ROOT)}:\n\n" + "".join(runner.output)
        )


@pytest.mark.parametrize("module_name", _package_modules())
def test_docstring_example(module_name: str) -> None:
    """Every ``Examples`` section in the package runs and is accurate."""
    module = importlib.import_module(module_name)
    finder = doctest.DocTestFinder(exclude_empty=True)
    try:
        tests = finder.find(module, module_name)
    except ValueError as exc:  # pragma: no cover - malformed docstring
        pytest.fail(f"could not parse docstrings in {module_name}: {exc}")

    tests = [t for t in tests if t.examples]
    if not tests:
        pytest.skip("no examples in this module")

    globs = _globs()
    runner = _Runner(optionflags=OPTIONFLAGS, verbose=False)
    for test in tests:
        test.globs.update(globs)
        runner._run(test)
    result = runner.summarize(verbose=False)
    if result.failed:
        pytest.fail(
            f"{result.failed} of {result.attempted} docstring examples failed "
            f"in {module_name}:\n\n" + "".join(runner.output)
        )
    if runner.missing_dependency:
        pytest.skip(
            "some examples need optional dependencies: "
            + ", ".join(sorted(set(runner.missing_dependency)))
        )


@pytest.mark.parametrize(
    "path", _rst_files(), ids=lambda p: str(p.relative_to(DOCS_SOURCE))
)
def test_code_blocks_reference_real_api(path: Path) -> None:
    """Names used in un-executed code blocks must still exist.

    Most examples are ``.. doctest::`` blocks, which the test above runs.
    A few must be ``.. code-block:: python`` because they cannot run here
    -- they need an optional dependency, a file on disk, or a third-party
    package. Those are exactly the snippets that rot unnoticed: this is
    how the guide came to document classes that had been removed.

    This checks the part that can be checked without running them: that
    every ``from qsarkit... import X`` resolves, and every ``qsarkit.x``
    attribute path exists.
    """
    text = path.read_text(encoding="utf-8")
    broken: List[str] = []

    for module_name, names in re.findall(
        r"^\s*from (qsarkit[\w.]*) import ([^\n#]+)$", text, flags=re.M
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            broken.append(f"{module_name} (no such module)")
            continue
        for name in (n.strip().rstrip(",") for n in names.split(",")):
            if not name or name in {"*", "("} or name.startswith("("):
                continue
            if not hasattr(module, name):
                broken.append(f"{module_name}.{name}")

    for dotted in set(re.findall(r"\bqsarkit\.(\w+)\.(\w+)", text)):
        sub, attribute = dotted
        if sub not in qsarkit._SUBPACKAGES:
            broken.append(f"qsarkit.{sub} (no such subpackage)")
            continue
        module = importlib.import_module(f"qsarkit.{sub}")
        # A second segment may be a private module rather than an export.
        if not hasattr(module, attribute) and not attribute.startswith("_"):
            try:
                importlib.import_module(f"qsarkit.{sub}.{attribute}")
            except ImportError:
                broken.append(f"qsarkit.{sub}.{attribute}")

    assert not broken, (
        f"{path.relative_to(ROOT)} references API that no longer exists: "
        f"{sorted(set(broken))}"
    )
