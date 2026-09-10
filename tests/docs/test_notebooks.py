"""Execute the example notebooks, so they cannot rot silently.

The notebooks are documentation, and documentation that is never run stops
being true. Running them here means a change that breaks an example fails
CI like any other regression.

They are marked ``slow`` because a full execution takes a minute or so;
run them with ``pytest -m slow`` or ``pytest tests/docs/test_notebooks.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "notebooks"


def _notebooks() -> List[Path]:
    return sorted(NOTEBOOKS.glob("*.ipynb"))


def test_notebook_directory_exists() -> None:
    """The examples ship with the repository."""
    assert NOTEBOOKS.is_dir()
    assert _notebooks(), "no notebooks found"


@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.stem)
def test_notebook_is_valid(path: Path) -> None:
    """Every notebook is well-formed and carries executed output.

    A notebook committed without output is one nobody checked, so this
    also guards against shipping a stale or unrun example.
    """
    nbformat = pytest.importorskip("nbformat")

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert code_cells, f"{path.name} has no code cells"

    with_output = [c for c in code_cells if c.get("outputs")]
    assert with_output, f"{path.name} was committed without executed output"

    for cell in code_cells:
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", (
                f"{path.name} contains a cell that raised "
                f"{output.get('ename')}: {output.get('evalue')}"
            )


@pytest.mark.slow
@pytest.mark.parametrize("path", _notebooks(), ids=lambda p: p.stem)
def test_notebook_executes(path: Path, tmp_path: Path) -> None:
    """Every notebook runs top to bottom without error."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    notebook = nbformat.read(path, as_version=4)
    # The notebooks import qsarkit; make sure the working tree is importable
    # even when the package is not installed into the kernel's environment.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    client = nbclient.NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    old_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = env["PYTHONPATH"]
    try:
        client.execute()
    finally:
        if old_path is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_path
