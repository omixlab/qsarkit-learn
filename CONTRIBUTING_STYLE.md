# qsarkit contribution style guide

This document defines the conventions every module in this package follows.
Read it before adding a new module.

## 1. Molecule I/O contract

Every public class/function that consumes molecules accepts
`Iterable[rdkit.Chem.Mol]` (never SMILES strings directly as the primary
input — SMILES parsing belongs in `qsarkit.chemistry` / `qsarkit.utils`
helpers, or in explicit `from_smiles` convenience constructors).

Use `qsarkit.base.ensure_mol_list` to validate/materialize input inside
`transform`.

## 2. Class shape

- Mol -> Mol curation/standardization steps subclass
  `qsarkit.base.MoleculeToMoleculeTransformer`.
- Mol -> features/vectors steps (fingerprints, descriptors, embeddings)
  subclass `qsarkit.base.MoleculeTransformer` or
  `qsarkit.base.FittableMoleculeTransformer` if they must learn parameters
  (e.g. Mol2Vec, a fitted scaler).
- All such classes implement `fit(mols, y=None)` and `_transform(mols)`
  (the public `transform` is provided by the base class and does
  validation). They are scikit-learn compatible (`BaseEstimator`,
  `TransformerMixin`), so `get_params`/`set_params`/`fit_transform` work
  out of the box — do not override `__init__` in a way that stores
  anything other than the constructor arguments verbatim (sklearn
  convention).
- Data records (e.g. `ActivityRecord`, `Relation`) are `@dataclass`.
- API clients (`qsarkit.databases.*`) are plain classes (not transformers)
  with explicit methods per the spec (e.g. `get_by_cid`), using `requests`
  with a `timeout`, raising `qsarkit.base.DatabaseClientError` on failure.
- Heavy optional dependencies (torch, transformers, gensim, shap, skopt,
  bs4/pdfminer/lxml, matplotlib, jinja2) are imported lazily via
  `qsarkit.base.require("torch")` inside `__init__`/`fit`/the method that
  needs them — never at module top level. This keeps `import qsarkit`
  cheap and lets users install only the extras they need
  (`pip install qsarkit-learn[nlp]`, see `pyproject.toml`).

## 3. Mandatory documentation

Every algorithm, model, metric, database interface, chemical
transformation and NLP method **must** have a docstring with a
`References` section citing the original publication (with DOI when
available) and, where relevant, the official implementation/API docs it
wraps (RDKit, scikit-learn, PubChem PUG REST, ChEMBL API, etc.). Follow
the NumPy docstring style used in `qsarkit/base/transformer.py`. A class
with a non-trivial algorithm and no `References` section is considered
incomplete.

Example:

```python
class FooTransformer(MoleculeTransformer):
    """One-line summary.

    Longer description of what it does and why.

    Parameters
    ----------
    radius : int
        ...

    References
    ----------
    - Author et al. (Year). "Title." Journal, vol(issue), pages.
      https://doi.org/xxxx
    - RDKit documentation: https://www.rdkit.org/docs/...
    """
```

## 4. Error handling

Only catch/validate at real boundaries (parsing untrusted text, network
calls, user-supplied SMILES). Do not wrap internal RDKit calls in
defensive `try/except` "just in case" — trust that a `Chem.Mol` produced
by an earlier validated step is a valid `Mol`. Raise the specific
exception from `qsarkit.base.exceptions` that matches the failure, not a
bare `Exception`.

## 5. No premature abstraction

Implement exactly the classes/methods named in `PROMPT.md` for your
module. Do not invent extra plugin systems, registries or config
frameworks. Where an algorithm genuinely requires a heavy pretrained
model (ChemBERTa, MPNN, generative models) that cannot be trained/loaded
in this environment, still implement the full class with a real
`__init__`/`fit`/`transform`/`forward` contract and real tensor
plumbing — using `require()` for the heavy dependency — rather than a
placeholder that just raises `NotImplementedError`. It is fine for such a
class to *download or expect* a pretrained checkpoint (document the
expected source in `References`); it must not silently return fake data.

## 6. Package layout

Each leaf subpackage (e.g. `qsarkit/chemistry/glycans/`) has:
- one module file per major class (or a couple of closely related classes),
  e.g. `_detector.py`, `_remover.py`, `_descriptors.py`
- an `__init__.py` that re-exports the public classes

Look at `qsarkit/chemistry/standardization/` for a worked example of this
pattern before writing a new module.

## 7. Static typing (mypy / mypyc)

The package is checked with `mypy --strict` (see `[tool.mypy]` in
`pyproject.toml`) and must stay mypyc-compilable. That means:

- **Every** function, method and `__init__` has full parameter and return
  annotations. No bare `def f(x):`.
- `from __future__ import annotations` at the top of every module.
- Annotate RDKit molecules as `Mol` via a `TYPE_CHECKING` import:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from rdkit.Chem import Mol
  ```
  and use `"Mol"` / `Iterable["Mol"]` in signatures. Do not annotate them
  as `Any` — RDKit has no stubs, but our own signatures must still be
  precise for readers.
- Use `npt.NDArray[np.float64]` (`import numpy.typing as npt`) for array
  returns, `list[...]`/`dict[...]` builtins (safe under
  `from __future__ import annotations` on 3.9), `Optional[X]` not `X | None`
  in runtime-evaluated positions, and `Sequence`/`Iterable` for inputs.
- Prefer `@dataclass` with annotated fields, `Protocol` for duck-typed
  interfaces, and `Literal[...]` for string-enum parameters
  (e.g. `mode: Literal["binary", "count"]`).
- mypyc-friendliness: avoid monkey-patching instance methods, avoid
  reassigning a name to a different type, and keep class attributes
  declared at class level with annotations.

Run `python -m mypy qsarkit/<your_module>` and fix every error in your
own code before finishing.

## 8. Plotting

**All plotting uses Plotly, not matplotlib.** `plotly` is a core
dependency. Plotting functions return a `plotly.graph_objects.Figure`
(never call `.show()` internally, never write a file unless explicitly
asked). Import it normally at the top of plotting modules:

```python
import plotly.graph_objects as go
```

Static image export (`fig.write_image`) needs `kaleido`, which is in the
`reporting` extra — guard that path with `qsarkit.base.require("kaleido")`.

## 9. Testing

Target **100% statement and branch coverage** of your module
(`pytest --cov=qsarkit.<module> --cov-branch`). Every public class,
method, branch and raised exception needs a test. Where a line is only
reachable with an optional dependency installed, cover it behind
`pytest.importorskip` rather than excluding it.



Add lightweight `pytest` unit tests under `tests/<module_path>/` mirroring
the package path, using small hardcoded molecules (aspirin, benzene,
ethanol, a flavonoid glycoside, etc.) — not network calls. Tests for
network-dependent clients (`qsarkit.databases.*`, live entity linking)
should mock `requests` rather than hitting real APIs.

## Documentation examples must be executable

Every example in a docstring or a documentation page is run by the test
suite (`pytest tests/docs`). Write them as `>>>` doctests, not as
`.. code-block:: python`, so that an example which stops being true fails
CI like any other regression.

A `code-block` is acceptable only where the snippet genuinely cannot run in
CI — it needs an optional dependency, a file that does not exist, or a
third-party package. `tests/docs/test_documentation_examples.py` still
checks that every `qsarkit` name such a block references actually exists.

Numbers in an example must be **measured, not plausible**. Run the code and
paste what it prints, including when the result is unflattering: a guide
that quotes an invented R² teaches the reader to expect something the
package does not deliver.
