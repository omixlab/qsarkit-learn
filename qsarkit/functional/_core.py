"""Core types for the functional pipe API: :class:`MoleculeSet` and :class:`Step`."""

from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from rdkit.Chem import Mol

__all__ = [
    "MoleculeSet",
    "FeatureSet",
    "PipeStep",
    "Step",
    "FeatureStep",
    "molecules",
    "step",
    "feature_step",
    "pipeline",
]

# What a molecule step's underlying function receives and returns.
_Payload = Tuple[List[Any], Optional[npt.NDArray[Any]]]
# What a feature step's underlying function receives and returns:
# the feature matrix, the labels, and the molecules they came from.
_FeaturePayload = Tuple[
    "npt.NDArray[Any]", Optional["npt.NDArray[Any]"], Optional[List[Any]]
]


class MoleculeSet:
    """A set of molecules with optional labels, flowing through a pipe.

    This is the value that moves left to right through a
    :mod:`qsarkit.functional` pipeline. It carries the molecules, the
    optional labels ``y`` kept index-aligned with them, and a provenance
    log recording what each step did.

    Unpacks as ``(mols, y)``::

        X, y = molecules(smiles, y) >> desalt() >> remove_duplicates()

    Parameters
    ----------
    mols : sequence of Mol
        The molecules. ``None`` entries are allowed and represent
        molecules that failed an earlier parsing or curation step; they
        keep positional alignment with ``y`` until you call
        :func:`~qsarkit.functional.drop_invalid`.
    y : array-like, optional
        Labels, one per molecule.
    history : list of str, optional
        Provenance log; steps append to it.

    Attributes
    ----------
    mols : list of Mol
        The molecules.
    y : ndarray or None
        The labels.
    history : list of str
        What each step did, in order.

    Examples
    --------
    >>> from qsarkit.functional import molecules
    >>> ms = molecules(["CCO", "c1ccccc1"], [1.0, 2.0])
    >>> len(ms)
    2
    >>> mols, y = ms
    >>> len(mols), y.tolist()
    (2, [1.0, 2.0])

    References
    ----------
    - Bache, S. M. & Wickham, H. (2014). "magrittr: A Forward-Pipe
      Operator for R." https://CRAN.R-project.org/package=magrittr
    - RDKit: Open-source cheminformatics. https://www.rdkit.org
    """

    __slots__ = ("mols", "y", "history")

    def __init__(
        self,
        mols: Sequence[Any],
        y: Optional[npt.ArrayLike] = None,
        history: Optional[List[str]] = None,
    ) -> None:
        self.mols: List[Any] = list(mols)
        self.y: Optional[npt.NDArray[Any]] = (
            None if y is None else np.asarray(y)
        )
        if self.y is not None and len(self.y) != len(self.mols):
            raise ValueError(
                f"y has length {len(self.y)} but there are {len(self.mols)} molecules."
            )
        self.history: List[str] = list(history) if history else []

    # -- piping ---------------------------------------------------------

    def __rshift__(self, other: "PipeStep") -> Any:
        """Apply a step: ``molecule_set >> some_step()``.

        Most steps return another :class:`MoleculeSet`. A step that
        computes features (:func:`~qsarkit.functional.featurize` and
        friends) returns a :class:`FeatureSet` instead, which is how a
        pipeline crosses from curation into modelling.
        """
        if not isinstance(other, PipeStep):
            raise TypeError(
                f"Can only pipe into a Step, got {type(other).__name__}. "
                "Did you forget to call the step, e.g. `>> desalt()` "
                "rather than `>> desalt`?"
            )
        return other(self)

    def __or__(self, other: "PipeStep") -> Any:
        """Alias for :meth:`__rshift__`, for those who prefer ``|``."""
        return self.__rshift__(other)

    def __gt__(self, other: Any) -> Any:
        """Reject ``>`` as a pipe operator, loudly.

        Python treats ``a > b > c`` as the *chained comparison*
        ``(a > b) and (b > c)``, so a ``>``-based pipe silently discards
        everything but the last two stages. That failure is invisible —
        you get a result, just not the one you asked for — so this raises
        rather than letting it through.
        """
        raise TypeError(
            "'>' cannot be used as a pipe operator in Python: 'a > b > c' is "
            "parsed as the chained comparison '(a > b) and (b > c)', which "
            "silently throws away your data. Use '>>' instead:\n"
            "    X, y = molecules(X, y) >> desalt() >> remove_duplicates()"
        )

    # -- unpacking and inspection ---------------------------------------

    def __iter__(self) -> Iterator[Any]:
        """Yield ``(mols, y)`` so the set unpacks as ``X, y = ...``.

        To iterate the molecules themselves, use :attr:`mols` or
        :meth:`iter_mols`.
        """
        yield self.mols
        yield self.y

    def iter_mols(self) -> Iterator[Any]:
        """Iterate the molecules (``__iter__`` is reserved for unpacking)."""
        return iter(self.mols)

    def __len__(self) -> int:
        """Number of molecules currently in the set."""
        return len(self.mols)

    def __repr__(self) -> str:
        labelled = "unlabelled" if self.y is None else f"y shape {self.y.shape}"
        n_invalid = sum(1 for m in self.mols if m is None)
        invalid = f", {n_invalid} invalid" if n_invalid else ""
        steps = f", {len(self.history)} steps" if self.history else ""
        return f"<MoleculeSet {len(self.mols)} molecules, {labelled}{invalid}{steps}>"

    @property
    def smiles(self) -> List[Optional[str]]:
        """Canonical SMILES for each molecule (``None`` where invalid)."""
        from rdkit import Chem

        return [None if m is None else Chem.MolToSmiles(m) for m in self.mols]

    def to_frame(self) -> "pd.DataFrame":
        """Render as a DataFrame with ``smiles`` and, if present, ``y``.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        data: Dict[str, Any] = {"smiles": self.smiles}
        if self.y is not None:
            data["y"] = self.y
        return pd.DataFrame(data)

    # -- drawing --------------------------------------------------------

    def to_dot(self, rankdir: str = "TB", include_input: bool = True) -> str:
        """Graphviz DOT source for this pipeline's flowchart.

        Parameters
        ----------
        rankdir : {"TB", "LR"}, default "TB"
            Layout direction.
        include_input : bool, default True
            Draw the input node.

        Returns
        -------
        str
            DOT source. See :func:`~qsarkit.functional.to_dot`.
        """
        from qsarkit.functional._viz import to_dot

        return to_dot(self, rankdir=rankdir, include_input=include_input)

    def plot(self, **kwargs: Any) -> Any:
        """Plotly flowchart of this pipeline.

        Parameters
        ----------
        **kwargs
            Passed to :func:`~qsarkit.functional.plot_pipeline`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from qsarkit.functional._viz import plot_pipeline

        return plot_pipeline(self, **kwargs)

    def render(self, path: str, **kwargs: Any) -> str:
        """Write this pipeline's flowchart to a PNG, PDF or SVG file.

        Parameters
        ----------
        path : str
            Output file; the extension chooses the format.
        **kwargs
            Passed to :func:`~qsarkit.functional.render_pipeline`.

        Returns
        -------
        str
            The path written.
        """
        from qsarkit.functional._viz import render_pipeline

        return render_pipeline(self, path, **kwargs)

    def replace(
        self,
        mols: Sequence[Any],
        y: Optional[npt.ArrayLike] = None,
        note: Optional[str] = None,
    ) -> "MoleculeSet":
        """Return a new set with different contents and an extended history.

        Steps use this instead of mutating, so a pipeline never modifies
        the set handed to it.

        Parameters
        ----------
        mols : sequence of Mol
            The new molecules.
        y : array-like, optional
            The new labels.
        note : str, optional
            Line to append to the provenance log.

        Returns
        -------
        MoleculeSet
        """
        history = list(self.history)
        if note:
            history.append(note)
        return MoleculeSet(mols, y, history)


class FeatureSet:
    """A feature matrix with labels, flowing through a pipe.

    What a :class:`MoleculeSet` becomes once it has been featurized. It
    carries the matrix ``X``, the labels ``y``, the molecules the rows
    came from (so a downstream step can still reach the chemistry), and
    the same growing provenance log.

    Unpacks as ``(X, y)``::

        X, y = molecules(smiles, y) >> desalt() >> featurize(MorganFingerprint())

    Parameters
    ----------
    X : array-like
        Feature matrix, one row per molecule.
    y : array-like, optional
        Labels, one per row.
    mols : sequence of Mol, optional
        The molecules the rows were computed from, kept index-aligned.
    history : list of str, optional
        Provenance log; steps append to it.
    feature_names : sequence of str, optional
        Column names, propagated from the transformer where it provides
        ``get_feature_names_out()``.

    Attributes
    ----------
    X : ndarray
        The feature matrix.
    y : ndarray or None
        The labels.
    mols : list of Mol or None
        The molecules, still aligned with the rows.
    history : list of str
        What each step did, in order.
    feature_names : list of str or None
        Column names, where known.

    Examples
    --------
    >>> from qsarkit.functional import featurize, molecules
    >>> from qsarkit.representation import MorganFingerprint
    >>> fs = molecules(["CCO", "c1ccccc1"], [1.0, 2.0]) >> featurize(
    ...     MorganFingerprint(n_bits=64))
    >>> fs.shape
    (2, 64)
    >>> X, y = fs
    >>> X.shape, y.tolist()
    ((2, 64), [1.0, 2.0])
    """

    __slots__ = ("X", "y", "mols", "history", "feature_names")

    def __init__(
        self,
        X: npt.ArrayLike,
        y: Optional[npt.ArrayLike] = None,
        mols: Optional[Sequence[Any]] = None,
        history: Optional[List[str]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.X: npt.NDArray[Any] = np.asarray(X)
        self.y: Optional[npt.NDArray[Any]] = None if y is None else np.asarray(y)
        self.mols: Optional[List[Any]] = None if mols is None else list(mols)
        if self.y is not None and len(self.y) != len(self.X):
            raise ValueError(
                f"y has length {len(self.y)} but X has {len(self.X)} rows."
            )
        if self.mols is not None and len(self.mols) != len(self.X):
            raise ValueError(
                f"Got {len(self.mols)} molecules but X has {len(self.X)} rows."
            )
        self.history: List[str] = list(history) if history else []
        self.feature_names: Optional[List[str]] = (
            None if feature_names is None else list(feature_names)
        )

    # -- piping ---------------------------------------------------------

    def __rshift__(self, other: "PipeStep") -> Any:
        """Apply a step: ``feature_set >> some_step()``."""
        if not isinstance(other, PipeStep):
            raise TypeError(
                f"Can only pipe into a Step, got {type(other).__name__}. "
                "Did you forget to call the step, e.g. `>> scale()` "
                "rather than `>> scale`?"
            )
        return other(self)

    def __or__(self, other: "PipeStep") -> Any:
        """Alias for :meth:`__rshift__`."""
        return self.__rshift__(other)

    def __gt__(self, other: Any) -> Any:
        """Reject ``>``; see :meth:`MoleculeSet.__gt__`."""
        raise TypeError(
            "'>' cannot be used as a pipe operator in Python: 'a > b > c' is "
            "parsed as the chained comparison '(a > b) and (b > c)', which "
            "silently throws away your data. Use '>>' instead."
        )

    # -- unpacking and inspection ---------------------------------------

    def __iter__(self) -> Iterator[Any]:
        """Yield ``(X, y)`` so the set unpacks as ``X, y = ...``."""
        yield self.X
        yield self.y

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.X)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the feature matrix."""
        return tuple(self.X.shape)

    def __repr__(self) -> str:
        labelled = "unlabelled" if self.y is None else f"y shape {self.y.shape}"
        steps = f", {len(self.history)} steps" if self.history else ""
        return f"<FeatureSet X shape {self.shape}, {labelled}{steps}>"

    def to_frame(self) -> "pd.DataFrame":
        """Render as a DataFrame, using :attr:`feature_names` where known.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        frame = pd.DataFrame(self.X, columns=self.feature_names)
        if self.y is not None:
            frame["y"] = self.y
        return frame

    # -- drawing --------------------------------------------------------

    def to_dot(self, rankdir: str = "TB", include_input: bool = True) -> str:
        """Graphviz DOT source for this pipeline's flowchart.

        Parameters
        ----------
        rankdir : {"TB", "LR"}, default "TB"
            Layout direction.
        include_input : bool, default True
            Draw the input node.

        Returns
        -------
        str
            DOT source. See :func:`~qsarkit.functional.to_dot`.
        """
        from qsarkit.functional._viz import to_dot

        return to_dot(self, rankdir=rankdir, include_input=include_input)

    def plot(self, **kwargs: Any) -> Any:
        """Plotly flowchart of this pipeline.

        Parameters
        ----------
        **kwargs
            Passed to :func:`~qsarkit.functional.plot_pipeline`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from qsarkit.functional._viz import plot_pipeline

        return plot_pipeline(self, **kwargs)

    def render(self, path: str, **kwargs: Any) -> str:
        """Write this pipeline's flowchart to a PNG, PDF or SVG file.

        Parameters
        ----------
        path : str
            Output file; the extension chooses the format.
        **kwargs
            Passed to :func:`~qsarkit.functional.render_pipeline`.

        Returns
        -------
        str
            The path written.
        """
        from qsarkit.functional._viz import render_pipeline

        return render_pipeline(self, path, **kwargs)

    def replace(
        self,
        X: npt.ArrayLike,
        y: Optional[npt.ArrayLike] = None,
        mols: Optional[Sequence[Any]] = None,
        note: Optional[str] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "FeatureSet":
        """Return a new set with different contents and an extended history.

        Parameters
        ----------
        X : array-like
            The new feature matrix.
        y : array-like, optional
            The new labels.
        mols : sequence of Mol, optional
            The new molecules.
        note : str, optional
            Line to append to the provenance log.
        feature_names : sequence of str, optional
            The new column names.

        Returns
        -------
        FeatureSet
        """
        history = list(self.history)
        if note:
            history.append(note)
        return FeatureSet(X, y, mols, history, feature_names)


class PipeStep:
    """Base class for anything that can appear on the right of ``>>``.

    A pipe step holds a function plus the arguments it was configured
    with, and applies them when a value is piped in. Steps also compose
    with each other, so a pipeline can be built once and reused.

    Three concrete kinds exist, distinguished by what they consume and
    produce:

    :class:`Step`
        :class:`MoleculeSet` -> :class:`MoleculeSet`. Curation,
        filtering, anything that stays in the chemistry domain.
    ``featurize`` and its shorthands
        :class:`MoleculeSet` -> :class:`FeatureSet`. The transition into
        the modelling domain.
    :class:`FeatureStep`
        :class:`FeatureSet` -> :class:`FeatureSet`. Scaling, selection,
        anything that reshapes the matrix.

    A chain that mixes them is checked as it runs, and a mismatch names
    both the step and what it received.

    Parameters
    ----------
    name : str
        Display name, used in the provenance log.
    params : dict, optional
        Keyword arguments applied when the step runs.
    """

    __slots__ = ("name", "params")

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.params = dict(params) if params else {}

    def __call__(self, data: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _describe(self, n_before: int = 0) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{self.name}({args})" if args else f"{self.name}()"

    def __rshift__(self, other: "PipeStep") -> "PipeStep":
        """Compose two steps into one reusable step."""
        if not isinstance(other, PipeStep):
            raise TypeError(
                f"Can only compose a Step with another Step, got "
                f"{type(other).__name__}."
            )
        return _Composed([self, other])

    def __or__(self, other: "PipeStep") -> "PipeStep":
        """Alias for :meth:`__rshift__`."""
        return self.__rshift__(other)

    def __gt__(self, other: Any) -> Any:
        """Reject ``>``; see :meth:`MoleculeSet.__gt__`."""
        raise TypeError(
            "'>' cannot be used as a pipe operator in Python (it is parsed as "
            "a chained comparison and silently discards data). Use '>>'."
        )

    # -- drawing --------------------------------------------------------

    def to_dot(self, rankdir: str = "TB", include_input: bool = True) -> str:
        """Graphviz DOT source for this pipeline's flowchart.

        Parameters
        ----------
        rankdir : {"TB", "LR"}, default "TB"
            Layout direction.
        include_input : bool, default True
            Draw the input node.

        Returns
        -------
        str
            DOT source. See :func:`~qsarkit.functional.to_dot`.
        """
        from qsarkit.functional._viz import to_dot

        return to_dot(self, rankdir=rankdir, include_input=include_input)

    def plot(self, **kwargs: Any) -> Any:
        """Plotly flowchart of this pipeline.

        Parameters
        ----------
        **kwargs
            Passed to :func:`~qsarkit.functional.plot_pipeline`.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from qsarkit.functional._viz import plot_pipeline

        return plot_pipeline(self, **kwargs)

    def render(self, path: str, **kwargs: Any) -> str:
        """Write this pipeline's flowchart to a PNG, PDF or SVG file.

        Parameters
        ----------
        path : str
            Output file; the extension chooses the format.
        **kwargs
            Passed to :func:`~qsarkit.functional.render_pipeline`.

        Returns
        -------
        str
            The path written.
        """
        from qsarkit.functional._viz import render_pipeline

        return render_pipeline(self, path, **kwargs)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self._describe()}>"


class Step(PipeStep):
    """One deferred molecule -> molecule operation in a pipe.

    Holds a function plus the arguments it was configured with, and
    applies them when a :class:`MoleculeSet` is piped in.

    Parameters
    ----------
    func : callable
        ``func(mols, y, **params) -> (mols, y)``.
    name : str
        Display name, used in the provenance log.
    params : dict
        Keyword arguments applied when the step runs.

    Examples
    --------
    >>> from qsarkit.functional import desalt, drop_invalid, molecules
    >>> curate = desalt() >> drop_invalid()      # reusable pipeline
    >>> mols, y = molecules(["CCO.[Na+]"]) >> curate
    >>> len(mols)
    1
    """

    __slots__ = ("func",)

    def __init__(
        self,
        func: Callable[..., _Payload],
        name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, params)
        self.func = func

    def __call__(self, data: Union[MoleculeSet, Sequence[Any]]) -> MoleculeSet:
        """Apply the step to a :class:`MoleculeSet` (or a raw molecule list)."""
        if isinstance(data, FeatureSet):
            raise TypeError(
                f"Step '{self.name}' works on molecules, but it received a "
                "FeatureSet. Molecule steps (curation, filtering) must come "
                "before the featurize() step that turns molecules into a "
                "feature matrix."
            )
        molecule_set = (
            data if isinstance(data, MoleculeSet) else MoleculeSet(data)
        )
        mols, y = self.func(molecule_set.mols, molecule_set.y, **self.params)
        return molecule_set.replace(mols, y, note=self._describe(len(molecule_set)))


class FeatureStep(PipeStep):
    """One deferred features -> features operation in a pipe.

    The :class:`FeatureSet` counterpart of :class:`Step`: scaling,
    feature selection, and anything else that reshapes the matrix while
    keeping ``y`` (and the originating molecules) aligned with it.

    Parameters
    ----------
    func : callable
        ``func(X, y, mols, **params) -> (X, y, mols)``.
    name : str
        Display name, used in the provenance log.
    params : dict
        Keyword arguments applied when the step runs.

    Examples
    --------
    >>> from qsarkit.functional import featurize, molecules, scale
    >>> from qsarkit.representation import PhysicochemicalDescriptors
    >>> fs = (
    ...     molecules(["CCO", "c1ccccc1", "CCN"], [1.0, 2.0, 3.0])
    ...     >> featurize(PhysicochemicalDescriptors())
    ...     >> scale()
    ... )
    >>> bool(abs(fs.X.mean()) < 1e-9)     # standardized to zero mean
    True
    """

    __slots__ = ("func",)

    def __init__(
        self,
        func: Callable[..., _FeaturePayload],
        name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, params)
        self.func = func

    def __call__(self, data: Any) -> FeatureSet:
        """Apply the step to a :class:`FeatureSet`."""
        if not isinstance(data, FeatureSet):
            what = "molecules" if isinstance(data, MoleculeSet) else type(data).__name__
            raise TypeError(
                f"Step '{self.name}' works on a feature matrix, but it "
                f"received {what}. Insert a featurize(...) step first:\n"
                "    molecules(X, y) >> desalt() >> featurize(MorganFingerprint()) "
                f">> {self.name}()"
            )
        X, y, mols = self.func(data.X, data.y, data.mols, **self.params)
        names = data.feature_names
        if names is not None and X.shape[1] != len(names):
            names = None
        return data.replace(X, y, mols, note=self._describe(), feature_names=names)


class _Composed(PipeStep):
    """A step that runs several steps in sequence.

    Composition is deliberately untyped at build time: a pipeline may
    legitimately start on molecules and end on features. Each constituent
    step checks what it actually receives when the pipeline runs, so a
    mis-ordered chain fails with a message naming the offending step.
    """

    __slots__ = ("steps",)

    def __init__(self, steps: Sequence[PipeStep]) -> None:
        # Flatten so `a >> b >> c` is one three-step pipeline, not nested pairs.
        flat: List[PipeStep] = []
        for s in steps:
            flat.extend(s.steps if isinstance(s, _Composed) else [s])
        self.steps: List[PipeStep] = flat
        super().__init__(name="pipeline")

    def __call__(self, data: Any) -> Any:
        """Run every constituent step in order."""
        result = (
            data
            if isinstance(data, (MoleculeSet, FeatureSet))
            else MoleculeSet(data)
        )
        for s in self.steps:
            result = s(result)
        return result

    def __repr__(self) -> str:
        return f"<Pipeline {' >> '.join(s._describe() for s in self.steps)}>"


def _looks_like_data(obj: Any) -> bool:
    """True if ``obj`` is data to run on, rather than a step parameter.

    The dual-mode decorators take the data as their first positional
    argument, which makes ``scale("robust")`` ambiguous: is ``"robust"``
    a dataset or the ``method`` parameter? Resolving it by *type* rather
    than by position lets both readings work, so the natural call spells
    what it means.
    """
    if isinstance(obj, (MoleculeSet, FeatureSet, np.ndarray)):
        return True
    if isinstance(obj, (str, bytes, int, float, bool)):
        return False
    # A sequence of molecules, SMILES or numbers is data; anything else
    # (a dict of options, an estimator) is not.
    if isinstance(obj, (list, tuple)):
        return True
    return hasattr(obj, "__array__") or hasattr(obj, "iloc")


def _bind_params(
    func: Callable[..., Any],
    skip: int,
    args: Tuple[Any, ...],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Bind positional step arguments to ``func``'s parameters after the data.

    Parameters
    ----------
    func : callable
        The step implementation.
    skip : int
        How many leading parameters are data (1 for molecule steps,
        3 for feature steps).
    args : tuple
        Positional arguments the caller supplied.
    params : dict
        Keyword arguments the caller supplied.

    Returns
    -------
    dict
        The merged keyword arguments.

    Raises
    ------
    TypeError
        If a positional argument has no parameter to bind to, or
        duplicates one given by keyword.
    """
    import inspect

    names = [
        name
        for name, p in inspect.signature(func).parameters.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ][skip:]
    if len(args) > len(names):
        raise TypeError(
            f"{func.__name__}() takes at most {len(names)} configuration "
            f"argument(s) ({', '.join(names) or 'none'}), got {len(args)}."
        )
    merged = dict(params)
    for name, value in zip(names, args):
        if name in merged:
            raise TypeError(
                f"{func.__name__}() got multiple values for argument {name!r}."
            )
        merged[name] = value
    return merged


def molecules(
    X: Sequence[Any],
    y: Optional[npt.ArrayLike] = None,
    fmt: Literal["auto", "smiles", "inchi", "mol"] = "auto",
) -> MoleculeSet:
    """Start a pipeline from RDKit molecules, SMILES or InChI.

    The entry point of the functional API. Accepts
    :class:`rdkit.Chem.Mol` objects, SMILES strings, InChI strings, or
    any mixture of the three. Strings that fail to parse become ``None``
    rather than raising, so they stay aligned with ``y`` until you decide
    what to do with them -- normally a
    :func:`~qsarkit.functional.drop_invalid` step, which removes the
    matching labels too.

    Parameters
    ----------
    X : sequence of Mol or str
        Molecules, SMILES strings, or InChI strings.
    y : array-like, optional
        Labels, one per molecule.
    fmt : {"auto", "smiles", "inchi", "mol"}, default "auto"
        How to read string entries. ``"auto"`` treats a string beginning
        with ``InChI=`` as InChI and anything else as SMILES, deciding
        per entry so mixed input works. Name the format explicitly when
        you would rather have malformed input fail than be silently
        reinterpreted.

    Returns
    -------
    MoleculeSet
        The set to pipe onward.

    Raises
    ------
    ValueError
        If ``fmt`` is not one of the four accepted values, or if
        ``fmt="mol"`` and an entry is not an RDKit molecule.

    Examples
    --------
    From SMILES:

    >>> from qsarkit.functional import desalt, drop_invalid, molecules
    >>> mols, y = (
    ...     molecules(["CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", "CCO"], [1.0, 2.0])
    ...     >> desalt()
    ...     >> drop_invalid()
    ... )
    >>> from rdkit import Chem
    >>> Chem.MolToSmiles(mols[0])
    'CC(=O)Oc1ccccc1C(=O)[O-]'

    From InChI, recognised without being told:

    >>> ms = molecules(["InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"])
    >>> Chem.MolToSmiles(ms.mols[0])
    'CCO'

    From RDKit molecules, or any mixture of the three:

    >>> mixed = molecules([
    ...     Chem.MolFromSmiles("c1ccccc1"),
    ...     "CCN",
    ...     "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
    ... ])
    >>> [Chem.MolToSmiles(m) for m in mixed.mols]
    ['c1ccccc1', 'CCN', 'CCO']

    Unparseable entries survive as ``None`` so nothing shifts out of
    alignment with ``y``:

    >>> ms = molecules(["CCO", "not-a-molecule"], [1.0, 2.0])
    >>> [m is None for m in ms.mols]
    [False, True]
    >>> mols, y = ms >> drop_invalid()
    >>> y.tolist()
    [1.0]

    References
    ----------
    - Bache, S. M. & Wickham, H. (2014). "magrittr: A Forward-Pipe
      Operator for R." https://CRAN.R-project.org/package=magrittr
    - Heller, S. R. et al. (2015). "InChI - the Worldwide Chemical
      Structure Identifier Standard." J. Cheminform., 7, 23.
      https://doi.org/10.1186/s13321-015-0068-4
    - RDKit: Open-source cheminformatics. https://www.rdkit.org
    """
    if fmt not in ("auto", "smiles", "inchi", "mol"):
        raise ValueError(
            f"Unknown fmt {fmt!r}. Choose from 'auto', 'smiles', 'inchi' or 'mol'."
        )

    from rdkit import Chem, rdBase

    parsed: List[Any] = []
    # RDKit logs a message per unparseable record. A curation pipeline is
    # expected to receive bad input -- that is what drop_invalid() is for --
    # so the None in the output is the signal, not a screenful of warnings.
    #
    # BlockLogs restores whatever logging state the caller had, rather than
    # unconditionally re-enabling: a caller who disabled RDKit logging for
    # their whole session must not have it switched back on underneath them.
    _blocker = rdBase.BlockLogs()
    try:
        for i, item in enumerate(X):
            if isinstance(item, Chem.Mol) or item is None:
                parsed.append(item)
                continue
            if fmt == "mol":
                raise ValueError(
                    f"Element {i} is not an rdkit.Chem.Mol (got "
                    f"{type(item).__name__!r}) and fmt='mol' was requested."
                )
            if not isinstance(item, str):
                raise ValueError(
                    f"Element {i} is neither an rdkit.Chem.Mol nor a string "
                    f"(got {type(item).__name__!r})."
                )
            text = item.strip()
            as_inchi = fmt == "inchi" or (fmt == "auto" and text.startswith("InChI="))
            parsed.append(
                Chem.MolFromInchi(text) if as_inchi else Chem.MolFromSmiles(text)
            )
    finally:
        del _blocker

    return MoleculeSet(parsed, y, history=[f"molecules(n={len(parsed)})"])


def step(func: Callable[..., _Payload]) -> Callable[..., Any]:
    """Turn an ``(X, y=None, **params) -> (X, y)`` function into a pipe step.

    The decorated callable works two ways, which is what lets the same
    function serve both the pipe API and ordinary imperative code:

    - Called with no molecules — ``desalt()``, ``balance(method="under")`` —
      it returns a deferred :class:`Step` for use in a pipe.
    - Called with molecules — ``desalt(mols, y)`` — it runs immediately
      and returns ``(mols, y)``.

    Parameters
    ----------
    func : callable
        Implementation taking ``(mols, y, **params)`` and returning
        ``(mols, y)``.

    Returns
    -------
    callable
        The dual-mode wrapper.

    Examples
    --------
    >>> from qsarkit.functional import step
    >>> @step
    ... def keep_first(mols, y=None, n=1):
    ...     return mols[:n], (None if y is None else y[:n])
    >>> from qsarkit.functional import molecules
    >>> mols, y = molecules(["CCO", "CCN", "CCC"]) >> keep_first(n=2)
    >>> len(mols)
    2
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **params: Any) -> Any:
        # The first positional argument is the data only when it looks
        # like data; otherwise every positional is a step parameter, so
        # `keep_if(predicate)` and `balance("under")` read naturally.
        if args and _looks_like_data(args[0]):
            X, rest = args[0], args[1:]
            y = rest[0] if rest else params.pop("y", None)
            merged = _bind_params(func, 2, rest[1:], params)
            if isinstance(X, MoleculeSet):
                return Step(func, func.__name__, merged)(X)
            return func(list(X), None if y is None else np.asarray(y), **merged)
        return Step(func, func.__name__, _bind_params(func, 2, args, params))

    return wrapper


def feature_step(func: Callable[..., _FeaturePayload]) -> Callable[..., Any]:
    """Turn an ``(X, y=None, mols=None, **params) -> (X, y, mols)`` function into a pipe step.

    The :class:`FeatureSet` counterpart of :func:`step`, and dual-mode in
    the same way:

    - Called with no data — ``scale()``, ``select_features(k=10)`` — it
      returns a deferred :class:`FeatureStep` for use in a pipe.
    - Called with a matrix — ``scale(X, y)`` — it runs immediately and
      returns ``(X, y, mols)``.

    Parameters
    ----------
    func : callable
        Implementation taking ``(X, y, mols, **params)`` and returning
        ``(X, y, mols)``.

    Returns
    -------
    callable
        The dual-mode wrapper.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.functional import feature_step, featurize, molecules
    >>> from qsarkit.representation import PhysicochemicalDescriptors
    >>> @feature_step
    ... def first_columns(X, y=None, mols=None, n=2):
    ...     return X[:, :n], y, mols
    >>> fs = (
    ...     molecules(["CCO", "c1ccccc1"], [1.0, 2.0])
    ...     >> featurize(PhysicochemicalDescriptors())
    ...     >> first_columns(n=3)
    ... )
    >>> fs.shape
    (2, 3)
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **params: Any) -> Any:
        # As for `step`: a leading positional is data only if it looks
        # like data, so `scale("robust")` configures rather than fails.
        if args and _looks_like_data(args[0]):
            X, rest = args[0], args[1:]
            y = rest[0] if rest else params.pop("y", None)
            mols = rest[1] if len(rest) > 1 else params.pop("mols", None)
            merged = _bind_params(func, 3, rest[2:], params)
            if isinstance(X, FeatureSet):
                return FeatureStep(func, func.__name__, merged)(X)
            return func(
                np.asarray(X),
                None if y is None else np.asarray(y),
                None if mols is None else list(mols),
                **merged,
            )
        return FeatureStep(func, func.__name__, _bind_params(func, 3, args, params))

    return wrapper


def pipeline(*steps: PipeStep) -> PipeStep:
    """Compose steps into one reusable pipeline.

    Equivalent to chaining with ``>>``, but easier to build
    programmatically from a list.

    Parameters
    ----------
    *steps : PipeStep
        Steps to run in order.

    Returns
    -------
    PipeStep
        A single step running all of them.

    Examples
    --------
    >>> from qsarkit.functional import desalt, drop_invalid, molecules, pipeline
    >>> curate = pipeline(desalt(), drop_invalid())
    >>> mols, y = molecules(["CCO.[Na+]", "not-a-molecule"]) >> curate
    >>> len(mols)
    1
    """
    if not steps:
        raise ValueError("pipeline() needs at least one step.")
    for s in steps:
        if not isinstance(s, PipeStep):
            raise TypeError(
                f"pipeline() takes Step objects, got {type(s).__name__}. "
                "Remember to call each step, e.g. desalt() not desalt."
            )
    return _Composed(list(steps))
