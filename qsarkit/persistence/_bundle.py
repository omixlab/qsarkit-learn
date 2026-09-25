"""Save and load QSAR models and their pipelines, without pickle.

Why not pickle
--------------
``pickle`` is the obvious choice and the wrong one for a model you intend
to keep. It embeds the exact class layout of every object, so a file
written under one scikit-learn or NumPy release can fail to load -- or,
worse, load into a subtly different object -- under the next. It also
executes arbitrary code on load, which makes a shared model file a
security problem rather than a data file.

This module writes a **directory bundle** instead:

.. code-block:: text

    model.qsar/
      manifest.json      what is in here, and what wrote it
      metadata.json      endpoint, task, feature names, provenance
      estimator.skops    the fitted estimator, in skops' inspectable format
      pipeline.skops     the preprocessing pipeline, when there is one

Everything except the estimator itself is plain JSON, readable without
importing qsarkit at all. The estimator uses `skops
<https://skops.readthedocs.io>`_, the format scikit-learn recommends for
persistence: it stores parameters as data rather than as a pickled object
graph, and refuses to reconstruct types that were not explicitly trusted.

The trade-off is honest: skops can persist scikit-learn estimators and
NumPy arrays, not arbitrary Python. A model wrapping something exotic may
need :func:`save_bundle`'s ``allow_pickle_fallback``, which is available
and loudly named so nobody reaches for it accidentally.

References
----------
- skops documentation, "Secure persistence with skops":
  https://skops.readthedocs.io/en/stable/persistence.html
- scikit-learn, "Model persistence":
  https://scikit-learn.org/stable/model_persistence.html
- OECD (2007). "Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models," ENV/JM/MONO(2007)2.
  https://doi.org/10.1787/9789264085442-en
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from qsarkit.base.exceptions import OptionalDependencyError
from qsarkit.persistence._metadata import BUNDLE_FORMAT_VERSION, ModelMetadata

__all__ = [
    "ModelBundle",
    "save_model",
    "load_model",
    "inspect_bundle",
]

PathLike = Union[str, Path]

#: Filenames inside a bundle. Fixed, so an older reader can find its way
#: around a bundle written by a newer version.
_MANIFEST = "manifest.json"
_METADATA = "metadata.json"
_ESTIMATOR = "estimator.skops"
_PIPELINE = "pipeline.skops"
_ESTIMATOR_PICKLE = "estimator.joblib"
_PIPELINE_PICKLE = "pipeline.joblib"

#: Default suffix for a bundle directory.
BUNDLE_SUFFIX = ".qsar"


def _require_skops() -> Any:
    """Import skops, with an error naming the extra that provides it."""
    from qsarkit.base import require

    return require("skops.io")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


class ModelBundle:
    """A fitted model, its preprocessing, and the provenance to interpret it.

    The unit qsarkit saves and loads. Holding the three together is the
    point: an estimator without its representation is not a model, and a
    model without its endpoint and feature layout cannot be used safely a
    year later.

    Parameters
    ----------
    estimator : object
        The fitted model. Any scikit-learn-compatible estimator.
    pipeline : object, optional
        The transformer (or :class:`sklearn.pipeline.Pipeline`) that turns
        molecules into the feature matrix ``estimator`` expects. Supply it
        whenever you have one: it is what makes :meth:`predict_mols`
        possible, and what stops a reloaded model being fed the wrong
        features.
    metadata : ModelMetadata, optional
        Provenance. A default is created if omitted, but an empty
        ``endpoint`` is worth filling in.

    Attributes
    ----------
    estimator : object
    pipeline : object or None
    metadata : ModelMetadata

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from qsarkit.persistence import ModelBundle, ModelMetadata
    >>> rng = np.random.default_rng(0)
    >>> X, y = rng.normal(size=(40, 5)), rng.normal(size=40)
    >>> bundle = ModelBundle(
    ...     Ridge().fit(X, y),
    ...     metadata=ModelMetadata(name="demo", endpoint="pIC50 (-log10 M)"),
    ... )
    >>> bundle.predict(X).shape
    (40,)

    Round-tripping through a directory preserves the predictions exactly:

    >>> import tempfile, os
    >>> path = os.path.join(tempfile.mkdtemp(), "demo.qsar")
    >>> _ = bundle.save(path)
    >>> reloaded = ModelBundle.load(path)
    >>> bool(np.allclose(reloaded.predict(X), bundle.predict(X)))
    True
    >>> reloaded.metadata.endpoint
    'pIC50 (-log10 M)'

    References
    ----------
    - skops documentation: https://skops.readthedocs.io/en/stable/persistence.html
    """

    def __init__(
        self,
        estimator: Any,
        pipeline: Optional[Any] = None,
        metadata: Optional[ModelMetadata] = None,
    ) -> None:
        self.estimator = estimator
        self.pipeline = pipeline
        self.metadata = metadata if metadata is not None else ModelMetadata()
        if self.metadata.n_features is None:
            self.metadata.n_features = _infer_n_features(estimator)
        if not self.metadata.name:
            self.metadata.name = type(estimator).__name__

    # -- use ------------------------------------------------------------

    def predict(self, X: Any) -> Any:
        """Predict from an already-featurized matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray

        Raises
        ------
        ValueError
            If ``X`` has a different number of columns than the model was
            fitted on. Checked rather than passed through, because a width
            mismatch otherwise produces confident nonsense.
        """
        self._check_width(X)
        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Class probabilities, for a classification bundle.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)

        Raises
        ------
        AttributeError
            If the underlying estimator has no ``predict_proba``.
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(
                f"{type(self.estimator).__name__} does not support "
                "predict_proba."
            )
        self._check_width(X)
        return self.estimator.predict_proba(X)

    def predict_mols(self, mols: Sequence[Any]) -> Any:
        """Predict straight from molecules, using the stored pipeline.

        Parameters
        ----------
        mols : sequence of Mol or str
            RDKit molecules, or SMILES if the stored pipeline begins with
            a SMILES parser.

        Returns
        -------
        ndarray

        Raises
        ------
        ValueError
            If the bundle carries no pipeline, so there is no way to know
            which representation the estimator expects.
        """
        if self.pipeline is None:
            raise ValueError(
                "This bundle has no pipeline, so molecules cannot be "
                "featurized. Either save the bundle with the transformer "
                "that produced its features, or featurize yourself and "
                "call predict()."
            )
        return self.estimator.predict(self.pipeline.transform(mols))

    def _check_width(self, X: Any) -> None:
        """Reject a feature matrix of unexpected width."""
        expected = self.metadata.n_features
        if expected is None:
            return
        try:
            width = int(X.shape[1])
        except (AttributeError, IndexError, TypeError):
            return
        if width != expected:
            raise ValueError(
                f"This model expects {expected} features but was given "
                f"{width}. The representation used at prediction time must "
                "match the one used for training -- check the fingerprint "
                "size and radius, or use predict_mols() with the stored "
                "pipeline."
            )

    # -- persistence ----------------------------------------------------

    def save(
        self,
        path: PathLike,
        allow_pickle_fallback: bool = False,
        overwrite: bool = True,
    ) -> str:
        """Write the bundle to a directory. See :func:`save_model`."""
        return save_model(
            self,
            path,
            allow_pickle_fallback=allow_pickle_fallback,
            overwrite=overwrite,
        )

    @classmethod
    def load(
        cls,
        path: PathLike,
        trusted: Optional[List[str]] = None,
        warn_on_environment_change: bool = True,
    ) -> "ModelBundle":
        """Read a bundle from a directory. See :func:`load_model`."""
        return load_model(
            path,
            trusted=trusted,
            warn_on_environment_change=warn_on_environment_change,
        )

    def __repr__(self) -> str:
        pipeline = type(self.pipeline).__name__ if self.pipeline else "none"
        return (
            f"<ModelBundle {self.metadata.name!r} "
            f"({self.metadata.task}, {self.metadata.n_features} features, "
            f"pipeline: {pipeline})>"
        )


def _infer_n_features(estimator: Any) -> Optional[int]:
    """Feature count from a fitted estimator, if it exposes one."""
    for attribute in ("n_features_in_", "n_features_"):
        value = getattr(estimator, attribute, None)
        if isinstance(value, (int,)):
            return int(value)
    # qsarkit facades hold the real estimator one level down.
    inner = getattr(estimator, "estimator_", None)
    if inner is not None:
        return _infer_n_features(inner)
    return None


def save_model(
    bundle: Union[ModelBundle, Any],
    path: PathLike,
    pipeline: Optional[Any] = None,
    metadata: Optional[ModelMetadata] = None,
    allow_pickle_fallback: bool = False,
    overwrite: bool = True,
) -> str:
    """Write a model to a qsarkit bundle directory.

    Parameters
    ----------
    bundle : ModelBundle or estimator
        A prepared bundle, or a bare fitted estimator -- in which case
        ``pipeline`` and ``metadata`` are used to build one.
    path : str or Path
        Destination directory. ``.qsar`` is appended when the path has no
        suffix, purely as a convention.
    pipeline : object, optional
        Only used when ``bundle`` is a bare estimator.
    metadata : ModelMetadata, optional
        Only used when ``bundle`` is a bare estimator.
    allow_pickle_fallback : bool, default False
        If skops cannot represent an object, fall back to
        :mod:`joblib` (which pickles). Off by default: the fallback
        reintroduces exactly the fragility and the arbitrary-code-execution
        risk this module exists to avoid, so it has to be asked for.
    overwrite : bool, default True
        Replace an existing bundle at ``path``. With ``False``, an
        existing directory raises.

    Returns
    -------
    str
        The directory written.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is False.
    OptionalDependencyError
        If ``skops`` is not installed and ``allow_pickle_fallback`` is False.

    Examples
    --------
    >>> import numpy as np, tempfile, os
    >>> from sklearn.linear_model import Ridge
    >>> from qsarkit.persistence import load_model, save_model
    >>> rng = np.random.default_rng(0)
    >>> X, y = rng.normal(size=(30, 4)), rng.normal(size=30)
    >>> out = os.path.join(tempfile.mkdtemp(), "ridge")
    >>> written = save_model(Ridge().fit(X, y), out)
    >>> written.endswith(".qsar")
    True
    >>> sorted(p.name for p in __import__("pathlib").Path(written).iterdir())
    ['estimator.skops', 'manifest.json', 'metadata.json']

    References
    ----------
    - skops: https://skops.readthedocs.io/en/stable/persistence.html
    """
    if not isinstance(bundle, ModelBundle):
        bundle = ModelBundle(bundle, pipeline=pipeline, metadata=metadata)

    target = Path(path)
    if not target.suffix:
        target = target.with_suffix(BUNDLE_SUFFIX)
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists. Pass overwrite=True to replace it."
            )
        import shutil

        shutil.rmtree(target)
    target.mkdir(parents=True)

    files: Dict[str, str] = {}
    files["estimator"] = _dump_object(
        bundle.estimator, target, _ESTIMATOR, _ESTIMATOR_PICKLE, allow_pickle_fallback
    )
    if bundle.pipeline is not None:
        files["pipeline"] = _dump_object(
            bundle.pipeline, target, _PIPELINE, _PIPELINE_PICKLE, allow_pickle_fallback
        )

    _write_json(target / _METADATA, bundle.metadata.to_dict())
    _write_json(
        target / _MANIFEST,
        {
            "format": "qsarkit-model-bundle",
            "format_version": BUNDLE_FORMAT_VERSION,
            "files": files,
            "estimator_class": _qualname(bundle.estimator),
            "pipeline_class": (
                _qualname(bundle.pipeline) if bundle.pipeline is not None else None
            ),
        },
    )
    return str(target)


def _qualname(obj: Any) -> str:
    """Fully qualified class name, for the manifest."""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _dump_object(
    obj: Any,
    directory: Path,
    skops_name: str,
    pickle_name: str,
    allow_pickle_fallback: bool,
) -> str:
    """Write one object, preferring skops and falling back only if allowed.

    Returns
    -------
    str
        The filename written, so the manifest can record which format was
        actually used.
    """
    try:
        skops_io = _require_skops()
    except OptionalDependencyError:
        if not allow_pickle_fallback:
            raise
        return _dump_pickle(obj, directory / pickle_name)

    try:
        skops_io.dump(obj, directory / skops_name)
    except Exception as exc:
        # Deliberately broad. skops raises TypeError for an object it cannot
        # represent, but also surfaces pickle's own errors and its internal
        # types, and the set differs between versions. Every one of them
        # means "this object cannot be stored this way", and the original is
        # chained into the message below, so nothing is hidden.
        if not allow_pickle_fallback:
            raise TypeError(
                f"skops could not serialize {_qualname(obj)}: {exc}. It "
                "supports scikit-learn estimators and NumPy data, not "
                "arbitrary Python objects. Either wrap the object in a "
                "scikit-learn-compatible estimator, or pass "
                "allow_pickle_fallback=True and accept that the file will "
                "be pickle-based, version-fragile and unsafe to load from "
                "an untrusted source."
            ) from exc
        return _dump_pickle(obj, directory / pickle_name)
    return skops_name


def _dump_pickle(obj: Any, path: Path) -> str:
    """Last-resort joblib dump, with a warning naming the consequence."""
    import joblib

    warnings.warn(
        f"Falling back to joblib (pickle) for {_qualname(obj)}. The "
        "resulting file may not load under a different scikit-learn or "
        "NumPy version, and loading it executes code, so treat it as "
        "trusted input only.",
        UserWarning,
        stacklevel=4,
    )
    joblib.dump(obj, path)
    return path.name


def load_model(
    path: PathLike,
    trusted: Optional[List[str]] = None,
    warn_on_environment_change: bool = True,
) -> ModelBundle:
    """Read a qsarkit bundle written by :func:`save_model`.

    Parameters
    ----------
    path : str or Path
        The bundle directory.
    trusted : list of str, optional
        Extra type names to allow skops to reconstruct, beyond what it
        trusts by default. Inspect a bundle first with
        :func:`inspect_bundle` and pass only what you recognize -- this is
        the mechanism that makes loading a third-party model safe, so
        blanket-trusting everything defeats it.
    warn_on_environment_change : bool, default True
        Warn when the current package versions differ from those recorded
        at save time. A model is not guaranteed to reproduce its original
        predictions across versions, and silence would hide that.

    Returns
    -------
    ModelBundle

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a bundle directory.
    ValueError
        If the manifest is missing, unreadable, or written in a bundle
        format this version does not understand.

    Examples
    --------
    >>> import numpy as np, tempfile, os
    >>> from sklearn.linear_model import Ridge
    >>> from qsarkit.persistence import load_model, save_model
    >>> rng = np.random.default_rng(0)
    >>> X, y = rng.normal(size=(30, 4)), rng.normal(size=30)
    >>> out = save_model(Ridge().fit(X, y), os.path.join(tempfile.mkdtemp(), "m"))
    >>> bundle = load_model(out)
    >>> bundle.metadata.task
    'regression'
    >>> bundle.predict(X).shape
    (30,)
    """
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} is not a directory. A qsarkit bundle is a "
            "directory containing manifest.json, metadata.json and the "
            "serialized estimator."
        )
    manifest_path = directory / _MANIFEST
    if not manifest_path.is_file():
        raise ValueError(
            f"{directory} has no {_MANIFEST}, so it is not a qsarkit "
            "bundle. If this is a bare .skops or .joblib file, load it "
            "with that library directly."
        )
    manifest = _read_json(manifest_path)

    written = str(manifest.get("format_version", "0"))
    if written > BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"This bundle uses format version {written}, but this qsarkit "
            f"({BUNDLE_FORMAT_VERSION}) can only read up to "
            f"{BUNDLE_FORMAT_VERSION}. Upgrade qsarkit to load it."
        )

    metadata_path = directory / _METADATA
    metadata = (
        ModelMetadata.from_dict(_read_json(metadata_path))
        if metadata_path.is_file()
        else ModelMetadata()
    )

    if warn_on_environment_change:
        differences = metadata.environment_differences()
        if differences:
            detail = "; ".join(f"{k} ({v})" for k, v in sorted(differences.items()))
            warnings.warn(
                f"Model {metadata.name!r} was saved under different package "
                f"versions: {detail}. Predictions may differ from the "
                "originals; re-validate before relying on them.",
                UserWarning,
                stacklevel=2,
            )

    files = manifest.get("files", {})
    estimator = _load_object(directory, files.get("estimator", _ESTIMATOR), trusted)
    pipeline = (
        _load_object(directory, files["pipeline"], trusted)
        if files.get("pipeline")
        else None
    )
    return ModelBundle(estimator, pipeline=pipeline, metadata=metadata)



#: Namespaces qsarkit trusts without the caller listing individual types.
#:
#: skops refuses, by design, to reconstruct any type it does not explicitly
#: trust, and several estimators qsarkit offers are built from types on that
#: list: ``rf`` from ``sklearn.tree._tree.Tree``, ``gbm`` from
#: ``TreePredictor``, ``mlp`` from ``AdamOptimizer``, ``gp`` from the
#: Gaussian-process kernels. skops 0.16 stopped trusting ``Tree`` by
#: default, which silently broke loading a saved random forest -- the
#: package default, and every persistence example in the documentation.
#:
#: This is a namespace policy rather than a list of type names because the
#: set of refused types is not stable enough to enumerate: for the same
#: estimator on the same data, skops 0.13 refuses five types from ``gbm``
#: (``IdentityLink``, ``Interval``, ``HalfSquaredError``, ``_BinMapper``,
#: ``TreePredictor``) where skops 0.16 refuses one, and which
#: Gaussian-process kernels appear depends on the fitted kernel. A
#: hardcoded list would break on every upgrade on either side.
#:
#: What this does and does not promise: the class is always resolved from
#: the installed library, never built from the file, so no code arrives with
#: the bundle. The file does supply the attributes, and for ``Tree`` those
#: are raw node indices that scikit-learn follows without bounds checking,
#: so a hostile bundle can still crash the process at ``predict`` time.
#: Loading a bundle you did not produce remains an act of trust. Anything
#: outside these namespaces -- a third-party estimator, a hand-written
#: class, a pickled callable -- is still refused until the caller names it,
#: and :func:`inspect_bundle` reports both groups separately.
_AUTO_TRUSTED_PREFIXES: Tuple[str, ...] = ("qsarkit.", "sklearn.")


def _is_auto_trusted(name: str) -> bool:
    """Whether qsarkit trusts ``name`` without the caller listing it."""
    return name.startswith(_AUTO_TRUSTED_PREFIXES)


def _auto_trusted_types(path: Path) -> List[str]:
    """The types inside a skops file that qsarkit trusts on its own."""
    skops_io = _require_skops()
    try:
        found = skops_io.get_untrusted_types(file=path)
    except Exception:  # pragma: no cover - unreadable file, reported later
        return []
    return [name for name in found if _is_auto_trusted(name)]


def _load_object(
    directory: Path, filename: str, trusted: Optional[List[str]]
) -> Any:
    """Load one serialized object, dispatching on its filename."""
    target = directory / filename
    if not target.is_file():
        raise ValueError(f"{directory} is missing its {filename}.")
    if target.suffix == ".joblib":
        import joblib

        return joblib.load(target)

    skops_io = _require_skops()
    allowed = _auto_trusted_types(target) + list(trusted or ())
    try:
        return skops_io.load(target, trusted=allowed)
    except Exception as exc:
        # Deliberately broad: skops signals an untrusted type with its own
        # UntrustedTypesFoundException, a corrupt file with a JSON or zip
        # error, and a version mismatch with something else again. The
        # caller's next step is the same in every case, and the original
        # exception is chained.
        raise ValueError(
            f"skops declined to load {filename}: {exc}\n"
            "Run qsarkit.persistence.inspect_bundle() on this bundle to see "
            "which types it contains, then pass the ones you recognize as "
            "load_model(..., trusted=[...]). qsarkit's own classes are "
            "trusted automatically; this list is for everything else."
        ) from exc


def inspect_bundle(path: PathLike) -> Dict[str, Any]:
    """Describe a bundle without reconstructing any object from it.

    The safe first step with a model from someone else: it reports the
    manifest, the metadata and the list of types skops would need to
    build, so you can decide what to trust before anything is executed.

    Parameters
    ----------
    path : str or Path
        The bundle directory.

    Returns
    -------
    dict
        ``manifest``, ``metadata``, ``untrusted`` and ``auto_trusted``.
        ``untrusted`` holds the type names that would block a load until
        they are passed to ``load_model(trusted=...)``; ``auto_trusted``
        holds the ones qsarkit accepts on your behalf -- its own classes,
        plus the scikit-learn internals its estimator menu produces (see
        :data:`_AUTO_TRUSTED_PREFIXES`). An empty ``untrusted`` means
        the bundle loads as it stands, not that nothing in it is
        executable, so inspect ``auto_trusted`` too for a bundle you did
        not produce.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a directory.
    ValueError
        If the manifest is missing.

    Examples
    --------
    >>> import numpy as np, tempfile, os
    >>> from sklearn.linear_model import Ridge
    >>> from qsarkit.persistence import inspect_bundle, save_model
    >>> rng = np.random.default_rng(0)
    >>> X, y = rng.normal(size=(30, 4)), rng.normal(size=30)
    >>> out = save_model(Ridge().fit(X, y), os.path.join(tempfile.mkdtemp(), "m"))
    >>> report = inspect_bundle(out)
    >>> report["manifest"]["estimator_class"]
    'sklearn.linear_model._ridge.Ridge'
    >>> report["untrusted"]
    []

    References
    ----------
    - skops, "Visualize and trust": https://skops.readthedocs.io/en/stable/persistence.html
    """
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory.")
    manifest_path = directory / _MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"{directory} has no {_MANIFEST}.")

    manifest = _read_json(manifest_path)
    metadata_path = directory / _METADATA
    metadata = _read_json(metadata_path) if metadata_path.is_file() else {}

    untrusted: List[str] = []
    auto_trusted: List[str] = []
    for filename in manifest.get("files", {}).values():
        target = directory / filename
        if target.suffix != ".skops" or not target.is_file():
            continue
        skops_io = _require_skops()
        # Split rather than filter: `untrusted` is what would block a load,
        # so it is the actionable list, but the types qsarkit trusts on the
        # caller's behalf are still reported -- hiding them would make this
        # a less honest answer to "what is in this file" than skops gives.
        for name in skops_io.get_untrusted_types(file=target):
            (auto_trusted if _is_auto_trusted(name) else untrusted).append(name)

    return {
        "manifest": manifest,
        "metadata": metadata,
        "untrusted": sorted(set(untrusted)),
        "auto_trusted": sorted(set(auto_trusted)),
    }
