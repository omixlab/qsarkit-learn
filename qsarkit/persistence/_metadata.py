"""Provenance metadata stored beside every serialized model.

A model file that does not record what produced it is unusable a year
later: you cannot tell which qsarkit version wrote it, which descriptor
block it expects, or what the numbers meant. This is the part of
persistence that matters for OECD principle 2 (an unambiguous algorithm),
and it is plain JSON so it can be read without importing anything.

References
----------
- OECD (2007). "Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models." OECD Series on Testing
  and Assessment No. 69, ENV/JM/MONO(2007)2.
  https://doi.org/10.1787/9789264085442-en
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = ["ModelMetadata", "environment_summary"]

#: Format version of the qsarkit bundle layout itself. Bumped only when the
#: on-disk structure changes in a way an older reader could not handle.
BUNDLE_FORMAT_VERSION = "1"


def environment_summary() -> Dict[str, str]:
    """Versions of the packages a saved model's behaviour depends on.

    Recorded at save time so a later load can warn about a mismatch
    rather than silently predicting something different.

    Returns
    -------
    dict
        Package name -> version, plus ``python`` and ``platform``.

    Examples
    --------
    >>> from qsarkit.persistence import environment_summary
    >>> summary = environment_summary()
    >>> "qsarkit" in summary and "python" in summary
    True
    """
    import numpy
    import sklearn

    import qsarkit

    versions: Dict[str, str] = {
        "qsarkit": qsarkit.__version__,
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "platform": platform.system(),
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    # Optional, and only interesting when actually installed.
    for name, module in (("rdkit", "rdkit"), ("scipy", "scipy")):
        try:
            versions[name] = __import__(module).__version__
        except Exception:  # pragma: no cover - import guard
            pass
    return versions


@dataclass
class ModelMetadata:
    """What a saved model needs to carry to remain interpretable.

    Attributes
    ----------
    name : str
        Human-readable model name.
    endpoint : str
        What the model predicts, and in what units -- ``"pIC50
        (-log10 M)"`` rather than ``"activity"``.
    task : str
        ``"regression"`` or ``"classification"``.
    qsarkit_version : str
        Version that wrote the file.
    format_version : str
        Version of the bundle layout.
    created : str
        UTC ISO-8601 timestamp.
    environment : dict
        Output of :func:`environment_summary` at save time.
    n_features : int, optional
        Expected width of the feature matrix. Checked on load, because a
        width mismatch is the failure that otherwise produces confident
        nonsense.
    feature_names : list of str, optional
        Column names, where the representation provides them.
    n_training_samples : int, optional
        How many compounds the model was fitted on.
    description : str
        Free text.
    extra : dict
        Anything else worth recording -- dataset DOI, assay, curation
        settings, validation scores.

    Examples
    --------
    >>> from qsarkit.persistence import ModelMetadata
    >>> meta = ModelMetadata(name="demo", endpoint="pIC50", task="regression")
    >>> meta.task
    'regression'
    >>> restored = ModelMetadata.from_dict(meta.to_dict())
    >>> restored.name == meta.name
    True
    """

    name: str = ""
    endpoint: str = ""
    task: str = "regression"
    qsarkit_version: str = ""
    format_version: str = BUNDLE_FORMAT_VERSION
    created: str = ""
    environment: Dict[str, str] = field(default_factory=dict)
    n_features: Optional[int] = None
    feature_names: Optional[List[str]] = None
    n_training_samples: Optional[int] = None
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in ("regression", "classification"):
            raise ValueError(
                f"task must be 'regression' or 'classification', got {self.task!r}."
            )
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.environment:
            self.environment = environment_summary()
        if not self.qsarkit_version:
            self.qsarkit_version = self.environment.get("qsarkit", "")

    def to_dict(self) -> Dict[str, Any]:
        """Render as a JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys.

        Unknown keys are dropped rather than raising, so a bundle written
        by a newer qsarkit that added a field still loads here.

        Parameters
        ----------
        data : dict

        Returns
        -------
        ModelMetadata
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def environment_differences(self) -> Dict[str, str]:
        """Packages whose current version differs from the recorded one.

        Returns
        -------
        dict
            Package name -> ``"saved X, now Y"`` for each mismatch.

        Examples
        --------
        >>> from qsarkit.persistence import ModelMetadata
        >>> meta = ModelMetadata(name="demo")
        >>> meta.environment_differences()          # same session, so none
        {}
        >>> meta.environment["numpy"] = "0.0.1"
        >>> "numpy" in meta.environment_differences()
        True
        """
        current = environment_summary()
        return {
            package: f"saved {saved}, now {current[package]}"
            for package, saved in self.environment.items()
            if package in current and current[package] != saved
        }
