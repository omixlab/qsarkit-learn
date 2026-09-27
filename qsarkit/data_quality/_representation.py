"""What the representation makes impossible, before any model is fitted.

A QSAR model can only distinguish compounds its features distinguish. When two
molecules map to the same feature vector and carry different activities, no
estimator reading that matrix can be right about both: the error is irreducible
and belongs to the representation, not to the learner. Tuning cannot remove it,
and a score reported without it looks like a modelling failure when it is a
descriptor choice.

This module measures that floor. It is the feature-space counterpart of
:class:`~qsarkit.data_quality.DuplicateDetector`, which finds duplicate
*structures*: two distinct structures can be perfectly resolved by InChIKey and
still collide under a hashed fingerprint. Stereoisomers are the clearest case —
*cis*- and *trans*-stilbene have identical Morgan fingerprints — and
homologues are the commonest, since a radius-2 environment cannot count how
many times it repeats.

The practical use is as a ceiling to report beside a score: a balanced accuracy
of 0.62 against an attainable 0.64 is a different result from 0.62 against
1.00, and only the first says the model is nearly done.

References
----------
- Maggiora, G. M. (2006). "On Outliers and Activity Cliffs -- Why QSAR Often
  Disappoints." J. Chem. Inf. Model., 46(4), 1535.
  https://doi.org/10.1021/ci060117s
- Stumpfe, D. & Bajorath, J. (2012). "Exploring Activity Cliffs in Medicinal
  Chemistry." J. Med. Chem., 55(7), 2932-2942.
  https://doi.org/10.1021/jm201706b
- Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
  J. Chem. Inf. Model., 50(5), 742-754. https://doi.org/10.1021/ci100050t
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["RepresentationConflict", "representation_conflicts"]


@dataclass
class RepresentationConflict:
    """One group of compounds the representation cannot tell apart.

    Attributes
    ----------
    indices : list of int
        Rows sharing a feature vector.
    labels : list
        Their activities, in the same order.
    n_distinct_labels : int
        How many different activities the group carries. A group with one is
        a harmless collision; more than one is irreducible error.
    spread : float
        For a continuous endpoint, ``max - min`` within the group. ``0.0``
        when every member agrees.
    majority_label : Any
        The label a model minimising error would have to predict for the whole
        group.
    n_misassigned : int
        Members that prediction necessarily gets wrong.
    """

    indices: List[int]
    labels: List[Any]
    n_distinct_labels: int
    spread: float
    majority_label: Any
    n_misassigned: int


@dataclass
class RepresentationConflictReport:
    """The representation's ceiling on a dataset.

    Attributes
    ----------
    n_samples, n_features : int
    n_distinct_rows : int
        Feature vectors that occur at least once.
    n_collision_groups : int
        Distinct feature vectors shared by more than one compound, whether or
        not the labels agree.
    n_conflicting_groups : int
        Collision groups whose members disagree about the activity. These are
        the irreducible ones.
    n_compounds_in_conflict : int
    n_irreducible_errors : int
        Compounds that any model reading this matrix must get wrong.
    irreducible_error_rate : float
        ``n_irreducible_errors / n_samples``.
    max_accuracy : float or None
        The best accuracy attainable on this matrix (classification only).
    max_balanced_accuracy : float or None
    irreducible_rmse : float or None
        The best RMSE attainable (regression only), from predicting each
        group's mean.
    conflicts : list of RepresentationConflict
        The conflicting groups, largest first.
    """

    n_samples: int
    n_features: int
    n_distinct_rows: int
    n_collision_groups: int
    n_conflicting_groups: int
    n_compounds_in_conflict: int
    n_irreducible_errors: int
    irreducible_error_rate: float
    max_accuracy: Optional[float] = None
    max_balanced_accuracy: Optional[float] = None
    irreducible_rmse: Optional[float] = None
    conflicts: List[RepresentationConflict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-friendly summary, without the per-group detail."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_distinct_rows": self.n_distinct_rows,
            "n_collision_groups": self.n_collision_groups,
            "n_conflicting_groups": self.n_conflicting_groups,
            "n_compounds_in_conflict": self.n_compounds_in_conflict,
            "n_irreducible_errors": self.n_irreducible_errors,
            "irreducible_error_rate": self.irreducible_error_rate,
            "max_accuracy": self.max_accuracy,
            "max_balanced_accuracy": self.max_balanced_accuracy,
            "irreducible_rmse": self.irreducible_rmse,
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<RepresentationConflictReport {self.n_conflicting_groups} "
            f"conflicting groups, {self.n_irreducible_errors} irreducible "
            f"errors of {self.n_samples}>"
        )


def representation_conflicts(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    task: Literal["classification", "regression"] = "classification",
    tolerance: float = 0.0,
    max_conflicts: Optional[int] = 50,
) -> RepresentationConflictReport:
    """Find compounds the features cannot distinguish but the labels do.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The feature matrix a model would be fitted on.
    y : array-like of shape (n_samples,)
        Activities.
    task : {"classification", "regression"}, default "classification"
        Classification compares labels exactly and reports an attainable
        accuracy; regression treats a group as conflicting when its spread
        exceeds ``tolerance`` and reports an attainable RMSE.
    tolerance : float, default 0.0
        Regression only: activity differences at or below this are treated as
        agreement. Set it to the assay's experimental error, below which a
        disagreement is noise rather than signal.
    max_conflicts : int or None, default 50
        Keep at most this many conflicting groups in ``conflicts``, largest
        first. ``None`` keeps all of them. The summary counts are unaffected.

    Returns
    -------
    RepresentationConflictReport

    Raises
    ------
    ValueError
        If ``X`` is not 2-dimensional, the lengths disagree, ``task`` is
        unknown, or ``tolerance`` is negative.

    Notes
    -----
    Rows are grouped by exact equality, which is the right test for
    fingerprints and for descriptor matrices that have not been scaled.
    Two descriptor vectors differing in the last floating-point digit are
    *not* grouped, so on continuous descriptors this reports a lower bound on
    the ambiguity rather than all of it.

    Examples
    --------
    Two compounds share a feature vector and disagree, so one of them must be
    predicted wrongly whatever the model:

    >>> import numpy as np
    >>> from qsarkit.data_quality import representation_conflicts
    >>> X = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    >>> y = np.array([1, 0, 1, 0])
    >>> report = representation_conflicts(X, y)
    >>> report.n_conflicting_groups, report.n_irreducible_errors
    (1, 1)
    >>> round(report.max_accuracy, 2)
    0.75

    A dataset with no collisions has no ceiling below 1:

    >>> clean = representation_conflicts(np.eye(4), np.array([0, 1, 0, 1]))
    >>> clean.n_conflicting_groups, clean.max_accuracy
    (0, 1.0)

    On a continuous endpoint the floor is an RMSE rather than an error count:

    >>> X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> y = np.array([5.0, 7.0, 3.0])
    >>> report = representation_conflicts(X, y, task="regression")
    >>> report.n_conflicting_groups, round(report.irreducible_rmse, 3)
    (1, 0.816)
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()

    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X_arr.shape}.")
    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"X has {len(X_arr)} rows but y has {len(y_arr)} entries."
        )
    if task not in ("classification", "regression"):
        raise ValueError(
            f"task must be 'classification' or 'regression', got {task!r}."
        )
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}.")
    if len(X_arr) == 0:
        raise ValueError("Cannot analyse an empty dataset.")

    # Group identical rows. np.unique on axis=0 sorts lexicographically, which
    # is O(n log n) and needs no hashing scheme of our own.
    _, inverse, counts = np.unique(X_arr, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()

    n_samples = len(y_arr)
    conflicts: List[RepresentationConflict] = []
    n_irreducible = 0
    n_in_conflict = 0
    squared_error = 0.0

    for group_id in np.flatnonzero(counts > 1):
        members = np.flatnonzero(inverse == group_id)
        labels = y_arr[members]

        if task == "classification":
            values, value_counts = np.unique(labels, return_counts=True)
            if len(values) < 2:
                continue
            majority = values[int(np.argmax(value_counts))]
            misassigned = int(len(members) - value_counts.max())
            spread = 0.0
            if np.issubdtype(labels.dtype, np.number):
                spread = float(labels.max() - labels.min())
        else:
            spread = float(labels.max() - labels.min())
            if spread <= tolerance:
                continue
            majority = float(labels.mean())
            misassigned = int(len(members))
            squared_error += float(np.sum((labels - labels.mean()) ** 2))
            values = np.unique(labels)

        n_irreducible += misassigned
        n_in_conflict += len(members)
        conflicts.append(
            RepresentationConflict(
                indices=[int(i) for i in members],
                labels=[labels[i].item() for i in range(len(labels))],
                n_distinct_labels=int(len(values)),
                spread=round(spread, 6),
                majority_label=(
                    majority.item() if hasattr(majority, "item") else majority
                ),
                n_misassigned=misassigned,
            )
        )

    conflicts.sort(key=lambda c: (-len(c.indices), c.indices[0]))
    if max_conflicts is not None:
        conflicts = conflicts[:max_conflicts]

    report = RepresentationConflictReport(
        n_samples=n_samples,
        n_features=int(X_arr.shape[1]),
        n_distinct_rows=int(len(counts)),
        n_collision_groups=int((counts > 1).sum()),
        # Counted over every group, not over the (possibly capped) sample
        # kept in `conflicts`.
        n_conflicting_groups=_count_conflicting_groups(
            y_arr, inverse, counts, task, tolerance
        ),
        n_compounds_in_conflict=n_in_conflict,
        n_irreducible_errors=n_irreducible,
        irreducible_error_rate=round(n_irreducible / n_samples, 6),
        conflicts=conflicts,
    )

    if task == "classification":
        report.max_accuracy = round(1.0 - n_irreducible / n_samples, 6)
        report.max_balanced_accuracy = _max_balanced_accuracy(
            y_arr, inverse, counts
        )
    else:
        report.irreducible_rmse = round(float(np.sqrt(squared_error / n_samples)), 6)

    return report


def _count_conflicting_groups(
    y_arr: "npt.NDArray[Any]",
    inverse: "npt.NDArray[np.int64]",
    counts: "npt.NDArray[np.int64]",
    task: str,
    tolerance: float,
) -> int:
    """Conflicting groups, counted independently of the `max_conflicts` cap."""
    total = 0
    for group_id in np.flatnonzero(counts > 1):
        labels = y_arr[inverse == group_id]
        if task == "classification":
            if len(np.unique(labels)) > 1:
                total += 1
        elif float(labels.max() - labels.min()) > tolerance:
            total += 1
    return total


def _max_balanced_accuracy(
    y_arr: "npt.NDArray[Any]",
    inverse: "npt.NDArray[np.int64]",
    counts: "npt.NDArray[np.int64]",
) -> Optional[float]:
    """The best balanced accuracy any model on this matrix could reach.

    Balanced accuracy weights the two classes equally, so the optimal label for
    a group is the one maximising the *weighted* gain, not the majority. With
    class sizes n_pos and n_neg, predicting positive for a group gains
    ``pos/n_pos`` and predicting negative gains ``neg/n_neg``.
    """
    classes = np.unique(y_arr)
    if len(classes) != 2:
        return None

    negative, positive = classes[0], classes[1]
    n_pos = int(np.sum(y_arr == positive))
    n_neg = int(np.sum(y_arr == negative))
    if n_pos == 0 or n_neg == 0:
        return None

    true_pos = 0.0
    true_neg = 0.0
    for group_id in range(len(counts)):
        labels = y_arr[inverse == group_id]
        pos = int(np.sum(labels == positive))
        neg = int(len(labels) - pos)
        if pos / n_pos >= neg / n_neg:
            true_pos += pos
        else:
            true_neg += neg
    return round(0.5 * (true_pos / n_pos + true_neg / n_neg), 6)
