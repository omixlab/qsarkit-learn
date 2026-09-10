"""Shared plumbing for dense, named scalar molecular-descriptor blocks.

Every descriptor transformer in :mod:`qsarkit.representation.descriptors` is
a stateless :class:`~qsarkit.base.MoleculeTransformer` built from a list of
``(name, function)`` pairs, where each ``function`` maps one RDKit ``Mol`` to
a single float. Batching, ``None``-molecule handling, per-descriptor failure
handling and ``get_feature_names_out`` are provided here so every concrete
descriptor class is NaN-safe by construction: many RDKit descriptor
functions raise (rather than return NaN) on unusual valences, disconnected
fragments or zero-heavy-atom molecules, and a batch calculator must not let
one such molecule abort the whole design matrix.

References
----------
- RDKit documentation, "List of Available Descriptors":
  https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
- Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
  Chemoinformatics." Wiley-VCH. https://doi.org/10.1002/9783527628766
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from qsarkit.base import MoleculeTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["BaseDescriptorTransformer"]


class BaseDescriptorTransformer(MoleculeTransformer):
    """Abstract base for dense, named scalar molecular-descriptor blocks.

    Notes
    -----
    ``missing_value`` (a constructor argument on every concrete subclass,
    following the sklearn convention of storing constructor arguments
    verbatim) is substituted whenever a descriptor function raises or
    returns a non-finite value, and for every column of a ``None`` input
    molecule -- keeping the output positionally aligned with the input, as
    elsewhere in :mod:`qsarkit.representation`.

    Catching a bare ``Exception`` around each descriptor call is a
    deliberate exception to the "no defensive try/except around internal
    calls" rule: the functions dispatched here are heterogeneous,
    third-party (RDKit) callables applied to arbitrary user molecules, and
    a handful of them (e.g. ``Ipc`` on large fused-ring systems, most 3-D
    descriptors on disconnected inputs) are documented to raise rather than
    return NaN. This is exactly the kind of untrusted-input boundary the
    project style guide carves out for explicit error handling.

    References
    ----------
    - RDKit documentation, "List of Available Descriptors":
      https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
    """

    #: Value substituted for a descriptor that raised or was non-finite.
    #: Every concrete subclass declares this as a constructor argument
    #: (default ``nan``) and stores it verbatim, per the sklearn estimator
    #: convention; declared here so base-class methods can reference it.
    missing_value: float

    @abstractmethod
    def _descriptor_functions(self) -> List[Tuple[str, Callable[["Mol"], float]]]:
        """Return the ``(name, function)`` pairs computed by this block."""

    def _compute_row(
        self,
        mol: "Mol",
        functions: Sequence[Tuple[str, Callable[["Mol"], float]]],
    ) -> npt.NDArray[np.float64]:
        """Evaluate every descriptor function on ``mol``, NaN-safe."""
        row = np.full(len(functions), self.missing_value, dtype=np.float64)
        for j, (_, fn) in enumerate(functions):
            try:
                value = float(fn(mol))
            except Exception:  # noqa: BLE001 - heterogeneous third-party callables
                continue
            row[j] = value if np.isfinite(value) else self.missing_value
        return row

    def _transform(self, mols: List[Optional["Mol"]]) -> npt.NDArray[np.float64]:
        functions = self._descriptor_functions()
        out = np.full(
            (len(mols), len(functions)),
            self.missing_value,
            dtype=np.float64,
        )
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            out[i] = self._compute_row(mol, functions)
        return out

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> npt.NDArray[np.object_]:
        """Return the descriptor names produced by this transformer.

        Parameters
        ----------
        input_features : sequence of str, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            Array of ``str`` names, one per output column.

        Examples
        --------
        >>> from qsarkit.representation.descriptors import LipinskiDescriptors
        >>> "MolWt" in LipinskiDescriptors().get_feature_names_out()
        True
        """
        return np.asarray(
            [name for name, _ in self._descriptor_functions()], dtype=object
        )
