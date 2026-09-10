"""3-D shape descriptors from an ETKDGv3-embedded, MMFF94-minimized conformer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from qsarkit.representation.descriptors._base import BaseDescriptorTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["Descriptors3D"]

#: The ten descriptors exposed by ``rdkit.Chem.Descriptors3D``.
_DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "NPR1",
    "NPR2",
    "PMI1",
    "PMI2",
    "PMI3",
    "RadiusOfGyration",
    "SpherocityIndex",
)


class Descriptors3D(BaseDescriptorTransformer):
    """3-D shape descriptors from an ETKDGv3-embedded, MMFF94-optimized conformer.

    Each molecule is protonated and embedded with RDKit's ETKDGv3 distance
    geometry algorithm (``n_confs`` trial conformers, seeded via
    ``random_state`` for reproducibility). Every embedded conformer is then
    geometry-optimized with the MMFF94 force field and the lowest-energy
    one is kept; the ten shape descriptors of ``rdkit.Chem.Descriptors3D``
    (principal-moments-of-inertia ratios NPR1/NPR2, asphericity,
    eccentricity, radius of gyration, spherocity index, ...) are then
    computed on that single best conformer.

    Parameters
    ----------
    n_confs : int, default 10
        Number of trial conformers generated per molecule; the
        lowest-MMFF94-energy one (after optimization) is kept.
    random_state : int, default 42
        Seed for the ETKDGv3 embedding (``randomSeed``), for deterministic,
        reproducible output.
    optimize : bool, default True
        Run an MMFF94 geometry minimization on every embedded conformer
        before ranking by energy. Skipping it (``False``) is faster but
        yields noisier, un-relaxed geometries.
    max_iters : int, default 200
        Maximum MMFF94 minimization iterations per conformer.
    missing_value : float, default nan
        Value substituted when embedding fails entirely (e.g. molecules
        RDKit cannot parametrize, or with fewer than 2 atoms) or a
        descriptor raises.

    Notes
    -----
    Distance-geometry embedding is inherently stochastic; determinism here
    comes entirely from fixing ``randomSeed`` in ``AllChem.ETKDGv3()``,
    which is RDKit's documented approach to reproducible conformer
    generation.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.descriptors import Descriptors3D
    >>> d3d = Descriptors3D(n_confs=2, random_state=0)
    >>> X = d3d.fit_transform([Chem.MolFromSmiles("CCO")])
    >>> X.shape
    (1, 10)

    References
    ----------
    - Riniker, S. & Landrum, G. A. (2015). "Better Informed Distance
      Geometry: Using What We Know To Improve Conformation Generation."
      J. Chem. Inf. Model., 55(12), 2562-2574.
      https://doi.org/10.1021/acs.jcim.5b00654
    - Halgren, T. A. (1996). "Merck Molecular Force Field. I. Basis, Form,
      Scope, Parameterization, and Performance of MMFF94." J. Comput.
      Chem., 17(5-6), 490-519.
      https://doi.org/10.1002/(SICI)1096-987X(199604)17:5/6%3C490::AID-JCC1%3E3.0.CO;2-P
    - RDKit ``rdkit.Chem.Descriptors3D`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors3D.html
    """

    def __init__(
        self,
        n_confs: int = 10,
        random_state: int = 42,
        optimize: bool = True,
        max_iters: int = 200,
        missing_value: float = float("nan"),
    ) -> None:
        self.n_confs = n_confs
        self.random_state = random_state
        self.optimize = optimize
        self.max_iters = max_iters
        self.missing_value = missing_value

    def _descriptor_functions(self) -> List[Tuple[str, Callable[["Mol"], float]]]:
        from rdkit.Chem import Descriptors3D as _d3d

        return [(name, getattr(_d3d, name)) for name in _DESCRIPTOR_NAMES]

    def _embed_best_conformer(self, mol: "Mol") -> Optional["Mol"]:
        """Embed ``n_confs`` ETKDGv3 conformers and keep the lowest-energy one."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        if mol.GetNumAtoms() < 2:
            return None

        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(self.random_state)
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=int(self.n_confs), params=params))
        if not conf_ids:
            return None

        best_cid = conf_ids[0]
        if self.optimize:
            best_energy = float("inf")
            props = AllChem.MMFFGetMoleculeProperties(mol_h)
            for cid in conf_ids:
                if props is None:
                    break
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=cid)
                if ff is None:
                    continue
                ff.Minimize(maxIts=int(self.max_iters))
                energy = ff.CalcEnergy()
                if energy < best_energy:
                    best_energy, best_cid = energy, cid

        # Isolate the winning conformer so the confId=-1 default used by
        # Descriptors3D functions resolves to it unambiguously.
        single = Chem.Mol(mol_h)
        single.RemoveAllConformers()
        single.AddConformer(mol_h.GetConformer(best_cid), assignId=True)
        return single

    def _transform(self, mols: List[Optional["Mol"]]) -> npt.NDArray[np.float64]:
        functions = self._descriptor_functions()
        out = np.full((len(mols), len(functions)), self.missing_value, dtype=np.float64)
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            try:
                embedded = self._embed_best_conformer(mol)
            except Exception:  # noqa: BLE001 - embedding is best-effort per molecule
                continue
            if embedded is None:
                continue
            out[i] = self._compute_row(embedded, functions)
        return out
