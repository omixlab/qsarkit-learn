"""Molecular standardization pipeline."""

from __future__ import annotations

from typing import Any, List, Optional

from qsarkit.base import InvalidMoleculeError, MoleculeToMoleculeTransformer


class MolecularStandardizer(MoleculeToMoleculeTransformer):
    """Standardize a batch of molecules into a canonical, model-ready form.

    Applies, in order: sanitization, salt/solvent removal (keep the largest
    organic fragment), charge neutralization, tautomer normalization,
    stereochemistry handling, and explicit-hydrogen normalization. This
    mirrors the "structure normalization" stage expected before any
    downstream QSAR/curation step (OECD QSAR guidance recommends
    normalized, unambiguous structures prior to model building).

    Parameters
    ----------
    remove_salts : bool, default True
        Keep only the largest organic fragment (strips counter-ions,
        solvates, hydrates).
    neutralize : bool, default True
        Neutralize charges where a neutral tautomer/protomer exists
        (e.g. carboxylates, ammoniums), leaving permanent charges
        (e.g. quaternary ammonium) untouched.
    normalize_tautomers : bool, default True
        Canonicalize to the RDKit-preferred tautomer using the
        Sybyl/MolVS-derived tautomer scoring rules.
    handle_stereochemistry : str, default "retain"
        One of ``"retain"`` (keep stereo as parsed, but reassign
        stereocenters from the 2D/3D structure), or ``"remove"`` (strip
        all stereochemistry, useful when comparing 2D scaffolds).
    normalize_hydrogens : bool, default True
        Strip explicit hydrogens except where required for correct
        valence/stereo perception (RDKit's implicit-H convention).
    on_error : str, default "none"
        ``"none"`` -> failed molecules become ``None`` in the output list
        (positional alignment preserved); ``"raise"`` -> raise
        ``InvalidMoleculeError`` on the first failure.

    Examples
    --------
    Sodium acetylsalicylate loses its counter-ion and its charge:

    >>> from rdkit import Chem
    >>> from qsarkit.chemistry import MolecularStandardizer
    >>> standardizer = MolecularStandardizer()
    >>> mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
    >>> Chem.MolToSmiles(standardizer.transform([mol])[0])
    'CC(=O)Oc1ccccc1C(=O)O'

    Failures do not shift the batch. An unparseable record becomes
    ``None`` in place, so a parallel array of activities stays aligned:

    >>> out = standardizer.transform([mol, None])
    >>> [m is None for m in out]
    [False, True]

    ``handle_stereochemistry="remove"`` strips stereocentres, which is
    what you want when comparing 2D scaffolds rather than modelling
    enantiomer-specific activity:

    >>> flat = MolecularStandardizer(handle_stereochemistry="remove")
    >>> Chem.MolToSmiles(flat.transform([Chem.MolFromSmiles("C[C@H](N)C(=O)O")])[0])
    'CC(N)C(=O)O'

    References
    ----------
    - Sitzmann et al. (2010). "Tautomerism in Large Databases." J. Comput.
      Aided Mol. Des., 24, 521-551. https://doi.org/10.1007/s10822-010-9346-4
    - RDKit MolStandardize documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.html
    - OECD (2007). "Guidance Document on the Validation of (Quantitative)
      Structure-Activity Relationship [(Q)SAR] Models," ENV/JM/MONO(2007)2.
      https://doi.org/10.1787/9789264085442-en
    """

    def __init__(
        self,
        remove_salts: bool = True,
        neutralize: bool = True,
        normalize_tautomers: bool = True,
        handle_stereochemistry: str = "retain",
        normalize_hydrogens: bool = True,
        on_error: str = "none",
    ):
        self.remove_salts = remove_salts
        self.neutralize = neutralize
        self.normalize_tautomers = normalize_tautomers
        self.handle_stereochemistry = handle_stereochemistry
        self.normalize_hydrogens = normalize_hydrogens
        self.on_error = on_error

    def _standardize_one(self, mol: Any) -> Optional[Any]:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        try:
            Chem.SanitizeMol(mol)

            if self.remove_salts:
                remover = rdMolStandardize.LargestFragmentChooser()
                mol = remover.choose(mol)

            if self.neutralize:
                uncharger = rdMolStandardize.Uncharger()
                mol = uncharger.uncharge(mol)

            if self.normalize_tautomers:
                enumerator = rdMolStandardize.TautomerEnumerator()
                # By default RDKit strips sp3 stereo at atoms it considers
                # tautomeric, which would silently destroy genuine
                # stereocentres (e.g. the alpha carbon of an amino acid).
                # Only allow that when the caller asked to drop stereo.
                enumerator.SetRemoveSp3Stereo(self.handle_stereochemistry == "remove")
                mol = enumerator.Canonicalize(mol)

            if self.handle_stereochemistry == "remove":
                Chem.RemoveStereochemistry(mol)
            elif self.handle_stereochemistry == "retain":
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            else:
                raise ValueError(
                    "handle_stereochemistry must be 'retain' or 'remove', "
                    f"got {self.handle_stereochemistry!r}"
                )

            if self.normalize_hydrogens:
                mol = Chem.RemoveHs(mol)

            Chem.SanitizeMol(mol)
            return mol
        except Exception as exc:
            if self.on_error == "raise":
                raise InvalidMoleculeError(f"Standardization failed: {exc}") from exc
            return None

    def _transform(self, mols: List[Any]) -> List[Any]:
        return [None if m is None else self._standardize_one(m) for m in mols]
