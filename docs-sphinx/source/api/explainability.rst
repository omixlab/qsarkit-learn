Explainability
==============

.. currentmodule:: qsarkit.explainability

SHAP, LIME, permutation importance, per-atom contribution maps,
fragment analysis, partial dependence and counterfactuals.

For a QSAR model the interesting question is not "which feature index
mattered" but "which part of *this molecule* mattered", which is what the
atom-level tools answer.

Permutation importance
----------------------

Model-agnostic, and the only importance measure here that reflects the
model's actual predictive reliance rather than its internal structure:

.. doctest::

   >>> from qsarkit.explainability import PermutationImportance
   >>> from qsarkit.models import QSARRegressor
   >>> X, y = demo_fingerprints(256), DEMO_Y
   >>> model = QSARRegressor("rf", random_state=0).fit(X, y)
   >>> importance = PermutationImportance(n_repeats=5, random_state=0).fit(model, X, y)
   >>> importance.importances_mean_.shape
   (256,)

Measure it on held-out data. Permuting a feature on the training set
reports how much the model *memorized* through it, which is not the same
quantity and is usually larger.

Per-atom contributions
----------------------

The chemist-facing explanation: mask each atom in turn, re-predict, and
attribute the change to that atom.

.. doctest::

   >>> from qsarkit.explainability import AtomicContributionMap
   >>> from qsarkit.representation import MorganFingerprint
   >>> explainer = AtomicContributionMap(model, MorganFingerprint(n_bits=256))
   >>> result = explainer.explain(demo_mols[0])
   >>> result.weights.shape
   (9,)

One weight per heavy atom, ready for RDKit's similarity maps.

.. note::

   Masking one atom of a symmetric ring changes little, because the
   remaining atoms set the same bits. That is not a defect of the method
   — it is a true statement about the model: no single one of those atoms
   is necessary, because the others are redundant with it. Read flat
   weights across a ring as "this ring matters as a unit".

SHAP and LIME
-------------

Both need the ``explainability`` extra::

   from qsarkit.explainability import LIMEExplainer, SHAPExplainer

   shap_values = SHAPExplainer(model).explain(X)      # needs qsarkit-learn[explainability]

SHAP's ``TreeExplainer`` is exact and fast for tree ensembles;
``KernelExplainer`` is model-agnostic and slow enough that you will want
to subsample. LIME fits a local surrogate, which makes it cheap and
local-only — a LIME explanation says nothing about a different molecule.

From bits back to atoms
-----------------------

Both explainers attribute to *fingerprint bits*, which is not an answer a
chemist can act on. :class:`AttributionAtomMapper` carries those
attributions back onto the molecule, so either explainer ends in the same
picture :func:`draw_atom_weights` draws:

.. doctest::

   >>> from qsarkit.explainability import AttributionAtomMapper, draw_atom_weights
   >>> from qsarkit.representation import MorganFingerprint
   >>> fingerprint = MorganFingerprint(radius=2, n_bits=256)
   >>> mapper = AttributionAtomMapper(fingerprint)
   >>> weights = mapper.from_lime(demo_mols[0], {next(iter(fingerprint.bit_info(demo_mols[0]))): 1.0})
   >>> weights.shape
   (9,)
   >>> svg = draw_atom_weights(demo_mols[0], weights, fmt="svg")

``from_shap`` takes a :class:`SHAPExplainer`; ``from_lime`` takes whatever
LIME gives you — an ``explain_one`` frame, a LIME ``Explanation``, a
``{bit: weight}`` mapping, or pairs.

The mapper asks the fingerprint which atoms set each bit, through
:meth:`~qsarkit.representation.MorganFingerprint.bit_info`, rather than
re-deriving the bit numbering from ``radius`` and ``n_bits``. This matters
more than it sounds: ``use_chirality``, ``use_bond_types`` and feature
invariants all change the hashing, so a re-derived map can disagree with
the matrix the model was fitted on, and the resulting picture is wrong
while looking entirely plausible.

.. warning::

   Atom mapping needs bits that *mean* local atom environments. MACCS keys
   and path-based fingerprints have no such correspondence, so there is
   nothing to place on an atom; :class:`AttributionAtomMapper` refuses them
   at construction rather than guessing.

   Check ``mapper.collision_rate(mol)`` before reading a map closely. One
   bit set by several environments bounds how specifically any attribution
   can be assigned, however confident the colours look.

Fragments and partial dependence
--------------------------------

.. doctest::

   >>> from qsarkit.explainability import FragmentContributionAnalyzer, PartialDependence
   >>> analyzer = FragmentContributionAnalyzer(model, MorganFingerprint(n_bits=256))
   >>> hasattr(analyzer, "analyze")
   True
   >>> pd = PartialDependence(model, grid_resolution=10)
   >>> hasattr(pd, "compute")
   True

Fragment contributions aggregate atom-level attributions over chemically
meaningful groups, which is usually more actionable than either the atom
or the bit.

API
---

.. automodule:: qsarkit.explainability
   :members:
   :show-inheritance:

References
----------

- Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to
  Interpreting Model Predictions." NeurIPS 2017, 4765-4774.
  https://papers.nips.cc/paper/7062
- Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). "Why Should I Trust
  You?" KDD 2016, 1135-1144. :doi:`10.1145/2939672.2939778`
- Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
  :doi:`10.1023/A:1010933404324`
- Riniker, S. & Landrum, G. A. (2013). "Similarity Maps — A Visualization
  Strategy for Molecular Fingerprints and Machine-Learning Methods."
  J. Cheminform., 5, 43. :doi:`10.1186/1758-2946-5-43`
- Polishchuk, P. et al. (2016). "Universal Approach for Structural
  Interpretation of QSAR/QSPR Models." Mol. Inform., 32(9-10), 843-853.
  :doi:`10.1002/minf.201300029`
