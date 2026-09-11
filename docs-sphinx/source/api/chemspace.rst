Chemical space
==============

.. currentmodule:: qsarkit.chemspace

Embedding, diversity, clustering, nearest-neighbour and scaffold analysis
of compound collections.

These are the questions you ask *before* modelling: does this library
cover one region or several, does the test set sit inside the training
set's cloud, is this screening hit an outlier.

Embedding
---------

.. doctest::

   >>> from qsarkit.chemspace import ChemicalSpaceAnalyzer
   >>> analyzer = ChemicalSpaceAnalyzer(random_state=0).fit(demo_mols)
   >>> analyzer.embedding_.shape
   (24, 2)
   >>> type(analyzer.plot()).__name__
   'Figure'

.. warning::

   The three methods are not interchangeable. PCA preserves global
   variance and its axes are interpretable. t-SNE and UMAP preserve local
   neighbourhoods and give the familiar island plots — but
   between-cluster distances in those plots are **not** meaningful.
   Reading them as chemical distance is the commonest misuse of the
   technique.

Diversity
---------

.. doctest::

   >>> from qsarkit.chemspace import DiversityAnalyzer
   >>> report = DiversityAnalyzer().analyze(demo_mols)
   >>> round(report["internal_diversity"], 3)
   0.763
   >>> report["n_scaffolds"], report["acyclic_fraction"]
   (3.0, 0.0)

Comparing libraries needs no extra code:

.. doctest::

   >>> frame = DiversityAnalyzer().compare({
   ...     "benzoic acids": demo_mols[:6],
   ...     "benzimidazoles": demo_mols[18:],
   ... })
   >>> len(frame)
   2

Scaffolds
---------

.. doctest::

   >>> from qsarkit.chemspace import ScaffoldAnalyzer
   >>> scaffolds = ScaffoldAnalyzer().fit(demo_mols)
   >>> scaffolds.n_scaffolds
   3
   >>> scaffolds.most_common(2)[0][1]
   12

Acyclic molecules have no Bemis-Murcko framework at all. They are
reported separately rather than counted as a scaffold of their own —
otherwise a library of straight chains would look scaffold-diverse:

.. doctest::

   >>> scaffolds.summary()["acyclic_fraction"]
   0.0

Novelty and coverage
--------------------

.. doctest::

   >>> from qsarkit.chemspace import NearestNeighborAnalyzer
   >>> nn = NearestNeighborAnalyzer().fit(demo_mols[:18])
   >>> novelty = nn.novelty(demo_mols[18:])
   >>> novelty.shape
   (6,)
   >>> bool((novelty > 0).all())
   True

``redundancy`` compares a library against itself, excluding each
molecule's own row — so an exact duplicate is detected rather than masked:

.. doctest::

   >>> library = demo_mols + [demo_mols[0]]
   >>> analyzer = NearestNeighborAnalyzer().fit(library)
   >>> round(analyzer.redundancy(library, threshold=0.99), 3)
   0.08

.. doctest::

   >>> from qsarkit.chemspace import ChemicalSpaceCoverage
   >>> coverage = ChemicalSpaceCoverage(threshold=0.5).compare(
   ...     demo_mols[18:], demo_mols[:18])
   >>> round(coverage["novel_fraction"], 3)
   1.0

The benzimidazoles are entirely novel relative to the other three series
— which is exactly why a scaffold split holding them out is a hard test.

API
---

.. automodule:: qsarkit.chemspace
   :members:
   :show-inheritance:

References
----------

- van der Maaten, L. & Hinton, G. (2008). "Visualizing Data Using
  t-SNE." J. Mach. Learn. Res., 9, 2579-2605.
  https://jmlr.org/papers/v9/vandermaaten08a.html
- McInnes, L., Healy, J. & Melville, J. (2018). "UMAP." :arxiv:`1802.03426`
- Wattenberg, M., Viegas, F. & Johnson, I. (2016). "How to Use t-SNE
  Effectively." Distill. :doi:`10.23915/distill.00002`
- Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known Drugs. 1.
  Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
  :doi:`10.1021/jm9602928`
- Sheridan, R. P. et al. (2004). "Similarity to Molecules in the Training
  Set Is a Good Discriminator for Prediction Accuracy in QSAR." J. Chem.
  Inf. Comput. Sci., 44(6), 1912-1928. :doi:`10.1021/ci049782w`
