Clustering
==========

.. currentmodule:: qsarkit.cluster

Cheminformatics clustering and diversity picking behind a scikit-learn
API: Taylor–Butina, sphere exclusion, MaxMin and hierarchical linkage,
all on Tanimoto distance.

.. warning::

   Euclidean distance is wrong for sparse binary fingerprints. Two
   molecules sharing no bits at all are "close" in Euclidean terms
   because they agree on the thousands of bits that are jointly zero —
   which carry no chemical information. Everything here uses
   Tanimoto/Jaccard.

Taylor–Butina
-------------

The standard cheminformatics clustering: no ``n_clusters`` to choose,
just a similarity cutoff, and every cluster has a real molecule at its
centre rather than an average that corresponds to nothing.

.. doctest::

   >>> from qsarkit.cluster import ButinaClustering
   >>> X = demo_fingerprints(256)
   >>> labels = ButinaClustering(cutoff=0.6).fit_predict(X)
   >>> labels[:8].tolist()
   [0, 0, 0, 0, 0, 0, 0, 1]
   >>> len(set(labels.tolist()))
   6

The cutoff is a Tanimoto *distance*, so a larger value merges more:

.. doctest::

   >>> len(set(ButinaClustering(cutoff=0.4).fit_predict(X).tolist()))
   14

Sphere exclusion and hierarchical
---------------------------------

.. doctest::

   >>> from qsarkit.cluster import HierarchicalClustering, SphereExclusionClustering
   >>> len(set(SphereExclusionClustering(cutoff=0.6).fit_predict(X).tolist()))
   6
   >>> HierarchicalClustering(n_clusters=3).fit_predict(X)[-4:].tolist()
   [1, 1, 2, 2]

Diversity picking
-----------------

MaxMin selects a maximally diverse subset — the right way to choose
compounds for a screening plate, or a representative subset of a large
library:

.. doctest::

   >>> from qsarkit.cluster import MaxMinPicker
   >>> MaxMinPicker(n_to_pick=4, seed_index=0).fit(X).picks_.tolist()
   [0, 21, 22, 8]

Each pick is the compound furthest from everything already picked, so the
selection covers the space rather than clustering in its densest region —
which is what a random sample would do.

API
---

.. automodule:: qsarkit.cluster
   :members:
   :show-inheritance:

References
----------

- Butina, D. (1999). "Unsupervised Data Base Clustering Based on Daylight's
  Fingerprint and Tanimoto Similarity." J. Chem. Inf. Comput. Sci.,
  39(4), 747-750. :doi:`10.1021/ci9803381`
- Taylor, R. (1995). "Simulation Analysis of Experimental Design
  Strategies for Screening Random Compounds." J. Chem. Inf. Comput.
  Sci., 35(1), 59-67. :doi:`10.1021/ci00023a009`
- Ashton, M. et al. (2002). "Identification of Diverse Database Subsets
  using Property-Based and Fragment-Based Molecular Descriptions."
  Quant. Struct.-Act. Relat., 21(6), 598-604.
  :doi:`10.1002/qsar.200290002`
- Willett, P. (2006). "Similarity-Based Virtual Screening Using 2D
  Fingerprints." Drug Discov. Today, 11(23-24), 1046-1053.
  :doi:`10.1016/j.drudis.2006.10.005`
