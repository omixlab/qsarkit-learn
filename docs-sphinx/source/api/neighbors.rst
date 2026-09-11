Neighbors
=========

.. currentmodule:: qsarkit.neighbors

Tanimoto/Jaccard similarity search and k-NN estimators for fingerprints.

.. warning::

   Euclidean distance is the wrong metric for sparse binary
   fingerprints. Two molecules that share no substructures are "close" in
   Euclidean terms because they agree on the thousands of bits that are
   jointly zero — bits that carry no chemical information. Every distance
   here is Tanimoto/Jaccard, verified against RDKit's
   ``BulkTanimotoSimilarity``.

Distances
---------

.. doctest::

   >>> from qsarkit.neighbors import is_binary, jaccard_distance, tanimoto_similarity_matrix
   >>> X = demo_fingerprints(256)
   >>> round(float(jaccard_distance(X[0], X[1])), 3)
   0.45
   >>> is_binary(X)
   True
   >>> similarity = tanimoto_similarity_matrix(X)
   >>> similarity.shape, round(float(similarity[0, 1]), 3)
   ((24, 24), 0.55)

Similarity search
-----------------

.. doctest::

   >>> from qsarkit.neighbors import JaccardNeighborSearch
   >>> search = JaccardNeighborSearch(n_neighbors=3).fit(X)
   >>> distances, indices = search.kneighbors(X[:1])
   >>> indices.tolist()
   [[0, 3, 2]]

A threshold search returns everything similar enough, rather than a fixed
count — which is what a chemist actually wants when asking "what else
looks like this":

.. doctest::

   >>> hits = search.similarity_search(X[:1], threshold=0.5)
   >>> [(i, round(s, 2)) for i, s in hits[0]]
   [(0, 1.0), (3, 0.58), (1, 0.55), (2, 0.55), (5, 0.55)]

k-NN estimators
---------------

.. doctest::

   >>> from qsarkit.neighbors import JaccardKNeighborsRegressor
   >>> model = JaccardKNeighborsRegressor(n_neighbors=3).fit(X, DEMO_Y)
   >>> round(float(model.predict(X[:1])[0]), 2)
   6.1

``weights="similarity"`` weights neighbours by Tanimoto similarity rather
than inverse distance, which is the natural reading for fingerprints:

.. doctest::

   >>> weighted = JaccardKNeighborsRegressor(
   ...     n_neighbors=3, weights="similarity").fit(X, DEMO_Y)
   >>> weighted.predict(X[:1]).shape
   (1,)

A k-NN model on Tanimoto distance is worth fitting even when you intend
to use something else: it is the direct expression of the similar
property principle, so it is the baseline any more complex model has to
beat to justify itself.

API
---

.. automodule:: qsarkit.neighbors
   :members:
   :show-inheritance:

References
----------

- Willett, P. (2006). "Similarity-Based Virtual Screening Using 2D
  Fingerprints." Drug Discov. Today, 11(23-24), 1046-1053.
  :doi:`10.1016/j.drudis.2006.10.005`
- Bajusz, D., Racz, A. & Heberger, K. (2015). "Why Is Tanimoto Index an
  Appropriate Choice for Fingerprint-Based Similarity Calculations?"
  J. Cheminform., 7, 20. :doi:`10.1186/s13321-015-0069-3`
- Johnson, M. A. & Maggiora, G. M. (1990). "Concepts and Applications of
  Molecular Similarity." Wiley.
