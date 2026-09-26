"""Trustworthiness of a chemical-space projection.

The figure a projection produces looks the same whether or not its
neighbourhoods survived the embedding, so this score is what distinguishes the
two. The tests pin the contract rather than particular values: a projection
that preserves structure must score above one that destroys it, the scale
argument must behave like the validators' ``scoring``, and the guards must fire
before sklearn produces something misleading.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsarkit.chemspace import ChemicalSpaceAnalyzer, projection_trustworthiness


@pytest.fixture(scope="module")
def binary_data():
    """Clustered binary vectors, so there is real neighbourhood structure."""
    rng = np.random.default_rng(0)
    centres = (rng.random((4, 48)) > 0.6).astype(float)
    rows = []
    for centre in centres:
        for _ in range(20):
            noise = rng.random(48) < 0.05
            rows.append(np.abs(centre - noise))
    return np.asarray(rows, dtype=float)


class TestScoreRange:
    def test_a_score_is_a_probability_like_number(self, binary_data):
        embedding = ChemicalSpaceAnalyzer(
            method="pca", metric="euclidean", random_state=0
        ).fit_transform(binary_data)
        score = projection_trustworthiness(
            binary_data, embedding, n_neighbors=5, metric="euclidean"
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_a_structure_preserving_projection_beats_a_random_one(self, binary_data):
        """The score has to be able to tell the two apart, or it is useless."""
        real = ChemicalSpaceAnalyzer(
            method="pca", metric="euclidean", random_state=0
        ).fit_transform(binary_data)
        rng = np.random.default_rng(1)
        scrambled = rng.random(real.shape)

        good = projection_trustworthiness(
            binary_data, real, n_neighbors=10, metric="euclidean"
        )
        bad = projection_trustworthiness(
            binary_data, scrambled, n_neighbors=10, metric="euclidean"
        )
        assert good > bad
        assert bad < 0.7


class TestSingleVersusMultiple:
    """Same convention as the validators: one asks, one answers."""

    def test_one_size_gives_a_float(self, binary_data):
        embedding = ChemicalSpaceAnalyzer(
            method="pca", metric="euclidean", random_state=0
        ).fit_transform(binary_data)
        assert isinstance(
            projection_trustworthiness(
                binary_data, embedding, n_neighbors=5, metric="euclidean"
            ),
            float,
        )

    def test_several_sizes_give_an_array_in_order(self, binary_data):
        embedding = ChemicalSpaceAnalyzer(
            method="pca", metric="euclidean", random_state=0
        ).fit_transform(binary_data)
        sizes = [5, 10, 20]
        together = projection_trustworthiness(
            binary_data, embedding, n_neighbors=sizes, metric="euclidean"
        )
        assert isinstance(together, np.ndarray)
        assert together.shape == (3,)

        separately = [
            projection_trustworthiness(
                binary_data, embedding, n_neighbors=k, metric="euclidean"
            )
            for k in sizes
        ]
        assert np.allclose(together, separately)


class TestSubsampling:
    def test_a_subsample_is_reproducible(self, binary_data):
        embedding = ChemicalSpaceAnalyzer(
            method="pca", metric="euclidean", random_state=0
        ).fit_transform(binary_data)
        kwargs = dict(n_neighbors=5, metric="euclidean", subsample=40, random_state=0)
        first = projection_trustworthiness(binary_data, embedding, **kwargs)
        second = projection_trustworthiness(binary_data, embedding, **kwargs)
        assert first == second

    def test_a_subsample_larger_than_the_data_is_a_no_op(self, binary_data):
        embedding = ChemicalSpaceAnalyzer(
            method="pca", metric="euclidean", random_state=0
        ).fit_transform(binary_data)
        full = projection_trustworthiness(
            binary_data, embedding, n_neighbors=5, metric="euclidean"
        )
        oversized = projection_trustworthiness(
            binary_data,
            embedding,
            n_neighbors=5,
            metric="euclidean",
            subsample=len(binary_data) * 10,
        )
        assert full == oversized


class TestGuards:
    def test_mismatched_sample_counts_are_refused(self, binary_data):
        with pytest.raises(ValueError, match="same compounds"):
            projection_trustworthiness(binary_data, np.zeros((5, 2)))

    def test_a_one_dimensional_input_is_refused(self, binary_data):
        with pytest.raises(ValueError, match="2-dimensional"):
            projection_trustworthiness(binary_data[:, 0], np.zeros((len(binary_data), 2)))

    def test_too_many_neighbours_is_refused_with_a_usable_message(self, binary_data):
        embedding = np.zeros((len(binary_data), 2))
        with pytest.raises(ValueError, match="too large for"):
            projection_trustworthiness(
                binary_data, embedding, n_neighbors=len(binary_data)
            )

    def test_an_empty_size_list_is_refused(self, binary_data):
        embedding = np.zeros((len(binary_data), 2))
        with pytest.raises(ValueError, match="at least one neighbourhood size"):
            projection_trustworthiness(binary_data, embedding, n_neighbors=[])

    def test_a_tiny_subsample_is_refused(self, binary_data):
        embedding = np.zeros((len(binary_data), 2))
        with pytest.raises(ValueError, match="subsample must be at least 3"):
            projection_trustworthiness(
                binary_data, embedding, n_neighbors=2, subsample=2
            )


class TestAnalyzerMethod:
    def test_it_scores_its_own_projection(self):
        from rdkit import Chem

        smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "c1ccccc1O", "CCCl", "CCBr", "CCI"]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        analyzer = ChemicalSpaceAnalyzer(method="pca", random_state=0).fit(mols)
        score = analyzer.trustworthiness(mols, n_neighbors=3)
        assert 0.0 <= score <= 1.0

    def test_it_refuses_before_fit(self):
        from rdkit import Chem

        mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN", "CCC", "CCCl")]
        with pytest.raises(AttributeError, match="Call fit"):
            ChemicalSpaceAnalyzer().trustworthiness(mols)

    def test_it_uses_the_analyzer_metric(self):
        """A jaccard projection must not be scored with euclidean distances."""
        rng = np.random.default_rng(0)
        X = (rng.random((40, 24)) > 0.6).astype(float)
        analyzer = ChemicalSpaceAnalyzer(
            method="pca", metric="jaccard", random_state=0
        ).fit(X)
        via_method = analyzer.trustworthiness(X, n_neighbors=5)
        via_function = projection_trustworthiness(
            X, analyzer.embedding_, n_neighbors=5, metric="jaccard"
        )
        assert via_method == via_function


class TestFitDoesNotMutateConstructorArguments:
    """``fit`` must leave the constructor's arguments alone.

    The projection branches consumed entries from ``self.kwargs`` with
    ``pop()``. That mutates a constructor argument, so a user's
    ``perplexity`` was honoured by the first ``fit()`` and silently forgotten
    by the second: two identical calls on one object returned different
    embeddings.
    """

    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng(0)
        return (rng.random((60, 32)) > 0.7).astype(float)

    @pytest.mark.parametrize(
        "method, extra",
        [
            ("tsne", {"perplexity": 5.0}),
            ("mds", {"n_init": 2}),
            ("pca", {}),
        ],
    )
    def test_kwargs_survive_a_fit(self, data, method, extra):
        analyzer = ChemicalSpaceAnalyzer(
            method=method, metric="euclidean", random_state=0, **extra
        )
        before = dict(analyzer.kwargs)
        analyzer.fit(data)
        assert analyzer.kwargs == before

    @pytest.mark.parametrize(
        "method, extra",
        [
            ("tsne", {"perplexity": 5.0}),
            ("mds", {"n_init": 2}),
            ("pca", {}),
        ],
    )
    def test_two_identical_fits_agree(self, data, method, extra):
        analyzer = ChemicalSpaceAnalyzer(
            method=method, metric="euclidean", random_state=0, **extra
        )
        first = analyzer.fit_transform(data)
        second = analyzer.fit_transform(data)
        assert np.allclose(first, second)
