"""Mol2vec: unsupervised molecular embeddings from Morgan-identifier sentences."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

from qsarkit.base import FittableMoleculeTransformer, ensure_mol_list, require

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["Mol2VecTransformer"]

#: Default placeholder substituted for substructure identifiers seen fewer
#: than ``min_count`` times during training, so the Word2Vec model learns a
#: usable vector for substructures it has never seen at inference time.
_UNSEEN_TOKEN = "UNK"


def mol_to_sentence(mol: "Mol", radius: int) -> List[str]:
    """Convert a molecule into its Mol2vec "sentence" of Morgan identifiers.

    Reproduces ``mol2alt_sentence`` from the reference Mol2vec
    implementation: a Morgan fingerprint is computed with bit-info tracking
    (mapping every circular-substructure identifier to the atoms/radii it
    was generated from); the identifiers are then read back out ordered by
    atom index and, within an atom, by increasing radius from 0 to
    ``radius``. The resulting list of identifiers is the "sentence" fed to
    Word2Vec, with each distinct circular substructure playing the role of
    a word and each molecule the role of a sentence.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule to decompose.
    radius : int
        Maximum Morgan radius; identifiers for every radius in
        ``0, ..., radius`` are included.

    Returns
    -------
    list of str
        The molecule's sentence, one token per (atom, radius) substructure.

    References
    ----------
    - Jaeger, S., Fulle, S. & Turk, S. (2018). "Mol2vec: Unsupervised
      Machine Learning Approach with Chemical Intuition." J. Chem. Inf.
      Model., 58(1), 27-35. https://doi.org/10.1021/acs.jcim.7b00616
    - Reference implementation: https://github.com/samoturk/mol2vec
    """
    from rdkit.Chem import AllChem

    radii = list(range(int(radius) + 1))
    bit_info: dict = {}
    AllChem.GetMorganFingerprint(mol, int(radius), bitInfo=bit_info)

    # identifier_by_atom_radius[atom_idx][r] = identifier, or None if that
    # atom has no substructure of radius r (e.g. terminal atoms at r > 0).
    identifier_by_atom_radius: dict = {
        atom.GetIdx(): {r: None for r in radii} for atom in mol.GetAtoms()
    }
    for identifier, envs in bit_info.items():
        for atom_idx, env_radius in envs:
            identifier_by_atom_radius[atom_idx][env_radius] = identifier

    sentence: List[str] = []
    for atom_idx in sorted(identifier_by_atom_radius):
        for r in radii:
            token = identifier_by_atom_radius[atom_idx][r]
            if token is not None:
                sentence.append(str(token))
    return sentence


def _insert_unseen_token(
    sentences: Sequence[Sequence[str]], min_count: int, unseen_token: str
) -> List[List[str]]:
    """Replace identifiers occurring fewer than ``min_count`` times with a shared token.

    This is the standard Mol2vec trick for handling out-of-vocabulary
    substructures at inference time: rather than letting Word2Vec's own
    ``min_count`` filter silently drop rare words (which would leave no
    vector at all for them), rare identifiers across the *training*
    corpus are collapsed into one shared placeholder before training, so
    Word2Vec learns a genuine (averaged) embedding for it. At inference,
    any never-before-seen identifier is mapped to that same placeholder.
    """
    counts = Counter(token for sentence in sentences for token in sentence)
    return [
        [token if counts[token] >= min_count else unseen_token for token in sentence]
        for sentence in sentences
    ]


class Mol2VecTransformer(FittableMoleculeTransformer):
    """Mol2vec: unsupervised molecular embeddings from Morgan-identifier sentences.

    Mol2vec treats a molecule as a "sentence" of circular substructure
    identifiers (Morgan/ECFP-style environments around each atom, one
    "word" per (atom, radius) pair) and trains a Word2Vec skip-gram model
    over a corpus of such sentences, exactly as in NLP. A trained model
    therefore embeds a *substructure* into a dense vector such that
    chemically related substructures (e.g. two different aromatic-ring
    contexts) end up nearby; a whole molecule's embedding is the
    (optionally weighted) sum or mean of its substructures' vectors.

    Parameters
    ----------
    radius : int, default 1
        Maximum Morgan radius used when building sentences; identifiers
        for every radius in ``0, ..., radius`` are included per atom.
    vector_size : int, default 100
        Dimensionality of the learned substructure/molecule embeddings.
    window : int, default 10
        Word2Vec context window (in tokens of the sentence).
    min_count : int, default 3
        Minimum corpus frequency for a substructure identifier to get its
        own vector; rarer identifiers are collapsed into ``unseen_token``
        (see :func:`_insert_unseen_token`) before training, when
        ``unseen_token`` is not ``None``.
    epochs : int, default 10
        Number of Word2Vec training epochs.
    sg : {0, 1}, default 1
        Word2Vec training algorithm: 1 = skip-gram (the algorithm used in
        the original paper), 0 = CBOW.
    agg : {"sum", "mean"}, default "sum"
        How per-substructure vectors are combined into the molecule
        embedding. The original paper sums them (``"sum"``, the "MOL2VEC"
        method); ``"mean"`` gives a length-normalized alternative.
    unseen_token : str or None, default "UNK"
        Placeholder substituted for identifiers below ``min_count`` during
        training, and used at inference time for any identifier absent
        from the trained vocabulary. ``None`` disables this: substructures
        not in the vocabulary simply do not contribute to the embedding.
    seed : int, default 42
        Word2Vec training seed, for reproducibility (combined with
        ``workers=1`` to make training deterministic, since gensim's
        multi-threaded training is only reproducible single-threaded).
    workers : int, default 1
        Number of Word2Vec worker threads. Kept at 1 by default because
        gensim's Word2Vec training is only bit-for-bit reproducible with a
        single worker.

    Attributes
    ----------
    is_fitted : bool
        Whether :meth:`fit` (or :meth:`from_pretrained`) has been called.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.mol2vec import Mol2VecTransformer
    >>> mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]]
    >>> m2v = Mol2VecTransformer(vector_size=8, min_count=1, epochs=5)
    >>> X = m2v.fit_transform(mols)
    >>> X.shape
    (4, 8)

    References
    ----------
    - Jaeger, S., Fulle, S. & Turk, S. (2018). "Mol2vec: Unsupervised
      Machine Learning Approach with Chemical Intuition." J. Chem. Inf.
      Model., 58(1), 27-35. https://doi.org/10.1021/acs.jcim.7b00616
    - Mikolov, T. et al. (2013). "Distributed Representations of Words and
      Phrases and Their Compositionality." NeurIPS 2013, 3111-3119.
      https://papers.nips.cc/paper/5021
    - Reference implementation: https://github.com/samoturk/mol2vec
    - gensim ``Word2Vec`` documentation:
      https://radimrehurek.com/gensim/models/word2vec.html
    """

    def __init__(
        self,
        radius: int = 1,
        vector_size: int = 100,
        window: int = 10,
        min_count: int = 3,
        epochs: int = 10,
        sg: int = 1,
        agg: str = "sum",
        unseen_token: Optional[str] = _UNSEEN_TOKEN,
        seed: int = 42,
        workers: int = 1,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.agg = agg
        self.unseen_token = unseen_token
        self.seed = seed
        self.workers = workers

    def fit(
        self, mols: Iterable[Any], y: Optional[Iterable[Any]] = None
    ) -> "Mol2VecTransformer":
        """Train the Word2Vec model on the Morgan-identifier sentences of ``mols``.

        Parameters
        ----------
        mols : Iterable[rdkit.Chem.Mol]
            Training molecules. ``None`` entries are ignored.
        y : ignored
            Present for scikit-learn API compatibility; Mol2vec training
            is unsupervised.

        Returns
        -------
        Mol2VecTransformer
            self.

        Raises
        ------
        ValueError
            If ``agg`` is not ``"sum"``/``"mean"``, or no non-``None``
            molecule is supplied.
        """
        if self.agg not in ("sum", "mean"):
            raise ValueError(f"agg must be 'sum' or 'mean', got {self.agg!r}.")

        word2vec = require("gensim.models.word2vec")
        mol_list = ensure_mol_list(mols)
        sentences = [
            mol_to_sentence(mol, self.radius) for mol in mol_list if mol is not None
        ]
        if not sentences:
            raise ValueError("Mol2VecTransformer.fit needs at least one valid molecule.")

        min_count = int(self.min_count)
        if self.unseen_token is not None:
            sentences = _insert_unseen_token(sentences, min_count, self.unseen_token)
            min_count = 1  # rare tokens are already folded into unseen_token

        self._model = word2vec.Word2Vec(
            sentences=sentences,
            vector_size=int(self.vector_size),
            window=int(self.window),
            min_count=min_count,
            sg=int(self.sg),
            epochs=int(self.epochs),
            seed=int(self.seed),
            workers=int(self.workers),
        )
        self._is_fitted = True
        return self

    def _sentence_vectors(self, sentence: Sequence[str]) -> List[npt.NDArray[np.float64]]:
        wv = self._model.wv
        vectors = []
        for token in sentence:
            if token in wv:
                vectors.append(np.asarray(wv[token], dtype=np.float64))
            elif self.unseen_token is not None and self.unseen_token in wv:
                vectors.append(np.asarray(wv[self.unseen_token], dtype=np.float64))
        return vectors

    def _transform(self, mols: List[Optional["Mol"]]) -> npt.NDArray[np.float64]:
        self._check_is_fitted()
        width = int(self._model.vector_size)
        out = np.zeros((len(mols), width), dtype=np.float64)
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            vectors = self._sentence_vectors(mol_to_sentence(mol, self.radius))
            if not vectors:
                continue
            stacked = np.stack(vectors, axis=0)
            out[i] = stacked.sum(axis=0) if self.agg == "sum" else stacked.mean(axis=0)
        return out

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> npt.NDArray[np.object_]:
        """Return ``vector_size`` embedding-dimension names.

        Parameters
        ----------
        input_features : sequence of str, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            Array of ``str`` names ``"mol2vec_0"``, ``"mol2vec_1"``, ...
        """
        self._check_is_fitted()
        width = int(self._model.vector_size)
        return np.asarray([f"mol2vec_{i}" for i in range(width)], dtype=object)

    def save(self, path: str) -> None:
        """Persist the trained Word2Vec model to disk.

        Parameters
        ----------
        path : str
            Destination path, forwarded to ``gensim.models.Word2Vec.save``.
        """
        self._check_is_fitted()
        self._model.save(path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "Mol2VecTransformer":
        """Load a previously trained (or third-party) Word2Vec checkpoint.

        Parameters
        ----------
        path : str
            Path to a Word2Vec model saved via ``gensim.models.Word2Vec.save``
            (e.g. the public Mol2vec checkpoint distributed by the paper's
            authors, ``model_300dim.pkl``:
            https://github.com/samoturk/mol2vec/tree/master/examples/models).
        **kwargs
            Extra constructor arguments (e.g. ``radius``, ``agg``) forwarded
            to ``Mol2VecTransformer.__init__``; use these to match the
            hyperparameters the checkpoint was trained with.

        Returns
        -------
        Mol2VecTransformer
            A fitted transformer wrapping the loaded model.

        References
        ----------
        - Jaeger, S., Fulle, S. & Turk, S. (2018). J. Chem. Inf. Model.,
          58(1), 27-35. https://doi.org/10.1021/acs.jcim.7b00616
        - Pretrained checkpoints: https://github.com/samoturk/mol2vec
        """
        word2vec = require("gensim.models.word2vec")
        instance = cls(**kwargs)
        instance._model = word2vec.Word2Vec.load(path)
        instance.vector_size = int(instance._model.vector_size)
        instance._is_fitted = True
        return instance
