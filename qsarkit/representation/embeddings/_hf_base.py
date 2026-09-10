"""Shared batching/pooling plumbing for Hugging Face encoder embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from qsarkit.base import FittableMoleculeTransformer, ensure_mol_list, require

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__: List[str] = []


class _BaseHFEncoderTransformer(FittableMoleculeTransformer):
    """Shared batching/pooling plumbing for Hugging Face encoder embeddings.

    Concrete subclasses (:class:`~qsarkit.representation.embeddings.ChemBERTaTransformer`,
    :class:`~qsarkit.representation.embeddings.MolecularTransformer`) load a
    Hugging Face ``AutoTokenizer``/``AutoModel`` pair lazily in :meth:`fit`,
    convert molecules to one or more SMILES-derived text strings via
    :meth:`_mol_to_texts` (overridden to add canonicalization/augmentation),
    encode them in batches under ``torch.no_grad()``, and pool the
    resulting token embeddings into one fixed-width vector per molecule
    (averaging across the molecule's text variants when there are several).

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier or local path, forwarded to
        ``AutoTokenizer.from_pretrained`` / ``AutoModel.from_pretrained``.
    pooling : {"mean", "cls"}, default "mean"
        Token-pooling strategy. ``"mean"`` averages token embeddings
        weighted by the attention mask (robust to padding); ``"cls"`` uses
        the first token's embedding, the BERT-family convention for a
        sequence-level representation.
    max_length : int, default 128
        Maximum token sequence length; longer SMILES are truncated.
    batch_size : int, default 32
        Number of text variants encoded per forward pass.
    device : str, optional
        Torch device (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ...). ``None``
        (default) uses CUDA when available, else CPU.

    References
    ----------
    - Wolf, T. et al. (2020). "Transformers: State-of-the-Art Natural
      Language Processing." EMNLP 2020: System Demonstrations, 38-45.
      https://doi.org/10.18653/v1/2020.emnlp-demos.6
    - Hugging Face ``transformers`` documentation:
      https://huggingface.co/docs/transformers
    """

    _feature_prefix: str = "hf_embed"

    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        max_length: int = 128,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    def fit(
        self, mols: Iterable[Any], y: Optional[Iterable[Any]] = None
    ) -> "_BaseHFEncoderTransformer":
        """Load the tokenizer and encoder model.

        Parameters
        ----------
        mols : Iterable[rdkit.Chem.Mol]
            Ignored beyond validation: loading a pretrained encoder does
            not depend on the molecules it will later embed.
        y : ignored

        Returns
        -------
        _BaseHFEncoderTransformer
            self.

        Raises
        ------
        ValueError
            If ``pooling`` is not ``"mean"``/``"cls"``.
        """
        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {self.pooling!r}.")
        ensure_mol_list(mols)

        transformers_mod = require("transformers")
        torch = require("torch")
        self._tokenizer = transformers_mod.AutoTokenizer.from_pretrained(self.model_name)
        self._model = transformers_mod.AutoModel.from_pretrained(self.model_name)
        self._device = torch.device(
            self.device if self.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model.to(self._device)
        self._model.eval()
        self._is_fitted = True
        return self

    def _mol_to_texts(self, mol: "Mol") -> List[str]:
        """Return the text variant(s) representing ``mol``. Default: canonical SMILES."""
        from rdkit import Chem

        return [Chem.MolToSmiles(mol)]

    def _encode_batch(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        """Tokenize, forward-pass and pool one batch of text variants."""
        torch = require("torch")
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=int(self.max_length),
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self._model(**encoded)
        hidden = output.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return np.asarray(pooled.detach().cpu().numpy(), dtype=np.float64)

    def _transform(self, mols: List[Optional["Mol"]]) -> npt.NDArray[np.float64]:
        self._check_is_fitted()
        hidden_size = int(self._model.config.hidden_size)
        out = np.zeros((len(mols), hidden_size), dtype=np.float64)

        pending: List[Tuple[int, str]] = []
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            pending.extend((i, text) for text in self._mol_to_texts(mol))
        if not pending:
            return out

        sums = np.zeros((len(mols), hidden_size), dtype=np.float64)
        counts = np.zeros(len(mols), dtype=np.float64)
        batch_size = int(self.batch_size)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            idxs = [idx for idx, _ in batch]
            texts = [text for _, text in batch]
            pooled = self._encode_batch(texts)
            for row, idx in zip(pooled, idxs):
                sums[idx] += row
                counts[idx] += 1.0

        has_output = counts > 0
        out[has_output] = sums[has_output] / counts[has_output, None]
        return out

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> npt.NDArray[np.object_]:
        """Return ``hidden_size`` embedding-dimension names.

        Parameters
        ----------
        input_features : sequence of str, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            Array of ``str`` names, one per output column.
        """
        self._check_is_fitted()
        hidden_size = int(self._model.config.hidden_size)
        return np.asarray(
            [f"{self._feature_prefix}_{i}" for i in range(hidden_size)], dtype=object
        )
