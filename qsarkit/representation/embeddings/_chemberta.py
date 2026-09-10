"""ChemBERTa: a RoBERTa masked-language model pretrained on SMILES."""

from __future__ import annotations

from typing import Optional

from qsarkit.representation.embeddings._hf_base import _BaseHFEncoderTransformer

__all__ = ["ChemBERTaTransformer"]


class ChemBERTaTransformer(_BaseHFEncoderTransformer):
    """ChemBERTa molecular embeddings from a pretrained SMILES RoBERTa model.

    ChemBERTa is a RoBERTa-architecture masked-language model pretrained on
    millions of SMILES strings from PubChem/ZINC, following the BERT
    pretrain-then-finetune recipe applied to chemistry. This transformer
    loads a pretrained checkpoint (by default
    ``seyonec/ChemBERTa-zinc-base-v1`` from the Hugging Face Hub), encodes
    each molecule's canonical SMILES, and pools the final hidden states
    into a fixed-width embedding usable directly as a QSAR feature matrix.

    Parameters
    ----------
    model_name : str, default "seyonec/ChemBERTa-zinc-base-v1"
        Hugging Face Hub identifier or local path of a ChemBERTa-family
        checkpoint. Requires network access on first use (or a local
        Hugging Face cache / offline path) to download model weights.
    pooling : {"mean", "cls"}, default "mean"
        Token-pooling strategy; see
        :class:`~qsarkit.representation.embeddings._hf_base._BaseHFEncoderTransformer`.
    max_length : int, default 128
        Maximum SMILES token length; longer SMILES are truncated.
    batch_size : int, default 32
        Number of molecules encoded per forward pass.
    device : str, optional
        Torch device. ``None`` (default) uses CUDA when available, else CPU.

    Notes
    -----
    Loading requires the ``nlp`` extra (``pip install qsarkit[nlp]``, which
    installs ``torch`` and ``transformers``) and, for the default
    checkpoint, either network access to the Hugging Face Hub or a
    previously populated local HF cache / offline ``model_name`` path. No
    fine-tuning is performed here: embeddings come directly from the
    pretrained encoder (feature extraction / "frozen ChemBERTa" mode), the
    setting under which the original paper reports its representation
    benchmarks.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.embeddings import ChemBERTaTransformer
    >>> cb = ChemBERTaTransformer()  # doctest: +SKIP
    >>> cb.fit([])  # doctest: +SKIP
    >>> cb.transform([Chem.MolFromSmiles("CCO")]).shape  # doctest: +SKIP
    (1, 768)

    References
    ----------
    - Chithrananda, S., Grand, G. & Ramsundar, B. (2020). "ChemBERTa:
      Large-Scale Self-Supervised Pretraining for Molecular Property
      Prediction." arXiv:2010.09885. https://arxiv.org/abs/2010.09885
    - Liu, Y. et al. (2019). "RoBERTa: A Robustly Optimized BERT
      Pretraining Approach." arXiv:1907.11692.
      https://arxiv.org/abs/1907.11692
    - Pretrained checkpoint:
      https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
    - Hugging Face ``transformers`` documentation:
      https://huggingface.co/docs/transformers
    """

    _feature_prefix = "chemberta"

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        pooling: str = "mean",
        max_length: int = 128,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pooling=pooling,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
