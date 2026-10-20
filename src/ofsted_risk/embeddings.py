"""Turning improvement text into fixed-length vectors.

The sentence encoder is a frozen, pre-trained model, so encoding every row
once before any train-test split leaks no label information. Embeddings are
cached on disk because encoding about 8,300 texts on a 2-vCPU machine takes
minutes. A small TF-IDF and SVD encoder stands in for the sentence model in
tests, where the pre-trained model cannot be downloaded.
"""

import hashlib
from pathlib import Path
from typing import List, Sequence

import numpy as np


class SentenceEncoder:
    """Encode texts with a pre-trained Sentence-BERT model."""

    def __init__(self, model_name: str) -> None:
        """Remember the model name; the model loads on first use.

        Args:
            model_name: Hugging Face identifier of the sentence model.
        """
        self.model_name = model_name
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into normalised embeddings.

        Args:
            texts: Texts to encode; empty strings stand for missing text.

        Returns:
            An array with one row per text.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return np.asarray(
            self._model.encode(list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        )


class LsaEncoder:
    """Encode texts with TF-IDF reduced by SVD, a stand-in for tests."""

    def __init__(self, dimensions: int = 32) -> None:
        """Set the output size.

        Args:
            dimensions: Number of SVD components to keep.
        """
        self.dimensions = dimensions
        self.model_name = f"lsa-{dimensions}"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Fit TF-IDF and SVD on the texts and return the reduced vectors.

        Args:
            texts: Texts to encode.

        Returns:
            An array with one row per text.
        """
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        matrix = TfidfVectorizer(min_df=1).fit_transform(list(texts))
        components = min(self.dimensions, matrix.shape[1] - 1)
        return TruncatedSVD(n_components=components, random_state=0).fit_transform(matrix)


def cached_encode(encoder, texts: List[str], cache_dir: Path) -> np.ndarray:
    """Encode texts, reusing a cached result for exactly the same inputs.

    Args:
        encoder: Object with ``model_name`` and ``encode``.
        texts: Texts in the order the rows will be used.
        cache_dir: Folder for cached arrays.

    Returns:
        The embedding array, from cache when available.
    """
    digest = hashlib.sha256(("\n".join(texts) + encoder.model_name).encode("utf-8")).hexdigest()[:16]
    path = cache_dir / f"embeddings_{encoder.model_name.replace('/', '_')}_{digest}.npy"
    if path.exists():
        return np.load(path)
    vectors = encoder.encode(texts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors
