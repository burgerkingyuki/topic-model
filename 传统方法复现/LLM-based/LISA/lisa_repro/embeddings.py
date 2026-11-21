from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False))

