"""Exact inner-product search over L2-normalised embeddings (= cosine similarity).

FAISS IndexFlatIP when faiss is installed (Kaggle: faiss-cpu); otherwise an exact numpy fallback that returns the
same neighbours (used by the local tests). The index only ever holds TRAIN-split studies (CLAUDE.md §3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - exercised only where faiss is missing
    faiss = None


class FlatIP:
    def __init__(self, emb: np.ndarray):
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("embeddings must be L2-normalised")
        self.n, self.d = emb.shape
        self.emb = emb
        self.backend = "faiss" if faiss is not None else "numpy"
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.d)
            self.index.add(emb)

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(np.atleast_2d(q), dtype=np.float32)
        k = min(k, self.n)
        if self.backend == "faiss":
            return self.index.search(q, k)
        s = q @ self.emb.T
        idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(s, idx, axis=1), idx

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "faiss":
            faiss.write_index(self.index, str(path))
        else:
            np.save(path.with_suffix(".npy"), self.emb)
