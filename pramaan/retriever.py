"""Dense retrieval over the clause set.

FAISS is used when it is installed; otherwise we fall back to an exact numpy
dot product. At corpus sizes like ours (thousands of clauses, not millions) the
fallback is exact and fast, so a missing FAISS wheel never blocks a demo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import EMBED_MODEL, INDEX_DIR, TOP_K
from .ingest import Clause, load_clauses

CLAUSES_FILE = INDEX_DIR / "clauses.jsonl"
VECTORS_FILE = INDEX_DIR / "vectors.npy"
META_FILE = INDEX_DIR / "meta.json"

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(EMBED_MODEL)
    return _encoder


def embed(texts: list[str]) -> np.ndarray:
    vecs = get_encoder().encode(
        texts, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=len(texts) > 500,
    )
    return vecs.astype("float32")


def build_index(clauses: list[Clause]) -> None:
    from .ingest import save_clauses

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    save_clauses(clauses, CLAUSES_FILE)
    vecs = embed([c.text for c in clauses])
    np.save(VECTORS_FILE, vecs)
    META_FILE.write_text(
        json.dumps({"model": EMBED_MODEL, "count": len(clauses), "dim": int(vecs.shape[1])}),
        encoding="utf-8",
    )


class Retriever:
    def __init__(self) -> None:
        if not VECTORS_FILE.exists():
            raise FileNotFoundError(
                "No index found. Run:  python scripts/build_index.py"
            )
        self.clauses = load_clauses(CLAUSES_FILE)
        self.vectors = np.load(VECTORS_FILE)
        self._faiss = None
        try:
            import faiss  # noqa: F401

            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
        except Exception:
            self._faiss = None  # numpy fallback

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Clause, float]]:
        q = embed([query])
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            pairs = zip(idx[0].tolist(), scores[0].tolist())
        else:
            sims = (self.vectors @ q[0])
            top = np.argsort(-sims)[:k]
            pairs = ((int(i), float(sims[i])) for i in top)
        return [(self.clauses[i], float(s)) for i, s in pairs if i >= 0]
