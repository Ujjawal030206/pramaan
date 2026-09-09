"""Dense retrieval over the clause set.

FAISS is used when it is installed; otherwise we fall back to an exact numpy
dot product. At corpus sizes like ours (thousands of clauses, not millions) the
fallback is exact and fast, so a missing FAISS wheel never blocks a demo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bm25 import BM25, rrf
from .config import (
    BM25_WEIGHT,
    EMBED_MODEL,
    HYBRID_RETRIEVAL,
    INDEX_DIR,
    RETRIEVAL_MODE,
    RRF_DEPTH_MULTIPLIER,
    TOP_K,
)
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
        if not CLAUSES_FILE.exists():
            raise FileNotFoundError(
                "No index found. Run:  python scripts/build_index.py"
            )
        self.clauses = load_clauses(CLAUSES_FILE)
        self.mode = RETRIEVAL_MODE

        if self.mode == "lexical":
            # No vectors, no encoder, no torch.
            self.vectors = None
            self._bm25 = BM25([c.text for c in self.clauses])
            self._faiss = None
            return

        if not VECTORS_FILE.exists():
            raise FileNotFoundError(
                "No vectors found. Run scripts/build_index.py, or set "
                "PRAMAAN_RETRIEVAL_MODE=lexical to run without embeddings."
            )
        # The vectors were produced by one specific encoder. Loading them with a
        # different one yields plausible-looking numbers and silently wrong
        # retrieval, which for this project is the worst possible failure.
        if META_FILE.exists():
            built_with = json.loads(META_FILE.read_text(encoding="utf-8")).get("model")
            if built_with and built_with != EMBED_MODEL:
                raise RuntimeError(
                    f"Index was built with {built_with!r} but PRAMAAN_EMBED_MODEL is "
                    f"{EMBED_MODEL!r}. Rebuild with scripts/build_index.py, or unset "
                    "the override. Mixing encoders silently corrupts retrieval."
                )
        self.vectors = np.load(VECTORS_FILE)
        # Cheap enough to rebuild at load (a few hundred clauses) that it is not
        # worth another artefact on disk to keep in sync with the vectors.
        self._bm25 = BM25([c.text for c in self.clauses])
        self._faiss = None
        try:
            import faiss  # noqa: F401

            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)
        except Exception:
            self._faiss = None  # numpy fallback

    def rank_against(self, text: str, clauses: list[Clause]) -> list[Clause]:
        """Order `clauses` by embedding similarity to `text`, best first.

        Used to decide which premises are worth handing to the cross-encoder.
        Falls back to the given order when embeddings are unavailable.
        """
        if self.vectors is None or not clauses:
            return list(clauses)
        if not hasattr(self, "_row"):
            self._row = {c.clause_id: i for i, c in enumerate(self.clauses)}
        rows = [self._row.get(c.clause_id) for c in clauses]
        if any(i is None for i in rows):
            return list(clauses)
        q = embed([text])[0]
        sims = self.vectors[rows] @ q
        order = np.argsort(-sims)
        return [clauses[i] for i in order]

    @property
    def backend(self) -> str:
        if self.mode == "lexical":
            return "bm25-only"
        return ("faiss" if self._faiss is not None else "numpy") + f"+{self.mode}"

    def dense_ranking(self, query: str, depth: int) -> tuple[list[int], np.ndarray]:
        q = embed([query])
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, min(depth, len(self.clauses)))
            order = [i for i in idx[0].tolist() if i >= 0]
            sims = np.zeros(len(self.clauses), dtype="float32")
            for i, s in zip(idx[0].tolist(), scores[0].tolist()):
                if i >= 0:
                    sims[i] = s
        else:
            sims = self.vectors @ q[0]
            order = np.argsort(-sims)[:depth].tolist()
        return order, sims

    def search(self, query: str, k: int = TOP_K) -> list[tuple[Clause, float]]:
        """Hybrid dense + BM25, fused with RRF.

        Returned score is the dense cosine similarity, kept because it is the
        interpretable one to show a user; ordering comes from the fusion.
        """
        depth = max(k * RRF_DEPTH_MULTIPLIER, 30)

        if self.mode == "lexical":
            lex = self._bm25.scores(query)
            top = sorted(range(len(lex)), key=lambda j: -lex[j])[:k]
            hi = max(lex) or 1.0
            return [(self.clauses[i], lex[i] / hi) for i in top if lex[i] > 0]

        dense_order, sims = self.dense_ranking(query, depth)

        if not HYBRID_RETRIEVAL:
            return [(self.clauses[i], float(sims[i])) for i in dense_order[:k]]

        lex = self._bm25.scores(query)
        lex_order = [
            i for i in sorted(range(len(lex)), key=lambda j: -lex[j])[:depth]
            if lex[i] > 0
        ]
        fused = rrf([dense_order, lex_order], weights=[1.0, BM25_WEIGHT])
        return [(self.clauses[i], float(sims[i])) for i, _ in fused[:k]]
