"""Minimal BM25, because dense retrieval alone keeps missing exact terms.

Benchmarked on this corpus, a pure dense retriever ranked the PM-KISAN
income-tax exclusion 35th for the question "I paid income tax last year. Can I
get PM-KISAN?" -- even though the phrase "income tax" appears verbatim in the
governing clause. Eligibility questions turn on exactly this kind of literal
token: amounts (6000, 436), ages (18, 40), and named categories. Lexical search
is very good at those and embeddings are not, so we run both.

Implemented here rather than pulled in as a dependency: it is forty lines, and
a deployment that has to install one fewer package is one that fails to boot
less often.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    # Digits are kept and never split: "6000" and "436" carry as much meaning
    # here as any word, and dropping them was part of why dense-only failed.
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in corpus]
        self.n = len(self.docs)
        self.lengths = [len(d) for d in self.docs]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.docs]

        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        # Robertson/Sparck-Jones idf with the +1 guard, so a term appearing in
        # every document scores ~0 instead of going negative.
        self.idf = {
            term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = [0.0] * self.n
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


def rrf(rank_lists: list[list[int]], k: int = 60, weights: list[float] | None = None):
    """Reciprocal Rank Fusion.

    Combines rankings without needing their scores to be on comparable scales,
    which matters because BM25 is unbounded and cosine similarity is not.
    """
    weights = weights or [1.0] * len(rank_lists)
    fused: dict[int, float] = {}
    for ranking, w in zip(rank_lists, weights):
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])
