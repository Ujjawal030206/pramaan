"""Compare sentence encoders on this corpus. Retrieval is the ceiling on the
whole system -- if the governing clause is not retrieved, the gate correctly
refuses to answer, and we look useless rather than wrong.

Each probe names a question and a distinctive substring that MUST appear in one
of the retrieved clauses. Reports recall@k and MRR.

    python scripts/bench_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from pramaan.ingest import build_clause_set  # noqa: E402

# (question, substring that identifies the governing clause)
PROBES = [
    ("I own farmland. What do I get under PM-KISAN?", "6000"),
    ("How much money do I get under PM-KISAN?", "6000"),
    ("How is the PM-KISAN amount paid during the year?", "three installments"),
    ("I paid income tax last year. Can I get PM-KISAN?", "income tax"),
    ("Are government employees eligible for PM-KISAN?", "serving or retired officers"),
    ("Who counts as a family under PM-KISAN?", "husband, wife and minor children"),
    ("What is the age range to join Atal Pension Yojana?", "18"),
    ("What is the entry age for Kisan Maan Dhan Yojana?", "18"),
    ("How much does PMJJBY cost per year?", "436"),
    ("What does PMSBY pay for accidental death?", "2 lakh"),
]

CANDIDATES = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "intfloat/multilingual-e5-small",
]

# e5 models are trained with asymmetric prefixes and degrade badly without them.
PREFIX = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
}


def _rank_of(order, clauses, needle):
    for pos, idx in enumerate(order, 1):
        if needle.lower() in clauses[idx].text.lower():
            return pos
    return None


def _score(orders, clauses, k):
    hits, rr, misses = 0, [], []
    for (question, needle), order in zip(PROBES, orders):
        rank = _rank_of(order, clauses, needle)
        if rank is None:
            rr.append(0.0)
            misses.append((question, "not found"))
            continue
        rr.append(1.0 / rank)
        if rank <= k:
            hits += 1
        else:
            misses.append((question, f"rank {rank}"))
    return hits / len(PROBES), float(np.mean(rr)), misses


def evaluate(model_name: str, clauses, k: int = 6):
    """Returns (dense_result, hybrid_result), each (recall@k, MRR, misses)."""
    from sentence_transformers import SentenceTransformer

    from pramaan.bm25 import BM25, rrf

    qp, pp = PREFIX.get(model_name, ("", ""))
    model = SentenceTransformer(model_name)
    corpus = model.encode(
        [pp + c.text for c in clauses], batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True,
    )
    bm25 = BM25([c.text for c in clauses])
    depth = 60

    dense_orders, hybrid_orders = [], []
    for question, _ in PROBES:
        qv = model.encode([qp + question], convert_to_numpy=True,
                          normalize_embeddings=True)[0]
        d_order = np.argsort(-(corpus @ qv)).tolist()
        dense_orders.append(d_order)

        lex = bm25.scores(question)
        l_order = [i for i in sorted(range(len(lex)), key=lambda j: -lex[j])[:depth]
                   if lex[i] > 0]
        fused = [i for i, _ in rrf([d_order[:depth], l_order])]
        hybrid_orders.append(fused)

    return _score(dense_orders, clauses, k), _score(hybrid_orders, clauses, k)


def main() -> int:
    print("Extracting clauses...\n")
    clauses = build_clause_set()
    print(f"\n{len(clauses)} clauses\n")
    print(f"{'model':<56} {'dense R@6':>10} {'hybrid R@6':>11} {'hybrid MRR':>11}")
    print("-" * 90)
    results = []
    for name in CANDIDATES:
        try:
            dense, hybrid = evaluate(name, clauses)
        except Exception as exc:
            print(f"{name:<56}  FAILED {type(exc).__name__}: {exc}")
            continue
        results.append((hybrid[0], hybrid[1], name, hybrid[2]))
        print(f"{name:<56} {dense[0]:>9.0%} {hybrid[0]:>10.0%} {hybrid[1]:>11.3f}")

    if results:
        results.sort(reverse=True)
        best = results[0]
        print(f"\nBest: {best[2]}  hybrid recall@6 {best[0]:.0%}, MRR {best[1]:.3f}")
        if best[3]:
            print("Still missed:")
            for q, why in best[3]:
                print(f"  - {why:<14} {q}")
        else:
            print("No misses on the probe set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
