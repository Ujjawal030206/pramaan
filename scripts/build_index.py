"""Extract clauses from the downloaded PDFs and build the retrieval index."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pramaan.ingest import build_clause_set  # noqa: E402
from pramaan.retriever import build_index  # noqa: E402


def main() -> int:
    print("Extracting clauses from official documents\n")
    clauses = build_clause_set()
    if not clauses:
        print("\nNo clauses extracted. Run scripts/fetch_corpus.py first.")
        return 1

    schemes = sorted({c.scheme for c in clauses})
    print(f"\n{len(clauses)} clauses across {len(schemes)} schemes:")
    for s in schemes:
        print(f"  - {s}")

    print("\nEmbedding and indexing (first run downloads the encoder)...")
    build_index(clauses)
    print("Index written to index/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
