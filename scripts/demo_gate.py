"""Show the gate working, with no API key and no network.

Runs a hand-written draft -- the kind of fluent, mostly-right, partly-invented
answer a chatbot actually produces -- through the real verifier against real
clauses retrieved from the real corpus.

Use this as the fallback demo if the API key or the venue wifi dies. It proves
the part that is ours; the drafting step is the part anyone can buy.

    python scripts/demo_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pramaan.config import ENTAILMENT_THRESHOLD  # noqa: E402
from pramaan.retriever import Retriever  # noqa: E402
from pramaan.verifier import make_verifier, verify  # noqa: E402

QUESTION = "I own farmland. What do I get under PM-KISAN and who is left out?"

# Deliberately realistic: four true sentences drawn from the guidelines and
# three confident inventions of exactly the kind that cause real-world harm.
DRAFT = (
    "Under the scheme an income support of Rs. 6000 per year is provided to all "
    "land holding farmer families. "
    "The amount is paid in three equal instalments. "
    "Institutional land holders are not eligible for the benefit. "
    "Persons who paid income tax in the last assessment year are excluded. "
    "You will also receive a free tractor once your application is approved. "
    "Farmers below 25 years of age get double the annual amount. "
    "The money is credited every month on the first working day."
)


def main() -> int:
    print("=" * 74)
    print("PRAMAAN - verification gate demo (no API key, no network)")
    print("=" * 74)
    print(f"\nQuestion: {QUESTION}\n")

    retriever = Retriever()
    hits = retriever.search(QUESTION)
    clauses = [c for c, _ in hits]
    print(f"Retrieved {len(clauses)} clauses via {retriever.backend}:")
    for c, s in hits:
        print(f"  {s:.3f}  {c.scheme} - {c.doc_title}, p.{c.page}")

    print("\nDraft handed to the gate (7 sentences, 3 of them invented):\n")
    print("  " + DRAFT.replace(". ", ".\n  "))

    verifier = make_verifier()
    print(f"\nVerifying with backend '{verifier.name}' at threshold {ENTAILMENT_THRESHOLD}...\n")
    res = verify(DRAFT, clauses, verifier=verifier)

    print("-" * 74)
    for i, v in enumerate(res.verdicts, 1):
        mark = "KEPT   " if v.supported else "BLOCKED"
        print(f"{mark} {v.score:.3f}  {i}. {v.sentence}")
        if v.supported:
            print(f"                  -> {v.citation}")
    print("-" * 74)

    print(f"\nDrafted {len(res.verdicts)} | kept {len(res.kept)} | "
          f"blocked {len(res.stripped)} | catch-rate {res.catch_rate:.0%}")
    print(f"Abstained: {res.abstained}")
    print("\nWhat the user would have seen:\n")
    print("  " + (res.answer or "(abstained)"))
    print("\nWhat the gate deleted before they saw it:\n")
    for v in res.stripped:
        print(f"  x {v.sentence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
