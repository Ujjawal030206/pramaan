"""The real gate, against real clause text. No API key needed.

test_verifier.py pins the control flow with a stub. This file checks the thing
that actually matters: that a genuine NLI model, on genuine government clause
text, separates claims the document supports from claims it does not.

If this passes, the central claim of the project is true rather than asserted.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pramaan.config import ENTAILMENT_THRESHOLD  # noqa: E402
from pramaan.ingest import Clause  # noqa: E402
from pramaan.verifier import NLIVerifier, verify  # noqa: E402

# Verbatim from the PM-KISAN Revised Operational Guidelines / FAQ.
PMKISAN_BENEFIT = Clause(
    clause_id="pmkisan-og#benefit",
    text=(
        "Under the scheme an income support of Rs. 6000 per year in three equal "
        "instalments will be provided to all land holding farmer families."
    ),
    scheme="PM-KISAN", doc_id="pmkisan-og",
    doc_title="Revised PM-KISAN Operational Guidelines (English)",
    authority="Department of Agriculture & Farmers Welfare", page=2,
    url="https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf",
)

PMKISAN_EXCLUSION = Clause(
    clause_id="pmkisan-og#exclusion",
    text=(
        "The following categories of beneficiaries of higher economic status "
        "shall not be eligible for benefit under the scheme: All Institutional "
        "Land holders; and all persons who paid income tax in last assessment year."
    ),
    scheme="PM-KISAN", doc_id="pmkisan-og",
    doc_title="Revised PM-KISAN Operational Guidelines (English)",
    authority="Department of Agriculture & Farmers Welfare", page=3,
    url="https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf",
)

CLAUSES = [PMKISAN_BENEFIT, PMKISAN_EXCLUSION]

SUPPORTED = [
    "Land holding farmer families receive Rs. 6000 per year under the scheme.",
    "The money is paid in three equal instalments.",
    "People who paid income tax in the last assessment year are not eligible.",
]

FABRICATED = [
    "The scheme also provides a free tractor to every registered farmer.",
    "Applicants receive Rs. 50,000 per year under this scheme.",
    "The benefit is paid in twelve monthly instalments.",
    "Farmers under 25 years of age are given double the benefit.",
]

_verifier = None


def get_verifier():
    global _verifier
    if _verifier is None:
        _verifier = NLIVerifier()
    return _verifier


def test_supported_claims_score_above_threshold():
    v = get_verifier()
    for claim in SUPPORTED:
        best = max(v.entailment_scores([c.text for c in CLAUSES], claim))
        print(f"    {best:.3f}  SUPPORTED  {claim}")
        assert best >= ENTAILMENT_THRESHOLD, f"should have passed: {claim} ({best:.3f})"


def test_fabricated_claims_score_below_threshold():
    v = get_verifier()
    for claim in FABRICATED:
        best = max(v.entailment_scores([c.text for c in CLAUSES], claim))
        print(f"    {best:.3f}  FABRICATED {claim}")
        assert best < ENTAILMENT_THRESHOLD, f"should have been blocked: {claim} ({best:.3f})"


def test_mixed_draft_is_partially_stripped():
    """The realistic case: a fluent answer where most is right and one line is invented."""
    draft = (
        "Land holding farmer families receive Rs. 6000 per year under the scheme. "
        "The money is paid in three equal instalments. "
        "The scheme also provides a free tractor to every registered farmer."
    )
    res = verify(draft, CLAUSES, verifier=get_verifier())
    assert not res.abstained
    assert len(res.verdicts) == 3
    assert len(res.kept) == 2
    assert len(res.stripped) == 1
    assert "tractor" in res.stripped[0].sentence
    assert "tractor" not in res.answer
    for v in res.kept:
        assert v.clause is not None
    print(f"    catch-rate {res.catch_rate:.0%}, answer: {res.answer[:70]}...")


def test_fully_fabricated_draft_abstains():
    draft = " ".join(FABRICATED)
    res = verify(draft, CLAUSES, verifier=get_verifier())
    assert res.abstained, "a wholly invented answer must not reach the user"
    assert res.answer == ""
    print(f"    abstained, catch-rate {res.catch_rate:.0%}")


def test_off_topic_claim_is_blocked():
    """Topical overlap is not entailment."""
    v = get_verifier()
    best = max(v.entailment_scores([c.text for c in CLAUSES],
                                   "The weather in Ghaziabad tomorrow will be sunny."))
    print(f"    {best:.3f}  OFF-TOPIC")
    assert best < ENTAILMENT_THRESHOLD


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print("Loading NLI model (first run downloads ~700 MB)...\n")
    failed = 0
    for fn in fns:
        print(f"  {fn.__name__}")
        try:
            fn()
            print("  PASS\n")
        except Exception:
            failed += 1
            print("  FAIL")
            traceback.print_exc()
            print()
    print(f"{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
