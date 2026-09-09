"""Gate behaviour, with a stub entailment model.

No torch and no API key needed -- these pin the control flow of the gate
itself: what survives, what is stripped, and when we abstain.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pramaan.ingest import Clause  # noqa: E402
from pramaan.verifier import split_sentences, verify  # noqa: E402


def clause(text: str, cid: str = "doc#1") -> Clause:
    return Clause(
        clause_id=cid, text=text, scheme="Test Scheme", doc_id="doc",
        doc_title="Test Document", authority="Test Authority",
        url="https://example.gov.in/doc.pdf", page=1,
    )


class StubVerifier:
    """Scores a sentence by whether a marker word appears in the premise."""

    name = "stub"

    def __init__(self, supported_markers: set[str]):
        self.markers = supported_markers

    def entailment_scores(self, premises, hypothesis):
        hit = any(m.lower() in hypothesis.lower() for m in self.markers)
        return [0.95 if hit else 0.05 for _ in premises]


def test_split_sentences_basic():
    got = split_sentences("You are eligible. Payment is Rs. 6000 per year. Apply now!")
    assert got == [
        "You are eligible.",
        "Payment is Rs. 6000 per year.",
        "Apply now!",
    ], got


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_supported_sentence_survives_with_citation():
    res = verify(
        "The benefit is six thousand rupees per year.",
        [clause("An income support of Rs 6000 per year is provided.")],
        verifier=StubVerifier({"six thousand"}),
    )
    assert not res.abstained
    assert len(res.kept) == 1
    assert res.kept[0].clause is not None
    assert res.kept[0].citation.startswith("Test Document, p. 1")


def test_unsupported_sentence_is_stripped():
    draft = (
        "The benefit is six thousand rupees per year. "
        "You will also receive a free tractor."
    )
    res = verify(
        draft,
        [clause("An income support of Rs 6000 per year is provided.")],
        verifier=StubVerifier({"six thousand"}),
    )
    assert len(res.verdicts) == 2
    assert len(res.kept) == 1
    assert len(res.stripped) == 1
    assert "tractor" in res.stripped[0].sentence
    assert "tractor" not in res.answer          # never reaches the user
    assert res.catch_rate == 0.5


def test_abstains_when_nothing_is_supported():
    res = verify(
        "You will receive a free tractor. Delivery is next Tuesday.",
        [clause("An income support of Rs 6000 per year is provided.")],
        verifier=StubVerifier({"nothing-matches-this"}),
    )
    assert res.abstained
    assert res.answer == ""
    assert res.catch_rate == 1.0


def test_does_not_abstain_when_most_survives():
    draft = "Alpha claim here. Alpha again here. Unsupported tail sentence."
    res = verify(
        draft,
        [clause("Alpha is documented.")],
        verifier=StubVerifier({"alpha"}),
    )
    assert not res.abstained
    assert len(res.kept) == 2


def test_stripped_sentences_carry_no_citation():
    res = verify(
        "Alpha claim. Beta invention.",
        [clause("Alpha is documented.")],
        verifier=StubVerifier({"alpha"}),
    )
    for v in res.stripped:
        assert v.clause is None
        assert v.citation == ""


def test_empty_clause_list_abstains():
    res = verify("Anything at all.", [], verifier=StubVerifier({"anything"}))
    assert res.abstained


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
