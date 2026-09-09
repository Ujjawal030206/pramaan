"""Ask -> retrieve -> draft -> verify -> deliver or abstain."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import TOP_K
from .drafter import NO_EVIDENCE, draft_answer
from .ingest import Clause
from .retriever import Retriever
from .verifier import VerificationResult, make_verifier, verify

ABSTAIN_MESSAGE = (
    "I could not verify an answer to this from the official documents I hold. "
    "Rather than guess, I am not answering."
)


@dataclass
class Outcome:
    question: str
    answer: str
    abstained: bool
    retrieved: list[tuple[Clause, float]] = field(default_factory=list)
    verification: VerificationResult | None = None
    draft: str = ""
    elapsed: float = 0.0
    reason: str = ""

    @property
    def citations(self) -> list[Clause]:
        return self.verification.cited_clauses() if self.verification else []

    @property
    def catch_rate(self) -> float:
        return self.verification.catch_rate if self.verification else 0.0

    def next_step(self) -> str:
        """What a user should do when we decline to answer."""
        schemes = sorted({c.scheme for c, _ in self.retrieved})
        if schemes:
            return (
                "The closest documents I hold cover "
                + ", ".join(schemes)
                + ". For a decision on your own case, contact your Common Service "
                "Centre (CSC) or the scheme helpline named in those documents."
            )
        return (
            "Nothing in my corpus is close to this question. Contact your Common "
            "Service Centre (CSC) or the relevant department directly."
        )


class Pramaan:
    """Holds the loaded models so a UI can answer repeatedly without reloading."""

    def __init__(self, verifier=None, retriever=None) -> None:
        self.retriever = retriever or Retriever()
        self.verifier = verifier or make_verifier()

    def ask(self, question: str, k: int = TOP_K) -> Outcome:
        t0 = time.time()
        question = (question or "").strip()
        if not question:
            return Outcome(question, ABSTAIN_MESSAGE, True, reason="empty question")

        retrieved = self.retriever.search(question, k=k)
        clauses = [c for c, _ in retrieved]
        if not clauses:
            return Outcome(
                question, ABSTAIN_MESSAGE, True, retrieved=retrieved,
                elapsed=time.time() - t0, reason="nothing retrieved",
            )

        try:
            draft = draft_answer(question, clauses)
        except Exception as exc:
            # A missing key or a provider outage must not surface as a traceback
            # mid-demo, and must not be mistaken for a verified answer.
            return Outcome(
                question, ABSTAIN_MESSAGE, True, retrieved=retrieved,
                elapsed=time.time() - t0,
                reason=f"drafting step unavailable ({type(exc).__name__}: {exc})",
            )
        if not draft:
            return Outcome(
                question, ABSTAIN_MESSAGE, True, retrieved=retrieved,
                elapsed=time.time() - t0, reason="empty draft",
            )

        # The drafter signals "the clauses do not settle this" with a sentinel
        # rather than prose. Without it the model writes "the clauses do not
        # address X", which is a sentence about our own retrieval -- and one
        # that can pick up an entailment score and arrive at the user wearing a
        # citation to an unrelated clause. Refusal must not be citable.
        if draft.strip().upper().startswith(NO_EVIDENCE):
            return Outcome(
                question, ABSTAIN_MESSAGE, True, retrieved=retrieved,
                draft=draft, elapsed=time.time() - t0,
                reason="the drafter found nothing in the retrieved clauses that settles this",
            )

        # The gate gets the retriever too, so it can look for evidence the
        # drafting query missed rather than deleting true sentences.
        result = verify(draft, clauses, verifier=self.verifier, retriever=self.retriever)
        if result.abstained:
            return Outcome(
                question, ABSTAIN_MESSAGE, True, retrieved=retrieved,
                verification=result, draft=draft, elapsed=time.time() - t0,
                reason="too little of the draft was supported by any clause",
            )

        return Outcome(
            question, result.answer, False, retrieved=retrieved,
            verification=result, draft=draft, elapsed=time.time() - t0,
        )
