"""Ask -> retrieve -> draft -> verify -> deliver or abstain."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .config import TOP_K
from .drafter import NO_EVIDENCE, draft_answer, raw_answer
from .ingest import Clause
from .retriever import Retriever
from .verifier import VerificationResult, make_verifier, split_sentences, verify

ABSTAIN_MESSAGE = (
    "I could not verify an answer to this from the official documents I hold. "
    "Rather than guess, I am not answering."
)

# progress(stage, **detail). Stages, in order: "retrieve", "draft",
# "verify" (i, n, sentence), "done" (abstained). The comparison reports
# "plain_draft" and "plain_verify" (i, n). A UI uses this to show the pipeline
# working instead of a spinner.
Progress = Callable[..., None]


def _safe(progress: Progress | None) -> Progress:
    """Wrap a progress callback so a UI failure can never break an answer."""
    if progress is None:
        return lambda *_, **__: None

    def note(stage: str, **detail) -> None:
        try:
            progress(stage, **detail)
        except Exception:
            pass

    return note


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
    cached: bool = False

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


@dataclass
class PlainComparison:
    """The same model, asked the same question, with no documents and no gate.

    Its sentences are then scored by the gate against the clauses PRAMAAN
    retrieved. An unsupported sentence means "not found in the official
    documents we hold" -- which is not the same as "false", and any UI showing
    this has to say so.
    """

    text: str = ""
    verification: VerificationResult | None = None
    elapsed: float = 0.0
    error: str = ""


class Pramaan:
    """Holds the loaded models so a UI can answer repeatedly without reloading."""

    def __init__(self, verifier=None, retriever=None) -> None:
        self.retriever = retriever or Retriever()
        self.verifier = verifier or make_verifier()

    def ask(self, question: str, k: int = TOP_K,
            progress: Progress | None = None) -> Outcome:
        note = _safe(progress)
        t0 = time.time()

        def finish(outcome: Outcome) -> Outcome:
            outcome.elapsed = time.time() - t0
            note("done", abstained=outcome.abstained)
            return outcome

        question = (question or "").strip()
        if not question:
            return finish(Outcome(question, ABSTAIN_MESSAGE, True, reason="empty question"))

        note("retrieve")
        retrieved = self.retriever.search(question, k=k)
        clauses = [c for c, _ in retrieved]
        if not clauses:
            return finish(Outcome(
                question, ABSTAIN_MESSAGE, True, retrieved=retrieved,
                reason="nothing retrieved",
            ))

        note("draft", clauses=len(clauses))
        try:
            draft = draft_answer(question, clauses)
        except Exception as exc:
            # A missing key or a provider outage must not surface as a traceback
            # mid-demo, and must not be mistaken for a verified answer.
            return finish(Outcome(
                question, ABSTAIN_MESSAGE, True, retrieved=retrieved,
                reason=f"drafting step unavailable ({type(exc).__name__}: {exc})",
            ))
        if not draft:
            return finish(Outcome(
                question, ABSTAIN_MESSAGE, True, retrieved=retrieved,
                reason="empty draft",
            ))

        # The drafter signals "the clauses do not settle this" with a sentinel
        # rather than prose. Without it the model writes "the clauses do not
        # address X", which is a sentence about our own retrieval -- and one
        # that can pick up an entailment score and arrive at the user wearing a
        # citation to an unrelated clause. Refusal must not be citable.
        if draft.strip().upper().startswith(NO_EVIDENCE):
            return finish(Outcome(
                question, ABSTAIN_MESSAGE, True, retrieved=retrieved, draft=draft,
                reason="the drafter found nothing in the retrieved clauses that settles this",
            ))

        # The gate gets the retriever too, so it can look for evidence the
        # drafting query missed rather than deleting true sentences.
        note("verify", i=0, n=len(split_sentences(draft)))
        result = verify(
            draft, clauses, verifier=self.verifier, retriever=self.retriever,
            on_sentence=lambda i, n, s: note("verify", i=i, n=n, sentence=s),
        )
        if result.abstained:
            return finish(Outcome(
                question, ABSTAIN_MESSAGE, True, retrieved=retrieved,
                verification=result, draft=draft,
                reason="too little of the draft was supported by any clause",
            ))

        return finish(Outcome(
            question, result.answer, False, retrieved=retrieved,
            verification=result, draft=draft,
        ))

    def compare_plain(self, question: str, clauses: list[Clause],
                      progress: Progress | None = None) -> PlainComparison:
        """Answer with the bare model, then score its sentences with the gate.

        Scored against the same clauses PRAMAAN retrieved, with abstention
        switched off -- the point is to show every sentence a normal chatbot
        would have said, and which of them the documents actually support.
        """
        note = _safe(progress)
        t0 = time.time()
        note("plain_draft")
        try:
            text = raw_answer(question)
        except Exception as exc:
            return PlainComparison(error=f"{type(exc).__name__}: {exc}",
                                   elapsed=time.time() - t0)
        if not text or not clauses:
            return PlainComparison(text=text, elapsed=time.time() - t0)

        result = verify(
            text, clauses, verifier=self.verifier, retriever=self.retriever,
            min_surviving=0.0,
            on_sentence=lambda i, n, s: note("plain_verify", i=i, n=n),
        )
        return PlainComparison(text=text, verification=result, elapsed=time.time() - t0)
