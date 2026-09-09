"""The gate.

This is the part of PRAMAAN that is not a prompt. A drafted answer arrives as
prose; we split it into sentences and, for each one, ask whether ANY retrieved
clause entails it. Sentences that nothing entails are removed before the user
sees them. If too little survives, we abstain outright.

The model is never asked to police itself. It writes, and then a separate
classifier decides what is allowed through. That asymmetry is the whole design:
a prompt can be talked out of; a threshold cannot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import (
    ENTAILMENT_THRESHOLD,
    MIN_SURVIVING_FRACTION,
    NLI_MODEL,
    VERIFIER_BACKEND,
    VERIFY_TIME_K,
)
from .ingest import Clause


# ---------------------------------------------------------------- sentences
_ABBREV = r"(?<!\bNo)(?<!\bRs)(?<!\bSr)(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bi\.e)(?<!\be\.g)(?<!\bvs)"
_SENT_SPLIT = re.compile(rf"{_ABBREV}(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if len(p) > 2]


# ---------------------------------------------------------------- results
@dataclass
class SentenceVerdict:
    sentence: str
    supported: bool
    score: float
    clause: Clause | None = None

    @property
    def citation(self) -> str:
        return self.clause.citation() if self.clause else ""


@dataclass
class VerificationResult:
    verdicts: list[SentenceVerdict] = field(default_factory=list)
    abstained: bool = False
    backend: str = ""

    @property
    def kept(self) -> list[SentenceVerdict]:
        return [v for v in self.verdicts if v.supported]

    @property
    def stripped(self) -> list[SentenceVerdict]:
        return [v for v in self.verdicts if not v.supported]

    @property
    def answer(self) -> str:
        return " ".join(v.sentence for v in self.kept)

    @property
    def catch_rate(self) -> float:
        """Share of drafted sentences the gate blocked. Reported, not hidden."""
        if not self.verdicts:
            return 0.0
        return len(self.stripped) / len(self.verdicts)

    def cited_clauses(self) -> list[Clause]:
        seen, out = set(), []
        for v in self.kept:
            if v.clause and v.clause.clause_id not in seen:
                seen.add(v.clause.clause_id)
                out.append(v.clause)
        return out


# ---------------------------------------------------------------- backends
class NLIVerifier:
    """Local cross-encoder NLI. No network, deterministic, measurable."""

    name = "nli"

    def __init__(self, model_name: str = NLI_MODEL) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # Label order differs between NLI checkpoints; read it rather than
        # assuming, otherwise the gate silently inverts.
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(
            (i for i, lbl in id2label.items() if "entail" in lbl), None
        )
        if self.entail_idx is None:
            raise ValueError(f"No entailment label in {id2label}")

    def entailment_scores(self, premises: list[str], hypothesis: str) -> list[float]:
        if not premises:
            return []
        with self.torch.no_grad():
            batch = self.tok(
                premises, [hypothesis] * len(premises),
                return_tensors="pt", truncation=True, padding=True, max_length=512,
            )
            logits = self.model(**batch).logits
            probs = self.torch.softmax(logits, dim=-1)
        return probs[:, self.entail_idx].tolist()


class LLMVerifier:
    """Entailment judged by the API. For hosts that cannot carry torch.

    Weaker than the NLI gate -- it reintroduces a model into the checking loop
    -- but it is still a separate call with a single yes/no job, not a polite
    request to the drafter. Used only when NLIVerifier cannot load.
    """

    name = "llm"

    def __init__(self) -> None:
        from .drafter import detect_provider

        if detect_provider() == "extractive":
            raise RuntimeError("no LLM provider configured for the LLM verifier")

    def entailment_scores(self, premises: list[str], hypothesis: str) -> list[float]:
        from .drafter import judge_entailment

        return judge_entailment(premises, hypothesis)


def make_verifier(backend: str = VERIFIER_BACKEND):
    if backend == "nli":
        return NLIVerifier()
    if backend == "llm":
        return LLMVerifier()
    try:
        return NLIVerifier()
    except Exception as exc:  # torch missing, model not downloadable, etc.
        print(f"[verifier] NLI backend unavailable ({type(exc).__name__}); using LLM judge")
        return LLMVerifier()


# ---------------------------------------------------------------- the gate
def verify(
    draft: str,
    clauses: list[Clause],
    verifier=None,
    threshold: float = ENTAILMENT_THRESHOLD,
    min_surviving: float = MIN_SURVIVING_FRACTION,
    retriever=None,
    verify_k: int = VERIFY_TIME_K,
) -> VerificationResult:
    """Check each sentence of `draft` against the evidence.

    If a `retriever` is supplied, the gate does not simply trust the clauses
    that were handed to the drafter -- it runs its own retrieval per sentence
    and adds what it finds to the premise pool.

    This matters for compound questions. Asked "what do I get under PM-KISAN
    and who is left out", a single query embedding retrieves the benefit
    clauses and misses the exclusion list, so a perfectly true sentence about
    income-tax payers gets deleted for want of evidence that exists in the
    corpus. Retrieving per sentence, at the granularity the check actually
    operates on, fixes that. It can only ever add support; a fabricated
    sentence still finds nothing to entail it.
    """
    verifier = verifier or make_verifier()
    result = VerificationResult(backend=verifier.name)

    for sentence in split_sentences(draft):
        pool = list(clauses)
        if retriever is not None and verify_k:
            seen = {c.clause_id for c in pool}
            try:
                for extra, _ in retriever.search(sentence, k=verify_k):
                    if extra.clause_id not in seen:
                        seen.add(extra.clause_id)
                        pool.append(extra)
            except Exception:
                pass  # a retrieval failure must not break the gate

        scores = verifier.entailment_scores([c.text for c in pool], sentence)
        if scores:
            best = max(range(len(scores)), key=lambda i: scores[i])
            score, clause = scores[best], pool[best]
        else:
            score, clause = 0.0, None
        result.verdicts.append(
            SentenceVerdict(
                sentence=sentence,
                supported=score >= threshold,
                score=score,
                clause=clause if score >= threshold else None,
            )
        )

    if not result.verdicts:
        result.abstained = True
    else:
        surviving = len(result.kept) / len(result.verdicts)
        result.abstained = surviving < min_surviving
    return result
