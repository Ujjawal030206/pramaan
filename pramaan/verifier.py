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
        from .drafter import get_client

        self.client = get_client()

    def entailment_scores(self, premises: list[str], hypothesis: str) -> list[float]:
        import json

        from .config import DRAFT_MODEL

        if not premises:
            return []
        numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(premises))
        msg = self.client.messages.create(
            model=DRAFT_MODEL,
            max_tokens=400,
            system=(
                "You judge textual entailment. For each numbered PREMISE, decide "
                "whether it alone entails the CLAIM. Entailment means the premise "
                "states or directly implies the claim. Topical overlap is not "
                "entailment. Reply with JSON only: "
                '{"scores": [{"i": 0, "p": 0.0}, ...]} where p is the probability '
                "in [0,1] that the premise entails the claim."
            ),
            messages=[{
                "role": "user",
                "content": f"PREMISES:\n{numbered}\n\nCLAIM: {hypothesis}",
            }],
        )
        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        try:
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            out = [0.0] * len(premises)
            for item in data.get("scores", []):
                i = int(item["i"])
                if 0 <= i < len(out):
                    out[i] = float(item["p"])
            return out
        except Exception:
            return [0.0] * len(premises)


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
) -> VerificationResult:
    verifier = verifier or make_verifier()
    premises = [c.text for c in clauses]
    result = VerificationResult(backend=verifier.name)

    for sentence in split_sentences(draft):
        scores = verifier.entailment_scores(premises, sentence)
        if scores:
            best = max(range(len(scores)), key=lambda i: scores[i])
            score, clause = scores[best], clauses[best]
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
