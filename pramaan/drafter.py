"""Drafting step.

The model writes from the retrieved clauses and nothing else. This prompt is a
first line of defence, not the defence -- everything it produces still has to
get past verifier.py. We keep the instruction tight anyway, because a cleaner
draft means less work for the gate and less shredding of the final answer.
"""
from __future__ import annotations

import functools

from .config import ANTHROPIC_API_KEY, DRAFT_MAX_TOKENS, DRAFT_MODEL
from .ingest import Clause

SYSTEM = """You answer questions about Indian government welfare schemes for ordinary citizens.

Rules you must follow:
- Use ONLY the numbered clauses provided. They are verbatim extracts from official documents.
- If the clauses do not settle the question, say so plainly in one sentence. Do not fill the gap.
- Never invent figures, dates, age limits, income limits or scheme names.
- Write short, complete, standalone sentences. Each sentence must carry one factual claim
  that a reader could check against a single clause.
- No preamble, no greeting, no "based on the provided context". Start with the answer.
- Plain language. The reader may not have finished school.
- 120 words maximum.
"""


@functools.lru_cache(maxsize=1)
def get_client():
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def format_clauses(clauses: list[Clause]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.scheme} -- {c.doc_title}, p.{c.page})\n{c.text}"
        for i, c in enumerate(clauses)
    )


def draft_answer(question: str, clauses: list[Clause], client=None) -> str:
    if not clauses:
        return ""
    client = client or get_client()
    msg = client.messages.create(
        model=DRAFT_MODEL,
        max_tokens=DRAFT_MAX_TOKENS,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"CLAUSES:\n{format_clauses(clauses)}\n\nQUESTION: {question}",
        }],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
