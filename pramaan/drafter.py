"""Drafting step -- deliberately provider-agnostic.

The drafter is the one component of PRAMAAN that is a commodity. The gate is
what makes the system trustworthy, and the gate does not care which model wrote
the sentences it checks. So we support whichever provider you can actually get
a key for, and the free tiers are first-class rather than an afterthought:

    GROQ_API_KEY      console.groq.com          free, no card, very fast
    GEMINI_API_KEY    aistudio.google.com       free, no card
    OPENAI_API_KEY    any OpenAI-compatible URL (OpenRouter, Ollama, LM Studio)
    ANTHROPIC_API_KEY paid

Set one and PRAMAAN picks it up. Set none and it falls back to `extractive`,
which answers using only sentences lifted verbatim from the retrieved clauses.
That keeps the app usable offline, but be honest about it in a demo: an
extractive draft is made of clause text, so the gate passes it by construction
and is not being exercised. Use a real generative backend to show the gate
doing its job.
"""
from __future__ import annotations

import json
import os
import re

from .config import (
    DRAFT_MAX_TOKENS,
    DRAFT_MODEL,
    DRAFT_PROVIDER,
    OPENAI_BASE_URL,
)
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

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5",
}

_TIMEOUT = 60


def detect_provider() -> str:
    """Explicit setting wins; otherwise use whichever key is present."""
    if DRAFT_PROVIDER and DRAFT_PROVIDER != "auto":
        return DRAFT_PROVIDER
    for env, name in (
        ("GROQ_API_KEY", "groq"),
        ("GEMINI_API_KEY", "gemini"),
        ("GOOGLE_API_KEY", "gemini"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
    ):
        if os.getenv(env):
            return name
    return "extractive"


def model_for(provider: str) -> str:
    return DRAFT_MODEL or DEFAULT_MODELS.get(provider, "")


# ---------------------------------------------------------------- providers
def _post(url: str, payload: dict, headers: dict) -> dict:
    import requests

    r = requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"{url.split('/')[2]} returned {r.status_code}: {r.text[:300]}")
    return r.json()


def _chat_openai_compatible(system: str, user: str, *, key: str, base_url: str,
                            model: str, max_tokens: int) -> str:
    data = _post(
        f"{base_url.rstrip('/')}/chat/completions",
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    return data["choices"][0]["message"]["content"].strip()


def _chat_gemini(system: str, user: str, *, key: str, model: str,
                 max_tokens: int) -> str:
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
        },
        {"Content-Type": "application/json"},
    )
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError):
        return ""


def _chat_anthropic(system: str, user: str, *, key: str, model: str,
                    max_tokens: int) -> str:
    import anthropic

    msg = anthropic.Anthropic(api_key=key).messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content
                   if getattr(b, "type", "") == "text").strip()


def chat(system: str, user: str, max_tokens: int = DRAFT_MAX_TOKENS) -> str:
    """One completion from whichever provider is configured."""
    provider = detect_provider()
    model = model_for(provider)

    if provider == "groq":
        return _chat_openai_compatible(
            system, user, key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            model=model, max_tokens=max_tokens)

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        return _chat_gemini(system, user, key=key, model=model, max_tokens=max_tokens)

    if provider == "openai":
        return _chat_openai_compatible(
            system, user, key=os.getenv("OPENAI_API_KEY", "not-needed"),
            base_url=OPENAI_BASE_URL, model=model, max_tokens=max_tokens)

    if provider == "anthropic":
        return _chat_anthropic(
            system, user, key=os.environ["ANTHROPIC_API_KEY"],
            model=model, max_tokens=max_tokens)

    raise RuntimeError(
        "No LLM provider configured. Set GROQ_API_KEY (free, console.groq.com) "
        "or GEMINI_API_KEY (free, aistudio.google.com) in your .env, or set "
        "PRAMAAN_DRAFT_PROVIDER=extractive to run without one."
    )


# ---------------------------------------------------------------- extractive
_SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def extractive_draft(question: str, clauses: list[Clause], limit: int = 4) -> str:
    """Answer using only sentences copied verbatim out of the retrieved clauses.

    A zero-cost, zero-network fallback. Note that because every sentence comes
    from a clause, the gate will pass essentially all of it -- this proves
    retrieval and citation still work, but it does not demonstrate the gate.
    """
    from .bm25 import tokenize

    want = set(tokenize(question)) - {
        "what", "is", "the", "a", "an", "i", "am", "can", "get", "do", "does",
        "for", "of", "to", "in", "and", "my", "me", "you", "how", "much", "who",
    }
    scored: list[tuple[float, str]] = []
    for clause in clauses:
        for sentence in _SENT.split(clause.text):
            sentence = sentence.strip()
            if not (40 <= len(sentence) <= 300):
                continue
            toks = set(tokenize(sentence))
            if not toks:
                continue
            overlap = len(want & toks) / (len(want) or 1)
            if overlap > 0:
                scored.append((overlap, sentence))

    scored.sort(key=lambda x: -x[0])
    picked, seen = [], set()
    for _, sentence in scored:
        key = sentence[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(sentence if sentence.endswith(".") else sentence + ".")
        if len(picked) >= limit:
            break
    return " ".join(picked)


# ---------------------------------------------------------------- public API
def format_clauses(clauses: list[Clause]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.scheme} -- {c.doc_title}, p.{c.page})\n{c.text}"
        for i, c in enumerate(clauses)
    )


def draft_answer(question: str, clauses: list[Clause]) -> str:
    if not clauses:
        return ""
    if detect_provider() == "extractive":
        return extractive_draft(question, clauses)
    return chat(SYSTEM, f"CLAUSES:\n{format_clauses(clauses)}\n\nQUESTION: {question}")


def judge_entailment(premises: list[str], hypothesis: str) -> list[float]:
    """Entailment scored by the LLM. Used only when torch is unavailable."""
    if not premises:
        return []
    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(premises))
    raw = chat(
        "You judge textual entailment. For each numbered PREMISE, decide whether "
        "it alone entails the CLAIM. Entailment means the premise states or "
        "directly implies the claim. Topical overlap is not entailment. Reply "
        'with JSON only: {"scores": [{"i": 0, "p": 0.0}]} where p is the '
        "probability in [0,1] that the premise entails the claim.",
        f"PREMISES:\n{numbered}\n\nCLAIM: {hypothesis}",
        max_tokens=600,
    )
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
