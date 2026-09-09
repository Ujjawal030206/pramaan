"""Pick an NLI model on evidence, not vibes.

The gate is the project, so it cannot simply be made fast -- it has to stay
accurate while it gets fast. This measures both on the same fixed pairs:

  separation = min(score over true claims) - max(score over fabricated claims)

A big positive separation means the threshold sits in a wide, safe gap. A small
or negative one means the gate is guessing, however quickly it does it.

    python scripts/bench_verifier.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def _real_premises() -> list[str]:
    """Use clauses the retriever actually returns.

    An earlier version of this benchmark used short hand-written premises and
    picked a model that was 25x faster and looked accurate -- then abstained on
    real questions in production, because real clauses are 400-700 chars of
    OCR-scarred legal text, not tidy sentences. Benchmark on what ships.
    """
    from pramaan.retriever import Retriever

    r = Retriever()
    seen, out = set(), []
    for q in ("How much do I get under PM-KISAN and in how many instalments?",
              "Who is excluded from PM-KISAN for paying income tax?",
              "What is the age range for Atal Pension Yojana?"):
        for c, _ in r.search(q, k=4):
            if c.clause_id not in seen:
                seen.add(c.clause_id)
                out.append(c.text)
    return out


PREMISES = _real_premises()

TRUE_CLAIMS = [
    "Land holding farmer families receive Rs. 6000 per year under the scheme.",
    "The money is paid in three equal instalments.",
    "Persons who paid income tax in the last assessment year are not eligible.",
    "A farmer's family means husband, wife and minor children.",
]

FALSE_CLAIMS = [
    "The scheme also provides a free tractor to every registered farmer.",
    "Applicants receive Rs. 50,000 per year under this scheme.",
    "The benefit is paid in twelve monthly instalments.",
    "The money is credited every month on the first working day.",
    "The weather in Ghaziabad tomorrow will be sunny.",
]

CANDIDATES = [
    ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", False),   # current default
    ("cross-encoder/nli-deberta-v3-small", False),
    ("microsoft/deberta-base-mnli", False),                    # v1: no disentangled-attention tax
    ("typeform/distilbert-base-uncased-mnli", False),
    ("cross-encoder/nli-distilroberta-base", False),
]


def build(model_name: str, quantize: bool):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(max(1, __import__("os").cpu_count() or 2))
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    if quantize:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    idx = next(i for i, lbl in id2label.items() if "entail" in lbl)

    def score(hypothesis: str) -> list[float]:
        with torch.no_grad():
            batch = tok(PREMISES, [hypothesis] * len(PREMISES),
                        return_tensors="pt", truncation=True,
                        padding=True, max_length=384)
            probs = torch.softmax(model(**batch).logits, dim=-1)
        return probs[:, idx].tolist()

    return score


def main() -> int:
    mean_chars = sum(len(p) for p in PREMISES) // max(1, len(PREMISES))
    print(f"premises: {len(PREMISES)}, mean {mean_chars} chars\n")
    print(f"{'model':<46} {'quant':<6} {'s/claim':>8} {'min(true)':>10} "
          f"{'max(false)':>11} {'sep':>7}")
    print("-" * 92)
    rows = []
    for name, quant in CANDIDATES:
        try:
            score = build(name, quant)
            score("warmup sentence for a fair timing measurement.")
            t = time.time()
            trues = [max(score(c)) for c in TRUE_CLAIMS]
            falses = [max(score(c)) for c in FALSE_CLAIMS]
            per = (time.time() - t) / (len(TRUE_CLAIMS) + len(FALSE_CLAIMS))
        except Exception as exc:
            print(f"{name:<46} {'int8' if quant else '-':<6}  FAILED "
                  f"{type(exc).__name__}: {str(exc)[:40]}")
            continue
        lo, hi = min(trues), max(falses)
        rows.append((hi - lo, per, name, quant, lo, hi))
        print(f"{name:<46} {'int8' if quant else '-':<6} {per:>8.2f} "
              f"{lo:>10.3f} {hi:>11.3f} {lo - hi:>7.3f}")

    print()
    usable = [r for r in rows if r[4] > r[5]]
    if usable:
        usable.sort(key=lambda r: r[1])
        sep, per, name, quant, lo, hi = usable[0]
        print(f"Fastest with a clean gap: {name} "
              f"{'(int8)' if quant else ''} -- {per:.2f}s/claim, "
              f"true>={lo:.3f}, false<={hi:.3f}")
    else:
        print("No candidate separated true from fabricated claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
