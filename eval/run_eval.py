"""Measure the gate and publish the numbers.

Reports the three figures PRAMAAN commits to showing:
  catch-rate           share of drafted sentences the gate blocked
  abstention rate      share of questions we declined
  citation coverage    share of surviving sentences carrying a clause

plus the two that say whether the abstention is calibrated rather than merely
frequent -- a system that abstains on everything scores a perfect catch-rate
and is useless:
  answer rate          on questions the corpus DOES settle
  abstention precision on questions the corpus does NOT settle

Usage:
    python eval/run_eval.py [--out eval/results.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from pramaan.pipeline import Pramaan  # noqa: E402

QUESTIONS = Path(__file__).resolve().parent / "questions.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    spec = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        spec = spec[: args.limit]

    engine = Pramaan()
    rows, t0 = [], time.time()

    for i, item in enumerate(spec, 1):
        q, expect = item["q"], item["expect"]
        out = engine.ask(q)
        v = out.verification
        kept = len(v.kept) if v else 0
        cited = sum(1 for s in (v.kept if v else []) if s.clause is not None)
        # Separates the two ways a sentence dies, so a drop in accuracy can be
        # pinned on the contradiction veto or on plain low entailment.
        vetoed = sum(1 for s in (v.verdicts if v else []) if s.contradicted_by is not None)
        correct = (expect == "abstain") == out.abstained

        rows.append({
            "question": q,
            "expect": expect,
            "abstained": out.abstained,
            "correct": correct,
            "drafted": len(v.verdicts) if v else 0,
            "kept": kept,
            "blocked": len(v.stripped) if v else 0,
            "cited": cited,
            "vetoed": vetoed,
            "catch_rate": out.catch_rate,
            "answer": out.answer if not out.abstained else "",
            "elapsed": round(out.elapsed, 2),
        })
        flag = "ok " if correct else "MISS"
        print(f"[{i:>2}/{len(spec)}] {flag} expect={expect:<7} "
              f"abstained={str(out.abstained):<5} "
              f"kept={kept}/{len(v.verdicts) if v else 0} vetoed={vetoed}  {q[:50]}")

    answerable = [r for r in rows if r["expect"] == "answer"]
    unanswerable = [r for r in rows if r["expect"] == "abstain"]
    drafted_total = sum(r["drafted"] for r in rows)
    blocked_total = sum(r["blocked"] for r in rows)
    kept_total = sum(r["kept"] for r in rows)
    cited_total = sum(r["cited"] for r in rows)

    summary = {
        "questions": len(rows),
        "catch_rate": (blocked_total / drafted_total) if drafted_total else 0.0,
        "abstention_rate": sum(r["abstained"] for r in rows) / len(rows),
        "citation_coverage": (cited_total / kept_total) if kept_total else 1.0,
        "answer_rate_on_answerable": (
            sum(not r["abstained"] for r in answerable) / len(answerable)
            if answerable else 0.0
        ),
        "abstention_precision_on_unanswerable": (
            sum(r["abstained"] for r in unanswerable) / len(unanswerable)
            if unanswerable else 0.0
        ),
        "overall_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "median_latency_s": statistics.median(r["elapsed"] for r in rows),
        "wall_clock_s": round(time.time() - t0, 1),
    }

    print("\n" + "=" * 62)
    print("PRAMAAN evaluation")
    print("=" * 62)
    print(f"  Verification catch-rate            {summary['catch_rate']:.1%}")
    print(f"  Abstention rate                    {summary['abstention_rate']:.1%}")
    print(f"  Citation coverage                  {summary['citation_coverage']:.1%}")
    print("  " + "-" * 58)
    print(f"  Answer rate (answerable qs)        {summary['answer_rate_on_answerable']:.1%}")
    print(f"  Abstention precision (unanswerable){summary['abstention_precision_on_unanswerable']:>7.1%}")
    print(f"  Overall accuracy                   {summary['overall_accuracy']:.1%}")
    print(f"  Median latency                     {summary['median_latency_s']:.1f}s")
    print("=" * 62)

    Path(args.out).write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
