"""Run the offline evaluation and train the learned calibrator.

Usage:
    python evaluate.py                 # resume / run on the full question set
    python evaluate.py --fresh         # discard previous records and start over
    python evaluate.py --limit 10      # quick run on the first 10 questions
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from core.evaluation import RECORDS_PATH, REPORT_PATH, build_report, load_dataset, run_evaluation  # noqa: E402
from core.llm import LLM, PREFERRED_GENERATORS, PREFERRED_JUDGES, pick_model  # noqa: E402
from core.pipeline import DetectionConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", help="model under test")
    parser.add_argument("--judge", help="comparator model")
    parser.add_argument("--samples", type=int, default=3, help="stochastic samples per question")
    parser.add_argument("--limit", type=int, help="only evaluate the first N questions")
    parser.add_argument("--fresh", action="store_true", help="ignore previous records")
    args = parser.parse_args()

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is missing. Add it to .env first.")
    llm = LLM(key)
    models = llm.list_chat_models()
    cfg = DetectionConfig(
        generator=args.generator or pick_model(models, PREFERRED_GENERATORS),
        judge=args.judge or pick_model(models, PREFERRED_JUDGES),
        n_samples=args.samples,
    )
    items = load_dataset()[: args.limit] if args.limit else load_dataset()
    print(f"Generator: {cfg.generator} | Judge: {cfg.judge} | samples: {cfg.n_samples} | questions: {len(items)}\n")

    def progress(i: int, n: int, rec: dict | None) -> None:
        if rec:
            mark = "OK " if rec["correct"] else "BAD"
            f = rec["features"]
            print(f"[{i:>2}/{n}] {mark} cons={f['consistency']:.2f} ver={f['verification']:.2f} "
                  f"verb={f['verbalized']:.2f} | {rec['question'][:60]}", flush=True)

    records = run_evaluation(llm, cfg, items, fresh=args.fresh, progress=progress)
    report = build_report(records, cfg)

    print(f"\nGenerator accuracy: {report['accuracy']:.0%} on {report['n']} questions\n")
    print(f"{'Method':<34}{'AUROC':>8}{'ECE':>8}{'Brier':>8}")
    for name, m in report["methods"].items():
        au = f"{m['auroc']:.3f}" if m["auroc"] is not None else "  n/a"
        print(f"{name:<34}{au:>8}{m['ece']:>8.3f}{m['brier']:>8.3f}")
    print(f"\nSaved: {RECORDS_PATH.name}, {REPORT_PATH.name}, calibrator.json  (in results/)")
    print(f"Total API usage: {llm.usage()}")


if __name__ == "__main__":
    main()
