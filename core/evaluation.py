"""Offline evaluation: run the detector on a labelled question set, measure how well each
uncertainty signal predicts correctness, and train the learned calibrator."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from . import prompts as P
from .calibration import FEATURES, LogisticCalibrator, auroc, brier, cross_val_predict, ece, reliability_bins
from .llm import LLM
from .pipeline import DetectionConfig, classify, run_detection

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "eval_questions.json"
RESULTS_DIR = ROOT / "results"
RECORDS_PATH = RESULTS_DIR / "eval_records.jsonl"
REPORT_PATH = RESULTS_DIR / "eval_report.json"
CALIBRATOR_PATH = RESULTS_DIR / "calibrator.json"

CATEGORY_LABELS = {"common": "Well-known", "obscure": "Obscure facts", "false_premise": "False premise"}


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text())


def load_records() -> list[dict]:
    if not RECORDS_PATH.exists():
        return []
    return [json.loads(line) for line in RECORDS_PATH.read_text().splitlines() if line.strip()]


def load_report() -> dict | None:
    return json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else None


def grade(llm: LLM, judge: str, item: dict, answer: str) -> tuple[bool, str]:
    data = llm.chat_json(judge, [{"role": "user", "content": P.CORRECTNESS_JUDGE.format(
        question=item["question"], reference=item["reference"], answer=answer)}], max_tokens=600)
    correct = data.get("correct")
    if isinstance(correct, str):
        correct = correct.strip().lower() == "true"
    return bool(correct), str(data.get("reason", ""))


def evaluate_item(llm: LLM, cfg: DetectionConfig, item: dict) -> dict:
    res = run_detection(llm, item["question"], cfg, calibrator=None)
    correct, reason = grade(llm, cfg.judge, item, res["answer"])
    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "reference": item["reference"],
        "answer": res["answer"],
        "correct": correct,
        "grade_reason": reason,
        "features": res["features"],
        "heuristic": res["heuristic"],
        "verdict": res["verification"]["verdict"],
        "generator": cfg.generator,
        "judge": cfg.judge,
        "n_samples": cfg.n_samples,
        "elapsed": res["elapsed"],
    }


def run_evaluation(
    llm: LLM,
    cfg: DetectionConfig,
    items: list[dict],
    fresh: bool = False,
    progress: Callable[[int, int, dict | None], None] | None = None,
) -> list[dict]:
    """Evaluate items, appending each record to disk so an interrupted run can resume."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if fresh and RECORDS_PATH.exists():
        RECORDS_PATH.unlink()
    done = {r["id"]: r for r in load_records() if r.get("generator") == cfg.generator}
    records = []
    for i, item in enumerate(items):
        if item["id"] in done:
            records.append(done[item["id"]])
            if progress:
                progress(i + 1, len(items), done[item["id"]])
            continue
        for attempt in range(3):
            try:
                record = evaluate_item(llm, cfg, item)
                break
            except Exception:  # noqa: BLE001 - keep the run alive on transient API errors
                if attempt == 2:
                    raise
                time.sleep(20 * (attempt + 1))
        with RECORDS_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        records.append(record)
        if progress:
            progress(i + 1, len(items), record)
    return records


def _metrics(probs, labels) -> dict:
    return {
        "auroc": auroc(probs, labels),
        "ece": ece(probs, labels),
        "brier": brier(probs, labels),
        "mean_confidence": float(np.mean(probs)),
        "reliability": reliability_bins(probs, labels),
    }


def build_report(records: list[dict], cfg: DetectionConfig) -> dict:
    """Compare signals, train the calibrator on all records, save calibrator + report."""
    X = np.array([[r["features"][f] for f in FEATURES] for r in records])
    y = np.array([1 if r["correct"] else 0 for r in records])
    heuristic = np.array([r["heuristic"] for r in records])
    oof = cross_val_predict(X, y, k=5)

    methods = {
        "Verbalized confidence": X[:, 2],
        "Semantic consistency": X[:, 0],
        "Self-verification (CoVe)": X[:, 1],
        "Heuristic combination": heuristic,
        "Learned calibrator (5-fold CV)": oof,
    }
    report_methods = {name: _metrics(p, y) for name, p in methods.items()}

    calibrator = LogisticCalibrator(meta={
        "trained_on": len(records),
        "generator": records[0]["generator"] if records else cfg.generator,
        "judge": records[0]["judge"] if records else cfg.judge,
        "created": datetime.now().isoformat(timespec="seconds"),
    }).fit(X, y)
    calibrator.save(CALIBRATOR_PATH)

    by_category = {}
    for cat in CATEGORY_LABELS:
        idx = [i for i, r in enumerate(records) if r["category"] == cat]
        if idx:
            by_category[cat] = {
                "label": CATEGORY_LABELS[cat],
                "n": len(idx),
                "accuracy": float(y[idx].mean()),
                "mean_verbalized": float(X[idx, 2].mean()),
                "mean_calibrated": float(oof[idx].mean()),
            }

    rows = []
    for r, p in zip(records, oof):
        rows.append({
            "id": r["id"], "category": r["category"], "question": r["question"],
            "reference": r["reference"], "answer": r["answer"], "correct": r["correct"],
            "grade_reason": r.get("grade_reason", ""),
            "verbalized": r["features"]["verbalized"], "consistency": r["features"]["consistency"],
            "verification": r["features"]["verification"], "heuristic": r["heuristic"],
            "calibrated": float(p), "verdict": r["verdict"],
            "status": classify(float(p), r["verdict"], r["features"]["consistency"], cfg),
        })

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "generator": calibrator.meta["generator"],
        "judge": calibrator.meta["judge"],
        "n": len(records),
        "n_samples": records[0]["n_samples"] if records else cfg.n_samples,
        "accuracy": float(y.mean()) if len(y) else 0.0,
        "methods": report_methods,
        "by_category": by_category,
        "calibrator": calibrator.to_dict(),
        "rows": rows,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report
