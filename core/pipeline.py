"""Hallucination detection pipeline.

1. Generate the main answer (greedy).
2. Draw N stochastic samples and check every sentence of the main answer against each
   sample (SelfCheckGPT-style semantic consistency).
3. Self-verification (factored Chain-of-Verification): the model plans fact-check
   questions, answers them without seeing its original answer, and a comparator checks
   agreement. The model also reports a verbalized confidence.
4. Combine the three signals into one confidence: a heuristic weighted sum, or a learned
   logistic calibrator trained on a labelled evaluation set.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from . import prompts as P
from .calibration import LogisticCalibrator
from .llm import LLM

SENTENCE_SCORE = {"supported": 1.0, "neutral": 0.5, "contradicted": 0.0}
OVERALL_SCORE = {"agree": 1.0, "partial": 0.5, "disagree": 0.0}
CHECK_SCORE = {"consistent": 1.0, "unclear": 0.5, "inconsistent": 0.0}
VERDICT_SCORE = {"supported": 1.0, "partial": 0.5, "contradicted": 0.0, "abstained": 0.75}

_ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "st.", "jr.", "sr.", "vs.", "no.", "e.g.", "i.e.",
                  "etc.", "prof.", "inc.", "ltd.", "co.", "approx.", "u.s.", "u.k."}


@dataclass
class DetectionConfig:
    generator: str
    judge: str
    n_samples: int = 4
    sample_temperature: float = 1.0
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2)  # consistency, verification, verbalized
    low: float = 0.45   # below -> likely hallucination
    high: float = 0.75  # above -> reliable


# ------------------------------------------------------------------ text utils
def clean_text(text: str) -> str:
    text = re.sub(r"[*_`#]+", "", text or "")
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str, max_sentences: int = 8) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[\"“'(\[A-Z0-9])", text.strip())
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if sentences:
            last_word = sentences[-1].split()[-1].lower() if sentences[-1].split() else ""
            # Don't split after abbreviations or initials like "G." / "R."
            if last_word in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last_word):
                sentences[-1] += " " + part
                continue
        sentences.append(part)
    if len(sentences) > max_sentences:
        sentences = sentences[: max_sentences - 1] + [" ".join(sentences[max_sentences - 1:])]
    return sentences or ([text.strip()] if text.strip() else [])


def _label(value, allowed: dict, default: str) -> str:
    value = str(value or "").lower().strip()
    return value if value in allowed else default


def _mean(values, default: float = 0.5) -> float:
    values = list(values)
    return sum(values) / len(values) if values else default


# ------------------------------------------------------------------- detector
class Detector:
    def __init__(self, llm: LLM, cfg: DetectionConfig):
        self.llm, self.cfg = llm, cfg

    def answer(self, question: str, temperature: float) -> str:
        text = self.llm.chat(
            self.cfg.generator,
            [{"role": "system", "content": P.ANSWER_SYSTEM}, {"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=350,
        )
        return clean_text(text)

    def self_check(self, question: str, answer: str) -> tuple[float, list[str]]:
        data = self.llm.chat_json(
            self.cfg.generator,
            [{"role": "user", "content": P.SELF_CHECK.format(question=question, answer=answer)}],
            max_tokens=400,
        )
        try:
            confidence = float(data.get("confidence", 50))
        except (TypeError, ValueError):
            confidence = 50.0
        confidence = min(max(confidence, 0.0), 100.0)
        questions = [str(q).strip() for q in data.get("verification_questions", []) if str(q).strip()]
        return confidence / 100.0, questions[:3]

    def cove_answers(self, questions: list[str]) -> list[str]:
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
        data = self.llm.chat_json(
            self.cfg.generator,
            [{"role": "user", "content": P.COVE_ANSWER.format(questions=numbered)}],
            max_tokens=400,
        )
        answers = [clean_text(str(a)) for a in data.get("answers", [])]
        return (answers + ["unknown"] * len(questions))[: len(questions)]

    def consistency(self, question: str, sentences: list[str], sample: str) -> tuple[list[str], str]:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
        data = self.llm.chat_json(
            self.cfg.judge,
            [{"role": "user", "content": P.CONSISTENCY_JUDGE.format(
                question=question, sample=sample, sentences=numbered)}],
        )
        by_id = {}
        for item in data.get("labels", []):
            try:
                by_id[int(item.get("id"))] = _label(item.get("label"), SENTENCE_SCORE, "neutral")
            except (TypeError, ValueError):
                continue
        labels = [by_id.get(i, "neutral") for i in range(1, len(sentences) + 1)]
        return labels, _label(data.get("overall"), OVERALL_SCORE, "partial")

    def cove_judge(self, question: str, answer: str, questions: list[str], answers: list[str]):
        qa = "\n".join(f"{i}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(zip(questions, answers), 1))
        data = self.llm.chat_json(
            self.cfg.judge,
            [{"role": "user", "content": P.COVE_JUDGE.format(question=question, answer=answer, qa=qa)}],
        )
        by_id = {}
        for item in data.get("checks", []):
            try:
                by_id[int(item.get("id"))] = (
                    _label(item.get("label"), CHECK_SCORE, "unclear"), str(item.get("note", "")).strip())
            except (TypeError, ValueError):
                continue
        checks = []
        for i, (q, a) in enumerate(zip(questions, answers), 1):
            label, note = by_id.get(i, ("unclear", ""))
            checks.append({"question": q, "independent_answer": a, "label": label, "note": note})
        verdict = _label(data.get("verdict"), VERDICT_SCORE, "partial")
        issues = [str(x).strip() for x in data.get("issues", []) if str(x).strip()]
        return checks, verdict, issues


# --------------------------------------------------------------------- scoring
def classify(final: float, verdict: str, consistency: float, cfg: DetectionConfig) -> str:
    if verdict == "abstained" and consistency >= 0.6:
        return "abstained"
    if final >= cfg.high:
        return "reliable"
    if final >= cfg.low:
        return "uncertain"
    return "hallucination"


def run_detection(
    llm: LLM,
    question: str,
    cfg: DetectionConfig,
    calibrator: LogisticCalibrator | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> dict:
    say = progress or (lambda stage, msg: None)
    det = Detector(llm, cfg)
    usage_before = llm.usage()
    timings: dict[str, float] = {}
    t_start = time.time()

    # 1. Main answer --------------------------------------------------------
    say("answer", f"Generating the main answer with {cfg.generator}")
    t = time.time()
    answer = det.answer(question, temperature=0.0)
    sentences = split_sentences(answer)
    timings["answer"] = time.time() - t

    # 2a + 3a. Stochastic samples and self-verification plan, in parallel ----
    say("sample", f"Drawing {cfg.n_samples} stochastic samples (T={cfg.sample_temperature}) and planning self-verification")
    t = time.time()
    with ThreadPoolExecutor(max_workers=cfg.n_samples + 1) as pool:
        sample_futures = [pool.submit(det.answer, question, cfg.sample_temperature) for _ in range(cfg.n_samples)]
        plan_future = pool.submit(det.self_check, question, answer)
        samples = [f.result() for f in sample_futures]
        verbalized, vqs = plan_future.result()
    timings["sampling"] = time.time() - t

    # 2b + 3b. Sentence-level consistency and independent CoVe answers -------
    say("check", "Cross-checking every sentence against each sample, and answering the fact-check questions independently")
    t = time.time()
    with ThreadPoolExecutor(max_workers=cfg.n_samples + 1) as pool:
        cons_futures = [pool.submit(det.consistency, question, sentences, s) for s in samples]
        cove_future = pool.submit(det.cove_answers, vqs) if vqs else None
        consistency_results = [f.result() for f in cons_futures]
        vq_answers = cove_future.result() if cove_future else []
    timings["consistency"] = time.time() - t

    # 3c. Verification verdict ------------------------------------------------
    say("verify", "Comparing the independent fact-checks with the original answer")
    t = time.time()
    if vqs:
        checks, verdict, issues = det.cove_judge(question, answer, vqs, vq_answers)
    else:
        checks, verdict, issues = [], "partial", ["The model produced no fact-check questions"]
    timings["verification"] = time.time() - t

    # 4. Scoring -------------------------------------------------------------
    say("calibrate", "Combining the three signals into a calibrated confidence")
    sentence_rows = []
    for i, sentence in enumerate(sentences):
        labels = [labels_i[i] for labels_i, _ in consistency_results]
        sentence_rows.append({
            "text": sentence,
            "labels": labels,
            "score": _mean(SENTENCE_SCORE[l] for l in labels),
        })
    overall_labels = [overall for _, overall in consistency_results]
    sentence_mean = _mean(r["score"] for r in sentence_rows)
    agreement = _mean(OVERALL_SCORE[o] for o in overall_labels)
    consistency = 0.5 * sentence_mean + 0.5 * agreement

    check_mean = _mean((CHECK_SCORE[c["label"]] for c in checks), default=VERDICT_SCORE[verdict])
    verification = 0.5 * check_mean + 0.5 * VERDICT_SCORE[verdict]

    features = {"consistency": consistency, "verification": verification, "verbalized": verbalized}
    w = cfg.weights
    heuristic = (w[0] * consistency + w[1] * verification + w[2] * verbalized) / sum(w)
    calibrated = calibrator.predict_one(features) if calibrator else None
    final = calibrated if calibrated is not None else heuristic

    usage_after = llm.usage()
    return {
        "question": question,
        "answer": answer,
        "sentences": sentence_rows,
        "samples": [{"text": s, "overall": o} for s, o in zip(samples, overall_labels)],
        "verification": {"checks": checks, "verdict": verdict, "issues": issues},
        "features": features,
        "details": {"sentence_mean": sentence_mean, "agreement": agreement, "check_mean": check_mean},
        "heuristic": heuristic,
        "calibrated": calibrated,
        "final": final,
        "status": classify(final, verdict, consistency, cfg),
        "config": {"generator": cfg.generator, "judge": cfg.judge, "n_samples": cfg.n_samples,
                   "weights": list(cfg.weights), "low": cfg.low, "high": cfg.high,
                   "calibrator": bool(calibrator)},
        "timings": timings,
        "elapsed": time.time() - t_start,
        "usage": {k: usage_after[k] - usage_before[k] for k in usage_after},
    }
