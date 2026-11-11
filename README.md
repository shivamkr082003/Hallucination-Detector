# VeriLens

**AI-Driven Uncertainty Calibration for Hallucination Detection and Self-Verification in Large Language Models**

VeriLens checks whether an LLM's answer can be trusted. It estimates uncertainty from three independent
signals, combines them with a learned calibrator, and highlights which sentences are likely hallucinated.

## How it works

| Step | What happens |
|---|---|
| 1. Generate | The model under test answers the question (greedy decoding). |
| 2. Sample | The same question is asked N more times at temperature 1.0. |
| 3. Cross-check | Each sentence of the answer is labelled against every sample: supported / neutral / contradicted (SelfCheckGPT-style). |
| 4. Self-verify | Factored Chain-of-Verification: the model writes fact-check questions, answers them **without seeing its original answer**, and a comparator checks agreement. |
| 5. Calibrate | Semantic consistency (C), verification (V) and verbalized confidence (P) are combined by a logistic calibrator: `P(correct) = σ(w·[C,V,P] + b)`. |

The calibrator is trained on a labelled question set (`data/eval_questions.json`), and calibration is
measured with ECE, the Brier score and AUROC using 5-fold cross-validation.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your Groq key in .env
```

## Run

```bash
streamlit run app.py        # opens http://localhost:8501
python evaluate.py          # (optional) re-run evaluation + retrain the calibrator
```

## Project structure

```
app.py                  Streamlit UI (Analyze / Evaluation / Method tabs)
evaluate.py             Offline evaluation + calibrator training (CLI)
core/llm.py             Groq client: retries, rate-limit handling, JSON mode, usage tracking
core/prompts.py         All prompts
core/pipeline.py        Detection pipeline and scoring
core/calibration.py     ECE, Brier, AUROC, reliability bins, logistic calibrator
core/evaluation.py      Evaluation runner and report builder
core/ui.py              Styling and HTML components
data/eval_questions.json  36 labelled questions (well-known, obscure, false-premise)
results/                Evaluation records, report and trained calibrator
```

## Models

- **Generator** (model under test): `qwen/qwen3.8-27b` by default. It hallucinates noticeably on obscure facts.
- **Judge** (text comparator only): `openai/gpt-oss-120b`. It compares texts and never decides what is true.

Both can be changed in the sidebar. The Groq free tier allows about 30 requests/minute and 1,000
requests/day per model. One analysis with 4 samples uses about 12 calls.

## References

1. Manakul et al. *SelfCheckGPT*, EMNLP 2023.
2. Dhuliawala et al. *Chain-of-Verification Reduces Hallucination in LLMs*, 2023.
3. Kuhn et al. *Semantic Uncertainty*, ICLR 2023. Farquhar et al. *Semantic entropy*, Nature 2024.
4. Guo et al. *On Calibration of Modern Neural Networks*, ICML 2017.
5. Xiong et al. *Can LLMs Express Their Uncertainty?*, ICLR 2024.
# Hallucination-Detector
