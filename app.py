"""VeriLens: uncertainty calibration for hallucination detection and self-verification in LLMs.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from core import ui  # noqa: E402
from core.calibration import LogisticCalibrator  # noqa: E402
from core.evaluation import (CALIBRATOR_PATH, CATEGORY_LABELS, build_report, load_dataset,  # noqa: E402
                             load_report, run_evaluation)
from core.llm import LLM, PREFERRED_GENERATORS, PREFERRED_JUDGES, pick_model  # noqa: E402
from core.pipeline import DetectionConfig, run_detection  # noqa: E402

st.set_page_config(page_title="VeriLens · LLM Hallucination Detector", page_icon="🔍",
                   layout="wide", initial_sidebar_state="collapsed")
ui.inject_css()

# (chip label, full question): one well-known fact, two obscure facts, one false premise.
EXAMPLES = [
    ("Capital of Australia", "What is the capital of Australia?"),
    ("Who wrote Raag Darbari?", "Who wrote the Hindi novel Raag Darbari, and which award did it win?"),
    ("First Jnanpith winner", "Who was the first recipient of the Jnanpith Award?"),
    ("Gandhi's Nobel Prize", "In which year did Mahatma Gandhi win the Nobel Peace Prize?"),
]

PLOT_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c9cdd8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
)
GRID = dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
METHOD_COLORS = {
    "Verbalized confidence": "#a78bfa",
    "Semantic consistency": "#22d3ee",
    "Self-verification (CoVe)": "#60a5fa",
    "Heuristic combination": "#fbbf24",
    "Learned calibrator (5-fold CV)": "#34d399",
}

# --------------------------------------------------------------- state
for key, default in {"question": "", "result": None, "auto_run": False}.items():
    st.session_state.setdefault(key, default)


def _use_example(q: str) -> None:
    st.session_state.question = q
    st.session_state.auto_run = True


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str) -> LLM:
    return LLM(api_key)


@st.cache_data(ttl=900, show_spinner=False)
def get_models(api_key: str) -> list[str]:
    return get_llm(api_key).list_chat_models()


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    if ("rate_limit" in text or "429" in text) and "per day" in text.lower():
        return ("The daily Groq quota for this model is used up. Pick a different generator or judge model "
                "in Settings, or try again later.")
    if "rate_limit" in text or "429" in text:
        return "Groq rate limit reached. Wait about a minute and try again, or lower the number of samples."
    if "401" in text or "invalid_api_key" in text.lower():
        return "The Groq API key is invalid. Check GROQ_API_KEY in your .env file."
    if "model_not_found" in text or "decommissioned" in text:
        return "The selected model is not available on Groq any more. Pick another model in Settings."
    return f"Something went wrong: {text[:300]}"


# ------------------------------------------------------------ top bar
api_key = os.getenv("GROQ_API_KEY", "")
models: list[str] = []
model_error = None
if api_key:
    try:
        models = get_models(api_key)
    except Exception as exc:  # noqa: BLE001
        model_error = friendly_error(exc)
calibrator = LogisticCalibrator.load(CALIBRATOR_PATH)

nav_brand, nav_links, nav_settings = st.columns([1, 1.3, 1], vertical_alignment="center")
with nav_brand:
    ui.md(ui.nav_brand(bool(models)))
with nav_links, st.container(horizontal=True, horizontal_alignment="center"):
    page = st.segmented_control("Page", ["Analyze", "Evaluation", "Method"], default="Analyze",
                                required=True, key="page", label_visibility="collapsed") or "Analyze"
with nav_settings, st.container(horizontal=True, horizontal_alignment="right"):
    with st.popover("Settings", icon=":material/tune:", key="settings"):
        ui.md(ui.connection_line(bool(models), len(models)))
        if model_error:
            st.error(model_error)
        if not api_key:
            api_key = st.text_input("Groq API key", type="password", help="Or set GROQ_API_KEY in .env")
        options = models or PREFERRED_GENERATORS
        generator = st.selectbox("Generator · model under test", options,
                                 index=options.index(pick_model(options, PREFERRED_GENERATORS)))
        judge = st.selectbox("Judge · text comparator", options,
                             index=options.index(pick_model(options, PREFERRED_JUDGES)),
                             help="Only compares texts (like an NLI model). It never decides what is true.")
        n_samples = st.slider("Re-generated samples (N)", 2, 6, 4,
                              help="More samples give a better uncertainty estimate but use more API calls.")
        low, high = st.slider("Confidence bands", 0.0, 1.0, (0.45, 0.75), 0.05,
                              help="Below the first value: hallucination. Above the second: reliable.")
        use_calibrator = st.toggle("Use learned calibrator", value=calibrator is not None,
                                   disabled=calibrator is None,
                                   help="Trained by evaluate.py on labelled questions."
                                   if calibrator else "Run the evaluation first to train it.")
        if calibrator is not None and calibrator.meta.get("generator") not in (None, generator):
            st.caption(f"⚠️ Calibrator was trained on {calibrator.meta.get('generator')}.")
        st.caption("Heuristic weights (used when the calibrator is off)")
        w1, w2, w3 = st.columns(3)
        w_c = w1.number_input("C", 0.0, 1.0, 0.5, 0.05, help="Semantic consistency")
        w_v = w2.number_input("V", 0.0, 1.0, 0.3, 0.05, help="Self-verification")
        w_p = w3.number_input("P", 0.0, 1.0, 0.2, 0.05, help="Verbalized confidence")
        if api_key:
            ui.md(ui.usage_stats(get_llm(api_key).usage()))
weights = (w_c, w_v, w_p) if (w_c + w_v + w_p) > 0 else (0.5, 0.3, 0.2)

cfg = DetectionConfig(generator=generator, judge=judge, n_samples=n_samples,
                      weights=weights, low=min(low, high), high=max(low, high))


def heatmap(res: dict) -> go.Figure:
    score = {"supported": 1.0, "neutral": 0.5, "contradicted": 0.0}
    sents = res["sentences"]
    z = [[score[l] for l in s["labels"]] for s in sents]
    text = [[l for l in s["labels"]] for s in sents]
    y = [f"S{i + 1}" for i in range(len(sents))]
    x = [f"Sample {j + 1}" for j in range(len(res["samples"]))]
    hover = [[f"<b>{y[i]}</b>: {sents[i]['text'][:90]}<br>{x[j]} → {text[i][j]}" for j in range(len(x))]
             for i in range(len(y))]
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, text=text, texttemplate="%{text}", hovertext=hover, hoverinfo="text",
        zmin=0, zmax=1, xgap=4, ygap=4, showscale=False,
        colorscale=[[0, "#7f1d1d"], [0.5, "#78350f"], [1, "#065f46"]],
        textfont=dict(color="#fff", size=12),
    ))
    fig.update_layout(**PLOT_LAYOUT, height=90 + 56 * len(y))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    return fig


def render_result(res: dict) -> None:
    ui.md(ui.result_card(res))
    with st.expander("See how VeriLens decided"):
        ui.md(ui.signals_card(res))
        t1, t2, t3, t4 = st.tabs(["Re-generated answers", "Fact-checks", "Sentence matrix", "Raw"])
        with t1:
            ui.md(ui.samples_grid(res))
        with t2:
            ui.md(ui.cove_card(res))
        with t3:
            st.caption("How each re-generated answer judges each sentence. "
                       "A red row is a sentence the model cannot repeat consistently.")
            st.plotly_chart(heatmap(res), width="stretch", config={"displayModeBar": False})
            for i, s in enumerate(res["sentences"], 1):
                st.markdown(f"<span style='color:#8b91a5'>S{i}</span> &nbsp;{ui.esc(s['text'])}",
                            unsafe_allow_html=True)
        with t4:
            st.json(res, expanded=False)


# --------------------------------------------------------------- analyze
if page == "Analyze":
    with st.container(key="narrow"):
        hero_slot = st.empty()
        if not st.session_state.result:
            with hero_slot.container():
                ui.md(ui.home_hero())
        with st.container(key="ask"):
            with st.form("ask_form", border=False):
                c_input, c_button = st.columns([5, 1.25], vertical_alignment="bottom", gap="small")
                with c_input:
                    st.text_input("Question", key="question", label_visibility="collapsed",
                                  placeholder="Ask anything, e.g. Who wrote the Hindi novel Raag Darbari?")
                with c_button:
                    submitted = st.form_submit_button("Analyze", type="primary", width="stretch",
                                                      disabled=not api_key)

        try_slot = st.empty()
        if not st.session_state.result:
            try_slot.markdown('<div class="vl-try">Try an example</div>', unsafe_allow_html=True)
        with st.container(key="chips", horizontal=True, horizontal_alignment="center", gap="small"):
            for i, (label, q) in enumerate(EXAMPLES):
                st.button(label, key=f"chip_{i}", on_click=_use_example, args=(q,), disabled=not api_key)

        run_now = submitted or st.session_state.auto_run
        st.session_state.auto_run = False
        if run_now:
            question = st.session_state.question.strip()
            if not question:
                st.warning("Type a question first.")
            else:
                icons = {"answer": "✍️", "sample": "🎲", "check": "🔎", "verify": "🧪", "calibrate": "🎯"}
                with st.status("Checking the answer…", expanded=True) as status:
                    try:
                        result = run_detection(
                            get_llm(api_key), question, cfg,
                            calibrator=calibrator if use_calibrator else None,
                            progress=lambda stage, msg: st.write(f"{icons.get(stage, '•')}  {msg}"),
                        )
                        status.update(label=f"Checked in {result['elapsed']:.1f}s", state="complete",
                                      expanded=False)
                        st.session_state.result = result
                        hero_slot.empty()
                        try_slot.empty()
                    except Exception as exc:  # noqa: BLE001
                        status.update(label="Something went wrong", state="error", expanded=True)
                        st.error(friendly_error(exc))

        if st.session_state.result:
            render_result(st.session_state.result)
        else:
            ui.md(ui.footer())


# ------------------------------------------------------------ evaluation
def reliability_figure(report: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color="rgba(255,255,255,0.35)", dash="dash", width=1.5)))
    for name in ["Verbalized confidence", "Heuristic combination", "Learned calibrator (5-fold CV)"]:
        bins = report["methods"][name]["reliability"]
        fig.add_trace(go.Scatter(
            x=[b["confidence"] for b in bins], y=[b["accuracy"] for b in bins], mode="lines+markers", name=name,
            line=dict(color=METHOD_COLORS[name], width=2.5),
            marker=dict(size=[7 + 2 * b["count"] ** 0.5 for b in bins], line=dict(width=0)),
            hovertemplate="confidence %{x:.2f}<br>accuracy %{y:.2f}<br>%{customdata} questions<extra>" + name + "</extra>",
            customdata=[b["count"] for b in bins],
        ))
    fig.update_layout(**{**PLOT_LAYOUT, "legend": dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                                                       bgcolor="rgba(7,9,15,0.75)", font=dict(size=11))},
                      height=400)
    fig.update_xaxes(title="Predicted confidence", range=[-0.02, 1.02], **GRID)
    fig.update_yaxes(title="Observed accuracy", range=[-0.02, 1.05], **GRID)
    return fig


def comparison_figure(report: dict) -> go.Figure:
    names = list(report["methods"])
    au = [report["methods"][n]["auroc"] or 0 for n in names]
    ec = [report["methods"][n]["ece"] for n in names]
    short = [n.replace(" (5-fold CV)", "").replace(" (CoVe)", "").replace(" ", "<br>", 1) for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=short, y=au, name="AUROC ↑", marker_color="#8b5cf6",
                         text=[f"{v:.2f}" for v in au], textposition="outside"))
    fig.add_trace(go.Bar(x=short, y=ec, name="ECE ↓", marker_color="#22d3ee",
                         text=[f"{v:.2f}" for v in ec], textposition="outside"))
    fig.update_layout(**PLOT_LAYOUT, height=400, barmode="group", bargap=0.3)
    fig.update_yaxes(range=[0, 1.15], **GRID)
    fig.update_xaxes(tickangle=0, **GRID)
    return fig


def distribution_figure(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for correct, name, color in [(True, "Correct answers", "#34d399"), (False, "Wrong answers", "#f87171")]:
        sub = [r for r in rows if r["correct"] == correct]
        for key, label in [("verbalized", "Verbalized"), ("calibrated", "Calibrated")]:
            fig.add_trace(go.Box(
                y=[r[key] for r in sub], x=[label] * len(sub), name=name, marker_color=color,
                boxpoints="all", jitter=0.45, pointpos=0, line=dict(width=1.5), fillcolor=color + "22",
                legendgroup=name, showlegend=key == "verbalized",
                hovertext=[r["question"] for r in sub], hoverinfo="y+text",
            ))
    fig.update_layout(**PLOT_LAYOUT, height=380, boxmode="group")
    fig.update_yaxes(title="Confidence", range=[-0.05, 1.08], **GRID)
    return fig


if page == "Evaluation":
    report = load_report()
    if not report:
        ui.md(ui.empty_state(
            "No evaluation results yet",
            "Run the detector on the labelled question set to measure calibration and train the learned calibrator.",
            "python evaluate.py",
        ))
    else:
        m = report["methods"]
        learned, verbal = m["Learned calibrator (5-fold CV)"], m["Verbalized confidence"]
        ui.md(ui.section(
            "How well does confidence predict correctness?",
            f"{report['n']} labelled questions · generator {report['generator']} · judge {report['judge']} · "
            f"{report['n_samples']} samples per question · run {report['created'].replace('T', ' ')}"))

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            ui.md(ui.metric("Generator accuracy", f"{report['accuracy'] * 100:.0f}%",
                            f"{sum(r['correct'] for r in report['rows'])} of {report['n']} answers correct"))
        with c2:
            au_l, au_v = learned["auroc"] or 0, verbal["auroc"] or 0
            ui.md(ui.metric("AUROC ↑", f"{au_l:.2f}", f"vs <b>{au_v:.2f}</b> for verbalized confidence", "#a78bfa"))
        with c3:
            ui.md(ui.metric("ECE ↓", f"{learned['ece']:.2f}", f"vs <b>{verbal['ece']:.2f}</b> for verbalized confidence", "#22d3ee"))
        with c4:
            ui.md(ui.metric("Brier score ↓", f"{learned['brier']:.2f}", f"vs <b>{verbal['brier']:.2f}</b> for verbalized confidence", "#34d399"))

        obscure = report["by_category"].get("obscure")
        if obscure:
            ui.md(f"""<div class="vl-card" style="margin-top:16px;border-color:rgba(139,92,246,0.35);
            background:linear-gradient(135deg,rgba(139,92,246,0.12),rgba(34,211,238,0.05) 60%,rgba(255,255,255,0.01))">
            <div class="vl-card-title"><span>Key finding · {obscure['n']} obscure-fact questions</span></div>
            <div style="display:flex;gap:28px;flex-wrap:wrap;align-items:flex-end">
              <div><div class="vl-metric-v" style="color:#fff">{obscure['accuracy'] * 100:.0f}%</div>
                   <div class="vl-metric-d">actual accuracy</div></div>
              <div><div class="vl-metric-v" style="color:#f87171">{obscure['mean_verbalized'] * 100:.0f}%</div>
                   <div class="vl-metric-d">what the model claims (verbalized)</div></div>
              <div><div class="vl-metric-v" style="color:#34d399">{obscure['mean_calibrated'] * 100:.0f}%</div>
                   <div class="vl-metric-d">VeriLens calibrated confidence</div></div>
              <div style="flex:1;min-width:240px;color:#cbd5e1;font-size:.92rem;line-height:1.6">
                The model overstates its confidence by
                <b style="color:#f87171">{(obscure['mean_verbalized'] - obscure['accuracy']) * 100:.0f} points</b>.
                The calibrated score is within
                <b style="color:#34d399">{abs(obscure['mean_calibrated'] - obscure['accuracy']) * 100:.0f} points</b>
                of the real accuracy.</div>
            </div></div>""")

        st.write("")
        g1, g2 = st.columns(2, gap="medium")
        with g1:
            ui.md('<div class="vl-card-title" style="margin:10px 0 0 0"><span>Reliability diagram</span></div>')
            st.plotly_chart(reliability_figure(report), width="stretch", config={"displayModeBar": False})
            st.caption("Points on the dashed line mean the confidence matches real accuracy. "
                       "Points below the line are overconfident.")
        with g2:
            ui.md('<div class="vl-card-title" style="margin:10px 0 0 0"><span>Method comparison</span></div>')
            st.plotly_chart(comparison_figure(report), width="stretch", config={"displayModeBar": False})
            st.caption("AUROC: how well the score separates right from wrong answers (higher is better). "
                       "ECE: average gap between confidence and accuracy (lower is better).")

        g3, g4 = st.columns([1.1, 1], gap="medium")
        with g3:
            ui.md('<div class="vl-card-title" style="margin:10px 0 0 0"><span>Confidence on correct vs wrong answers</span></div>')
            st.plotly_chart(distribution_figure(report["rows"]), width="stretch", config={"displayModeBar": False})
        with g4:
            ui.md('<div class="vl-card-title" style="margin:10px 0 0 0"><span>By question type</span></div>')
            cat_df = pd.DataFrame([
                {"Type": v["label"], "N": v["n"], "Accuracy": v["accuracy"],
                 "Claimed": v["mean_verbalized"], "Calibrated": v["mean_calibrated"]}
                for v in report["by_category"].values()])
            st.dataframe(cat_df, hide_index=True, width="stretch", column_config={
                "Type": st.column_config.TextColumn(width="small"),
                "N": st.column_config.NumberColumn(width=40),
                "Accuracy": st.column_config.NumberColumn(format="percent"),
                "Claimed": st.column_config.NumberColumn(format="percent", help="Mean verbalized confidence"),
                "Calibrated": st.column_config.NumberColumn(format="percent", help="Mean calibrated confidence"),
            })
            st.caption("Claimed = what the model says about itself. Calibrated = VeriLens. "
                       "The closer a value is to Accuracy, the better calibrated it is.")
            ui.md(ui.side_label("Learned calibrator weights"))
            cal = report["calibrator"]
            wtxt = " + ".join(f"{w:+.2f}·{f[:4]}" for w, f in zip(cal["weights"], cal["features"]))
            ui.md(f'<div class="vl-formula">P(correct) = σ({wtxt} {cal["bias"]:+.2f})</div>')

        ui.md(ui.section("Per-question results", "Calibrated values are out-of-fold, so the model never saw that question during training."))
        df = pd.DataFrame(report["rows"])
        df["Type"] = df["category"].map(CATEGORY_LABELS)
        df["Correct"] = df["correct"].map({True: "✅", False: "❌"})
        df = df[["Type", "question", "answer", "Correct", "verbalized", "consistency", "verification", "calibrated", "reference"]]
        df.columns = ["Type", "Question", "Model answer", "Correct", "Verbalized", "Consistency", "Verification", "Calibrated", "Reference"]
        pct = dict(format="percent", min_value=0, max_value=1)
        st.dataframe(df, hide_index=True, width="stretch", height=420, column_config={
            "Question": st.column_config.TextColumn(width="medium"),
            "Model answer": st.column_config.TextColumn(width="large"),
            "Verbalized": st.column_config.ProgressColumn(**pct),
            "Consistency": st.column_config.ProgressColumn(**pct),
            "Verification": st.column_config.ProgressColumn(**pct),
            "Calibrated": st.column_config.ProgressColumn(**pct),
        })

    with st.expander("Run the evaluation again from the app"):
        st.caption(f"Evaluates all {len(load_dataset())} questions with the generator and judge selected in Settings "
                   "(3 samples each, about 10 calls per question). This takes roughly 5 to 15 minutes because of "
                   "Groq rate limits. Finished questions are saved, so an interrupted run can resume.")
        fresh = st.checkbox("Start from scratch (discard saved records)")
        if st.button("Run evaluation", disabled=not api_key):
            bar = st.progress(0.0, text="Starting…")
            eval_cfg = DetectionConfig(generator=generator, judge=judge, n_samples=3, weights=weights,
                                       low=cfg.low, high=cfg.high)
            try:
                records = run_evaluation(
                    get_llm(api_key), eval_cfg, load_dataset(), fresh=fresh,
                    progress=lambda i, n, rec: bar.progress(i / n, text=f"{i}/{n} · {rec['question'][:70] if rec else ''}"))
                build_report(records, eval_cfg)
                st.success("Evaluation finished and the calibrator was retrained.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(friendly_error(exc))


# ---------------------------------------------------------------- method
if page == "Method":
    ui.md(ui.section("How VeriLens works", "AI-Driven Uncertainty Calibration for Hallucination Detection and Self-Verification in Large Language Models"))
    ui.md(ui.section("Pipeline", "Five steps run for every question. Steps 2 and 4 run in parallel to keep latency low."))
    ui.md(ui.pipeline_diagram())

    ui.md(ui.section("The three uncertainty signals"))
    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        ui.md("""<div class="vl-card"><div class="vl-card-title"><span>C · Semantic consistency</span></div>
        <div style="color:#cbd5e1;font-size:.92rem;line-height:1.6">If a model knows a fact, it repeats it every time.
        If it is guessing, the facts change between samples. Each sentence of the answer is labelled against each
        sample by a comparator model, an approach based on SelfCheckGPT.</div>
        <div class="vl-formula">s_i = mean_j score(sentence_i, sample_j)<br>supported=1 · neutral=0.5 · contradicted=0<br>
        C = ½·mean_i(s_i) + ½·agreement</div></div>""")
    with m2:
        ui.md("""<div class="vl-card"><div class="vl-card-title"><span>V · Self-verification</span></div>
        <div style="color:#cbd5e1;font-size:.92rem;line-height:1.6">Factored Chain-of-Verification: the model plans
        fact-check questions about its own answer, then answers them in a fresh context without seeing that answer.
        This stops it from simply copying its earlier mistake.</div>
        <div class="vl-formula">V = ½·mean(check) + ½·verdict<br>consistent=1 · unclear=0.5 · inconsistent=0<br>
        supported=1 · partial=0.5 · contradicted=0</div></div>""")
    with m3:
        ui.md("""<div class="vl-card"><div class="vl-card-title"><span>P · Verbalized confidence</span></div>
        <div style="color:#cbd5e1;font-size:.92rem;line-height:1.6">The model is asked how confident it is. This
        baseline is cheap but usually overconfident: models often report 90%+ even when they are wrong. It is the
        baseline that the calibrated score is compared against.</div>
        <div class="vl-formula">P = self-reported confidence / 100</div></div>""")

    ui.md(ui.section("Calibration"))
    ui.md("""<div class="vl-card"><div style="color:#cbd5e1;font-size:.95rem;line-height:1.7">
    A confidence score is <b style="color:#fff">calibrated</b> when it matches reality: among answers given 80%
    confidence, about 80% should be correct. VeriLens learns a logistic calibrator (multi-feature Platt scaling) on a
    labelled question set. It learns how much to trust each signal and maps the result to a probability. Quality is
    measured with <b style="color:#fff">ECE</b> (Expected Calibration Error), the <b style="color:#fff">Brier score</b>
    and <b style="color:#fff">AUROC</b>, using 5-fold cross-validation so the scores are not measured on training data.</div>
    <div class="vl-formula" style="margin-top:14px">P(correct | C, V, P) = σ(w₁·C + w₂·V + w₃·P + b)
    &nbsp;&nbsp;·&nbsp;&nbsp; ECE = Σ_b (n_b / N) · |acc(b) − conf(b)|</div></div>""")

    ui.md(ui.section("References"))
    ui.md("""<div class="vl-card vl-ref">
    1. Manakul, Liusie, Gales. <i>SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs.</i> EMNLP 2023.<br>
    2. Dhuliawala et al. <i>Chain-of-Verification Reduces Hallucination in Large Language Models.</i> 2023.<br>
    3. Kuhn, Gal, Farquhar. <i>Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG.</i> ICLR 2023.<br>
    4. Farquhar et al. <i>Detecting hallucinations in large language models using semantic entropy.</i> Nature 2024.<br>
    5. Guo, Pleiss, Sun, Weinberger. <i>On Calibration of Modern Neural Networks.</i> ICML 2017.<br>
    6. Xiong et al. <i>Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs.</i> ICLR 2024.<br>
    7. Kadavath et al. <i>Language Models (Mostly) Know What They Know.</i> 2022.
    </div>""")
