"""Styling and HTML components for the Streamlit app."""
from __future__ import annotations

import html
import math

import streamlit as st

STATUS = {
    "reliable": {"label": "Reliable", "color": "#34d399", "icon": "✓",
                 "text": "The answer stays consistent across samples and survives self-verification."},
    "uncertain": {"label": "Uncertain", "color": "#fbbf24", "icon": "!",
                  "text": "Some claims change across samples or fail verification. Double-check before trusting."},
    "hallucination": {"label": "Likely hallucination", "color": "#f87171", "icon": "✕",
                      "text": "The model contradicts itself. Key claims are probably fabricated."},
    "abstained": {"label": "Model abstained", "color": "#60a5fa", "icon": "∅",
                  "text": "The model declined or flagged a false premise, which is a safe response."},
}

LABEL_COLORS = {
    "supported": "#34d399", "agree": "#34d399", "consistent": "#34d399",
    "neutral": "#94a3b8", "partial": "#fbbf24", "unclear": "#94a3b8",
    "contradicted": "#f87171", "disagree": "#f87171", "inconsistent": "#f87171",
    "abstained": "#60a5fa",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
  --bg: #07090f; --surface: rgba(255,255,255,0.035); --surface-2: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.08); --border-strong: rgba(255,255,255,0.14);
  --text: #e7e9f0; --muted: #8b91a5; --faint: #5d6377;
  --violet: #8b5cf6; --indigo: #6366f1; --cyan: #22d3ee;
  --green: #34d399; --amber: #fbbf24; --red: #f87171; --blue: #60a5fa;
}
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, select {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
  background:
    radial-gradient(1100px 560px at 8% -12%, rgba(139,92,246,0.20), transparent 60%),
    radial-gradient(900px 520px at 100% -5%, rgba(34,211,238,0.11), transparent 60%),
    var(--bg);
  color: var(--text);
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }

/* Sidebar */
[data-testid="stSidebar"] { background: rgba(9,11,18,0.92); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
[data-testid="stSidebar"] label p { font-size: 0.82rem; color: var(--muted); font-weight: 500; }

/* Inputs */
[data-testid="stTextArea"] textarea {
  background: rgba(255,255,255,0.025) !important; border-radius: 16px !important;
  font-size: 1.05rem !important; line-height: 1.55 !important; padding: 16px 18px !important;
  color: var(--text) !important;
}
[data-testid="stTextArea"] > div > div { border-radius: 16px !important; border-color: var(--border-strong) !important; }
[data-testid="stTextArea"] > div > div:focus-within { border-color: var(--violet) !important; box-shadow: 0 0 0 3px rgba(139,92,246,0.18); }

/* Buttons */
.stButton > button {
  border-radius: 12px; border: 1px solid var(--border); background: var(--surface);
  color: var(--text); font-weight: 500; transition: all .18s ease; text-align: left;
}
.stButton > button:not([kind="primary"]) { justify-content: flex-start; padding: 0.6rem 0.9rem; min-height: 3.1rem; height: 100%; }
.stButton > button:not([kind="primary"]) > div,
.stButton > button:not([kind="primary"]) span,
.stButton > button:not([kind="primary"]) .stMarkdownEllipsis,
.stButton > button:not([kind="primary"]) p {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
  text-align: left !important; justify-content: flex-start !important; max-width: 100%;
}
.stButton > button:not([kind="primary"]) p { font-size: .9rem; line-height: 1.4; }
.stButton > button:hover { border-color: rgba(139,92,246,0.55); background: rgba(139,92,246,0.08); color: #fff; transform: translateY(-1px); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 55%, #0ea5e9 100%) !important;
  border: 0 !important; color: #fff !important; font-weight: 650 !important; letter-spacing: .01em;
  box-shadow: 0 10px 30px -8px rgba(139,92,246,0.65); padding: 0.62rem 1.2rem; text-align: center;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  filter: brightness(1.08); box-shadow: 0 14px 36px -8px rgba(139,92,246,0.8);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px; background: var(--surface); padding: 5px; border-radius: 14px;
  border: 1px solid var(--border); width: fit-content; max-width: 100%; overflow-x: auto;
}
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 18px; height: auto; color: var(--muted); }
.stTabs [data-baseweb="tab"] p { font-weight: 550; font-size: .92rem; }
.stTabs [aria-selected="true"] { background: rgba(139,92,246,0.18) !important; color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
/* Newer Streamlit (react-aria tabs) */
.stTabs [role="tablist"] {
  gap: 4px; background: var(--surface); padding: 5px; border-radius: 14px; border: 1px solid var(--border);
  width: fit-content; max-width: 100%; overflow-x: auto; border-bottom: 1px solid var(--border) !important;
}
.stTabs [role="tablist"] [data-testid="stTab"] { border-radius: 10px; padding: 8px 18px; color: var(--muted); border-bottom: 0 !important; }
.stTabs [role="tablist"] [data-testid="stTab"][data-selected="true"],
.stTabs [role="tablist"] [data-testid="stTab"][aria-selected="true"] { background: rgba(139,92,246,0.18); color: #fff; }
.stTabs [role="tablist"] [data-testid="stTab"] p { font-weight: 550; font-size: .92rem; }
.stTabs [role="tablist"] [data-testid="stTab"]::after, .stTabs [role="tablist"] + div[aria-hidden] { display: none !important; }

/* Status / expander / dataframe */
[data-testid="stExpander"] details, [data-testid="stStatusWidget"] { border-radius: 14px; }
[data-testid="stExpander"] details { border: 1px solid var(--border); background: var(--surface); }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }

/* ---------- Custom components ---------- */
.vl-hero { margin: 0 0 1.6rem 0; }
.vl-eyebrow {
  display: inline-flex; align-items: center; gap: 8px; font-size: .72rem; font-weight: 650;
  letter-spacing: .14em; text-transform: uppercase; color: #c4b5fd;
  background: rgba(139,92,246,0.12); border: 1px solid rgba(139,92,246,0.3);
  padding: 6px 12px; border-radius: 999px;
}
.vl-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); box-shadow: 0 0 10px var(--green); }
.vl-title {
  font-size: clamp(2.2rem, 5vw, 3.4rem); font-weight: 800; letter-spacing: -0.035em; line-height: 1.05;
  margin: 18px 0 10px 0;
  background: linear-gradient(120deg, #ffffff 10%, #c4b5fd 45%, #67e8f9 90%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.vl-sub { color: var(--muted); font-size: 1.06rem; max-width: 760px; line-height: 1.6; }
.vl-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.vl-chip {
  font-size: .78rem; color: #cbd5e1; padding: 6px 12px; border-radius: 999px;
  background: var(--surface); border: 1px solid var(--border);
}
.vl-chip b { color: #fff; font-weight: 600; }

.vl-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.018));
  border: 1px solid var(--border); border-radius: 20px; padding: 22px 24px;
  box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 20px 50px -30px rgba(0,0,0,0.8);
  margin-bottom: 16px;
}
.vl-card-title {
  font-size: .74rem; font-weight: 650; letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;
}
.vl-label { font-size: .74rem; font-weight: 650; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 18px 0 10px 0; }
.vl-section { margin: 30px 0 14px 0; }
.vl-section h3 { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; color: #fff; }
.vl-section p { color: var(--muted); margin: 4px 0 0 0; font-size: .92rem; }

.vl-verdict { display: flex; gap: 18px; align-items: flex-start; position: relative; overflow: hidden; }
.vl-verdict-icon {
  flex: 0 0 auto; width: 52px; height: 52px; border-radius: 16px; display: grid; place-items: center;
  font-size: 1.5rem; font-weight: 800;
}
.vl-verdict h2 { margin: 0; font-size: 1.6rem; font-weight: 750; letter-spacing: -0.02em; }
.vl-verdict p { margin: 6px 0 0 0; color: #cbd5e1; line-height: 1.55; }
.vl-reasons { margin: 12px 0 0 0 !important; padding: 0 !important; list-style: none; display: flex; flex-wrap: wrap; gap: 8px; }
.vl-reasons li {
  margin: 0 !important; font-size: .8rem; padding: 5px 10px; border-radius: 8px; background: rgba(255,255,255,0.04);
  border: 1px solid var(--border); color: #cbd5e1;
}
.vl-q { font-size: .85rem; color: var(--muted); margin-top: 14px; }
.vl-q b { color: #e2e8f0; font-weight: 600; }

.vl-ring-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; }
.vl-ring-caption { font-size: .78rem; color: var(--muted); margin-top: 6px; text-align: center; }
.vl-mono { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }

.vl-answer { font-size: 1.08rem; line-height: 2.05; color: #f1f5f9; }
.vl-sent { padding: 3px 5px; border-radius: 7px; border-bottom: 2px solid; -webkit-box-decoration-break: clone; box-decoration-break: clone; cursor: help; }
.vl-sent.ok  { background: rgba(52,211,153,0.10); border-color: rgba(52,211,153,0.65); }
.vl-sent.mid { background: rgba(251,191,36,0.10); border-color: rgba(251,191,36,0.7); }
.vl-sent.bad { background: rgba(248,113,113,0.13); border-color: rgba(248,113,113,0.8); }
.vl-legend { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 16px; font-size: .78rem; color: var(--muted); }
.vl-legend span { display: inline-flex; align-items: center; gap: 6px; }
.vl-legend i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }

.vl-bar-row { margin-bottom: 16px; }
.vl-bar-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 7px; }
.vl-bar-name { font-size: .9rem; font-weight: 600; color: #e2e8f0; }
.vl-bar-hint { font-size: .74rem; color: var(--faint); margin-top: 2px; }
.vl-bar-val { font-family: 'JetBrains Mono', monospace; font-size: .92rem; font-weight: 700; }
.vl-bar { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.06); overflow: hidden; }
.vl-bar > div { height: 100%; border-radius: 999px; }
.vl-divider { height: 1px; background: var(--border); margin: 6px 0 16px 0; }
.vl-formula {
  font-family: 'JetBrains Mono', monospace; font-size: .76rem; color: #a5b4fc; background: rgba(99,102,241,0.08);
  border: 1px solid rgba(99,102,241,0.2); border-radius: 10px; padding: 10px 12px; line-height: 1.6; margin-top: 4px;
}

.vl-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }
.vl-sample { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 16px; }
.vl-sample-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.vl-sample-head span:first-child { font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); font-weight: 650; }
.vl-sample p { margin: 0; color: #cbd5e1; font-size: .92rem; line-height: 1.6; }
.vl-pill {
  font-size: .72rem; font-weight: 650; padding: 3px 10px; border-radius: 999px; text-transform: capitalize;
  border: 1px solid; white-space: nowrap;
}

.vl-cove-row { display: grid; grid-template-columns: 28px 1fr 1fr 120px; gap: 14px; padding: 14px 0; border-top: 1px solid var(--border); align-items: start; }
.vl-cove-row:first-of-type { border-top: 0; }
.vl-cove-num { width: 26px; height: 26px; border-radius: 8px; background: rgba(139,92,246,0.15); color: #c4b5fd; display: grid; place-items: center; font-size: .8rem; font-weight: 700; }
.vl-cove-k { font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; color: var(--faint); margin-bottom: 3px; font-weight: 650; }
.vl-cove-v { color: #e2e8f0; font-size: .92rem; line-height: 1.5; }
.vl-cove-note { font-size: .76rem; color: var(--muted); margin-top: 6px; }
.vl-issues { margin-top: 14px; padding: 12px 14px; border-radius: 12px; background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.2); font-size: .86rem; color: #fecaca; }

.vl-metric { background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015)); border: 1px solid var(--border); border-radius: 18px; padding: 18px 20px; height: 100%; }
.vl-metric-k { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 650; }
.vl-metric-v { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; margin: 6px 0 2px 0; font-family: 'JetBrains Mono', monospace; }
.vl-metric-d { font-size: .8rem; color: var(--muted); }
.vl-metric-d b { color: #e2e8f0; }

.vl-pipe { display: flex; align-items: stretch; gap: 8px; }
.vl-step { flex: 1 1 0; min-width: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 16px; position: relative; }
.vl-step-n { font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: #a78bfa; font-weight: 700; }
.vl-step h4 { margin: 6px 0 6px 0; font-size: .98rem; color: #fff; font-weight: 650; }
.vl-step p { margin: 0; font-size: .82rem; color: var(--muted); line-height: 1.5; }
.vl-arrow { display: grid; place-items: center; color: var(--faint); font-size: 1.2rem; }
.vl-ref { font-size: .88rem; color: #cbd5e1; line-height: 1.9; }
.vl-ref i { color: var(--muted); }

.vl-side-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.vl-logo { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; font-weight: 800; color: #fff;
  background: linear-gradient(135deg, #8b5cf6, #22d3ee); box-shadow: 0 8px 24px -8px rgba(139,92,246,0.8); }
.vl-side-name { font-weight: 750; font-size: 1.05rem; color: #fff; letter-spacing: -0.01em; }
.vl-side-tag { font-size: .74rem; color: var(--muted); }
.vl-conn { display: flex; align-items: center; gap: 8px; font-size: .8rem; color: #cbd5e1; padding: 8px 12px; border-radius: 10px; background: var(--surface); border: 1px solid var(--border); margin: 12px 0 6px 0; }
.vl-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
.vl-stat { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; }
.vl-stat div:first-child { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; font-weight: 650; }
.vl-stat div:last-child { font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 700; color: #fff; margin-top: 2px; }

.vl-empty { text-align: center; padding: 40px 20px; color: var(--muted); }
.vl-empty h4 { color: #fff; font-size: 1.1rem; margin: 0 0 6px 0; }
.vl-code { font-family: 'JetBrains Mono', monospace; background: rgba(0,0,0,0.35); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; color: #a5f3fc; display: inline-block; margin-top: 10px; font-size: .86rem; }

/* ---------- v2: minimal layout ---------- */
.block-container { max-width: 1160px !important; padding-top: 1.4rem !important; }
.st-key-narrow { max-width: 780px; margin: 0 auto; width: 100%; }
.vl-brand { display: flex; align-items: center; gap: 10px; font-weight: 750; font-size: 1.05rem; color: #fff; letter-spacing: -0.01em; height: 40px; }
.vl-brand .vl-logo { width: 30px; height: 30px; border-radius: 9px; font-size: .9rem; }
.vl-brand-dot { width: 7px; height: 7px; border-radius: 50%; margin-left: 2px; }

/* Nav: segmented control as quiet text links */
.st-key-page [data-testid="stButtonGroup"] { gap: 2px; }
.st-key-page button {
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  color: var(--muted) !important; padding: 6px 14px !important; border-radius: 10px !important; min-height: 0 !important;
}
.st-key-page button:hover { color: #fff !important; background: rgba(255,255,255,0.04) !important; }
.st-key-page button[kind="segmented_controlActive"], .st-key-page button[aria-checked="true"],
.st-key-page button[data-testid="stBaseButton-segmented_controlActive"] {
  color: #fff !important; background: rgba(255,255,255,0.07) !important;
}
.st-key-page button p { font-size: .9rem !important; font-weight: 550 !important; }


/* Settings popover trigger */
.st-key-settings button, [data-testid="stPopover"] > div > button {
  border-radius: 10px !important; min-height: 0 !important; padding: 6px 12px !important; height: auto !important;
  background: rgba(255,255,255,0.04) !important;
}
[data-testid="stPopoverBody"] { background: #0e1119 !important; border: 1px solid var(--border-strong) !important; border-radius: 16px !important; }

/* Home */
.vl-home { text-align: center; margin: 8vh 0 30px 0; }
.st-key-narrow > div:first-child:empty + div .vl-home { margin-top: 0; }
.st-key-ask { margin-top: 18px; }
.vl-home-title {
  font-size: clamp(2.1rem, 5.2vw, 3.5rem); font-weight: 800; letter-spacing: -0.04em; line-height: 1.06;
  background: linear-gradient(120deg, #ffffff 20%, #c4b5fd 60%, #67e8f9 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.vl-home-sub { color: var(--muted); font-size: 1.05rem; margin-top: 16px; }

/* Search bar */
.st-key-ask [data-testid="stForm"] { border: 0; padding: 0; }
.st-key-ask [data-testid="stTextInputRootElement"] {
  height: 58px !important; border-radius: 16px !important; background: rgba(255,255,255,0.035) !important;
  border: 1px solid var(--border-strong) !important; box-shadow: 0 20px 60px -30px rgba(139,92,246,0.45);
}
.st-key-ask [data-testid="stTextInputRootElement"]:focus-within {
  border-color: rgba(139,92,246,0.7) !important; box-shadow: 0 0 0 4px rgba(139,92,246,0.15) !important;
}
.st-key-ask [data-testid="stTextInputField"] {
  height: 56px !important; font-size: 1.05rem !important; padding: 0 20px !important; color: var(--text) !important;
  background: transparent !important;
}
.st-key-ask button { height: 58px !important; border-radius: 16px !important; }
[data-testid="stBaseButton-primaryFormSubmit"], button[kind="primaryFormSubmit"] {
  background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 55%, #0ea5e9 100%) !important;
  border: 0 !important; color: #fff !important; font-weight: 650 !important;
  box-shadow: 0 10px 30px -8px rgba(139,92,246,0.65);
}
[data-testid="stBaseButton-primaryFormSubmit"]:hover { filter: brightness(1.08); }

/* Example chips */
.st-key-chips { justify-content: center; margin-top: 4px; }
.st-key-chips button {
  border-radius: 999px !important; min-height: 0 !important; height: auto !important; padding: 7px 15px !important;
  background: transparent !important; border: 1px solid var(--border) !important;
}
.st-key-chips button p { font-size: .84rem !important; color: var(--muted) !important; white-space: nowrap !important; }
.st-key-chips button:hover p { color: #fff !important; }
.vl-try { text-align: center; color: var(--faint); font-size: .8rem; margin: 18px 0 8px 0; }

/* Result card */
.vl-result {
  position: relative; overflow: hidden; margin-top: 26px; padding: 26px 28px 22px 28px; border-radius: 22px;
  border: 1px solid var(--border); background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
  box-shadow: 0 30px 80px -40px rgba(0,0,0,0.9);
}
.vl-result-glow { position: absolute; inset: 0; pointer-events: none; }
.vl-result-top { position: relative; display: flex; gap: 22px; align-items: center; }
.vl-result-ring { flex: 0 0 auto; margin: -6px 0 -6px -6px; }
.vl-result-head h2 { margin: 10px 0 4px 0; font-size: 1.65rem; font-weight: 750; letter-spacing: -0.025em; color: #fff; line-height: 1.2; }
.vl-result-head p { margin: 0; color: var(--muted); font-size: .92rem; }
.vl-status { display: inline-block; font-size: .76rem; font-weight: 650; padding: 4px 11px; border-radius: 999px; border: 1px solid; letter-spacing: .02em; }
.vl-gap {
  position: relative; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-top: 18px; padding: 10px 14px;
  border-radius: 12px; background: rgba(248,113,113,0.07); border: 1px solid rgba(248,113,113,0.2); font-size: .86rem; color: #fecaca;
}
.vl-gap b { color: #fff; }
.vl-gap-arrow { color: #f87171; }
.vl-result-q { position: relative; margin-top: 22px; padding-top: 18px; border-top: 1px solid var(--border); color: var(--muted); font-size: .86rem; }
.vl-result .vl-answer { position: relative; margin-top: 8px; }
.vl-footer { text-align: center; color: var(--faint); font-size: .76rem; margin: 70px auto 0 auto; max-width: 560px; line-height: 1.6; }
.vl-conn { margin: 0 0 12px 0; }

/* Details expander */
.st-key-narrow [data-testid="stExpander"] { margin-top: 14px; }
.st-key-narrow [data-testid="stExpander"] summary p { font-weight: 600; color: #cbd5e1; }

@media (max-width: 640px) {
  .vl-result { padding: 20px 18px; }
  .vl-result-top { flex-direction: column; align-items: flex-start; gap: 6px; }
  .vl-home { margin-top: 4vh; }
}

@media (max-width: 900px) {
  .vl-pipe { flex-direction: column; }
  .vl-arrow { transform: rotate(90deg); }
}
@media (max-width: 640px) {
  .vl-cove-row { grid-template-columns: 26px 1fr; }
  .vl-cove-row > div:nth-child(3), .vl-cove-row > div:nth-child(4) { grid-column: 2; }
  .vl-verdict h2 { font-size: 1.3rem; }
}
</style>
"""


# ------------------------------------------------------------------ helpers
def md(markup: str) -> None:
    """Render HTML. Lines are flattened so Markdown never treats indentation as a code block."""
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text)).replace("$", "&#36;")


def score_color(value: float) -> str:
    if value >= 0.75:
        return "#34d399"
    if value >= 0.45:
        return "#fbbf24"
    return "#f87171"


def pill(label: str) -> str:
    color = LABEL_COLORS.get(label, "#94a3b8")
    return (f'<span class="vl-pill" style="color:{color};border-color:{color}55;'
            f'background:{color}14">{esc(label)}</span>')


def ring(value: float, color: str, size: int = 168) -> str:
    r = 52
    c = 2 * math.pi * r
    filled = max(0.0, min(1.0, value)) * c
    gid = f"g{abs(hash((value, color))) % 10**8}"
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 120 120">
  <defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="{color}"/><stop offset="100%" stop-color="{color}" stop-opacity="0.55"/>
  </linearGradient></defs>
  <circle cx="60" cy="60" r="{r}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="9"/>
  <circle cx="60" cy="60" r="{r}" fill="none" stroke="url(#{gid})" stroke-width="9" stroke-linecap="round"
    stroke-dasharray="{filled:.2f} {c:.2f}" transform="rotate(-90 60 60)"
    style="filter: drop-shadow(0 0 6px {color}88)"/>
  <text x="60" y="58" text-anchor="middle" fill="#fff" font-size="26" font-weight="800"
    font-family="JetBrains Mono, monospace">{value * 100:.0f}<tspan font-size="13" fill="#8b91a5">%</tspan></text>
  <text x="60" y="78" text-anchor="middle" fill="#8b91a5" font-size="8.5" letter-spacing="1.2"
    font-family="Inter, sans-serif">CONFIDENCE</text>
</svg>"""


# --------------------------------------------------------------- components
HEADLINES = {
    "reliable": "This answer looks reliable",
    "uncertain": "Double-check this answer",
    "hallucination": "This answer is likely made up",
    "abstained": "The model chose not to answer",
}


def nav_brand(connected: bool) -> str:
    dot = "#34d399" if connected else "#f87171"
    return f"""
<div class="vl-brand"><div class="vl-logo">V</div><span>VeriLens</span>
<span class="vl-brand-dot" style="background:{dot};box-shadow:0 0 8px {dot}"
 title="{'Connected to Groq' if connected else 'Not connected'}"></span></div>"""


def home_hero() -> str:
    return """
<div class="vl-home">
  <div class="vl-home-title">Know when your AI<br>is making things up.</div>
  <div class="vl-home-sub">Ask a question. VeriLens checks the answer and tells you how much to trust it.</div>
</div>"""


def footer() -> str:
    return """<div class="vl-footer">Final-year project · AI-Driven Uncertainty Calibration for Hallucination
Detection and Self-Verification in Large Language Models</div>"""


def connection_line(connected: bool, n_models: int) -> str:
    dot = "#34d399" if connected else "#f87171"
    text = f"Groq connected · {n_models} models available" if connected else "Not connected to Groq"
    return (f'<div class="vl-conn"><span style="width:8px;height:8px;border-radius:50%;background:{dot};'
            f'box-shadow:0 0 8px {dot}"></span>{text}</div>')


def result_card(res: dict) -> str:
    meta = STATUS[res["status"]]
    color = meta["color"]
    n = len(res["samples"])
    agree = sum(1 for s in res["samples"] if s["overall"] == "agree")
    checks = res["verification"]["checks"]
    passed = sum(1 for c in checks if c["label"] == "consistent")
    summary = f"{agree} of {n} re-generated answers agree"
    if checks:
        summary += f" · {passed} of {len(checks)} fact-checks passed"

    spans = []
    for s in res["sentences"]:
        cls = "ok" if s["score"] >= 0.75 else ("mid" if s["score"] >= 0.45 else "bad")
        sup = sum(1 for l in s["labels"] if l == "supported")
        tip = f"Supported by {sup} of {n} re-generated answers"
        spans.append(f'<span class="vl-sent {cls}" title="{esc(tip)}">{esc(s["text"])}</span>')

    claimed, final = res["features"]["verbalized"], res["final"]
    gap = ""
    if claimed - final > 0.25:
        gap = (f'<div class="vl-gap"><span>Model claimed <b>{claimed * 100:.0f}%</b> confidence</span>'
               f'<span class="vl-gap-arrow">→</span><span>evidence supports <b>{final * 100:.0f}%</b></span></div>')

    return f"""
<div class="vl-result" style="border-color:{color}38">
  <div class="vl-result-glow" style="background:radial-gradient(600px 180px at 0% 0%, {color}1f, transparent 70%)"></div>
  <div class="vl-result-top">
    <div class="vl-result-ring">{ring(final, color, size=124)}</div>
    <div class="vl-result-head">
      <span class="vl-status" style="color:{color};background:{color}17;border-color:{color}45">{meta['icon']}&nbsp; {meta['label']}</span>
      <h2>{HEADLINES[res['status']]}</h2>
      <p>{esc(summary)}</p>
    </div>
  </div>
  {gap}
  <div class="vl-result-q">{esc(res['question'])}</div>
  <div class="vl-answer">{' '.join(spans)}</div>
  <div class="vl-legend">
    <span><i style="background:#34d399"></i>Consistent</span>
    <span><i style="background:#fbbf24"></i>Uncertain</span>
    <span><i style="background:#f87171"></i>Likely hallucinated</span>
  </div>
</div>"""


def side_label(text: str) -> str:
    return f'<div class="vl-label">{esc(text)}</div>'


def section(title: str, subtitle: str = "") -> str:
    sub = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    return f'<div class="vl-section"><h3>{esc(title)}</h3>{sub}</div>'


def usage_stats(usage: dict) -> str:
    return f"""
<div class="vl-stats">
  <div class="vl-stat"><div>LLM calls</div><div>{usage['calls']}</div></div>
  <div class="vl-stat"><div>Tokens</div><div>{usage['total_tokens']:,}</div></div>
</div>"""


def _bar(name: str, hint: str, value: float, color: str | None = None) -> str:
    color = color or score_color(value)
    return f"""
<div class="vl-bar-row">
  <div class="vl-bar-head"><div><div class="vl-bar-name">{esc(name)}</div><div class="vl-bar-hint">{esc(hint)}</div></div>
    <div class="vl-bar-val" style="color:{color}">{value * 100:.0f}%</div></div>
  <div class="vl-bar"><div style="width:{max(2, value * 100):.1f}%;background:linear-gradient(90deg,{color}99,{color})"></div></div>
</div>"""


def signals_card(res: dict) -> str:
    f, d, cfg = res["features"], res["details"], res["config"]
    w = cfg["weights"]
    total = sum(w)
    rows = [
        _bar("Semantic consistency", f"sentence support {d['sentence_mean']:.2f} · answer agreement {d['agreement']:.2f}", f["consistency"]),
        _bar("Self-verification (CoVe)", f"verdict: {res['verification']['verdict']} · checks {d['check_mean']:.2f}", f["verification"]),
        _bar("Verbalized confidence", "what the model claims about itself", f["verbalized"], "#a78bfa"),
        '<div class="vl-divider"></div>',
        _bar("Heuristic score", f"{w[0] / total:.2f}·C + {w[1] / total:.2f}·V + {w[2] / total:.2f}·P", res["heuristic"]),
    ]
    if res["calibrated"] is not None:
        rows.append(_bar("Calibrated confidence", "σ(w·[C, V, P] + b), learned on labelled data", res["calibrated"]))
    gap = f["verbalized"] - res["final"]
    note = ""
    if gap > 0.25:
        note = (f'<div class="vl-issues" style="margin-top:4px">The model says it is {f["verbalized"] * 100:.0f}% sure, '
                f'but the evidence supports only {res["final"] * 100:.0f}%. It is overconfident by {gap * 100:.0f} points.</div>')
    return f"""
<div class="vl-card"><div class="vl-card-title"><span>Uncertainty signals</span></div>{''.join(rows)}{note}</div>"""


def samples_grid(res: dict) -> str:
    cards = []
    for i, s in enumerate(res["samples"], 1):
        cards.append(f"""
<div class="vl-sample"><div class="vl-sample-head"><span>Sample {i}</span>{pill(s['overall'])}</div>
<p>{esc(s['text'])}</p></div>""")
    return f"""
<div style="color:var(--muted);font-size:.88rem;margin:4px 0 14px 0">The same question was asked
{len(res['samples'])} more times at temperature {1.0}. When the model really knows the answer, the samples
agree. When it is guessing, the facts change from sample to sample.</div>
<div class="vl-grid">{''.join(cards)}</div>"""


def cove_card(res: dict) -> str:
    v = res["verification"]
    rows = []
    for i, c in enumerate(v["checks"], 1):
        note = f'<div class="vl-cove-note">{esc(c["note"])}</div>' if c["note"] else ""
        rows.append(f"""
<div class="vl-cove-row">
  <div class="vl-cove-num">{i}</div>
  <div><div class="vl-cove-k">Fact-check question</div><div class="vl-cove-v">{esc(c['question'])}</div></div>
  <div><div class="vl-cove-k">Independent answer</div><div class="vl-cove-v">{esc(c['independent_answer'])}</div>{note}</div>
  <div style="text-align:right">{pill(c['label'])}</div>
</div>""")
    issues = ""
    if v["issues"]:
        issues = '<div class="vl-issues"><b>Issues found:</b> ' + " · ".join(esc(x) for x in v["issues"]) + "</div>"
    body = "".join(rows) or '<div style="color:var(--muted)">No fact-check questions were produced.</div>'
    return f"""
<div style="color:var(--muted);font-size:.88rem;margin:4px 0 14px 0">The model wrote its own fact-check
questions, then answered them <b style="color:#e2e8f0">without seeing its original answer</b> (factored
Chain-of-Verification). Disagreement means the original claim is not stable.</div>
<div class="vl-card"><div class="vl-card-title"><span>Verification trace</span>{pill(v['verdict'])}</div>{body}{issues}</div>"""


def metric(label: str, value: str, detail: str = "", color: str = "#fff") -> str:
    return f"""
<div class="vl-metric"><div class="vl-metric-k">{esc(label)}</div>
<div class="vl-metric-v" style="color:{color}">{value}</div><div class="vl-metric-d">{detail}</div></div>"""


def pipeline_diagram() -> str:
    steps = [
        ("01", "Generate", "The model under test answers the question (greedy decoding)."),
        ("02", "Sample", "The same question is asked N more times at temperature 1.0."),
        ("03", "Cross-check", "Each sentence is checked against every sample: supported, neutral or contradicted."),
        ("04", "Self-verify", "The model writes fact-check questions and answers them without seeing its answer."),
        ("05", "Calibrate", "The signals are combined into P(correct) by a learned logistic calibrator."),
    ]
    parts = []
    for i, (n, title, text) in enumerate(steps):
        if i:
            parts.append('<div class="vl-arrow">→</div>')
        parts.append(f'<div class="vl-step"><div class="vl-step-n">{n}</div><h4>{title}</h4><p>{text}</p></div>')
    return f'<div class="vl-pipe">{"".join(parts)}</div>'


def empty_state(title: str, text: str, code: str = "") -> str:
    code_html = f'<div class="vl-code">{esc(code)}</div>' if code else ""
    return f'<div class="vl-card vl-empty"><h4>{esc(title)}</h4><div>{text}</div>{code_html}</div>'
