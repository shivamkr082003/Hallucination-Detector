"""Thin wrapper around the Groq chat API with retries, JSON parsing and usage tracking."""
from __future__ import annotations

import json
import re
import threading
import time

from groq import BadRequestError, Groq, RateLimitError

# Models that are not chat/completion models.
NON_CHAT_HINTS = ("whisper", "orpheus", "prompt-guard", "safeguard", "tts", "playai", "allam")

# Generator = the model whose answers we test. Qwen hallucinates noticeably on
# obscure facts, which makes it a good subject for a hallucination detector.
PREFERRED_GENERATORS = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
# Judge = neutral comparator used only to compare texts (like an NLI model).
PREFERRED_JUDGES = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def pick_model(available: list[str], preferred: list[str]) -> str:
    for model in preferred:
        if model in available:
            return model
    return available[0] if available else preferred[0]


def _extra_params(model: str) -> dict:
    """Keep reasoning models fast and their output clean."""
    m = model.lower()
    if "gpt-oss" in m:
        return {"reasoning_effort": "low"}
    if "qwen3" in m:
        return {"reasoning_effort": "none"}
    return {}


def _retry_after(exc: Exception, default: float) -> float:
    """Seconds to wait, from the retry-after header or the 'try again in Xs' message."""
    seconds = None
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            seconds = float(response.headers.get("retry-after"))
        except (TypeError, ValueError):
            seconds = None
    if seconds is None:
        match = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", str(exc))
        if match:
            minutes = float(match.group(1) or 0)
            value = float(match.group(2)) / (1000 if match.group(3) == "ms" else 1)
            seconds = minutes * 60 + value
    return min(max(seconds if seconds is not None else default, 1.0), 45.0) + 0.5


def parse_json(text: str) -> dict | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text or "", re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class LLM:
    def __init__(self, api_key: str):
        # The SDK retries 429/5xx itself and honours the server's retry-after header.
        self.client = Groq(api_key=api_key, max_retries=2, timeout=90)
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # ------------------------------------------------------------------ usage
    def usage(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            }

    def _record(self, resp) -> None:
        with self._lock:
            self.calls += 1
            if getattr(resp, "usage", None):
                self.prompt_tokens += resp.usage.prompt_tokens or 0
                self.completion_tokens += resp.usage.completion_tokens or 0

    # ----------------------------------------------------------------- models
    def list_chat_models(self) -> list[str]:
        ids = [m.id for m in self.client.models.list().data]
        return sorted(i for i in ids if not any(h in i.lower() for h in NON_CHAT_HINTS))

    # ------------------------------------------------------------------- chat
    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 400,
        json_mode: bool = False,
    ) -> str:
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        kwargs.update(_extra_params(model))
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._create(kwargs)
        self._record(resp)
        text = resp.choices[0].message.content or ""
        return _THINK_RE.sub("", text).strip()

    def _create(self, kwargs: dict):
        """Create a completion, waiting out per-minute rate limits (free tier: 30 RPM / 8K TPM)."""
        for attempt in range(6):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                # Daily quota exhausted: waiting a few seconds will not help.
                if attempt == 5 or "per day" in str(exc).lower():
                    raise
                time.sleep(_retry_after(exc, default=5.0 * (attempt + 1)))
            except BadRequestError as exc:
                # Some models reject reasoning params; retry without them.
                if "reasoning" in str(exc).lower() and "reasoning_effort" in kwargs:
                    kwargs = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
                    continue
                raise
        raise RuntimeError("unreachable")

    def chat_json(self, model: str, messages: list[dict], max_tokens: int = 900) -> dict:
        """Call the model in JSON mode; retry once if the output is not valid JSON."""
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                text = self.chat(model, messages, temperature=0.0 if attempt == 0 else 0.3,
                                 max_tokens=max_tokens, json_mode=True)
            except BadRequestError as exc:  # e.g. json_validate_failed
                last_error = exc
                continue
            data = parse_json(text)
            if data is not None:
                return data
        if last_error:
            raise last_error
        return {}
