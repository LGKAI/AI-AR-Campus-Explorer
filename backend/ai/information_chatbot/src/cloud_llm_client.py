"""
STELLAR-RAG — CloudLLMClient
Supports any OpenAI-compatible API: Groq, DeepSeek, OpenRouter, Together AI.

Free tiers (recommended):
  Groq       → https://console.groq.com       (30 RPM free, Llama 3.3 70B)
  DeepSeek   → https://platform.deepseek.com  (free credits, deepseek-chat)
  OpenRouter → https://openrouter.ai          (free models incl. Llama, DS)

Rate-limit handling:
  - Sliding-window RPM guard (configurable)
  - Minimum gap between calls
  - Exponential back-off on 429 (up to BACKOFF_MAX_S)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# Per-provider + per-model defaults
# max_tpm: tokens-per-minute quota for the FREE tier.
#   0 = no TPM tracking (paid tier or unknown).
# Groq free tier TPM limits (verified 2025-05 from console.groq.com dashboard):
#   llama-3.3-70b-versatile : 12,000 TPM  (RPM: 30)
#   llama-3.1-8b-instant    : 30,000 TPM  (RPM: 30)
_PROVIDERS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model":    "qwen/qwen3-32b",   # 32B, 6K TPM, 500K tok/day, 60 RPM
        "max_rpm":  60,
        "min_gap":  2.0,
        "max_tpm":  6_000,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model":    "deepseek-chat",
        "max_rpm":  58,
        "min_gap":  1.1,
        "max_tpm":  0,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "meta-llama/llama-3.3-70b-instruct:free",
        "max_rpm":  18,
        "min_gap":  3.5,
        "max_tpm":  0,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "model":    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "max_rpm":  58,
        "min_gap":  1.1,
        "max_tpm":  0,
    },
}

# Per-model TPM limits — verified 2025-05 from console.groq.com/settings/limits.
# Columns: MODEL | RPM | Requests/Day | Tokens/Minute | Tokens/Day
#
#   llama-3.1-8b-instant    : 30 RPM | 14.4K req/day |  6K TPM | 500K tok/day
#   llama-3.3-70b-versatile : 30 RPM |    1K req/day | 12K TPM | 100K tok/day (*)
#   qwen/qwen3-32b          : 60 RPM |    1K req/day |  6K TPM | 500K tok/day
#   llama-4-scout-17b       : 30 RPM |    1K req/day | 30K TPM | 500K tok/day
#
# (*) 70B daily cap: 100K tokens. Eval call budget (Qwen3/LLaMA tokenizer):
#     System ~4K chars + context ~6.5K chars = 10.5K chars ~ 3,500 tokens @ 3.0c/tok
#     Output ~80 tokens -> ~3,580 total. 60 questions * 3,580 = 215K > 100K daily limit.
#     70B exhausts daily budget after ~27 eval questions. Use qwen3-32b instead.
#
# Why qwen/qwen3-32b for eval:
#   - 32B vs Ollama 7B: meaningful quality gap to compare
#   - 500K tok/day: handles full 60-question eval (60 * 3,580 = 215K << 500K)
#   - 60 RPM: double throughput headroom
#   - 6K TPM: 1 eval call per ~60s window (guard sleeps ~20s after each call)
_GROQ_MODEL_TPM: dict[str, int] = {
    "llama-3.3-70b-versatile":                   12_000,
    "llama-3.1-70b-versatile":                   12_000,
    "llama-3.1-8b-instant":                       6_000,
    "qwen/qwen3-32b":                             6_000,
    "meta-llama/llama-4-scout-17b-16e-instruct": 30_000,
    "mixtral-8x7b-32768":                         5_000,
    "gemma2-9b-it":                              15_000,
}

# Recommended min_gap_s for the NER+relation ingest pipeline
# (relation-only prompts ~700 tokens — much smaller than eval calls).
_GROQ_MODEL_MIN_GAP: dict[str, float] = {
    "llama-3.3-70b-versatile":                   5.0,   # 12K TPM / ~700 tok
    "llama-3.1-70b-versatile":                   5.0,
    "llama-3.1-8b-instant":                      3.0,   #  6K TPM / ~700 tok
    "qwen/qwen3-32b":                            2.0,   #  6K TPM, 60 RPM
    "meta-llama/llama-4-scout-17b-16e-instruct": 2.0,   # 30K TPM / ~700 tok
}

BACKOFF_BASE = 30.0
BACKOFF_MAX  = 300.0
MAX_RETRIES  = 6


def recommended_min_gap(provider: str, model: str) -> float:
    """Return the recommended min_gap_s for a given provider/model combination."""
    if provider == "groq":
        return _GROQ_MODEL_MIN_GAP.get(model, _PROVIDERS["groq"]["min_gap"])
    return _PROVIDERS.get(provider, _PROVIDERS["groq"])["min_gap"]

class CloudLLMClient:
    """
    Rate-limited, retry-safe client for any OpenAI-compatible LLM API.

    Usage::
        client = CloudLLMClient()          # reads CLOUD_PROVIDER + CLOUD_API_KEY from env
        text   = client.chat(messages, temperature=0.0)
    """

    def __init__(
        self,
        provider:  str  = "",
        api_key:   str  = "",
        base_url:  str  = "",
        model:     str  = "",
        max_rpm:   int  = 0,
        min_gap_s: float = 0.0,
    ) -> None:
        import os

        self.provider = (provider or os.getenv("CLOUD_PROVIDER", "groq")).lower()
        defaults      = _PROVIDERS.get(self.provider, _PROVIDERS["groq"])

        self.api_key  = api_key  or os.getenv("CLOUD_API_KEY",  "")
        self.base_url = base_url or os.getenv("CLOUD_API_BASE", defaults["base_url"])
        self.model    = model    or os.getenv("CLOUD_MODEL",    defaults["model"])
        self._max_rpm = max_rpm  or defaults["max_rpm"]
        self._min_gap = min_gap_s or defaults["min_gap"]

        if not self.api_key:
            raise ValueError(
                f"CLOUD_API_KEY not set for provider '{self.provider}'. "
                "Add it to your .env file."
            )

        # Lazy import openai (pip install openai)
        try:
            from openai import OpenAI
            # max_retries=0: disable SDK's own retry so our rate limiter is in full control
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
        except ImportError:
            raise ImportError(
                "openai package not found. Run: pip install openai --break-system-packages"
            )

        # TPM limit: check model-specific override, then provider default
        if self.provider == "groq" and self.model in _GROQ_MODEL_TPM:
            self._max_tpm: int = _GROQ_MODEL_TPM[self.model]
        else:
            self._max_tpm = defaults.get("max_tpm", 0)

        # Rate-limit state (thread-safe)
        self._lock       = threading.Lock()
        self._timestamps: deque[float] = deque()   # RPM tracking
        self._tpm_log:    deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._last_call  = 0.0

        logger.info(
            f"[CloudLLMClient] provider={self.provider}  model={self.model}  "
            f"max_rpm={self._max_rpm}  max_tpm={self._max_tpm or 'unlimited'}"
        )

    # ─
    # Public API
    # ─

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        """
        Estimate token count before sending to the API.

        Vietnamese text tokenizes at ~2.0 chars/token with the LLaMA tokenizer
        (diacritics count as one char but one token; syllable-based splitting).
        English system prompts tokenize at ~4.0 chars/token.
        Mixed content (typical STELLAR-RAG message): ~2.5 chars/token is safe.

        Using 2.0 (conservative) to prevent the TPM guard from underestimating
        and allowing calls that would exceed the actual quota.
        """
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return int(total_chars / 2.0) + len(messages) * 10

    def chat(
        self,
        messages:    list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens:  int   = 2048,
    ) -> str:
        """Send messages and return the assistant text."""
        estimated_in  = self._estimate_tokens(messages)
        estimated_out = min(max_tokens, 400)          # conservative output estimate
        estimated_total = estimated_in + estimated_out

        backoff = BACKOFF_BASE
        for attempt in range(1, MAX_RETRIES + 1):
            self._wait_for_slot(estimated_total)
            try:
                resp = self._client.chat.completions.create(
                    model       = self.model,
                    messages    = messages,
                    temperature = temperature,
                    max_tokens  = max_tokens,
                )
                # Record actual tokens if available, otherwise use estimate
                usage = getattr(resp, "usage", None)
                actual_tokens = (
                    (usage.prompt_tokens + usage.completion_tokens)
                    if usage else estimated_total
                )
                self._record_call(actual_tokens)
                return resp.choices[0].message.content or ""

            except Exception as exc:
                err = str(exc)
                if _is_rate_limit(err):
                    wait = min(backoff, BACKOFF_MAX)
                    # Extract Groq's specific limit header/message if available
                    limit_detail = _extract_limit_detail(exc)
                    logger.warning(
                        f"[CloudLLM/{self.provider}] 429 rate-limit "
                        f"(attempt {attempt}/{MAX_RETRIES}). "
                        + (f"Reason: {limit_detail}  " if limit_detail else "")
                        + f"Sleeping {wait:.0f}s …"
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, BACKOFF_MAX)
                elif attempt < MAX_RETRIES:
                    logger.warning(
                        f"[CloudLLM/{self.provider}] error attempt {attempt}: "
                        f"{err[:120]}. Retrying in 5s …"
                    )
                    time.sleep(5)
                else:
                    logger.error(
                        f"[CloudLLM/{self.provider}] all {MAX_RETRIES} attempts failed: "
                        f"{err[:200]}"
                    )
                    raise
        return ""

    # ─
    # Rate limiting
    # ─

    def _wait_for_slot(self, estimated_tokens: int = 0) -> None:
        with self._lock:
            now = time.monotonic()

            # 1. Min gap between consecutive calls
            gap = now - self._last_call
            if gap < self._min_gap:
                time.sleep(self._min_gap - gap)
                now = time.monotonic()

            # 2. RPM sliding-window guard
            cutoff = now - 60.0
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) >= self._max_rpm:
                sleep_needed = 60.0 - (now - self._timestamps[0]) + 0.5
                if sleep_needed > 0:
                    logger.info(f"[CloudLLM/{self.provider}] RPM full — sleeping {sleep_needed:.1f}s")
                    time.sleep(sleep_needed)
                    now    = time.monotonic()
                    cutoff = now - 60.0
                    while self._timestamps and self._timestamps[0] < cutoff:
                        self._timestamps.popleft()

            # 3. TPM sliding-window guard
            if self._max_tpm > 0 and estimated_tokens > 0:
                cutoff = now - 60.0
                while self._tpm_log and self._tpm_log[0][0] < cutoff:
                    self._tpm_log.popleft()

                used_tokens = sum(t for _, t in self._tpm_log)
                if used_tokens + estimated_tokens > self._max_tpm * 0.80:  # 80% safety buffer
                    # Sleep until oldest entry exits the 60s window
                    if self._tpm_log:
                        sleep_needed = 60.0 - (now - self._tpm_log[0][0]) + 1.0
                        if sleep_needed > 0:
                            logger.info(
                                f"[CloudLLM/{self.provider}] TPM guard "
                                f"(used={used_tokens}+est={estimated_tokens} "
                                f">= {int(self._max_tpm*0.8)}) — sleeping {sleep_needed:.1f}s"
                            )
                            time.sleep(sleep_needed)
                            now    = time.monotonic()
                            cutoff = now - 60.0
                            while self._tpm_log and self._tpm_log[0][0] < cutoff:
                                self._tpm_log.popleft()

            self._last_call = time.monotonic()

    def _record_call(self, tokens: int = 0) -> None:
        with self._lock:
            now = time.monotonic()
            self._timestamps.append(now)
            if tokens > 0:
                self._tpm_log.append((now, tokens))

    def rpm_snapshot(self) -> int:
        with self._lock:
            cutoff = time.monotonic() - 60.0
            return sum(1 for t in self._timestamps if t >= cutoff)

# Helpers

def _extract_limit_detail(exc: Exception) -> str:
    """
    Pull the most useful fragment from a Groq 429 response.

    Groq returns rate-limit info in three places:
      1. Response headers  (x-ratelimit-limit-tokens, x-ratelimit-remaining-tokens, ...)
      2. Response body     (error.message, error.code)
      3. Exception string  (fallback)
    """
    try:
        # OpenAI SDK wraps Groq errors as RateLimitError with a .response attribute
        response = getattr(exc, "response", None)
        if response is not None:
            # Headers carry per-minute limits and retry-after
            headers = getattr(response, "headers", {}) or {}
            parts: list[str] = []
            for h in ("x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
                      "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                      "retry-after"):
                v = headers.get(h)
                if v is not None:
                    parts.append(f"{h}={v}")
            if parts:
                return "  ".join(parts)
            # Body message
            body = getattr(response, "text", "") or ""
            if body:
                return body[:200]
    except Exception:
        pass
    # Fallback: first 200 chars of exception string
    return str(exc)[:200]


def _is_rate_limit(err: str) -> bool:
    keywords = (
        "429", "rate", "quota", "too many requests",
        "ratelimit", "rate_limit", "resource_exhausted",
        "overloaded", "tokens per minute",
    )
    el = err.lower()
    return any(k in el for k in keywords)
