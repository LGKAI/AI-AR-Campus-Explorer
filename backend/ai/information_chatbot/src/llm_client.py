"""
Unified LLM Client — Ollama (local) + Cloud LLM (Groq/DeepSeek/OpenRouter).

Supports three modes (set LLM_BACKEND in .env):
  "ollama"  — local Ollama only (default, no API key needed)
  "cloud"   — Cloud LLM only (Groq / DeepSeek / OpenRouter / Together)
  "both"    — Ollama primary, Cloud fallback

Dual-answer mode (answer_dual in agent.py):
  Calls both Ollama and Cloud LLM in parallel via chat_dual().

Set in .env:
  CLOUD_PROVIDER = groq          # groq | deepseek | openrouter | together
  CLOUD_API_KEY  = gsk_...       # your API key
  CLOUD_MODEL    =               # leave blank for provider default
"""
from __future__ import annotations

import logging
import time
from typing import Any, Generator

from config import settings

logger = logging.getLogger(__name__)

class LLMResponse:
    """Unified response wrapper."""
    def __init__(
        self,
        content:  str,
        backend:  str,
        latency:  float,
        model:    str,
    ) -> None:
        self.content = content
        self.backend = backend
        self.latency = latency
        self.model   = model

    def __str__(self) -> str:
        return self.content

class LLMClient:
    """
    Unified chat client.  Instantiate once (e.g. on Agent.__init__) and
    call `chat()` or `stream()` per query.

    Parameters mirror what agent.py passes to ollama.Client.chat():
        messages : OpenAI-style list[{"role": ..., "content": ...}]
        options  : {"temperature": 0.0, ...}  (Ollama-compatible)
    """

    def __init__(self) -> None:
        self.backend = getattr(settings, "llm_backend", "ollama").lower()
        self._ollama = None
        self._cloud:  "Any | None" = None   # CloudLLMClient instance
        self._init_backends()

    # ─
    # Initialisation
    # ─

    def _init_backends(self) -> None:
        if self.backend in ("ollama", "both"):
            try:
                from ollama import Client as OllamaClient  # type: ignore
                self._ollama = OllamaClient(host=settings.ollama_host)
                logger.info(f"[LLMClient] Ollama ready at {settings.ollama_host}")
            except Exception as exc:
                logger.warning(f"[LLMClient] Ollama init failed: {exc}")

        if self.backend in ("cloud", "both"):
            self._try_init_cloud()

    def _try_init_cloud(self) -> bool:
        """Try to init CloudLLMClient. Returns True on success."""
        if not getattr(settings, "cloud_api_key", ""):
            logger.warning("[LLMClient] CLOUD_API_KEY not set — cloud backend disabled")
            return False
        try:
            from cloud_llm_client import CloudLLMClient
            self._cloud = CloudLLMClient(
                provider = settings.cloud_provider,
                api_key  = settings.cloud_api_key,
                base_url = settings.cloud_api_base,
                model    = settings.cloud_model,
            )
            logger.info(
                f"[LLMClient] Cloud ready: provider={settings.cloud_provider} "
                f"model={self._cloud.model}"
            )
            return True
        except Exception as exc:
            logger.warning(f"[LLMClient] Cloud init failed: {exc}")
            return False

    # ─
    # Public chat (blocking)
    # ─

    def chat(
        self,
        model:    str,
        messages: list[dict[str, Any]],
        options:  dict[str, Any] | None = None,
    ) -> "LLMResponse":
        """
        Ollama-compatible .chat() interface.

        Returns an LLMResponse with .message.content  ← same attribute
        path that agent.py currently uses.
        """
        options = options or {}

        if self.backend == "ollama":
            return self._chat_ollama(model, messages, options)
        if self.backend == "cloud":
            return self._chat_cloud(messages, options)
        # both: try Ollama first, fall back to Cloud
        try:
            return self._chat_ollama(model, messages, options)
        except Exception as exc:
            logger.warning(f"[LLMClient] Ollama failed ({exc}), falling back to Cloud")
            return self._chat_cloud(messages, options)

    # Ollama

    def _chat_ollama(
        self,
        model:    str,
        messages: list[dict],
        options:  dict,
    ) -> "LLMResponse":
        t0   = time.perf_counter()
        resp = self._ollama.chat(model=model, messages=messages, options=options)
        lat  = time.perf_counter() - t0
        text = resp["message"]["content"]
        return _OllamaLLMResponse(text, lat, model)

    # Cloud LLM (Groq / DeepSeek / OpenRouter / Together)

    def _chat_cloud(
        self,
        messages: list[dict],
        options:  dict,
    ) -> "LLMResponse":
        if self._cloud is None:
            raise RuntimeError("Cloud backend not initialised (check CLOUD_API_KEY in .env)")
        temp = float(options.get("temperature", 0.0))
        t0   = time.perf_counter()
        text = self._cloud.chat(
            messages    = messages,
            temperature = temp,
            max_tokens  = int(options.get("num_predict", 2048)),
        )
        lat = time.perf_counter() - t0
        return LLMResponse(text, f"cloud/{settings.cloud_provider}", lat, self._cloud.model)

    # ─
    # Dual generation (Ollama + Cloud in parallel)
    # ─

    def chat_dual(
        self,
        model:    str,
        messages: list[dict[str, Any]],
        options:  dict[str, Any] | None = None,
    ) -> tuple["LLMResponse | None", "LLMResponse | None", dict[str, str]]:
        """
        Run Ollama AND Cloud LLM in parallel (ThreadPoolExecutor).

        Returns (ollama_resp, cloud_resp, errors_dict).
        Either resp may be None if that backend is unavailable or failed.
        CloudLLMClient already handles 429 with exponential back-off.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        options = options or {}
        results: dict[str, "LLMResponse | None"] = {"ollama": None, "cloud": None}
        errors:  dict[str, str] = {}

        tasks: dict = {}
        with ThreadPoolExecutor(max_workers=2) as exe:
            if self._ollama:
                tasks[exe.submit(self._chat_ollama, model, messages, options)] = "ollama"
            if self._cloud:
                tasks[exe.submit(self._chat_cloud, messages, options)] = "cloud"

            for fut in as_completed(tasks):
                backend = tasks[fut]
                try:
                    results[backend] = fut.result()
                except Exception as exc:
                    errors[backend] = str(exc)
                    logger.warning(f"[LLMClient] dual {backend} failed: {exc}")

        return results["ollama"], results["cloud"], errors

    def ensure_dual(self) -> None:
        """Force-initialise both Ollama and Cloud backends for dual mode."""
        if self._ollama is None:
            try:
                from ollama import Client as OllamaClient
                self._ollama = OllamaClient(host=settings.ollama_host)
                logger.info("[LLMClient] Ollama lazy-init OK")
            except Exception as exc:
                logger.warning(f"[LLMClient] Ollama lazy-init failed: {exc}")
        if self._cloud is None:
            # Check prerequisites before attempting init (surfaces the real error)
            if not settings.cloud_api_key:
                logger.warning(
                    "[LLMClient] Cloud backend disabled: CLOUD_API_KEY not set in .env. "
                    "Get a free key at https://console.groq.com"
                )
            else:
                try:
                    import openai as _openai_check  # noqa: F401
                except ImportError:
                    logger.warning(
                        "[LLMClient] Cloud backend disabled: 'openai' package not installed. "
                        "Fix: pip install openai>=1.30.0"
                    )
                    return
                ok = self._try_init_cloud()
                if not ok:
                    logger.warning("[LLMClient] Cloud init failed — dual mode runs Ollama only")

    def stream(
        self,
        model:    str,
        messages: list[dict[str, Any]],
        options:  dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        """
        Yield text tokens one by one.
        Uses Ollama streaming if available, else buffers the Cloud LLM response.
        """
        options = options or {}

        if self.backend in ("ollama", "both") and self._ollama:
            try:
                yield from self._stream_ollama(model, messages, options)
                return
            except Exception as exc:
                if self.backend == "ollama":
                    raise
                logger.warning(f"[LLMClient] Ollama stream failed: {exc}")

        # Cloud: buffer full response, yield words for UI responsiveness
        resp = self._chat_cloud(messages, options)
        for word in resp.content.split(" "):
            yield word + " "

    def _stream_ollama(
        self,
        model:    str,
        messages: list[dict],
        options:  dict,
    ) -> Generator[str, None, None]:
        for chunk in self._ollama.chat(
            model=model, messages=messages, options=options, stream=True
        ):
            token = chunk["message"]["content"]
            if token:
                yield token

# ─
# Helper classes / functions
# ─

class _OllamaLLMResponse(LLMResponse):
    """Duck-type shim: exposes .message.content for agent.py compatibility."""
    def __init__(self, text: str, latency: float, model: str) -> None:
        super().__init__(text, "ollama", latency, model)
        # agent.py accesses resp["message"]["content"] via dict protocol
        self._dict = {"message": {"content": text}}

    def __getitem__(self, key):
        return self._dict[key]

    # Also keep .message.content for newer code
    @property
    def message(self):
        return type("M", (), {"content": self.content})()

def _extract_system_and_last_user(messages: list[dict]) -> tuple[str, str]:
    """Extract system prompt and last user message from messages list."""
    system = ""
    user   = ""
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system = content
        elif role == "user":
            user = content
    return system, user
