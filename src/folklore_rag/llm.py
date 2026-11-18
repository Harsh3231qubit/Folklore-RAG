"""Groq chat client with on-disk caching, retries and model auto-discovery."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import NON_CHAT_MARKERS, Settings
from .utils import strip_think

log = logging.getLogger(__name__)

# Reasoning models spend part of max_tokens on hidden thinking.
REASONING_MARKERS = ("gpt-oss", "qwen")


def _retryable_errors():
    from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    return (RateLimitError, APIConnectionError, InternalServerError, APITimeoutError)


class LLMClient:
    """Thin wrapper around the Groq SDK.

    Every uncached call is written to ``cache_dir`` keyed by a hash of
    (model, system, user, temperature, max_tokens), which makes experiments
    resumable after a rate limit or crash.
    """

    def __init__(self, settings: Settings):
        from groq import Groq

        settings.validate()
        self.settings = settings
        self.client = Groq(api_key=settings.groq_api_key)
        self.cache_dir = Path(settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.writer, self.critic, self.judge = self._resolve_models()
        log.info("writer=%s critic=%s judge=%s", self.writer, self.critic, self.judge)

    # ---- model selection -------------------------------------------------
    def _resolve_models(self) -> tuple[str, str, str]:
        s = self.settings
        available = sorted(m.id for m in self.client.models.list().data)
        chat = [m for m in available if not any(x in m for x in NON_CHAT_MARKERS)]
        if not chat:
            raise RuntimeError(f"No chat models available for this key: {available}")

        def pick(*prefs: str, exclude: tuple[str, ...] = ()) -> str | None:
            for p in prefs:
                for m in chat:
                    if p in m and m not in exclude:
                        return m
            return None

        writer = s.writer_model or pick("gpt-oss-120b") or chat[0]
        critic = s.critic_model or pick("gpt-oss-20b") or writer
        judge = s.judge_model or pick("qwen", "llama", "gemma", "kimi", "mistral", exclude=(writer, critic)) or writer
        if "gpt-oss" in judge and "gpt-oss" in writer:
            log.warning("Judge and writer share a model family; expect self-preference bias in scores.")
        return writer, critic, judge

    # ---- calls -----------------------------------------------------------
    def _call(self, model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
        @retry(
            retry=retry_if_exception_type(_retryable_errors()),
            wait=wait_exponential(multiplier=2, min=5, max=90),
            stop=stop_after_attempt(8),
            reraise=True,
        )
        def _do() -> str:
            kwargs = {}
            if "gpt-oss" in model:
                kwargs["reasoning_effort"] = "low"
            tokens = max(max_tokens, 2000) if any(x in model for x in REASONING_MARKERS) else max_tokens
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=tokens,
                **kwargs,
            )
            return resp.choices[0].message.content or ""

        return _do()

    def complete(self, model: str, system: str, user: str, temperature: float = 0.7, max_tokens: int = 700) -> str:
        key = hashlib.sha256(json.dumps([model, system, user, temperature, max_tokens]).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["out"]

        out = ""
        for mult in (1, 2):  # retry once with more room if a reasoning model returned nothing
            out = strip_think(self._call(model, system, user, temperature, max_tokens * mult))
            time.sleep(self.settings.rate_sleep)
            if out:
                break
        if not out:
            raise RuntimeError(f"Model {model} returned empty output twice")
        path.write_text(json.dumps({"out": out}))
        return out
