"""Облачные модели: OpenAI (Chat Completions) и Gemini (generateContent) через HTTP.

Ключи — только из переменных окружения OPENAI_API_KEY / GEMINI_API_KEY; в журнал не пишутся.
Модель и доп. параметры задаются в config/show.yaml (brain.*), цены — для оценки расходов.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np

from ..registry import Registry
from .prompt import gemini_schema, openai_schema, system_text


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0  # включая «размышления» модели
    reasoning_tokens: int = 0


@dataclass
class Prices:
    """$ за 1M токенов."""

    input: float = 0.0
    cached_input: float = 0.0
    output: float = 0.0

    def cost(self, u: Usage) -> float:
        fresh = max(0, u.input_tokens - u.cached_tokens)
        return (fresh * self.input + u.cached_tokens * self.cached_input + u.output_tokens * self.output) / 1e6


@dataclass
class BrainResult:
    actions: list[dict[str, Any]]
    usage: Usage
    latency: float
    cost: float
    error: str | None = None
    raw: str = ""


def encode_jpeg(rgb: np.ndarray, quality: int = 70) -> bytes:
    from PIL import Image  # type: ignore

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def parse_actions(text: str) -> list[dict[str, Any]]:
    data = json.loads(text)
    actions = data.get("actions", []) if isinstance(data, dict) else []
    out = []
    for a in actions[:2]:
        if isinstance(a, dict) and a.get("type"):
            out.append({"type": a["type"], "id": a.get("id"), "reason": str(a.get("reason", ""))[:200],
                        "confidence": a.get("confidence")})
    return out


class Brain(ABC):
    name = "base"

    def __init__(self, registry: Registry, model: str, prices: Prices, timeout: float = 8.0,
                 extra: dict[str, Any] | None = None):
        self.registry = registry
        self.model = model
        self.prices = prices
        self.timeout = timeout
        self.extra = dict(extra or {})
        self.system = system_text(registry)
        self._session: aiohttp.ClientSession | None = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def close(self) -> None:
        if self._session:
            await self._session.close()

    async def decide(self, context: str, image: bytes | None) -> BrainResult:
        t0 = time.monotonic()
        try:
            text, usage = await self._call(context, image)
            actions = parse_actions(text)
            return BrainResult(actions, usage, time.monotonic() - t0, self.prices.cost(usage), raw=text[:2000])
        except Exception as e:  # noqa: BLE001 — сеть/формат: решение пропускается, шоу продолжается
            return BrainResult([], Usage(), time.monotonic() - t0, 0.0, error=f"{type(e).__name__}: {e}"[:500])

    @abstractmethod
    async def _call(self, context: str, image: bytes | None) -> tuple[str, Usage]: ...


class OpenAIBrain(Brain):
    name = "openai"
    URL = "https://api.openai.com/v1/chat/completions"

    async def _call(self, context: str, image: bytes | None) -> tuple[str, Usage]:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("нет OPENAI_API_KEY")
        content: list[dict[str, Any]] = [{"type": "text", "text": context}]
        if image:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(image).decode(), "detail": "low"}})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "sound_actions", "strict": True, "schema": openai_schema(self.registry)}},
            **self.extra,
        }
        s = await self.session()
        async with s.post(self.URL, json=body, headers={"Authorization": f"Bearer {key}"}) as r:
            data = await r.json(content_type=None)
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {json.dumps(data, ensure_ascii=False)[:300]}")
        msg = data["choices"][0]["message"]
        if msg.get("refusal"):
            raise RuntimeError(f"отказ модели: {msg['refusal'][:200]}")
        u = data.get("usage", {})
        usage = Usage(
            input_tokens=int(u.get("prompt_tokens", 0)),
            cached_tokens=int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)),
            output_tokens=int(u.get("completion_tokens", 0)),
            reasoning_tokens=int((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)),
        )
        return msg.get("content") or "{}", usage


class GeminiBrain(Brain):
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def _call(self, context: str, image: bytes | None) -> tuple[str, Usage]:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("нет GEMINI_API_KEY")
        parts: list[dict[str, Any]] = [{"text": context}]
        if image:
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(image).decode()}})
        gen = {"responseMimeType": "application/json", "responseSchema": gemini_schema(self.registry),
               "temperature": 0.2}
        gen.update(self.extra.get("generationConfig", {}))
        body = {
            "systemInstruction": {"parts": [{"text": self.system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen,
            **{k: v for k, v in self.extra.items() if k != "generationConfig"},
        }
        s = await self.session()
        async with s.post(self.URL.format(model=self.model), json=body, headers={"x-goog-api-key": key}) as r:
            data = await r.json(content_type=None)
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {json.dumps(data, ensure_ascii=False)[:300]}")
        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"пустой ответ: {json.dumps(data, ensure_ascii=False)[:300]}")
        text = "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", [])
                       if not p.get("thought"))
        m = data.get("usageMetadata", {})
        usage = Usage(
            input_tokens=int(m.get("promptTokenCount", 0)),
            cached_tokens=int(m.get("cachedContentTokenCount", 0)),
            output_tokens=int(m.get("candidatesTokenCount", 0)) + int(m.get("thoughtsTokenCount", 0)),
            reasoning_tokens=int(m.get("thoughtsTokenCount", 0)),
        )
        return text or "{}", usage


class FakeBrain(Brain):
    """Подставной «мозг» для тестов: правила «слово в тексте → действие»."""

    def __init__(self, registry: Registry, rules: list[tuple[str, dict[str, Any]]] | None = None,
                 latency: float = 0.0):
        super().__init__(registry, "fake", Prices())
        self.rules = rules or []
        self.fake_latency = latency
        self.contexts: list[str] = []
        self.images = 0

    name = "fake"

    async def _call(self, context: str, image: bytes | None) -> tuple[str, Usage]:
        import asyncio

        self.contexts.append(context)
        self.images += 1 if image else 0
        if self.fake_latency:
            await asyncio.sleep(self.fake_latency)
        new = context.split("НОВОЕ:", 1)[-1].lower()
        actions = [a for word, a in self.rules if word in new]
        return json.dumps({"actions": actions[:2]}, ensure_ascii=False), Usage(len(context) // 4, 0, 20)


def make_brain(kind: str | None, registry: Registry, cfg: dict[str, Any] | None = None) -> Brain | None:
    cfg = cfg or {}
    if not kind or kind == "off":
        return None
    p = (cfg.get("prices") or {}).get(kind, {})
    prices = Prices(float(p.get("input", 0)), float(p.get("cached_input", 0)), float(p.get("output", 0)))
    sub = cfg.get(kind) or {}
    if kind == "openai":
        return OpenAIBrain(registry, sub.get("model", "gpt-5-mini"), prices, float(cfg.get("timeout", 8.0)),
                           sub.get("extra", {"reasoning_effort": "minimal"}))
    if kind == "gemini":
        return GeminiBrain(registry, sub.get("model", "gemini-3.5-flash"), prices, float(cfg.get("timeout", 8.0)),
                           sub.get("extra", {}))
    raise ValueError(f"неизвестный провайдер: {kind} (openai | gemini | off)")
