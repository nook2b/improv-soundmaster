import asyncio
import json
from pathlib import Path

import numpy as np
from aiohttp import web

from soundmaster.brain.providers import GeminiBrain, OpenAIBrain, Prices, encode_jpeg, parse_actions
from soundmaster.brain.prompt import gemini_schema, openai_schema, system_text
from soundmaster.brain.scheduler import BrainScheduler, SchedulerConfig
from soundmaster.media.asr import Transcript
from soundmaster.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent


def registry():
    from conftest import with_test_music

    return with_test_music(load_registry(ROOT / "config" / "registry.yaml"))


async def serve(handler):
    app = web.Application()
    app.router.add_post("/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def test_openai_request_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen = {}

    async def handler(req):
        seen["auth"] = req.headers.get("Authorization")
        seen["body"] = await req.json()
        return web.json_response({
            "choices": [{"message": {"content": json.dumps({"actions": [
                {"type": "effect", "id": "doorbell", "reason": "звонок", "confidence": "high"}]})}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50,
                      "prompt_tokens_details": {"cached_tokens": 800},
                      "completion_tokens_details": {"reasoning_tokens": 10}},
        })

    async def main():
        runner, base = await serve(handler)
        b = OpenAIBrain(registry(), "gpt-5-mini", Prices(0.25, 0.025, 2.0), extra={"reasoning_effort": "minimal"})
        b.URL = base + "/v1/chat/completions"
        img = encode_jpeg(np.zeros((36, 64, 3), np.uint8))
        r = await b.decide("НОВОЕ:\n[0:01] кто-то звонит в дверь", img)
        await b.close()
        await runner.cleanup()
        return r

    r = asyncio.run(main())
    assert r.error is None, r.error
    assert r.actions == [{"type": "effect", "id": "doorbell", "reason": "звонок", "confidence": "high"}]
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "gpt-5-mini" and body["reasoning_effort"] == "minimal"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # 200 свежих * 0.25 + 800 кэш * 0.025 + 50 * 2.0 (за 1M)
    assert abs(r.cost - (200 * 0.25 + 800 * 0.025 + 50 * 2.0) / 1e6) < 1e-12
    assert r.usage.reasoning_tokens == 10


def test_gemini_request_and_usage(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    seen = {}

    async def handler(req):
        seen["key"] = req.headers.get("x-goog-api-key")
        seen["path"] = req.path
        seen["body"] = await req.json()
        return web.json_response({
            "candidates": [{"content": {"parts": [
                {"text": "думаю...", "thought": True},
                {"text": json.dumps({"actions": [{"type": "background", "id": "rain", "reason": "дождь",
                                                  "confidence": "medium"}]})}]}}],
            "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 30, "thoughtsTokenCount": 70},
        })

    async def main():
        runner, base = await serve(handler)
        b = GeminiBrain(registry(), "gemini-3.5-flash", Prices(1.5, 0.15, 9.0))
        b.URL = base + "/v1beta/models/{model}:generateContent"
        r = await b.decide("НОВОЕ:\n[0:02] опять этот дождь", None)
        await b.close()
        await runner.cleanup()
        return r

    r = asyncio.run(main())
    assert r.error is None, r.error
    assert r.actions[0]["id"] == "rain"
    assert seen["key"] == "g-key" and seen["path"].endswith("gemini-3.5-flash:generateContent")
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert r.usage.output_tokens == 100  # ответ + «размышления»


def test_http_error_is_reported_not_raised(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    async def handler(req):
        return web.json_response({"error": {"message": "rate limit"}}, status=429)

    async def main():
        runner, base = await serve(handler)
        b = OpenAIBrain(registry(), "gpt-5-mini", Prices())
        b.URL = base + "/x"
        r = await b.decide("ctx", None)
        await b.close()
        await runner.cleanup()
        return r

    r = asyncio.run(main())
    assert r.actions == [] and "429" in r.error


def test_missing_key_reported(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = asyncio.run(OpenAIBrain(registry(), "m", Prices()).decide("ctx", None))
    assert "OPENAI_API_KEY" in r.error


def test_schema_and_prompt():
    reg = registry()
    schema = openai_schema(reg)
    ids = schema["properties"]["actions"]["items"]["properties"]["id"]["enum"]
    assert "doorbell" in ids and "restaurant" not in ids and None in ids  # атмосфера — только вручную
    assert "scene_end" not in ids  # отбивка — отдельным типом действия, не как эффект
    # Со словарём музыка выбирается через music (жанр/настроение/…), а не id состояния-файла.
    assert "romantic" not in ids
    props = schema["properties"]["actions"]["items"]["properties"]
    assert "music" in props["type"]["enum"]
    assert "noir" in props["genre"]["enum"] and None in props["mood"]["enum"]
    assert set(schema["properties"]["actions"]["items"]["required"]) >= {"genre", "mood", "energy", "tempo"}
    gem = gemini_schema(reg)["properties"]["actions"]["items"]["properties"]
    assert "tense" in gem["mood"]["enum"]
    text = system_text(reg)
    assert "doorbell" in text and "не создаёт новых событий" in text
    assert "{background_rules}" not in text and "genre:" in text and "После «монтажа»" in text


def test_schema_without_vocabulary_keeps_music_states():
    reg = registry()
    reg.vocabulary = None
    schema = openai_schema(reg)
    ids = schema["properties"]["actions"]["items"]["properties"]["id"]["enum"]
    assert "romantic" in ids and "music" not in schema["properties"]["actions"]["items"]["properties"]["type"]["enum"]
    assert "genre" not in schema["properties"]["actions"]["items"]["properties"]
    text = system_text(reg)
    assert "{background_rules}" not in text and "background — музыка или атмосфера" in text


def test_parse_actions_keeps_music_style():
    raw = json.dumps({"actions": [{"type": "music", "id": None, "reason": "суд", "confidence": "high",
                                   "genre": "noir", "mood": "tense", "energy": "low", "tempo": "slow"}]})
    acts = parse_actions(raw)
    assert acts[0]["genre"] == "noir" and acts[0]["tempo"] == "slow"


def test_parse_actions_limits_and_cleans():
    raw = json.dumps({"actions": [{"type": "effect", "id": "a", "reason": "x" * 500, "confidence": "low"},
                                  {"type": "note", "id": None, "reason": "ресторан", "confidence": "high"},
                                  {"type": "effect", "id": "b", "reason": "", "confidence": "low"}]})
    acts = parse_actions(raw)
    assert len(acts) == 2 and len(acts[0]["reason"]) == 200


def test_scheduler_context_marks_new_lines(system, clock):
    from soundmaster.brain.providers import FakeBrain

    make, engine, mixer, journal = system
    c = make()
    reg = c.registry
    brain = FakeBrain(reg, rules=[("дверь", {"type": "effect", "id": "doorbell", "reason": "r", "confidence": "high"})])
    s = BrainScheduler(brain, c, journal, clock, SchedulerConfig(min_interval=0))
    s.on_transcript(Transcript("Привет, как дела", clock.now() - 3, clock.now() - 2, clock.now(), 0.1))
    asyncio.run(s.step())
    s.on_transcript(Transcript("Кто-то стучит в дверь", clock.now() - 1, clock.now() - 0.5, clock.now(), 0.1))
    asyncio.run(s.step())
    ctx = brain.contexts[-1]
    old, new = ctx.split("НОВОЕ:")
    assert "Привет" in old and "дверь" in new and "Привет" not in new
    assert any(r.get("call") == "suggest_sfx" and r.get("effect") == "doorbell" for r in journal.records)
    # новая сцена очищает контекст модели
    c.new_scene("лес")
    assert not s.transcripts and s.note == ""


def test_scheduler_budget_stops_requests(system, clock):
    from soundmaster.brain.providers import FakeBrain

    make, engine, mixer, journal = system
    c = make()
    brain = FakeBrain(c.registry)
    brain.prices = Prices(input=1e6, output=0)  # каждый запрос «стоит» дорого
    s = BrainScheduler(brain, c, journal, clock, SchedulerConfig(min_interval=0, budget_usd=1.0))
    s.on_transcript(Transcript("раз", 0, clock.now(), clock.now(), 0.1))
    asyncio.run(s.step())
    s.on_transcript(Transcript("два", 0, clock.now(), clock.now(), 0.1))
    assert s.stopped_by_budget and s.should_request() is None
