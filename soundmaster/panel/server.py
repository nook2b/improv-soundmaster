"""Пульт техника: веб-страница на localhost + WebSocket. Работает без интернета."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from ..app import App

STATIC = Path(__file__).parent / "static"
SEND_TIMEOUT = 1.0  # медленный клиент дольше этого — отключаем, цикл пульта его не ждёт


def handle_command(app: App, msg: dict[str, Any]) -> dict[str, Any] | None:
    c = app.controller
    cmd = msg.get("cmd")
    if cmd == "play":
        d = c.manual_play(msg["effect"])
        return {"result": d.result, "blocked_by": d.blocked_by, "effect": d.effect}
    if cmd == "stop_effect":
        c.manual_stop(msg["effect"])
    elif cmd == "music":
        return {"result": c.manual_music(msg.get("state"))}
    elif cmd == "pool":
        return {"result": c.manual_pool(str(msg["genre"]))}
    elif cmd == "release_music":
        return {"result": "released" if c.release_music() else "nothing_to_release"}
    elif cmd == "stop_channel":
        c.manual_stop_channel("ambience" if msg.get("channel") == "ambience" else "music")
    elif cmd == "confirm":
        d = c.confirm(int(msg["id"]))
        return {"result": d.result, "blocked_by": d.blocked_by}
    elif cmd == "confirm_kind":
        # Отдельная клавиша на каждый тип подсказки: подтверждается самая свежая подсказка этого типа.
        kinds = {"music", "music_style"} if msg["kind"] == "music" else {msg["kind"]}
        items = [s for s in c.state.suggestions.values() if s.kind in kinds]
        if items:
            d = c.confirm(max(items, key=lambda s: s.t_created).id)
            return {"result": d.result, "blocked_by": d.blocked_by}
    elif cmd == "dismiss":
        c.dismiss(int(msg["id"]))
    elif cmd == "new_scene":
        c.new_scene(msg.get("hint", ""), keep_music=bool(msg.get("keep_music", False)))
    elif cmd == "automation":
        c.set_automation(bool(msg["enabled"]))
    elif cmd == "stop_ai":
        c.stop_ai_sound()
    elif cmd == "stop_all":
        c.stop_all()
    elif cmd == "generator":
        app.music.set_generator_enabled(bool(msg["enabled"]))
    elif cmd == "set_mode":
        c.set_mode(msg["effect"], msg["mode"])
    elif cmd == "fire":
        d = c.fire("operator")
        return {"result": d.result, "effect": d.effect, "blocked_by": d.blocked_by}
    elif cmd == "disarm":
        if c.state.armed:
            c._disarm("operator")
    elif cmd == "brain":
        if app.scheduler:
            app.scheduler.enabled = bool(msg["enabled"])
            app.journal.write("operator", action="brain_enabled", enabled=app.scheduler.enabled)
    elif cmd == "scene_cut_mode":
        return {"result": c.set_scene_cut_mode(str(msg["mode"]))}  # неизвестный режим → ошибка на пульт
    elif cmd == "mark":
        # Отметка техника «здесь было событие» — черновик эталонной дорожки для разметки.
        app.journal.write("mark", label=msg.get("label", ""), scene_id=c.state.scene_id)
    else:
        return {"error": f"неизвестная команда {cmd}"}
    return None


def make_web_app(app: App) -> web.Application:
    clients: set[web.WebSocketResponse] = set()

    async def index(_req: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC / "index.html")

    async def library(_req: web.Request) -> web.Response:
        return web.json_response(app.describe_library())

    async def ws_handler(req: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=5)
        await ws.prepare(req)
        clients.add(ws)
        try:
            async for m in ws:
                if m.type != WSMsgType.TEXT:
                    continue
                try:
                    reply = handle_command(app, json.loads(m.data))
                except Exception as e:  # noqa: BLE001 — ошибку показать на пульте, сервер не ронять
                    reply = {"error": f"{type(e).__name__}: {e}"}
                if reply:
                    await ws.send_json({"type": "reply", **reply})
                await ws.send_json({"type": "state", **app.status()})
        finally:
            clients.discard(ws)
        return ws

    sending: dict[web.WebSocketResponse, asyncio.Task] = {}

    async def send_one(ws: web.WebSocketResponse, data: str) -> None:
        try:
            await asyncio.wait_for(ws.send_str(data), SEND_TIMEOUT)
        except Exception:  # noqa: BLE001 — оборванный или зависший клиент: отключаем, остальные работают
            clients.discard(ws)
            if not ws.closed:
                asyncio.get_running_loop().create_task(ws.close())
        finally:
            sending.pop(ws, None)

    def broadcast(data: str) -> None:
        """Состояние всем клиентам в отдельных задачах: tick не ждёт сокеты и не падает от них."""
        for ws in list(clients):
            if ws.closed:
                clients.discard(ws)
            elif ws not in sending:  # предыдущее ещё не ушло — это состояние пропускаем
                sending[ws] = asyncio.create_task(send_one(ws, data))

    async def ticker(_web_app: web.Application):
        async def loop():
            n = 0
            while True:
                try:
                    app.controller.tick()
                except Exception as e:  # noqa: BLE001 — цикл пульта не должен умирать
                    app.journal.write("error", where="tick", error=f"{type(e).__name__}: {e}")
                n += 1
                if n % 4 == 0 and clients:
                    try:
                        state = json.dumps({"type": "state", **app.status()}, ensure_ascii=False)
                    except Exception as e:  # noqa: BLE001 — без состояния пульт поживёт, tick — нет
                        app.journal.write("error", where="status", error=f"{type(e).__name__}: {e}")
                    else:
                        broadcast(state)
                await asyncio.sleep(0.05)

        task = asyncio.create_task(loop())
        bg = app.attach_loop(asyncio.get_running_loop())
        yield
        task.cancel()
        for t in list(sending.values()):
            t.cancel()
        for t in bg:
            t.cancel()
        await app.detach_loop()

    web_app = web.Application()
    web_app.router.add_get("/", index)
    web_app.router.add_get("/library", library)
    web_app.router.add_get("/ws", ws_handler)
    web_app.router.add_static("/static", STATIC)
    web_app.cleanup_ctx.append(ticker)
    return web_app


def run_panel(app: App, host: str, port: int) -> None:
    app.start()
    try:
        web.run_app(make_web_app(app), host=host, port=port, print=lambda *_: None)
    finally:
        app.stop()
