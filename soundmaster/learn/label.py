"""Разметка кандидатов «монтажа»: локальная страница в браузере, ролики не покидают компьютер.

    python -m soundmaster label-scenecuts

Ответы — в data/scenecut/labels.json: {id кандидата: "cut" | "no" | "unsure"}. Этот файл и рамки
людей (data/scenecut/*.json) отправляются в git; ролики остаются в recordings/ (не в git).
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

LABELS = ("cut", "no", "unsure")


def load_items(data_dir: Path) -> list[dict]:
    items = []
    for p in sorted(data_dir.glob("*.json")):
        if p.name == "labels.json":
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for c in d.get("candidates", []):
            items.append({"id": c["id"], "clip": c["clip"], "video": d["video"], "title": d["title"],
                          "t": c["t_start"], "sources": c.get("sources", [])})
    return items


def load_labels(data_dir: Path) -> dict[str, str]:
    p = data_dir / "labels.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_label(data_dir: Path, cid: str, label: str | None) -> dict[str, str]:
    labels = load_labels(data_dir)
    if label is None:
        labels.pop(cid, None)
    else:
        if label not in LABELS:
            raise ValueError(f"метка {label!r}: нужна одна из {LABELS}")
        labels[cid] = label
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = data_dir / "labels.json.tmp"
    tmp.write_text(json.dumps(dict(sorted(labels.items())), ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(data_dir / "labels.json")
    return labels


def make_app(data_dir: Path, clip_dir: Path):
    from aiohttp import web

    async def index(_req):
        return web.Response(text=PAGE, content_type="text/html")

    async def items(_req):
        return web.json_response({"items": [i for i in load_items(data_dir) if (clip_dir / i["clip"]).exists()],
                                  "labels": load_labels(data_dir)})

    async def label(req):
        body = await req.json()
        try:
            labels = save_label(data_dir, str(body["id"]), body.get("label"))
        except (KeyError, ValueError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "count": len(labels)})

    async def clip(req):
        name = req.match_info["name"]
        path = clip_dir / name
        if "/" in name or "\\" in name or ".." in name or not path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/api/items", items), web.post("/api/label", label),
                    web.get("/clips/{name}", clip)])
    return app


def serve(data_dir: Path, clip_dir: Path, port: int = 8766, open_browser: bool = True) -> None:
    from aiohttp import web

    url = f"http://127.0.0.1:{port}"
    print(f"Разметка: {url}  (остановить — Ctrl+C)")
    if open_browser:
        webbrowser.open(url)
    web.run_app(make_app(data_dir, clip_dir), host="127.0.0.1", port=port, print=None)


PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Разметка «монтажа»</title>
<style>
:root{--bg:#f4f2ee;--fg:#1d1b18;--muted:#6f6a61;--line:#ddd8cf;--surface:#fff;--cut:#2f7d4f;--no:#b3261e;--unsure:#8a6d1d;--accent:#b5462c}
@media (prefers-color-scheme:dark){:root{--bg:#15130f;--fg:#efeae2;--muted:#a39b8f;--line:#36312a;--surface:#1f1c17;--cut:#5cc488;--no:#f07a70;--unsure:#e0c060;--accent:#e0805f;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:16px;display:grid;gap:14px}
h1{font-size:20px;margin:4px 0}.muted{color:var(--muted)}
video{width:100%;background:#000;border-radius:10px;aspect-ratio:16/9}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{font:inherit;padding:10px 16px;border-radius:10px;border:1px solid var(--line);background:var(--surface);color:var(--fg);cursor:pointer}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.cut[aria-pressed=true]{background:var(--cut);color:#fff;border-color:var(--cut)}
.no[aria-pressed=true]{background:var(--no);color:#fff;border-color:var(--no)}
.unsure[aria-pressed=true]{background:var(--unsure);color:#fff;border-color:var(--unsure)}
.grow{flex:1}kbd{font:12px ui-monospace,Menlo,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px}
.bar{height:6px;background:var(--line);border-radius:3px}.bar i{display:block;height:100%;background:var(--accent);border-radius:3px;width:0}
</style></head><body><div class="wrap">
<h1>Разметка «монтажа»</h1>
<p class="muted">«Монтаж» — актёр проходит перед сценой, чтобы начать новую сцену. Жёлтая рамка — человек, на которого
среагировала программа (если рамки нет — сработал детектор по картинке). Ролик повторяется по кругу.</p>
<div class="row"><b id="pos"></b><span class="muted" id="title"></span><span class="grow"></span><span id="stat" class="muted"></span></div>
<div class="bar"><i id="prog"></i></div>
<video id="v" autoplay muted loop playsinline></video>
<div class="row">
<button class="cut" id="bCut">Монтаж <kbd>1</kbd></button>
<button class="no" id="bNo">Не монтаж <kbd>2</kbd></button>
<button class="unsure" id="bUnsure">Непонятно <kbd>3</kbd></button>
<span class="grow"></span>
<button id="bPrev">← Назад</button><button id="bNext">Дальше →</button>
</div>
<label class="muted"><input type="checkbox" id="skip" checked> Показывать только неразмеченные</label>
<p class="muted">Пробел — пауза, <kbd>←</kbd> <kbd>→</kbd> — листать. Отметка сохраняется сразу; можно закрыть и продолжить позже.</p>
</div><script>
let items=[],labels={},i=0;const $=id=>document.getElementById(id),v=$("v");
async function load(){const r=await fetch("/api/items");const d=await r.json();items=d.items;labels=d.labels;
 i=Math.max(0,items.findIndex(x=>!labels[x.id]));show()}
function show(){if(!items.length){$("pos").textContent="Пока нет роликов — сбор ещё идёт";return}
 i=(i+items.length)%items.length;const it=items[i];$("pos").textContent=`${i+1} / ${items.length}`;
 $("title").textContent=`${it.title} · ${Math.floor(it.t/60)}:${String(Math.floor(it.t%60)).padStart(2,"0")}`;
 if(!v.src.endsWith(it.clip)){v.src="/clips/"+it.clip;v.play().catch(()=>{})}
 const l=labels[it.id];[["bCut","cut"],["bNo","no"],["bUnsure","unsure"]].forEach(([b,k])=>$(b).setAttribute("aria-pressed",l===k));
 const n=Object.keys(labels).filter(k=>items.some(x=>x.id===k)).length;$("stat").textContent=`размечено ${n} из ${items.length}`;
 $("prog").style.width=(n/items.length*100)+"%"}
function step(d){if(!items.length)return;const only=$("skip").checked;for(let k=0;k<items.length;k++){i=(i+d+items.length)%items.length;
 if(!only||!labels[items[i].id])break}show()}
async function mark(l){const it=items[i];if(!it)return;const cur=labels[it.id];const nl=cur===l?null:l;
 const r=await fetch("/api/label",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:it.id,label:nl})});
 if(!r.ok){alert("Не сохранилось");return}if(nl)labels[it.id]=nl;else delete labels[it.id];show();if(nl)setTimeout(()=>step(1),150)}
$("bCut").onclick=()=>mark("cut");$("bNo").onclick=()=>mark("no");$("bUnsure").onclick=()=>mark("unsure");
$("bPrev").onclick=()=>{i--;show()};$("bNext").onclick=()=>{i++;show()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;
 if(e.key==="1")mark("cut");else if(e.key==="2")mark("no");else if(e.key==="3")mark("unsure");
 else if(e.key==="ArrowRight"){i++;show()}else if(e.key==="ArrowLeft"){i--;show()}
 else if(e.key===" "){e.preventDefault();v.paused?v.play():v.pause()}});
setInterval(async()=>{const r=await fetch("/api/items");const d=await r.json();if(d.items.length!==items.length){const cur=items[i]&&items[i].id;
 items=d.items;labels=d.labels;const j=items.findIndex(x=>x.id===cur);i=j<0?i:j;show()}},20000);
load();
</script></body></html>"""
