"""Разметка кандидатов «монтажа»: локальная страница в браузере, ролики не покидают компьютер.

    python -m soundmaster label-scenecuts

Ответы — в data/scenecut/labels.json: {id кандидата: "cut" | "cut_other" | "no" | "unsure"}
(cut_other — монтаж есть, но рамка не на том, кто его делает). Пропущенные программой монтажи Иван
отмечает временем в записи — data/scenecut/manual.json {id видео: [секунды]}; при следующем сборе по ним
нарезаются ролики с меткой «монтаж». Эти файлы и рамки людей (data/scenecut/*.json) отправляются в git;
ролики остаются в recordings/ (не в git).
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

LABELS = ("cut", "cut_other", "no", "unsure")
SERVICE = ("labels.json", "manual.json")


def load_items(data_dir: Path) -> list[dict]:
    items = []
    for p in sorted(data_dir.glob("*.json")):
        if p.name in SERVICE:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        src = d.get("source", "")
        for c in d.get("candidates", []):
            items.append({"id": c["id"], "clip": c["clip"], "wide": c.get("clip_wide"), "video": d["video"],
                          "title": d["title"], "t": c.get("t_clip", c["t_start"]), "sources": c.get("sources", []),
                          "url": src if src.startswith("http") else ""})
    return items


def load_labels(data_dir: Path) -> dict[str, str]:
    p = data_dir / "labels.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_labels(data_dir: Path, labels: dict[str, str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = data_dir / "labels.json.tmp"
    tmp.write_text(json.dumps(dict(sorted(labels.items())), ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(data_dir / "labels.json")


def save_label(data_dir: Path, cid: str, label: str | None) -> dict[str, str]:
    labels = load_labels(data_dir)
    if label is None:
        labels.pop(cid, None)
    else:
        if label not in LABELS:
            raise ValueError(f"метка {label!r}: нужна одна из {LABELS}")
        labels[cid] = label
    save_labels(data_dir, labels)
    return labels


def load_manual(data_dir: Path) -> dict[str, list[float]]:
    p = data_dir / "manual.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_manual(data_dir: Path, video: str, t: float, remove: bool = False) -> dict[str, list[float]]:
    """Добавить (или убрать) момент пропущенного монтажа: id видео и секунда записи."""
    manual = load_manual(data_dir)
    times = [x for x in manual.get(video, []) if abs(x - t) > 0.5]
    if not remove:
        times.append(round(float(t), 1))
    if times:
        manual[video] = sorted(times)
    else:
        manual.pop(video, None)
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = data_dir / "manual.json.tmp"
    tmp.write_text(json.dumps(manual, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(data_dir / "manual.json")
    return manual


def videos(data_dir: Path) -> list[dict]:
    out = []
    for p in sorted(data_dir.glob("*.json")):
        if p.name not in SERVICE:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({"video": d["video"], "title": d["title"], "seconds": d.get("seconds", 0)})
    return out


def make_app(data_dir: Path, clip_dir: Path):
    from aiohttp import web

    async def index(_req):
        return web.Response(text=PAGE, content_type="text/html")

    async def items(_req):
        return web.json_response({"items": [i for i in load_items(data_dir) if (clip_dir / i["clip"]).exists()],
                                  "labels": load_labels(data_dir), "manual": load_manual(data_dir),
                                  "videos": videos(data_dir)})

    async def manual(req):
        body = await req.json()
        try:
            m = save_manual(data_dir, str(body["video"]), float(body["t"]), bool(body.get("remove")))
        except (KeyError, ValueError, TypeError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "manual": m})

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
    app.add_routes([web.get("/", index), web.get("/api/items", items), web.post("/api/label", label), web.post("/api/manual", manual),
                    web.get("/clips/{name}", clip)])
    return app


def serve(data_dir: Path, clip_dir: Path, port: int = 8766, open_browser: bool = True) -> None:
    from aiohttp import web

    url = f"http://127.0.0.1:{port}"
    print(f"Разметка: {url}  (остановить — Ctrl+C)")
    if open_browser:
        webbrowser.open(url)
    web.run_app(make_app(data_dir, clip_dir), host="127.0.0.1", port=port, print=None)


PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Разметка «монтажа»</title>
<style>
:root{--bg:#f4f2ee;--fg:#1d1b18;--muted:#6f6a61;--line:#ddd8cf;--surface:#fff;--cut:#2f7d4f;--no:#b3261e;--unsure:#8a6d1d;--accent:#b5462c}
@media (prefers-color-scheme:dark){:root{--bg:#15130f;--fg:#efeae2;--muted:#a39b8f;--line:#36312a;--surface:#1f1c17;--cut:#5cc488;--no:#f07a70;--unsure:#e0c060;--accent:#e0805f;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:16px;display:grid;gap:14px}
h1{font-size:20px;margin:4px 0}.muted{color:var(--muted)}
video{width:100%;background:#000;border-radius:10px;aspect-ratio:16/9}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
select{font:inherit;padding:6px 8px;border-radius:8px;border:1px solid var(--line);background:var(--surface);color:var(--fg);max-width:100%}
button{font:inherit;padding:10px 16px;border-radius:10px;border:1px solid var(--line);background:var(--surface);color:var(--fg);cursor:pointer}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.cut[aria-pressed=true]{background:var(--cut);color:#fff;border-color:var(--cut)}
.no[aria-pressed=true]{background:var(--no);color:#fff;border-color:var(--no)}
.cut2[aria-pressed=true]{background:var(--cut);color:#fff;border-color:var(--cut);opacity:.8}
input[type=text]{font:inherit;padding:6px 8px;border-radius:8px;border:1px solid var(--line);background:var(--surface);color:var(--fg);width:6em}
a{color:var(--accent)}fieldset{border:1px solid var(--line);border-radius:10px;padding:10px 14px;display:grid;gap:8px}
#bWide[aria-pressed=true],#bSlow[aria-pressed=true]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.unsure[aria-pressed=true]{background:var(--unsure);color:#fff;border-color:var(--unsure)}
.grow{flex:1}kbd{font:12px ui-monospace,Menlo,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px}
.bar{height:6px;background:var(--line);border-radius:3px}.bar i{display:block;height:100%;background:var(--accent);border-radius:3px;width:0}
</style></head><body><div class="wrap">
<h1>Разметка «монтажа»</h1>
<p class="muted">Вопрос один: есть ли в ролике «монтаж» (актёр проходит перед сценой, чтобы начать новую). Жёлтая рамка —
человек, на которого среагировала программа, — подсказка: если монтаж есть, но рамка на другом человеке, жмите
«Монтаж, рамка мимо». Непонятно, откуда человек вышел и что было после, — «Шире» (5 с до прохода и 7 после).</p>
<div class="row"><b id="pos"></b><span class="muted" id="title"></span><a id="yt" target="_blank" rel="noopener">открыть на YouTube</a><span class="grow"></span><span id="stat" class="muted"></span></div>
<div class="bar"><i id="prog"></i></div>
<video id="v" autoplay muted loop playsinline></video>
<div class="row">
<button class="cut" id="bCut">Монтаж <kbd>1</kbd></button>
<button class="cut2" id="bCut2">Монтаж, рамка мимо <kbd>6</kbd></button>
<button class="no" id="bNo">Не монтаж <kbd>2</kbd></button>
<button class="unsure" id="bUnsure">Непонятно <kbd>3</kbd></button>
<span class="grow"></span>
<button id="bWide" aria-pressed="false">Шире <kbd>4</kbd></button><button id="bSlow" aria-pressed="false">Медленно <kbd>5</kbd></button>
<button id="bPrev">← Назад</button><button id="bNext">Дальше →</button>
</div>
<div class="row"><label class="muted" for="vid">Видео:</label><select id="vid"><option value="">все</option></select>
<label class="muted"><input type="checkbox" id="skip" checked> Показывать только неразмеченные</label></div>
<p class="muted">Пробел — пауза, <kbd>←</kbd> <kbd>→</kbd> — листать. Отметка сохраняется сразу; можно закрыть и продолжить позже.</p>
<fieldset><legend>Пропущенный монтаж</legend>
<p class="muted" style="margin:0">Монтаж, которого нет среди роликов: выберите видео и время в записи (как на YouTube). При следующем сборе
программа нарежет по нему ролик с меткой «монтаж».</p>
<div class="row"><select id="mVid"></select><input type="text" id="mTime" placeholder="12:30" aria-label="Время в записи">
<button id="mAdd">Добавить</button><span id="mMsg" class="muted"></span></div>
<div id="mList" class="row"></div></fieldset>
</div><script>
let all=[],items=[],labels={},i=0,wide=false,manual={},vids=[];const $=id=>document.getElementById(id),v=$("v");
function videos(){const sel=$("vid"),cur=sel.value,seen=new Map();all.forEach(x=>{const s=seen.get(x.video)||{title:x.title,n:0,done:0};
 s.n++;if(labels[x.id])s.done++;seen.set(x.video,s)});sel.innerHTML='<option value="">все видео</option>'+
 [...seen].map(([k,s])=>`<option value="${k}">${s.title.replace(/</g,"&lt;")} — ${s.done}/${s.n}</option>`).join("");sel.value=cur}
function filter(){const f=$("vid").value;items=f?all.filter(x=>x.video===f):all}
const fmt=t=>`${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,"0")}`;
function renderManual(){const sel=$("mVid"),cur=sel.value||$("vid").value;sel.innerHTML=vids.map(x=>`<option value="${x.video}">${x.title.replace(/</g,"&lt;")}</option>`).join("");
 if(cur)sel.value=cur;const list=manual[sel.value]||[];$("mList").innerHTML=list.length?list.map(t=>`<button data-t="${t}" title="Убрать">${fmt(t)} ✕</button>`).join(""):'<span class="muted">для этого видео отметок нет</span>';
 $("mList").querySelectorAll("button").forEach(b=>b.onclick=()=>saveManual(sel.value,+b.dataset.t,true))}
async function saveManual(video,t,remove){const r=await fetch("/api/manual",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({video,t,remove})});
 const d=await r.json();if(!r.ok){$("mMsg").textContent="Не сохранилось";return}manual=d.manual;$("mMsg").textContent=remove?"Убрано":"Добавлено — появится после следующего сбора";renderManual()}
$("mVid").onchange=renderManual;
$("mAdd").onclick=()=>{const m=$("mTime").value.trim().match(/^(?:(\d+):)?(\d{1,2}):(\d{2})$/);
 if(!m){$("mMsg").textContent="Время в виде 12:30 или 1:02:30";return}const t=(+(m[1]||0))*3600+(+m[2])*60+(+m[3]);
 const vd=vids.find(x=>x.video===$("mVid").value);if(vd&&vd.seconds&&t>vd.seconds){$("mMsg").textContent="Видео короче";return}
 saveManual($("mVid").value,t,false);$("mTime").value=""};
async function load(){const r=await fetch("/api/items");const d=await r.json();all=d.items;labels=d.labels;manual=d.manual||{};vids=d.videos||[];videos();filter();renderManual();
 i=Math.max(0,items.findIndex(x=>!labels[x.id]));show()}
$("vid").onchange=()=>{if($("vid").value){$("mVid").value=$("vid").value;renderManual()}filter();i=Math.max(0,items.findIndex(x=>!labels[x.id]));show()};
function show(){if(!items.length){$("pos").textContent="Пока нет роликов — сбор ещё идёт";return}
 i=(i+items.length)%items.length;const it=items[i];$("pos").textContent=`${i+1} / ${items.length}`;
 $("title").textContent=`${it.title} · ${fmt(it.t)}`;
 const yt=$("yt");if(it.url){yt.hidden=false;yt.href=`https://www.youtube.com/watch?v=${it.video}&t=${Math.max(0,Math.floor(it.t)-3)}s`}else yt.hidden=true;
 const clip=wide&&it.wide?it.wide:it.clip;if(!v.src.endsWith(clip)){v.src="/clips/"+clip;v.playbackRate=$("bSlow").getAttribute("aria-pressed")==="true"?0.5:1;v.play().catch(()=>{})}
 $("bWide").disabled=!it.wide;$("bWide").setAttribute("aria-pressed",wide&&!!it.wide);
 const l=labels[it.id];[["bCut","cut"],["bCut2","cut_other"],["bNo","no"],["bUnsure","unsure"]].forEach(([b,k])=>$(b).setAttribute("aria-pressed",l===k));
 const n=Object.keys(labels).filter(k=>items.some(x=>x.id===k)).length;$("stat").textContent=`размечено ${n} из ${items.length}`;
 $("prog").style.width=(n/items.length*100)+"%"}
function step(d){if(!items.length)return;const only=$("skip").checked;for(let k=0;k<items.length;k++){i=(i+d+items.length)%items.length;
 if(!only||!labels[items[i].id])break}show()}
async function mark(l){const it=items[i];if(!it)return;const cur=labels[it.id];const nl=cur===l?null:l;
 const r=await fetch("/api/label",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:it.id,label:nl})});
 if(!r.ok){alert("Не сохранилось");return}if(nl)labels[it.id]=nl;else delete labels[it.id];videos();show();if(nl)setTimeout(()=>step(1),150)}
$("bCut").onclick=()=>mark("cut");$("bCut2").onclick=()=>mark("cut_other");$("bNo").onclick=()=>mark("no");$("bUnsure").onclick=()=>mark("unsure");
function toggleWide(){wide=!wide;show()}
function toggleSlow(){const b=$("bSlow"),on=b.getAttribute("aria-pressed")!=="true";b.setAttribute("aria-pressed",on);v.playbackRate=on?0.5:1}
$("bWide").onclick=toggleWide;$("bSlow").onclick=toggleSlow;
$("bPrev").onclick=()=>{i--;show()};$("bNext").onclick=()=>{i++;show()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;
 if(e.key==="1")mark("cut");else if(e.key==="2")mark("no");else if(e.key==="3")mark("unsure");else if(e.key==="4")toggleWide();else if(e.key==="5")toggleSlow();else if(e.key==="6")mark("cut_other");
 else if(e.key==="ArrowRight"){i++;show()}else if(e.key==="ArrowLeft"){i--;show()}
 else if(e.key===" "){e.preventDefault();v.paused?v.play():v.pause()}});
setInterval(async()=>{const r=await fetch("/api/items");const d=await r.json();if(d.items.length!==all.length){const cur=items[i]&&items[i].id;
 all=d.items;labels=d.labels;manual=d.manual||{};vids=d.videos||[];videos();filter();renderManual();const j=items.findIndex(x=>x.id===cur);i=j<0?i:j;show()}},20000);
load();
</script></body></html>"""
