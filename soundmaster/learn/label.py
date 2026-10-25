"""Разметка кандидатов «монтажа»: одна локальная страница в браузере.

    python -m soundmaster label-scenecuts

Видео играет во встроенном плеере YouTube (на диск не скачивается); поверх — рамки людей из сбора,
жёлтая — человек, на которого среагировала программа. Под плеером — таймлайн шоу с роликами-кандидатами.

Ответы — в data/scenecut/labels.json: {id кандидата: "cut" | "cut_other" | "no" | "unsure"}
(cut_other — монтаж есть, но нужного человека нет среди рамок). Если жёлтая рамка не на том, Иван
указывает нужного человека кликом — data/scenecut/tracks.json {id кандидата: id траектории}.
Пропущенные программой монтажи — временем в записи, data/scenecut/manual.json {id видео: [секунды]};
при следующем сборе по ним появляются кандидаты с меткой «монтаж». Всё это — в git (только числа).
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

LABELS = ("cut", "cut_other", "no", "unsure")
SERVICE = ("labels.json", "manual.json", "tracks.json")


def load_items(data_dir: Path) -> list[dict]:
    items = []
    for p in sorted(data_dir.glob("*.json")):
        if p.name in SERVICE:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        src = d.get("source", "")
        for c in d.get("candidates", []):
            items.append({"id": c["id"], "clip": c.get("clip"), "wide": c.get("clip_wide"), "video": d["video"],
                          "title": d["title"], "t": c.get("t_clip", c["t_start"]), "sources": c.get("sources", []),
                          "t_start": c["t_start"], "t_end": c.get("t_pass_end", c.get("t_end", c["t_start"])),
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
            src = d.get("source", "")
            out.append({"video": d["video"], "title": d["title"], "seconds": d.get("seconds", 0),
                        "youtube": "youtube.com" in src or "youtu.be" in src})
    return out


def load_tracks(data_dir: Path) -> dict[str, str]:
    """Кого Иван указал как делающего монтаж, если программа выбрала не того: {id ролика: id траектории}."""
    p = data_dir / "tracks.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_track(data_dir: Path, cid: str, track: str | None) -> dict[str, str]:
    tracks = load_tracks(data_dir)
    if track is None:
        tracks.pop(cid, None)
    else:
        tracks[cid] = str(track)
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = data_dir / "tracks.json.tmp"
    tmp.write_text(json.dumps(dict(sorted(tracks.items())), ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(data_dir / "tracks.json")
    return tracks


def boxes(data_dir: Path, video: str) -> list[dict]:
    """Рамки людей по роликам одного видео — для рамок поверх плеера: [{id, track, frames: [[t, {id: рамка}]]}]."""
    p = data_dir / f"{video}.json"
    if any(ch in video for ch in "/\\:") or ".." in video or not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    fixed = load_tracks(data_dir)
    return [{"id": c["id"], "track": fixed.get(c["id"], str(c.get("track", -1))), "fixed": c["id"] in fixed,
             "frames": c.get("frames", [])} for c in d.get("candidates", [])]


def make_app(data_dir: Path, clip_dir: Path):
    from aiohttp import web

    async def index(_req):
        return web.Response(text=PAGE, content_type="text/html")

    async def watch(_req):  # старый адрес второй страницы
        raise web.HTTPFound("/")

    async def items(_req):
        return web.json_response({"items": load_items(data_dir),
                                  "labels": load_labels(data_dir), "manual": load_manual(data_dir),
                                  "videos": videos(data_dir)})

    async def boxes_api(req):
        return web.json_response({"boxes": boxes(data_dir, req.query.get("video", ""))})

    async def track(req):
        body = await req.json()
        try:
            cid = str(body["id"])
            save_track(data_dir, cid, body.get("track"))
            if body.get("track") is not None:
                save_label(data_dir, cid, "cut")  # указали, кто делал монтаж, — значит, монтаж
        except (KeyError, ValueError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True})

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
        if any(ch in name for ch in "/\\:") or ".." in name or not path.exists():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/watch", watch), web.get("/api/items", items), web.post("/api/label", label), web.post("/api/manual", manual), web.get("/api/boxes", boxes_api), web.post("/api/track", track),
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
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Разметка монтажей</title>
<style>
:root{--bg:#f4f2ee;--fg:#1d1b18;--muted:#6f6a61;--line:#ddd8cf;--surface:#fff;--cut:#2f7d4f;--no:#b3261e;--unsure:#c9a227;--todo:#9a948a;--manual:#2f6fd0;--accent:#b5462c}
@media (prefers-color-scheme:dark){:root{--bg:#15130f;--fg:#efeae2;--muted:#a39b8f;--line:#36312a;--surface:#1f1c17;--cut:#5cc488;--no:#f07a70;--unsure:#e0c060;--todo:#6f6a61;--manual:#6fa8ff;--accent:#e0805f;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px;display:grid;gap:12px}
h1{font-size:20px;margin:0}h2{font-size:13px;margin:0;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.muted{color:var(--muted)}.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.grow{flex:1}
select,button{font:inherit;padding:8px 12px;border-radius:10px;border:1px solid var(--line);background:var(--surface);color:var(--fg);cursor:pointer;max-width:100%}
button:disabled{opacity:.45;cursor:default}button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
kbd{font:12px ui-monospace,Menlo,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-left:4px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--cut)}
.vids{display:flex;flex-wrap:wrap;gap:6px}.vids button{padding:5px 10px;font-size:13px}.vids button.on{border-color:var(--fg);font-weight:600}
.vids .done{color:var(--cut)}
.player{position:relative;width:100%;aspect-ratio:16/9;background:#000;border-radius:10px;overflow:hidden}
.player>div,.player iframe{position:absolute;inset:0;width:100%;height:100%}
#ov{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#ov.pick{pointer-events:auto;cursor:crosshair;background:rgba(0,0,0,.18)}
canvas.tl{width:100%;display:block;border-radius:6px;cursor:pointer;background:var(--surface);border:1px solid var(--line)}
#tlFull{height:40px}#tlZoom{height:56px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px 14px;display:grid;gap:10px}
.cut[aria-pressed=true]{background:var(--cut);color:#fff;border-color:var(--cut)}
.no[aria-pressed=true]{background:var(--no);color:#fff;border-color:var(--no)}
.unsure[aria-pressed=true]{background:var(--unsure);color:#000;border-color:var(--unsure)}
#bPick[aria-pressed=true]{background:#ffd400;color:#000;border-color:#ffd400}
.tog[aria-pressed=true]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
#bMiss{border-color:var(--manual);color:var(--manual)}
.status{font-weight:600}.status.cut{color:var(--cut)}.status.no{color:var(--no)}.status.unsure{color:var(--unsure)}
.legend{font-size:13px}.legend span{display:inline-flex;align-items:center;gap:4px;margin-right:12px}.legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
#msg{min-height:1.4em}
</style></head><body><div class="wrap">

<div class="row"><h1>Разметка монтажей</h1><span class="grow"></span><span class="muted" id="total"></span></div>
<div class="bar" title="Размечено роликов во всех видео"><i id="totalBar"></i></div>
<div class="vids" id="vids" aria-label="Видео"></div>

<div class="player"><div id="yt"></div><canvas id="ov"></canvas></div>
<canvas class="tl" id="tlFull" aria-label="Всё видео"></canvas>
<canvas class="tl" id="tlZoom" aria-label="±1 минута вокруг текущего места"></canvas>
<div class="row legend muted"><span><i style="background:var(--cut)"></i>монтаж</span><span><i style="background:var(--no)"></i>не монтаж</span>
 <span><i style="background:var(--unsure)"></i>непонятно</span><span><i style="background:var(--todo)"></i>не размечено</span>
 <span><i style="background:var(--manual)"></i>пропущенный, отмечен вручную</span><span class="grow"></span>
 <label><input type="checkbox" id="others" checked> рамки всех людей</label></div>

<div class="panel" aria-label="Текущий ролик">
 <div class="row"><b id="clipPos">—</b><span class="status" id="clipStatus"></span><span class="grow"></span><span class="muted" id="clock">0:00</span></div>
 <h2>1. Есть ли здесь монтаж?</h2>
 <div class="row">
  <button class="cut" id="bCut">Монтаж<kbd>1</kbd></button>
  <button class="no" id="bNo">Не монтаж<kbd>2</kbd></button>
  <button class="unsure" id="bUnsure">Непонятно<kbd>3</kbd></button>
 </div>
 <h2>2. Если монтаж, а жёлтая рамка не на том человеке</h2>
 <div class="row">
  <button id="bPick" aria-pressed="false">Указать человека кликом<kbd>B</kbd></button>
  <button class="cut" id="bCut2">Нужного человека нет в рамках<kbd>6</kbd></button>
 </div>
 <h2>3. Переход</h2>
 <div class="row">
  <button id="bPrev">← Предыдущий<kbd>P</kbd></button>
  <button id="bNext" title="Следующий неразмеченный в этом шоу; если всё размечено — просто следующий">Следующий →<kbd>N</kbd></button>
  <button class="tog" id="bLoop" aria-pressed="true">Повтор ролика<kbd>L</kbd></button>
  <select id="rate" aria-label="Скорость"><option value="0.5">×0.5</option><option value="1" selected>×1</option><option value="1.5">×1.5</option><option value="2">×2</option></select>
  <label class="muted"><input type="checkbox" id="auto" checked> после отметки — сразу к следующему</label>
 </div>
</div>


<div class="panel">
 <h2>Пропущенный монтаж</h2>
 <div class="row"><button id="bMiss">Здесь монтаж, а ролика нет<kbd>M</kbd></button>
 <span class="muted">Смотрите шоу (можно на ×1.5–2); увидели монтаж без полоски на таймлайне — пауза и M. Убрать отметку — клик по синему флажку на нижнем таймлайне с Alt.</span></div>
</div>
<p class="muted" id="msg"></p>
<p class="muted">Пробел — пауза, <kbd>←</kbd><kbd>→</kbd> — 5 с (с Shift — 1 с), клик по таймлайну — перемотка. Отметки сохраняются сразу; закончили — <code>git add data/scenecut</code>, <code>git commit</code>, <code>git push</code>.</p>
</div>
<script src="https://www.youtube.com/iframe_api"></script>
<script>
const $=id=>document.getElementById(id);
let all=[],labels={},manual={},vids=[],video="",dur=1,player=null,ready=false,t=0,bx=[],cur=null,pick=false,loop=true;
const NAMES={cut:"монтаж",cut_other:"монтаж (человека нет в рамках)",no:"не монтаж",unsure:"непонятно"};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt=s=>{s=Math.max(0,Math.floor(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=String(s%60).padStart(2,"0");return h?`${h}:${String(m).padStart(2,"0")}:${x}`:`${m}:${x}`};
const isCut=l=>l==="cut"||l==="cut_other";
const cands=()=>all.filter(x=>x.video===video).sort((a,b)=>a.t_start-b.t_start);
const curC=()=>all.find(x=>x.id===cur)||null;
const say=m=>{$("msg").textContent=m};
async function post(url,body){const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});if(!r.ok){say("Не сохранилось — проверьте, что окно с командой label-scenecuts открыто");return null}return r.json()}

// ---- данные и прогресс ----
async function load(){const r=await fetch("/api/items");const d=await r.json();all=d.items;labels=d.labels;manual=d.manual||{};vids=(d.videos||[]).filter(v=>v.youtube);
 if(!vids.length){say("Пока нет собранных видео с YouTube — сначала запустите collect-scenecuts.");return}
 if(!video||!vids.some(v=>v.video===video)){const first=vids.find(v=>todo(v.video)>0)||vids[0];openVideo(first.video)}
 renderVids();draw()}
const todo=v=>all.filter(x=>x.video===v&&!labels[x.id]).length;
function renderVids(){let n=0,done=0,full=0;const el=$("vids");el.innerHTML="";
 for(const v of vids){const cs=all.filter(x=>x.video===v.video),dn=cs.filter(x=>labels[x.id]).length;n+=cs.length;done+=dn;if(cs.length&&dn===cs.length)full++;
  const b=document.createElement("button");b.className=v.video===video?"on":"";b.innerHTML=`${v.title.replace(/</g,"&lt;")} <span class="${dn===cs.length?"done":"muted"}">${dn===cs.length&&cs.length?"✓":dn+"/"+cs.length}</span>`;
  b.onclick=()=>openVideo(v.video);el.appendChild(b)}
 $("total").textContent=`размечено ${done} из ${n} роликов · видео готово ${full} из ${vids.length}`;$("totalBar").style.width=(n?done/n*100:0)+"%"}

// ---- плеер ----
async function openVideo(v){if(v===video)return;video=v;cur=null;t=0;bx=[];const vd=vids.find(x=>x.video===v);dur=vd&&vd.seconds||1;
 try{const r=await fetch("/api/boxes?video="+encodeURIComponent(v));bx=(await r.json()).boxes||[]}catch(e){bx=[]}
 const c=cands().find(x=>!labels[x.id])||cands()[0];if(c){cur=c.id;t=c.t}renderVids();
 // плеер может быть ещё не готов: тогда нужное видео загрузит onReady
 if(player&&ready)player.loadVideoById({videoId:v,startSeconds:c?c.t:0});else makePlayer();draw()}
function makePlayer(){if(player||!video||!(window.YT&&YT.Player))return;player=new YT.Player("yt",{videoId:video,playerVars:{rel:0,modestbranding:1,playsinline:1},
 events:{onReady:()=>{ready=true;const c=curC();
   if(player.getVideoData().video_id!==video)player.loadVideoById({videoId:video,startSeconds:c?c.t:0});
   else if(c)player.seekTo(c.t,true);
   const d=player.getDuration();if(d)dur=d;player.setPlaybackRate(+$("rate").value)},
  onStateChange:e=>{if(e.data===1){const d=player.getDuration();if(d)dur=d;if(player.getPlaybackRate()!==+$("rate").value)player.setPlaybackRate(+$("rate").value)}}}})}
// клик по видео отдаёт клавиатуру плееру YouTube (M — его «без звука»): возвращаем её странице
addEventListener("blur",()=>setTimeout(()=>{const a=document.activeElement;if(a&&a.tagName==="IFRAME"){a.blur();window.focus()}},150));
window.onYouTubeIframeAPIReady=()=>makePlayer();
function seek(s){s=Math.max(0,Math.min(dur-1,s));t=s;if(player&&ready)player.seekTo(s,true);draw()}
function select(c,play=true){cur=c.id;seek(c.t);if(play&&player&&ready)player.playVideo();draw()}
function pickNear(){const c=curC();if(c&&t>=c.t-2&&t<=c.t_end+8)return;let best=null;
 for(const x of cands()){const d=t<x.t_start?x.t_start-t:t>x.t_end?t-x.t_end:0;if(d<=4&&(!best||d<best.d))best={x,d}}cur=best?best.x.id:null}

// ---- рамки поверх видео ----
function frameAt(fr,tt){let lo=0,hi=fr.length-1;if(hi<0||tt<fr[0][0]-0.3||tt>fr[hi][0]+0.3)return null;
 while(hi-lo>1){const m=(lo+hi)>>1;if(fr[m][0]<=tt)lo=m;else hi=m}return Math.abs(fr[lo][0]-tt)<=Math.abs(fr[hi][0]-tt)?fr[lo]:fr[hi]}
function overlay(){const cv=$("ov"),w=cv.clientWidth,h=cv.clientHeight,dpr=devicePixelRatio||1;if(cv.width!==Math.round(w*dpr)){cv.width=w*dpr;cv.height=h*dpr}
 const g=cv.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);const others=$("others").checked||pick;
 const list=cur?[bx.find(b=>b.id===cur)].filter(Boolean):bx;const seen=new Set();
 for(const c of list){const f=frameAt(c.frames,t);if(!f)continue;
  for(const [k,b] of Object.entries(f[1])){const yellow=k===c.track,key=k+":"+b;if(!yellow&&(!others||seen.has(key)))continue;seen.add(key);
   g.strokeStyle=yellow?"#ffd400":pick?"rgba(255,255,255,.95)":"rgba(220,220,220,.5)";g.lineWidth=yellow?3:pick?2:1;g.strokeRect(b[0]*w,b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h)}}}

// ---- таймлайн ----
function color(c){const l=labels[c.id];return isCut(l)?css("--cut"):l==="no"?css("--no"):l==="unsure"?css("--unsure"):css("--todo")}
function draw(){pickNear();
 for(const [id,zoom] of [["tlFull",false],["tlZoom",true]]){const cv=$(id),w=cv.clientWidth,h=cv.clientHeight,dpr=devicePixelRatio||1;
  cv.width=w*dpr;cv.height=h*dpr;const g=cv.getContext("2d");g.scale(dpr,dpr);
  const a=zoom?t-60:0,b=zoom?t+60:dur,X=s=>(s-a)/(b-a)*w;g.fillStyle=css("--muted");g.font="11px system-ui";const step=zoom?10:(dur>3600?600:300);
  for(let s=Math.max(0,Math.ceil(a/step)*step);s<=Math.min(b,dur);s+=step){const x=X(s);g.fillRect(x,h-6,1,6);if(x>2&&x<w-30)g.fillText(fmt(s),x+3,h-8)}
  for(const c of cands()){const x1=X(c.t_start),x2=Math.max(X(c.t_end),x1+(zoom?4:2));if(x2<0||x1>w)continue;g.fillStyle=color(c);g.fillRect(x1,zoom?8:6,x2-x1,h-(zoom?26:20));
   if(c.id===cur){g.strokeStyle=css("--fg");g.lineWidth=2;g.strokeRect(x1-2,zoom?6:4,x2-x1+4,h-(zoom?22:16))}}
  g.fillStyle=css("--manual");for(const m of manual[video]||[]){const x=X(m);if(x>=0&&x<=w){g.beginPath();g.moveTo(x-6,2);g.lineTo(x+6,2);g.lineTo(x,12);g.fill();g.fillRect(x-1,2,2,h-20)}}
  g.fillStyle=css("--fg");g.fillRect(X(t)-1,0,2,h);if(!zoom){g.strokeStyle=css("--accent");g.lineWidth=1;g.strokeRect(X(t-60),1,X(t+60)-X(t-60),h-2)}}
 $("clock").textContent=`${fmt(t)} / ${fmt(dur)}`;const c=curC(),cs=cands();
 if(c){const l=labels[c.id];$("clipPos").textContent=`Ролик ${cs.indexOf(c)+1} из ${cs.length} · ${fmt(c.t_start)}`;$("clipStatus").textContent=l?"— "+NAMES[l]:"— не размечен";$("clipStatus").className="status "+(isCut(l)?"cut":l||"")}
 else{$("clipPos").textContent="Здесь ролика программы нет";$("clipStatus").textContent="";$("clipStatus").className="status"}
 for(const [b,k] of [["bCut","cut"],["bCut2","cut_other"],["bNo","no"],["bUnsure","unsure"]]){$(b).disabled=!c;$(b).setAttribute("aria-pressed",!!c&&labels[c.id]===k)}
 $("bPick").disabled=!c;$("bPick").setAttribute("aria-pressed",pick)}
for(const [id,zoom] of [["tlFull",false],["tlZoom",true]])$(id).addEventListener("click",async e=>{const r=e.currentTarget.getBoundingClientRect(),f=(e.clientX-r.left)/r.width;
 if(zoom&&e.altKey){const near=(manual[video]||[]).find(m=>Math.abs(m-(t-60+f*120))<3);if(near){const d=await post("/api/manual",{video,t:near,remove:true});if(d){manual=d.manual;say(`Ручная отметка ${fmt(near)} убрана`);draw()}}return}
 cur=null;seek(zoom?t-60+f*120:f*dur)});

// ---- отметки ----
async function label(l,force){const c=curC();if(!c)return;const nl=labels[c.id]===l&&!force?null:l;const d=await post("/api/label",{id:c.id,label:nl});if(!d)return;
 if(nl)labels[c.id]=nl;else delete labels[c.id];say(`Ролик ${fmt(c.t_start)}: ${nl?NAMES[nl]:"отметка снята"}`);renderVids();draw();
 if(nl&&$("auto").checked)setTimeout(()=>next(),250)}
// Видео сама страница не переключает: только кнопкой шоу сверху (Иван перепроверяет размеченные шоу).
function next(){const cs=cands(),c=curC(),from=c?c.t_start:t;
 const n=cs.find(x=>!labels[x.id]&&x.t_start>from)||cs.find(x=>!labels[x.id]);if(n){select(n);return}
 const r=cs.find(x=>x.t_start>from+0.5);  // шоу размечено — просто следующий ролик, для перепроверки
 if(r){select(r);say("В этом шоу всё размечено — идём по роликам для перепроверки");return}
 const ov=vids.find(v=>v.video!==video&&todo(v.video)>0);
 say(ov?`Это последний ролик шоу. Неразмеченные есть в «${ov.title}» — выберите его кнопкой сверху.`:"Это последний ролик шоу. Всё размечено — отправьте: git add data/scenecut, git commit, git push")}
function prev(){const cs=cands(),c=curC(),from=c?c.t_start:t;const p=[...cs].reverse().find(x=>x.t_start<from-0.5);if(p)select(p)}
function setPick(on){pick=on&&!!curC();$("ov").classList.toggle("pick",pick);if(pick){if(player&&ready)player.pauseVideo();say("Кликните по рамке человека, который делает монтаж (Esc — отмена)")}draw()}
$("ov").addEventListener("click",async e=>{if(!pick)return;const c=bx.find(x=>x.id===cur);if(!c){setPick(false);return}
 const r=e.currentTarget.getBoundingClientRect(),px=(e.clientX-r.left)/r.width,py=(e.clientY-r.top)/r.height,f=frameAt(c.frames,t);let hit=null;
 if(f)for(const [k,b] of Object.entries(f[1]))if(px>=b[0]&&px<=b[2]&&py>=b[1]&&py<=b[3]){const a=(b[2]-b[0])*(b[3]-b[1]);if(!hit||a<hit.a)hit={k,a}}
 if(!hit){say("Здесь нет рамки — кликните внутрь рамки человека. Если нужного человека нет в рамках — кнопка 6.");return}
 const d=await post("/api/track",{id:c.id,track:hit.k});if(!d)return;c.track=hit.k;labels[c.id]="cut";setPick(false);say("Монтаж, человек указан вручную");renderVids();draw();
 if($("auto").checked)setTimeout(()=>next(),400)});
async function missed(){const c=curC();if(c&&Math.abs(t-c.t_start)<4){if(!isCut(labels[c.id]))await label("cut",true);return}
 const d=await post("/api/manual",{video,t});if(!d)return;manual=d.manual;say(`Пропущенный монтаж ${fmt(t)} отмечен — больше ничего нажимать не нужно, после следующего сбора по нему появится ролик с меткой «монтаж».`);draw()}
function toggleLoop(){loop=!loop;$("bLoop").setAttribute("aria-pressed",loop)}

$("bCut").onclick=()=>label("cut");$("bCut2").onclick=()=>label("cut_other");$("bNo").onclick=()=>label("no");$("bUnsure").onclick=()=>label("unsure");
$("bPick").onclick=()=>setPick(!pick);$("bPrev").onclick=prev;$("bNext").onclick=next;$("bLoop").onclick=toggleLoop;$("bMiss").onclick=missed;
$("rate").onchange=e=>{if(player&&ready)player.setPlaybackRate(+e.target.value)};$("others").onchange=overlay;
document.addEventListener("keydown",e=>{if(e.target.tagName==="SELECT"||e.metaKey||e.ctrlKey||e.altKey)return;const k=e.key.toLowerCase(),d=e.shiftKey?1:5;
 if(k===" "){e.preventDefault();if(player&&ready)(player.getPlayerState()===1?player.pauseVideo():player.playVideo())}
 else if(e.key==="ArrowRight"){e.preventDefault();cur=null;seek(t+d)}else if(e.key==="ArrowLeft"){e.preventDefault();cur=null;seek(t-d)}
 else if(e.key==="1")label("cut");else if(e.key==="2")label("no");else if(e.key==="3")label("unsure");else if(e.key==="6")label("cut_other");
 else if(k==="b"||k==="и")setPick(!pick);else if(e.key==="Escape")setPick(false);
 else if(k==="n"||k==="т")next();else if(k==="p"||k==="з")prev();else if(k==="l"||k==="д")toggleLoop();else if(k==="m"||k==="ь")missed()});

let lastDraw=0;
(function tick(now){if(player&&ready&&player.getCurrentTime){const nt=player.getCurrentTime();
  const c=curC();if(loop&&c&&nt>c.t_end+1.5&&nt<c.t_end+4){player.seekTo(c.t,true)}
  if(Math.abs(nt-t)>0.02){t=nt;if(now-lastDraw>250){lastDraw=now;draw()}}}overlay();requestAnimationFrame(tick)})(0);
addEventListener("resize",draw);setInterval(load,30000);load();
</script></body></html>"""
