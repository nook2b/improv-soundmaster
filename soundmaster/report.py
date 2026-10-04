"""Отчёт по журналу прогона: что распознано, что решила модель, что сделал контроллер, сколько стоило."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_journal(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))], 3)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    start = next((r for r in records if r["kind"] == "replay_start"), None)
    t0 = start["media_t0"] if start else 0.0
    end = next((r for r in reversed(records) if r["kind"] == "replay_end"), None)
    duration = (end["media_duration"] if end else (records[-1]["t"] - t0 if records else 0.0)) or 0.0
    tr = [r for r in records if r["kind"] == "transcript"]
    resp = [r for r in records if r["kind"] == "brain_response"]
    dec = [r for r in records if r["kind"] == "decision"]
    cost = resp[-1]["cost_total"] if resp else 0.0
    return {
        "duration_s": round(duration, 1),
        "utterances": sum(1 for r in records if r["kind"] == "utterance"),
        "transcripts": len(tr),
        "asr_delay_p50": _pct([r["delay"] for r in tr], 0.5),
        "asr_delay_p95": _pct([r["delay"] for r in tr], 0.95),
        "brain_requests": len(resp),
        "brain_errors": sum(1 for r in resp if r.get("error")),
        "brain_latency_p50": _pct([r["latency"] for r in resp], 0.5),
        "brain_latency_p95": _pct([r["latency"] for r in resp], 0.95),
        "input_tokens": sum(r.get("input_tokens", 0) for r in resp),
        "cached_tokens": sum(r.get("cached_tokens", 0) for r in resp),
        "output_tokens": sum(r.get("output_tokens", 0) for r in resp),
        "cost_usd": round(cost, 4),
        "cost_per_hour_usd": round(cost / duration * 3600, 2) if duration > 0 else None,
        "decisions": dict(Counter(f"{r.get('call')}:{r.get('result')}" for r in dec)),
        "scene_cuts": sum(1 for r in records if r["kind"] == "scene_cut_detected"),
        "new_scenes": sum(1 for r in records if r.get("action") == "new_scene"),
    }


def _fmt(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{t % 60:04.1f}"


def timeline(records: list[dict[str, Any]]) -> list[tuple[float, str, str, str]]:
    start = next((r for r in records if r["kind"] == "replay_start"), None)
    t0 = start["media_t0"] if start else 0.0
    rows = []
    for r in records:
        k = r["kind"]
        if k == "transcript":
            rows.append((r["t_start"] - t0, "речь", r["text"], f"текст готов через {r['delay']} с"))
        elif k == "decision":
            what = f"{r.get('call')} → {r.get('effect')}: {r.get('result')}"
            if r.get("blocked_by"):
                what += f" ({r['blocked_by']})"
            cls = "звук" if r.get("result") in ("played", "confirmed") else "решение"
            rows.append((r["t"] - t0, cls, what, r.get("reason") or r.get("word") or ""))
        elif k == "brain_response":
            acts = ", ".join(f"{a['type']}:{a.get('id')}" for a in r.get("actions", [])) or "ничего"
            note = f"{r['latency']} с, ${r['cost']}" + (f" — ОШИБКА {r['error']}" if r.get("error") else "")
            rows.append((r["t"] - t0, "модель", acts, note))
        elif k == "scene_cut_detected":
            rows.append((r["t"] - t0, "монтаж", f"проход {r['direction']}", f"уверенность {r['confidence']}"))
        elif k in ("system", "operator") and r.get("action") == "new_scene":
            rows.append((r["t"] - t0, "сцена", f"новая сцена {r['scene_id']}", r.get("source", "")))
        elif k == "rule_negated":
            rows.append((r["t"] - t0, "правило", f"отрицание: {r['word']}", r["rule"]))
        elif k == "arm":
            rows.append((r["t"] - t0, "взведение", f"снято: {r['effect']}", r.get("why", "")))
    rows.sort(key=lambda x: x[0])
    return rows


def write_html(records: list[dict[str, Any]], path: str | Path, title: str) -> Path:
    s = summarize(records)
    rows = timeline(records)
    colors = {"речь": "#2b6cb0", "модель": "#6b46c1", "звук": "#2f855a", "решение": "#975a16",
              "монтаж": "#c05621", "сцена": "#c53030", "правило": "#718096", "взведение": "#718096"}
    trs = "\n".join(
        f"<tr><td>{_fmt(t)}</td><td style='color:{colors.get(kind, '#000')}'>{html.escape(kind)}</td>"
        f"<td>{html.escape(str(what))}</td><td class='m'>{html.escape(str(note))}</td></tr>"
        for t, kind, what, note in rows
    )
    summ = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(json.dumps(v, ensure_ascii=False))}</td></tr>"
                   for k, v in s.items())
    doc = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:14px/1.4 -apple-system,sans-serif;margin:16px;color:#1a202c}}table{{border-collapse:collapse;width:100%}}
td{{border-bottom:1px solid #e2e8f0;padding:4px 8px;vertical-align:top}}td.m{{color:#718096}}
h1{{font-size:18px}}h2{{font-size:15px;margin-top:24px}}</style></head><body>
<h1>{html.escape(title)}</h1><h2>Итоги</h2><table>{summ}</table>
<h2>Хронология (время записи)</h2><table>{trs}</table></body></html>"""
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path
