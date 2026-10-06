"""Контроллер: решает, можно ли выполнить команду, и ведёт журнал решений (PLAN.md, разд. 5).

Ручные команды техника выполняются всегда (кроме отсутствующего файла) и не тратят лимиты
автоматики. Команды AI проходят проверки по порядку из плана и становятся автозапуском,
подсказкой технику или отказом с причиной.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .clock import Clock
from .engine import Engine
from .journal import Journal
from .music.director import MusicDirector
from .registry import MusicStyle, Registry


@dataclass
class Decision:
    # played | suggested | confirmed | dismissed | blocked | late_dropped | deduped | expired | armed | started
    result: str
    effect: str | None = None
    blocked_by: str | None = None
    suggestion_id: int | None = None
    voice_id: int | None = None


@dataclass
class Suggestion:
    id: int
    kind: str  # "effect" | "music" | "scene_end"
    target: str  # id эффекта / музыкального состояния
    scene_id: int
    t_created: float
    t_expires: float
    reason: str = ""
    confidence: str | None = None
    request_id: str | None = None

    def as_dict(self, now: float) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "target": self.target, "reason": self.reason,
            "confidence": self.confidence, "ttl": round(max(0.0, self.t_expires - now), 1),
        }


@dataclass
class _Played:
    t: float
    effect: str
    source: str  # manual | ai


@dataclass
class Armed:
    """Взведённый синхронный эффект: сыграет один раз от разрешённого триггера до t_expires."""

    effect: str
    source: str  # fast_layer | ai | manual
    triggers: tuple[str, ...]
    scene_id: int
    t_armed: float
    t_expires: float
    shadow: bool  # True — режим SUGGEST/автоматика выкл: срабатывание станет подсказкой, а не звуком
    reason: str = ""
    request_id: str | None = None


@dataclass
class ControllerState:
    scene_id: int = 1
    connection_id: int = 0
    automation_enabled: bool = True
    scene_hint: str = ""
    suggestions: dict[int, Suggestion] = field(default_factory=dict)
    armed: Armed | None = None
    t_scene_start: float = 0.0


class Controller:
    def __init__(
        self,
        registry: Registry,
        engine: Engine,
        music: MusicDirector,
        journal: Journal,
        clock: Clock,
        clean_actor_feed: bool,
        scene_cut_mode: str = "SUGGEST",
        scene_cut_music: str = "stop",
        scene_cut_cooldown: float = 10.0,
        music_mode: str = "SUGGEST",
        music_min_hold: float = 30.0,
        music_first_hold: float = 10.0,
        music_neutral: str | None = None,
    ):
        self.registry = registry
        self.engine = engine
        self.music = music
        self.journal = journal
        self.clock = clock
        self.clean_actor_feed = clean_actor_feed
        self.state = ControllerState()
        self.played: deque[_Played] = deque(maxlen=500)
        self.ai_actions: deque[tuple[float, str]] = deque(maxlen=500)  # (t, effect) — для лимитов AI
        self.dismissed: dict[str, float] = {}  # target → когда техник отклонил (в этой сцене)
        self.scene_end_count = 0
        self.scene_end_pause_until = 0.0
        self.scene_cut_mode = scene_cut_mode.upper()  # OFF | SUGGEST | AUTO
        self.scene_cut_music = scene_cut_music  # stop | keep
        self.scene_cut_cooldown = scene_cut_cooldown
        # Музыка от AI: SUGGEST — подсказка человеку; AUTO — включается сама (решение команды:
        # музыка автоматическая, эффекты — пока подсказки). min_hold — сколько секунд держать выбор
        # в сцене, прежде чем AI сможет его сменить (чтобы фон не «прыгал» от реплики к реплике).
        self.music_mode = music_mode.upper()
        if self.music_mode not in ("SUGGEST", "AUTO"):
            raise ValueError(f"music_mode: {music_mode} (SUGGEST | AUTO)")
        self.music_min_hold = music_min_hold
        # Первый выбор в сцене модель делает рано, по 1–2 репликам, — его можно поправить быстрее.
        self.music_first_hold = music_first_hold
        # Нейтральная переходная музыка после «монтажа» (scene_cut_music: neutral): сцена не стоит
        # в тишине, пока модель понимает новую сцену; потом генератор плавно перетекает в выбранное.
        self.music_neutral = MusicStyle.parse(music_neutral) if music_neutral else None
        if self.music_neutral and registry.vocabulary is not None:
            bad = registry.vocabulary.validate(self.music_neutral)
            if bad:
                raise ValueError(f"нейтральная музыка {music_neutral}: {bad}")
        self.music_t_set: float | None = None  # когда в этой сцене AI выбрал музыку
        self.music_picks = 0  # сколько раз AI выбирал музыку в этой сцене
        self.music_history: deque[tuple[int, str]] = deque(maxlen=8)  # (сцена, стиль) — для разнообразия
        self.new_scene_listeners: list = []  # вызываются с (scene_id, source) — например, сброс контекста AI
        self._ids = itertools.count(1)
        self.state.t_scene_start = clock.now()

    # ---- вспомогательное ---------------------------------------------------------------
    @property
    def limits(self):
        return self.registry.limits

    def _log(self, decision: Decision, source: str, call: str, **extra: Any) -> Decision:
        self.journal.write(
            "decision", source=source, call=call, scene_id=self.state.scene_id,
            effect=decision.effect, result=decision.result, blocked_by=decision.blocked_by,
            suggestion_id=decision.suggestion_id, voice_id=decision.voice_id, **extra,
        )
        return decision

    def _play(self, effect_id: str, owner: str, source: str, info: dict | None = None) -> int | None:
        effect = self.registry.effect(effect_id)
        vid = self.engine.play_effect(effect, owner, info)
        if vid is not None:
            self.played.append(_Played(self.clock.now(), effect_id, source))
        return vid

    def _drop_suggestions(self, *, target: str | None = None, kind: str | None = None) -> None:
        for sid in [s.id for s in self.state.suggestions.values()
                    if (target is None or s.target == target) and (kind is None or s.kind == kind)]:
            del self.state.suggestions[sid]

    def _suggest(self, kind: str, target: str, reason: str, confidence: str | None,
                 request_id: str | None, ttl: float | None = None) -> Suggestion:
        now = self.clock.now()
        s = Suggestion(next(self._ids), kind, target, self.state.scene_id, now,
                       now + (ttl if ttl is not None else self.limits.suggestion_ttl),
                       reason, confidence, request_id)
        self.state.suggestions[s.id] = s
        return s

    # ---- ручные команды ----------------------------------------------------------------
    def manual_play(self, effect_id: str) -> Decision:
        effect = self.registry.effect(effect_id)
        if not self.engine.available(effect):
            return self._log(Decision("blocked", effect_id, "file_missing"), "manual", "play")
        # Ручной запуск снимает конфликтующие подсказки того же эффекта.
        self._drop_suggestions(target=effect_id)
        vid = self._play(effect_id, "manual", "manual")
        if vid is None:
            return self._log(Decision("blocked", effect_id, "already_playing"), "manual", "play")
        return self._log(Decision("played", effect_id, voice_id=vid), "manual", "play")

    def manual_stop(self, effect_id: str) -> None:
        self.engine.mixer.stop(effect_id=effect_id, fade=self.registry.effect(effect_id).fade_out or 0.03)
        self.journal.write("operator", action="stop_effect", effect=effect_id)

    def manual_music(self, state_id: str | None) -> str:
        self._drop_suggestions(kind="music")
        if state_id is None:
            self.music.stop("music")
            result = "stopped"
        else:
            result = self.music.set_state(state_id, owner="manual")
        self.journal.write("operator", action="music", state=state_id, result=result)
        return result

    def confirm(self, suggestion_id: int) -> Decision:
        s = self.state.suggestions.pop(suggestion_id, None)
        now = self.clock.now()
        if s is None:
            return self._log(Decision("blocked", None, "no_such_suggestion"), "manual", "confirm")
        if now > s.t_expires:
            return self._log(Decision("expired", s.target, suggestion_id=s.id), "manual", "confirm")
        if s.scene_id != self.state.scene_id:
            return self._log(Decision("blocked", s.target, "stale_scene", s.id), "manual", "confirm")
        if s.kind == "new_scene":
            self.new_scene(keep_music=self.scene_cut_music == "keep", source="confirmed_scene_cut")
            return self._log(Decision("confirmed", "new_scene", suggestion_id=s.id), "manual", "confirm")
        if s.kind == "music_style":
            self.music.set_style(MusicStyle.parse(s.target.removeprefix("style:")), owner="manual")
            self.music_t_set = now
            return self._log(Decision("confirmed", s.target, suggestion_id=s.id), "manual", "confirm",
                             request_id=s.request_id)
        if s.kind == "music":
            self.music.set_state(s.target, owner="manual")
            return self._log(Decision("confirmed", s.target, suggestion_id=s.id), "manual", "confirm",
                             request_id=s.request_id)
        effect_id = s.target
        if s.kind == "scene_end":
            effect_id = self.registry.scene_end_effect
            if effect_id is None:
                return self._log(Decision("blocked", None, "no_scene_end_effect", s.id), "manual", "confirm")
        vid = self._play(effect_id, "manual", "manual", {"request_id": s.request_id})
        return self._log(Decision("confirmed", effect_id, suggestion_id=s.id, voice_id=vid), "manual", "confirm",
                         request_id=s.request_id)

    def dismiss(self, suggestion_id: int) -> Decision:
        s = self.state.suggestions.pop(suggestion_id, None)
        if s is None:
            return self._log(Decision("blocked", None, "no_such_suggestion"), "manual", "dismiss")
        now = self.clock.now()
        self.dismissed[s.target] = now
        if s.kind == "scene_end":
            self.scene_end_pause_until = now + float(self.limits.scene_end["pause_after_dismiss"])
        return self._log(Decision("dismissed", s.target, suggestion_id=s.id), "manual", "dismiss")

    def new_scene(self, hint: str = "", keep_music: bool = False, source: str = "manual") -> int:
        self.state.scene_id += 1
        self.state.scene_hint = hint
        self.state.t_scene_start = self.clock.now()
        self.state.suggestions.clear()
        if self.state.armed:
            self._disarm("new_scene")
        self.dismissed.clear()
        self.scene_end_count = 0
        self.scene_end_pause_until = 0.0
        if not keep_music:
            self.music_t_set = None
            self.music_picks = 0
            if self.scene_cut_music == "neutral" and self.start_neutral_music(cut=True):
                pass  # атмосфера прошлого места погасла, музыка плавно уходит в нейтральную (без паузы)
            else:
                self.music.stop_all()
        self.engine.mixer.stop(owner="ai", fade=0.3)
        self.journal.write("operator" if source == "manual" else "system", action="new_scene",
                           scene_id=self.state.scene_id, hint=hint, keep_music=keep_music, source=source)
        for cb in self.new_scene_listeners:
            cb(self.state.scene_id, source)
        return self.state.scene_id

    def start_neutral_music(self, cut: bool = False) -> bool:
        """Нейтральная музыка, пока модель не поняла сцену: в начале шоу и после «монтажа» (cut).

        Только если музыка автоматическая и словарь/нейтральный стиль заданы. False — не включали.
        """
        if not (self.music_neutral is not None and self.registry.vocabulary is not None
                and self.music_mode == "AUTO" and self.state.automation_enabled):
            return False
        self.music.stop("ambience")
        self.music.set_style(self.music_neutral, owner="system", cut=cut)
        self.journal.write("system", action="neutral_music", scene_id=self.state.scene_id,
                           style=self.music_neutral.id)
        return True

    def set_scene_hint(self, hint: str) -> None:
        self.state.scene_hint = hint
        self.journal.write("operator", action="scene_hint", scene_id=self.state.scene_id, hint=hint)

    # ---- смена сцены по сигналу («монтаж» проходом перед сценой) -----------------------
    def scene_cut(self, source: str, confidence: float = 0.0, t_signal: float | None = None) -> Decision:
        now = self.clock.now()
        extra = {"confidence": round(confidence, 2), "t_signal": None if t_signal is None else round(t_signal, 3),
                 "mode": self.scene_cut_mode}

        def log(d: Decision) -> Decision:
            return self._log(d, source, "scene_cut", **extra)

        if self.scene_cut_mode == "OFF":
            return log(Decision("blocked", "new_scene", "mode_off"))
        if now - self.state.t_scene_start < self.scene_cut_cooldown:
            return log(Decision("deduped", "new_scene", "scene_just_started"))
        if any(s.kind == "new_scene" for s in self.state.suggestions.values()):
            return log(Decision("deduped", "new_scene", "already_suggested"))
        if self.scene_cut_mode == "AUTO" and self.state.automation_enabled:
            self.new_scene(keep_music=self.scene_cut_music == "keep", source=source)
            return log(Decision("started", "new_scene"))
        s = self._suggest("new_scene", "new_scene", f"{source}: проход перед сценой", None, None, ttl=6.0)
        why = "mode_suggest" if self.scene_cut_mode != "AUTO" else "automation_off"
        return log(Decision("suggested", "new_scene", why, s.id))

    # ---- взведение и срабатывание синхронных эффектов -----------------------------------
    def _disarm(self, why: str) -> None:
        a = self.state.armed
        self.state.armed = None
        if a:
            self.journal.write("arm", action="disarm", effect=a.effect, why=why, scene_id=self.state.scene_id)

    def arm(self, effect_id: str, *, source: str, timeout: float | None = None, reason: str = "",
            scene_id: int | None = None, connection_id: int | None = None, t_input_ref: float | None = None,
            request_id: str | None = None) -> Decision:
        now = self.clock.now()
        extra = {"reason": reason, "request_id": request_id}

        def log(d: Decision) -> Decision:
            return self._log(d, source, "arm", **extra)

        effect = self.registry.effects.get(effect_id)
        if effect is None:
            return log(Decision("blocked", effect_id, "unknown_effect"))
        if effect.mode == "OFF":
            return log(Decision("blocked", effect_id, "mode_off"))
        if effect.cls != "sync":
            return log(Decision("blocked", effect_id, "not_sync_effect"))
        if source != "manual" and effect.human_only:
            return log(Decision("blocked", effect_id, "human_only"))
        if scene_id is not None and scene_id != self.state.scene_id:
            return log(Decision("blocked", effect_id, "stale_scene"))
        if connection_id is not None and connection_id != self.state.connection_id:
            return log(Decision("blocked", effect_id, "stale_connection"))
        if t_input_ref is not None and now - t_input_ref > self.limits.late_drop.get("context", 3.0):
            return log(Decision("late_dropped", effect_id))
        a = self.state.armed
        if a and a.effect == effect_id and now <= a.t_expires:
            a.t_expires = max(a.t_expires, now + min(timeout or effect.max_arm_timeout, effect.max_arm_timeout))
            return log(Decision("deduped", effect_id, "already_armed"))
        if a:
            self._disarm(f"replaced_by:{effect_id}")
        ttl = min(timeout or effect.max_arm_timeout, effect.max_arm_timeout)
        shadow = source != "manual" and (effect.mode != "AUTO" or not self.state.automation_enabled)
        self.state.armed = Armed(effect_id, source, effect.arm_triggers, self.state.scene_id, now, now + ttl,
                                 shadow, reason, request_id)
        return log(Decision("armed", effect_id, "shadow" if shadow else None))

    def fire(self, trigger: str, *, word: str | None = None, t_signal: float | None = None) -> Decision:
        now = self.clock.now()
        source = "manual" if trigger == "operator" else "fast_layer"
        extra = {"trigger": trigger, "word": word,
                 "t_signal": None if t_signal is None else round(t_signal, 3)}

        def log(d: Decision) -> Decision:
            return self._log(d, source, "fire", **extra)

        a = self.state.armed
        if a is None:
            return log(Decision("blocked", None, "not_armed"))
        if now > a.t_expires:
            self._disarm("expired")
            return log(Decision("expired", a.effect))
        if trigger not in a.triggers and trigger != "operator":
            return log(Decision("blocked", a.effect, "trigger_not_allowed"))
        effect = self.registry.effect(a.effect)
        if trigger == "keyword" and (word or "") not in effect.fire_keywords:
            return log(Decision("blocked", a.effect, "wrong_keyword"))
        self.state.armed = None
        extra["armed_by"] = a.source
        extra["armed_for"] = round(now - a.t_armed, 3)
        if trigger == "operator":
            vid = self._play(a.effect, "manual", "manual", {"request_id": a.request_id})
            return log(Decision("played", a.effect, voice_id=vid))
        if a.shadow:
            # Теневой режим: что сыграло бы — в журнал; технику — короткая подсказка.
            s = self._suggest("effect", a.effect, f"сработало бы: {word or trigger}", None, a.request_id, ttl=2.0)
            return log(Decision("suggested", a.effect, "shadow", s.id))
        vid = self._play(a.effect, "ai", "ai", {"request_id": a.request_id})
        return log(Decision("played", a.effect, voice_id=vid))

    def set_automation(self, enabled: bool) -> None:
        self.state.automation_enabled = enabled
        self.journal.write("operator", action="automation", enabled=enabled)

    def stop_ai_sound(self) -> None:
        """Отключить автоматику и остановить все звуки, запущенные AI (фон — с коротким fade)."""
        self.set_automation(False)
        self._drop_suggestions()
        if self.state.armed:
            self._disarm("stop_ai")
        self.engine.mixer.stop(owner="ai", fade=0.3)
        for ch, a in list(self.music.active.items()):
            if a.owner == "ai":
                self.music.stop(ch)
        self.journal.write("operator", action="stop_ai_sound")

    def stop_all(self) -> None:
        self.music.stop_all()
        self.engine.mixer.stop_all(fade=0.05)
        self.journal.write("operator", action="stop_all")

    def set_mode(self, effect_id: str, mode: str) -> None:
        effect = self.registry.effect(effect_id)
        mode = mode.upper()
        if mode not in ("OFF", "SUGGEST", "AUTO"):
            raise ValueError(mode)
        if mode == "AUTO" and effect.human_only:
            raise ValueError(f"{effect_id}: громкий эффект/отбивка не может быть AUTO")
        effect.mode = mode
        self.journal.write("operator", action="set_mode", effect=effect_id, mode=mode)

    def new_connection(self) -> int:
        """Новое соединение с AI: всё, что пришло по старому, считается устаревшим."""
        self.state.connection_id += 1
        self._drop_suggestions()
        if self.state.armed and self.state.armed.source == "ai":
            self._disarm("new_connection")
        self.journal.write("system", action="new_connection", connection_id=self.state.connection_id)
        return self.state.connection_id

    # ---- команды AI ---------------------------------------------------------------------
    def _ai_limits(self, effect_id: str, now: float) -> str | None:
        e = self.registry.effect(effect_id)
        mine = [t for t, eid in self.ai_actions if eid == effect_id]
        if mine and e.series.cooldown and now - mine[-1] < e.series.cooldown:
            return "cooldown"
        if e.series.count and e.series.window:
            if sum(1 for t in mine if now - t < e.series.window) >= e.series.count:
                return "series_limit"
        others = [t for t, eid in self.ai_actions if eid != effect_id]
        if others and now - others[-1] < self.limits.min_gap_between_different_ai_effects:
            return "global_gap"
        # Серия одного эффекта считается за одно вмешательство.
        recent = [eid for t, eid in self.ai_actions if now - t < 60.0]
        distinct_runs = sum(1 for i, eid in enumerate(recent) if i == 0 or recent[i - 1] != eid)
        if distinct_runs >= self.limits.max_ai_actions_per_minute and (not recent or recent[-1] != effect_id):
            return "per_minute_limit"
        return None

    def ai_effect(
        self,
        effect_id: str,
        *,
        scene_id: int,
        connection_id: int,
        t_input_ref: float,
        request_id: str | None = None,
        reason: str = "",
        confidence: str | None = None,
        call: str = "suggest_sfx",
    ) -> Decision:
        now = self.clock.now()
        extra = {"request_id": request_id, "reason": reason, "confidence": confidence,
                 "t_input_ref": round(t_input_ref, 4), "connection_id": connection_id}

        def log(d: Decision) -> Decision:
            return self._log(d, "ai", call, **extra)

        # 1. эффект и режим
        effect = self.registry.effects.get(effect_id)
        if effect is None:
            return log(Decision("blocked", effect_id, "unknown_effect"))
        if effect.mode == "OFF":
            return log(Decision("blocked", effect_id, "mode_off"))
        if not self.engine.available(effect):
            return log(Decision("blocked", effect_id, "file_missing"))
        # 2. актуальность
        if scene_id != self.state.scene_id:
            return log(Decision("blocked", effect_id, "stale_scene"))
        if connection_id != self.state.connection_id:
            return log(Decision("blocked", effect_id, "stale_connection"))
        if effect.cls == "sync":
            # Синхронный эффект прямым вызовом всегда опаздывает: только взведение или человек.
            return log(Decision("blocked", effect_id, "sync_requires_arm_or_human"))
        late = self.limits.late_drop.get(effect.cls, 0.0)
        if now - t_input_ref > late:
            return log(Decision("late_dropped", effect_id, f"late>{late}s"))
        # 3. уже выполнено / уже предложено
        if any(p.effect == effect_id and now - p.t < self.limits.dedupe_window for p in self.played):
            return log(Decision("deduped", effect_id, "recently_played"))
        if any(s.target == effect_id for s in self.state.suggestions.values()):
            return log(Decision("deduped", effect_id, "already_suggested"))
        # 4. отклонено техником
        t_dis = self.dismissed.get(effect_id)
        if t_dis is not None and now - t_dis < self.limits.dedupe_window:
            return log(Decision("blocked", effect_id, "dismissed_by_operator"))
        # 5. лимиты автоматики
        lim = self._ai_limits(effect_id, now)
        if lim:
            return log(Decision("blocked", effect_id, lim))
        self.ai_actions.append((now, effect_id))
        # 6. нужен ли человек
        why_suggest = None
        if effect.human_only:
            why_suggest = "human_only"
        elif effect.mode != "AUTO":
            why_suggest = "mode_suggest"
        elif not self.state.automation_enabled:
            why_suggest = "automation_off"
        elif not self.clean_actor_feed:
            why_suggest = "no_clean_actor_feed"
        if why_suggest:
            s = self._suggest("effect", effect_id, reason, confidence, request_id)
            return log(Decision("suggested", effect_id, why_suggest, s.id))
        vid = self._play(effect_id, "ai", "ai", {"request_id": request_id})
        return log(Decision("played", effect_id, voice_id=vid))

    def _music_gate(self, channel: str, target: str, *, scene_id: int, connection_id: int,
                    t_input_ref: float) -> str | None:
        """Общие проверки музыки от AI. None — можно; иначе причина отказа."""
        now = self.clock.now()
        if scene_id != self.state.scene_id or connection_id != self.state.connection_id:
            return "stale"
        if now - t_input_ref > self.limits.late_drop.get("background", 5.0):
            return "late"
        chosen = self.music.current(channel)
        if chosen and chosen.id == target:
            return "already_playing"
        if any(s.target == target for s in self.state.suggestions.values()):
            return "already_suggested"
        active = self.music.active.get(channel)
        if active and active.owner == "manual":
            return "manual_music"  # человек выбрал сам — AI не перебивает до новой сцены
        hold = self.music_first_hold if self.music_picks <= 1 else self.music_min_hold
        if (channel == "music" and chosen and self.music_t_set is not None
                and now - self.music_t_set < hold):
            return "min_hold"
        return None

    def ai_music(self, state_id: str, *, scene_id: int, connection_id: int, t_input_ref: float,
                 request_id: str | None = None, reason: str = "", confidence: str | None = None) -> Decision:
        """Фон-состояние из реестра (атмосфера или музыка-файл): AUTO — сразу, SUGGEST — подсказка."""
        extra = {"request_id": request_id, "reason": reason, "confidence": confidence}

        def log(d: Decision) -> Decision:
            return self._log(d, "ai", "suggest_background", **extra)

        if state_id not in self.registry.music_states:
            return log(Decision("blocked", state_id, "unknown_music_state"))
        if not self.registry.music_states[state_id].ai:
            return log(Decision("blocked", state_id, "manual_only"))
        channel = self.registry.music_states[state_id].channel
        why = self._music_gate(channel, state_id, scene_id=scene_id, connection_id=connection_id,
                               t_input_ref=t_input_ref)
        if why == "late":
            return log(Decision("late_dropped", state_id))
        if why in ("already_playing", "already_suggested"):
            return log(Decision("deduped", state_id, why))
        if why:
            return log(Decision("blocked", state_id, why))
        if self.music_mode != "AUTO" or not self.state.automation_enabled:
            why_s = "background_suggest_only" if self.music_mode != "AUTO" else "automation_off"
            s = self._suggest("music", state_id, reason, confidence, request_id, ttl=self.limits.suggestion_ttl * 2)
            return log(Decision("suggested", state_id, why_s, s.id))
        result = self.music.set_state(state_id, owner="ai")
        if channel == "music":
            self.music_t_set = self.clock.now()
            self.music_picks += 1
        return log(Decision("started", state_id, "no_source" if result == "no_source" else None))

    def ai_music_style(self, style: MusicStyle, *, scene_id: int, connection_id: int, t_input_ref: float,
                       request_id: str | None = None, reason: str = "", confidence: str | None = None) -> Decision:
        """Музыка сцены из словаря (жанр/настроение/энергия/темп) — для генератора."""
        extra = {"request_id": request_id, "reason": reason, "confidence": confidence}

        def log(d: Decision) -> Decision:
            return self._log(d, "ai", "music_style", **extra)

        vocab = self.registry.vocabulary
        if vocab is None:
            return log(Decision("blocked", style.id, "no_vocabulary"))
        bad = vocab.validate(style)
        if bad:
            return log(Decision("blocked", style.id, bad))
        kept = self.scene_instrument()
        if kept is not None and style.instrument != kept:
            # Состав выбран на сцену: поворот меняет настроение/энергию, а не инструменты.
            extra["asked_instrument"] = style.instrument
            style = MusicStyle(style.genre, style.mood, style.energy, style.tempo, kept)
        target = f"style:{style.id}"
        why = self._music_gate("music", target, scene_id=scene_id, connection_id=connection_id,
                               t_input_ref=t_input_ref)
        if why == "late":
            return log(Decision("late_dropped", style.id))
        if why in ("already_playing", "already_suggested"):
            return log(Decision("deduped", style.id, why))
        if why:
            return log(Decision("blocked", style.id, why))
        prev = self.prev_scene_music()
        if prev and (prev.genre, prev.instrument) == (style.genre, style.instrument):
            return log(Decision("blocked", style.id, "same_as_prev_scene"))
        if self.music_mode != "AUTO" or not self.state.automation_enabled:
            # Подсказка хранит стиль как строку — при подтверждении он собирается заново.
            why_s = "background_suggest_only" if self.music_mode != "AUTO" else "automation_off"
            s = self._suggest("music_style", target, reason or vocab.label(style), confidence, request_id,
                              ttl=self.limits.suggestion_ttl * 2)
            return log(Decision("suggested", style.id, why_s, s.id))
        result = self.music.set_style(style, owner="ai")
        self.music_t_set = self.clock.now()
        self.music_picks += 1
        self.music_history.append((self.state.scene_id, style.id))
        return log(Decision("started", style.id, "no_source" if result == "no_source" else None))

    def scene_instrument(self) -> str | None:
        """Состав, закреплённый за сценой: после первого выбора и его уточнения не меняется."""
        if self.music_picks < 2:
            return None
        cur = [sid for scene, sid in self.music_history if scene == self.state.scene_id]
        return MusicStyle.parse(cur[-1]).instrument if cur else None

    def prev_scene_music(self) -> MusicStyle | None:
        """Чем закончилась музыка прошлой сцены: соседние сцены не звучат одинаково (жанр + состав)."""
        past = [sid for scene, sid in self.music_history if scene < self.state.scene_id]
        return MusicStyle.parse(past[-1]) if past else None

    def ai_scene_end(self, *, scene_id: int, connection_id: int, request_id: str | None = None,
                     reason: str = "", confidence: str | None = None) -> Decision:
        now = self.clock.now()
        cfg = self.limits.scene_end
        extra = {"request_id": request_id, "reason": reason, "confidence": confidence}

        def log(d: Decision) -> Decision:
            return self._log(d, "ai", "suggest_scene_end", **extra)

        if scene_id != self.state.scene_id or connection_id != self.state.connection_id:
            return log(Decision("blocked", "scene_end", "stale"))
        active = [s for s in self.state.suggestions.values() if s.kind == "scene_end"]
        if len(active) >= int(cfg["max_active"]):
            return log(Decision("deduped", "scene_end", "already_suggested"))
        if now < self.scene_end_pause_until:
            return log(Decision("blocked", "scene_end", "pause_after_dismiss"))
        if self.scene_end_count >= int(cfg["max_per_scene"]):
            return log(Decision("blocked", "scene_end", "max_per_scene"))
        self.scene_end_count += 1
        s = self._suggest("scene_end", "scene_end", reason, confidence, request_id)
        return log(Decision("suggested", "scene_end", "human_only", s.id))

    # ---- периодическое -----------------------------------------------------------------
    def tick(self) -> None:
        now = self.clock.now()
        if self.state.armed and now > self.state.armed.t_expires:
            a = self.state.armed
            self._disarm("expired")
            self._log(Decision("expired", a.effect), "system", "arm_expire")
        for s in [s for s in self.state.suggestions.values() if now > s.t_expires]:
            del self.state.suggestions[s.id]
            if s.kind == "scene_end":
                self.scene_end_pause_until = now + float(self.limits.scene_end["pause_after_dismiss"])
            self._log(Decision("expired", s.target, suggestion_id=s.id), "system", "expire")
        self.music.tick()
        self.engine.pump_events()

    def snapshot(self) -> dict[str, Any]:
        now = self.clock.now()
        playing = sorted({(eid, owner) for _, eid, _, owner in self.engine.mixer.playing})
        return {
            "t": round(now, 2),
            "scene_id": self.state.scene_id,
            "scene_hint": self.state.scene_hint,
            "automation": self.state.automation_enabled,
            "clean_actor_feed": self.clean_actor_feed,
            "suggestions": [s.as_dict(now) for s in self.state.suggestions.values()],
            "armed": None if not self.state.armed else {
                "effect": self.state.armed.effect, "source": self.state.armed.source,
                "shadow": self.state.armed.shadow,
                "ttl": round(max(0.0, self.state.armed.t_expires - now), 1),
            },
            "scene_cut_mode": self.scene_cut_mode,
            "playing": [{"effect": e, "owner": o} for e, o in playing],
            "music": self.music.status(),
            "modes": {e.id: e.mode for e in self.registry.effects.values()},
        }
