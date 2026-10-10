"""Реестр эффектов и музыкальных состояний (config/registry.yaml).

Модель и пульт ссылаются на эффекты только по id из реестра. Изменять реестр во время шоу
может только техник (режим эффекта), но не AI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MODES = ("OFF", "SUGGEST", "AUTO")
CLASSES = ("sync", "context", "background")
RISKS = ("low", "loud")
CHANNELS = ("sfx", "music", "ambience", "sting")
TRIGGERS = ("operator", "keyword", "transient")
# Клавиши пульта: n — новая сцена, a — автоматика, s/m/b — подтвердить подсказку эффекта/музыки/отбивки,
# x — стоп AI-звук, escape — стоп всё, "." — отметка события, " " — не используется
# (прокрутка/нажатие кнопки в фокусе).
RESERVED_KEYS = {"n", "a", "s", "m", "b", "x", "f", ".", "escape", " "}  # f — сработать взведённое


class RegistryError(ValueError):
    pass


@dataclass
class Series:
    cooldown: float = 0.0
    count: int = 0  # 0 — без лимита
    window: float = 0.0


@dataclass
class Effect:
    id: str
    file: str
    channel: str
    cls: str = "context"
    risk: str = "low"
    mode: str = "SUGGEST"
    label: str = ""
    hotkey: str | None = None
    gain_db: float = 0.0
    series: Series = field(default_factory=Series)
    loop: bool = False
    fade_in: float = 0.0
    fade_out: float = 0.0
    do_not_restart_if_playing: bool = False
    arm_triggers: tuple[str, ...] = ("operator",)
    fire_keywords: tuple[str, ...] = ()
    arm_keywords: tuple[str, ...] = ()
    max_arm_timeout: float = 5.0

    @property
    def human_only(self) -> bool:
        """Громкие эффекты и отбивки запускает только человек."""
        return self.risk == "loud" or self.channel == "sting"


@dataclass
class MusicState:
    """Музыкальное состояние: модель выбирает его id, программа — файл и пресет генератора."""

    id: str
    channel: str = "music"
    label: str = ""
    hotkey: str | None = None
    file: str | None = None  # библиотечный вариант и резерв генератора
    generator: dict[str, Any] = field(default_factory=dict)  # проверенный пресет генератора
    fade_in: float = 2.0
    fade_out: float = 3.0
    gain_db: float = -6.0
    ai: bool = True  # False — модель не включает, только человек с пульта


@dataclass
class VocabItem:
    """Одно слово музыкального словаря: что модель выбирает и что из этого получает генератор."""

    id: str
    label: str = ""
    text: str = ""  # фрагмент описания для генератора (Lyria понимает английский)
    config: dict[str, Any] = field(default_factory=dict)  # параметры генератора: bpm, density, brightness
    fallback: str | None = None  # музыкальное состояние-файл на случай сбоя генератора (только у mood)


@dataclass
class MusicStyle:
    """Музыка сцены, собранная из словаря: жанр + настроение + энергия + темп (+ инструменты)."""

    genre: str
    mood: str
    energy: str
    tempo: str
    instrument: str = "auto"  # auto — как в жанре; иначе сольный инструмент / состав из словаря

    @property
    def id(self) -> str:
        base = f"{self.genre}/{self.mood}/{self.energy}/{self.tempo}"
        return base if self.instrument == "auto" else f"{base}/{self.instrument}"

    @classmethod
    def parse(cls, style_id: str) -> "MusicStyle":
        parts = style_id.split("/")
        if len(parts) not in (4, 5):
            raise RegistryError(f"стиль музыки {style_id!r}: нужно жанр/настроение/энергия/темп[/инструменты]")
        return cls(*parts)


@dataclass
class MusicVocabulary:
    """Словарь, из которого модель собирает музыку для генератора (а не пишет свободный текст)."""

    genres: dict[str, VocabItem]
    moods: dict[str, VocabItem]
    energy: dict[str, VocabItem]
    tempo: dict[str, VocabItem]
    instruments: dict[str, VocabItem] = field(default_factory=dict)  # auto — всегда разрешено
    fade_in: float = 3.0
    fade_out: float = 3.0
    gain_db: float = -8.0
    # Непусто — музыка только сольная: инструмент обязателен (auto нельзя), а этот текст
    # добавляется к запросу генератора, чтобы Lyria не дописывала аккомпанемент.
    solo: str = ""

    def validate(self, style: MusicStyle) -> str | None:
        for name, value, table in (("genre", style.genre, self.genres), ("mood", style.mood, self.moods),
                                   ("energy", style.energy, self.energy), ("tempo", style.tempo, self.tempo)):
            if value not in table:
                return f"unknown_{name}"
        if style.instrument == "auto":
            return "instrument_required" if self.solo else None
        if style.instrument not in self.instruments:
            return "unknown_instrument"
        return None

    def label(self, style: MusicStyle) -> str:
        parts = [t[v].label for t, v in ((self.genres, style.genre), (self.moods, style.mood),
                                         (self.energy, style.energy), (self.tempo, style.tempo))]
        if style.instrument != "auto":
            parts.insert(1, self.instruments[style.instrument].label)
        return ", ".join(parts)

    def preset(self, style: MusicStyle) -> dict[str, Any]:
        """Пресет генератора: описания (жанр, настроение, темп — у кого есть text) и параметры."""
        g, m, e, t = self.genres[style.genre], self.moods[style.mood], self.energy[style.energy], self.tempo[style.tempo]
        parts = [(g, 1.0), (m, 1.0), (e, 0.5), (t, 0.5)]
        if style.instrument != "auto":
            # Сольный инструмент/состав: жанр остаётся «манерой», состав задаёт инструмент (вес больше,
            # иначе Lyria добавляет инструменты из описания жанра).
            parts = [(g, 0.5), (self.instruments[style.instrument], 1.5), (m, 1.0), (e, 0.5), (t, 0.5)]
        prompts = [{"text": x.text, "weight": w} for x, w in parts if x.text]
        if self.solo and style.instrument != "auto":
            prompts.append({"text": self.solo, "weight": 1.0})
        config: dict[str, Any] = {}
        for x, _ in parts:
            config.update(x.config)
        return {"prompts": prompts, "config": config}


@dataclass
class PoolTrack:
    """Записанная мелодия пула: стиль словаря + файл (sounds/pool/<жанр>/<n>-<инструмент>.flac)."""

    style: MusicStyle
    file: str
    n: int  # номер в жанре, с 1


@dataclass
class GlobalLimits:
    min_gap_between_different_ai_effects: float = 3.0
    max_ai_actions_per_minute: int = 6
    dedupe_window: float = 3.0
    late_drop: dict[str, float] = field(
        default_factory=lambda: {"context": 3.0, "background": 5.0, "sync": 0.0}
    )
    suggestion_ttl: float = 4.0
    scene_end: dict[str, float] = field(
        default_factory=lambda: {"max_active": 1, "pause_after_dismiss": 20.0, "max_per_scene": 3}
    )


@dataclass
class Registry:
    effects: dict[str, Effect]
    music_states: dict[str, MusicState]
    limits: GlobalLimits
    sounds_dir: Path
    scene_end_effect: str | None = None  # какой эффект звучит при подтверждении подсказки отбивки
    vocabulary: MusicVocabulary | None = None  # словарь музыки для генератора; None — только состояния-файлы
    pool: dict[str, list[PoolTrack]] = field(default_factory=dict)  # жанр → записанные мелодии

    def music_style_state(self, style: MusicStyle) -> MusicState:
        """Музыкальное состояние для стиля из словаря: пресет генератора + резервный файл настроения."""
        v = self.vocabulary
        if v is None:
            raise RegistryError("в реестре нет music_vocabulary")
        fb = v.moods[style.mood].fallback
        file = self.music_states[fb].file if fb else None
        return MusicState(id=f"style:{style.id}", channel="music", label=v.label(style), file=file,
                          generator=v.preset(style), fade_in=v.fade_in, fade_out=v.fade_out, gain_db=v.gain_db)

    def effect(self, effect_id: str) -> Effect:
        try:
            return self.effects[effect_id]
        except KeyError:
            raise RegistryError(f"неизвестный эффект: {effect_id}") from None

    def path_of(self, rel: str) -> Path:
        return (self.sounds_dir / rel).resolve()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RegistryError(msg)


def _parse_effect(eid: str, raw: dict[str, Any]) -> Effect:
    _require(isinstance(raw, dict), f"{eid}: ожидается словарь")
    _require("file" in raw, f"{eid}: нет поля file")
    channel = raw.get("channel", "sfx")
    _require(channel in CHANNELS, f"{eid}: канал {channel!r} не из {CHANNELS}")
    cls = raw.get("class", "background" if channel in ("music", "ambience") else "context")
    _require(cls in CLASSES, f"{eid}: класс {cls!r} не из {CLASSES}")
    risk = raw.get("risk", "low")
    _require(risk in RISKS, f"{eid}: риск {risk!r} не из {RISKS}")
    mode = str(raw.get("mode", "SUGGEST")).upper()
    _require(mode in MODES, f"{eid}: режим {mode!r} не из {MODES}")

    s = raw.get("series") or {}
    series = Series(
        cooldown=float(s.get("cooldown", raw.get("cooldown", 0.0))),
        count=int(s.get("count", 0)),
        window=float(s.get("window", 0.0)),
    )
    arm = raw.get("arm") or {}
    triggers = tuple(arm.get("triggers", ["operator"]))
    for t in triggers:
        _require(t in TRIGGERS, f"{eid}: триггер {t!r} не из {TRIGGERS}")
    effect = Effect(
        id=eid,
        file=raw["file"],
        channel=channel,
        cls=cls,
        risk=risk,
        mode=mode,
        label=raw.get("label", eid),
        hotkey=(str(raw["hotkey"]) if raw.get("hotkey") is not None else None),
        gain_db=float(raw.get("gain_db", 0.0)),
        series=series,
        loop=bool(raw.get("loop", channel in ("music", "ambience"))),
        fade_in=float(raw.get("fade_in", 0.0)),
        fade_out=float(raw.get("fade_out", 0.0)),
        do_not_restart_if_playing=bool(raw.get("do_not_restart_if_playing", False)),
        arm_triggers=triggers,
        fire_keywords=tuple(arm.get("fire_keywords", ())),
        arm_keywords=tuple(arm.get("arm_keywords", ())),
        max_arm_timeout=float(arm.get("max_timeout", 5.0)),
    )
    if effect.human_only:
        # Громкий эффект или отбивка никогда не бывает AUTO и не взводится на автоматический триггер.
        _require(effect.mode != "AUTO", f"{eid}: громкий эффект/отбивка не может быть AUTO")
        _require(
            set(effect.arm_triggers) <= {"operator"},
            f"{eid}: громкий эффект/отбивка срабатывает только от оператора",
        )
    return effect


def load_registry(path: str | Path) -> Registry:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sounds_dir = (path.parent / data.get("sounds_dir", "../sounds")).resolve()

    g = data.get("global") or {}
    limits = GlobalLimits()
    for key in (
        "min_gap_between_different_ai_effects",
        "dedupe_window",
        "suggestion_ttl",
    ):
        if key in g:
            setattr(limits, key, float(g[key]))
    if "max_ai_actions_per_minute" in g:
        limits.max_ai_actions_per_minute = int(g["max_ai_actions_per_minute"])
    if "late_drop" in g:
        limits.late_drop.update({k: float(v) for k, v in g["late_drop"].items()})
    if "scene_end" in g:
        limits.scene_end.update(g["scene_end"])
    _require(
        limits.dedupe_window >= limits.late_drop.get("context", 0.0),
        "dedupe_window должен быть не меньше late_drop.context",
    )

    effects = {eid: _parse_effect(eid, raw) for eid, raw in (data.get("effects") or {}).items()}

    hotkeys: dict[str, str] = {}
    for e in effects.values():
        if e.hotkey:
            _require(e.hotkey not in hotkeys, f"клавиша {e.hotkey!r} у {hotkeys.get(e.hotkey)} и {e.id}")
            _require(e.hotkey.lower() not in RESERVED_KEYS, f"{e.id}: клавиша {e.hotkey!r} зарезервирована пультом")
            hotkeys[e.hotkey] = e.id

    states: dict[str, MusicState] = {}
    for sid, raw in (data.get("music_states") or {}).items():
        raw = raw or {}
        channel = raw.get("channel", "music")
        _require(channel in ("music", "ambience"), f"{sid}: канал музыки {channel!r}")
        states[sid] = MusicState(
            id=sid,
            channel=channel,
            label=raw.get("label", sid),
            hotkey=(str(raw["hotkey"]) if raw.get("hotkey") is not None else None),
            file=raw.get("file"),
            generator=dict(raw.get("generator") or {}),
            fade_in=float(raw.get("fade_in", 2.0)),
            fade_out=float(raw.get("fade_out", 3.0)),
            gain_db=float(raw.get("gain_db", -6.0)),
            ai=bool(raw.get("ai", True)),
        )
        _require(states[sid].file or states[sid].generator, f"{sid}: нужен file или generator")
        hk = states[sid].hotkey
        if hk:
            _require(hk not in hotkeys, f"клавиша {hk!r} у {hotkeys.get(hk)} и {sid}")
            _require(hk.lower() not in RESERVED_KEYS, f"{sid}: клавиша {hk!r} зарезервирована пультом")
            hotkeys[hk] = sid

    scene_end_effect = g.get("scene_end_effect", "scene_end" if "scene_end" in effects else None)
    _require(scene_end_effect is None or scene_end_effect in effects,
             f"scene_end_effect {scene_end_effect!r} нет в effects")

    vocabulary = _parse_vocabulary(data.get("music_vocabulary"), states)
    pool = _parse_pool(data.get("music_pool"), vocabulary)
    return Registry(effects=effects, music_states=states, limits=limits, sounds_dir=sounds_dir,
                    scene_end_effect=scene_end_effect, vocabulary=vocabulary, pool=pool)


def _parse_pool(raw: dict[str, Any] | None, vocab: MusicVocabulary | None) -> dict[str, list[PoolTrack]]:
    if not raw:
        return {}
    _require(vocab is not None, "music_pool: нужен music_vocabulary")
    assert vocab is not None
    pool: dict[str, list[PoolTrack]] = {}
    for genre, items in raw.items():
        _require(genre in vocab.genres, f"music_pool.{genre}: нет такого жанра в словаре")
        _require(isinstance(items, list) and items, f"music_pool.{genre}: нужен список мелодий")
        tracks = []
        for n, spec in enumerate(items, 1):
            style = MusicStyle.parse(f"{genre}/{spec}")
            bad = vocab.validate(style)
            _require(bad is None, f"music_pool.{genre}[{n}] {spec!r}: {bad}")
            tracks.append(PoolTrack(style, f"pool/{genre}/{n}-{style.instrument}.flac", n))
        pool[genre] = tracks
    return pool


def _parse_vocabulary(raw: dict[str, Any] | None, states: dict[str, MusicState]) -> MusicVocabulary | None:
    if not raw:
        return None
    tables: dict[str, dict[str, VocabItem]] = {}
    for key in ("genres", "moods", "energy", "tempo", "instruments"):
        items = raw.get(key) or {}
        if key == "instruments" and not items:
            tables[key] = {}
            continue
        _require(isinstance(items, dict) and items, f"music_vocabulary.{key}: нужен непустой словарь")
        _require(key != "instruments" or "auto" not in items, "music_vocabulary.instruments: id auto зарезервирован")
        table = {}
        for iid, it in items.items():
            it = it or {}
            _require("/" not in iid, f"music_vocabulary.{key}.{iid}: в id нельзя «/»")
            fb = it.get("fallback")
            _require(fb is None or fb in states, f"music_vocabulary.{key}.{iid}: нет состояния {fb!r}")
            _require(fb is None or states[fb].channel == "music",
                     f"music_vocabulary.{key}.{iid}: резерв {fb!r} должен быть на канале music")
            table[iid] = VocabItem(iid, it.get("label", iid), it.get("text", ""), dict(it.get("config") or {}), fb)
        tables[key] = table
    return MusicVocabulary(**tables, fade_in=float(raw.get("fade_in", 3.0)),
                           fade_out=float(raw.get("fade_out", 3.0)), gain_db=float(raw.get("gain_db", -8.0)),
                           solo=str(raw.get("solo") or ""))
