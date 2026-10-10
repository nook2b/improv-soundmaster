import pytest


def ai(c, effect, clock, **kw):
    args = dict(scene_id=c.state.scene_id, connection_id=c.state.connection_id, t_input_ref=clock.now() - 0.5)
    args.update(kw)
    return c.ai_effect(effect, **args)


def test_manual_play_always_plays(system):
    make, *_ = system
    c = make()
    assert c.manual_play("gunshot").result == "played"
    assert c.manual_play("doorbell").result == "played"


def test_ai_suggests_in_suggest_mode(system, clock):
    make, *_ = system
    c = make()
    d = ai(c, "doorbell", clock)
    assert d.result == "suggested" and d.blocked_by == "mode_suggest"
    assert c.confirm(d.suggestion_id).result == "confirmed"


def test_without_clean_feed_auto_becomes_suggestion(system, clock):
    make, *_ = system
    c = make(clean=False)
    c.set_mode("doorbell", "AUTO")
    d = ai(c, "doorbell", clock)
    assert d.result == "suggested" and d.blocked_by == "no_clean_actor_feed"


def test_auto_plays_with_clean_feed(system, clock):
    make, *_ = system
    c = make(clean=True)
    c.set_mode("doorbell", "AUTO")
    assert ai(c, "doorbell", clock).result == "played"


def test_automation_off_turns_auto_into_suggestion(system, clock):
    make, *_ = system
    c = make(clean=True)
    c.set_mode("doorbell", "AUTO")
    c.set_automation(False)
    assert ai(c, "doorbell", clock).blocked_by == "automation_off"


def test_loud_never_auto(system, clock):
    make, *_ = system
    c = make(clean=True)
    with pytest.raises(ValueError):
        c.set_mode("gunshot", "AUTO")
    with pytest.raises(ValueError):
        c.set_mode("scene_end", "AUTO")


def test_sync_direct_call_blocked(system, clock):
    make, *_ = system
    c = make(clean=True)
    assert ai(c, "camera_shutter", clock).blocked_by == "sync_requires_arm_or_human"


def test_stale_scene_and_connection(system, clock):
    make, *_ = system
    c = make()
    old_scene = c.state.scene_id
    c.new_scene("ресторан")
    assert ai(c, "doorbell", clock, scene_id=old_scene).blocked_by == "stale_scene"
    old_conn = c.state.connection_id
    c.new_connection()
    assert ai(c, "doorbell", clock, connection_id=old_conn).blocked_by == "stale_connection"


def test_late_dropped(system, clock):
    make, *_ = system
    c = make()
    assert ai(c, "doorbell", clock, t_input_ref=clock.now() - 3.5).result == "late_dropped"


def test_dedupe_after_manual_play(system, clock):
    make, *_ = system
    c = make()
    c.manual_play("doorbell")
    clock.advance(1.0)
    assert ai(c, "doorbell", clock).result == "deduped"
    clock.advance(3.0)
    assert ai(c, "doorbell", clock).result == "suggested"


def test_no_duplicate_suggestion(system, clock):
    make, *_ = system
    c = make()
    assert ai(c, "doorbell", clock).result == "suggested"
    assert ai(c, "doorbell", clock).blocked_by == "already_suggested"


def test_manual_play_clears_suggestion(system, clock):
    make, *_ = system
    c = make()
    ai(c, "doorbell", clock)
    c.manual_play("doorbell")
    assert not c.state.suggestions


def test_suggestion_expires_and_late_confirm_does_nothing(system, clock):
    make, engine, mixer, _ = system
    c = make()
    d = ai(c, "doorbell", clock)
    clock.advance(10)
    assert c.confirm(d.suggestion_id).result == "expired"
    d2 = ai(c, "knock", clock)
    clock.advance(10)
    c.tick()
    assert d2.suggestion_id not in c.state.suggestions


def test_dismissed_not_resuggested_immediately(system, clock):
    make, *_ = system
    c = make()
    d = ai(c, "doorbell", clock)
    c.dismiss(d.suggestion_id)
    clock.advance(1)
    assert ai(c, "doorbell", clock).blocked_by == "dismissed_by_operator"


def test_global_gap_between_different_effects(system, clock):
    make, *_ = system
    c = make()
    ai(c, "doorbell", clock)
    clock.advance(1)
    assert ai(c, "knock", clock).blocked_by == "global_gap"
    clock.advance(3)
    assert ai(c, "knock", clock).result == "suggested"


def test_scene_end_rules(system, clock):
    make, *_ = system
    c = make()
    kw = dict(scene_id=c.state.scene_id, connection_id=0)
    d = c.ai_scene_end(**kw)
    assert d.result == "suggested"
    assert c.ai_scene_end(**kw).blocked_by == "already_suggested"
    c.dismiss(d.suggestion_id)
    assert c.ai_scene_end(**kw).blocked_by == "pause_after_dismiss"
    clock.advance(21)
    assert c.ai_scene_end(**kw).result == "suggested"  # повторная подсказка после паузы разрешена


def test_scene_end_max_per_scene(system, clock):
    make, *_ = system
    c = make()
    for _ in range(3):
        d = c.ai_scene_end(scene_id=c.state.scene_id, connection_id=0)
        assert d.result == "suggested"
        c.dismiss(d.suggestion_id)
        clock.advance(21)
    assert c.ai_scene_end(scene_id=c.state.scene_id, connection_id=0).blocked_by == "max_per_scene"


def test_confirm_scene_end_plays_sting(system, clock):
    make, engine, mixer, _ = system
    c = make()
    d = c.ai_scene_end(scene_id=c.state.scene_id, connection_id=0)
    r = c.confirm(d.suggestion_id)
    assert r.result == "confirmed" and r.effect == "scene_end"


def test_new_scene_clears_suggestions(system, clock):
    make, *_ = system
    c = make()
    ai(c, "doorbell", clock)
    c.new_scene()
    assert not c.state.suggestions


def test_ai_music_only_suggests(system, clock):
    make, *_ = system
    c = make(clean=True)
    d = c.ai_music("romantic", scene_id=c.state.scene_id, connection_id=0, t_input_ref=clock.now())
    assert d.result == "suggested"
    assert c.confirm(d.suggestion_id).result == "confirmed"
    assert c.music.active["music"].state.id == "romantic"


# ---- автономная музыка ---------------------------------------------------------------------
from soundmaster.registry import MusicStyle  # noqa: E402

NOIR = MusicStyle("noir", "tense", "low", "slow", "saxophone")
COMEDY = MusicStyle("comedy", "playful", "medium", "fast", "ukulele")


def style(c, s, clock, **kw):
    kw.setdefault("t_input_ref", clock.now())
    return c.ai_music_style(s, scene_id=c.state.scene_id, connection_id=c.state.connection_id, **kw)


def auto(system):
    make = system[0]
    c = make()
    c.music_mode = "AUTO"
    return c


def test_music_style_auto_starts_without_human_even_without_clean_feed(system, clock):
    c = auto(system)
    d = style(c, NOIR, clock)
    assert d.result == "started"
    assert c.music.current("music").id == "style:noir/tense/low/slow/saxophone"
    assert c.music.active["music"].owner == "ai"
    assert style(c, NOIR, clock).result == "deduped"


def test_music_style_min_hold_then_change(system, clock):
    c = auto(system)
    style(c, NOIR, clock)
    clock.advance(5)
    d = style(c, COMEDY, clock)
    assert (d.result, d.blocked_by) == ("blocked", "min_hold")
    clock.advance(c.music_min_hold)
    assert style(c, COMEDY, clock).result == "started"
    assert c.music.current("music").id.endswith("comedy/playful/medium/fast/ukulele")


def test_new_scene_stops_music_and_allows_immediate_choice(system, clock):
    c = auto(system)
    style(c, NOIR, clock)
    clock.advance(2)
    c.new_scene(source="video")
    assert c.music.current("music") is None
    assert style(c, COMEDY, clock).result == "started"


def neutral_auto(system):
    c = auto(system)
    c.scene_cut_music = "neutral"
    c.scene_cut_mode = "AUTO"
    c.music_neutral = MusicStyle("underscore", "calm", "low", "slow", "piano")
    return c


def test_scene_cut_goes_to_neutral_then_model_choice_replaces_it_at_once(system, clock):
    c = neutral_auto(system)
    c.manual_music("restaurant")
    style(c, NOIR, clock)
    clock.advance(c.scene_cut_cooldown + 1)
    assert c.scene_cut("video").result == "started"
    cur = c.music.current("music")
    assert cur.id == "style:underscore/calm/low/slow/piano" and c.music.active["music"].owner == "system"
    assert c.music.current("ambience") is None  # атмосфера прошлого места гаснет
    clock.advance(1)
    assert style(c, COMEDY, clock).result == "started"  # нейтральную не держим


def test_first_pick_in_scene_can_be_corrected_sooner(system, clock):
    c = neutral_auto(system)
    style(c, NOIR, clock)
    clock.advance(c.music_first_hold + 0.5)
    assert style(c, COMEDY, clock).result == "started"  # первый выбор — держится только 10 с
    clock.advance(c.music_first_hold + 0.5)
    d = style(c, MusicStyle("drama", "sad", "low", "slow", "cello"), clock)
    assert (d.result, d.blocked_by) == ("blocked", "min_hold")  # дальше — 30 с
    assert [sid for _, sid in c.music_history] == [NOIR.id, COMEDY.id]


def test_neighbouring_scenes_do_not_repeat_instrument(system, clock):
    c = neutral_auto(system)
    drama = MusicStyle("drama", "tense", "low", "slow", "cello")
    assert style(c, drama, clock).result == "started"
    clock.advance(c.scene_cut_cooldown + 1)
    c.scene_cut("video")
    d = style(c, MusicStyle("comedy", "playful", "medium", "medium", "cello"), clock)  # другой жанр, тот же инструмент
    assert (d.result, d.blocked_by) == ("blocked", "same_as_prev_scene")
    assert style(c, MusicStyle("drama", "sad", "low", "slow", "piano"), clock).result == "started"
    clock.advance(c.scene_cut_cooldown + 1)
    c.scene_cut("video")
    assert style(c, drama, clock).result == "started"  # через сцену — можно


def test_solo_vocabulary_requires_an_instrument(system, clock):
    c = auto(system)
    d = style(c, MusicStyle("drama", "tense", "low", "slow"), clock)
    assert (d.result, d.blocked_by) == ("blocked", "instrument_required")


def test_scene_instrument_locked_after_first_correction(system, clock):
    c = neutral_auto(system)
    assert style(c, MusicStyle("drama", "tense", "low", "slow", "cello"), clock).result == "started"
    clock.advance(c.music_first_hold + 1)
    d = style(c, MusicStyle("drama", "tense", "low", "slow", "violin"), clock)  # первое уточнение — можно
    assert d.result == "started"
    clock.advance(c.music_min_hold + 1)
    d = style(c, MusicStyle("drama", "tense", "low", "slow", "piano"), clock)  # только инструмент — уже не меняем
    assert d.result == "deduped"
    d = style(c, MusicStyle("drama", "sad", "low", "slow", "piano"), clock)  # поворот: настроение да, инструмент нет
    assert d.result == "started" and c.music.current("music").id == "style:drama/sad/low/slow/violin"
    clock.advance(c.scene_cut_cooldown + 1)
    c.scene_cut("video")
    assert style(c, MusicStyle("folk", "tender", "low", "slow", "guitar"), clock).result == "started"  # новая сцена


def test_music_style_with_solo_instrument(system, clock):
    c = auto(system)
    s = MusicStyle("folk", "tender", "low", "slow", "guitar")
    assert style(c, s, clock).result == "started"
    st = c.music.current("music")
    assert st.id == "style:folk/tender/low/slow/guitar" and "гитара" in st.label
    texts = {p["text"]: p["weight"] for p in st.generator["prompts"]}
    guitar = next(t for t in texts if "guitar" in t)
    assert max(texts.values()) == texts[guitar]  # инструмент весит больше жанра
    assert any("no accompaniment" in t for t in texts)  # сольно: Lyria не дописывает аккомпанемент
    assert style(c, MusicStyle("folk", "tender", "low", "slow", "kazoo"), clock).blocked_by == "unknown_instrument"


def test_music_style_does_not_override_manual_music(system, clock):
    c = auto(system)
    c.manual_music("romantic")
    d = style(c, NOIR, clock)
    assert (d.result, d.blocked_by) == ("blocked", "manual_music")


def test_music_style_suggest_mode_and_confirm(system, clock):
    c = system[0]()
    d = style(c, NOIR, clock)
    assert (d.result, d.blocked_by) == ("suggested", "background_suggest_only")
    assert c.music.current("music") is None
    assert c.confirm(d.suggestion_id).result == "confirmed"
    assert c.music.current("music").id == "style:noir/tense/low/slow/saxophone"


def test_music_style_stop_ai_turns_auto_into_suggestion(system, clock):
    c = auto(system)
    style(c, NOIR, clock)
    c.stop_ai_sound()
    assert c.music.current("music") is None
    d = style(c, COMEDY, clock)
    assert (d.result, d.blocked_by) == ("suggested", "automation_off")


def test_music_style_rejects_unknown_words_and_late(system, clock):
    c = auto(system)
    assert style(c, MusicStyle("polka", "tense", "low", "slow", "piano"), clock).blocked_by == "unknown_genre"
    clock.advance(10)
    assert style(c, NOIR, clock, t_input_ref=clock.now() - 10).result == "late_dropped"


def test_ambience_is_manual_only(system, clock):
    c = auto(system)
    d = c.ai_music("restaurant", scene_id=c.state.scene_id, connection_id=c.state.connection_id,
                   t_input_ref=clock.now())
    assert (d.result, d.blocked_by) == ("blocked", "manual_only") and c.music.current("ambience") is None
    c.manual_music("restaurant")  # человек с пульта — можно
    assert c.music.current("ambience").id == "restaurant"


def test_model_does_not_see_manual_only_ambience(registry):
    from soundmaster.brain.prompt import action_types, allowed_ids, system_text

    assert "background" not in action_types(registry)
    assert "restaurant" not in allowed_ids(registry) and "restaurant" not in system_text(registry)


def test_show_starts_with_neutral_music(system, clock):
    c = neutral_auto(system)
    assert c.start_neutral_music()
    assert c.music.current("music").id == "style:underscore/calm/low/slow/piano"
    assert c.music.active["music"].owner == "system"
    assert style(c, COMEDY, clock).result == "started"  # модель сразу заменяет нейтральную


def test_no_neutral_music_when_music_is_not_automatic(system, clock):
    c = neutral_auto(system)
    c.music_mode = "SUGGEST"
    assert not c.start_neutral_music() and c.music.current("music") is None
