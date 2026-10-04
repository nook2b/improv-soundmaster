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
