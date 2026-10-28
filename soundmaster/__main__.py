"""Командная строка.

  python -m soundmaster run            — запустить пульт (http://127.0.0.1:8765)
  python -m soundmaster run --input mac-mic — домашний тест с микрофоном Mac
  python -m soundmaster devices        — список аудиоустройств (имена для config)
  python -m soundmaster check          — проверить конфигурацию и файлы
  python -m soundmaster make-test-sounds — создать тестовые звуки для всех файлов реестра
  python -m soundmaster import-sounds ПАПКА — взять звуки (mp3, mp4, wav …) по именам id из реестра
  python -m soundmaster make-music-pool — записать мелодии для кнопок жанров и резерва без интернета (ElevenLabs)
  python -m soundmaster replay ЗАПИСЬ.mp4 — прогнать запись через систему (теневой режим) + отчёт
  python -m soundmaster cameras        — список камер (имя для config: video.camera)
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "show.yaml"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="soundmaster", description="AI-звукарь для импров-шоу")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="путь к show.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="запустить пульт")
    r.add_argument("--null-audio", action="store_true", help="без звуковой карты (проверка интерфейса)")
    r.add_argument("--no-input", action="store_true", help="не открывать вход звука")
    r.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    input_help = "профиль входа вместо указанного в show.yaml: mac-mic, brio-mic, recorder или путь к YAML"
    r.add_argument("--input", help=input_help)
    r.add_argument("--asr", choices=["mlx", "faster", "off"], help="распознавание (вместо listen.asr в show.yaml)")
    r.add_argument("--asr-model", help="модель распознавания (faster: small / large-v3-turbo …)")
    r.add_argument("--brain", choices=["openai", "gemini", "off"], help="облачная модель (вместо brain.provider)")
    r.add_argument("--camera", help="включить камеру: имя из `cameras` (вместо video.enabled/camera)")
    sub.add_parser("devices", help="список аудиоустройств")
    sub.add_parser("cameras", help="список камер macOS")
    rp = sub.add_parser("replay", help="прогнать запись через систему (ничего не звучит, всё в отчёт)")
    rp.add_argument("media", help="видео/аудио файл записи или ссылка (YouTube, Drive — нужен yt-dlp)")
    rp.add_argument("--asr", choices=["mlx", "faster", "off"], help="распознавание (по умолчанию — из show.yaml)")
    rp.add_argument("--asr-model", help="модель распознавания (mlx: репозиторий HF, faster: small / large-v3-turbo …)")
    rp.add_argument("--brain", choices=["openai", "gemini", "off"], help="облачная модель (по умолчанию — из show.yaml)")
    rp.add_argument("--transcript", help="готовая расшифровка вместо распознавания: строки «0:12-0:15 текст»")
    rp.add_argument("--fast", action="store_true", help="быстрее реального времени (только без облачной модели)")
    rp.add_argument("--scene-cut", choices=["OFF", "SUGGEST", "AUTO"], help="режим «монтажа» для прогона")
    rp.add_argument("--scene-cut-detector", choices=["people", "pixel"],
                    help="чем ловить «монтаж»: обученной моделью (people) или старым детектором по пикселям")
    rp.add_argument("--clean-feed", action="store_true", help="считать звук чистым (как с петличек)")
    rp.add_argument("--no-video", action="store_true", help="не анализировать картинку")
    rp.add_argument("--max-seconds", type=float, help="прогнать только начало записи")
    ch = sub.add_parser("check", help="проверить конфигурацию")
    ch.add_argument("--input", help=input_help)
    m = sub.add_parser("make-test-sounds", help="создать тестовые звуки")
    m.add_argument("--force", action="store_true", help="перезаписать существующие файлы")
    im = sub.add_parser("import-sounds", help="импорт звуков из папки (имя файла = id эффекта/фона)")
    im.add_argument("folder", help="папка с файлами: doorbell.mp3, rain.mp4 …")
    im.add_argument("--force", action="store_true", help="перезаписать существующие (в т. ч. тестовые) файлы")
    mp = sub.add_parser("make-music-pool",
                        help="записать мелодии пула (кнопки жанров и резерв без интернета)")
    mp.add_argument("--only", help="только жанр (noir) или мелодия (noir/2); через запятую — несколько")
    mp.add_argument("--engine", choices=["elevenlabs", "lyria"], default="elevenlabs",
                    help="генератор: elevenlabs (ELEVENLABS_API_KEY, по умолчанию) или lyria (GEMINI_API_KEY)")
    mp.add_argument("--seconds", type=float, help="длина петли, с (elevenlabs — 30, lyria — 60)")
    mp.add_argument("--parallel", type=int, help="сколько записывать одновременно (elevenlabs — 2, lyria — 4)")
    mp.add_argument("--force", action="store_true", help="перезаписать уже записанные")
    mp.add_argument("--takes", type=int, help="дублей на мелодию, остаётся самый спокойный (elevenlabs — 1, lyria — 3)")
    cs = sub.add_parser("collect-scenecuts",
                        help="собрать примеры «монтажа» с записей (плейлист/ссылка/файл) для обучения детектора")
    cs.add_argument("sources", nargs="+", help="ссылка на плейлист или видео YouTube, или файлы записей")
    cs.add_argument("--limit", type=int, help="сколько видео пройти за этот запуск")
    cs.add_argument("--only-shows", action="store_true", help="только шоу (в названии «Компот» или «шоу»)")
    cs.add_argument("--out", default="data/scenecut", help="куда писать рамки людей (в git)")
    cs.add_argument("--clips", default="recordings/scenecut_clips", help="куда писать ролики (не в git)")
    cs.add_argument("--no-push", action="store_true", help="не отправлять собранное на GitHub самому")
    cs.add_argument("--save-clips", action="store_true",
                    help="нарезать ролики mp4 (не нужно для YouTube: разметка смотрит видео во встроенном плеере)")
    cs.add_argument("--model", default="yolo11s-pose.pt",
                    help="нейросеть для людей и их позы (yolo11n-pose.pt — быстрее, yolo11m-pose.pt — точнее)")
    cs.add_argument("--emb-weights", help="файл весов DINOv2 (model.safetensors), если Hugging Face не открывается")
    cs.add_argument("--labeled", action="store_true",
                    help="только видео, где уже есть разметка (пересбор для новых признаков, разметка сохраняется)")
    ls_ = sub.add_parser("label-scenecuts", help="разметить собранные ролики «монтаж / не монтаж» в браузере")
    ls_.add_argument("--out", default="data/scenecut")
    ls_.add_argument("--clips", default="recordings/scenecut_clips")
    ls_.add_argument("--port", type=int, default=8766)
    ls_.add_argument("--no-push", action="store_true", help="не отправлять отметки на GitHub самому")
    ts = sub.add_parser("train-scenecut", help="обучить модель «монтажа» на разметке (нужен scikit-learn)")
    ts.add_argument("--data", default="data/scenecut")
    ts.add_argument("--out", default="config/scenecut_model.json")
    ts.add_argument("--trees", type=int, default=200)
    args = p.parse_args(argv)

    if args.cmd == "collect-scenecuts":
        from .learn.collect import collect

        try:
            res = collect(args.sources, Path(args.out), Path(args.clips), limit=args.limit,
                          only_shows=args.only_shows, model=args.model, save_clips=args.save_clips,
                          push=not args.no_push, labeled_only=args.labeled,
                          emb_weights=args.emb_weights)
        except (RuntimeError, ImportError) as e:
            print(f"Ошибка: {e}")
            return 1
        print(f"Готово: видео {len(res)}, кандидатов {sum(len(r['candidates']) for r in res)}. "
              "Разметка: python -m soundmaster label-scenecuts")
        return 0

    if args.cmd == "label-scenecuts":
        from .learn.label import serve

        serve(Path(args.out), Path(args.clips), port=args.port, push=not args.no_push)
        return 0

    if args.cmd == "train-scenecut":
        from .learn.model import train

        train(Path(args.data), Path(args.out), trees=args.trees)
        return 0

    if args.cmd == "devices":
        from .audio.devices import list_devices

        print(list_devices())
        return 0

    if args.cmd == "cameras":
        from .media.ffmpeg import list_cameras

        print(list_cameras())
        return 0

    from .app import load_show_config, resolve_input_profile
    from .audio.inputs import load_input_profile
    from .registry import load_registry

    cfg = load_show_config(args.config)
    registry = load_registry(cfg.rel(cfg.raw.get("registry", "registry.yaml")))

    if args.cmd == "replay":
        from .replay import Replay, ReplayOptions, print_summary

        from .media.download import download, is_url

        media = Path(args.media).expanduser().resolve()
        if is_url(args.media):
            target = cfg.rel(cfg.get("recording", "dir", "../recordings")) / "downloads"
            print(f"Скачиваю {args.media} → {target} …")
            media = download(args.media, target)
            print(f"Готово: {media}")
        opts = ReplayOptions(
            media=media, config=Path(args.config), speed=0 if args.fast else 1,
            asr=args.asr, asr_model=args.asr_model, brain=args.brain,
            transcript=Path(args.transcript).expanduser() if args.transcript else None,
            scene_cut_mode=args.scene_cut, scene_cut_kind=args.scene_cut_detector, clean_feed=args.clean_feed, video=not args.no_video,
            max_seconds=args.max_seconds,
        )
        if not opts.media.exists():
            raise FileNotFoundError(f"нет файла {opts.media}")
        rp = Replay(opts)
        print(f"Прогон {opts.media.name}: {rp.info.duration:.0f} с, "
              f"{'быстрый режим' if rp.fast else 'реальный темп'}; "
              f"распознавание: {rp.asr.name if rp.asr else 'нет'}; "
              f"модель: {rp.brain.name + ':' + rp.brain.model if rp.brain else 'нет'}. Остановка: Ctrl+C")
        if rp.asr and rp.asr.name in ("mlx", "faster"):
            print("Загрузка модели распознавания (первый раз — скачивание, может занять несколько минут)…")
            rp.asr.warmup()
        print_summary(rp.run(), rp.out_dir)
        return 0


    if args.cmd == "make-test-sounds":
        from .testsounds import make_test_sounds

        written = make_test_sounds(registry, force=args.force)
        print(f"Создано файлов: {len(written)}" + "".join(f"\n  {w}" for w in written))
        return 0

    if args.cmd == "import-sounds":
        from .importsounds import import_sounds

        try:
            rep = import_sounds(registry, Path(args.folder), force=args.force)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Ошибка: {e}")
            return 1
        for sid, dst in rep.imported:
            print(f"  ✓ {sid} → {dst}")
        for sid, dst in rep.skipped:
            print(f"  = {sid}: уже есть {dst} (заменить: --force)")
        for src, err in rep.errors:
            print(f"  ✗ {src.name}: {err}")
        if rep.unknown:
            print("Нет такого id в реестре (переименуйте файл или добавьте запись в registry.yaml):\n  "
                  + "\n  ".join(p.name for p in rep.unknown))
        if rep.missing:
            print("Ещё нет файлов для: " + ", ".join(rep.missing) + " (временно: make-test-sounds)")
        print(f"Импортировано: {len(rep.imported)}")
        return 1 if rep.errors else 0

    if args.cmd == "make-music-pool":
        from .music.pool import make_pool

        try:
            eleven = args.engine == "elevenlabs"
            rep = make_pool(registry, seconds=args.seconds or (30.0 if eleven else 60.0),
                            parallel=args.parallel or (2 if eleven else 4), only=args.only, force=args.force,
                            takes=args.takes or (1 if eleven else 3), engine=args.engine)
        except (KeyError, RuntimeError) as e:
            print(f"Ошибка: {e}")
            return 1
        print(f"Записано: {len(rep.made)}, уже было: {len(rep.skipped)} (перезаписать: --force), "
              f"ошибок: {len(rep.errors)}" + (" — запустите ещё раз, допишутся только недостающие" if rep.errors else ""))
        return 1 if rep.errors else 0

    if args.cmd == "check":
        profile = load_input_profile(resolve_input_profile(cfg, args.input))
        print(profile.describe())
        print(f"\nЭффектов: {len(registry.effects)}, музыкальных состояний: {len(registry.music_states)}")
        print(scenecut_status(cfg.raw.get("scene_cut") or {}))
        missing = [e.file for e in registry.effects.values() if not registry.path_of(e.file).exists()]
        missing += [s.file for s in registry.music_states.values() if s.file and not registry.path_of(s.file).exists()]
        if missing:
            print("Нет файлов (можно создать тестовые: make-test-sounds):\n  " + "\n  ".join(missing))
            return 1
        print("Все файлы на месте.")
        return 0

    from .app import App
    from .panel.server import run_panel

    app = App(args.config, null_audio=args.null_audio, no_input=args.no_input, input_profile=args.input,
              overrides={"asr": args.asr, "asr_model": args.asr_model, "brain": args.brain, "camera": args.camera})
    host = cfg.get("panel", "host", "127.0.0.1")
    port = int(cfg.get("panel", "port", 8765))
    url = f"http://{host}:{port}"
    print(app.profile.describe())
    print(f"\nПульт: {url}\nЗапись и журнал: {app.run_dir}\nОстановка: Ctrl+C")
    if not args.no_browser:
        webbrowser.open(url)
    run_panel(app, host, port)
    return 0


def scenecut_status(sc: dict) -> str:
    """Чем будет ловиться «монтаж»: обученной моделью (и есть ли для неё YOLO) или пикселями."""
    import importlib.util

    from .learn.model import CutModel
    from .media.peoplecut import PeopleCutConfig

    if str(sc.get("kind", "people")) != "people":
        return "«Монтаж»: старый детектор по пикселям (scene_cut.kind: pixel)."
    pc = PeopleCutConfig.from_dict(sc.get("people"))
    if not Path(pc.model).exists():
        return f"«Монтаж»: нет модели {pc.model} — работает детектор по пикселям (python -m soundmaster train-scenecut)."
    info = CutModel.load(Path(pc.model)).info
    c = info.get("check", {})
    line = (f"«Монтаж»: модель {pc.model} (обучена {info.get('trained', '?')} на {info.get('videos', '?')} видео; "
            f"на невиданных видео находит {c.get('recall', 0):.0%}, точность {c.get('precision', 0):.0%}).")
    if importlib.util.find_spec("ultralytics") is None:
        line += ("\n  Нет ultralytics — на шоу будет детектор по пикселям. Установить: pip install -e \".[learn]\""
                 " (веса YOLO скачаются при первом запуске — сделайте это до шоу, с интернетом).")
    return line


def cli() -> int:
    try:
        return main()
    except (FileNotFoundError, ValueError, RuntimeError) as e:  # ошибки конфигурации — коротко, без трассировки
        print(f"Ошибка: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(cli())
