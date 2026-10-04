"""Командная строка.

  python -m soundmaster run            — запустить пульт (http://127.0.0.1:8765)
  python -m soundmaster run --input mac-mic — домашний тест с микрофоном Mac
  python -m soundmaster devices        — список аудиоустройств (имена для config)
  python -m soundmaster check          — проверить конфигурацию и файлы
  python -m soundmaster make-test-sounds — создать тестовые звуки для всех файлов реестра
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
    sub.add_parser("devices", help="список аудиоустройств")
    sub.add_parser("cameras", help="список камер macOS")
    rp = sub.add_parser("replay", help="прогнать запись через систему (ничего не звучит, всё в отчёт)")
    rp.add_argument("media", help="видео/аудио файл записи или ссылка (YouTube, Drive — нужен yt-dlp)")
    rp.add_argument("--asr", choices=["mlx", "faster", "off"], help="распознавание (по умолчанию — из show.yaml)")
    rp.add_argument("--brain", choices=["openai", "gemini", "off"], help="облачная модель (по умолчанию — из show.yaml)")
    rp.add_argument("--transcript", help="готовая расшифровка вместо распознавания: строки «0:12-0:15 текст»")
    rp.add_argument("--fast", action="store_true", help="быстрее реального времени (только без облачной модели)")
    rp.add_argument("--scene-cut", choices=["OFF", "SUGGEST", "AUTO"], help="режим «монтажа» для прогона")
    rp.add_argument("--clean-feed", action="store_true", help="считать звук чистым (как с петличек)")
    rp.add_argument("--no-video", action="store_true", help="не анализировать картинку")
    rp.add_argument("--max-seconds", type=float, help="прогнать только начало записи")
    ch = sub.add_parser("check", help="проверить конфигурацию")
    ch.add_argument("--input", help=input_help)
    m = sub.add_parser("make-test-sounds", help="создать тестовые звуки")
    m.add_argument("--force", action="store_true", help="перезаписать существующие файлы")
    args = p.parse_args(argv)

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
            asr=args.asr, brain=args.brain,
            transcript=Path(args.transcript).expanduser() if args.transcript else None,
            scene_cut_mode=args.scene_cut, clean_feed=args.clean_feed, video=not args.no_video,
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

    if args.cmd == "check":
        profile = load_input_profile(resolve_input_profile(cfg, args.input))
        print(profile.describe())
        print(f"\nЭффектов: {len(registry.effects)}, музыкальных состояний: {len(registry.music_states)}")
        missing = [e.file for e in registry.effects.values() if not registry.path_of(e.file).exists()]
        missing += [s.file for s in registry.music_states.values() if s.file and not registry.path_of(s.file).exists()]
        if missing:
            print("Нет файлов (можно создать тестовые: make-test-sounds):\n  " + "\n  ".join(missing))
            return 1
        print("Все файлы на месте.")
        return 0

    from .app import App
    from .panel.server import run_panel

    app = App(args.config, null_audio=args.null_audio, no_input=args.no_input, input_profile=args.input)
    host = cfg.get("panel", "host", "127.0.0.1")
    port = int(cfg.get("panel", "port", 8765))
    url = f"http://{host}:{port}"
    print(app.profile.describe())
    print(f"\nПульт: {url}\nЗапись и журнал: {app.run_dir}\nОстановка: Ctrl+C")
    if not args.no_browser:
        webbrowser.open(url)
    run_panel(app, host, port)
    return 0


def cli() -> int:
    try:
        return main()
    except (FileNotFoundError, ValueError, RuntimeError) as e:  # ошибки конфигурации — коротко, без трассировки
        print(f"Ошибка: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(cli())
