"""Командная строка.

  python -m soundmaster run            — запустить пульт (http://127.0.0.1:8765)
  python -m soundmaster devices        — список аудиоустройств (имена для config)
  python -m soundmaster check          — проверить конфигурацию и файлы
  python -m soundmaster make-test-sounds — создать тестовые звуки для всех файлов реестра
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
    sub.add_parser("devices", help="список аудиоустройств")
    sub.add_parser("check", help="проверить конфигурацию")
    m = sub.add_parser("make-test-sounds", help="создать тестовые звуки")
    m.add_argument("--force", action="store_true", help="перезаписать существующие файлы")
    args = p.parse_args(argv)

    if args.cmd == "devices":
        from .audio.devices import list_devices

        print(list_devices())
        return 0

    from .app import load_show_config
    from .audio.inputs import load_input_profile
    from .registry import load_registry

    cfg = load_show_config(args.config)
    registry = load_registry(cfg.rel(cfg.raw.get("registry", "registry.yaml")))

    if args.cmd == "make-test-sounds":
        from .testsounds import make_test_sounds

        written = make_test_sounds(registry, force=args.force)
        print(f"Создано файлов: {len(written)}" + "".join(f"\n  {w}" for w in written))
        return 0

    if args.cmd == "check":
        profile = load_input_profile(cfg.rel(cfg.raw["input_profile"]))
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

    app = App(args.config, null_audio=args.null_audio, no_input=args.no_input)
    host = cfg.get("panel", "host", "127.0.0.1")
    port = int(cfg.get("panel", "port", 8765))
    url = f"http://{host}:{port}"
    print(app.profile.describe())
    print(f"\nПульт: {url}\nЗапись и журнал: {app.run_dir}\nОстановка: Ctrl+C")
    if not args.no_browser:
        webbrowser.open(url)
    run_panel(app, host, port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
