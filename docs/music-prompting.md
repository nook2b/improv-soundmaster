# Музыка под речь: как промптить генераторы (исследование 2026-10-17)

Прочитано целиком (полный доступ к сети): официальные гайды Google (Lyria 3 / 3.5 / RealTime),
ElevenLabs (best practices, composition plans, API), Stability (Stable Audio 2.5 и 3.0), статьи про
андерскор и импров. Ссылки — в конце.

## Общие правила музыки под диалог

- **Тон и фактура, а не мелодия.** Андерскор не должен привлекать внимание; ни один инструмент не
  выходит вперёд с большой мелодией (Thinkspace, Tim Johnson). Циммер — тянущиеся струнные или
  пульсирующий синт под сценой.
- **Уйти из диапазона голоса.** Основная масса голоса — 500 Гц–1 кГц, разборчивость согласных —
  2–4 кГц. Виолончель «прямо в частоте диалога» звукорежиссёр уберёт почти в ноль. Значит: по-настоящему
  низко (ниже ~250 Гц) или высоко, а не «средний регистр».
- **Громкость.** BBC: под речью музыку ещё на 3–4 дБ ниже, без слов и тяжёлых ударных; немецкие
  вещатели — речь громче фона минимум на 7 LU. Плавное «сжатие вниз» (downward expansion) музыки
  заметно повышает разборчивость (Demonte, Salford) — у нас это ducking в микшере.
- **Импров:** несколько нот задают мир; музыка комментирует, но не забирает фокус; слушать и подстраиваться.

## Lyria (Google)

- Формула: жанр + настроение + инструменты + темп (можно словами или «slow tempo around 72 BPM») +
  «Instrumental only, no vocals». Тональность — «in D minor».
- **Lyria 3.5 сама переписывает промпт и планирует форму песни (вступление, куплет, припев).** Для
  андерскора это вредно — отсюда нарастания. Нужно явно: одна секция, без вступления и концовки,
  ровно; помогают таймкоды «[0:00 - 1:00] …».
- **Сначала Clip** (30 с, дёшево), потом Pro/3.5.
- **Список распознаваемых инструментов** (гайд): Cello, Harp, Kalimba, Marimba, Glockenspiel,
  Vibraphone, Bass Clarinet, Ocarina, Harmonica, Mandolin, Banjo, Slide Guitar, Warm Acoustic Guitar,
  Flamenco Guitar, Smooth Pianos, Rhodes Piano, Harpsichord, Clavichord, Mellotron, Synth Pads,
  Moog Oscillations, Spacey Synths, Alto Saxophone, Trumpet, Tuba, Woodwinds, Fiddle, Viola Ensemble,
  Koto, Sitar, Dulcimer, Hang Drum, Mbira, Steel Drum … Терменвокса, пилы, фагота, контрабаса,
  баритон-саксофона, флейты в списке нет — вероятная причина «не того инструмента» в пуле.
- **RealTime:** инструмент и настроение — отдельными weighted prompts (так их можно крутить
  независимо); `density` низкая = реже ноты; `guidance` выше = точнее, но резче переходы; `scale` —
  тональность (смена требует reset, как bpm).

## ElevenLabs Music

- **По умолчанию `model_id` = music_v1;** есть music_v2 и music_v2_5. Первый раунд сравнения шёл на v1.
- **Composition plan** (только v2/v2.5): блоки с `positive_styles` и **`negative_styles`** (до 50,
  «используйте щедро»). Первый блок задаёт тон — 6–7 стилей. Это настоящий негатив-промпт:
  drums, bass, piano, strings, chords, harmony, pads, build-up, crescendo, vocals.
- Петли — «упражнение в исключении»: слово **just** («just solo cello»), иначе модель заполняет тишину.
- «solo» перед инструментом; BPM и тональность держит точно; студийные слова работают
  (close-mic'd, bone-dry, dead room).

## Stable Audio

- 2.5 (API Stability: модели `stable-audio-2.5`, `stable-audio-2`): жанр → инструменты → настроение →
  детали; 60–80 BPM = медленно; помогают «perfect for …» (сценарий) и производственные слова
  (minimalist, stripped-back).
- **3.0 (август 2026): `TrackType: Instrument` — отдельный инструмент (стем)**, ровно наш случай.
  В API Stability его нет; есть на fal.ai (`fal-ai/stable-audio-3/small/music/text-to-audio`) и открытые
  веса (Small Music идёт локально — возможно, на Mac, бесплатно и без сети).

## Что из этого следует для пула

1. Инструменты пула — из списка распознаваемых (для Lyria) и в регистре вне голоса.
2. Промпт — «тон и фактура»: тянущиеся ноты / мотив из 2–4 нот, без «мелодии».
3. ElevenLabs — v2.5 + composition plan с negative_styles.
4. Попробовать Stable Audio 3.0 `TrackType: Instrument`.
5. Обработка записи: вырез 2–5 кГц, нормализация; вживую — ducking под речью.

## Источники

- https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-lyria-3-pro
- https://ai.google.dev/gemini-api/docs/lyria-prompt-guide
- https://ai.google.dev/gemini-api/docs/music-generation
- https://ai.google.dev/gemini-api/docs/realtime-music-generation
- https://elevenlabs.io/docs/overview/capabilities/music/best-practices
- https://elevenlabs.io/docs/eleven-api/guides/how-to/music/composition-plans
- https://elevenlabs.io/docs/api-reference/music/compose
- https://stability.ai/guides/stable-audio-25-prompt-guide-bg226-TgFnU
- https://stability.ai/guides/stable-audio-3-prompt-guide
- https://thinkspace.ac.uk/blog/blog-how-to-underscore/
- https://hooplaimpro.com/?p=5675
- https://comedykeys.podbean.com/
- https://salford-repository.worktribe.com/output/3636737/effect-of-background-music-arrangement-and-tempo-on-foreground-speech-intelligibility-listening-experiment-data-and-statistical-analyses-spreadsheets
- https://salford-repository.worktribe.com/output/3626626/downward-dynamic-range-expansion-versus-ducking-test-of-speech-intelligibility-data
