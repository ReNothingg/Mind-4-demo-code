# Mind 4

Mind 4 — это папка с кодом. Это лаборатория для тех, кто хочет понять, как работает +- генеративная модель, и попробовать её «вживую» — от запуска простого чата до экспериментов с ядрами на GPU.

Перед вами — экспериментальная архитектура трансформера для локального обучения, текстовой генерации и диалогового взаимодействия. В репозитории есть две основные линии исполнения: понятная и гибкая PyTorch‑версия и ускоренные компоненты на Triton/Metal. В комплекте — токенизатор, полезные утилиты и небольшая система оценки качества моделей.

Этот документ подскажет, что здесь к чему: какие требования, как установить и запустить, где менять параметры и как отлаживать. В конце — карта проекта, чтобы легко ориентироваться.

## Содержание

- Обзор
- Требования
- Установка
- Быстрый старт
- Конфигурация (пример и справочник полей)
- Обучение и инференс
- Оценка качества
- Сборка и использование `metal/`
- Токенизатор
- Производительность и рекомендации
- Устранение неполадок (FAQ)
- Приложение: Полная структура репозитория

---

## Обзор

В основе Mind 4 — хорошо знакомый трансформер, но с акцентом на практичность и скорость. Это «учебно‑боевой» конструктор: его можно использовать как справочную реализацию на PyTorch или подключать ускоренные ядра на Triton/Metal, если хочется выжать максимум из железа.

Поддерживаемые элементы:

- внимание со стандартной схемой SDPA;
- позиционирование RoPE и нормализация RMSNorm;
- семплинг с `top‑k`/`top‑p` и температурой;
- компоненты MoE (Mixture of Experts) для маршрутизации экспертов.

---

## Требования

Если коротко, нужен современный Python и доступ к подходящему ускорителю (по желанию). Подробно:

- Python 3.10+;
- ОС: Linux/macOS/Windows;
- для PyTorch‑инференса — установленный PyTorch (с CUDA, если есть NVIDIA‑GPU);
- для Triton/Metal — совместимая среда Triton; для `metal/` на macOS — Apple Metal, Xcode и CMake;
- рекомендуется использовать виртуальную среду (venv/conda), чтобы не «ломать» системный Python.

Зависимости подбираются под вашу систему: PyTorch ставится по инструкции с `pytorch.org`, Triton — согласно их документации. Специального «замороженного» списка пакетов мы не навязываем, чтобы оставить свободу конфигурации.

---

## Установка

Заведём отдельную среду и поставим всё необходимое. Это безопасно и удобно для экспериментов.

```bash
python -m venv .venv
# Linux/macOS (bash/zsh)
. .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (cmd)
.\.venv\Scripts\activate

pip install --upgrade pip
# Установите PyTorch по инструкции на `https://pytorch.org` (учтите CUDA/MPS)
# По желанию: pip install triton  # для экспериментов c Triton (чаще на Linux)
```

Зависимости для сборки `metal/`:

- macOS: Xcode, CMake 3.20+
- Windows: CMake и MSVC Build Tools (для Python‑обёрток сборка не требуется, если используете PyTorch‑путь)

---

## Быстрый старт

Начнём с самого приятного — заставим модель говорить.

- Генерация текста (PyTorch):

```bash
python model/generate.py
```

- Диалоговый режим (PyTorch):

```bash
python model/chat.py
```

Перед запуском загляните в `config.json`: там указываются размеры модели, пути к весам и токенизатору, параметры семплинга. Для обучения — `config/train.yaml`.

Пример указания пути к локальным модулям, если запускаете из корня (Windows cmd):

```cmd
set PYTHONPATH=%CD%
python model\generate.py
```

Linux/macOS:

```bash
export PYTHONPATH="$PWD"
python model/generate.py
```

---

## [Конфигурация (пример и справочник полей)](config.json)

В проекте два основных файла настроек:

- `config.json` — параметры модели/токенизатора и опции инференса;
- `config/train.yaml` — конфигурация обучения: размеры, контекст, оптимизатор, план обучения, данные.

Ключевые поля:

- Модель: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`.
- Словарь/контекст: `vocab_size`, `initial_context_length`, `max_context_length`, `sliding_window`.
- RoPE/NTK: `rope_theta`, `rope_scaling_factor`, `rope_ntk_alpha`, `rope_ntk_beta`.
- MoE: `num_experts`, `experts_per_token`.
- Активации: `swiglu_limit`.
- Исполнение: `backend` (triton|torch|vllm), `dtype` (float16|bfloat16|float32), `device` (cuda|cpu|mps|auto).
- Пути: `checkpoint` (SafeTensors), `tokenizer` (путь к токенизатору/вокабуляру).
- Семплинг: `sampling.temperature`, `sampling.top_k`, `sampling.top_p`, `sampling.max_new_tokens`.
- Логи: `logging.log_level` (debug|info|warning|error).

---

## Обучение и инференс

- Инференс (PyTorch): `model/generate.py` и `model/chat.py`. Параметры и пути — в `config.json`.
- Инференс (Triton): `model/triton/*` или примеры из `metal/examples/` — если у вас подходящая среда.
- Обучение: ориентируйтесь на `config/train.yaml` и `model/torch/*`. Подготовка/конвертация весов зависит от вашего пайплайна; для локальных экспериментов пригодится `metal/scripts/create-local-model.py`.

---

## Оценка качества

Чтобы не спорить «на глаз», используем измерения. В каталоге `evals/` — сценарии AIME, GPQA, ABCD и агрегатор отчётов. Подробные инструкции — в `evals/README.md`.

Мини‑пример запуска (если предусмотрено конкретным скриптом):

```bash
python evals/basic_eval.py --config config.json --report out/report.json
```

---

## Сборка и использование `metal/`

Если вам интересны низкоуровневые детали и хочется пообщаться с GPU напрямую — загляните в `metal/`. Понадобится macOS с Apple Metal, Xcode и CMake. Базовая схема сборки:

```bash
mkdir -p metal/build
cd metal/build
cmake ..
cmake --build . --config Release
```

Примеры Python (`metal/examples/*.py`) используют собранные нативные компоненты. Убедитесь, что Python видит модуль `metal` (настройте `PYTHONPATH` или установите пакет сборкой). На Windows раздел `metal/` полезен как «анатомия» ядра; исполнять Metal‑ядра там нельзя.

---

## Токенизатор

Символы становятся числами — и наоборот — благодаря токенизатору в `tokenizer/`. Точки входа: `tokenizer.py` и `token_generator.py`. Примеры использования и пояснения — в `tokenizer/README.md`.

---

## Производительность и рекомендации

Немного практических подсказок из «цеха»:

- используйте `dtype=bfloat16` или `float16` на GPU — так быстрее и экономнее по памяти;
- ограничивайте `max_new_tokens` и подбирайте `top_k`/`top_p` для ускорения генерации;
- при длинном контексте включайте `sliding_window` и аккуратно настраивайте `rope_theta`/NTK‑параметры;
- на macOS (M‑серия) для PyTorch выбирайте `device=mps`; Metal‑ядра у нас живут в каталоге `metal/`;
- на Windows без CUDA — `device=cpu` и `dtype=float32` (медленнее, зато стабильно).

---

## Устранение неполадок (FAQ)

- Не находится модуль при запуске Python:
  - проверьте `PYTHONPATH` — укажите корень проекта (см. «Быстрый старт»).
- PyTorch не видит GPU:
  - переустановите PyTorch под вашу версию CUDA и проверьте `torch.cuda.is_available()`.
- Ошибки сборки `metal/` на macOS:
  - установите Xcode и CMake; на Windows Metal не поддерживается.
- Конфликты версий Python/пакетов:
  - обновите `pip` и создайте чистую виртуальную среду.

---

## Приложение: Полная структура репозитория

Ниже — карта проекта. Это как легенда к атласу: быстро понять, где что лежит, и куда идти за нужной деталью.

### Корень

- `README.md` — этот документ.
- `LICENSE` — лицензия проекта.
- `.gitignore` — правила исключения Git.
- `.vscode/settings.json` — настройки рабочей области.
- `config.json` — JSON‑конфигурация модели.
- [`main.md`](https://renothingg.github.io/research/mind4/system) — системные инструкции/промпт и правила поведения модели.

### Каталог `config/`

- `config/train.yaml` — параметры обучения и экспериментов.

### Каталог `model/`

- `model/__init__.py` — инициализация пакета `model`.
- `model/generate.py` — авторегрессивная генерация (temperature, top‑k/top‑p, поток вывода).
- `model/chat.py` — диалоговый интерфейс с историей сообщений.

#### `model/torch/`

- `model/torch/__init__.py` — инициализация подпакета.
- `model/torch/model.py` — трансформер‑модель на PyTorch (внимание, MLP, нормализация, RoPE).
- `model/torch/utils.py` — служебные функции: инициализация, подсчёты и утилиты.
- `model/torch/weights.py` — загрузка/сохранение весов и конвертации форматов.

#### `model/triton/`

- `model/triton/__init__.py` — инициализация подпакета.
- `model/triton/model.py` — сборка модели на Triton.
- `model/triton/attention.py` — оптимизированные компоненты внимания.
- `model/triton/moe.py` — компоненты MoE: маршрутизация и агрегация.

### Каталог `tokenizer/`

- `tokenizer/README.md` — документация токенизатора.
- `tokenizer/tokenizer.py` — реализация токенизатора.
- `tokenizer/token_generator.py` — генерация последовательностей и утилиты инференса.

### Каталог `evals/`

- `evals/README.md` — запуск и метрики.
- `evals/__init__.py` — инициализация пакета.
- `evals/types.py` — типы и интерфейсы задач.
- `evals/basic_eval.py` — базовые сценарии оценки.
- `evals/abcd_grader.py` — оценивание по формату ABCD.
- `evals/aime_eval.py` — сценарии AIME.
- `evals/gpqa_eval.py` — сценарии GPQA.
- `evals/report.py` — агрегация и отчёт по результатам.

### Каталог `metal/` (ядра, обёртки и примеры)

- `metal/__init__.py` — инициализация Python‑пакета `metal`.
- `metal/CMakeLists.txt` — сборка нативных компонентов через CMake.

#### `metal/include/`

- `metal/include/mind-four.h` — публичный заголовок API.
- `metal/include/mind-four/functions.h` — публичные функции API.
- `metal/include/mind-four/macros.h` — общие макросы.

#### `metal/python/` (обёртки для CPython)

- `metal/python/module.h` — заголовок CPython‑модуля.
- `metal/python/module.c` — реализация CPython‑модуля.
- `metal/python/context.c` — управление контекстом.
- `metal/python/model.c` — связки модели.
- `metal/python/tokenizer.c` — связки токенизатора.

#### `metal/scripts/`

- `metal/scripts/create-local-model.py` — подготовка/конвертация локальной модели и весов.

#### `metal/benchmark/`

- `metal/benchmark/end-to-end.cc` — сквозной бенчмарк пути инференса.
- `metal/benchmark/f32-random.cc` — тесты с float32‑данными.
- `metal/benchmark/mf4-f32-convert.cc` — конвертация mf4↔f32, проверка корректности и скорости.
- `metal/benchmark/u32-random.cc` — тесты с u32‑данными.
- `metal/benchmark/f32-bf16w-rmsnorm.cc` — бенчмарк RMSNorm и форматов float.

#### `metal/examples/`

- `metal/examples/generate.py` — пример генерации через нативные ядра.
- `metal/examples/chat.py` — пример диалога через нативные ядра.

#### `metal/source/` (ядра и внутренние реализации)

- `metal/source/accumulate.metal` — аккумуляция.
- `metal/source/convert.metal` — конвертация типов/форматов.
- `metal/source/embeddings.metal` — эмбеддинги.
- `metal/source/generate.c` — функции генерации.
- `metal/source/log.c` — логирование.
- `metal/source/matmul.metal` — матричное умножение.
- `metal/source/metal-kernels.c` — точка регистрации GPU‑ядер.
- `metal/source/metal.m` — интеграция Metal API.
- `metal/source/model.c` — внутренняя логика модели.
- `metal/source/random.metal` — генератор случайных чисел.
- `metal/source/rmsnorm.metal` — RMSNorm.
- `metal/source/rope.metal` — Rotary Positional Embeddings.
- `metal/source/sample.metal` — сэмплинг токенов.
- `metal/source/sdpa.metal` — scaled dot‑product attention.
- `metal/source/topk.metal` — выбор top‑k.
- `metal/source/context.c` — управление контекстом.
- `metal/source/tokenizer.c` — токенизатор (внутренняя реализация).

##### `metal/source/include/internal/` (внутренние заголовки)

- `datatype.h`, `datatype.hpp` — типы данных и утилиты типов.
- `kernel-args.h` — аргументы ядер.
- `log.h` — интерфейсы логирования.
- `macros.h` — служебные макросы.
- `math.h` — математические утилиты.
- `metal-kernels.h` — объявления GPU‑ядер.
- `metal.h`, `metal.hpp` — интеграция с Metal.
- `model.h` — внутренние интерфейсы модели.
- `rng.h`, `rng.hpp` — генераторы случайных чисел.
- `storage.h` — абстракции хранения/буферов.
- `uuid.h` — идентификаторы.
