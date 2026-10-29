# Гайд по ML-проекту

Практический гайд по организации ML-проекта: структура репозитория, инструменты, конвенции.
Каждый раздел отвечает на вопрос «что делать и почему именно так».

Стек, который закрывает весь цикл работы и не требует самописной инфраструктуры:

| Задача | Инструмент | Почему он |
| --- | --- | --- |
| Конфигурация | **Hydra** (+ OmegaConf) | Фактический стандарт в ML: группы конфигов, override из CLI, multirun, версионирование конфига |
| Данные и пайплайн | **DVC** | Git для данных: версии, кэш, `dvc.yaml`-стадии, `dvc.lock`, удалённые remotes |
| Трекинг, артефакты, реестр | **MLflow** | Run'ы, метрики, параметры, артефакты, datasets, Model Registry — всё в одном месте |
| Зависимости | **uv** | Мгновенный resolve и установка, `uv.lock` для воспроизводимости |
| Тесты | **pytest** | Стандарт де-факто, огромная экосистема плагинов |
| Команды | **just** | Читаемый `justfile` вместо Makefile |
| Валидация данных | **Pandera** | Схема DataFrame проверяется кодом, а не глазами |
| Оркестрация (по необходимости) | **Prefect** / **Airflow** | Только когда нужны расписания, ретраи и удалённое исполнение |
| DL | **PyTorch Lightning** | Trainer решает циклы обучения, логирование, чекпойнты, AMP, детерминизм |

Что сознательно **не** используется: самописные форматы метрик (`metrics.json`), свои `Protocol`-контракты,
собственный трекер, собственный guard целостности, самописный локальный tracking-сервер.
Всё это уже есть в готовых инструментах.

---

## Содержание

- [Гайд по ML-проекту](#гайд-по-ml-проекту)
  - [Содержание](#содержание)
  - [Структура репозитория](#структура-репозитория)
  - [Слои данных (Medallion Architecture)](#слои-данных-medallion-architecture)
  - [Как уйти от ноутбуков](#как-уйти-от-ноутбуков)
    - [Роль ноутбука](#роль-ноутбука)
    - [Рабочий цикл «ноутбук → модуль»](#рабочий-цикл-ноутбук--модуль)
    - [Технические практики](#технические-практики)
    - [Куда деваются выводы ноутбуков](#куда-деваются-выводы-ноутбуков)
  - [Зависимости: uv](#зависимости-uv)
  - [Команды: just](#команды-just)
  - [Конфигурация: Hydra](#конфигурация-hydra)
    - [Альтернатива: pydantic + YAML (без Hydra)](#альтернатива-pydantic--yaml-без-hydra)
  - [Данные и пайплайн: DVC](#данные-и-пайплайн-dvc)
  - [.gitignore: как это работает и частые ловушки](#gitignore-как-это-работает-и-частые-ловушки)
    - [Как Git применяет правила](#как-git-применяет-правила)
    - [Ключевые символы](#ключевые-символы)
    - [Примеры из типичного ML-проекта](#примеры-из-типичного-ml-проекта)
    - [Нюансы](#нюансы)
    - [Итог одной фразой](#итог-одной-фразой)
  - [Масштабирование под тип проекта: уровни зрелости](#масштабирование-под-тип-проекта-уровни-зрелости)
  - [MLflow: трекинг, артефакты, реестр](#mlflow-трекинг-артефакты-реестр)
    - [Локальный сервер](#локальный-сервер)
    - [Базовый цикл в коде](#базовый-цикл-в-коде)
    - [Autolog: минимум ручного кода](#autolog-минимум-ручного-кода)
    - [Датасеты и версии данных](#датасеты-и-версии-данных)
    - [Сравнение и поиск run'ов](#сравнение-и-поиск-runов)
    - [Model Registry](#model-registry)
    - [Загрузка модели в inference](#загрузка-модели-в-inference)
    - [Асинхронное логирование](#асинхронное-логирование)
    - [Офлайн-режим для тестов](#офлайн-режим-для-тестов)
    - [Сводка](#сводка)
  - [Модели: BaseModel, sklearn, Lightning](#модели-basemodel-sklearn-lightning)
    - [BaseModel: свой ABC, а не библиотека](#basemodel-свой-abc-а-не-библиотека)
    - [Реестр моделей](#реестр-моделей)
    - [Одна парадигма: sklearn или Lightning](#одна-парадигма-sklearn-или-lightning)
    - [Skorch: обычно не нужен](#skorch-обычно-не-нужен)
  - [Композиция компонентов: sklearn.Pipeline](#композиция-компонентов-sklearnpipeline)
  - [PyTorch Lightning](#pytorch-lightning)
  - [Большие данные и распределённое обучение](#большие-данные-и-распределённое-обучение)
  - [Воспроизводимость: seed](#воспроизводимость-seed)
  - [Валидация данных: Pandera](#валидация-данных-pandera)
  - [Тесты: pytest](#тесты-pytest)
  - [subprocess в обучении](#subprocess-в-обучении)
  - [Оркестрация: DVC, Prefect, Airflow](#оркестрация-dvc-prefect-airflow)
    - [Что ещё бывает в индустрии](#что-ещё-бывает-в-индустрии)
    - [Сводная таблица инструментов по задачам](#сводная-таблица-инструментов-по-задачам)
  - [Kaggle: соревнования и удалённый запуск](#kaggle-соревнования-и-удалённый-запуск)
    - [Схема работы](#схема-работы)
    - [Что уходит в Kaggle](#что-уходит-в-kaggle)
    - [Цикл команд](#цикл-команд)
    - [MLflow из Kaggle](#mlflow-из-kaggle)
    - [Что здесь НЕ нужно](#что-здесь-не-нужно)
    - [Запуск по требованию](#запуск-по-требованию)
  - [Serving: что это и когда нужно](#serving-что-это-и-когда-нужно)
  - [Жизненный цикл модели в компании](#жизненный-цикл-модели-в-компании)
    - [Champion и Challenger](#champion-и-challenger)
    - [Промоушен по средам](#промоушен-по-средам)
    - [Мониторинг](#мониторинг)
    - [Training-serving skew](#training-serving-skew)
    - [CI/CD для ML](#cicd-для-ml)
    - [Контейнеризация](#контейнеризация)
  - [CI и защита кода](#ci-и-защита-кода)
  - [Антипаттерны](#антипаттерны)
  - [Архитектура пайплайна: стадии-функции + DVC](#архитектура-пайплайна-стадии-функции--dvc)
    - [Почему не sklearn.Pipeline для современных задач](#почему-не-sklearnpipeline-для-современных-задач)
    - [Паттерн: стадии-функции + DVC](#паттерн-стадии-функции--dvc)
      - [1. clean](#1-clean)
      - [2. split](#2-split)
      - [3. candidates](#3-candidates)
      - [4. features](#4-features)
      - [5. ranker](#5-ranker)
      - [6. evaluate](#6-evaluate)
    - [Практика работы с большими данными (3+ ГБ parquet)](#практика-работы-с-большими-данными-3-гб-parquet)
    - [Что взять из гайда, что заменить](#что-взять-из-гайда-что-заменить)
  - [EDA: систематика и хранение](#eda-систематика-и-хранение)
    - [1. Разведка (одноразовая)](#1-разведка-одноразовая)
    - [2. Находки (то, что стоит сохранить)](#2-находки-то-что-стоит-сохранить)
    - [3. Автоматический профиль данных (мониторинг)](#3-автоматический-профиль-данных-мониторинг)
    - [Чек-лист вопросов для EDA (общий, не только recsys)](#чек-лист-вопросов-для-eda-общий-не-только-recsys)
    - [Как делать EDA на больших данных](#как-делать-eda-на-больших-данных)
  - [Хранение графиков: по типу, а не в кучу](#хранение-графиков-по-типу-а-не-в-кучу)
  - [В каком порядке осваивать инженерную часть](#в-каком-порядке-осваивать-инженерную-часть)
  - [Чеклист перед коммитом](#чеклист-перед-коммитом)
  - [Что читать по инженерной части](#что-читать-по-инженерной-части)
  - [Практическое задание](#практическое-задание)

---

## Структура репозитория

Основа: шаблон Cookiecutter Data Science, адаптированный под современные практики.

```text
my-ml-project/
├── pyproject.toml              # зависимости, настройки инструментов (uv / poetry)
├── uv.lock                     # зафиксированные версии
├── README.md
├── Makefile                    # короткие команды: make data, make train, make test
├── dvc.yaml                    # граф стадий пайплайна
├── dvc.lock                    # хэши входов/выходов
├── .pre-commit-config.yaml     # ruff, nbstripout, проверки
├── .github/workflows/ci.yaml   # тесты и линтеры на каждый push
│
├── conf/ / configs/            # ВСЯ конфигурация здесь, не в коде (Hydra использует conf/)
│   ├── data.yaml               # пути, источники, период
│   ├── features.yaml           # список признаков, параметры
│   ├── split.yaml              # схема разбиения, seed
│   └── train/
│       ├── lgbm.yaml
│       └── mlp.yaml
│
├── data/                       # в .gitignore, управляется DVC
│   ├── raw/                    # bronze — неизменяемые исходные данные
│   ├── interim/                # silver — очищенные, типизированные, провалидированные
│   ├── processed/              # gold — признаки + разбиение, готово для обучения
│   └── external/               # справочники, сторонние данные
│
├── src/my_project/             # ЯДРО: код как пакет
│   ├── __init__.py
│   ├── config.py               # загрузка и валидация конфигов (pydantic / Hydra)
│   ├── data/
│   │   ├── ingest.py           # получение raw
│   │   ├── clean.py            # очистка, типы
│   │   ├── schemas.py          # pandera/pydantic схемы
│   │   ├── split.py            # разбиение (group/time-aware)
│   │   └── datasets.py         # PyTorch Dataset / tf.data
│   ├── features/
│   │   ├── transformers.py     # кастомные sklearn-трансформеры
│   │   ├── build.py            # сборка ColumnTransformer/Pipeline
│   │   └── text.py, image.py   # по модальностям
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── inference/
│   │   └── predict.py          # использует ТЕ ЖЕ трансформации
│   └── utils/                  # логирование, seed, io
│
├── scripts/ или cli.py         # точки входа: python -m my_project.cli ...
│
├── notebooks/                  # только исследование
│   ├── 01_eda_orders.ipynb
│   └── 02_label_noise_check.ipynb
│
├── tests/
│   ├── test_clean.py
│   ├── test_features.py        # на маленьких синтетических фикстурах
│   ├── test_split_no_leakage.py
│   └── test_schemas.py
│
├── models/                     # артефакты (под DVC/MLflow)
├── reports/                    # графики, отчёты, dataset card
└── docs/
```

Правила:

1. `notebooks/` — только исследование. Код из ноутбука переносится в `src/` перед тем, как попасть в эксперимент.
2. `data/`, `models/`, `reports/` — не в git, версионируются DVC.
3. Код лежит в `src/<package>/`, импорты абсолютные: `from my_project.data.clean import clean_orders`.
4. `conf/` / `configs/` — единственный источник гиперпараметров. Литералов в коде быть не должно.
5. `tests/` повторяет структуру `src/`.
6. `docs/` — документация живет рядом с кодом. Храните здесь:
   - архитектуру и ADR (почему выбрали то или иное решение);
   - описание API (если есть сервис);
   - гайд для контрибьюторов;
   - запуск, деплой, частые проблемы.
   Формат — Markdown, лежит в `docs/`, коммитится в git. Не храните туда секреты и большие бинарники.

Ключевая идея: **`src/` это библиотека, `configs/` это параметры, `dvc.yaml` это порядок выполнения, `notebooks/` это одноразовая разведка.**

---

## Слои данных (Medallion Architecture)

Данные проходят через слои, и каждый следующий получается из предыдущего кодом. Распространённая схема в индустрии — «medallion»: bronze / silver / gold.

```text
raw (bronze)      → неизменяемые исходные данные, как получили. Никогда не правятся руками.
interim (silver)  → очищенные, типизированные, провалидированные, соединённые
processed (gold)  → признаки + разбиение, готово для обучения
```

Правила:

- **Raw иммутабелен.** Всё, что вы делаете, воспроизводимо из raw.
- Каждый слой можно удалить и пересобрать командой.
- Между слоями стоят **проверки** (схема, диапазоны, доли пропусков).
- В git лежат не данные, а **указатели** на них (DVC, lakeFS).

---

## Как уйти от ноутбуков

Ноутбуки не зло, но у них проблемы: скрытое состояние (порядок выполнения ячеек), плохие диффы в git, сложно тестировать и запускать по расписанию, соблазн копипасты.

### Роль ноутбука

- **Допустимо:** EDA, прототипирование, визуализация, разбор ошибок модели, отчёты.
- **Недопустимо:** единственное место, где живёт логика очистки или признаков, из которой берётся датасет для обучения.

### Рабочий цикл «ноутбук → модуль»

1. Исследуете в ноутбуке.
2. Когда нашли нужное преобразование, оформляете его **функцией в `src/`** с типами и докстрингом.
3. Пишете на неё тест.
4. В ноутбуке теперь `from my_project.data.clean import fix_dates` и вызываете её. Логика в одном месте.
5. Ноутбук остаётся «клиентом» вашей библиотеки.

### Технические практики

- `pip install -e .` (или `uv sync`), чтобы `import my_project` работал из ноутбука.
- В первой ячейке: `%load_ext autoreload` и `%autoreload 2`, чтобы правки в `src/` подхватывались.
- **nbstripout** в pre-commit: чистит выводы перед коммитом, диффы становятся читаемыми.
- **Jupytext**: хранит ноутбук как `.py`-файл (формат percent), удобно ревьюить и диффить.
- Ноутбуки нумеруются и не «чинятся» вечно: сделал, выводы записал, устарело, архивируй.
- Если нужно, ноутбук можно запускать как задачу через papermill, но для пайплайна лучше обычные скрипты.
- Альтернативы с реактивным состоянием: marimo (хранится как чистый `.py`).

### Куда деваются выводы ноутбуков

Ноутбуки попадают в репозиторий **без сохранённых выводов**. `nbstripout` вырезает из `.ipynb` только вывод ячеек (графики, таблицы, логи), а код и markdown остаются. Причины: выводы раздувают файл, диффы превращаются в мусор, в выводах может утечь чувствительное, выводы не воспроизводимы.

**Практический минимум для вашего проекта:**

1. `nbstripout` в pre-commit, ноутбуки в git без выводов.
2. Важные выводы EDA пишете текстом в `docs/eda_notes.md` или в README.
3. Графики, нужные повторно, оформляете функциями и стадией в `dvc.yaml`, результат в `reports/figures/`.
4. Метрики и графики моделей логируете в MLflow.
5. Для итогового разбора делаете `nbconvert --execute --to html` и кладёте в `reports/`.

**Возможное развитие (всё, что не в минимуме):**

- Одноразовая разведка: никуда — перезапустите ноутбук, если нужно.
- Выводы, ставшие правилами: уходят в код (`src/`) с тестами.
- Регулярные графики: функции в `src/reporting/plots.py` + стадия DVC.
- Метрики экспериментов: MLflow / W&B / TensorBoard (`mlflow.log_metric`, `mlflow.log_figure`).
- Отчёты о данных: HTML-артефакты (ydata-profiling, Great Expectations, Evidently) — в CI artifacts, MLflow или GitHub Pages.
- Ноутбуки как отчёты: рендерим через `jupyter nbconvert --to html --execute`, papermill (параметризованные запуски), Quarto (сайты/PDF) — результат в `reports/` или на Pages/Confluence.
- Ноутбуки с выводами в отдельном месте: `notebooks/reports/` (исключение через `.gitattributes`), Wiki/Notion, корпоративные платформы (Databricks, SageMaker). Для портфолио на GitHub можно отключить `nbstripout` для отдельных файлов.

---

## Зависимости: uv

```bash
uv init                                  # создать проект
uv python pin 3.12                       # зафиксировать версию Python
uv add torch torchvision pandas numpy scikit-learn lightgbm
uv add mlflow pandera
uv add --dev pytest pytest-cov pytest-mock ruff mypy
uv sync                                  # поставить ровно то, что в uv.lock
uv run pytest                            # запуск в окружении проекта
uv lock                                  # пересобрать lock-файл
```

`uv.lock` — это то, что делает окружение воспроизводимым. Он в git, как и `pyproject.toml`.

Группы зависимостей удобно разделять, чтобы тяжёлое не ставилось на CI для тестов:

```toml
[dependency-groups]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]
```

`uv sync --group dev` — поставить вместе с dev-инструментами, `uv sync --no-dev` — только рантайм.

---

## Команды: just

`justfile` читается как список намерений и заменяет Makefile без его синтаксиса.

```make
# justfile
set shell := ["bash", "-uc"]

default:
    @just --list

install:
    uv sync --group dev

train exp="baseline":
    uv run python -m my_project.train experiment={{exp}}

test:
    uv run pytest -q

lint:
    uv run ruff check .
    uv run ruff format --check .

typecheck:
    uv run mypy src

format:
    uv run ruff format .

# полный цикл: подготовить данные, обучить, отладить пайплайн
repro:
    uv run dvc repro

# MLflow локально
mlflow-server:
    uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

```bash
just                 # список рецептов
just train baseline # аргументы по умолчанию
just train exp=sweep
```

---

## Конфигурация: Hydra

Hydra — стандарт в ML-проектах. OmegaConf лежит внутри неё и используется для доступа к значениям.

`conf/config.yaml` — корневой конфиг, который собирает остальные:

```yaml
defaults:
  - data: tabular
  - model: lgbm
  - _self_

experiment_name: baseline
seed: 42

paths:
  raw_dir: ${oc.env:DATA_RAW_DIR,data/raw}
  processed_dir: ${oc.env:DATA_PROCESSED_DIR,data/processed}

hydra:
  run:
    dir: outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}
  job:
    chdir: false
```

Группы конфигов (в `conf/model/`, `conf/data/`, `conf/experiment/`) переопределяются по именам:

```yaml
# conf/model/lgbm.yaml
type: lgbm
params:
  n_estimators: 800
  learning_rate: 0.03
  num_leaves: 31
  subsample: 0.8
```

Запуск и оверайды:

```bash
uv run python -m my_project.train                      # дефолт
uv run python -m my_project.train model=mlp            # сменить группу
uv run python -m my_project.train model.params.lr=0.01  # точечный override
uv run python -m my_project.train +new_key=value        # добавить новое поле
```

`+` перед новым ключом обязателен: без `+` Hydra требует, чтобы ключ уже существовал в схеме.

Multirun (свипы) — одна строка:

```bash
uv run python -m my_project.train --multirun model.params.lr=0.01,0.03,0.1
```

Типизация: Hydra умеет валидировать YAML по структуре (dataclass или Pydantic-модель), так что опечатка в имени параметра падает сразу, а не через час обучения:

```python
from dataclasses import dataclass


@dataclass
class ModelConfig:
    type: str
    params: dict[str, float]


with hydra.initialize_config_dir(config_dir="conf", version_base=None):
    cfg = hydra.compose(config_name="config")
```

Секреты — только в окружении, никогда в YAML:

```yaml
tracking_uri: ${oc.env:MLFLOW_TRACKING_URI}
```

Хэш финального конфига (полезно класть в MLflow как параметр):

```python
import hashlib
from omegaconf import OmegaConf

resolved = OmegaConf.to_yaml(cfg, resolve=True)
config_hash = hashlib.sha256(resolved.encode()).hexdigest()
```

### Альтернатива: pydantic + YAML (без Hydra)

Если не нужен Hydra, можно использовать pydantic для валидации YAML-конфигов:

```yaml
# configs/split.yaml
strategy: group_time
group_col: user_id
time_col: event_ts
test_start: "2025-01-01"
val_months: 2
seed: 42
```

```python
# src/my_project/config.py
from pydantic import BaseModel
from pathlib import Path
import yaml


class SplitConfig(BaseModel):
    strategy: str
    group_col: str
    time_col: str
    test_start: str
    val_months: int
    seed: int


def load(path: Path, cls):
    return cls(**yaml.safe_load(path.read_text()))
```

Конфиг валидируется при старте, опечатка в ключе падает сразу, а не через час.

---

## Данные и пайплайн: DVC

DVC отвечает за две вещи: версионирование данных и пайплайн.

Версионирование:

```bash
dvc init
dvc remote add -d s3://my-bucket/dvc
dvc add data/raw/train.csv      # создаёт data/raw/train.csv.dvc
dvc push                        # отправить в remote
dvc pull                        # забрать на другой машине
```

Пайплайн — стадии в `dvc.yaml`:

```yaml
stages:
  prepare:
    cmd: uv run python -m my_project.data.prepare
    deps:
      - src/my_project/data/prepare.py
      - data/raw/train.csv
    outs:
      - data/processed/train.parquet
      - data/processed/valid.parquet

  train:
    cmd: uv run python -m my_project.train
    deps:
      - src/my_project/train.py
      - data/processed/train.parquet
    params:
      - conf/config.yaml
      - conf/model/lgbm.yaml
    outs:
      - models/baseline/model.pkl
```

```bash
dvc repro                 # пересобрать изменившееся
dvc repro --dry           # показать, что изменится
dvc status                # что устарело
dvc.lock                  # зафиксированные хэши выходов (в git)
```

`dvc.lock` в git, сами данные — нет. Это и есть «git для данных».

Секреты и кэш:

```gitignore
.dvc/cache/
data/**.parquet
models/
```

DVC-эксперименты (вместо ручных копий папок):

```bash
dvc exp run                 # запуск эксперимента
dvc exp list
dvc exp show                # чем отличаются от базового
dvc exp diff
```

---

## .gitignore: как это работает и частые ловушки

`.gitignore` — файл с **шаблонами (patterns)**, по которым Git решает, какие **неотслеживаемые** файлы игнорировать. Важно: **если файл уже в индексе (закоммичен), `.gitignore` на него не повлияет** — нужно `git rm --cached`.

### Как Git применяет правила

1. Путь файла берётся **относительно корня репозитория** (например, `data/raw/sales.parquet`).
2. Прогоняется через все шаблоны в `.gitignore` (и `.git/info/exclude`, и глобальный `core.excludesFile`).
3. **Последнее совпадение выигрывает** — порядок строк важен.
4. `#` — комментарий, пустые строки игнорируются.
5. `!шаблон` — исключение (вернуть файл под версионирование, даже если он попал под предыдущее правило).

### Ключевые символы

| Символ | Значение |
| --- | --- |
| `*` | любые символы, **кроме `/`** |
| `**` | любое число вложенных каталогов (включая ноль) |
| `?` | ровно один символ, кроме `/` |
| `[abc]` | один из символов |
| `/` в начале | шаблон **только от корня репозитория** |
| `/` в конце | соответствует **только каталогу** |
| `/` в середине | шаблон «привязан» к относительному пути от места `.gitignore` |
| нет `/` вообще | шаблон ищется **на любом уровне вложенности** |

### Примеры из типичного ML-проекта

```gitignore
# Python — виртуальные окружения и секреты
/.venv          # только в корне, файл или папка
/venv/          # только в корне, только папка
.env            # везде: ./.env, src/.env, config/prod/.env

# Data — данные (управляются DVC)
data/**/*.parquet   # любая глубина: data/x.parquet, data/raw/2024/x.parquet
data/**/*.csv       # аналогично для CSV

# Артефакты
models/             # папки models на любом уровне
reports/            # папки reports на любом уровне

# DVC
.dvc/               # служебная папка DVC
dvc.lock            # lock-файл (решение команды: не коммитим)

# MLflow
mlruns/             # локальное хранилище MLflow (артефакты, метрики, модели)
```

### Нюансы

**`**` vs `*`**

- `data/*/*.csv` — ровно один промежуточный каталог (`data/2024/jan.csv` ✅, `data/raw/2024/jan.csv` ❌).
- `data/**/*.csv` — любая глубина (`data/jan.csv`, `data/raw/2024/jan.csv` ✅).

**Если файл уже в индексе:**

```bash
git rm --cached data/old.parquet        # убрать конкретный файл
git rm -r --cached models/              # убрать папку рекурсивно
```

Иначе `.gitignore` его «не увидит».

**Проверить, почему файл игнорируется (или нет):**

```bash
git check-ignore -v data/raw/sales.parquet
```

Покажет строку `.gitignore`, сработавшую для этого пути.

**Порядок строк — ловушка с исключениями:**

```gitignore
data/**/*.csv
!data/important/keep.csv
```

Вторая строка вернёт файл. Но **исключение не сработает**, если родительский каталог целиком проигнорирован (например, стоит просто `data/` без звёздочек) — частая ошибка.

### Итог одной фразой

> `.gitignore` говорит Git'у: «эти шаблоны путей не отслеживай», причём `/` на краях и внутри, `*`/`**` и порядок строк определяют, **где** и **что именно** будет отфильтровано. В ML-проекте: секреты и окружения — по всей репе, данные и артефакты (parquet/csv/models/reports/mlruns) — по своим путям, служебные файлы DVC/MLflow — целиком.

---

## Масштабирование под тип проекта: уровни зрелости

Не нужно сразу строить всё. Уровни зрелости:

**Уровень 0. Один ноутбук.** Подходит для первой недели исследования. Не подходит ни для чего, что дальше.

**Уровень 1. Ноутбуки + `src/`.** Логика вынесена в функции, `requirements`/`pyproject`, git, nbstripout. *Минимум для любого серьёзного проекта.*

**Уровень 2. Воспроизводимый пайплайн.** Конфиги, CLI, DVC или Make, схемы данных, тесты, pre-commit, CI. *Целевой уровень для большинства проектов и портфолио.*

**Уровень 3. Продакшн.** Оркестратор (Airflow, Prefect, Dagster), feature store, реестр моделей, Docker, мониторинг дрейфа, алерты, data contracts, автопереобучение.

Для обучения и пет-проектов стремитесь к уровню 2.

---

## MLflow: трекинг, артефакты, реестр

MLflow — единое место, где живёт **результат эксперимента**. Свой формат метрик (`metrics.json`) не нужен:
MLflow уже умеет всё, что такой файл давал, и умеет больше.

### Локальный сервер

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
# UI: http://127.0.0.1:5000
```

Удалённый сервер (в Docker, с артефактами на S3):

```bash
mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri postgresql://user:pass@db:5432/mlflow \
  --artifacts-destination s3://my-bucket/mlflow-artifacts \
  --serve-artifacts
```

`--serve-artifacts` обязателен, если клиент загружает артефакты: без него метрики запишутся, а файлы — нет.

Доступ из кода — через переменные окружения:

```bash
MLFLOW_TRACKING_URI=https://mlflow.example.com
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...
# или
MLFLOW_TRACKING_TOKEN=...
```

Секреты — в `.env` (в git не попадает) или в секретах CI/Kaggle.

### Базовый цикл в коде

```python
import mlflow
from omegaconf import OmegaConf

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("my-project")

with mlflow.start_run(
    run_name=cfg.experiment_name, tags={"seed": str(cfg.seed)}
) as run:
    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
    mlflow.log_param("config_hash", config_hash)
    mlflow.log_dict({"cfg": OmegaConf.to_container(cfg, resolve=True)}, "config.json")

    for epoch in range(cfg.epochs):
        train_loss = ...
        val_auc = ...
        mlflow.log_metrics({"train_loss": train_loss, "val_auc": val_auc}, step=epoch)

    mlflow.log_artifact("models/baseline/model.pkl", artifact_path="model")
    mlflow.log_artifact("reports/curves.png", artifact_path="reports")
```

`mlflow.log_metrics(..., step=epoch)` — метрика привязывается к шагу, поэтому в UI видно кривые обучения.

### Autolog: минимум ручного кода

Для sklearn/XGBoost/LightGBM/PyTorch MLflow умеет логировать сам:

```python
mlflow.autolog(sklearn=True, pytorch=True, log_models=False)
```

Autolog подхватывает параметры модели, метрики обучения и сохраняет модель. Ручной `log_params` после autolog избыточен.

### Датасеты и версии данных

```python
import mlflow.data

dataset = mlflow.data.from_pandas(
    train_df, source="data/processed/train.parquet", name="train"
)
mlflow.log_input(dataset, context="training")
```

Вместе с `dvc`-хэшем в параметрах это даёт полную прослеживаемость: какие данные, какая версия, какой код.

```python
mlflow.log_param("dvc_commit", dvc_hash)  # hash из .dvc-файла
mlflow.log_param("git_commit", git_sha)
```

### Сравнение и поиск run'ов

```python
import pandas as pd

df = mlflow.search_runs(
    experiment_names=["my-project"],
    filter_string="metrics.val_auc > 0.85 and attributes.seed = 42",
    order_by=["metrics.val_auc DESC"],
)
print(df[["run_id", "metrics.val_auc", "params.lr"]])
```

Всё то же самое умеет UI: сравнение, графики, удаление, теги.

### Model Registry

```python
mlflow.sklearn.log_model(model, "model", registered_model_name="lgbm-baseline")
```

```python
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias("lgbm-baseline", "champion", "3")
```

Alias (`champion`, `prod`) лучше, чем номер версии: переключение модели в проде не требует правок кода.

### Загрузка модели в inference

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/lgbm-baseline@champion")
preds = model.predict(pandas_df)
```

### Асинхронное логирование

MLflow 3 умеет писать метрики в фоне — полезно, когда логирование не должно тормозить эпоху:

```bash
MLFLOW_ENABLE_ASYNC_LOGGING=true
```

```python
mlflow.config.enable_async_logging(True)
```

Минус: при аварийном завершении процесса последние батчи могут не уйти. Для коротких экспериментов оставляйте синхронный режим.

### Офлайн-режим для тестов

Тесты не должны ходить в сеть. Локальный file-store решает это:

```python
mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
```

### Сводка

| Задача | API |
| --- | --- |
| Создать run | `mlflow.start_run(run_name=..., tags=...)` |
| Параметры | `mlflow.log_params`, `mlflow.log_param` |
| Метрики | `mlflow.log_metrics(metrics, step=...)` |
| Артефакты | `mlflow.log_artifact(path, artifact_path=...)` |
| Конфиг в артефакт | `mlflow.log_dict` |
| Датасет | `mlflow.data.from_pandas` + `mlflow.log_input` |
| Модель | `mlflow.sklearn.log_model` / `mlflow.pytorch.log_model` |
| Реестр | `mlflow.register_model`, `client.set_registered_model_alias` |
| Поиск | `mlflow.search_runs` |
| Скачать артефакт | `mlflow.artifacts.download` |

Альтернативы: **Weights & Biases** (удобнее UI, особенно для DL), **Aim**, **Neptune**. MLflow остаётся стандартом де-факто, потому что закрывает ещё и реестр моделей, и `pyfunc` для инференса.

---

## Модели: BaseModel, sklearn, Lightning

### BaseModel: свой ABC, а не библиотека

`BaseModel` в шаблонах ML-проектов — это **не пакет**, а небольшой абстрактный класс, который пишут в проекте сами, чтобы код обучения и оценки не зависел от конкретной библиотеки:

```python
# src/my_project/models/base.py
from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Общий интерфейс моделей: fit / predict / predict_proba."""

    @abstractmethod
    def fit(self, x: Any, y: Any) -> "BaseModel": ...

    @abstractmethod
    def predict(self, x: Any) -> Any: ...

    def predict_proba(self, x: Any) -> Any:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """Параметры для логирования и поиска по сетке."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}
```

Что это даёт:

- `evaluate.py` работает с любой моделью: sklearn, LightGBM, Lightning — код одинаковый;
- можно подменить модель в тесте на заглушку;
- параметры модели единообразно попадают в MLflow.

Близкие аналоги из библиотек:

- `sklearn.base.BaseEstimator` (+ `ClassifierMixin`) — базовый класс sklearn-моделей, даёт `get_params`/`set_params`; для sklearn-моделей свой `BaseModel` не нужен, достаточно наследоваться от `BaseEstimator`;
- `pydantic.BaseModel` — совершенно другое: валидация конфигов и данных, а не интерфейс модели. Не путать;
- `torch.nn.Module` — базовый класс нейросетей в PyTorch;
- `lightning.LightningModule` — аналогичный интерфейс для PyTorch Lightning (см. ниже).

Правило: **свой `BaseModel` имеет смысл, только если в проекте больше одной реализации**. Для одного LightGBM-модели — лишний слой.

### Реестр моделей

Не нужен, если моделей мало. Когда появляется десяток, помогает маленький словарь:

```python
MODELS = {"lgbm": LgbmModel, "mlp": MlpModel}


def build_model(cfg) -> BaseModel:
    return MODELS[cfg.model.type](**cfg.model.params)
```

Hydra умеет вызывать классы напрямую — `_target_`:

```yaml
model:
  _target_: my_project.models.lgbm.LgbmModel
  params:
    n_estimators: 800
```

```python
model = hydra.utils.instantiate(cfg.model)
```

---

### Одна парадигма: sklearn или Lightning

Выберите одну и не смешивайте в одном пайплайне:

| Задача | Парадигма |
| --- | --- |
| Табличные данные, классический ML (sklearn, LightGBM, XGBoost) | sklearn-пайплайн |
| CV / NLP / LLM, кастомные архитектуры и циклы обучения | PyTorch Lightning |

Граница проходит не по «классика против DL», а по контролю над обучением: цикл стандартный — берите sklearn-совместимый API, цикл нестандартный — Lightning.

`nn.Module` и `BaseEstimator` не конфликтуют, у них разные роли:

- `torch.nn.Module` — «сердце» PyTorch: регистрация параметров и слоёв, `.to(device)`, `.train()/.eval()`, `state_dict`;
- `sklearn.base.BaseEstimator` — «паспорт» интеграции в sklearn: `get_params`/`set_params`, работа в `Pipeline`, `GridSearchCV`, `cross_val_score`, логирование в MLflow.

### Skorch: обычно не нужен

`skorch` — sklearn-совместимая обёртка над `nn.Module` (`NeuralNetClassifier` и т.п.): даёт `fit`/`predict` обычному sklearn-пайплайну.

Но это лишний слой: он дублирует то, что уже делает Lightning, и ограничивает контроль над обучением. Нужен только в одном случае — нейросеть как один из шагов sklearn-пайплайна с `GridSearchCV`. В остальных случаях — прямой Lightning без посредников.

## Композиция компонентов: sklearn.Pipeline

Правило «скрипты — оркестраторы, библиотека — логика»: переиспользуемая логика живёт в `src/` в виде компонентов с общим API, а `scripts/train.py` — тонкая точка входа, которая читает конфиг, собирает компоненты и вызывает их.

Единый API компонента (это и есть «не 100 функций»):

| Метод | Смысл |
| --- | --- |
| `fit(X, y)` | обучить компонент, вернуть `self` |
| `transform(X)` | применить выученное к новым данным |
| `fit_transform(X, y)` | удобная комбинация (даётся `TransformerMixin`) |
| `predict(X)` | предсказания |
| `get_params()` / `set_params()` | параметры для логирования и перебора (даёт `BaseEstimator`) |

```python
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_na_value: float = 0.0):
        self.fill_na_value = fill_na_value

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.columns_:
            if X[column].isna().any():
                X[column] = X[column].fillna(self.fill_na_value)
        return X
```

Сборка в `Pipeline`:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    [
        ("features", FeatureEngineer(fill_na_value=0.0)),
        ("model", LgbmModel(n_estimators=800)),
    ]
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

`Pipeline` вызывает `fit_transform` на всех шагах, кроме последнего, и `transform` при предсказании. Это автоматически исключает data leakage: тестовые данные не попадают в обучение даже руками.

Что сразу начинает работать (никакого «клея»):

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

search = GridSearchCV(
    pipeline,
    param_grid={
        "features__fill_na_value": [0.0, -1.0],
        "model__n_estimators": [400, 800],
    },
    scoring="roc_auc",
    cv=5,
)
search.fit(X_train, y_train)
```

- `GridSearchCV` / `Optuna` — перебор гиперпараметров, включая параметры шагов (`шаг__параметр`);
- `cross_val_score(pipeline, X, y, cv=5)` — честная CV без утечек;
- `mlflow.sklearn.log_model(pipeline, "model")` — пайплайн сохраняется как один артефакт со всеми шагами;
- DVC-стадия — один вызов скрипта, кэширование по зависимостям.

Тонкий скрипт-оркестратор:

```python
# scripts/train.py
import mlflow
from omegaconf import OmegaConf

from my_project.data import load_split
from my_project.features import FeatureEngineer
from my_project.models import build_model


def main() -> None:
    cfg = OmegaConf.load("conf/config.yaml")
    X_train, X_valid, y_train, y_valid = load_split(cfg)

    pipeline = Pipeline(
        [
            ("features", FeatureEngineer(**cfg.features)),
            ("model", build_model(cfg.model)),
        ]
    )

    with mlflow.start_run(run_name=cfg.experiment_name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        pipeline.fit(X_train, y_train)
        mlflow.log_metrics(
            {"valid_auc": compute_auc(y_valid, pipeline.predict_proba(X_valid)[:, 1])}
        )
        mlflow.sklearn.log_model(pipeline, "model")
```

Меняете модель — правите одну строку в конфиге. Меняете признаки — правите один класс. Скрипт не трогаете.

---

## PyTorch Lightning

`LightningModule` — это «модель + как её обучать»: методы `training_step`, `validation_step`, `configure_optimizers`.

```python
import lightning as L
import torch
from torch import nn


class MlpClassifier(L.LightningModule):
    def __init__(self, in_dim: int, hidden: int = 128, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()  # попадёт в чекпойнт и в MLflow
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1)
        )
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        acc = ((self(x) > 0).float() == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
```

Trainer берёт на себя всё, что обычно пишут руками:

```python
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from mlflow.lightning import MLflowLogger

trainer = L.Trainer(
    accelerator="gpu",  # "cpu" / "gpu" / "mps"
    devices=1,
    precision="16-mixed",  # AMP
    max_epochs=50,
    deterministic=True,  # воспроизводимость
    logger=MLflowLogger(
        experiment_name="my-project",
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        log_model=True,
    ),
    callbacks=[
        ModelCheckpoint(monitor="val_loss", save_top_k=1, filename="best"),
        EarlyStopping(monitor="val_loss", patience=5),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    log_every_n_steps=50,
    limit_train_batches=0.1,  # быстрый smoke-run
    num_sanity_val_steps=0,
)
trainer.fit(model, datamodule=dm)
```

Что даёт Lightning по сравнению с «циклом на 30 строк»:

| Фича | Что делает |
| --- | --- |
| `Trainer` | циклы train/val/test, распределение по GPU/TPU, AMP, gradient clipping, accumulate, gradient accumulation |
| `LightningModule` | логирование метрик через `self.log`, чекпойнты с гиперпараметрами |
| `LightningDataModule` | данные и трансформации в одном месте, версии датасета |
| Callbacks | early stopping, checkpointing, LR monitor, SWA, EMA, профилирование |
| Логгеры | CSV, TensorBoard, W&B, **MLflow** из коробки |
| `seed_everything` | единый seed для всех библиотек и воркеров |
| `Trainer(limit_train_batches=...)` | smoke-тест пайплайна за секунды |
| `torch.compile`, `precision`, `detect_anomaly` | оптимизации и отладка без ручного кода |

Что использовать **обязательно**, если берёте Lightning:

1. `pl.seed_everything(seed, workers=True)` — иначе воспроизводимости не будет.
2. `deterministic=True` в `Trainer` (и `torch.use_deterministic_algorithms`).
3. `self.log(...)` вместо `print` и ручного сбора метрик.
4. `ModelCheckpoint` — сохранять лучшую модель, а не последнюю.
5. `save_hyperparameters()` — иначе чекпойнт не восстановить.
6. `LightningDataModule` — чтобы переиспользовать одни и те же данные в train и в eval.

Полезно: `torch.set_float32_matmul_precision("high")` (быстрее на Ampere+), `torch.compile` (обёртка модели), `profiler="simple"` для поиска узких мест, `sync_batchnorm=True` при DDP.

---

## Большие данные и распределённое обучение

Уровни масштабирования — выбирайте по необходимости, а не «на всякий случай».

**1. Батчи внутри обучения.** `DataLoader` отдаёт батчи, а не весь датасет:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=8,  # распараллеливает загрузку и препроцессинг
    prefetch_factor=2,  # сколько батчей готовить заранее
    pin_memory=True,  # ускоряет перенос на GPU
    persistent_workers=True,  # не пересоздавать воркеров на каждой эпохе
    generator=generator,  # фиксированный seed (см. раздел про seed)
    worker_init_fn=worker_init_fn,
)
```

`num_workers > 0` означает, что данные копируются в память воркеров — на больших датасетах это либо RAM, либо `persistent_workers` + ленивые датасеты.

**2. Out-of-core: обучение на данных, которые не влезают в RAM.** `partial_fit` дообучает модель чанк за чанком:

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
for chunk in read_chunks("data/raw/huge.parquet", size=100_000):
    model.partial_fit(chunk[features], chunk[target], classes=[0, 1])
```

**3. Распределённое обучение.** Одна GPU не хватает:

```python
# Lightning: несколько GPU на одной машине
trainer = L.Trainer(strategy="ddp", devices=4, accelerator="gpu")

# или вручную: torchrun --nproc_per_node=4 train.py
```

- **DDP** — реплицирует модель на несколько GPU, синхронизирует градиенты (подходит, когда модель влезает в память одной GPU);
- **FSDP / ZeRO** — шардирует веса и оптимизатор, когда модель не влезает;
- несколько машин — DDP поверх сети, обычно через инфраструктуру K8s/кластера.

**4. Модель не влезает никуда.** Тогда выносите обучение в отдельную среду с GPU: Kaggle (см. раздел ниже), удалённая машина по SSH, K8s Job, Spot-инстанс. Локально остаётся только код и конфиг.

**5. Инференс отдельно.** Обучение — batch-задача, инференс — сервис. Модель загружается из реестра (MLflow), а не «поднимается заново».

---

## Воспроизводимость: seed

Один seed на весь проект, а не на каждую библиотеку отдельно:

```python
# src/my_project/utils/seed.py
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Фиксирует seed для random, numpy, torch и их воркеров."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed
```

```python
# Lightning
import lightning as L

L.seed_everything(seed, workers=True)
```

Что ещё нужно для полной воспроизводимости:

```python
# генератор для DataLoader, иначе shuffle будет разным
generator = torch.Generator().manual_seed(seed)
loader = DataLoader(dataset, shuffle=True, generator=generator, num_workers=4)


# фиксированный порядок Workers
def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

`PYTHONHASHSEED` влияет на порядок ключей в `set`/`dict` — из-за него «одинаковые» результаты могут расходиться между запусками. Задавать его нужно **до** старта интерпретатора, то есть в окружении, а не в коде:

```bash
PYTHONHASHSEED=0 uv run python -m my_project.train
```

Ограничения, о которых стоит знать: `torch.use_deterministic_algorithms(True)` может бросить ошибку на операциях без детерминированной реализации (часть GPU-операций), а cuDNN-алгоритмы для свёрток всё равно могут давать мелкие расхождения — отсюда `warn_only=True` и `benchmark=False`.

Детерминизм GPU ≠ детерминизм порядка данных: при `num_workers > 0` порядок батчей зависиет от worker'ов, поэтому seed для воркеров обязателен.

---

## Валидация данных: Pandera

Pandera — библиотека для проверки схемы pandas/polars DataFrame кодом. Смысл: «действительные» данные проверяются до обучения, а не «на глаз в ноутбуке».

```python
# src/my_project/data/schema.py
import pandera.pandas as pa
from pandera.typing import Series

train_schema = pa.DataFrameSchema(
    {
        "id": Series(int, nullable=False),
        "target": Series(int, nullable=False, checks=pa.Check.isin([0, 1])),
        "age": Series(float, nullable=True, checks=pa.Check.in_range(0, 120)),
        "income": Series(float, nullable=True, checks=pa.Check.gt(0)),
    },
    checks=[
        pa.Check(lambda df: df["id"].is_unique, error="id должен быть уникальным"),
    ],
    coerce=True,
)
```

Применение:

```python
train_schema.validate(train_df, lazy=True)  # lazy=True — собирает все ошибки сразу
```

или как декоратор функции:

```python
@pa.check_output(train_schema)
def prepare(raw: pd.DataFrame) -> pd.DataFrame: ...
```

Или как «типизированный» слой, чтобы в остальном коде типы были известны:

```python
from pandera.typing import DataFrame, Series


def feature_engineering(
    df: DataFrame[pa.DataFrameSchema],
) -> DataFrame[pa.DataFrameSchema]: ...
```

Где применять: на границе стадий DVC (после `prepare`, перед `train`) — так «битые» данные ломают пайплайн сразу и с понятным сообщением, а не через три эпохи обучения.

Ошибки: `pandera.errors.SchemaErrors` — можно ловить и превращать в понятное сообщение.

Альтернативы: **Great Expectations** (наследие для датасета как «ожидания», дороже в настройке) и **TFDV** (валидация в TensorFlow-пайплайнах). Для одного проекта Pandera — самый короткий путь.

---

## Тесты: pytest

```bash
uv add --dev pytest pytest-cov pytest-mock
```

```toml
# pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = ["slow: долгие тесты"]
```

Что тестировать в ML-проекте (по убыванию ценности):

1. **Схема данных** — что Pandera ловит битые данные.
2. **Метрики** — что `compute_metrics` даёт ожидаемые значения на маленьком фикстуре.
3. **Сплиты** — нет утечки: train/valid не пересекаются, порядок детерминирован.
4. **Модель** — `fit`/`predict` работают и дают правильную форму.
5. **Smoke-тест обучения** — 10 батчей, 1 эпоха, `limit_train_batches=0.1`; проверяет, что пайплайн не развалился.

```python
# tests/conftest.py
import pytest


@pytest.fixture
def tiny_dataset():
    """Маленький DataFrame, на котором быстро прогоняются тесты."""
    import pandas as pd

    return pd.DataFrame({"id": range(100), "x": range(100), "target": [0, 1] * 50})


@pytest.fixture
def tmp_mlflow(tmp_path, monkeypatch):
    """Локальный file-store, чтобы тесты не ходили в сеть."""
    import mlflow

    uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(uri)
    return uri
```

```python
# tests/models/test_lgbm.py
import pytest

from my_project.models.lgbm import LgbmModel


def test_fit_predict(tiny_dataset):
    model = LgbmModel(n_estimators=10)
    model.fit(tiny_dataset[["x"]], tiny_dataset["target"])

    predictions = model.predict(tiny_dataset[["x"]])

    assert len(predictions) == len(tiny_dataset)
    assert set(predictions.unique()) <= {0, 1}
```

Правила:

- тесты не ходят в сеть и не пишут в реальный MLflow — всегда `tmp_mlflow`;
- GPU-тесты помечать `@pytest.mark.slow` и не запускать в дефолтном CI;
- `pytest --cov=src --cov-report=term-missing` для оценки покрытия, но не гнаться за 100%;
- фикстуры — в `conftest.py`, параметризация — через `@pytest.mark.parametrize`.

---

## subprocess в обучении

Иногда обучение запускают как отдельный процесс. Это не обязательная практика, но у неё есть конкретные применения.

Зачем:

1. **Изоляция.** Ошибка в коде не роняет родительский процесс; GPU-память освобождается при выходе из процесса.
2. **Таймауты и отмена.** Процесс можно убить, если он завис.
3. **Логи в файл.** Весь stdout/stderr обучения пишется в файл без перенаправлений.
4. **Единая точка входа.** `python -m my_project.train` можно звать из anywhere — ноутбука, DVC-стадии, CI.
5. **Проверка «код вообще запускается».** Дешёвый smoke-тест: запустить процесс с таймаутом и проверить код возврата.

Пример в `train.py` (или в обёртке над ним):

```python
import os
import subprocess
import sys
from pathlib import Path


def run_training(command: list[str], log_path: str, timeout: int | None = None) -> int:
    """Запустить обучение в отдельном процессе и записать лог."""
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    return completed.returncode
```

Правила безопасности:

- **не использовать `shell=True`** — иначе строка команды превращается в shell-инъекцию;
- передавать аргументы списком: `["python", "-m", "my_project.train", "model=mlp"]`;
- `check=False` + ручная проверка `returncode`, если нужно «мягко» обработать падение;
- `timeout=` для защиты от зависания;
- `cwd=` и `env=` — чтобы процесс видел то же окружение, что и вызывающий.

Ограничение: если обучение идёт через `subprocess`, то метрики в MLflow пишутся внутри процесса — родительский процесс не знает результат, только код возврата. Поэтому в MLflow кладите всё, что нужно для анализа: параметры, метрики, артефакты.

Когда subprocess **не нужен**: если обучение вызывается как обычная функция из DVC-стадии, `dvc repro` уже даёт отдельный процесс, изоляцию, логи и код возврата.

---

## Оркестрация: DVC, Prefect, Airflow

| Инструмент | Когда нужен | Чего не умеет из коробки |
| --- | --- | --- |
| **DVC** | локальный пайплайн: prepare → train → evaluate; кэш и версии данных | расписаний, распределённого исполнения |
| **Prefect** | расписание, ретраи, запуск на удалённой машине, наблюдаемость | — |
| **Airflow** | если в компании уже есть Airflow как стандарт | — |

Рекомендация для pet-проекта: **DVC + just**. Prefect подключайте, когда появится потребность в «запустить ночью по расписанию и получить ссылку на результат». Airflow — когда он уже есть в компании, а не «на всякий случай».

Prefect (примерно):

```python
from prefect import flow, task


@task(retries=2, retry_delay_seconds=60)
def train(experiment: str) -> str:
    return experiment


@flow(log_prints=True)
def pipeline(experiment: str = "baseline") -> None:
    run_id = train(experiment).result()
    print(f"MLflow run: {run_id}")
```

```bash
prefect deploy -m pipeline.py
```

Airflow даёт то же самое, но в виде DAG-файлов, планировщика, веб-UI и общей инфраструктурой компании (XCom, S3, Hive, SLA). Для одного разработчика он тяжелее: нужен свой сервер Airflow.

Если вычисления должны идти на GPU-машине, а не на ноутбуке, оркестрация — не «ещё один слой», а необходимость: Flyte, Prefect, Airflow, Kubeflow, или простой `ssh`/`rsync`-скрипт.

---

### Что ещё бывает в индустрии

Полезно знать, что существует, чтобы понимать, чего у вас сознательно нет:

| Инструмент | Что даёт | Когда смотреть |
| --- | --- | --- |
| **Dagster** | ассеты вместо задач, автоматический lineage | когда данные — главный объект работы |
| **Metaflow** | «человеко-ориентированный» фреймворк для ML, заточен под data science | когда нужен удобный UI и запуск локально/в облаке |
| **Kubeflow Pipelines** | K8s-нативный оркестратор с GPU-осведомлённостью | когда инфраструктура уже в Kubernetes |
| **Great Expectations** | датасет как набор «ожиданий» | когда Pandera не хватает (сложные зависимости между колонками) |
| **Feast / Tecton** | feature store: фичи считаются один раз и переиспользуются | когда появилась training-serving skew и несколько команд |
| **Weights & Biases** | трекинг с удобным UI, особенно для DL | если MLflow UI не устраивает |
| **Optuna** | подбор гиперпараметров, лучше GridSearchCV по времени | когда перебор становится дорогим |
| **Kedro** | готовый каркас пайплайнов, каталог данных | когда нужен структурированный старт без боилерплейта |
| **Snakemake** | декларативные пайплайны с wildcard'ами | для биоинформатики и сложных DAG'ов |
| **lakeFS / Delta Lake** | Git-подобное версионирование данных поверх S3 | когда DVC не хватает (branch/merge данных) |
| **marimo** | реактивные ноутбуки, хранятся как `.py` | альтернатива Jupyter с воспроизводимым состоянием |
| **papermill / jupytext** | параметризация и версияция ноутбуков | когда ноутбуки нужно запускать в пайплайне |

Правило: не добавлять инструмент, пока не упёрлись в его отсутствие.

---

### Сводная таблица инструментов по задачам

Если выбрать минимальный современный стек: **uv + pydantic + pandera + sklearn Pipeline + DVC + pytest + ruff + pre-commit + MLflow**.

| Задача | Варианты |
| --- | --- |
| Окружение и зависимости | **uv**, poetry, conda; Docker для полной воспроизводимости |
| Структура проекта | Cookiecutter Data Science, kedro (даёт готовый каркас пайплайнов) |
| Конфиги | pydantic-settings, **Hydra** / OmegaConf, YAML |
| Версионирование данных | **DVC**, lakeFS, Delta Lake |
| Граф стадий | DVC pipelines, Make, Snakemake, Kedro |
| Оркестрация | Prefect (проще), Dagster (удобен для данных, ассеты), Airflow (стандарт индустрии) |
| Валидация | pandera, Great Expectations, pydantic |
| Трекинг экспериментов | MLflow, Weights & Biases |
| Качество кода | ruff, mypy, pre-commit, pytest |
| CI | GitHub Actions / GitLab CI |
| Feature store | Feast (для продакшн-уровня) |
| Ноутбуки | nbstripout, jupytext, marimo, papermill |

---

## Kaggle: соревнования и удалённый запуск

Отдельный частный случай: у Kaggle своя модель исполнения, здесь не нужен ни Airflow, ни Docker, ни локальный MLflow-сервер.

### Схема работы

```text
локально                          Kaggle
────────                          ──────
папка Kernel
├── kernel-metadata.json
├── train.py
├── src/  conf/          ──push──▶  /kaggle/input   (данные, только чтение)
└── requirements.txt               /kaggle/working  (результаты)
        ◀──output───  kaggle kernels output
```

### Что уходит в Kaggle

Код и конфиги — да, данные — нет. Данные подключаются через `competition_sources` и `dataset_sources` и доступны в `/kaggle/input` только на чтение.

```json
{
  "id": "my-user/my-experiment",
  "title": "my-experiment",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "competition_sources": ["my-competition"],
  "dataset_sources": []
}
```

### Цикл команд

```bash
kaggle kernels push -p ./kernel            # отправить и запустить
kaggle kernels status my-user/my-experiment # статус последнего запуска
kaggle kernels logs my-user/my-experiment   # логи
kaggle kernels output my-user/my-experiment -p ./outputs -o   # забрать артефакты
```

Артефакты (`submission.csv`, `oof.npy`, графики) скачиваются в `outputs/` — это и есть результат эксперимента.

### MLflow из Kaggle

На Kaggle нельзя поднять свой tracking-сервер (нет фоновых процессов) — нужен внешний. Настройки приходят через **Kaggle Secrets** (`Add-ons → Secrets` в UI), в коде читаются через `kaggle_secrets`:

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["MLFLOW_TRACKING_URI"] = secrets.get_secret("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_TOKEN"] = secrets.get_secret("MLFLOW_TRACKING_TOKEN")
```

Готовый бесплатный удалённый MLflow — **DagsHub**: `https://dagshub.com/<user>/<repo>.mlflow`.

Код в Kaggle должен совпадать с кодом в git: перед отправкой фиксируйте commit SHA и логируйте его в MLflow как параметр — иначе эксперимент невоспроизводим.

### Что здесь НЕ нужно

| Не нужно | Почему |
| --- | --- |
| Airflow и его Kaggle-провайдер | Airflow требует постоянно работающего инстанса; для ручных запусков это лишний слой |
| MLflow-сервер внутри Kaggle | нет фоновых процессов и внешнего доступа |
| Docker | Kaggle запускает скрипт из папки, образ не нужен |
| Свой DVC remote | версионирование экспериментов Kaggle ведётся в MLflow |

### Запуск по требованию

Три способа попросить «запустить эксперимент»:

1. **Локально из CLI или из агента** — скрипт вызывает `kaggle kernels push` и потом опрашивает `status`. Это самый прямой путь.
2. **GitHub Actions** — workflow с `workflow_dispatch` (запуск по кнопке) или по `push` в ветку:

```yaml
name: train
on:
  workflow_dispatch:
  push:
    branches: [experiments]

jobs:
  kaggle:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install kaggle
      - run: kaggle kernels push -p kernel
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
```

1. **Отдельный скрипт-очередь**, если экспериментов много: он держит список, отправляет, опрашивает статус и забирает результаты.

---

## Serving: что это и когда нужно

Serving — это «отдать обученную модель наружу»: HTTP-сервис, который принимает признаки и возвращает предсказания.

Когда он нужен:

- модель обслуживается в проде и вызывается из приложения;
- нужна предсказательная API с авторизацией, лимитами и мониторингом;
- модель переобучается, и старые версии должны жить параллельно.

Когда не нужен:

- вы только учитесь и считаете метрики;
- результаты идут в batch-выгрузку (CSV, parquet);
- вы делаете Kaggle-подобные соревнования — там нужен `submission.csv`, а не API.

Варианты (от простого к сложному):

| Вариант | Что даёт | Когда брать |
| --- | --- | --- |
| `mlflow.pyfunc` + FastAPI | 30 строк: загрузили модель, обернули в endpoint | умеренно |
| **FastAPI** вручную | полный контроль над форматом запроса/ответа | умеренно |
| **BentoML** | сервинг из коробки: версии, батчи, наблюдаемость | средний |
| **MLServer** (KServe) | стандартный протокол инференса, K8s | крупный |
| **Triton** | несколько моделей/фреймворков, низкая задержка | крупный |

Минимальный FastAPI-вариант:

```python
from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/lgbm-baseline@champion")


@app.post("/predict")
def predict(payload: dict) -> dict:
    import pandas as pd

    frame = pd.DataFrame(payload["rows"])
    return {"predictions": model.predict(frame).tolist()}
```

Прод-версия добавляет: версионирование (alias), кэш, health-check, таймауты, метрики запросов, canary-деплой. Это отдельная тема, и в pet-проекте до неё лучше не доходить раньше времени.

---

## Жизненный цикл модели в компании

Это то, что появляется, когда за проектом стоит не только вы.

### Champion и Challenger

Новая модель не вытесняет текущую «просто потому что новее». Обе обучаются, обе оцениваются на одной и той же отложенной выборке, и решение принимается по метрикам с доверительными интервалами:

```python
# сравнение двух run'ов из MLflow
import mlflow


def compare(run_a: str, run_b: str) -> pd.DataFrame:
    runs = mlflow.search_runs(
        experiment_names=["my-project"],
        filter_string=f"run_id in ('{run_a}', '{run_b}')",
    )
    return runs[["run_id", "metrics.valid_auc"]]
```

Если разница меньше шума — оставляем champion.

### Промоушен по средам

```text
development → staging → production
```

В каждой среде свой конфиг (адреса данных, лимиты, флаги) и свой уровень мониторинга. Модель в production — это alias в реестре (`champion`), а не «код с закомментированными путями».

### Мониторинг

- **Дрейф данных** (PSI): сравниваем распределения входных признаков с обучающими. Простой порог: PSI > 0.2 — повод разобраться;
- **Деградация метрик**: качество на «свежих» данных, а не на тестовой выборке;
- **Триггер переобучения**: по расписанию или по событию (появились новые данные, метрики упали).

### Training-serving skew

Главный источник расхождений между обучением и продакшеном — разная логика обработки данных. Решения:

- один и тот же код трансформаций в train и serve (самый простой и достаточный вариант);
- feature store (Feast/Tecton), когда фичи считаются один раз и переиспользуются;
- инференс-пайплайн, собранный из тех же компонентов, что и обучение (см. раздел про `sklearn.Pipeline`).

### CI/CD для ML

Кроме тестов кода, в CI полезно проверять:

- **данные**: схема (Pandera) на маленьком фикстуре;
- **модель**: обучение за 10 секунд на маленьком датасете не падает и даёт метрику в ожидаемом диапазоне;
- **конкуренцию**: новый run не хуже champion'а сильнее, чем на N эпохах — «зелёный бенчмарк» перед мержем.

```bash
uv run pytest -q                              # юнит-тесты и schema-тесты
uv run python -m my_project.train --smoke    # 10 секунд, проверка живости пайплайна
```

### Контейнеризация

Когда вычисления уезжают с ноутбука, нужен воспроизводимый образ:

```dockerfile
FROM python:3.12-slim
COPY uv.lock pyproject.toml ./
RUN pip install uv && uv sync --frozen --no-dev
COPY src/ src/
CMD ["uv", "run", "python", "-m", "my_project.train"]
```

Такой образ запускается в K8s Job, в Spot-инстансе, в Airflow-контейнере — везде одинаково.

---

## CI и защита кода

Защиту «нельзя менять код оценки» делают не самописным SHA-guard, а средствами git и CI:

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --group dev --locked
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest -q
```

Проверка, что защищённые файлы не менялись:

```bash
git diff --exit-code origin/main -- src/my_project/evaluation/
```

Дополнительно:

- **CODEOWNERS** — файл защиты с ревью владельцем;
- **protected branch** — merge только после CI;
- **pre-commit** — `ruff`, `detect-secrets`, `nbstripout`;
- **DVC repro check** — в PR запускать `dvc repro --dry`, чтобы поймать «забытые» изменения пайплайна;
- **лицензии и секреты** — `detect-secrets` в pre-commit, `.env` в `.gitignore`.

---

## Антипаттерны

| Не делать | Почему | Что делать вместо |
| --- | --- | --- |
| Свой формат метрик рядом с MLflow | два источника истины, расхождения | MLflow как единственный источник |
| Свои `Protocol`/`ABC` ради «чистоты» | лишние файлы и слои без второй реализации | ABC модели, если моделей несколько; иначе — прямой код |
| Собственный tracking-сервер | MLflow уже это умеет | `mlflow server` |
| Собственный guard целостности | не защищает от намеренной подмены | git + CI + CODEOWNERS |
| `shell=True` в subprocess | инъекция | список аргументов |
| Конфиги в коде | нельзя переопределить и зафиксировать | Hydra |
| Данные в git | раздувает репозиторий | DVC |
| Notebook как источник истины | не воспроизводится | код в `src/`, ноутбук — исследование |
| Секреты в YAML или в коде | утечка | `.env` / секреты CI |
| `torch.manual_seed` без остальных | расходится между запусками | `seed_everything` + seed воркеров |
| Логика признаков живёт в ноутбуке, а в проде её переписывают заново | вечный training-serving skew | один sklearn Pipeline / ColumnTransformer, сохраняется целиком (joblib / MLflow) |
| Пути вида `C:/Users/me/Downloads/data_final_v3_NEW.csv` в коде | непереносимость | пути в конфиге, данные через DVC |
| Файлы `data_clean_final2.csv` вместо версий и пайплайна | невозможно воспроизвести | DVC-стадии, версии данных |
| Правки raw-данных руками | нарушена иммутабельность raw | никак не править raw, всё — кодом из raw |
| Магические числа и списки колонок прямо в коде, а не в конфиге | не параметризовано | всё в configs/ |
| Один гигантский `preprocess.py` на 2000 строк без функций и тестов | не тестируется, не читается | мелкие чистые функции в `src/`, тесты на каждую |
| Нет seed, нет фиксации версий библиотек | невоспроизводимость | `seed_everything`, `uv.lock` / `pyproject.toml` |
| Копипаста препроцессинга для train, val, test вместо одного `transform` | утечки, расхождения | `fit` на train, `transform` везде — sklearn Pipeline |
| Pickle от версии sklearn, которой уже нет | не загрузится | сохранять через MLflow / joblib с версией, контейнеризация |
| Ноутбук, который работает только при выполнении ячеек в определённом порядке | скрытое состояние | `%autoreload 2`, код в `src/`, тесты |
| Нет проверки данных на входе: пайплайн «успешно» отрабатывает на мусоре | молчаливые ошибки | Pandera / Great Expectations на границе стадий DVC |

---

## Архитектура пайплайна: стадии-функции + DVC

Паттерн **«стадия = чистая функция, читает parquet → пишет parquet»** — это индустриальный стандарт для батч-ML. Он работает поверх любого dataframe-инструмента (polars, pandas, Spark, DuckDB) и не зависит от `sklearn.Pipeline`.

### Почему не sklearn.Pipeline для современных задач

`sklearn.Pipeline` создан для одной таблицы `X → y` с построчными преобразованиями (`fit` на train, `transform` везде). Он отлично работает для классических табличных задач, но ломается там, где:

- **Данные — это лог событий** `(user, item, ts)`, а не плоская таблица. Признаки (популярность, активность, ко-встречаемость) — агрегаты по логу, а не построчные трансформы.
- **Утечка временная, а не построчная.** `Pipeline` защищает от утечки только внутри `fit/transform`. Если вы посчитали популярность товара по всему логу, а потом сплитили — `Pipeline` этого не заметит. Правильная защита — признаки «на момент cutoff» (as-of features).
- **Сплит по времени**, а не `train_test_split`. Метрики — ранжирующие (Recall@k, NDCG@k), а не AUC по строкам.
- **Инструмент — polars/DuckDB/Spark** с ленивыми запросами. `Pipeline` требует материализованные pandas/numpy.

`sklearn.Pipeline` остаётся уместен **только** для финального ранкера: когда у вас уже готова матрица признаков `(user, item) → features`, и вам нужно сохранить «предобработка + модель» как один артефакт (joblib/MLflow).

### Паттерн: стадии-функции + DVC

Каждая стадия — это Python-функция (или скрипт), которая:

1. Читает входные parquet через `scan_parquet` (лениво).
2. Делает работу.
3. Пишет результат `sink_parquet` (материализует).
4. Зависящие стадии читают этот parquet.

DVC (`dvc.yaml`) связывает стадии графом зависимостей, кэширует по хэшам входов и перезапускает только изменившееся.

```text
raw ─▶ clean ─▶ split ─▶ candidates ─▶ features ─▶ ranker ─▶ evaluate
     (типы,   (cutoff   (популярное,  (user/item/   (LightGBM  (Recall@k,
      id→int,  по        ALS/ItemKNN,  пара, как     lambdarank NDCG@k vs
      дубли)   времени)  co-visit.)    будто в T)   / CatBoost) baseline)
```

#### 1. clean

`scan_parquet` → приведение типов (`Int32`, `Categorical`/`Enum`) → удаление дублей → `sink_parquet` в `data/interim/`. Валидация схемы — Pandera (поддерживает polars).

#### 2. split

Не копии данных, а **cutoff-метки времени** в конфиге (`train < T1`, `valid ∈ [T1, T2)`, `test ≥ T2`). Фильтрация лениво за секунды.

#### 3. candidates

Сначала бейзлайны (популярное, «последние действия пользователя»), потом ALS/ItemKNN/two-tower. Результат — таблица `(user, item, source, score)`. Качество кандидатов = потолок всей системы (Recall@N кандидатов до ранкера).

#### 4. features

Функции `build_features(interactions, cutoff)`, использующие **только события до `cutoff`**. Это аналог «fit на train»: статистики считаются из окна до cutoff и сохраняются как parquet. Для признаков документов, меняющихся во времени — `join_asof` (polars умеет).

#### 5. ranker

LightGBM `lambdarank` / CatBoost `YetiRank`. Обучающие метки — из окна после cutoff. Данные отсортированы по группам (пользователям), размеры групп переданы модели.

#### 6. evaluate

Recall@k / NDCG@k / MAP@k по пользователям, **всегда с бейзлайном «топ популярного»**. Если сложная модель не бьёт популярное — это важнее любого графика.

### Практика работы с большими данными (3+ ГБ parquet)

- `pl.scan_parquet` + ленивые запросы; колонки выбирайте как можно раньше.
- `sink_parquet` для результатов, не влезающих в память.
- `collect(engine="streaming")` (проверьте версию polars).
- id → `Int32`: существенная экономия памяти.
- Конвертация в pandas/numpy **только** на последнем шаге (матрица для ранкера), и только нужное; негативы сэмплируете.

### Что взять из гайда, что заменить

- **Оставить:** DVC-стадии, Hydra/pydantic для cutoff и параметров, MLflow для метрик, pytest.
- **Заменить:** раздел про `sklearn.Pipeline` для задач с логами событий → раздел про «стадии + признаки as-of cutoff».
- **Тесты:** именно на отсутствие утечки по времени: «ни один признак для события в момент T не использует данные позже T».

---

## EDA: систематика и хранение

Текущий подход «выводы текстом в `eda_notes.md`, графики никуда» — худший вариант: доказательство теряется, текст остаётся без опоры. Разделите EDA на три вида с разной судьбой.

### 1. Разведка (одноразовая)

Ноутбук, где вы смотрите на данные и задаёте вопросы. **Выбрасывается без сожаления**, ничего хранить не нужно.

### 2. Находки (то, что стоит сохранить)

Для каждой находки, повлиявшей на решение, запишите в `docs/eda.md` три вещи:

- **Вопрос** — что проверяли.
- **Результат** — 1–2 ключевых графика или числа.
- **Решение** — что из этого следует (идёт в конфиг, а не в память).

Пример: *«медиана активности пользователя — 1 событие, 99% — 298, значит `min_interactions=5` для обучения, а холодных обрабатываем популярным»*. Это попадает в конфиг, а не в память.

Графики небольшие — коммитьте как PNG рядом с заметкой в `docs/` (или генерируйте функцией для воспроизведения одной командой).

### 3. Автоматический профиль данных (мониторинг)

Стадия DVC `profile`, которая считает фиксированный набор чисел:

- число строк, доля пропусков, диапазон времени
- число пользователей/товаров
- доля холодных пользователей/товаров в valid
- популярность топ-k, дрейф распределений

Пишет `reports/data_profile.json`. DVC умеет сравнивать метрики между версиями (`dvc metrics diff`), поэтому при новой версии данных вы сразу видите, что изменилось.

**Что НЕ нужно:** хранить в git графики по сырым данным или результаты каждого ноутбука. Графики дешёво перегенерировать, если данные и код версионированы.

### Чек-лист вопросов для EDA (общий, не только recsys)

- Распределение целевой переменной / событий на сущность (длинный хвост, log-log), доля с одним событием.
- **Доля «холодных» сущностей** в valid относительно train — определяет стратегию холодного старта.
- Покрытие по времени, сезонность, пропуски и «дыры» в логе, дрейф популярности.
- **Повторное потребление**: возвращаются ли сущности к тем же объектам?
- Дубли, порядок меток времени, события «из будущего» (проверка на утечку).
- Типы событий и их соотношение; смещение показа (exposure bias).
- Длина сессий; изменения каталога (появление/исчезновение объектов).
- Бейзлайны: что даёт простые эвристики (популярное, последнее, случайное) на valid.

### Как делать EDA на больших данных

Никогда не рисуйте сырые данные. Сначала агрегируйте лениво (polars/DuckDB), потом рисуйте маленькую таблицу (`.to_pandas()` на агрегате) через matplotlib/seaborn/plotly/altair.

---

## Хранение графиков: по типу, а не в кучу

Ручные сохранения быстро превращаются в `plot_final2.png`. Лучше по типу графика:

| Тип графика | Куда класть | Как воспроизводить |
| ------------- | ------------- | --------------------- |
| **Обучение/оценка модели** (кривые loss, NDCG@k по эпохам) | **MLflow** — `mlflow.log_figure(fig, "plots/ndcg_by_k.png")` | Привязаны к run'у, параметрам, версии кода. Сравнение в UI. |
| **Данные для графика, а не PNG** | Агрегированная таблица (CSV/parquet) + функция рисования | `dvc plots` умеет рисовать из CSV/JSON; PNG по запросу. |
| **Профиль данных / распределения** (выход стадии EDA/profile) | **DVC** — `reports/figures/<стадия>/<смысл>.png` | Не в git, под DVC. Воспроизводятся `dvc repro`, перезаписываются. Имена по смыслу, без дат/«final». |
| **Доказательства из EDA** (ключевые графики для решений) | **git** — `docs/` рядом с заметкой, где объясняют решение | Единственное исключение: графики в git как доказательства. |
| **Интерактивные / отчёты** (plotly, Quarto) | Один HTML-артефакт | `jupyter nbconvert --execute --to html` или Quarto. |

**Правило:** график либо привязан к запуску (MLflow), либо воспроизводится из стадии (DVC), либо является доказательством в документе (git). Всё остальное — одноразовое и выбрасывается.

---

## В каком порядке осваивать инженерную часть

1. **Git** и нормальная работа с ветками.
2. **Python-пакет:** `pyproject.toml`, `src`-layout, `pip install -e .`, импорты.
3. **Окружения:** uv/venv, фиксация зависимостей.
4. **Функции вместо ячеек:** рефакторинг ноутбука в модули.
5. **pytest** и тесты на трансформации и утечки.
6. **Конфиги:** pydantic + YAML, затем Hydra.
7. **sklearn Pipeline/ColumnTransformer** + сохранение целого пайплайна.
8. **Валидация:** pandera, затем Great Expectations.
9. **DVC:** версии данных и граф стадий.
10. **CLI** (typer / click), Makefile / justfile.
11. **pre-commit + CI.**
12. **MLflow** для экспериментов.
13. **Docker.**
14. **Оркестратор** (Prefect или Dagster).
15. **Мониторинг и feature store.**

---

## Чеклист перед коммитом

```bash
just format      # ruff format
just lint        # ruff check
just typecheck   # mypy
just test        # pytest
just repro       # dvc repro — пайплайн собирается
```

- [ ] Нет секретов в коде, YAML и ноутбуках (`.env` в `.gitignore`).
- [ ] Все параметры эксперимента — в `conf/`, а не в коде.
- [ ] Seed зафиксирован и одинаков для всех библиотек.
- [ ] Датасет прошёл Pandera-схему.
- [ ] В MLflow записаны: параметры, метрики со `step`, артефакты, версия кода и данных.
- [ ] `dvc.lock` и `uv.lock` обновлены и в git.
- [ ] Код в `src/`, а не только в ноутбуке.
- [ ] Тест на маленьком фикстуре проходит.

---

## Что читать по инженерной части

- *Designing Machine Learning Systems*, Chip Huyen.
- Документация DVC (Get Started + Pipelines).
- Cookiecutter Data Science: документация и философия.
- Kedro docs: даже если не будете использовать, учит мыслить пайплайнами.
- *Software Engineering for Data Scientists* и подобные материалы; Google «Rules of Machine Learning» (раздел про данные и пайплайны).
- «Hidden Technical Debt in Machine Learning Systems» (Sculley et al.), статья о том, почему код вокруг модели важнее самой модели.

---

## Практическое задание

Возьмите любой свой ноутбук с Kaggle и превратите его в проект уровня 2: репозиторий со структурой выше, `dvc repro` собирает всё от raw до метрики, `pytest` проверяет отсутствие утечек, CI это запускает, а README объясняет, как воспроизвести результат одной командой. Это лучший учебный проект по теме и сильная часть портфолио.
