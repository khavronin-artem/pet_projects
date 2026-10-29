# Подготовка данных

## Столбцы таблиц ESCI (из docs/readme-of-esci.md)

### shopping_queries_dataset_examples.parquet

- `example_id` — уникальный id пары (query, product)
- `query` — текст запроса
- `query_id` — id запроса
- `product_id` — id товара
- `product_locale` — язык/регион (us/es/jp)
- `esci_label` — метка релевантности: E (Exact), S (Substitute), C (Complement), I (Irrelevant)
- `small_version` — флаг уменьшенной версии (Task 1)
- `large_version` — флаг полной версии (Task 2, 3)
- `split` — train / test

- Exact - точное совпадение с запросом
- Substitute - товар-заменитель (например, если оригинала нет или он дороже)
- Complement - товар, который покупают вместе с искомым ("iPhone" - "Защитное стекло")
- Irrelevant - нерелеватный

### shopping_queries_dataset_products.parquet

- `product_id` — id товара
- `product_title` — заголовок товара
- `product_description` — описание
- `product_bullet_point` — пункты (bullet points)
- `product_brand` — бренд
- `product_color` — цвет
- `product_locale` — локаль товара

### shopping_queries_dataset_sources.csv

- `query_id` — id запроса
- `source` — источник

## 3 постановки задачи (из readme к датасету)

1. **Query-Product Ranking (Task 1)** — по запросу и списку найденных товаров нужно упорядочить их так, чтобы более релевантные были выше. Используется small_version (48 300 запросов, ~1,1 млн пар). Метрика — nDCG. Модели: cross-encoder на query + product_title.

2. **Multi-class Product Classification (Task 2)** — для каждой пары (query, product) предсказать одну из 4 меток ESCI: Exact / Substitute / Complement / Irrelevant. Используется large_version (~2,6 млн пар). Метрики — Macro F1, Micro F1.

3. **Product Substitute Identification (Task 3)** — бинарная задача на базе large_version: найти товары-заменители (label == "S"). Остальные классы сворачиваются в 0. Метрики те же — Macro/Micro F1.

## EDA инсайты

### Дубликаты

1. Точных дубликатов нет ни в одной из двух таблиц.
2. Одному `product_id` соответствует несколько `product_locale`. Один документ может быть на нескольких языках.
3. В таблице нет одинаковых `["query_id", "product_id"]` в `train` и `test` (нет утечки). `example_id` уникален (действительно является id). `example_id` соотносится с `["query_id", "product_id"]` (взаимозаменяемо).
4. Fuzzy дубликаты (почти дубликаты) не находились, пока слишком сложно

### Пропуски

1. В таблице `examples` нет пропусков.
2. В таблице `products` могут отсутствовать `product_description`, `product_bullet_point`, `product_brand`, `product_color` (значение null)
3. Есть много "магических" null: "None" (строка), "-", "---" и т.д. в таблице `products`

### Типы и согласованность

1. В queries много non-ASCII символов (т.к. есть японский и испанский языки)
2. В таблице `products` много "грязных" значений, не содержащих смысла и т.д. (например `product_color` содержит "*1", "4030" или "100 cm")

## Этапы очистки

Какие действия выполняются для очистки

1.

## Валидация данных
