"""nlu.py - разбор запросов на естественном языке."""
import re
from dataclasses import dataclass, field

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "май": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}

INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("help", ("помощь", "что ты умеешь", "как пользоваться", "справк")),
    ("list_dashboards", ("мои дашборд", "список дашборд", "все дашборд", "сохраненные дашборд")),
    ("open_dashboard", ("открой дашборд", "вернись к дашборд", "покажи дашборд")),
    ("delete_dashboard", ("удали дашборд", "удалить дашборд")),
    ("rename_dashboard", ("переимен",)),
    ("save_dashboard", ("сохрани", "сохранить", "запомни")),
    ("export", ("скачай", "скачать", "экспорт", "выгрузи")),
    ("add_calc", ("добавь колонк", "добавь столбец", "добавь вычислен",
                  "добавь метрик", "добавь формул", "добавь динамик", "добавь медиан",
                  "добавь минимум", "добавь максимум", "добавь среднее",
                  "посчитай", "вычисли", "формула")),
    ("rebuild_chart", ("перестрой", "отсортируй", "измени график", "обнови график",
                       "поменяй", "переключи", "только", "сделай")),
    ("show_table", ("покажи таблиц", "покажи данные", "покажи выгрузк", "таблицу")),
    ("discounts", ("скидк",)),
]

DIM_PATTERNS: list[tuple[str, str]] = [
    ("category", r"(?:по|на)\s+категори|разрез\w*\s+(?:на\s+)?категори"),
    ("region_name", r"(?:по|на)\s+регион|разрез\w*\s+(?:на\s+)?регион"),
    ("client_name", r"(?:по|на)\s+клиент|разрез\w*\s+(?:на\s+)?клиент"),
    ("outlet_name", r"(?:по|на)\s+(?:торгов\w*\s+)?точк|разрез\w*\s+(?:на\s+)?(?:торгов\w*\s+)?точк|(?:по|на)\s+магазин"),
    ("product_name", r"(?:по|на)\s+(?:продукт|товар)|разрез\w*\s+(?:на\s+)?(?:продукт|товар)"),
]

SORT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("completion_pct", ("выполнени",)),
    ("deviation_pct", ("отклонени",)),
    ("share_pct", ("доля", "доли", "долю", "долей")),
    ("mom_pct", ("динамик",)),
    ("discount_amount", ("скидк",)),
    ("plan_amount", ("плану",)),
    ("fact_amount", ("факт", "объем", "объём", "продаж", "выручк")),
]

FILTER_COLUMNS = ["region_name", "city", "category", "client_name",
                  "product_name", "outlet_name"]

KNOWN_VALUES: dict[str, list[str]] = {}


@dataclass
class Intent:
    intent: str = "build_dashboard"
    dimension: str | None = None
    granularity: str | None = None
    period: dict | None = None
    filters: dict = field(default_factory=dict)
    calc_query: str | None = None
    sort_by: str | None = None
    top_n: int | None = None
    dashboard_name: str | None = None
    export_format: str | None = None
    text: str = ""
    dashboard_name: str | None = None
    new_name: str | None = None


def set_known_values(mapping: dict[str, list[str]]) -> None:
    KNOWN_VALUES.clear()
    KNOWN_VALUES.update(mapping)


def _clean(text: str) -> str:
    t = re.sub(r"[^\w\s№%-]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def _norm_value(value: str) -> str:
    v = value.lower().replace("«", "").replace("»", "").replace('"', "")
    v = re.sub(r"\b(ооо|ип|пао|ао)\b", " ", v)
    v = re.sub(r"[^\w\s]", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def _parse_period(t: str) -> dict | None:
    m = re.search(r"последн\w*\s+(\d+)\s+(лет|года)", t)
    if m:
        return {"type": "last_years", "value": int(m.group(1))}
    if "последний год" in t or "за год" in t or "последние 12 месяцев" in t:
        return {"type": "last_year"}
    if "последний квартал" in t:
        return {"type": "last_quarter"}
    if "последний месяц" in t:
        return {"type": "last_month"}
    m = re.search(r"(\d)\s*[-й]?\s*квартал\w*\s*(20\d{2})?", t)
    if m:
        return {"type": "quarter", "quarter": int(m.group(1)),
                "year": int(m.group(2)) if m.group(2) else None}
    for stem, num in MONTHS.items():
        if stem in t:
            m = re.search(r"20\d{2}", t)
            return {"type": "month", "month": num,
                    "year": int(m.group(0)) if m else None}
    m = re.search(r"20\d{2}", t)
    if m:
        return {"type": "year", "year": int(m.group(0))}
    return None


def _match_filters(t: str) -> dict:
    filters = {}
    for col in FILTER_COLUMNS:
        hits = [v for v in KNOWN_VALUES.get(col, [])
                if len(_norm_value(v)) >= 4 and _norm_value(v) in t]
        if hits:
            filters[col] = hits
    return filters


def parse(text: str) -> Intent:
    """Разбирает сообщение пользователя в структуру Intent."""
    t = _clean(text)
    it = Intent(text=text)
    for intent, keys in INTENT_RULES:
        if any(k in t for k in keys):
            it.intent = intent
            break
    for dim, pattern in DIM_PATTERNS:
        if re.search(pattern, t):
            it.dimension = dim
            break
    if "поквартальн" in t or "по кварталам" in t:
        it.granularity = "quarter"
    elif "по годам" in t:
        it.granularity = "year"
    elif "помесячн" in t or "по месяцам" in t:
        it.granularity = "month"
    it.period = _parse_period(t)
    it.filters = _match_filters(t)
    m = re.search(r"топ[- ]?(\d+)", t) or re.search(r"первы[ех] (\d+)", t)
    if m:
        it.top_n = int(m.group(1))
    if it.intent == "rebuild_chart" or "отсортируй" in t or "сортировк" in t or it.top_n:
        for col, keys in SORT_RULES:
            if any(k in t for k in keys):
                it.sort_by = col
                break
    names = re.findall(r"«([^»]+)»", text) or re.findall(r'"([^"]+)"', text)
    if names:
        it.dashboard_name = names[0].strip()
        if len(names) > 1:
            it.new_name = names[1].strip()
    else:
        m = re.search(r"под названием\s+([^.,!]+)", text)
        if m:
            it.dashboard_name = m.group(1).strip()
    if "csv" in t:
        it.export_format = "csv"
    elif "excel" in t or "xlsx" in t:
        it.export_format = "xlsx"
    elif "html" in t:
        it.export_format = "html"
    it.calc_query = text
    return it