"""calculations.py - производные колонки и агрегации единой таблицы."""
import ast
import operator
import re

import numpy as np
import pandas as pd

EXTRA_LABELS: dict[str, str] = {}

BASE_LABELS = {
    "period": "Период",
    "fact_amount": "Факт, ₽",
    "plan_amount": "План, ₽",
    "fact_qty": "Факт, шт",
    "plan_qty": "План, шт",
    "discount_amount": "Скидка, ₽",
    "sale_amount": "Сумма продажи, ₽",
    "region_name": "Регион",
    "city": "Город",
    "outlet_type": "Тип точки",
    "outlet_name": "Торговая точка",
    "product_name": "Продукт",
    "category": "Категория",
    "client_name": "Клиент",
    "reason": "Причина",
}

CALC_LABELS = {
    "deviation_amt": "Отклонение, ₽",
    "deviation_pct": "Отклонение, %",
    "completion_pct": "Выполнение плана, %",
    "mom_pct": "Динамика к пред. периоду, %",
    "yoy_pct": "Динамика год к году, %",
    "share_pct": "Доля в периоде, %",
    "cum_amount": "Накопительный итог, ₽",
    "rank_in_period": "Ранг в периоде",
    "group_avg": "Среднее по разрезу, ₽",
    "group_median": "Медиана по разрезу, ₽",
    "group_min": "Минимум по разрезу, ₽",
    "group_max": "Максимум по разрезу, ₽",
}

ALL_LABELS = {**BASE_LABELS, **CALC_LABELS}

DIM_COLUMNS = ["region_name", "outlet_name", "product_name", "category",
               "client_name", "city", "outlet_type", "reason"]

METRIC_COLUMNS = ["fact_amount", "plan_amount", "fact_qty", "plan_qty",
                  "discount_amount", "sale_amount"]

GRAN_FREQ = {"month": "M", "quarter": "Q", "year": "Y"}

def _pick_metric(df: pd.DataFrame, preferred: str = "fact_amount") -> str:
    """Возвращает первую доступную числовую метрику из витрины."""
    if preferred in df.columns:
        return preferred
    for col in METRIC_COLUMNS:
        if col in df.columns:
            return col
    raise CalcError("В таблице нет числовой метрики для вычисления")

class CalcError(Exception):
    """Вычисление неприменимо к текущей таблице."""


def column_label(col: str) -> str:
    return ALL_LABELS.get(col, EXTRA_LABELS.get(col, col))


def infer_dimension(df: pd.DataFrame) -> str:
    for col in DIM_COLUMNS:
        if col in df.columns:
            return col
    for col in df.columns:
        if col != "period" and df[col].dtype == object:
            return col
    raise CalcError("В таблице нет колонки-разреза")


def _floor_period(series: pd.Series, granularity: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    freq = GRAN_FREQ.get(granularity, "M")
    return ts.dt.to_period(freq).dt.to_timestamp()


def aggregate_view(df: pd.DataFrame, dimension: str, granularity: str) -> pd.DataFrame:
    """Агрегат единой таблицы до зерна период x разрез."""
    d = df.copy()
    d["period"] = _floor_period(d["period"], granularity)
    metrics = [c for c in METRIC_COLUMNS if c in d.columns]
    out = d.groupby(["period", dimension], as_index=False, dropna=False)[metrics].sum()
    return out.sort_values(["period", dimension]).reset_index(drop=True)


def _require_plan(df: pd.DataFrame) -> None:
    if "plan_amount" not in df.columns or "fact_amount" not in df.columns:
        raise CalcError("Для этого вычисления нужны план и факт")


def add_deviation_amt(df: pd.DataFrame, dimension: str):
    _require_plan(df)
    d = df.copy()
    d["deviation_amt"] = d["fact_amount"] - d["plan_amount"]
    return d, "Отклонение факта от плана, ₽", ["deviation_amt"]


def add_deviation_pct(df: pd.DataFrame, dimension: str):
    _require_plan(df)
    d = df.copy()
    d["deviation_pct"] = np.where(d["plan_amount"] > 0,
                                  (d["fact_amount"] - d["plan_amount"]) / d["plan_amount"] * 100,
                                  np.nan)
    return d, "Отклонение факта от плана, %", ["deviation_pct"]


def add_completion(df: pd.DataFrame, dimension: str):
    _require_plan(df)
    d = df.copy()
    d["completion_pct"] = np.where(d["plan_amount"] > 0,
                                   d["fact_amount"] / d["plan_amount"] * 100,
                                   np.nan)
    return d, "Процент выполнения плана, %", ["completion_pct"]


def add_mom(df: pd.DataFrame, dimension: str):
    d = df.copy().sort_values(["period", dimension])
    metric = _pick_metric(d)
    d["mom_pct"] = d.groupby(dimension)[metric].pct_change() * 100
    return d, "Динамика к прошлому периоду, %", ["mom_pct"]


def add_share(df: pd.DataFrame, dimension: str):
    d = df.copy()
    metric = _pick_metric(d)
    total = d.groupby("period")[metric].transform("sum")
    d["share_pct"] = np.where(total > 0, d[metric] / total * 100, np.nan)
    return d, "Доля в объёме периода, %", ["share_pct"]


def add_cumsum(df: pd.DataFrame, dimension: str):
    d = df.copy().sort_values(["period", dimension])
    metric = _pick_metric(d)
    d["cum_amount"] = d.groupby(dimension)[metric].cumsum()
    return d, "Накопительный итог, ₽", ["cum_amount"]


def add_rank(df: pd.DataFrame, dimension: str):
    d = df.copy()
    metric = _pick_metric(d)
    d["rank_in_period"] = d.groupby("period")[metric].rank(ascending=False,
                                                           method="min").astype(int)
    return d, "Ранг в периоде по факту", ["rank_in_period"]


def add_group_avg(df: pd.DataFrame, dimension: str):
    d = df.copy()
    metric = _pick_metric(d)
    d["group_avg"] = d.groupby(dimension)[metric].transform("mean")
    return d, "Среднее по разрезу, ₽", ["group_avg"]


def add_group_median(df: pd.DataFrame, dimension: str):
    d = df.copy()
    metric = _pick_metric(d)
    d["group_median"] = d.groupby(dimension)[metric].transform("median")
    return d, "Медиана по разрезу, ₽", ["group_median"]


def add_group_min(df: pd.DataFrame, dimension: str):
    d = df.copy()
    metric = _pick_metric(d)
    d["group_min"] = d.groupby(dimension)[metric].transform("min")
    return d, "Минимум по разрезу, ₽", ["group_min"]


def add_group_max(df: pd.DataFrame, dimension: str):
    d = df.copy()
    metric = _pick_metric(d)
    d["group_max"] = d.groupby(dimension)[metric].transform("max")
    return d, "Максимум по разрезу, ₽", ["group_max"]


def add_yoy(df: pd.DataFrame, dimension: str):
    d = df.copy()
    metric = _pick_metric(d)
    prev = d[[dimension, "period", metric]].copy()
    prev["period"] = prev["period"] + pd.DateOffset(years=1)
    prev = prev.rename(columns={metric: "_prev_year"})
    d = d.merge(prev, on=[dimension, "period"], how="left")
    d["yoy_pct"] = np.where(d["_prev_year"] > 0,
                            (d[metric] - d["_prev_year"]) / d["_prev_year"] * 100,
                            np.nan)
    d.drop(columns=["_prev_year"], inplace=True)
    return d, "Динамика год к году, %", ["yoy_pct"]


_PRESET_FUNCS = {
    "deviation_amt": add_deviation_amt,
    "deviation_pct": add_deviation_pct,
    "completion": add_completion,
    "mom": add_mom,
    "yoy": add_yoy,
    "share": add_share,
    "cumsum": add_cumsum,
    "rank": add_rank,
    "group_avg": add_group_avg,
    "group_median": add_group_median,
    "group_min": add_group_min,
    "group_max": add_group_max,
}

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": np.abs, "round": np.round, "minimum": np.minimum, "maximum": np.maximum}


def safe_eval_formula(df: pd.DataFrame, formula: str) -> pd.Series:
    """Безопасный разбор арифметической формулы над колонками таблицы."""
    try:
        node = ast.parse(formula, mode="eval").body
    except SyntaxError as exc:
        raise CalcError(f"Не удалось разобрать формулу: {formula}") from exc

    def ev(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                return n.value
            raise CalcError("В формуле допустимы только числа и имена колонок")
        if isinstance(n, ast.Name):
            if n.id in df.columns:
                return df[n.id]
            raise CalcError(f"Неизвестная колонка в формуле: {n.id}")
        if isinstance(n, ast.BinOp):
            op = _BIN_OPS.get(type(n.op))
            if op is None:
                raise CalcError("Неподдерживаемый оператор в формуле")
            return op(ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            op = _UNARY_OPS.get(type(n.op))
            if op is None:
                raise CalcError("Неподдерживаемый оператор в формуле")
            return op(ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise CalcError("Неподдерживаемое выражение в формуле")

    res = ev(node)
    if isinstance(res, pd.Series):
        return res
    return pd.Series(np.full(len(df), float(res)), index=df.index)


def normalize_query(text: str) -> tuple[str, str]:
    """Распознаёт тип вычисления: (ключ пресета, текст формулы)."""
    t = text.lower()
    m = re.search(r"формула\s*[:\-]?\s*(.+)", t)
    if m:
        expr = re.match(r"[a-z0-9_+\-*/%().,\s]+", m.group(1))
        return "formula", expr.group(0).strip() if expr else ""
    if "отклонени" in t or ("разниц" in t and "план" in t):
        if "%" in text or "процент" in t:
            return "deviation_pct", ""
        return "deviation_amt", ""
    if "выполнени" in t:
        return "completion", ""
    if "медиан" in t:
        return "group_median", ""
    if "минимум" in t or "наименьш" in t:
        return "group_min", ""
    if "максимум" in t or "наибольш" in t:
        return "group_max", ""
    if "год к году" in t or "к прошлому году" in t or "к предыдущему году" in t or "yoy" in t:
        return "yoy", ""
    if "динамик" in t or "к прошлому" in t or "к предыдущему" in t:
        return "mom", ""
    if "доля" in t or "доли" in t or "долю" in t or "долей" in t:
        return "share", ""
    if "накопительн" in t or "с начала" in t:
        return "cumsum", ""
    if "ранжир" in t or "ранг" in t:
        return "rank", ""
    if "средн" in t:
        return "group_avg", ""
    return "", ""


def apply_preset(df: pd.DataFrame, query: str, dimension: str | None = None):
    """Применяет вычисление к таблице. Возвращает (df, описание, колонки) или None."""
    key, formula = normalize_query(query)
    if not key:
        return None
    dim = dimension or infer_dimension(df)
    if key == "formula":
        if not formula:
            return None
        d = df.copy()
        name = "formula_" + re.sub(r"[^a-zа-яё0-9]+", "_", formula)[:24].strip("_")
        d[name] = safe_eval_formula(d, formula)
        desc = f"Формула: {formula}"
        EXTRA_LABELS[name] = desc
        return d, desc, [name]
    return _PRESET_FUNCS[key](df, dim)