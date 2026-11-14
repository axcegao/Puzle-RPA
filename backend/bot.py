"""bot.py - диалоговая логика: сессии, обработчики, дашборды."""
import json
import os
import re
import uuid
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

import calculations as calc
import db
import nlu

STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
DASHBOARDS_FILE = os.path.join(STORAGE_DIR, "dashboards.json")
EXPORT_DIR = os.path.join(STORAGE_DIR, "exports")

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6",
           "#0ea5e9", "#f472b6", "#84cc16"]

DIM_LABELS = {
    "region_name": "Регион",
    "outlet_name": "Торговая точка",
    "product_name": "Продукт",
    "category": "Категория",
    "client_name": "Клиент",
    "city": "Город",
    "outlet_type": "Тип точки",
    "reason": "Причина",
}

ADDITIVE = {"fact_amount", "plan_amount", "fact_qty", "plan_qty",
            "discount_amount", "sale_amount", "deviation_amt", "cum_amount"}

CLARIFY_TEXT = ("Не хватает разреза. Уточните: по регионам, по торговым точкам, "
                "по продуктам, по категориям или по клиентам?")

HELP_TEXT = (
    "Я строю интерактивные дашборды по текстовому запросу.\n"
    "Примеры:\n"
    "- Собери дашборд по продажам: план-факт по регионам за последний год\n"
    "- покажи таблицу\n"
    "- добавь колонку с процентом выполнения плана\n"
    "- формула: (fact_amount - plan_amount) / plan_amount * 100\n"
    "- переключи разрез на торговые точки\n"
    "- покажи только Центральный регион\n"
    "- сделай поквартально\n"
    "- покажи скидки по клиенту за последние 3 года\n"
    "- сохрани дашборд под названием «Мой дашборд»\n"
    "- покажи мои дашборды / открой дашборд «Мой дашборд»\n"
    "- скачай таблицу csv / скачай дашборд в html"
)

sessions = {}


def refresh_known_values() -> None:
    """Передает в NLU справочные значения для распознавания фильтров."""
    mapping = {}
    frames = [db.get_sales_flat(), db.get_discounts()]
    for col in nlu.FILTER_COLUMNS:
        values = set()
        for frame in frames:
            if col in frame.columns:
                values.update(frame[col].dropna().unique().tolist())
        mapping[col] = sorted(values)
    nlu.set_known_values(mapping)


def period_bounds(period: dict | None):
    if not period:
        return None
    today = pd.Timestamp.today().normalize()
    kind = period["type"]
    if kind == "last_year":
        return today - pd.DateOffset(months=12), today
    if kind == "last_years":
        return today - pd.DateOffset(years=period["value"]), today
    if kind == "last_quarter":
        return today - pd.DateOffset(months=3), today
    if kind == "last_month":
        return today - pd.DateOffset(months=1), today
    if kind == "year":
        year = period["year"]
        return pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    if kind == "quarter":
        year = period.get("year") or today.year
        start = pd.Timestamp(year, 3 * (period["quarter"] - 1) + 1, 1)
        return start, start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    if kind == "month":
        year = period.get("year") or today.year
        start = pd.Timestamp(year, period["month"], 1)
        return start, start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    return None


def period_label(ts: pd.Timestamp, granularity: str) -> str:
    if granularity == "quarter":
        return f"{ts.year}-Q{(ts.month - 1) // 3 + 1}"
    if granularity == "year":
        return str(ts.year)
    return ts.strftime("%Y-%m")


def fmt_money(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


class Session:
    def __init__(self, sid: str):
        self.id = sid
        self.source = "sales"
        self.base = db.get_sales_with_plan()
        self.config = {"dimension": None, "granularity": "month", "period": None,
                       "filters": {}, "sort_by": None, "top_n": None, "metric": None}
        self.calc_history = []
        self.pending_clarify = False
        self.name = None
        self._fact_end = None

    def set_source(self, source: str) -> None:
        if source == self.source:
            return
        self.source = source
        self.base = db.get_sales_with_plan() if source == "sales" else db.get_discounts()
        self.calc_history = []
        self._fact_end = None
        self.config.update({"dimension": None, "filters": {}, "sort_by": None,
                            "top_n": None, "metric": None})

    def fact_end(self) -> pd.Timestamp:
        if self._fact_end is None:
            metric = "fact_amount" if "fact_amount" in self.base.columns else "discount_amount"
            mask = self.base[metric] > 0
            last = self.base.loc[mask, "period"].max()
            if pd.isna(last):
                last = self.base["period"].max()
            self._fact_end = last + pd.offsets.MonthEnd(0)
        return self._fact_end

    def view(self) -> pd.DataFrame:
        d = self.base
        bounds = period_bounds(self.config.get("period"))
        if bounds:
            start, end = bounds
            if self.config["period"]["type"] in ("last_year", "last_years",
                                                 "last_quarter", "last_month"):
                end = min(end, self.fact_end())
            d = d[(d["period"] >= start) & (d["period"] <= end)]
        for col, values in self.config.get("filters", {}).items():
            if col in d.columns and values:
                d = d[d[col].isin(values)]
        dim = self.config["dimension"]
        gran = self.config["granularity"]
        v = calc.aggregate_view(d, dim, gran)
        for item in self.calc_history:
            res = calc.apply_preset(v, item["query"], dim)
            if res:
                v = res[0]
        v["period_label"] = [period_label(t, gran) for t in v["period"]]
        return v


def _totals(v: pd.DataFrame, dim: str) -> pd.DataFrame:
    metrics = [c for c in v.columns if c not in ("period", "period_label", dim)]
    agg = {c: ("sum" if c in ADDITIVE else "mean") for c in metrics}
    return v.groupby(dim, as_index=False).agg(agg)


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.02, y=0.97, xanchor="left", yanchor="top"),
                      template="plotly_white",
                      font=dict(family="Segoe UI, Arial", size=12),
                      margin=dict(l=50, r=30, t=60, b=40),
                      legend=dict(orientation="h", y=-0.25),
                      modebar=dict(orientation="v", bgcolor="rgba(255,255,255,0.7)",
                                   color="#64748b", activecolor="#2563eb"))
    return fig


def build_charts(v: pd.DataFrame, cfg: dict) -> list:
    dim = cfg["dimension"]
    has_plan = "plan_amount" in v.columns
    main = "fact_amount" if has_plan else "discount_amount"
    sel = cfg.get("metric") or main
    if sel not in v.columns:
        sel = main
    totals = _totals(v, dim)
    sort_col = cfg.get("sort_by") or main
    if sort_col not in totals.columns:
        sort_col = main
    totals = totals.sort_values(sort_col, ascending=False)
    top_n = cfg.get("top_n") or 10
    cats = totals[dim].head(top_n).tolist()
    charts = []

    fig = go.Figure()
    t = totals[totals[dim].isin(cats)]
    names = t[dim].astype(str).tolist()
    short = [n if len(n) <= 32 else n[:31] + "…" for n in names]
    horizontal = dim in ("outlet_name", "product_name", "client_name")
    if horizontal:
        fig.add_bar(y=short, x=t[main], orientation="h",
                    name="Факт" if has_plan else "Скидки",
                    marker_color=PALETTE[0], customdata=names)
        if has_plan:
            fig.add_bar(y=short, x=t["plan_amount"], orientation="h", name="План",
                        marker_color=PALETTE[1], customdata=names)
        fig.update_traces(hovertemplate="%{customdata}<br>%{x:,.0f} ₽")
        fig.update_yaxes(autorange="reversed", automargin=True)
    else:
        fig.add_bar(x=short, y=t[main], name="Факт" if has_plan else "Скидки",
                    marker_color=PALETTE[0], customdata=names)
        if has_plan:
            fig.add_bar(x=short, y=t["plan_amount"], name="План",
                        marker_color=PALETTE[1], customdata=names)
        fig.update_traces(hovertemplate="%{customdata}<br>%{y:,.0f} ₽")
        fig.update_xaxes(tickangle=-35, automargin=True)
    charts.append(_layout(fig, f"План и факт по разрезам: {DIM_LABELS.get(dim, dim)}"))

    per_cols = [main] + (["plan_amount"] if has_plan else [])
    per = v.groupby("period_label", as_index=False)[per_cols].sum()
    per = per.sort_values("period_label")
    fig = go.Figure()
    fig.add_scatter(x=per["period_label"], y=per[main], mode="lines+markers",
                    name="Факт" if has_plan else "Скидки", line=dict(color=PALETTE[0]))
    if has_plan:
        fig.add_scatter(x=per["period_label"], y=per["plan_amount"], mode="lines+markers",
                        name="План", line=dict(color=PALETTE[1]))
    fig.update_traces(hovertemplate="%{x}<br>%{y:,.0f} ₽")
    charts.append(_layout(fig, "Динамика по периодам").update_xaxes(type="category"))

    if len(totals) > 12:
        top = totals.head(8)
        names = top[dim].astype(str).tolist()
        short = [n if len(n) <= 32 else n[:31] + "…" for n in names]
        fig = go.Figure(go.Bar(y=short, x=top[main], orientation="h",
                               marker_color=PALETTE[0], customdata=names))
        fig.update_traces(hovertemplate="%{customdata}<br>%{x:,.0f} ₽")
        fig.update_yaxes(autorange="reversed", automargin=True)
        charts.append(_layout(fig, "Структура объёма: топ-8"))
    else:
        pie_src = totals.head(8)
        labels = pie_src[dim].tolist()
        values = pie_src[main].tolist()
        rest = totals[main].sum() - sum(values)
        if rest > 0:
            labels.append("Прочие")
            values.append(rest)
        fig = go.Figure(go.Pie(labels=labels, values=values, marker_colors=PALETTE, hole=0.45))
        charts.append(_layout(fig, "Структура объёма"))

    top5 = totals[dim].head(5).tolist()
    aggfunc = "sum" if sel in ADDITIVE else "mean"
    pivot = v[v[dim].isin(top5)].pivot_table(index="period_label", columns=dim,
                                             values=sel, aggfunc=aggfunc)
    pivot = pivot.sort_index()
    fig = go.Figure()
    for i, col in enumerate(pivot.columns):
        fig.add_scatter(x=pivot.index.tolist(), y=pivot[col].tolist(), mode="lines+markers",
                        name=str(col), line=dict(color=PALETTE[i % len(PALETTE)]))
    unit = " ₽" if sel in ADDITIVE else ""
    fig.update_traces(hovertemplate="%{x}<br>%{y:,.2f}" + unit)
    charts.append(_layout(fig, f"Топ-5 разрезов в динамике: {calc.column_label(sel)}")
                  .update_xaxes(type="category"))
    return [json.loads(f.to_json()) for f in charts]


def compute_kpis(v: pd.DataFrame, cfg: dict) -> list:
    if "plan_amount" in v.columns:
        fact = v["fact_amount"].sum()
        plan = v["plan_amount"].sum()
        comp = fact / plan * 100 if plan else 0.0
        return [
            {"label": "Факт, ₽", "value": fmt_money(fact)},
            {"label": "План, ₽", "value": fmt_money(plan)},
            {"label": "Выполнение, %", "value": f"{comp:.1f}"},
            {"label": "Отклонение, ₽", "value": fmt_money(fact - plan)},
        ]
    disc = v["discount_amount"].sum()
    sale = v["sale_amount"].sum()
    share = disc / sale * 100 if sale else 0.0
    return [
        {"label": "Скидки, ₽", "value": fmt_money(disc)},
        {"label": "Оборот со скидками, ₽", "value": fmt_money(sale)},
        {"label": "Доля скидок, %", "value": f"{share:.1f}"},
        {"label": "Значений разреза", "value": str(v[cfg["dimension"]].nunique())},
    ]


def summarize(v: pd.DataFrame, cfg: dict) -> str:
    dim = cfg["dimension"]
    has_plan = "plan_amount" in v.columns
    main = "fact_amount" if has_plan else "discount_amount"
    totals = _totals(v, dim).sort_values(main, ascending=False)
    top = totals.iloc[0]
    lines = [f"Период: {v['period_label'].min()} - {v['period_label'].max()}.",
             f"Лидер по объёму: {top[dim]} ({fmt_money(top[main])} ₽)."]
    if has_plan:
        totals["comp"] = totals["fact_amount"] / totals["plan_amount"] * 100
        worst = totals.sort_values("comp").iloc[0]
        lines.append(f"Минимальное выполнение плана: {worst[dim]} ({worst['comp']:.1f}%).")
    if cfg.get("filters"):
        parts = [f"{DIM_LABELS.get(k, k)}: {', '.join(vals)}"
                 for k, vals in cfg["filters"].items() if vals]
        if parts:
            lines.append("Фильтры: " + "; ".join(parts) + ".")
    return " ".join(lines)


def table_payload(v: pd.DataFrame, cfg: dict) -> dict:
    dim = cfg["dimension"]
    cols = ["period_label", dim] + [c for c in v.columns
                                    if c not in ("period", "period_label", dim)]
    labels = ["Период", DIM_LABELS.get(dim, dim)] + [calc.column_label(c) for c in cols[2:]]
    rows = json.loads(v[cols].head(200).to_json(orient="values"))
    return {"columns": cols, "labels": labels, "rows": rows, "total_rows": len(v)}


def _export_frame(v: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dim = cfg["dimension"]
    cols = ["period_label", dim] + [c for c in v.columns
                                    if c not in ("period", "period_label", dim)]
    labels = ["Период", DIM_LABELS.get(dim, dim)] + [calc.column_label(c) for c in cols[2:]]
    out = v[cols].copy()
    out.columns = labels
    return out


def filter_options(sess: Session) -> dict:
    options = {}
    for col in nlu.FILTER_COLUMNS:
        if col in sess.base.columns:
            options[col] = sorted(sess.base[col].dropna().unique().tolist())[:500]
    return options


def reply(sess: Session, kind: str, text: str, **extra) -> dict:
    payload = {"session_id": sess.id, "kind": kind, "text": text, "config": sess.config}
    payload.update(extra)
    return payload


def render_dashboard(sess: Session, intro: str = "") -> dict:
    dim = sess.config["dimension"]
    if not dim or dim not in sess.base.columns:
        sess.config["dimension"] = None
        sess.pending_clarify = True
        return reply(sess, "clarify", CLARIFY_TEXT)
    v = sess.view()
    if v.empty:
        return reply(sess, "text", "По заданным условиям данных нет. Ослабьте фильтры или период.")
    return reply(sess, "dashboard", intro or "Готово.",
                 kpis=compute_kpis(v, sess.config),
                 charts=build_charts(v, sess.config),
                 summary=summarize(v, sess.config),
                 table=table_payload(v, sess.config),
                 filter_options=filter_options(sess))


def _load_store() -> list:
    if os.path.exists(DASHBOARDS_FILE):
        with open(DASHBOARDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_store(items: list) -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(DASHBOARDS_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _apply_intent_config(sess: Session, it: nlu.Intent, replace_filters: bool) -> None:
    if it.dimension:
        sess.config["dimension"] = it.dimension
    if it.granularity:
        sess.config["granularity"] = it.granularity
    if it.period:
        sess.config["period"] = it.period
    if it.filters:
        if replace_filters:
            sess.config["filters"] = dict(it.filters)
        else:
            sess.config["filters"].update(it.filters)
    if it.sort_by:
        sess.config["sort_by"] = it.sort_by
    if it.top_n:
        sess.config["top_n"] = it.top_n


def h_help(sess: Session, it: nlu.Intent) -> dict:
    return reply(sess, "text", HELP_TEXT)


def h_build(sess: Session, it: nlu.Intent) -> dict:
    sess.set_source("sales")
    _apply_intent_config(sess, it, replace_filters=False)
    return render_dashboard(sess)


def h_rebuild(sess: Session, it: nlu.Intent) -> dict:
    _apply_intent_config(sess, it, replace_filters=True)
    return render_dashboard(sess, "Перестроил с учетом уточнений.")


def h_show_table(sess: Session, it: nlu.Intent) -> dict:
    if not sess.config["dimension"]:
        sess.pending_clarify = True
        return reply(sess, "clarify", CLARIFY_TEXT)
    v = sess.view()
    return reply(sess, "table", f"Таблица: {len(v)} строк, показаны первые 200.",
                 table=table_payload(v, sess.config))


def h_calc(sess: Session, it: nlu.Intent) -> dict:
    if not sess.config["dimension"]:
        sess.pending_clarify = True
        return reply(sess, "clarify", CLARIFY_TEXT)
    v = sess.view()
    try:
        res = calc.apply_preset(v, it.calc_query, sess.config["dimension"])
    except calc.CalcError as exc:
        return reply(sess, "text", str(exc))
    if not res:
        return reply(sess, "text", "Не распознал вычисление. Примеры: добавь колонку с процентом "
                                   "выполнения плана; посчитай долю; формула: fact_amount - plan_amount")
    sess.calc_history.append({"query": it.calc_query, "cols": res[2]})
    sess.config["metric"] = res[2][0]
    return render_dashboard(sess, f"Добавлено вычисление: {res[1]}.")


def h_discounts(sess: Session, it: nlu.Intent) -> dict:
    sess.set_source("discounts")
    sess.config["dimension"] = it.dimension or "client_name"
    if it.granularity:
        sess.config["granularity"] = it.granularity
    if it.period:
        sess.config["period"] = it.period
    if it.filters:
        sess.config["filters"] = dict(it.filters)
    return render_dashboard(sess, "Витрина скидок.")


def h_save(sess: Session, it: nlu.Intent) -> dict:
    if not sess.config["dimension"]:
        sess.pending_clarify = True
        return reply(sess, "clarify", CLARIFY_TEXT)
    store = _load_store()
    name = it.dashboard_name or f"Дашборд {len(store) + 1}"
    state = {"id": uuid.uuid4().hex[:10], "name": name, "source": sess.source,
             "config": sess.config, "calc_history": sess.calc_history,
             "updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
    store = [s for s in store if s["name"] != name] + [state]
    _save_store(store)
    sess.name = name
    return reply(sess, "text", f"Дашборд «{name}» сохранен.")


def h_list(sess: Session, it: nlu.Intent) -> dict:
    store = _load_store()
    if not store:
        return reply(sess, "text", "Сохраненных дашбордов пока нет.")
    items = [{"name": s["name"], "updated": s["updated"], "source": s["source"]} for s in store]
    return reply(sess, "list", "Ваши дашборды:", dashboards=items)


def _find_state(store: list, key: str):
    for s in store:
        if s["name"].lower() == key.lower() or s["id"] == key:
            return s
    return None


def h_open(sess: Session, it: nlu.Intent) -> dict:
    state = _find_state(_load_store(), it.dashboard_name or "")
    if not state:
        return reply(sess, "text", f"Дашборд «{it.dashboard_name}» не найден.")
    sess.set_source(state["source"])
    sess.config = dict(state["config"])
    sess.calc_history = list(state["calc_history"])
    sess.name = state["name"]
    return render_dashboard(sess, f"Открыл дашборд «{state['name']}».")


def h_delete(sess: Session, it: nlu.Intent) -> dict:
    store = _load_store()
    state = _find_state(store, it.dashboard_name or "")
    if not state:
        return reply(sess, "text", f"Дашборд «{it.dashboard_name}» не найден.")
    _save_store([s for s in store if s["id"] != state["id"]])
    return reply(sess, "text", f"Дашборд «{state['name']}» удален.")


def h_rename(sess: Session, it: nlu.Intent) -> dict:
    return reply(sess, "text", "Переименование: напишите «переименуй дашборд «Старое имя» в «Новое имя»». "
                               "Пока поддерживается удаление и повторное сохранение под новым именем.")


def h_export(sess: Session, it: nlu.Intent) -> dict:
    if not sess.config["dimension"]:
        sess.pending_clarify = True
        return reply(sess, "clarify", CLARIFY_TEXT)
    v = sess.view()
    fmt = it.export_format or "csv"
    os.makedirs(EXPORT_DIR, exist_ok=True)
    stem = re.sub(r"[^\w]+", "_", sess.name or "dashboard").strip("_")
    fname = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
    path = os.path.join(EXPORT_DIR, fname)
    frame = _export_frame(v, sess.config)
    if fmt == "csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    elif fmt == "xlsx":
        try:
            from openpyxl.utils import get_column_letter
        except ImportError:
            return reply(sess, "text", "Для Excel-экспорта установите openpyxl: pip install openpyxl")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Данные")
            sheet = writer.sheets["Данные"]
            for idx, column in enumerate(frame.columns, 1):
                width = max(len(str(column)), frame[column].astype(str).str.len().max()) + 2
                sheet.column_dimensions[get_column_letter(idx)].width = min(width, 60)
    else:
        parts = ["<html><head><meta charset='utf-8'>",
                 "<script>" + get_plotlyjs() + "</script>",
                 "<style>body{font-family:'Segoe UI',Arial,sans-serif;background:#f1f5f9;"
                 "color:#0f172a;margin:0;padding:24px;}"
                 ".wrap{max-width:1100px;margin:0 auto;}"
                 ".kpis{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0;}"
                 ".kpi{background:#fff;border:1px solid #e2e8f0;border-radius:8px;"
                 "padding:8px 14px;min-width:150px;}"
                 ".kpi b{display:block;font-size:11px;color:#64748b;font-weight:400;}"
                 ".kpi span{font-size:20px;font-weight:600;}"
                 ".summary{background:#fff;border-left:3px solid #2563eb;padding:8px 10px;"
                 "border-radius:4px;font-size:13px;margin:12px 0;}"
                 ".chart-box{background:#fff;border:1px solid #e2e8f0;border-radius:8px;"
                 "padding:6px;margin:12px 0;height:380px;}"
                 "</style></head><body><div class='wrap'>",
                 f"<h2>{sess.name or 'Дашборд'}</h2>",
                 "<div class='kpis'>"]
        for kpi in compute_kpis(v, sess.config):
            parts.append(f"<div class='kpi'><b>{kpi['label']}</b><span>{kpi['value']}</span></div>")
        parts.append("</div>")
        parts.append(f"<div class='summary'>{summarize(v, sess.config)}</div>")
        for fig_json in build_charts(v, sess.config):
            fig = go.Figure(fig_json)
            parts.append("<div class='chart-box'>")
            parts.append(fig.to_html(full_html=False, include_plotlyjs=False,
                                     config={"responsive": True, "displaylogo": False},
                                     default_width="100%", default_height="100%"))
            parts.append("</div>")
        parts.append("</div></body></html>")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
    return reply(sess, "text", f"Файл сохранен: storage/exports/{fname}", file_url=f"/files/{fname}")


HANDLERS = {
    "help": h_help,
    "build_dashboard": h_build,
    "rebuild_chart": h_rebuild,
    "show_table": h_show_table,
    "add_calc": h_calc,
    "discounts": h_discounts,
    "save_dashboard": h_save,
    "list_dashboards": h_list,
    "open_dashboard": h_open,
    "delete_dashboard": h_delete,
    "rename_dashboard": h_rename,
    "export": h_export,
}


def handle_message(session_id: str | None, text: str) -> dict:
    sess = sessions.get(session_id) if session_id else None
    if sess is None:
        sess = Session(uuid.uuid4().hex[:12])
        sessions[sess.id] = sess
    it = nlu.parse(text)
    if sess.pending_clarify and it.dimension:
        sess.config["dimension"] = it.dimension
        sess.pending_clarify = False
        return render_dashboard(sess, f"Разрез: {DIM_LABELS.get(it.dimension, it.dimension)}.")
    handler = HANDLERS.get(it.intent, h_build)
    return handler(sess, it)


def rebuild_interactive(session_id: str, controls: dict) -> dict:
    sess = sessions.get(session_id)
    if sess is None:
        return {"error": "Сессия не найдена, отправьте запрос заново."}
    toggles = controls.get("toggles") or {}
    if toggles.get("dimension"):
        sess.config["dimension"] = toggles["dimension"]
    if toggles.get("granularity"):
        sess.config["granularity"] = toggles["granularity"]
    if "metric" in toggles:
        sess.config["metric"] = toggles["metric"] or None
    sess.config["filters"] = {k: v for k, v in (controls.get("filters") or {}).items() if v}
    return render_dashboard(sess, "Обновил по фильтрам.")