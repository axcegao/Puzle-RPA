"""test_flow.py - смоук-тест сквозного сценария бота."""
import db
import bot

db.init()
bot.refresh_known_values()


def send(sid, text):
    r = bot.handle_message(sid, text)
    print(">>>", text)
    print("   ", r["kind"], "|", r["text"][:110])
    if r.get("kpis"):
        print("    KPI:", [(k["label"], k["value"]) for k in r["kpis"]])
    if r.get("charts"):
        print("    charts:", len(r["charts"]))
    if r.get("table"):
        print("    table rows:", r["table"]["total_rows"])
    if r.get("dashboards"):
        print("    dashboards:", [d["name"] for d in r["dashboards"]])
    return r


r = send(None, "Собери дашборд по продажам: план-факт по регионам за последний год")
sid = r["session_id"]
send(sid, "покажи таблицу")
send(sid, "добавь колонку с процентом выполнения плана")
send(sid, "перестрой график - отсортируй по проценту выполнения")
send(sid, "переключи разрез на торговые точки")
send(sid, "покажи только Центральный регион")
send(sid, "сохрани дашборд под названием «Продажи Q3»")
send(sid, "покажи мои дашборды")
send(sid, "открой дашборд «Продажи Q3»")
send(sid, "покажи скидки по клиенту Русский Оазис за последние 3 года")
send(sid, "собери дашборд")
send(sid, "по торговым точкам")
send(sid, "формула: (fact_amount - plan_amount) / plan_amount * 100 добавь колонку")
send(sid, "сделай поквартально")
send(sid, "скачай таблицу csv")
send(sid, "скачай дашборд в html")
send(sid, "что ты умеешь")

out = bot.rebuild_interactive(sid, {"filters": {"region_name": ["Центральный", "Северо-Западный"]},
                                    "toggles": {"granularity": "month"}})
print("REBUILD:", out.get("kind"), "charts:", len(out.get("charts", [])))