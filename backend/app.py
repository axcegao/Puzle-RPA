"""app.py - HTTP API чат-бота и раздача файлов."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from plotly.offline import get_plotlyjs
from pydantic import BaseModel, Field

import bot
import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    bot.refresh_known_values()
    yield


app = FastAPI(title="Puzzle RPA BI chat-bot", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str | None = None
    text: str


class RebuildRequest(BaseModel):
    session_id: str
    filters: dict = Field(default_factory=dict)
    toggles: dict = Field(default_factory=dict)


@app.post("/api/chat")
def chat(req: ChatRequest):
    return bot.handle_message(req.session_id, req.text)


@app.post("/api/rebuild")
def rebuild(req: RebuildRequest):
    out = bot.rebuild_interactive(req.session_id,
                                  {"filters": req.filters, "toggles": req.toggles})
    if out.get("error"):
        raise HTTPException(status_code=404, detail=out["error"])
    return out


@app.get("/files/{name}")
def files(name: str):
    if os.path.basename(name) != name:
        raise HTTPException(status_code=400, detail="Некорректное имя файла")
    path = os.path.join(bot.EXPORT_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(path, filename=name)


@app.get("/plotly.js")
def plotly_js():
    return Response(get_plotlyjs(), media_type="application/javascript")


@app.get("/")
def index():
    path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Фронтенд еще не создан")
    return FileResponse(path)