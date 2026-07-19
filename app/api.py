"""FastAPI backend for the Soccer Game Predictor.

Run:
    uvicorn app.api:app --reload
then open http://127.0.0.1:8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from src import predict as P
from src import roster_fetch

app = FastAPI(title="Soccer Game Predictor", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent / "web"


class PredictRequest(BaseModel):
    home: str
    away: str
    neutral: bool = True
    include_rosters: bool = False


@app.get("/api/teams")
def teams():
    return {"teams": P.available_teams()}


@app.post("/api/predict")
def predict(req: PredictRequest):
    known = {t["name"] for t in P.available_teams()}
    for team in (req.home, req.away):
        if team not in known:
            raise HTTPException(400, f"Unknown team: {team}")
    if req.home == req.away:
        raise HTTPException(400, "Pick two different teams.")

    pred = P.predict_match(req.home, req.away, neutral=req.neutral).to_dict()
    pred["home_flag"] = config.get_flag_code(req.home)
    pred["away_flag"] = config.get_flag_code(req.away)

    if req.include_rosters:
        pred["home_roster"] = roster_fetch.get_current_squad(req.home)
        pred["away_roster"] = roster_fetch.get_current_squad(req.away)
        pred["rosters_available"] = roster_fetch.has_key()
    return pred


@app.get("/api/roster/{team}")
def roster(team: str):
    return {
        "team": team,
        "key_configured": roster_fetch.has_key(),
        "squad": roster_fetch.get_current_squad(team),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "teams": len(P.available_teams())}


# Serve the single-page UI at the root.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
