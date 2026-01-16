from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint

from config import START_DATE, TIME_STEP
from src.data import load_data, get_last_window
from src.models import load_trained_model, make_forecast
from src.rag_copilot.qa import answer_question

app = FastAPI(
    title="Stock Forecast API",
    description="Production-style stock forecasting API (LSTM + baselines) with RAG copilot.",
    version="1.0.0",
)

class ForecastRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    days: conint(ge=1, le=30) = Field(..., examples=[5])

class ForecastPoint(BaseModel):
    date: str
    forecast: float

class ForecastResponse(BaseModel):
    ticker: str
    days: int
    points: List[ForecastPoint]

class HorizonRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    days: conint(ge=1, le=30) = Field(..., examples=[30])

class HorizonResult(BaseModel):
    horizon: int
    date: str
    forecast: float

class MultiHorizonResponse(BaseModel):
    ticker: str
    requested_days: int
    best_horizon: int
    horizons: List[HorizonResult]

class AskRequest(BaseModel):
    ticker: str = Field(..., examples=["AAPL"])
    question: str = Field(..., examples=["What is the goal of this project?"])
    k: conint(ge=1, le=12) = Field(6, examples=[6])

def _get_latest_data(ticker: str) -> pd.DataFrame:
    end_date = date.today().strftime("%Y-%m-%d")
    df = load_data(ticker, START_DATE, end_date)
    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}.")
    return df

def _get_scaler(ticker: str):
    p = f"models/{ticker.upper()}_scaler.pkl"
    try:
        return joblib.load(p)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler not found or failed to load at {p}. Train first. Error: {e}")

def _business_dates(start_dt: pd.Timestamp, n: int) -> List[str]:
    start_dt = pd.to_datetime(start_dt).normalize()
    dr = pd.bdate_range(start=start_dt, periods=int(n))
    return [d.strftime("%Y-%m-%d") for d in dr]

def _inverse_scale_price(scaler, y_hat_scaled: float) -> float:
    arr = np.array([[float(y_hat_scaled)]], dtype=float)
    return float(scaler.inverse_transform(arr)[0, 0])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    df = _get_latest_data(req.ticker)
    model = load_trained_model(req.ticker)
    scaler = _get_scaler(req.ticker)

    X_last = get_last_window(df, TIME_STEP, scaler)
    dates = _business_dates(df.index[-1], req.days)

    points: List[ForecastPoint] = []
    for h in range(1, req.days + 1):
        y_hat_scaled = make_forecast(model, X_last, horizon=h)
        y_hat_price = _inverse_scale_price(scaler, y_hat_scaled)
        points.append(ForecastPoint(date=dates[h - 1], forecast=y_hat_price))

    return ForecastResponse(ticker=req.ticker.upper(), days=int(req.days), points=points)

@app.post("/multi_horizon_forecast", response_model=MultiHorizonResponse)
def multi_horizon_forecast(req: HorizonRequest):
    df = _get_latest_data(req.ticker)
    model = load_trained_model(req.ticker)
    scaler = _get_scaler(req.ticker)

    X_last = get_last_window(df, TIME_STEP, scaler)
    dates = _business_dates(df.index[-1], req.days)

    horizons: List[HorizonResult] = []
    for h in range(1, req.days + 1):
        y_hat_scaled = make_forecast(model, X_last, horizon=h)
        y_hat_price = _inverse_scale_price(scaler, y_hat_scaled)
        horizons.append(HorizonResult(horizon=h, date=dates[h - 1], forecast=y_hat_price))

    return MultiHorizonResponse(
        ticker=req.ticker.upper(),
        requested_days=int(req.days),
        best_horizon=int(req.days),
        horizons=horizons,
    )

@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    try:
        return answer_question(question=req.question, ticker=req.ticker, k=int(req.k))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG /ask failed: {e}")

@app.get("/")
def root():
    return {"message": "Stock Forecast API is running.", "endpoints": ["/health", "/forecast", "/multi_horizon_forecast", "/ask"]}
