# app.py

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import joblib
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint

from config import START_DATE, TIME_STEP
from src.models import load_trained_model, make_forecast

app = FastAPI(
    title="Stock Forecast API",
    description="Forecast future stock prices using ML models.",
    version="1.1.0",
)


# ---------------------------
# Pydantic models
# ---------------------------

class ForecastRequest(BaseModel):
    ticker: str = Field(default="AAPL", json_schema_extra={"example": "AAPL"})
    days: conint(ge=1, le=30) = Field(default=7, json_schema_extra={"example": 7})


class ForecastPoint(BaseModel):
    date: date
    forecast: float


class ForecastResponse(BaseModel):
    ticker: str
    days: int
    points: List[ForecastPoint]


class HorizonRequest(BaseModel):
    ticker: str = Field(default="AAPL", json_schema_extra={"example": "AAPL"})
    days: conint(ge=1, le=30) = Field(default=30, json_schema_extra={"example": 30})


class HorizonResult(BaseModel):
    horizon: int
    date: date
    forecast: float


class MultiHorizonResponse(BaseModel):
    ticker: str
    requested_days: int
    best_horizon: int
    horizons: List[HorizonResult]


# ---------------------------
# Helpers
# ---------------------------

def _download_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Deterministic download: set auto_adjust explicitly.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume (no NaNs), sorted by date.
    """
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,   # critical for reproducibility
        progress=False,
    )

    if df is None or len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}.")

    # Some yfinance versions return MultiIndex columns; normalize if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing columns in yfinance data: {missing}")

    df = df[needed].dropna().sort_index()
    return df


def get_latest_data(ticker: str) -> pd.DataFrame:
    end_date = date.today().strftime("%Y-%m-%d")
    return _download_ohlcv(ticker, START_DATE, end_date)


def _scaler_path(ticker: str) -> Path:
    return Path("models") / f"{ticker.upper()}_scaler.pkl"


def load_scaler(ticker: str):
    p = _scaler_path(ticker)
    if not p.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Scaler not found at {p}. Run training to generate it.",
        )
    return joblib.load(p)


def build_latest_window_scaled(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Build the latest scaled feature window: shape (1, TIME_STEP, 1)
    """
    closes = df["Close"].astype("float32").values.reshape(-1, 1)
    if len(closes) < TIME_STEP:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data to build TIME_STEP={TIME_STEP} window.",
        )

    scaled = scaler.transform(closes)
    X_last = scaled[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    return X_last


def business_day_dates_after(last_timestamp: pd.Timestamp, n: int) -> List[date]:
    start = last_timestamp + pd.Timedelta(days=1)
    return list(pd.bdate_range(start=start, periods=n).date)


# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    print("DEBUG /forecast req:", req)

    df = get_latest_data(req.ticker)
    model = load_trained_model(req.ticker)
    scaler = load_scaler(req.ticker)

    X_last = build_latest_window_scaled(df, scaler)

    # Predict in scaled space, invert to USD
    forecasts_usd: List[float] = []
    for h in range(1, req.days + 1):
        y_hat_scaled = make_forecast(model, X_last, horizon=h)
        y_hat_usd = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])
        forecasts_usd.append(y_hat_usd)

    dates = business_day_dates_after(df.index[-1], req.days)
    points = [ForecastPoint(date=d, forecast=f) for d, f in zip(dates, forecasts_usd)]

    return ForecastResponse(ticker=req.ticker.upper(), days=req.days, points=points)


@app.post("/multi_horizon_forecast", response_model=MultiHorizonResponse)
def multi_horizon_forecast(req: HorizonRequest):
    print("DEBUG /multi_horizon_forecast req:", req)

    df = get_latest_data(req.ticker)
    model = load_trained_model(req.ticker)
    scaler = load_scaler(req.ticker)
    X_last = build_latest_window_scaled(df, scaler)

    dates = business_day_dates_after(df.index[-1], req.days)

    horizons: List[HorizonResult] = []
    for h in range(1, req.days + 1):
        y_hat_scaled = make_forecast(model, X_last, horizon=h)
        y_hat_usd = float(scaler.inverse_transform([[y_hat_scaled]])[0, 0])
        horizons.append(HorizonResult(horizon=h, date=dates[h - 1], forecast=y_hat_usd))

    best_horizon = req.days  # placeholder (later pick based on eval)
    return MultiHorizonResponse(
        ticker=req.ticker.upper(),
        requested_days=req.days,
        best_horizon=best_horizon,
        horizons=horizons,
    )


@app.get("/")
def root():
    return {
        "message": "Stock Forecast API is running.",
        "endpoints": ["/health", "/forecast", "/multi_horizon_forecast"],
    }
