# src/main_training.py

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

from config import STOCK_SYMBOL, START_DATE, TIME_STEP, TEST_SIZE_RATIO
from src.data import build_rf_feature_matrix, get_last_window
from src.models import get_model_path


# ---------------- Config ----------------

TICKER = STOCK_SYMBOL.upper()
END_DATE = datetime.today().strftime("%Y-%m-%d")

TEST_RATIO = float(TEST_SIZE_RATIO)
SEED = 42
N_LAGS = 10
FUTURE_DAYS = 30

np.random.seed(SEED)

Path("outputs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)


# ---------------- Helpers ----------------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def download_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,   # critical for reproducibility
        progress=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index()
    return df


def create_sequences(scaled_close: np.ndarray, time_step: int):
    X, y = [], []
    for i in range(time_step, len(scaled_close)):
        X.append(scaled_close[i - time_step : i, 0])
        y.append(scaled_close[i, 0])
    X = np.array(X).reshape(-1, time_step, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def compute_regime(close_series) -> str:
    if isinstance(close_series, pd.Series):
        s = close_series
    elif isinstance(close_series, pd.DataFrame):
        s = close_series.iloc[:, 0]
    else:
        s = pd.Series(close_series)

    s = pd.to_numeric(s, errors="coerce")

    if s.shape[0] < 252:
        return "unknown"

    last = s.iloc[-1]
    prev = s.iloc[-252]
    if pd.isna(last) or pd.isna(prev) or prev == 0:
        return "unknown"

    ret_1y = last / prev - 1.0
    if ret_1y > 0.2:
        return "bull"
    elif ret_1y < -0.2:
        return "bear"
    return "sideways"


def save_scaler(ticker: str, scaler: MinMaxScaler):
    out = Path("models") / f"{ticker.upper()}_scaler.pkl"
    joblib.dump(scaler, out)
    print(f"Saved scaler to {out}")


# ---------------- Main ----------------

def main():
    # 1) Load data
    df_raw = download_ohlcv(TICKER, START_DATE, END_DATE)
    print(f"Downloaded {len(df_raw)} rows for {TICKER}.")

    close = df_raw[["Close"]].astype("float32").values  # shape (N, 1)

    # 2) Train-only scaler fit (NO leakage)
    n_total = len(close)
    split_point_raw = int(n_total * (1 - TEST_RATIO))
    train_close = close[:split_point_raw]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_close)

    scaled_all = scaler.transform(close)  # transform full series using train-fitted scaler

    X, y = create_sequences(scaled_all, TIME_STEP)

    split_idx = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 3) Build and train LSTM
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=32,
        callbacks=[es],
        verbose=1,
    )

    # 4) Evaluate LSTM in USD
    test_pred = model.predict(X_test, verbose=0)  # scaled
    y_test_usd = scaler.inverse_transform(y_test)
    pred_usd = scaler.inverse_transform(test_pred)

    rmse_lstm = rmse(y_test_usd, pred_usd)
    print(f"LSTM RMSE (Close): {rmse_lstm:.4f}")

    # 5) Save model + scaler for API
    model_path = get_model_path(TICKER)  # models/AAPL_lstm.h5
    model.save(model_path)
    print(f"Saved LSTM model to {model_path}")

    save_scaler(TICKER, scaler)

    # 6) Plot (basic)
    Path("outputs").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))

    # Create a date index aligned to y samples:
    # y corresponds to original close indices TIME_STEP..end
    y_dates = df_raw.index[TIME_STEP:]
    test_dates = y_dates[split_idx:]  # aligned with X_test

    plt.plot(test_dates, y_test_usd.flatten(), label="Actual Close")
    plt.plot(test_dates, pred_usd.flatten(), label="LSTM Predicted Close")
    plt.title(f"{TICKER} Close Price - LSTM Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_pred_lstm.jpg")
    plt.close()

    # 7) Baselines aligned to test window (USD)
    close_series = df_raw["Close"].astype(float)

    # First test sample corresponds to original index (TIME_STEP + split_idx)
    test_start_orig = TIME_STEP + split_idx
    true_test = close_series.iloc[test_start_orig:].values.reshape(-1, 1)
    naive_pred = close_series.iloc[test_start_orig - 1 : -1].values.reshape(-1, 1)

    rmse_naive = rmse(true_test, naive_pred)
    print(f"Naive RMSE (Close): {rmse_naive:.4f}")

    ma5 = close_series.rolling(window=5).mean()
    ma5_pred = ma5.shift(1).iloc[test_start_orig:].values.reshape(-1, 1)
    rmse_ma5 = rmse(true_test, ma5_pred)
    print(f"MA_5 RMSE (Close): {rmse_ma5:.4f}")

    # 8) RandomForest regression on indicators (price)
    X_rf, y_rf = build_rf_feature_matrix(df_raw, n_lags=N_LAGS)
    split_rf = int(len(X_rf) * (1 - TEST_RATIO))
    X_rf_train, X_rf_test = X_rf[:split_rf], X_rf[split_rf:]
    y_rf_train, y_rf_test = y_rf[:split_rf], y_rf[split_rf:]

    rf_price = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        random_state=SEED,
        n_jobs=-1,
    )
    rf_price.fit(X_rf_train, y_rf_train)
    y_rf_pred = rf_price.predict(X_rf_test)

    rmse_rf = rmse(y_rf_test, y_rf_pred)
    print(f"RandomForest RMSE (Close): {rmse_rf:.4f}")

    joblib.dump(rf_price, "models/rf_price.pkl")
    print("Saved RF price model to models/rf_price.pkl")

    # 9) Save metrics
    metrics_df = pd.DataFrame(
        {
            "model": ["naive", "lstm", "random_forest", "ma_5"],
            "rmse": [rmse_naive, rmse_lstm, rmse_rf, rmse_ma5],
        }
    )
    metrics_df.to_csv("metrics.csv", index=False)
    print("Saved metrics.csv")

    # 10) Future forecasting (USD) using train-fitted scaler
    last_window = get_last_window(df_raw, TIME_STEP, scaler)  # scaled window
    future_preds_scaled = []
    curr_window = last_window.copy()

    for _ in range(FUTURE_DAYS):
        pred_scaled = model.predict(curr_window, verbose=0)
        future_preds_scaled.append(pred_scaled[0, 0])
        new_window = np.append(curr_window[0, 1:, 0], pred_scaled[0, 0])
        curr_window = new_window.reshape(1, TIME_STEP, 1)

    future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
    future_preds = scaler.inverse_transform(future_preds_scaled).flatten()

    last_date = df_raw.index[-1]
    future_dates = []
    d = last_date
    while len(future_dates) < FUTURE_DAYS:
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            future_dates.append(d)

    future_forecast_df = pd.DataFrame({"date": future_dates, "forecast_close": future_preds})
    future_forecast_df.to_csv("future_forecast_lstm.csv", index=False)
    print("Saved future_forecast_lstm.csv")

    # 11) Walk-forward RF vs naive (regime-aware RMSE)
    window_size = 252
    step_size = 126
    walk_metrics = []

    for start in range(0, len(df_raw) - window_size, step_size):
        end = start + window_size
        window_df = df_raw.iloc[start:end]
        regime = compute_regime(window_df["Close"])

        X_wf, y_wf = build_rf_feature_matrix(window_df, n_lags=N_LAGS)
        if len(X_wf) < 50:
            continue

        split_wf = int(len(X_wf) * 0.8)
        X_wf_train, X_wf_test = X_wf[:split_wf], X_wf[split_wf:]
        y_wf_train, y_wf_test = y_wf[:split_wf], y_wf[split_wf:]

        rf_wf = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=SEED,
            n_jobs=-1,
        )
        rf_wf.fit(X_wf_train, y_wf_train)
        y_wf_pred = rf_wf.predict(X_wf_test)
        rmse_wf = rmse(y_wf_test, y_wf_pred)

        walk_metrics.append(
            {
                "window_id": len(walk_metrics),
                "start_date": window_df.index[0].strftime("%Y-%m-%d"),
                "end_date": window_df.index[-1].strftime("%Y-%m-%d"),
                "regime": regime,
                "model": "random_forest",
                "rmse": rmse_wf,
            }
        )

        naive_pred_wf = window_df["Close"].shift(1).iloc[-len(y_wf_test):].values
        rmse_naive_wf = rmse(y_wf_test, naive_pred_wf)
        walk_metrics.append(
            {
                "window_id": len(walk_metrics),
                "start_date": window_df.index[0].strftime("%Y-%m-%d"),
                "end_date": window_df.index[-1].strftime("%Y-%m-%d"),
                "regime": regime,
                "model": "naive",
                "rmse": rmse_naive_wf,
            }
        )

    if walk_metrics:
        walk_df = pd.DataFrame(walk_metrics)
        walk_df.to_csv("walkforward_metrics.csv", index=False)
        print("Saved walkforward_metrics.csv")


if __name__ == "__main__":
    main()
