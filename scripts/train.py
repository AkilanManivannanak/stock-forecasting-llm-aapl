from __future__ import annotations

import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from config import STOCK_SYMBOL, START_DATE, TIME_STEP
from src.data import load_data, prepare_lstm_data
from src.models import build_lstm, get_model_path
from src.plots import plot_actual_vs_pred
from src.reports import write_experiment_report

def rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("data/kb", exist_ok=True)

    end_date = date.today().strftime("%Y-%m-%d")
    df = load_data(STOCK_SYMBOL, START_DATE, end_date)
    if df is None or len(df) == 0:
        raise SystemExit("No data downloaded. Check internet / ticker.")

    X_train, X_test, y_train, y_test, scaler, split = prepare_lstm_data(df, TIME_STEP, test_ratio=0.2)

    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    yhat_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    y_test_scaled = y_test.reshape(-1, 1)

    yhat = scaler.inverse_transform(yhat_scaled).flatten()
    y_true = scaler.inverse_transform(y_test_scaled).flatten()

    lstm_rmse = rmse(y_true, yhat)

    close = df["Close"].astype(float).to_numpy()
    close = close.reshape(-1)
    test_close = close[split:]
    naive_pred = close[split - 1 : -1]
    naive_rmse = rmse(test_close, naive_pred)

    ma5 = pd.Series(close.squeeze()).rolling(5).mean().to_numpy()
    ma5_pred = ma5[split - 1 : -1]
    ma5_rmse = rmse(test_close, ma5_pred)

    rf_rmse = np.nan
    try:
        X_rf = []
        y_rf = []
        for i in range(5, len(close) - 1):
            X_rf.append(close[i - 5 : i])
            y_rf.append(close[i + 0])
        X_rf = np.array(X_rf)
        y_rf = np.array(y_rf)
        rf_split = int(len(X_rf) * 0.8)
        rf = RandomForestRegressor(n_estimators=200, random_state=7)
        rf.fit(X_rf[:rf_split], y_rf[:rf_split])
        rf_pred = rf.predict(X_rf[rf_split:])
        rf_true = y_rf[rf_split:]
        rf_rmse = rmse(rf_true, rf_pred)
        joblib.dump(rf, "models/rf_price.pkl")
    except Exception:
        pass

    model_path = get_model_path(STOCK_SYMBOL)
    model.save(model_path)
    joblib.dump(scaler, f"models/{STOCK_SYMBOL.upper()}_scaler.pkl")

    metrics = pd.DataFrame(
        [
            {"model": "naive", "rmse": naive_rmse},
            {"model": "lstm", "rmse": lstm_rmse},
            {"model": "ma_5", "rmse": ma5_rmse},
            {"model": "random_forest", "rmse": float(rf_rmse) if rf_rmse == rf_rmse else 9999.0},
        ]
    ).sort_values("rmse")

    metrics_path = "outputs/metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    metrics.to_csv("metrics.csv", index=False)

    plot_actual_vs_pred(y_true=y_true, y_pred=yhat, plots_dir="plots")

    write_experiment_report(metrics_csv=metrics_path, out_path="data/kb/experiment_report.md")

    print(f"LSTM RMSE (Close): {lstm_rmse:.4f}")
    print(f"Naive RMSE (Close): {naive_rmse:.4f}")
    print(f"MA_5 RMSE (Close): {ma5_rmse:.4f}")
    print(f"RandomForest RMSE (Close): {rf_rmse}")
    print(f"Saved model: {model_path}")
    print("Saved scaler: models/AAPL_scaler.pkl")
    print("Saved metrics: outputs/metrics.csv and ./metrics.csv")
    print("Saved plot: plots/actual_vs_pred_lstm.png")
    print("Saved report: data/kb/experiment_report.md")

if __name__ == "__main__":
    main()
