import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def load_data(stock_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.sort_index(inplace=True)
    return df

def create_sequences(scaled_close: np.ndarray, time_step: int):
    X, y = [], []
    for i in range(time_step, len(scaled_close)):
        X.append(scaled_close[i - time_step:i, 0])
        y.append(scaled_close[i, 0])
    X = np.array(X).reshape(-1, time_step, 1)
    y = np.array(y)
    return X, y

def prepare_lstm_data(df: pd.DataFrame, time_step: int, test_ratio: float = 0.2):
    close = df[["Close"]].values.astype(float)
    n = len(close)
    split = int(n * (1 - test_ratio))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close[:split])

    scaled = scaler.transform(close)
    X_all, y_all = create_sequences(scaled, time_step)

    split_seq = split - time_step
    X_train, X_test = X_all[:split_seq], X_all[split_seq:]
    y_train, y_test = y_all[:split_seq], y_all[split_seq:]

    return X_train, X_test, y_train, y_test, scaler, split

def get_last_window(df: pd.DataFrame, time_step: int, scaler) -> np.ndarray:
    close = df[["Close"]].values.astype(float)
    scaled = scaler.transform(close)
    last_window = scaled[-time_step:].reshape(1, time_step, 1)
    return last_window
