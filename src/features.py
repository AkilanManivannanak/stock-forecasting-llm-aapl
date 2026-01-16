import numpy as np
import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["return_1d"] = data["Close"].pct_change()
    data["log_return_1d"] = np.log1p(data["return_1d"])

    data["ma_10"] = data["Close"].rolling(window=10).mean()
    data["ma_20"] = data["Close"].rolling(window=20).mean()
    data["ma_50"] = data["Close"].rolling(window=50).mean()
    data["vol_10"] = data["Close"].pct_change().rolling(window=10).std()
    data["vol_20"] = data["Close"].pct_change().rolling(window=20).std()

    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    data["macd"] = macd
    data["macd_signal"] = signal

    ma_20 = data["Close"].rolling(window=20).mean()
    std_20 = data["Close"].rolling(window=20).std()
    data["bb_upper"] = ma_20 + 2 * std_20
    data["bb_lower"] = ma_20 - 2 * std_20

    return data.dropna()
