"""
feature_engineering.py
-----------------------
Turns raw OHLCV data into the feature set the model trains on, and builds
the target label (the "upcoming trend").

All indicators are implemented from scratch with pandas/numpy only, so you
don't need TA-Lib or any extra install that can be painful to set up.

Feature list (18 features):
    Raw:            Open, High, Low, Close, Volume
    Trend:          SMA_10, SMA_20, EMA_12, EMA_26, MACD, MACD_Signal
    Momentum:       RSI_14, Stoch_%K, Stoch_%D, Momentum_10
    Volatility:     BB_Width, ATR_14, Daily_Return_Std_10

Target:
    Trend_Target -> looks FORWARD_DAYS ahead and labels the move as:
        2 = Up       (future return >  UP_THRESHOLD)
        1 = Sideways (future return between the two thresholds)
        0 = Down      (future return < DOWN_THRESHOLD)
"""

import numpy as np
import pandas as pd

FORWARD_DAYS = 5        # how many trading days ahead we're predicting the trend for
UP_THRESHOLD = 0.015     # +1.5% or more over FORWARD_DAYS  -> "Up"
DOWN_THRESHOLD = -0.015  # -1.5% or worse over FORWARD_DAYS -> "Down"
                          # anything in between -> "Sideways"

TREND_LABELS = {0: "Down", 1: "Sideways", 2: "Up"}


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return ema_fast, ema_slow, macd_line, signal_line


def _stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    percent_k = 100 * (close - lowest_low) / denom
    percent_k = percent_k.fillna(50)
    percent_d = percent_k.rolling(d_period).mean()
    return percent_k, percent_d


def _atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a DataFrame with Date, Open, High, Low, Close, Volume columns
    (sorted ascending by Date) and returns it with technical indicator
    columns appended."""
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()

    ema_12, ema_26, macd_line, signal_line = _macd(close)
    df["EMA_12"] = ema_12
    df["EMA_26"] = ema_26
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line

    df["RSI_14"] = _rsi(close, 14)

    percent_k, percent_d = _stochastic(high, low, close)
    df["Stoch_%K"] = percent_k
    df["Stoch_%D"] = percent_d

    df["Momentum_10"] = close - close.shift(10)

    sma_20 = df["SMA_20"]
    std_20 = close.rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    df["BB_Width"] = (bb_upper - bb_lower) / sma_20.replace(0, np.nan)

    df["ATR_14"] = _atr(high, low, close, 14)

    df["Daily_Return"] = close.pct_change()
    df["Daily_Return_Std_10"] = df["Daily_Return"].rolling(10).std()

    return df


def add_trend_target(
    df: pd.DataFrame,
    forward_days: int = FORWARD_DAYS,
    up_threshold: float = UP_THRESHOLD,
    down_threshold: float = DOWN_THRESHOLD,
) -> pd.DataFrame:
    """Adds Future_Return and Trend_Target (0=Down, 1=Sideways, 2=Up) columns.
    Trend_Target is NaN for the last `forward_days` rows, since the future
    isn't known yet for those."""
    df = df.copy()
    future_close = df["Close"].shift(-forward_days)
    df["Future_Return"] = (future_close - df["Close"]) / df["Close"]

    conditions = [
        df["Future_Return"] >= up_threshold,
        df["Future_Return"] <= down_threshold,
    ]
    choices = [2, 0]
    df["Trend_Target"] = np.select(conditions, choices, default=1)
    df.loc[df["Future_Return"].isna(), "Trend_Target"] = np.nan

    return df


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_20", "EMA_12", "EMA_26", "MACD", "MACD_Signal",
    "RSI_14", "Stoch_%K", "Stoch_%D", "Momentum_10",
    "BB_Width", "ATR_14", "Daily_Return_Std_10",
]


def build_feature_dataset(raw_df: pd.DataFrame):
    """Full pipeline: raw OHLCV -> indicators -> target -> clean model-ready
    DataFrame. Returns (X, y, full_df_with_features)."""
    df = add_technical_indicators(raw_df)
    df = add_trend_target(df)

    model_df = df.dropna(subset=FEATURE_COLUMNS + ["Trend_Target"]).reset_index(drop=True)

    X = model_df[FEATURE_COLUMNS]
    y = model_df["Trend_Target"].astype(int)
    return X, y, df


if __name__ == "__main__":
    raw = pd.read_csv("data/sample_stock_data.csv", parse_dates=["Date"])
    X, y, full_df = build_feature_dataset(raw)
    print("Feature matrix shape:", X.shape)
    print("Target distribution:\n", y.value_counts().rename(index=TREND_LABELS))
    print(X.head())
