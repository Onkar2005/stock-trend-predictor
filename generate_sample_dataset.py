"""
generate_sample_dataset.py
---------------------------
Creates a realistic-looking synthetic daily OHLCV dataset so the project
has something to train/run on immediately, with zero external downloads.

IMPORTANT (read this):
This is SYNTHETIC data (random-walk based), not a real stock. It exists so
you can run the whole pipeline (features -> training -> Streamlit app) right
away and see that everything works end-to-end.

For a real project submission, swap this for real data using one of:
  1. data_fetcher.py in this folder (uses the `yfinance` library, needs
     internet) - fetches real historical OHLCV for any ticker, e.g. RELIANCE.NS,
     TCS.NS, AAPL, INFY.NS, ^NSEI (Nifty 50), etc.
  2. A Kaggle dataset, e.g. search "NSE stock market data" or
     "Huge Stock Market Dataset" (Kaggle, user: borismarjanovic) which has
     OHLCV CSVs for thousands of stocks.
  3. Your college / exchange-provided historical data dump.

Whatever source you use, just make sure the CSV has these columns:
    Date, Open, High, Low, Close, Volume
and point train_model.py at it (see README.md).
"""

import numpy as np
import pandas as pd

def generate_synthetic_ohlcv(
    n_days: int = 1500,
    start_price: float = 500.0,
    annual_drift: float = 0.10,
    annual_vol: float = 0.28,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    dt = 1 / 252  # one trading day
    mu = annual_drift
    sigma = annual_vol

    # simulate close prices with GBM + occasional regime shocks so the
    # series has trending / mean-reverting phases (more realistic than pure GBM)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)

    # inject a few random "regime" drifts (mini bull/bear phases)
    n_regimes = 8
    regime_points = np.sort(rng.choice(np.arange(50, n_days - 50), size=n_regimes, replace=False))
    regime_bias = rng.uniform(-0.0025, 0.0025, size=n_regimes + 1)
    regime_id = np.searchsorted(regime_points, np.arange(n_days))
    log_returns = log_returns + regime_bias[regime_id]

    close = start_price * np.exp(np.cumsum(log_returns))

    # build open/high/low around each day's close using small intraday noise
    intraday_vol = sigma * np.sqrt(dt) * 0.6
    open_ = close * np.exp(rng.normal(0, intraday_vol, n_days)) * np.roll(
        np.exp(rng.normal(0, intraday_vol * 0.2, n_days)), 1
    )
    open_[0] = start_price

    high_bump = np.abs(rng.normal(0, intraday_vol, n_days))
    low_bump = np.abs(rng.normal(0, intraday_vol, n_days))
    high = np.maximum(open_, close) * (1 + high_bump)
    low = np.minimum(open_, close) * (1 - low_bump)

    # volume: base level + higher volume on big moves (realistic correlation)
    base_vol = 1_000_000
    move_size = np.abs(log_returns)
    volume = (base_vol * (1 + 8 * move_size) * rng.lognormal(0, 0.25, n_days)).astype(int)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(open_, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Volume": volume,
        }
    )

    # sanity fix: guarantee High >= max(O,C) and Low <= min(O,C)
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

    return df


if __name__ == "__main__":
    df = generate_synthetic_ohlcv()
    out_path = "data/sample_stock_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head())
    print(df.tail())
