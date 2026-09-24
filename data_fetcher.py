"""
data_fetcher.py
----------------
OPTIONAL helper to download REAL historical OHLCV data using the
`yfinance` library, so you can train on an actual stock instead of the
synthetic sample data. Needs an internet connection and `pip install yfinance`.

Usage:
    python data_fetcher.py --ticker RELIANCE.NS --years 6
    python data_fetcher.py --ticker AAPL --years 8
    python data_fetcher.py --ticker ^NSEI --years 10        # Nifty 50 index

Common ticker formats:
    NSE (India) stocks : SYMBOL.NS   e.g. TCS.NS, INFY.NS, RELIANCE.NS
    BSE (India) stocks : SYMBOL.BO
    US stocks          : plain symbol, e.g. AAPL, MSFT, TSLA
    Indices             : ^NSEI (Nifty 50), ^BSESN (Sensex), ^GSPC (S&P 500)

Saves the result to data/sample_stock_data.csv (same filename the rest of
the project expects), so after running this you can go straight to:
    python train_model.py
"""

import argparse

import pandas as pd


def fetch(ticker: str, years: int, out_path: str = "data/sample_stock_data.csv"):
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit(
            "yfinance is not installed. Run: pip install yfinance"
        )

    period = f"{years}y"
    print(f"Downloading {ticker} ({period}) from Yahoo Finance...")
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)

    if df.empty:
        raise SystemExit(f"No data returned for ticker '{ticker}'. Check the symbol.")

    df = df.reset_index()
    # yfinance sometimes returns MultiIndex columns for a single ticker; flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={"Date": "Date"})[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="e.g. RELIANCE.NS, AAPL, ^NSEI")
    parser.add_argument("--years", type=int, default=6)
    parser.add_argument("--out", default="data/sample_stock_data.csv")
    args = parser.parse_args()
    fetch(args.ticker, args.years, args.out)
