# src/data_fetch.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance can return MultiIndex columns like ('Open','SPY') -> flatten to 'Open'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def fetch_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    # Explicitly set auto_adjust to avoid future default changes
    df = yf.download(
        ticker,
        start=start,
        progress=False,
        group_by="column",
        auto_adjust=False,
        threads=True,
    )

    if df is None or len(df) == 0:
        raise RuntimeError(f"No data returned for ticker={ticker!r}. Try another ticker or start date.")

    df = _flatten_columns(df)

    # Ensure Date is a real column
    df = df.reset_index()

    # Standardize column names
    rename_map = {
        "Adj_Close": "Adj_Close",
        "Adj Close": "Adj_Close",
    }
    df = df.rename(columns=rename_map)

    if "Date" not in df.columns:
        # yfinance usually gives Date on reset_index, but keep a safe fallback
        if "index" in df.columns:
            df = df.rename(columns={"index": "Date"})
        else:
            raise RuntimeError("Could not find a Date column after downloading data.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # Make numeric columns numeric
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in df.columns])

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="SPY")
    ap.add_argument("--start", type=str, default="2005-01-01")
    ap.add_argument("--out", type=str, default="data/spy.csv")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_ohlcv(args.ticker, args.start)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
