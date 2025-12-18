"""
Fetch daily OHLCV data using yfinance and save to CSV.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class FetchArgs:
    ticker: str
    start: str
    out: Path


def fetch_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for ticker={ticker!r}.")
    # normalize column names
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df.index.name = "Date"
    df = df.reset_index()
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--out", default="data/spy.csv")
    args_ns = p.parse_args()
    args = FetchArgs(ticker=args_ns.ticker, start=args_ns.start, out=Path(args_ns.out))

    df = fetch_ohlcv(args.ticker, args.start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
