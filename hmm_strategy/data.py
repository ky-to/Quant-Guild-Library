from pathlib import Path

import numpy as np
import pandas as pd

from .config import StrategyConfig


def _find_column(columns, candidates):
    lookup = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    raise KeyError(f"None of these columns were found: {candidates}")


def _resample_to_interval(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule_map = {
        "1m": None,
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1H",
        "1d": "1D",
        "1wk": "W-FRI",
        "1mo": "MS",
    }

    rule = rule_map.get(interval)
    if rule is None:
        return frame.sort_index()

    out = frame.resample(rule).agg({"price": "last", "vol": "sum"})
    out.dropna(subset=["price"], inplace=True)
    return out.sort_index()


def fetch_data(cfg: StrategyConfig) -> pd.DataFrame:
    """Read Databento CSV files from a folder and compute log returns."""
    data_dir = Path(cfg.data_folder)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in: {data_dir}")

    print(f"📥 Reading {cfg.ticker} data from {data_dir}...")
    print(f"  → Found {len(csv_files)} CSV files")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(f"All CSV files were empty in: {data_dir}")

    data = pd.concat(frames, ignore_index=True)

    ts_col = _find_column(data.columns, ["ts_event", "datetime", "timestamp", "date", "time"])
    price_col = _find_column(data.columns, ["close", "price"])

    try:
        volume_col = _find_column(data.columns, ["volume", "vol"])
    except KeyError:
        volume_col = None

    symbol_col = next((col for col in data.columns if str(col).lower() == "symbol"), None)

    result = data.copy()
    result[ts_col] = pd.to_datetime(result[ts_col], errors="coerce", utc=True)
    result[price_col] = pd.to_numeric(result[price_col], errors="coerce")

    if volume_col is not None:
        result["vol"] = pd.to_numeric(result[volume_col], errors="coerce").fillna(0.0)
    else:
        result["vol"] = 0.0

    if symbol_col is not None and cfg.ticker:
        result = result[result[symbol_col].astype(str).str.upper() == cfg.ticker.upper()]
        if result.empty:
            available_symbols = sorted(data[symbol_col].dropna().astype(str).str.upper().unique().tolist())
            raise ValueError(
                f"No rows found for ticker '{cfg.ticker}' in folder '{data_dir}'. "
                f"Available symbols: {available_symbols[:10]}"
            )

    result.dropna(subset=[ts_col, price_col], inplace=True)
    result = result[np.isfinite(result[price_col])]
    result = result[result[price_col] > 0]
    result.sort_values(ts_col, inplace=True)
    result.drop_duplicates(subset=[ts_col], keep="last", inplace=True)
    result.set_index(ts_col, inplace=True)

    result["price"] = result[price_col].astype(float)
    result = result[["price", "vol"]]
    result = _resample_to_interval(result, cfg.interval)

    result["log_return"] = np.log(result["price"] / result["price"].shift(1))
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result.dropna(subset=["price", "log_return"], inplace=True)

    if result.empty:
        raise ValueError(
            f"No usable rows remained after cleaning for ticker '{cfg.ticker}'. "
            f"Check the folder contents and CSV column names."
        )

    print(f"  → {len(result)} bars loaded ({result.index[0]} to {result.index[-1]})")
    return result