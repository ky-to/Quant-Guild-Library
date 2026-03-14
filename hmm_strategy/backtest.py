from typing import List

import numpy as np
import pandas as pd

from .config import StrategyConfig


def compute_returns(wf_data: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Compute strategy and buy-hold returns for the test period."""
    wf = wf_data.copy()
    wf["wf_signal"] = wf["wf_signal"].fillna(0.0)
    wf["wf_return"] = wf["wf_signal"].shift(1).fillna(0.0) * wf["log_return"]

    mask = wf.index >= cfg.strategy_start
    test = wf[mask].copy()
    test["cum_strategy"] = test["wf_return"].cumsum()
    test["cum_buyhold"] = test["log_return"].cumsum()
    return test


def build_trade_log(test_data: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Build a trade-by-trade log from signal changes."""
    if test_data.empty:
        return pd.DataFrame()

    td = test_data.copy()
    td["prev_signal"] = td["wf_signal"].shift(1)

    signal_label = {
        0.0: "FLAT (0%)",
        0.5: "HALF (50%)",
        1.0: "FULL (100%)",
    }

    trades: List[dict] = []
    cash = cfg.initial_investment
    shares = 0.0

    for i in range(len(td)):
        price = float(td["price"].iloc[i])
        volume = float(td["vol"].iloc[i])
        vol_int = int(volume) if not np.isnan(volume) else 0
        sig = float(td["wf_signal"].iloc[i])

        prev = td["prev_signal"].iloc[i]
        prev_sig = float(prev) if not pd.isna(prev) else 0.0

        if i == 0 or sig != prev_sig:
            target_value = (cash + shares * price) * sig
            trade_value = target_value - shares * price
            trade_shares = trade_value / price if price > 0 else 0.0

            if abs(trade_shares) > 0.001:
                action = "BUY" if trade_shares > 0 else "SELL"
                cash -= trade_shares * price
                shares += trade_shares

                dt_fmt = "%Y-%m-%d %H:%M" if cfg.interval not in ("1d", "1wk", "1mo") else "%Y-%m-%d"

                trades.append(
                    {
                        "Date": td.index[i].strftime(dt_fmt),
                        "Action": action,
                        "Signal": signal_label.get(sig, f"{sig:.0%}"),
                        "Price": price,
                        "Shares": abs(trade_shares),
                        "Trade $": abs(trade_shares * price),
                        "Position Shares": shares,
                        "Portfolio $": cash + shares * price,
                        "Volume (Market)": f"{vol_int:,}",
                    }
                )

    return pd.DataFrame(trades)


def extract_major_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Extract only trades involving FLAT (enter/exit market)."""
    if trades_df.empty:
        return trades_df

    major_rows = []
    prev_label = ""

    for _, trade in trades_df.iterrows():
        is_to_flat = "FLAT" in trade["Signal"]
        is_from_flat = "FLAT" in prev_label

        if is_to_flat or is_from_flat or prev_label == "":
            major_rows.append(trade)

        prev_label = trade["Signal"]

    return pd.DataFrame(major_rows)
