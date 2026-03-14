import numpy as np
import pandas as pd

from .backtest import extract_major_trades
from .config import StrategyConfig


def print_results(test_data: pd.DataFrame, trades_df: pd.DataFrame, cfg: StrategyConfig) -> None:
    """Print performance summary and trade log to console."""
    bh_log_ret = test_data["log_return"].sum()
    strat_log_ret = test_data["wf_return"].sum()

    print(f"\n📊 RESULTS ({cfg.strategy_start} onwards, {cfg.interval_label}):")
    print("-" * 60)

    results = [
        ("Buy & Hold (Benchmark)", bh_log_ret),
        ("Walk-Forward HMM", strat_log_ret),
    ]

    for name, log_ret in sorted(results, key=lambda item: item[1], reverse=True):
        ret_pct = (np.exp(log_ret) - 1) * 100
        final_value = cfg.initial_investment * np.exp(log_ret)
        print(f"  {name:30s}: {ret_pct:8.2f}%  (${final_value:,.0f})")

    major_df = extract_major_trades(trades_df)

    print(f"\n{'=' * 80}")
    print("📌 MAJOR REGIME CHANGES (Enter/Exit Market):")
    print(f"{'=' * 80}")

    date_width = 18 if cfg.interval not in ("1d", "1wk", "1mo") else 12

    if not major_df.empty:
        print(
            f"{'Date':<{date_width}} {'Act':<5} {'Signal':<14} {'Price':>10} "
            f"{'Shares':>10} {'Trade $':>10} {'Pos':>10} {'Portfolio':>12} {'Volume':>14}"
        )
        print("-" * (date_width + 95))

        for _, trade in major_df.iterrows():
            print(
                f"{trade['Date']:<{date_width}} {trade['Action']:<5} {trade['Signal']:<14} "
                f"${trade['Price']:>9,.2f} {trade['Shares']:>10.2f} "
                f"${trade['Trade $']:>9,.2f} {trade['Position Shares']:>10.2f} "
                f"${trade['Portfolio $']:>11,.2f} {trade['Volume (Market)']:>14}"
            )

        print("-" * (date_width + 95))

    print(f"  Major trades: {len(major_df)}  |  Total rebalances: {len(trades_df)}")

    print(f"\n{'=' * 60}")
    print("⚙️  CONFIG:")
    print(f"{'=' * 60}")
    print(f"  Ticker:          {cfg.ticker}")
    print(f"  Interval:        {cfg.interval_label}")
    print(f"  Train window:    {cfg.train_bars} bars")
    print(f"  Retrain every:   {cfg.retrain_bars} bars")
    print(f"  Min hold:        {cfg.min_hold_bars} bars")
    print(f"  Confirmation:    {cfg.confirmation_bars} bars")
    print(
        f"  Signals:         bull={cfg.signal_bull}, "
        f"neutral={cfg.signal_neutral}, bear={cfg.signal_bear}"
    )