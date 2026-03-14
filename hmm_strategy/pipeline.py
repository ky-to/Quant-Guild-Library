from typing import Tuple

import pandas as pd

from .backtest import build_trade_log, compute_returns
from .config import StrategyConfig
from .data import fetch_data
from .model import fit_hmm, label_regimes
from .plotting import plot_performance, plot_regimes, plot_trade_signals, plot_trade_tables
from .reporting import print_results
from .signals import walk_forward


def main(cfg: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full HMM strategy pipeline."""
    data = fetch_data(cfg)
    returns = data["log_return"].values.reshape(-1, 1)

    print("\n🔧 Fitting full-history HMM for visualization...")
    model = fit_hmm(returns, cfg)
    data["regime"] = model.predict(returns)
    labels, stats = label_regimes(model, returns)

    print("\n📊 Regime Statistics (Full History):")
    for regime_id, mean_ret, vol in stats:
        regime_name = [name for name, rid in labels.items() if rid == regime_id]
        regime_name = regime_name[0] if regime_name else f"regime_{regime_id}"
        print(f"  {regime_name:>10s} (id={regime_id}): mean={mean_ret:+.6f}, vol={vol:.6f}")

    wf_data = walk_forward(data, cfg)
    test_data = compute_returns(wf_data, cfg)
    trades_df = build_trade_log(test_data, cfg)

    print_results(test_data, trades_df, cfg)
    plot_regimes(data, model, cfg)
    plot_performance(test_data, cfg)
    plot_trade_signals(test_data, trades_df, cfg)
    plot_trade_tables(trades_df, cfg)

    return test_data, trades_df
