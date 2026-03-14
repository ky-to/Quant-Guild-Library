from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """All tunable strategy parameters in one place."""

    # --- Asset & Date Range ---
    ticker: str = "GOOGL"
    data_start: str = "2015-01-01"
    data_end: str = ""
    strategy_start: str = "2023-01-01"
    data_folder: str = r"D:\playground\Quant-Guild-Library\GOOGL_2018-05-01T09-30-00_to_2026-03-14T16-00-00"

    # --- Timeframe ---
    interval: str = "1d"

    # --- HMM Model ---
    n_regimes: int = 3
    covariance_type: str = "diag"
    hmm_iter: int = 1000
    hmm_iter_walkforward: int = 500
    random_state: int = 42

    # --- Signal Mapping (long-only) ---
    signal_bull: float = 1.0
    signal_neutral: float = 0.5
    signal_bear: float = 0.0

    # --- Walk-Forward Validation ---
    train_bars: int = 756
    retrain_bars: int = 126

    # --- Signal Smoothing ---
    min_hold_bars: int = 1
    confirmation_bars: int = 1

    # --- Portfolio ---
    initial_investment: float = 10000.0

    # --- Display ---
    show_full_trade_log: bool = True

    @property
    def interval_label(self) -> str:
        mapping = {
            "1m": "1-Minute",
            "5m": "5-Minute",
            "15m": "15-Minute",
            "30m": "30-Minute",
            "1h": "Hourly",
            "1d": "Daily",
            "1wk": "Weekly",
            "1mo": "Monthly",
        }
        return mapping.get(self.interval, self.interval)