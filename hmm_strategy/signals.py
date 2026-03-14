import numpy as np
import pandas as pd

from .config import StrategyConfig
from .model import fit_hmm, label_regimes, regime_to_signal


def smooth_signals(raw_signals: pd.Series, cfg: StrategyConfig) -> pd.Series:
    signals = raw_signals.copy()

    if cfg.confirmation_bars > 1:
        confirmed = signals.copy()
        for i in range(cfg.confirmation_bars - 1, len(signals)):
            window = signals.iloc[max(0, i - cfg.confirmation_bars + 1): i + 1]
            if window.nunique() == 1:
                confirmed.iloc[i] = window.iloc[0]
            else:
                confirmed.iloc[i] = confirmed.iloc[i - 1] if i > 0 else signals.iloc[0]
        signals = confirmed

    if cfg.min_hold_bars > 1:
        held = signals.copy()
        last_switch_idx = 0
        last_signal = signals.iloc[0]

        for i in range(1, len(signals)):
            if signals.iloc[i] != last_signal:
                if (i - last_switch_idx) >= cfg.min_hold_bars:
                    last_signal = signals.iloc[i]
                    last_switch_idx = i
                    held.iloc[i] = last_signal
                else:
                    held.iloc[i] = last_signal
            else:
                held.iloc[i] = last_signal

        signals = held

    return signals


def walk_forward(data: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    wf = data.copy()
    wf["wf_regime"] = np.nan
    wf["wf_signal_raw"] = np.nan

    n = len(wf.index)

    print(f"\n{'=' * 60}")
    print(f"🔬 WALK-FORWARD VALIDATION ({cfg.interval_label} bars)")
    print(f"{'=' * 60}")
    print(f"  Training window:  {cfg.train_bars} bars")
    print(f"  Retrain every:    {cfg.retrain_bars} bars")
    print(f"  Signal smoothing: hold={cfg.min_hold_bars}, confirm={cfg.confirmation_bars}")
    print(f"  Total data:       {n} bars")

    window_count = 0
    skipped_windows = 0

    for start_test in range(cfg.train_bars, n, cfg.retrain_bars):
        train_start = start_test - cfg.train_bars
        test_end = min(start_test + cfg.retrain_bars, n)

        train_series = wf.iloc[train_start:start_test]["log_return"]
        train_values = train_series.to_numpy(dtype=float)
        train_values = train_values[np.isfinite(train_values)]

        if len(train_values) < max(cfg.n_regimes * 20, 60):
            skipped_windows += 1
            continue

        try:
            model = fit_hmm(train_values.reshape(-1, 1), cfg, n_iter=cfg.hmm_iter_walkforward)
            labels, _ = label_regimes(model, train_values.reshape(-1, 1))
        except Exception as exc:
            print(f"  Skipping window starting {wf.index[start_test]}: {exc}")
            skipped_windows += 1
            continue

        test_series = wf.iloc[start_test:test_end]["log_return"]
        test_values = test_series.to_numpy(dtype=float)
        finite_mask = np.isfinite(test_values)

        if not finite_mask.any():
            skipped_windows += 1
            continue

        clean_test_returns = test_values[finite_mask].reshape(-1, 1)
        clean_test_index = test_series.index[finite_mask]

        test_regimes = model.predict(clean_test_returns)

        wf.loc[clean_test_index, "wf_regime"] = test_regimes
        wf.loc[clean_test_index, "wf_signal_raw"] = [
            regime_to_signal(int(regime), labels, cfg) for regime in test_regimes
        ]

        window_count += 1

    print(f"  Walk-forward windows: {window_count}")
    print(f"  Skipped windows:      {skipped_windows}")

    wf["wf_signal"] = wf["wf_signal_raw"]

    valid_raw = wf["wf_signal_raw"].dropna()
    if not valid_raw.empty:
        smoothed = smooth_signals(valid_raw, cfg)
        wf.loc[smoothed.index, "wf_signal"] = smoothed

    wf["wf_signal"] = wf["wf_signal"].ffill().fillna(0.0)
    return wf