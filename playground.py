"""
HMM Regime-Switching Trading Strategy
======================================
Walk-forward validated Hidden Markov Model that detects market regimes
(bull / neutral / bear) and generates long-only trading signals.

All tunable parameters live in StrategyConfig at the top.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn.hmm import GaussianHMM
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ============================================================
# CONFIGURATION — Change these to experiment
# ============================================================

@dataclass
class StrategyConfig:
    """All tunable strategy parameters in one place."""

    # --- Asset & Date Range ---
    ticker: str = "COST"
    data_start: str = "2015-01-01"      # Start of historical data for training
    data_end: str = ""                  # End of historical data ("" = today, 
                                        #   treated as such in the code)
    strategy_start: str = "2023-01-01"  # When to begin live/test trading

    # --- Timeframe ---
    # Options: "1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"
    # Note: intraday intervals have limited history on yfinance
    #   - "1m"  → last 7 days only
    #   - "5m"  → last 60 days
    #   - "1h"  → last 730 days
    #   - "1d"  → unlimited
    interval: str = "1d"

    # --- HMM Model ---
    n_regimes: int = 3                  # Number of hidden states (bull/neutral/bear)
    covariance_type: str = "full"       # "full", "diag", "tied", "spherical"
    hmm_iter: int = 1000                # Max EM iterations for initial fit
    hmm_iter_walkforward: int = 500     # Max EM iterations during walk-forward
    random_state: int = 42

    # --- Signal Mapping (long-only) ---
    # What fraction of portfolio to allocate in each regime
    signal_bull: float = 1.0            # Bull regime  → 100% invested
    signal_neutral: float = 0.5         # Neutral      → 50% invested
    signal_bear: float = 0.0            # Bear regime  → 0% (flat / cash)

    # --- Walk-Forward Validation ---
    train_bars: int = 756               # Training window size in bars (756 ≈ 3yr daily)
    retrain_bars: int = 126             # Retrain every N bars (126 ≈ 6mo daily)

    # --- Signal Smoothing (reduce noisy flipping) ---
    min_hold_bars: int = 1              # Minimum bars to hold before allowing signal change
                                        #   1 = no smoothing (original behavior)
                                        #   5 = hold at least 5 bars before switching
    confirmation_bars: int = 1          # Regime must persist N consecutive bars to trigger
                                        #   1 = no confirmation (original behavior)
                                        #   3 = regime must be same for 3 bars to switch

    # --- Portfolio ---
    initial_investment: float = 10000.0

    # --- Display ---
    show_full_trade_log: bool = True    # Show scrollable table of ALL trades (Fig 5)

    @property
    def interval_label(self) -> str:
        """Human-readable interval name."""
        mapping = {
            "1m": "1-Minute", "5m": "5-Minute", "15m": "15-Minute",
            "30m": "30-Minute", "1h": "Hourly", "1d": "Daily",
            "1wk": "Weekly", "1mo": "Monthly",
        }
        return mapping.get(self.interval, self.interval)


# ============================================================
# DATA
# ============================================================

def fetch_data(cfg: StrategyConfig) -> pd.DataFrame:
    """Download price data and compute log returns."""
    end_date = cfg.data_end if cfg.data_end else pd.Timestamp.now().strftime("%Y-%m-%d")
    print(f"📥 Downloading {cfg.ticker} ({cfg.interval_label}) "
          f"from {cfg.data_start} to {end_date}...")

    data = yf.download(
        cfg.ticker,
        start=cfg.data_start,
        end=end_date,
        interval=cfg.interval,
    )

    if data.empty:
        raise ValueError(f"No data returned for {cfg.ticker} with interval={cfg.interval}")

    # Compute log returns
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
    data.dropna(inplace=True)

    # Flatten MultiIndex columns from yfinance (e.g. ('Close', 'COST') → float)
    data["price"] = data["Close"].values.flatten().astype(float)
    if "Volume" in data.columns:
        data["vol"] = data["Volume"].values.flatten().astype(float)
    else:
        data["vol"] = 0.0

    print(f"  → {len(data)} bars loaded ({data.index[0]} to {data.index[-1]})")
    return data


# ============================================================
# HMM MODEL
# ============================================================

def fit_hmm(returns: np.ndarray, cfg: StrategyConfig, n_iter: Optional[int] = None) -> GaussianHMM:
    """Fit a Gaussian HMM to return data."""
    model = GaussianHMM(
        n_components=cfg.n_regimes,
        covariance_type=cfg.covariance_type,
        n_iter=n_iter or cfg.hmm_iter,
        random_state=cfg.random_state,
    )
    model.fit(returns)
    return model


def label_regimes(model: GaussianHMM, returns: np.ndarray) -> Tuple[Dict[str, int], List[Tuple[int, float, float]]]:
    """
    Predict regimes and sort by mean return to assign labels.

    Returns:
        label_map: {"bear": id, "neutral": id, "bull": id}
        regime_stats: [(id, mean, vol), ...] sorted by mean ascending
    """
    predictions = model.predict(returns)
    stats = []
    for i in range(model.n_components):
        mask = predictions == i
        mean_ret = returns[mask].mean() if mask.sum() > 0 else 0.0
        vol = returns[mask].std() if mask.sum() > 1 else 0.0
        stats.append((i, mean_ret, vol))

    stats = sorted(stats, key=lambda x: x[1])

    # For 3 regimes: bear, neutral, bull (lowest to highest mean)
    # For 2 regimes: bear, bull
    # For 4+: bear, ...neutrals..., bull
    labels: Dict[str, int] = {}
    labels["bear"] = stats[0][0]
    labels["bull"] = stats[-1][0]
    if len(stats) == 3:
        labels["neutral"] = stats[1][0]
    elif len(stats) > 3:
        for j, (rid, _, _) in enumerate(stats[1:-1], start=1):
            labels[f"neutral_{j}"] = rid

    return labels, stats


def regime_to_signal(regime: int, labels: Dict[str, int], cfg: StrategyConfig) -> float:
    """Map a regime ID to allocation signal using config."""
    if regime == labels["bull"]:
        return cfg.signal_bull
    elif regime == labels.get("neutral"):
        return cfg.signal_neutral
    elif regime == labels["bear"]:
        return cfg.signal_bear
    else:
        # Additional neutral regimes → use neutral signal
        return cfg.signal_neutral


# ============================================================
# SIGNAL SMOOTHING
# ============================================================

def smooth_signals(raw_signals: pd.Series, cfg: StrategyConfig) -> pd.Series:
    """
    Apply signal smoothing to reduce noisy flipping.

    1. confirmation_bars: regime must persist N bars before we switch
    2. min_hold_bars: once we switch, hold for at least N bars
    """
    signals = raw_signals.copy()

    # --- Confirmation filter ---
    if cfg.confirmation_bars > 1:
        confirmed = signals.copy()
        for i in range(cfg.confirmation_bars - 1, len(signals)):
            window = signals.iloc[max(0, i - cfg.confirmation_bars + 1):i + 1]
            if window.nunique() == 1:
                confirmed.iloc[i] = window.iloc[0]
            else:
                # Not confirmed → keep previous confirmed signal
                confirmed.iloc[i] = confirmed.iloc[i - 1] if i > 0 else signals.iloc[0]
        signals = confirmed

    # --- Minimum hold filter ---
    if cfg.min_hold_bars > 1:
        held = signals.copy()
        last_switch_idx = 0
        last_signal = signals.iloc[0]
        for i in range(1, len(signals)):
            if signals.iloc[i] != last_signal:
                if (i - last_switch_idx) >= cfg.min_hold_bars:
                    # Allowed to switch
                    last_signal = signals.iloc[i]
                    last_switch_idx = i
                    held.iloc[i] = last_signal
                else:
                    # Forced to hold
                    held.iloc[i] = last_signal
            else:
                held.iloc[i] = last_signal
        signals = held

    return signals


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward(data: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Walk-forward validation: train on past, predict forward, retrain.
    Returns data with 'wf_regime', 'wf_signal_raw', 'wf_signal' columns added.
    """
    wf = data.copy()
    wf["wf_regime"] = np.nan
    wf["wf_signal_raw"] = 0.0

    dates = wf.index.to_list()
    n = len(dates)

    print(f"\n{'='*60}")
    print(f"🔬 WALK-FORWARD VALIDATION ({cfg.interval_label} bars)")
    print(f"{'='*60}")
    print(f"  Training window:  {cfg.train_bars} bars")
    print(f"  Retrain every:    {cfg.retrain_bars} bars")
    print(f"  Signal smoothing: hold={cfg.min_hold_bars}, confirm={cfg.confirmation_bars}")
    print(f"  Total data:       {n} bars")

    window_count = 0
    for start_test in range(cfg.train_bars, n, cfg.retrain_bars):
        train_start = start_test - cfg.train_bars
        test_end = min(start_test + cfg.retrain_bars, n)

        # Train
        train_returns = wf.iloc[train_start:start_test]["log_return"].values.reshape(-1, 1)
        model = fit_hmm(train_returns, cfg, n_iter=cfg.hmm_iter_walkforward)
        labels, _ = label_regimes(model, train_returns)

        # Predict on test window (out-of-sample)
        test_returns = wf.iloc[start_test:test_end]["log_return"].values.reshape(-1, 1)
        test_regimes = model.predict(test_returns)

        # Store
        test_indices = wf.index[start_test:test_end]
        wf.loc[test_indices, "wf_regime"] = test_regimes
        for idx, regime in zip(test_indices, test_regimes):
            wf.loc[idx, "wf_signal_raw"] = regime_to_signal(regime, labels, cfg)

        window_count += 1

    print(f"  Walk-forward windows: {window_count}")

    # Apply smoothing
    mask_has_signal = wf["wf_signal_raw"].notna()
    wf["wf_signal"] = wf["wf_signal_raw"]
    if mask_has_signal.sum() > 0:
        smoothed = smooth_signals(wf.loc[mask_has_signal, "wf_signal_raw"], cfg)
        wf.loc[mask_has_signal, "wf_signal"] = smoothed

    return wf


# ============================================================
# BACKTEST
# ============================================================

def compute_returns(wf_data: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Compute strategy and buy-hold returns for the test period."""
    wf = wf_data.copy()

    # Strategy return: yesterday's signal × today's return
    wf["wf_return"] = wf["wf_signal"].shift(1) * wf["log_return"]
    wf["wf_return"] = wf["wf_return"].fillna(0)

    # Filter to strategy period
    mask = wf.index >= cfg.strategy_start
    test = wf[mask].copy()
    test["cum_strategy"] = test["wf_return"].cumsum()
    test["cum_buyhold"] = test["log_return"].cumsum()

    return test


def build_trade_log(test_data: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Build a trade-by-trade log from signal changes."""
    td = test_data.copy()
    td["prev_signal"] = td["wf_signal"].shift(1)

    signal_label = {0.0: "FLAT (0%)", 0.5: "HALF (50%)", 1.0: "FULL (100%)"}

    trades: List[dict] = []
    cash = cfg.initial_investment
    shares = 0.0

    for i in range(len(td)):
        price = float(td["price"].iloc[i])
        volume = float(td["vol"].iloc[i])
        vol_int = int(volume) if not np.isnan(volume) else 0
        sig = float(td["wf_signal"].iloc[i])
        prev = td["prev_signal"].iloc[i]
        prev_sig = float(prev) if not (isinstance(prev, float) and np.isnan(prev)) else 0.0

        if i == 0 or sig != prev_sig:
            target_value = (cash + shares * price) * sig
            trade_value = target_value - shares * price
            trade_shares = trade_value / price if price > 0 else 0

            if abs(trade_shares) > 0.001:
                action = "BUY" if trade_shares > 0 else "SELL"
                cash -= trade_shares * price
                shares += trade_shares

                # Format date: include time for intraday data
                dt_fmt = "%Y-%m-%d %H:%M" if cfg.interval not in ("1d", "1wk", "1mo") else "%Y-%m-%d"
                trades.append({
                    "Date": td.index[i].strftime(dt_fmt),
                    "Action": action,
                    "Signal": signal_label.get(sig, f"{sig:.0%}"),
                    "Price": price,
                    "Shares": abs(trade_shares),
                    "Trade $": abs(trade_shares * price),
                    "Position Shares": shares,
                    "Portfolio $": cash + shares * price,
                    "Volume (Market)": f"{vol_int:,}",
                })

    return pd.DataFrame(trades)


def extract_major_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Extract only trades involving FLAT (enter/exit market)."""
    if trades_df.empty:
        return trades_df

    major_rows = []
    prev_label = ""
    for _, t in trades_df.iterrows():
        is_to_flat = "FLAT" in t["Signal"]
        is_from_flat = "FLAT" in prev_label
        if is_to_flat or is_from_flat or prev_label == "":
            major_rows.append(t)
        prev_label = t["Signal"]

    return pd.DataFrame(major_rows)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_results(test_data: pd.DataFrame, trades_df: pd.DataFrame, cfg: StrategyConfig):
    """Print performance summary and trade log to console."""
    bh_log_ret = test_data["log_return"].sum()
    strat_log_ret = test_data["wf_return"].sum()

    print(f"\n📊 RESULTS ({cfg.strategy_start} onwards, {cfg.interval_label}):")
    print("-" * 60)
    for name, lr in sorted(
        [("Buy & Hold (Benchmark)", bh_log_ret), ("Walk-Forward HMM", strat_log_ret)],
        key=lambda x: x[1], reverse=True
    ):
        ret_pct = (np.exp(lr) - 1) * 100
        final = cfg.initial_investment * np.exp(lr)
        print(f"  {name:30s}: {ret_pct:8.2f}%  (${final:,.0f})")

    # Major trades
    major_df = extract_major_trades(trades_df)
    print(f"\n{'='*80}")
    print(f"📌 MAJOR REGIME CHANGES (Enter/Exit Market):")
    print(f"{'='*80}")

    date_w = 18 if cfg.interval not in ("1d", "1wk", "1mo") else 12
    if not major_df.empty:
        print(f"{'Date':<{date_w}} {'Act':<5} {'Signal':<14} {'Price':>10} {'Shares':>10} "
              f"{'Trade $':>10} {'Pos':>10} {'Portfolio':>12} {'Volume':>14}")
        print("-" * (date_w + 95))
        for _, t in major_df.iterrows():
            print(f"{t['Date']:<{date_w}} {t['Action']:<5} {t['Signal']:<14} "
                  f"${t['Price']:>9,.2f} {t['Shares']:>10.2f} "
                  f"${t['Trade $']:>9,.2f} {t['Position Shares']:>10.2f} "
                  f"${t['Portfolio $']:>11,.2f} {t['Volume (Market)']:>14}")
        print("-" * (date_w + 95))
    print(f"  Major trades: {len(major_df)}  |  Total rebalances: {len(trades_df)}")

    print(f"\n{'='*60}")
    print(f"⚙️  CONFIG:")
    print(f"{'='*60}")
    print(f"  Ticker:          {cfg.ticker}")
    print(f"  Interval:        {cfg.interval_label}")
    print(f"  Train window:    {cfg.train_bars} bars")
    print(f"  Retrain every:   {cfg.retrain_bars} bars")
    print(f"  Min hold:        {cfg.min_hold_bars} bars")
    print(f"  Confirmation:    {cfg.confirmation_bars} bars")
    print(f"  Signals:         bull={cfg.signal_bull}, neutral={cfg.signal_neutral}, bear={cfg.signal_bear}")


# ============================================================
# VISUALIZATION
# ============================================================

def plot_regimes(data: pd.DataFrame, model: GaussianHMM, cfg: StrategyConfig):
    """Fig 1: Full-history regime scatter on price."""
    colors = ["green", "red", "blue", "orange", "purple"]
    fig = go.Figure()
    for i in range(model.n_components):
        subset = data[data["regime"] == i]
        fig.add_trace(go.Scatter(
            x=subset.index,
            y=subset["price"],
            mode='markers',
            name=f"Regime {i}",
            marker=dict(color=colors[i % len(colors)], size=4),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    fig.update_layout(
        title=f"{cfg.ticker} Market Regimes via HMM ({cfg.interval_label})",
        xaxis_title="Date", yaxis_title="Price",
        hovermode='closest', template='plotly_white'
    )
    fig.show()


def plot_performance(test_data: pd.DataFrame, cfg: StrategyConfig):
    """Fig 2: Cumulative return comparison + signal timeline."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Walk-Forward HMM vs Buy & Hold", "Position Signal"),
        vertical_spacing=0.15
    )
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=(np.exp(test_data["cum_buyhold"]) - 1) * 100,
        name="Buy & Hold", line=dict(color='orange', width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=test_data.index,
        y=(np.exp(test_data["cum_strategy"]) - 1) * 100,
        name="Walk-Forward HMM", line=dict(color='green', width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=test_data.index, y=test_data["wf_signal"],
        name="Signal", fill='tozeroy', line=dict(color='blue')
    ), row=2, col=1)
    fig.update_layout(
        title=f"{cfg.ticker}: Walk-Forward ({cfg.interval_label}, "
              f"hold={cfg.min_hold_bars}, confirm={cfg.confirmation_bars})",
        template='plotly_white', height=700, hovermode='x unified'
    )
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Signal", row=2, col=1)
    fig.show()


def plot_trade_signals(test_data: pd.DataFrame, trades_df: pd.DataFrame, cfg: StrategyConfig):
    """Fig 3: Price with regime shading, buy/sell arrows, signal step chart, volume."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"{cfg.ticker} Price with Regime Shading & Key Signals",
            "Position Signal Over Time",
            "Volume"
        ),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.2, 0.3],
        shared_xaxes=True
    )

    # Row 1: Price line
    fig.add_trace(go.Scatter(
        x=test_data.index, y=test_data["price"],
        name="Price", line=dict(color='black', width=1.5),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Regime background shading
    bg_colors = {0.0: "rgba(255,0,0,0.15)", 0.5: "rgba(0,0,255,0.08)", 1.0: "rgba(0,180,0,0.12)"}
    label_names = {0.0: "Bear (Flat)", 0.5: "Neutral (Half)", 1.0: "Bull (Full)"}

    sig_vals = test_data["wf_signal"].values
    block_start = 0
    for i in range(1, len(sig_vals) + 1):
        if i == len(sig_vals) or sig_vals[i] != sig_vals[block_start]:
            fig.add_vrect(
                x0=test_data.index[block_start],
                x1=test_data.index[min(i, len(sig_vals) - 1)],
                fillcolor=bg_colors.get(sig_vals[block_start], "rgba(128,128,128,0.1)"),
                layer="below", line_width=0, row=1, col=1
            )
            block_start = i

    # Legend entries for regime shading
    for sv, c in {0.0: "red", 0.5: "blue", 1.0: "green"}.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            name=label_names.get(sv, f"Signal {sv}"),
            marker=dict(color=c, size=10, symbol='square'), showlegend=True
        ), row=1, col=1)

    # Major buy/sell arrows
    if not trades_df.empty:
        major_buys, major_sells = [], []
        prev = ""
        for _, t in trades_df.iterrows():
            if "FLAT" in t["Signal"]:
                major_sells.append(t)
            elif "FLAT" in prev or prev == "":
                major_buys.append(t)
            prev = t["Signal"]

        if major_buys:
            mb = pd.DataFrame(major_buys)
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(mb["Date"]), y=mb["Price"],
                mode='markers+text', name='Enter Position',
                marker=dict(symbol='triangle-up', size=16, color='green',
                           line=dict(width=2, color='darkgreen')),
                text=mb["Signal"].apply(lambda s: s.split("(")[0].strip()),
                textposition='top center',
                textfont=dict(size=9, color='green', family='Arial Black'),
                hovertemplate='<b>ENTER</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)

        if major_sells:
            ms = pd.DataFrame(major_sells)
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(ms["Date"]), y=ms["Price"],
                mode='markers+text', name='Exit to Flat',
                marker=dict(symbol='triangle-down', size=16, color='red',
                           line=dict(width=2, color='darkred')),
                text=["EXIT"] * len(ms),
                textposition='bottom center',
                textfont=dict(size=9, color='red', family='Arial Black'),
                hovertemplate='<b>EXIT</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)

    # Row 2: Signal step chart
    fig.add_trace(go.Scatter(
        x=test_data.index, y=test_data["wf_signal"],
        name="Position Signal", line=dict(color='purple', width=2, shape='hv'),
        fill='tozeroy', fillcolor='rgba(128,0,128,0.1)',
        hovertemplate='Date: %{x}<br>Signal: %{y}<extra></extra>'
    ), row=2, col=1)

    # Row 3: Volume bars colored by signal
    vol_colors_map = {0.0: "red", 0.5: "blue", 1.0: "green"}
    bar_colors = [vol_colors_map.get(s, 'gray') for s in test_data["wf_signal"]]
    fig.add_trace(go.Bar(
        x=test_data.index, y=test_data["vol"],
        name="Volume", marker_color=bar_colors, opacity=0.7,
        hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
    ), row=3, col=1)

    fig.update_layout(
        title=f"{cfg.ticker}: Trade Signals ({cfg.interval_label}, {cfg.strategy_start} onwards)",
        template='plotly_white', height=900, hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Signal", tickvals=[0, 0.5, 1.0],
                     ticktext=["Flat", "Half", "Full"], row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.show()


def plot_trade_tables(trades_df: pd.DataFrame, cfg: StrategyConfig):
    """Fig 4: Major trades table.  Fig 5 (optional): Full trade log."""
    if trades_df.empty:
        return

    def _make_table(df: pd.DataFrame) -> go.Table:
        return go.Table(
            header=dict(
                values=["Date", "Action", "Signal", "Price", "Shares",
                        "Trade Value", "Position", "Portfolio", "Mkt Volume"],
                fill_color='rgb(50, 50, 80)',
                font=dict(color='white', size=12), align='center'
            ),
            cells=dict(
                values=[
                    df["Date"], df["Action"], df["Signal"],
                    df["Price"].apply(lambda x: f"${x:,.2f}"),
                    df["Shares"].apply(lambda x: f"{x:.2f}"),
                    df["Trade $"].apply(lambda x: f"${x:,.2f}"),
                    df["Position Shares"].apply(lambda x: f"{x:.2f}"),
                    df["Portfolio $"].apply(lambda x: f"${x:,.2f}"),
                    df["Volume (Market)"],
                ],
                fill_color=[
                    ['rgba(144,238,144,0.3)' if a == 'BUY' else 'rgba(255,182,182,0.3)'
                     for a in df["Action"]]
                ] * 9,
                font=dict(size=11), align='center', height=28
            )
        )

    # Fig 4: Major trades
    major = extract_major_trades(trades_df)
    if not major.empty:
        fig4 = go.Figure(data=[_make_table(major)])
        fig4.update_layout(
            title=f"{cfg.ticker}: Key Trades — Enter/Exit ({len(major)} major)",
            template='plotly_white',
            height=max(400, 50 + len(major) * 30)
        )
        fig4.show()

    # Fig 5: Full log
    if cfg.show_full_trade_log:
        fig5 = go.Figure(data=[_make_table(trades_df)])
        fig5.update_layout(
            title=f"{cfg.ticker}: Complete Trade Log — All {len(trades_df)} rebalances",
            template='plotly_white', height=600
        )
        fig5.show()


# ============================================================
# MAIN
# ============================================================

def main(cfg: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full HMM strategy pipeline."""

    # 1. Fetch data
    data = fetch_data(cfg)
    returns = data["log_return"].values.reshape(-1, 1)

    # 2. Fit full-history HMM (for regime visualization only — NOT for trading)
    print("\n🔧 Fitting full-history HMM for visualization...")
    model = fit_hmm(returns, cfg)
    data["regime"] = model.predict(returns)
    labels, stats = label_regimes(model, returns)

    print("\n📊 Regime Statistics (Full History):")
    for rid, mean, vol in stats:
        regime_name = [k for k, v in labels.items() if v == rid]
        regime_name = regime_name[0] if regime_name else f"regime_{rid}"
        print(f"  {regime_name:>10s} (id={rid}): mean={mean:+.6f}, vol={vol:.6f}")

    # 3. Walk-forward validation
    wf_data = walk_forward(data, cfg)

    # 4. Compute returns for strategy period
    test_data = compute_returns(wf_data, cfg)

    # 5. Build trade log
    trades_df = build_trade_log(test_data, cfg)

    # 6. Print results
    print_results(test_data, trades_df, cfg)

    # 7. Visualize
    plot_regimes(data, model, cfg)
    plot_performance(test_data, cfg)
    plot_trade_signals(test_data, trades_df, cfg)
    plot_trade_tables(trades_df, cfg)

    return test_data, trades_df


# ============================================================
# RUN — Pick a config and go
# ============================================================

if __name__ == "__main__":

    # =============================================
    #  DEFAULT: Daily COST, same as original
    # =============================================
    config = StrategyConfig(
        ticker="COST",
        interval="1d",
        data_start="2015-01-01",
        data_end="",
        strategy_start="2023-01-01",
        train_bars=756,          # ~3 years of daily bars
        retrain_bars=126,        # ~6 months
        min_hold_bars=1,         # No smoothing (original behavior)
        confirmation_bars=1,     # No confirmation (original behavior)
    )

    # =============================================
    #  EXAMPLES — Uncomment any block to try it
    # =============================================

    # --- Weekly bars, longer holding ---
    # config = StrategyConfig(
    #     ticker="COST",
    #     interval="1wk",
    #     data_start="2010-01-01",
    #     data_end="2025-01-01",
    #     strategy_start="2023-01-01",
    #     train_bars=156,        # ~3 years of weekly bars
    #     retrain_bars=26,       # ~6 months
    #     min_hold_bars=4,       # Hold at least 4 weeks
    #     confirmation_bars=2,   # Regime must persist 2 weeks
    # )

    # --- Daily with smoothing (reduce daily flipping) ---
    # config = StrategyConfig(
    #     ticker="COST",
    #     interval="1d",
    #     data_start="2015-01-01",
    #     data_end="2025-01-01",
    #     strategy_start="2023-01-01",
    #     train_bars=756,
    #     retrain_bars=126,
    #     min_hold_bars=5,       # Hold at least 5 days
    #     confirmation_bars=3,   # Regime must persist 3 days
    # )

    # --- Hourly data (limited to ~2 years on yfinance) ---
    # config = StrategyConfig(
    #     ticker="COST",
    #     interval="1h",
    #     data_start="2024-01-01",
    #     data_end="2025-01-01",
    #     strategy_start="2024-06-01",
    #     train_bars=1764,       # ~3 months hourly (252d × 7 bars/day)
    #     retrain_bars=441,      # ~3 weeks
    #     min_hold_bars=7,       # Hold at least 7 hours
    #     confirmation_bars=3,
    # )

    # --- Binary signal (no half position) ---
    # config = StrategyConfig(
    #     ticker="COST",
    #     interval="1d",
    #     data_start="2015-01-01",
    #     data_end="2025-01-01",
    #     strategy_start="2023-01-01",
    #     signal_bull=1.0,
    #     signal_neutral=1.0,    # Neutral → also fully invested
    #     signal_bear=0.0,       # Only go flat in bear
    #     min_hold_bars=5,
    # )

    test_data, trades = main(config)
