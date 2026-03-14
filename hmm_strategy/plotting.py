import numpy as np
import pandas as pd
import plotly.graph_objects as go
from hmmlearn.hmm import GaussianHMM
from plotly.subplots import make_subplots

from .backtest import extract_major_trades
from .config import StrategyConfig


def plot_regimes(data: pd.DataFrame, model: GaussianHMM, cfg: StrategyConfig) -> None:
    """Fig 1: Full-history regime scatter on price."""
    colors = ["green", "red", "blue", "orange", "purple"]

    fig = go.Figure()
    for regime_id in range(model.n_components):
        subset = data[data["regime"] == regime_id]
        fig.add_trace(
            go.Scatter(
                x=subset.index,
                y=subset["price"],
                mode="markers",
                name=f"Regime {regime_id}",
                marker=dict(color=colors[regime_id % len(colors)], size=4),
                hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{cfg.ticker} Market Regimes via HMM ({cfg.interval_label})",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="closest",
        template="plotly_white",
    )
    fig.show()


def plot_performance(test_data: pd.DataFrame, cfg: StrategyConfig) -> None:
    """Fig 2: Cumulative return comparison + signal timeline."""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Walk-Forward HMM vs Buy & Hold", "Position Signal"),
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Scatter(
            x=test_data.index,
            y=(np.exp(test_data["cum_buyhold"]) - 1) * 100,
            name="Buy & Hold",
            line=dict(color="orange", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=test_data.index,
            y=(np.exp(test_data["cum_strategy"]) - 1) * 100,
            name="Walk-Forward HMM",
            line=dict(color="green", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data["wf_signal"],
            name="Signal",
            fill="tozeroy",
            line=dict(color="blue"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=(
            f"{cfg.ticker}: Walk-Forward ({cfg.interval_label}, "
            f"hold={cfg.min_hold_bars}, confirm={cfg.confirmation_bars})"
        ),
        template="plotly_white",
        height=700,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Signal", row=2, col=1)
    fig.show()


def plot_trade_signals(test_data: pd.DataFrame, trades_df: pd.DataFrame, cfg: StrategyConfig) -> None:
    """Fig 3: Price with regime shading, buy/sell arrows, signal step chart, volume."""
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            f"{cfg.ticker} Price with Regime Shading & Key Signals",
            "Position Signal Over Time",
            "Volume",
        ),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.2, 0.3],
        shared_xaxes=True,
    )

    fig.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data["price"],
            name="Price",
            line=dict(color="black", width=1.5),
            hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    bg_colors = {
        0.0: "rgba(255,0,0,0.15)",
        0.5: "rgba(0,0,255,0.08)",
        1.0: "rgba(0,180,0,0.12)",
    }
    label_names = {
        0.0: "Bear (Flat)",
        0.5: "Neutral (Half)",
        1.0: "Bull (Full)",
    }

    sig_vals = test_data["wf_signal"].values
    block_start = 0

    for i in range(1, len(sig_vals) + 1):
        if i == len(sig_vals) or sig_vals[i] != sig_vals[block_start]:
            fig.add_vrect(
                x0=test_data.index[block_start],
                x1=test_data.index[min(i, len(sig_vals) - 1)],
                fillcolor=bg_colors.get(sig_vals[block_start], "rgba(128,128,128,0.1)"),
                layer="below",
                line_width=0,
                row=1,
                col=1,
            )
            block_start = i

    for signal_value, color in {0.0: "red", 0.5: "blue", 1.0: "green"}.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=label_names.get(signal_value, f"Signal {signal_value}"),
                marker=dict(color=color, size=10, symbol="square"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    if not trades_df.empty:
        major_buys = []
        major_sells = []
        prev_signal = ""

        for _, trade in trades_df.iterrows():
            if "FLAT" in trade["Signal"]:
                major_sells.append(trade)
            elif "FLAT" in prev_signal or prev_signal == "":
                major_buys.append(trade)
            prev_signal = trade["Signal"]

        if major_buys:
            mb = pd.DataFrame(major_buys)
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(mb["Date"]),
                    y=mb["Price"],
                    mode="markers+text",
                    name="Enter Position",
                    marker=dict(
                        symbol="triangle-up",
                        size=16,
                        color="green",
                        line=dict(width=2, color="darkgreen"),
                    ),
                    text=mb["Signal"].apply(lambda value: value.split("(")[0].strip()),
                    textposition="top center",
                    textfont=dict(size=9, color="green", family="Arial Black"),
                    hovertemplate="<b>ENTER</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if major_sells:
            ms = pd.DataFrame(major_sells)
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(ms["Date"]),
                    y=ms["Price"],
                    mode="markers+text",
                    name="Exit to Flat",
                    marker=dict(
                        symbol="triangle-down",
                        size=16,
                        color="red",
                        line=dict(width=2, color="darkred"),
                    ),
                    text=["EXIT"] * len(ms),
                    textposition="bottom center",
                    textfont=dict(size=9, color="red", family="Arial Black"),
                    hovertemplate="<b>EXIT</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data["wf_signal"],
            name="Position Signal",
            line=dict(color="purple", width=2, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(128,0,128,0.1)",
            hovertemplate="Date: %{x}<br>Signal: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    vol_colors_map = {0.0: "red", 0.5: "blue", 1.0: "green"}
    bar_colors = [vol_colors_map.get(signal, "gray") for signal in test_data["wf_signal"]]

    fig.add_trace(
        go.Bar(
            x=test_data.index,
            y=test_data["vol"],
            name="Volume",
            marker_color=bar_colors,
            opacity=0.7,
            hovertemplate="Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=f"{cfg.ticker}: Trade Signals ({cfg.interval_label}, {cfg.strategy_start} onwards)",
        template="plotly_white",
        height=900,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(
        title_text="Signal",
        tickvals=[0, 0.5, 1.0],
        ticktext=["Flat", "Half", "Full"],
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.show()


def plot_trade_tables(trades_df: pd.DataFrame, cfg: StrategyConfig) -> None:
    """Fig 4: Major trades table. Fig 5 (optional): Full trade log."""
    if trades_df.empty:
        return

    def _make_table(df: pd.DataFrame) -> go.Table:
        return go.Table(
            header=dict(
                values=[
                    "Date",
                    "Action",
                    "Signal",
                    "Price",
                    "Shares",
                    "Trade Value",
                    "Position",
                    "Portfolio",
                    "Mkt Volume",
                ],
                fill_color="rgb(50, 50, 80)",
                font=dict(color="white", size=12),
                align="center",
            ),
            cells=dict(
                values=[
                    df["Date"],
                    df["Action"],
                    df["Signal"],
                    df["Price"].apply(lambda x: f"${x:,.2f}"),
                    df["Shares"].apply(lambda x: f"{x:.2f}"),
                    df["Trade $"].apply(lambda x: f"${x:,.2f}"),
                    df["Position Shares"].apply(lambda x: f"{x:.2f}"),
                    df["Portfolio $"].apply(lambda x: f"${x:,.2f}"),
                    df["Volume (Market)"],
                ],
                fill_color=[
                    [
                        "rgba(144,238,144,0.3)" if action == "BUY" else "rgba(255,182,182,0.3)"
                        for action in df["Action"]
                    ]
                ] * 9,
                font=dict(size=11),
                align="center",
                height=28,
            ),
        )

    major = extract_major_trades(trades_df)
    if not major.empty:
        fig4 = go.Figure(data=[_make_table(major)])
        fig4.update_layout(
            title=f"{cfg.ticker}: Key Trades — Enter/Exit ({len(major)} major)",
            template="plotly_white",
            height=max(400, 50 + len(major) * 30),
        )
        fig4.show()

    if cfg.show_full_trade_log:
        fig5 = go.Figure(data=[_make_table(trades_df)])
        fig5.update_layout(
            title=f"{cfg.ticker}: Complete Trade Log — All {len(trades_df)} rebalances",
            template="plotly_white",
            height=600,
        )
        fig5.show()
        