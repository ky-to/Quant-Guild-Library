import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn.hmm import GaussianHMM

# Download Google stock data
# ticker = "GOOGL"
ticker = "COST"
data = yf.download(ticker, start="2015-01-01", end="2025-01-01")

# Compute log returns
data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
data.dropna(inplace=True)

returns = data["log_return"].values.reshape(-1, 1)

# Initialize HMM
hmm = GaussianHMM(
    n_components=3,        # bull / bear / sideways
    covariance_type="full",
    n_iter=1000,
    random_state=42
)

# Fit model
hmm.fit(returns)

hidden_states = hmm.predict(returns)

data["regime"] = hidden_states

for i in range(hmm.n_components):
    print(f"Regime {i}")
    print("Mean return:", hmm.means_[i][0])
    print("Volatility:", np.sqrt(hmm.covars_[i][0][0]))
    print()

colors = ["green", "red", "blue"]

fig1 = go.Figure()
for i in range(hmm.n_components):
    subset = data[data["regime"] == i]
    fig1.add_trace(go.Scatter(
        x=subset.index,
        y=subset["Close"].values.flatten(),
        mode='markers',
        name=f"Regime {i}",
        marker=dict(color=colors[i], size=5),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))

fig1.update_layout(
    title="GOOGL Market Regimes via Hidden Markov Model",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode='closest',
    template='plotly_white'
)
fig1.show()

# 1) Turn HMM Regimes into Trading Signals
# Idea
    # Each hidden state already encodes:
    # Expected return (drift)
    # Volatility
    # Persistence (transition probability)
    # We map regimes to actions.
# Example policy:
    # Bull regime → long stock or sell puts
    # Bear regime → short stock or buy puts
    # Sideways → options premium selling (iron condor / straddle)
# Signal Construction
regime_stats = []

# n_components = 3. Meaning 3 regimes (bull, bear, sideways)
# hmm.means_ is a 2D array with shape (n_components, n_features)
# This loop extracts the mean return and volatility for each of the 3 HMM regimes, 
# so you can rank them and decide which is bull/bear/neutral.
for i in range(hmm.n_components):
    mean = hmm.means_[i][0]
    vol = np.sqrt(hmm.covars_[i][0][0])
    regime_stats.append((i, mean, vol))

regime_stats

# Sort by mean return: rank them and decide which is bull/bear/neutral
regime_stats = sorted(regime_stats, key=lambda x: x[1])
bear, neutral, bull = [r[0] for r in regime_stats]

initial_investment = 10000
strategy_start_date = "2023-01-01"  # Set your strategy start date here

# ============================================
# WALK-FORWARD VALIDATION (NO LOOK-AHEAD BIAS)
# ============================================
print("\n" + "="*60)
print("🔬 WALK-FORWARD VALIDATION (TRUE OUT-OF-SAMPLE)")
print("="*60)

# Parameters
train_years = 3  # Train on 3 years of data
retrain_months = 6  # Retrain every 6 months

# Create walk-forward data
wf_data = data.copy()
wf_data["wf_regime"] = np.nan
wf_data["wf_signal"] = 0.0

# Get unique dates for walk-forward windows
dates = wf_data.index.to_list()
train_days = train_years * 252
retrain_days = int(retrain_months * 21)

print(f"\nWalk-Forward Parameters:")
print(f"  Training window: {train_years} years ({train_days} days)")
print(f"  Retrain every:   {retrain_months} months ({retrain_days} days)")
print(f"  Total data:      {len(dates)} days")

# Walk-forward loop
current_hmm = None
current_bear = None
current_neutral = None  
current_bull = None

for start_test_idx in range(train_days, len(dates), retrain_days):
    # Training period: [start_test_idx - train_days, start_test_idx)
    train_start = start_test_idx - train_days
    train_end = start_test_idx
    
    # Test period: [start_test_idx, start_test_idx + retrain_days)
    test_end = min(start_test_idx + retrain_days, len(dates))
    
    # Get training data
    train_returns = wf_data.iloc[train_start:train_end]["log_return"].values.reshape(-1, 1)
    
    # Train NEW HMM on training data only
    wf_hmm = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=500,
        random_state=42
    )
    wf_hmm.fit(train_returns)
    
    # Determine regime labels from TRAINING data only
    train_regimes = wf_hmm.predict(train_returns)
    wf_regime_stats = []
    for i in range(3):
        mask = train_regimes == i
        if mask.sum() > 0:
            mean_ret = train_returns[mask].mean()
        else:
            mean_ret = 0
        wf_regime_stats.append((i, mean_ret))
    
    wf_regime_stats = sorted(wf_regime_stats, key=lambda x: x[1])
    current_bear, current_neutral, current_bull = [r[0] for r in wf_regime_stats]
    current_hmm = wf_hmm
    
    # Now predict on TEST data (out-of-sample!)
    test_returns = wf_data.iloc[start_test_idx:test_end]["log_return"].values.reshape(-1, 1)
    test_regimes = current_hmm.predict(test_returns)
    
    # Store predictions
    test_indices = wf_data.index[start_test_idx:test_end]
    wf_data.loc[test_indices, "wf_regime"] = test_regimes
    
    # Generate signals based on current regime labels
    for idx, regime in zip(test_indices, test_regimes):
        if regime == current_bull:
            wf_data.loc[idx, "wf_signal"] = 1.0
        elif regime == current_neutral:
            wf_data.loc[idx, "wf_signal"] = 0.5
        else:  # bear
            wf_data.loc[idx, "wf_signal"] = 0.0  # Long-only, go flat in bear

# Calculate walk-forward strategy returns
wf_data["wf_return"] = wf_data["wf_signal"].shift(1) * wf_data["log_return"]
wf_data["wf_return"] = wf_data["wf_return"].fillna(0)
wf_data["cum_wf"] = wf_data["wf_return"].cumsum()

# Compare: Only look at the strategy start date onwards
test_mask = wf_data.index >= strategy_start_date

# Calculate returns only for test period
test_buy_hold = wf_data.loc[test_mask, "log_return"].sum()
test_wf_strategy = wf_data.loc[test_mask, "wf_return"].sum()

print(f"\n📊 WALK-FORWARD RESULTS (Test Period: {strategy_start_date} onwards):")
print("-" * 60)

results_comparison = {
    "Buy & Hold (Benchmark)": test_buy_hold,
    "Walk-Forward HMM": test_wf_strategy,
}

for name, log_ret in sorted(results_comparison.items(), key=lambda x: x[1], reverse=True):
    actual_ret = (np.exp(log_ret) - 1) * 100
    final_value = initial_investment * np.exp(log_ret)
    print(f"  {name:30s}: {actual_ret:8.2f}%  (${final_value:,.0f})")

# -------------------------
# Visualization
# -------------------------
fig2 = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Walk-Forward HMM vs Buy & Hold",
        "Walk-Forward Regime Detection Over Time"
    ),
    vertical_spacing=0.15
)

# Only plot test period
test_data = wf_data[test_mask].copy()

# Cumulative returns starting from 0 at test start
test_data["cum_wf_plot"] = test_data["wf_return"].cumsum()
test_data["cum_bh_plot"] = test_data["log_return"].cumsum()

fig2.add_trace(go.Scatter(
    x=test_data.index, y=(np.exp(test_data["cum_bh_plot"]) - 1) * 100,
    name="Buy & Hold", line=dict(color='orange', width=2)
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=test_data.index, y=(np.exp(test_data["cum_wf_plot"]) - 1) * 100,
    name="Walk-Forward HMM", line=dict(color='green', width=2)
), row=1, col=1)

# Regime plot
fig2.add_trace(go.Scatter(
    x=test_data.index, y=test_data["wf_signal"],
    name="Signal (WF)", fill='tozeroy', line=dict(color='blue')
), row=2, col=1)

fig2.update_layout(
    title=f"{ticker}: Walk-Forward Validation (No Look-Ahead Bias)",
    template='plotly_white',
    height=700,
    hovermode='x unified'
)
fig2.update_yaxes(title_text="Return (%)", row=1, col=1)
fig2.update_yaxes(title_text="Signal (0=Flat, 0.5=Half, 1=Long)", row=2, col=1)
fig2.show()

print("\n" + "="*60)
print("💡 KEY INSIGHTS FROM WALK-FORWARD VALIDATION:")
print("="*60)
print(f"""
1. WHAT WALK-FORWARD DOES:
   • Train HMM on past {train_years} years ONLY
   • Predict on next {retrain_months} months (unseen data)
   • Retrain and repeat - never uses future data!

2. REALISTIC EXPECTATIONS:
   • Walk-forward results are what you'd actually get
   • If WF beats buy & hold → strategy has real edge
   • If WF underperforms → strategy only works with hindsight

3. FOR REAL TRADING:
   • Always use walk-forward or expanding window validation
   • Never trust in-sample backtests
   • Consider transaction costs, slippage, and fees
""")

# ============================================
# DETAILED TRADE SIGNALS: PRICE CHART + TRADE LOG
# ============================================

# --- Build trade log from signal changes ---
test_data = wf_data[test_mask].copy()
test_data["prev_signal"] = test_data["wf_signal"].shift(1)
test_data["signal_change"] = test_data["wf_signal"] != test_data["prev_signal"]

# Get Close price as a flat Series (handle MultiIndex columns from yfinance)
close_vals = test_data["Close"].values.flatten()
test_data = test_data.copy()
test_data["price"] = close_vals

# Volume (may also be MultiIndex)
if "Volume" in test_data.columns:
    test_data["vol"] = test_data["Volume"].values.flatten().astype(float)
else:
    test_data["vol"] = 0.0

# Build detailed trade log
trades = []
position_shares = 0.0
cash = float(initial_investment)
portfolio_value = float(initial_investment)

signal_label = {0.0: "FLAT (0%)", 0.5: "HALF (50%)", 1.0: "FULL (100%)"}

for i in range(len(test_data)):
    date = test_data.index[i]
    price = float(test_data["price"].iloc[i])
    volume = int(test_data["vol"].iloc[i]) if not np.isnan(test_data["vol"].iloc[i]) else 0
    current_signal = float(test_data["wf_signal"].iloc[i])
    prev_sig = test_data["prev_signal"].iloc[i]
    prev_signal = float(prev_sig) if not (isinstance(prev_sig, float) and np.isnan(prev_sig)) else 0.0
    
    # On signal change, compute what trade happens
    if i == 0 or current_signal != prev_signal:
        # Target allocation
        target_value = (cash + position_shares * price) * current_signal
        current_stock_value = position_shares * price
        trade_value = target_value - current_stock_value
        trade_shares = trade_value / price if price > 0 else 0
        
        if abs(trade_shares) > 0.001:  # Skip negligible trades
            action = "BUY" if trade_shares > 0 else "SELL"
            cash -= trade_shares * price
            position_shares += trade_shares
            portfolio_value = cash + position_shares * price
            
            trades.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Action": action,
                "Signal": signal_label.get(current_signal, str(current_signal)),
                "Price": price,
                "Shares": abs(trade_shares),
                "Trade $": abs(trade_shares * price),
                "Position Shares": position_shares,
                "Portfolio $": portfolio_value,
                "Volume (Market)": f"{volume:,}",
            })

trades_df = pd.DataFrame(trades)

# --- Print trade log table ---
print("\n" + "="*80)
print("📋 DETAILED TRADE LOG (Every Position Change)")
print("="*80)
if len(trades_df) > 0:
    print(f"{'Date':<12} {'Action':<6} {'New Signal':<14} {'Price':>10} {'Shares':>10} "
          f"{'Trade $':>10} {'Pos Shares':>12} {'Portfolio $':>12} {'Mkt Volume':>14}")
    print("-" * 110)
    for _, t in trades_df.iterrows():
        print(f"{t['Date']:<12} {t['Action']:<6} {t['Signal']:<14} "
              f"${t['Price']:>9,.2f} {t['Shares']:>10.2f} "
              f"${t['Trade $']:>9,.2f} {t['Position Shares']:>12.2f} "
              f"${t['Portfolio $']:>11,.2f} {t['Volume (Market)']:>14}")
    print("-" * 110)
    print(f"  Total trades: {len(trades_df)}")
    final_val = cash + position_shares * float(test_data["price"].iloc[-1])
    print(f"  Final portfolio value: ${final_val:,.2f}  (started with ${initial_investment:,})")
else:
    print("  No trades generated in the test period.")

# --- Fig 3: Price chart with buy/sell arrows + volume ---
fig3 = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        f"{ticker} Price with Trade Signals",
        "Position Signal Over Time",
        "Daily Volume"
    ),
    vertical_spacing=0.08,
    row_heights=[0.5, 0.2, 0.3],
    shared_xaxes=True
)

# Row 1: Price line
fig3.add_trace(go.Scatter(
    x=test_data.index,
    y=test_data["price"],
    name="Price",
    line=dict(color='gray', width=1.5),
    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
), row=1, col=1)

# Color the price line by regime
regime_colors = {0.0: "red", 0.5: "blue", 1.0: "green"}
regime_names = {0.0: "Bear (Flat)", 0.5: "Neutral (Half)", 1.0: "Bull (Full)"}
for sig_val, color in regime_colors.items():
    mask = test_data["wf_signal"] == sig_val
    if mask.sum() > 0:
        fig3.add_trace(go.Scatter(
            x=test_data.index[mask],
            y=test_data["price"][mask],
            mode='markers',
            name=regime_names[sig_val],
            marker=dict(color=color, size=4, opacity=0.7),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<br>' + regime_names[sig_val] + '<extra></extra>'
        ), row=1, col=1)

# Add buy/sell markers from trade log
if len(trades_df) > 0:
    buys = trades_df[trades_df["Action"] == "BUY"]
    sells = trades_df[trades_df["Action"] == "SELL"]
    
    if len(buys) > 0:
        fig3.add_trace(go.Scatter(
            x=pd.to_datetime(buys["Date"]),
            y=buys["Price"],
            mode='markers+text',
            name='BUY',
            marker=dict(symbol='triangle-up', size=14, color='green', line=dict(width=1, color='darkgreen')),
            text=["BUY"] * len(buys),
            textposition='top center',
            textfont=dict(size=9, color='green'),
            hovertemplate='<b>BUY</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>'
                         + 'Shares: ' + buys["Shares"].apply(lambda x: f"{x:.2f}").values
                         + '<br>Trade: $' + buys["Trade $"].apply(lambda x: f"{x:,.2f}").values
                         + '<extra></extra>'
        ), row=1, col=1)
    
    if len(sells) > 0:
        fig3.add_trace(go.Scatter(
            x=pd.to_datetime(sells["Date"]),
            y=sells["Price"],
            mode='markers+text',
            name='SELL',
            marker=dict(symbol='triangle-down', size=14, color='red', line=dict(width=1, color='darkred')),
            text=["SELL"] * len(sells),
            textposition='bottom center',
            textfont=dict(size=9, color='red'),
            hovertemplate='<b>SELL</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>'
                         + 'Shares: ' + sells["Shares"].apply(lambda x: f"{x:.2f}").values
                         + '<br>Trade: $' + sells["Trade $"].apply(lambda x: f"{x:,.2f}").values
                         + '<extra></extra>'
        ), row=1, col=1)

# Row 2: Signal step chart
fig3.add_trace(go.Scatter(
    x=test_data.index,
    y=test_data["wf_signal"],
    name="Position Signal",
    line=dict(color='purple', width=2, shape='hv'),  # step function
    fill='tozeroy',
    fillcolor='rgba(128,0,128,0.1)',
    hovertemplate='Date: %{x}<br>Signal: %{y}<extra></extra>'
), row=2, col=1)

# Row 3: Volume bars colored by signal
vol_colors = [regime_colors.get(s, 'gray') for s in test_data["wf_signal"]]
fig3.add_trace(go.Bar(
    x=test_data.index,
    y=test_data["vol"],
    name="Volume",
    marker_color=vol_colors,
    opacity=0.7,
    hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
), row=3, col=1)

fig3.update_layout(
    title=f"{ticker}: Detailed Trade Signals with Price & Volume ({strategy_start_date} onwards)",
    template='plotly_white',
    height=900,
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig3.update_yaxes(title_text="Price ($)", row=1, col=1)
fig3.update_yaxes(title_text="Signal", tickvals=[0, 0.5, 1.0], 
                  ticktext=["Flat", "Half", "Full"], row=2, col=1)
fig3.update_yaxes(title_text="Volume", row=3, col=1)
fig3.update_xaxes(title_text="Date", row=3, col=1)
fig3.show()

# --- Fig 4: Plotly Table of trades ---
if len(trades_df) > 0:
    fig4 = go.Figure(data=[go.Table(
        header=dict(
            values=["Date", "Action", "New Signal", "Price", "Shares Traded", 
                    "Trade Value", "Position (Shares)", "Portfolio Value", "Market Volume"],
            fill_color='rgb(50, 50, 80)',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=[
                trades_df["Date"],
                trades_df["Action"],
                trades_df["Signal"],
                trades_df["Price"].apply(lambda x: f"${x:,.2f}"),
                trades_df["Shares"].apply(lambda x: f"{x:.2f}"),
                trades_df["Trade $"].apply(lambda x: f"${x:,.2f}"),
                trades_df["Position Shares"].apply(lambda x: f"{x:.2f}"),
                trades_df["Portfolio $"].apply(lambda x: f"${x:,.2f}"),
                trades_df["Volume (Market)"],
            ],
            fill_color=[
                ['rgba(144,238,144,0.3)' if a == 'BUY' else 'rgba(255,182,182,0.3)' 
                 for a in trades_df["Action"]]
            ] * 9,
            font=dict(size=11),
            align='center',
            height=28
        )
    )])
    fig4.update_layout(
        title=f"{ticker}: Complete Trade Log ({strategy_start_date} onwards) — {len(trades_df)} trades",
        template='plotly_white',
        height=max(400, 50 + len(trades_df) * 30)
    )
    fig4.show()
