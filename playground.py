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
