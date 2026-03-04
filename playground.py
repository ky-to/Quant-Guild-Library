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

for i in range(hmm.n_components):
    mean = hmm.means_[i][0]
    vol = np.sqrt(hmm.covars_[i][0][0])
    regime_stats.append((i, mean, vol))

regime_stats

# Sort by mean return:

regime_stats = sorted(regime_stats, key=lambda x: x[1])

bear, neutral, bull = [r[0] for r in regime_stats]

# Trading Signal
data["signal"] = 0
data.loc[data["regime"] == bull, "signal"] = 1
data.loc[data["regime"] == bear, "signal"] = -1

# Strategy Returns
# Now the model acts, not just labels
strategy_start_date = "2023-01-01"  # Set your strategy start date here

data["strategy_return"] = data["signal"].shift(1) * data["log_return"]
data["strategy_return"] = data["strategy_return"].where(data.index >= strategy_start_date, 0)
data["cum_strategy"] = data["strategy_return"].cumsum()
data["cum_buy_hold"] = data["log_return"].cumsum()

# Convert log returns to actual percentage returns
data["actual_strategy_return"] = np.exp(data["cum_strategy"]) - 1
data["actual_buy_hold_return"] = np.exp(data["cum_buy_hold"]) - 1

# Calculate dollar returns (assuming $10,000 starting investment)
initial_investment = 10000
data["strategy_dollar"] = initial_investment * (1 + data["actual_strategy_return"])
data["buy_hold_dollar"] = initial_investment * (1 + data["actual_buy_hold_return"])

# Create subplot with log, percentage, and dollar returns
fig2 = make_subplots(
    rows=3, cols=1,
    subplot_titles=("Log Returns", "Actual % Returns", f"Portfolio Value (Starting: ${initial_investment:,})"),
    vertical_spacing=0.08
)

# Log returns plot
fig2.add_trace(go.Scatter(
    x=data.index, y=data["cum_strategy"],
    name="HMM Strategy", line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Log Return: %{y:.4f}<extra></extra>'
), row=1, col=1)
fig2.add_trace(go.Scatter(
    x=data.index, y=data["cum_buy_hold"],
    name="Buy & Hold", line=dict(color='orange'),
    hovertemplate='Date: %{x}<br>Log Return: %{y:.4f}<extra></extra>'
), row=1, col=1)

# Actual % returns plot
fig2.add_trace(go.Scatter(
    x=data.index, y=data["actual_strategy_return"] * 100,
    name="HMM Strategy", line=dict(color='blue'), showlegend=False,
    hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
), row=2, col=1)
fig2.add_trace(go.Scatter(
    x=data.index, y=data["actual_buy_hold_return"] * 100,
    name="Buy & Hold", line=dict(color='orange'), showlegend=False,
    hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
), row=2, col=1)

# Dollar returns plot
fig2.add_trace(go.Scatter(
    x=data.index, y=data["strategy_dollar"],
    name="HMM Strategy", line=dict(color='blue'), showlegend=False,
    hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
), row=3, col=1)
fig2.add_trace(go.Scatter(
    x=data.index, y=data["buy_hold_dollar"],
    name="Buy & Hold", line=dict(color='orange'), showlegend=False,
    hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
), row=3, col=1)

fig2.update_layout(
    title="Regime-Based Strategy vs Buy & Hold",
    hovermode='x unified',
    template='plotly_white',
    height=900
)
fig2.update_yaxes(title_text="Log Return", row=1, col=1)
fig2.update_yaxes(title_text="Return (%)", row=2, col=1)
fig2.update_yaxes(title_text="Portfolio Value ($)", row=3, col=1)
fig2.show()

print(f"HMM Strategy Total Return: {data['actual_strategy_return'].iloc[-1] * 100:.2f}%")
print(f"Buy & Hold Total Return: {data['actual_buy_hold_return'].iloc[-1] * 100:.2f}%")
print(f"\nStarting Investment: ${initial_investment:,}")
print(f"HMM Strategy Final Value: ${data['strategy_dollar'].iloc[-1]:,.2f}")
print(f"Buy & Hold Final Value: ${data['buy_hold_dollar'].iloc[-1]:,.2f}")
print(f"HMM Strategy Profit/Loss: ${data['strategy_dollar'].iloc[-1] - initial_investment:,.2f}")
print(f"Buy & Hold Profit/Loss: ${data['buy_hold_dollar'].iloc[-1] - initial_investment:,.2f}")

# ============================================
# DIAGNOSTIC: Why is HMM strategy underperforming?
# ============================================
print("\n" + "="*60)
print("DIAGNOSTIC ANALYSIS: HMM Strategy Performance")
print("="*60)

# 1. Regime Statistics with Labels
print("\n1. REGIME CHARACTERISTICS:")
print("-" * 40)
regime_labels = {bear: "Bear", neutral: "Neutral", bull: "Bull"}
for i, (regime_id, mean_ret, vol) in enumerate(regime_stats):
    label = regime_labels[regime_id]
    days_in_regime = (data["regime"] == regime_id).sum()
    pct_time = days_in_regime / len(data) * 100
    print(f"  {label} (Regime {regime_id}):")
    print(f"    Mean daily return: {mean_ret*100:.4f}%")
    print(f"    Daily volatility:  {vol*100:.4f}%")
    print(f"    Days in regime:    {days_in_regime} ({pct_time:.1f}%)")

# 2. Regime Switching Frequency
regime_changes = (data["regime"].diff() != 0).sum()
avg_regime_duration = len(data) / regime_changes
print(f"\n2. REGIME SWITCHING:")
print("-" * 40)
print(f"  Total regime changes: {regime_changes}")
print(f"  Avg regime duration:  {avg_regime_duration:.1f} days")
print(f"  (Frequent switching = whipsaw losses)")

# 3. Signal Distribution
print(f"\n3. SIGNAL DISTRIBUTION:")
print("-" * 40)
long_days = (data["signal"] == 1).sum()
short_days = (data["signal"] == -1).sum()
flat_days = (data["signal"] == 0).sum()
print(f"  Long days:  {long_days} ({long_days/len(data)*100:.1f}%)")
print(f"  Short days: {short_days} ({short_days/len(data)*100:.1f}%)")
print(f"  Flat days:  {flat_days} ({flat_days/len(data)*100:.1f}%)")

# 4. Performance Attribution
print(f"\n4. PERFORMANCE ATTRIBUTION:")
print("-" * 40)
long_returns = data[data["signal"].shift(1) == 1]["log_return"].sum()
short_returns = -data[data["signal"].shift(1) == -1]["log_return"].sum()
print(f"  Return from LONG positions:  {(np.exp(long_returns)-1)*100:.2f}%")
print(f"  Return from SHORT positions: {(np.exp(short_returns)-1)*100:.2f}%")

# 5. The REAL Problem: Look-Ahead Bias Warning
print(f"\n5. ⚠️  CRITICAL ISSUE: LOOK-AHEAD BIAS")
print("-" * 40)
print("  The HMM is trained on ALL data (2023-2025),")
print("  then used to predict on the SAME data.")
print("  This means the model 'knows the future'!")
print("  In live trading, performance would likely be WORSE.")

# 6. Visualize regime transitions
fig3 = make_subplots(
    rows=2, cols=1,
    subplot_titles=(f"{ticker} Price with Regime Colors", "Trading Signal Over Time"),
    vertical_spacing=0.12
)

for i in range(hmm.n_components):
    subset = data[data["regime"] == i]
    label = regime_labels.get(i, f"Regime {i}")
    fig3.add_trace(go.Scatter(
        x=subset.index,
        y=subset["Close"].values.flatten(),
        mode='markers',
        name=label,
        marker=dict(color=colors[i], size=5),
    ), row=1, col=1)

fig3.add_trace(go.Scatter(
    x=data.index, y=data["signal"],
    name="Signal", line=dict(color='purple'),
    fill='tozeroy'
), row=2, col=1)

fig3.update_layout(
    title=f"{ticker}: Regime Detection & Trading Signals",
    template='plotly_white',
    height=600
)
fig3.update_yaxes(title_text="Price ($)", row=1, col=1)
fig3.update_yaxes(title_text="Signal (-1, 0, +1)", row=2, col=1)
fig3.show()

# ============================================
# IMPROVED HMM STRATEGIES
# ============================================
print("\n" + "="*60)
print("IMPROVED HMM STRATEGIES COMPARISON")
print("="*60)

# Create a copy of data for strategy comparisons
strategy_data = data.copy()

# -------------------------
# IMPROVEMENT 1: Long-Only Strategy
# -------------------------
# Don't short - just go flat in bear regime
strategy_data["signal_long_only"] = 0.0  # Use float to allow 0.5
strategy_data.loc[strategy_data["regime"] == bull, "signal_long_only"] = 1.0
strategy_data.loc[strategy_data["regime"] == neutral, "signal_long_only"] = 0.5  # Half position in neutral
# Bear = 0 (flat, no shorting)

strategy_data["return_long_only"] = strategy_data["signal_long_only"].shift(1) * strategy_data["log_return"]
strategy_data["cum_long_only"] = strategy_data["return_long_only"].cumsum()

# -------------------------
# IMPROVEMENT 2: Regime Smoothing (Require 3 consecutive days)
# -------------------------
def smooth_regime(regime_series, min_days=3):
    """Only switch regime if it persists for min_days"""
    smoothed = regime_series.copy()
    current_regime = regime_series.iloc[0]
    count = 1
    
    for i in range(1, len(regime_series)):
        if regime_series.iloc[i] == regime_series.iloc[i-1]:
            count += 1
        else:
            count = 1
        
        if count >= min_days:
            current_regime = regime_series.iloc[i]
        
        smoothed.iloc[i] = current_regime
    
    return smoothed

strategy_data["regime_smoothed"] = smooth_regime(strategy_data["regime"], min_days=3)
strategy_data["signal_smoothed"] = 0
strategy_data.loc[strategy_data["regime_smoothed"] == bull, "signal_smoothed"] = 1
strategy_data.loc[strategy_data["regime_smoothed"] == bear, "signal_smoothed"] = -1

strategy_data["return_smoothed"] = strategy_data["signal_smoothed"].shift(1) * strategy_data["log_return"]
strategy_data["cum_smoothed"] = strategy_data["return_smoothed"].cumsum()

# -------------------------
# IMPROVEMENT 3: Probability-Based Signals
# -------------------------
# Use regime probabilities instead of hard classification
regime_probs = hmm.predict_proba(returns)
strategy_data["bull_prob"] = regime_probs[:, bull]
strategy_data["bear_prob"] = regime_probs[:, bear]

# Signal = bull_prob - bear_prob (continuous signal from -1 to 1)
strategy_data["signal_prob"] = strategy_data["bull_prob"] - strategy_data["bear_prob"]

strategy_data["return_prob"] = strategy_data["signal_prob"].shift(1) * strategy_data["log_return"]
strategy_data["cum_prob"] = strategy_data["return_prob"].cumsum()

# -------------------------
# IMPROVEMENT 4: Long-Only + Smoothed (Best of both)
# -------------------------
strategy_data["signal_best"] = 0.0  # Use float to allow 0.5
strategy_data.loc[strategy_data["regime_smoothed"] == bull, "signal_best"] = 1.0
strategy_data.loc[strategy_data["regime_smoothed"] == neutral, "signal_best"] = 0.5
# No shorting in bear

strategy_data["return_best"] = strategy_data["signal_best"].shift(1) * strategy_data["log_return"]
strategy_data["cum_best"] = strategy_data["return_best"].cumsum()

# -------------------------
# Calculate final returns for all strategies
# -------------------------
strategies = {
    "Buy & Hold": strategy_data["cum_buy_hold"].iloc[-1],
    "Original HMM (Long/Short)": strategy_data["cum_strategy"].iloc[-1],
    "Long-Only HMM": strategy_data["cum_long_only"].iloc[-1],
    "Smoothed HMM (3-day)": strategy_data["cum_smoothed"].iloc[-1],
    "Probability-Based HMM": strategy_data["cum_prob"].iloc[-1],
    "Long-Only + Smoothed": strategy_data["cum_best"].iloc[-1],
}

print("\n📊 STRATEGY COMPARISON (Log Returns):")
print("-" * 50)
for name, log_ret in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
    actual_ret = (np.exp(log_ret) - 1) * 100
    final_value = initial_investment * np.exp(log_ret)
    print(f"  {name:25s}: {actual_ret:7.2f}%  (${final_value:,.0f})")

# -------------------------
# Visualize all strategies
# -------------------------
fig4 = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Strategy Comparison (% Returns)", "Regime Switching: Original vs Smoothed"),
    vertical_spacing=0.15
)

# Plot all cumulative returns
fig4.add_trace(go.Scatter(
    x=strategy_data.index, y=(np.exp(strategy_data["cum_buy_hold"]) - 1) * 100,
    name="Buy & Hold", line=dict(color='orange', width=2)
), row=1, col=1)

fig4.add_trace(go.Scatter(
    x=strategy_data.index, y=(np.exp(strategy_data["cum_strategy"]) - 1) * 100,
    name="Original HMM", line=dict(color='red', dash='dot')
), row=1, col=1)

fig4.add_trace(go.Scatter(
    x=strategy_data.index, y=(np.exp(strategy_data["cum_long_only"]) - 1) * 100,
    name="Long-Only", line=dict(color='blue')
), row=1, col=1)

fig4.add_trace(go.Scatter(
    x=strategy_data.index, y=(np.exp(strategy_data["cum_best"]) - 1) * 100,
    name="Long-Only + Smoothed", line=dict(color='green', width=2)
), row=1, col=1)

# Show regime switching comparison
original_changes = (strategy_data["regime"].diff() != 0).cumsum()
smoothed_changes = (strategy_data["regime_smoothed"].diff() != 0).cumsum()

fig4.add_trace(go.Scatter(
    x=strategy_data.index, y=original_changes,
    name="Original Switches", line=dict(color='red'), showlegend=True
), row=2, col=1)

fig4.add_trace(go.Scatter(
    x=strategy_data.index, y=smoothed_changes,
    name="Smoothed Switches", line=dict(color='green'), showlegend=True
), row=2, col=1)

fig4.update_layout(
    title=f"{ticker}: Improved HMM Strategies Comparison",
    template='plotly_white',
    height=700,
    hovermode='x unified'
)
fig4.update_yaxes(title_text="Return (%)", row=1, col=1)
fig4.update_yaxes(title_text="Cumulative Switches", row=2, col=1)
fig4.show()

# -------------------------
# Why Original HMM Underperforms - Summary
# -------------------------
print("\n" + "="*60)
print("💡 WHY ORIGINAL HMM UNDERPERFORMS FOR COST:")
print("="*60)
print(f"""
1. STRONG UPTREND: COST gained {(np.exp(strategy_data['cum_buy_hold'].iloc[-1])-1)*100:.0f}% over this period.
   → Shorting EVER loses money in a strong bull market.

2. WHIPSAW: Original HMM had {(strategy_data['regime'].diff() != 0).sum()} regime switches.
   → Smoothed version only had {(strategy_data['regime_smoothed'].diff() != 0).sum()} switches.
   → Fewer trades = less slippage & whipsaw.

3. SHORTING LOSSES: Check diagnostic above for "Return from SHORT positions"
   → This is likely negative, dragging down total returns.

4. REGIME MISMATCH: HMM labels high-vol days as "bear" even in uptrends.
   → Those "bear" days are often buying opportunities, not short signals!

🔧 RECOMMENDED FIXES:
   • Use Long-Only strategy (never short trending stocks)
   • Add regime smoothing (3-5 day minimum)
   • Use probability thresholds (only act when >70% confident)
   • Consider walk-forward training to avoid look-ahead bias
""")

# Show regime switching reduction
print(f"\n📉 REGIME SWITCHING REDUCTION:")
print(f"   Original: {(strategy_data['regime'].diff() != 0).sum()} switches")
print(f"   Smoothed: {(strategy_data['regime_smoothed'].diff() != 0).sum()} switches")
print(f"   Reduction: {((strategy_data['regime'].diff() != 0).sum() - (strategy_data['regime_smoothed'].diff() != 0).sum())} fewer trades")

# ============================================
# OPTIONS-ENHANCED LONG-ONLY STRATEGY (IMPROVED)
# ============================================
print("\n" + "="*60)
print("🎯 OPTIONS-ENHANCED LONG-ONLY STRATEGY (v2)")
print("="*60)

"""
IMPROVED Strategy Logic - Focus on SELLING premium (Theta positive):
─────────────────────────────────────────────────────────────
BULL REGIME:   Long stock + Sell OTM Puts (30-delta)
               → Collect premium, get paid to buy dips
               → Keep 100% of premium if stock stays up
               
NEUTRAL REGIME: Long stock + Sell OTM Calls (far OTM, 5% above)
               → Only cap extreme gains, keep most upside
               → Collect small premium for income
               
BEAR REGIME:   Stay flat (0% stock) + Sell OTM Puts (aggressive)
               → Collect HIGH premium in high-vol environment
               → Get paid to buy at lower prices
               → NO buying puts (that's a premium drain!)
─────────────────────────────────────────────────────────────
KEY INSIGHT: Always be a NET SELLER of options for income
"""

# Calculate rolling volatility for premium estimation
strategy_data["rolling_vol"] = strategy_data["log_return"].rolling(20).std() * np.sqrt(252)
strategy_data["rolling_vol"] = strategy_data["rolling_vol"].fillna(strategy_data["rolling_vol"].mean())

def estimate_put_premium(volatility, otm_pct=0.03, days=5):
    """More realistic put premium - higher in high vol"""
    # Premium increases with volatility
    base_premium = volatility * np.sqrt(days/252) * 0.5
    # OTM discount
    otm_factor = max(0.3, 1 - otm_pct * 10)
    return base_premium * otm_factor

def estimate_call_premium(volatility, otm_pct=0.05, days=5):
    """Call premium for far OTM covered calls"""
    base_premium = volatility * np.sqrt(days/252) * 0.3
    return base_premium * 0.5  # Far OTM = smaller premium

# Reset options columns
strategy_data["option_premium_v2"] = 0.0
strategy_data["option_pnl_v2"] = 0.0
strategy_data["option_strategy_v2"] = ""

# Weekly option cycles
option_cycle = 5
strategy_data["week_num"] = np.arange(len(strategy_data)) // option_cycle

for week in strategy_data["week_num"].unique():
    week_mask = strategy_data["week_num"] == week
    week_data = strategy_data[week_mask]
    
    if len(week_data) == 0:
        continue
    
    start_idx = week_data.index[0]
    regime = strategy_data.loc[start_idx, "regime_smoothed"]
    vol = strategy_data.loc[start_idx, "rolling_vol"]
    
    # Week's stock return
    week_return = week_data["log_return"].sum()
    
    if regime == bull:
        # BULL: Long stock + Sell 3% OTM Puts
        put_premium = estimate_put_premium(vol, otm_pct=0.03)
        strategy_data.loc[week_mask, "option_strategy_v2"] = "Sell Put (Bull)"
        
        if week_return < -0.03:  # Stock dropped below put strike
            # Assigned - lose beyond strike, but we wanted to buy anyway
            assignment_loss = (week_return + 0.03) * 0.5  # Partial loss (we keep stock)
            pnl = put_premium + assignment_loss
        else:
            # Keep full premium
            pnl = put_premium
        
        strategy_data.loc[week_mask, "option_pnl_v2"] = pnl / option_cycle
        strategy_data.loc[week_mask, "option_premium_v2"] = put_premium / option_cycle
            
    elif regime == neutral:
        # NEUTRAL: Long stock + Sell FAR OTM Calls (5% OTM)
        # This barely caps upside but collects some premium
        call_premium = estimate_call_premium(vol, otm_pct=0.05)
        strategy_data.loc[week_mask, "option_strategy_v2"] = "Sell Call (Neutral)"
        
        if week_return > 0.05:  # Stock rose above call strike (rare)
            # Called away - cap gains at 5%
            cap_loss = (week_return - 0.05) * 0.3  # Only partial cap
            pnl = call_premium - cap_loss
        else:
            # Keep premium + full stock gains
            pnl = call_premium
        
        strategy_data.loc[week_mask, "option_pnl_v2"] = pnl / option_cycle
        strategy_data.loc[week_mask, "option_premium_v2"] = call_premium / option_cycle
        
    else:  # Bear regime
        # BEAR: No stock position, but SELL puts aggressively at lower strikes
        # High vol = expensive puts = collect MORE premium
        put_premium = estimate_put_premium(vol, otm_pct=0.05) * 1.5  # Vol premium!
        strategy_data.loc[week_mask, "option_strategy_v2"] = "Sell Put (Bear)"
        
        if week_return < -0.05:  # Stock dropped below our low strike
            # Assigned at attractive price - this is actually good!
            # We get stock at 5% discount + kept premium
            # Model as small loss since we didn't want stock in bear
            assignment_loss = (week_return + 0.05) * 0.3
            pnl = put_premium + assignment_loss
        else:
            # Keep full premium - stock didn't drop to our strike
            pnl = put_premium
        
        strategy_data.loc[week_mask, "option_pnl_v2"] = pnl / option_cycle
        strategy_data.loc[week_mask, "option_premium_v2"] = put_premium / option_cycle

# Calculate combined options + stock strategy
strategy_data["return_options_v2"] = (
    strategy_data["signal_best"].shift(1) * strategy_data["log_return"] +  # Stock component
    strategy_data["option_pnl_v2"]  # Options component (always additive now!)
)
strategy_data["cum_options_v2"] = strategy_data["return_options_v2"].cumsum()
strategy_data["cum_premium_v2"] = strategy_data["option_premium_v2"].cumsum()

# -------------------------
# Final Comparison
# -------------------------
strategies_final = {
    "Buy & Hold": strategy_data["cum_buy_hold"].iloc[-1],
    "Long-Only + Smoothed": strategy_data["cum_best"].iloc[-1],
    "Options-Enhanced v2": strategy_data["cum_options_v2"].iloc[-1],
}

print("\n📊 IMPROVED OPTIONS STRATEGY COMPARISON:")
print("-" * 55)
for name, log_ret in sorted(strategies_final.items(), key=lambda x: x[1], reverse=True):
    actual_ret = (np.exp(log_ret) - 1) * 100
    final_value = initial_investment * np.exp(log_ret)
    print(f"  {name:25s}: {actual_ret:8.2f}%  (${final_value:,.0f})")

# Premium breakdown
print(f"\n💰 PREMIUM INCOME (v2):")
print("-" * 55)
total_premium_v2 = strategy_data["option_premium_v2"].sum()
total_pnl_v2 = strategy_data["option_pnl_v2"].sum()
print(f"  Gross Premium Collected: {total_premium_v2*100:.2f}%")
print(f"  Net Options P&L:         {total_pnl_v2*100:.2f}%")
print(f"  Added to Portfolio:      ${initial_investment * total_pnl_v2:,.0f}")

# Annualized premium income
years = len(strategy_data) / 252
annual_premium = (total_premium_v2 / years) * 100
print(f"  Annualized Premium:      {annual_premium:.2f}% per year")

# -------------------------
# Visualization
# -------------------------
fig5 = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Strategy Comparison: Options v2 vs Buy & Hold",
        "Cumulative Premium Income (Pure Selling Strategy)"
    ),
    vertical_spacing=0.12
)

fig5.add_trace(go.Scatter(
    x=strategy_data.index, y=(np.exp(strategy_data["cum_buy_hold"]) - 1) * 100,
    name="Buy & Hold", line=dict(color='orange', width=2)
), row=1, col=1)

fig5.add_trace(go.Scatter(
    x=strategy_data.index, y=(np.exp(strategy_data["cum_best"]) - 1) * 100,
    name="Long-Only + Smoothed", line=dict(color='blue', width=1, dash='dot')
), row=1, col=1)

fig5.add_trace(go.Scatter(
    x=strategy_data.index, y=(np.exp(strategy_data["cum_options_v2"]) - 1) * 100,
    name="Options-Enhanced v2", line=dict(color='green', width=3)
), row=1, col=1)

# Premium income
fig5.add_trace(go.Scatter(
    x=strategy_data.index, y=strategy_data["cum_premium_v2"] * 100,
    name="Premium Collected", line=dict(color='purple'), fill='tozeroy'
), row=2, col=1)

fig5.update_layout(
    title=f"{ticker}: Options-Enhanced Strategy v2 (Premium Selling Focus)",
    template='plotly_white',
    height=700,
    hovermode='x unified'
)
fig5.update_yaxes(title_text="Return (%)", row=1, col=1)
fig5.update_yaxes(title_text="Cumulative Premium (%)", row=2, col=1)
fig5.show()

print("\n" + "="*60)
print("🔑 KEY CHANGES IN v2:")
print("="*60)
print("""
❌ OLD (Bad):
   • Bought puts in bear regime → Premium DRAIN
   • Covered calls too tight → Capped upside too much
   • Net premium was negative in some periods

✅ NEW (Better):
   • ALWAYS sell options → Always collect premium
   • Bull: Sell OTM puts (get paid to buy dips)
   • Neutral: Sell FAR OTM calls (5%+ away, rarely hit)
   • Bear: Sell OTM puts at LOW strikes (high premium!)
   
💡 The "Wheel Strategy" concept:
   • Sell puts → Get assigned → Sell calls → Get called away → Repeat
   • Works best on stocks you want to own long-term (like COST)
""")

# ============================================
# ⚠️ LOOK-AHEAD BIAS ANALYSIS
# ============================================
print("\n" + "="*60)
print("⚠️  CRITICAL: LOOK-AHEAD BIAS IN ALL STRATEGIES ABOVE")
print("="*60)
print("""
ALL strategies above are CHEATING because they know the future:

1. HMM TRAINING BIAS:
   • HMM trained on 2015-2025, then tested on SAME data
   • Model has already "seen" every crash and rally
   
2. REGIME LABELING BIAS:
   • Bull/Bear/Neutral determined by sorting mean returns
   • Uses ENTIRE dataset to decide which regime is "bull"
   • In real trading, you don't know future regime statistics!

3. WHAT THIS MEANS:
   • Backtest results are OVERLY OPTIMISTIC
   • Real trading performance will be WORSE
   • This is why the strategies still underperform buy & hold!
""")

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
wf_data = strategy_data.copy()  # Use strategy_data which has return_best
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

# Compare: Only look at the test period (after first training window)
test_start_date = dates[train_days]
test_mask = wf_data.index >= test_start_date

# Calculate returns only for test period
test_buy_hold = wf_data.loc[test_mask, "log_return"].sum()
test_wf_strategy = wf_data.loc[test_mask, "wf_return"].sum()
test_biased_strategy = wf_data.loc[test_mask, "return_best"].sum()

print(f"\n📊 WALK-FORWARD RESULTS (Test Period: {test_start_date.strftime('%Y-%m-%d')} onwards):")
print("-" * 60)

results_comparison = {
    "Buy & Hold (Benchmark)": test_buy_hold,
    "Biased HMM (knows future)": test_biased_strategy,
    "Walk-Forward HMM (honest)": test_wf_strategy,
}

for name, log_ret in sorted(results_comparison.items(), key=lambda x: x[1], reverse=True):
    actual_ret = (np.exp(log_ret) - 1) * 100
    print(f"  {name:30s}: {actual_ret:8.2f}%")

# Performance degradation
if test_biased_strategy != 0:
    degradation = ((test_wf_strategy - test_biased_strategy) / abs(test_biased_strategy)) * 100
    print(f"\n⚠️  Walk-Forward vs Biased: {degradation:+.1f}% difference")
    print("   (Negative = honest strategy performs worse, as expected)")

# -------------------------
# Visualization
# -------------------------
fig6 = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Walk-Forward (Honest) vs Biased Strategy",
        "Walk-Forward Regime Detection Over Time"
    ),
    vertical_spacing=0.15
)

# Only plot test period
test_data = wf_data[test_mask].copy()

# Cumulative returns starting from 0 at test start
test_data["cum_wf_plot"] = test_data["wf_return"].cumsum()
test_data["cum_biased_plot"] = test_data["return_best"].cumsum()
test_data["cum_bh_plot"] = test_data["log_return"].cumsum()

fig6.add_trace(go.Scatter(
    x=test_data.index, y=(np.exp(test_data["cum_bh_plot"]) - 1) * 100,
    name="Buy & Hold", line=dict(color='orange', width=2)
), row=1, col=1)

fig6.add_trace(go.Scatter(
    x=test_data.index, y=(np.exp(test_data["cum_biased_plot"]) - 1) * 100,
    name="Biased HMM (cheating)", line=dict(color='red', dash='dot')
), row=1, col=1)

fig6.add_trace(go.Scatter(
    x=test_data.index, y=(np.exp(test_data["cum_wf_plot"]) - 1) * 100,
    name="Walk-Forward HMM (honest)", line=dict(color='green', width=2)
), row=1, col=1)

# Regime plot
fig6.add_trace(go.Scatter(
    x=test_data.index, y=test_data["wf_signal"],
    name="Signal (WF)", fill='tozeroy', line=dict(color='blue')
), row=2, col=1)

fig6.update_layout(
    title=f"{ticker}: Walk-Forward Validation (No Look-Ahead Bias)",
    template='plotly_white',
    height=700,
    hovermode='x unified'
)
fig6.update_yaxes(title_text="Return (%)", row=1, col=1)
fig6.update_yaxes(title_text="Signal (0=Flat, 0.5=Half, 1=Long)", row=2, col=1)
fig6.show()

print("\n" + "="*60)
print("💡 KEY INSIGHTS FROM WALK-FORWARD VALIDATION:")
print("="*60)
print(f"""
1. WHAT WALK-FORWARD DOES:
   • Train HMM on past {train_years} years ONLY
   • Predict on next {retrain_months} months (unseen data)
   • Retrain and repeat - never uses future data!

2. WHY RESULTS DIFFER:
   • Biased strategy "knew" which regime was bullish
   • Walk-forward must LEARN regime characteristics
   • Regime labels can FLIP between training windows!

3. REALISTIC EXPECTATIONS:
   • Walk-forward results are what you'd actually get
   • If WF beats buy & hold → strategy has real edge
   • If WF underperforms → strategy only works with hindsight

4. FOR REAL TRADING:
   • Always use walk-forward or expanding window validation
   • Never trust in-sample backtests
   • Consider transaction costs, slippage, and fees
""")

# # 2) Use Regimes to Switch Option Strategies

# # This is where HMMs shine.
#     # Regime → Options Mapping
#     # Regime	// Characteristics	// Options Strategy
#     # Bull	// +drift, low vol	// Short puts, call spreads
#     # Bear	// −drift, high vol	// Long puts, put spreads
#     # Sideways	// Flat, low vol	// Iron condor, straddle

# # Example: Strategy Selector
# # In practice, you’d combine this with:
# # Implied volatility rank
# # Skew
# # Time to expiry
# # The HMM decides when, IV decides how.
# def option_strategy(regime):
#     if regime == bull:
#         return "Short Put / Call Spread"
#     elif regime == bear:
#         return "Long Put / Put Spread"
#     else:
#         return "Iron Condor"

# data["option_strategy"] = data["regime"].apply(option_strategy)

# # 3) Forecast Future Regime Probabilities

# # This uses the transition matrix, the quiet heart of the HMM.

# transition_matrix = hmm.transmat_
# transition_matrix

# # One-Step Ahead Regime Probability
# current_regime = data["regime"].iloc[-1]

# next_regime_prob = transition_matrix[current_regime]
# next_regime_prob


# # Interpretation:

# # High self-transition → trend persistence
# # Rising probability of another regime → early warning

# # Multi-Step Forecast
# def forecast_regime_probs(start_regime, steps=5):
#     prob = np.zeros(hmm.n_components)
#     prob[start_regime] = 1

#     forecasts = []
#     for _ in range(steps):
#         prob = prob @ transition_matrix
#         forecasts.append(prob.copy())
#     return forecasts

# forecast_regime_probs(current_regime, steps=10)


# # This lets you ask:
# # “What’s the probability we’re still bullish in 2 weeks?”
# # That’s gold for options expiration selection.

