"""
Kalman Filter Trading System (KTS) — OU Mean-Level Estimation
-------------------------------------------------------------
The Kalman filter estimates the current mean level (fair value) of an Ornstein–Uhlenbeck (OU)
process. The horizontal line on the chart is this estimated mean level and updates every tick.
You can calibrate the OU model to historical data and control how much the filter trusts the
OU model vs. observed prices via the noise lever. IBKR API only.

• Calibrate OU: Fits AR(1) to the last N bars (bar size + calib window). Produces φ, μ, σ
  and builds the Kalman filter. Use "Refresh & Calibrate" or start stream (which calibrates first).

• Kalman update: On every tick we run predict (OU step) then update (blend with price).
  The state x is the estimated mean level; the horizontal line is drawn at x.

• Noise lever: "Trust prices" = filter follows price closely; "Trust OU" = filter stays near
  OU prediction. Implemented via observation noise R (high R ⇒ trust OU more).

• OU forecast: Purple dots show pure OU forward prediction from current state (no new observations).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from collections import deque
from datetime import datetime

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

# -----------------------------------------------------------------------------
# OU / AR(1) estimation from historical closes (discrete approximation to OU)
# -----------------------------------------------------------------------------
def estimate_ar1(closes):
    """
    Fit AR(1) on calibration-window closes. Returns (phi, mu, sigma).
    phi, sigma from regression; mu = sample mean of the window so the OU mean
    is anchored to the 60 bars. The Kalman filter then adjusts the mean level over time.
    """
    if closes is None or len(closes) < 5:
        return None
    try:
        y = np.array(closes, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 5:
            return None
        mu = float(np.mean(y))
        x_lag = y[:-1]
        x_curr = y[1:]
        X = np.column_stack([np.ones_like(x_lag), x_lag])
        beta = np.linalg.lstsq(X, x_curr, rcond=None)[0]
        c, phi = float(beta[0]), float(beta[1])
        phi = np.clip(phi, 0.01, 0.99)
        resid = x_curr - (c + phi * x_lag)
        sigma = np.sqrt(np.mean(resid ** 2))
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = max(np.std(y) * 0.01, 1e-9)
        return phi, mu, sigma
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Kalman filter: state = mean level (OU state), observation = price
# The horizontal line is drawn at self.x (updated every tick).
# -----------------------------------------------------------------------------
class KalmanOU:
    """
    One-dimensional Kalman filter with OU transition. State x is the estimated mean level
    (fair value). Parameters (phi, mu, Q, R) are set at construction; only (x, P) update on each tick.
    """

    def __init__(self, phi, mu, sigma_process, obs_noise_scale=1.0):
        self.phi = phi
        self.mu = mu
        self.Q = (sigma_process ** 2) * max(1 - phi ** 2, 1e-6)
        self.R = (sigma_process ** 2) * max(obs_noise_scale, 0.01)
        self.x = mu
        self.P = self.R

    def predict(self):
        """OU transition: mean level and variance evolve one step without observation."""
        self.x = self.phi * self.x + (1 - self.phi) * self.mu
        self.P = self.phi ** 2 * self.P + self.Q

    def update(self, z):
        """Predict then measurement update with observed price z. State (x, P) only."""
        self.predict()
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P

    def forecast(self, steps):
        """Pure OU prediction: mean level at 1..steps ahead, no new data."""
        x, mu, phi = self.x, self.mu, self.phi
        return [mu + (phi ** k) * (x - mu) for k in range(1, steps + 1)]


# -----------------------------------------------------------------------------
# IBKR API wrapper: historical bars, live ticks, orders
# -----------------------------------------------------------------------------
class IBApp(EWrapper, EClient):
    """TWS/Gateway connection: stores historical bars by reqId, forwards last price to on_tick."""

    def __init__(self, on_tick=None):
        EClient.__init__(self, self)
        self.connected = False
        self.next_order_id = None
        self.on_tick = on_tick
        self.last_price = None
        self.bid = self.ask = None
        self.historical_data = {}
        self.hist_done = threading.Event()
        self.positions = {}
        self.account_value = None

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in (2104, 2106, 2158, 2176):
            return
        print(f"IB Error {reqId}: {errorCode} - {errorString}")

    def nextValidId(self, orderId: int):
        self.connected = True
        self.next_order_id = orderId

    def tickPrice(self, reqId, tickType, price, attrib):
        if price <= 0:
            return
        if tickType == 4:
            self.last_price = price
            if self.on_tick:
                self.on_tick(price, datetime.now())
        elif tickType == 1:
            self.bid = price
        elif tickType == 2:
            self.ask = price

    def historicalData(self, reqId, bar):
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append({
            'date': bar.date, 'open': bar.open, 'high': bar.high,
            'low': bar.low, 'close': bar.close, 'volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        self.hist_done.set()

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        self.positions[contract.symbol] = {'position': position, 'avgCost': avgCost}

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        if tag == "NetLiquidation":
            try:
                self.account_value = float(value)
            except ValueError:
                pass


# -----------------------------------------------------------------------------
# Map noise lever [0, 100] to observation noise scale: 0 = trust prices, 100 = trust OU
# -----------------------------------------------------------------------------
def noise_lever_to_scale(lever_percent):
    """
    lever 0 -> scale 0.1 (low R, trust prices); lever 100 -> scale 1e8 (R >> P so K≈0, pure OU).
    At 100% Trust OU the Kalman gain is effectively zero so the KF mean equals the OU process.
    """
    p = max(0, min(100, lever_percent)) / 100.0
    if p >= 1.0:
        return 1e8
    return 0.1 + (10.0 - 0.1) * p


# -----------------------------------------------------------------------------
# Main app: GUI + OU calibration + Kalman (mean-level) + trading
# -----------------------------------------------------------------------------
class KalmanTradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kalman Filter Trading System — OU Mean-Level Estimation")
        self.root.geometry("1150x800")
        self.root.configure(bg='#0d1117')

        self.ib = IBApp(on_tick=self.on_tick)
        self.connected = False
        self.streaming = False
        self.api_thread = None
        self.symbol_var = tk.StringVar(value="AAPL")
        self.host_var = tk.StringVar(value="127.0.0.1")
        self.port_var = tk.StringVar(value="7497")

        self.bar_size_var = tk.StringVar(value="1 m")
        self.calib_window_var = tk.StringVar(value="60")
        self.online_params_var = tk.BooleanVar(value=False)
        self.noise_lever_var = tk.DoubleVar(value=50.0)
        self._bar_sec = 60
        self.max_bars = 120
        self.ohlc_bars = deque(maxlen=self.max_bars)
        self.current_bar = None
        self.bar_start = None
        self.prices = deque(maxlen=500)
        self.kalman_prices = []
        self.forecast_prices = []

        self.phi = self.mu = self.sigma = None
        self.kalman = None

        self.position = 0
        self.entry_price = 0.0
        self.cash = 100000.0
        self.order_id = None
        self._chart_update_scheduled = False
        self._last_redraw_time = 0.0

        self.setup_styles()
        self.setup_ui()
        self.setup_chart()
        self.refresh_timer()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        bg, fg = '#0d1117', '#c9d1d9'
        self.style.configure('TFrame', background=bg)
        self.style.configure('TLabelframe', background=bg, foreground=fg)
        self.style.configure('TLabelframe.Label', background=bg, foreground=fg, font=('Segoe UI', 10, 'bold'))
        self.style.configure('TLabel', background=bg, foreground=fg)
        self.style.configure('TButton', background='#238636', foreground='white', padding=(8, 4))
        self.style.map('TButton', background=[('active', '#2ea043')])
        self.style.configure('Accent.TButton', background='#da3633')
        self.style.map('Accent.TButton', background=[('active', '#f85149')])

    def setup_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky='nsew')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(5, weight=1)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        ttk.Label(header, text="Kalman Filter Trading System — OU Mean-Level", font=('Segoe UI', 14, 'bold')).pack(side='left')
        self.status_lbl = tk.Label(header, text="● Disconnected", font=('Segoe UI', 10, 'bold'), bg='#0d1117', fg='#f85149')
        self.status_lbl.pack(side='right')

        ctrl = ttk.LabelFrame(main, text="Connection & Symbol", padding=8)
        ctrl.grid(row=1, column=0, sticky='ew', pady=(0, 8))
        ctrl.columnconfigure(1, weight=1)
        row0 = ttk.Frame(ctrl)
        row0.grid(row=0, column=0, sticky='ew')
        ttk.Label(row0, text="Host").pack(side='left', padx=(0, 4))
        ttk.Entry(row0, textvariable=self.host_var, width=12).pack(side='left', padx=(0, 12))
        ttk.Label(row0, text="Port").pack(side='left', padx=(0, 4))
        ttk.Entry(row0, textvariable=self.port_var, width=6).pack(side='left', padx=(0, 12))
        self.connect_btn = ttk.Button(row0, text="Connect", command=self.connect_ib)
        self.connect_btn.pack(side='left', padx=(0, 6))
        self.disc_btn = ttk.Button(row0, text="Disconnect", command=self.disconnect_ib, state='disabled', style='Accent.TButton')
        self.disc_btn.pack(side='left')
        row1 = ttk.Frame(ctrl)
        row1.grid(row=1, column=0, sticky='ew', pady=(6, 0))
        ttk.Label(row1, text="Symbol").pack(side='left', padx=(0, 4))
        ttk.Entry(row1, textvariable=self.symbol_var, width=10).pack(side='left', padx=(0, 8))
        self.stream_btn = ttk.Button(row1, text="▶ Start stream", command=self.toggle_stream, state='disabled')
        self.stream_btn.pack(side='left', padx=(0, 8))
        self.refresh_btn = ttk.Button(row1, text="⟳ Refresh & Calibrate OU", command=self.refresh_30m, state='disabled')
        self.refresh_btn.pack(side='left', padx=(0, 8))
        ttk.Button(row1, text="Clear chart", command=self.clear_chart, style='Accent.TButton').pack(side='left', padx=(0, 8))
        self.price_lbl = tk.Label(row1, text="Last: ---", font=('Consolas', 12, 'bold'), bg='#0d1117', fg='#7ee787')
        self.price_lbl.pack(side='right')

        ou_frame = ttk.LabelFrame(main, text="OU Model & Calibration", padding=8)
        ou_frame.grid(row=2, column=0, sticky='ew', pady=(0, 4))
        ou_frame.columnconfigure(1, weight=1)
        row_ou = ttk.Frame(ou_frame)
        row_ou.grid(row=0, column=0, sticky='ew')
        ttk.Label(row_ou, text="Bar size").pack(side='left', padx=(0, 4))
        bar_combo = ttk.Combobox(row_ou, textvariable=self.bar_size_var, values=("30 s", "1 m", "5 m"), width=8, state="readonly")
        bar_combo.pack(side='left', padx=(0, 12))
        ttk.Label(row_ou, text="Calib window (bars)").pack(side='left', padx=(0, 4))
        ttk.Entry(row_ou, textvariable=self.calib_window_var, width=6).pack(side='left', padx=(0, 12))
        self.online_cb = ttk.Checkbutton(row_ou, text="Recalibrate OU each new bar", variable=self.online_params_var)
        self.online_cb.pack(side='left', padx=(0, 16))
        ttk.Label(row_ou, text="φ (phi):").pack(side='left', padx=(0, 2))
        self.phi_lbl = tk.Label(row_ou, text="—", font=('Consolas', 10), bg='#0d1117', fg='#8b949e')
        self.phi_lbl.pack(side='left', padx=(0, 8))
        ttk.Label(row_ou, text="μ (mean):").pack(side='left', padx=(0, 2))
        self.mu_lbl = tk.Label(row_ou, text="—", font=('Consolas', 10), bg='#0d1117', fg='#8b949e')
        self.mu_lbl.pack(side='left', padx=(0, 8))
        ttk.Label(row_ou, text="σ:").pack(side='left', padx=(0, 2))
        self.sigma_lbl = tk.Label(row_ou, text="—", font=('Consolas', 10), bg='#0d1117', fg='#8b949e')
        self.sigma_lbl.pack(side='left')

        noise_frame = ttk.LabelFrame(main, text="Kalman: Trust prices ↔ Trust OU model", padding=8)
        noise_frame.grid(row=3, column=0, sticky='ew', pady=(0, 4))
        noise_frame.columnconfigure(1, weight=1)
        row_noise = ttk.Frame(noise_frame)
        row_noise.grid(row=0, column=0, sticky='ew')
        ttk.Label(row_noise, text="Trust prices").pack(side='left', padx=(0, 6))
        self.noise_slider = ttk.Scale(row_noise, from_=0, to=100, variable=self.noise_lever_var, orient='horizontal', length=280, command=self._on_noise_lever)
        self.noise_slider.pack(side='left', padx=(0, 6))
        ttk.Label(row_noise, text="Trust OU").pack(side='left', padx=(0, 8))
        self.noise_val_lbl = tk.Label(row_noise, text="50%", font=('Consolas', 10), bg='#0d1117', fg='#58a6ff')
        self.noise_val_lbl.pack(side='left')

        trade = ttk.LabelFrame(main, text="Trading & Portfolio", padding=8)
        trade.grid(row=4, column=0, sticky='ew', pady=(0, 8))
        row2 = ttk.Frame(trade)
        row2.grid(row=0, column=0, sticky='ew')
        ttk.Button(row2, text="Long", command=lambda: self.place_trade(1)).pack(side='left', padx=(0, 6))
        ttk.Button(row2, text="Short", command=lambda: self.place_trade(-1)).pack(side='left', padx=(0, 6))
        ttk.Button(row2, text="Close", command=lambda: self.place_trade(0), style='Accent.TButton').pack(side='left', padx=(0, 12))
        ttk.Label(row2, text="Position:").pack(side='left', padx=(0, 4))
        self.pos_lbl = tk.Label(row2, text="0", font=('Consolas', 11, 'bold'), bg='#0d1117', fg='#8b949e')
        self.pos_lbl.pack(side='left', padx=(0, 12))
        ttk.Label(row2, text="Portfolio:").pack(side='left', padx=(0, 4))
        self.port_lbl = tk.Label(row2, text="100000.00", font=('Consolas', 11, 'bold'), bg='#0d1117', fg='#58a6ff')
        self.port_lbl.pack(side='left')

        chart_frame = ttk.LabelFrame(main, text="OHLC | Orange = Kalman mean (history) | Solid line = live mean (updates every tick) | Purple = OU forecast", padding=6)
        chart_frame.grid(row=5, column=0, sticky='nsew')
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.chart_container = ttk.Frame(chart_frame)
        self.chart_container.grid(row=0, column=0, sticky='nsew')
        self.chart_container.columnconfigure(0, weight=1)
        self.chart_container.rowconfigure(0, weight=1)

    def _on_noise_lever(self, _):
        v = self.noise_lever_var.get()
        self.noise_val_lbl.config(text=f"{v:.0f}%")
        if self.kalman is not None and self.sigma is not None:
            scale = noise_lever_to_scale(v)
            self.kalman.R = (self.sigma ** 2) * max(scale, 0.01)
            self.redraw_chart()

    def setup_chart(self):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(11, 5), facecolor='#0d1117')
        self.ax.set_facecolor('#161b22')
        self.ax.tick_params(colors='#8b949e')
        self.ax.spines['bottom'].set_color('#30363d')
        self.ax.spines['top'].set_color('#30363d')
        self.ax.spines['left'].set_color('#30363d')
        self.ax.spines['right'].set_color('#30363d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

    def contract(self):
        c = Contract()
        c.symbol = self.symbol_var.get().strip().upper()
        c.secType = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

    def connect_ib(self):
        if self.connected:
            return
        try:
            port = int(self.port_var.get())
            self.ib.connect(self.host_var.get(), port, 98)
            self.api_thread = threading.Thread(target=self.ib.run, daemon=True)
            self.api_thread.start()
            for _ in range(50):
                time.sleep(0.1)
                if self.ib.connected:
                    break
            if self.ib.connected:
                self.connected = True
                self.order_id = self.ib.next_order_id
                self.connect_btn.config(state='disabled')
                self.disc_btn.config(state='normal')
                self.stream_btn.config(state='normal')
                self.refresh_btn.config(state='normal')
                self.status_lbl.config(text="● Connected", fg='#3fb950')
                self.ib.reqAccountSummary(9001, "All", "NetLiquidation")
                self.ib.reqPositions()
                time.sleep(0.3)
                sym = self.symbol_var.get().strip().upper()
                if sym in self.ib.positions:
                    self.position = int(self.ib.positions[sym]['position'])
                    self.entry_price = self.ib.positions[sym]['avgCost']
            else:
                messagebox.showerror("Connection", "Could not connect to TWS/Gateway")
        except Exception as e:
            messagebox.showerror("Connection", str(e))

    def disconnect_ib(self):
        if not self.connected:
            return
        if self.streaming:
            self.toggle_stream()
        self.ib.disconnect()
        self.connected = False
        self.connect_btn.config(state='normal')
        self.disc_btn.config(state='disabled')
        self.stream_btn.config(state='disabled')
        self.refresh_btn.config(state='disabled')
        self.status_lbl.config(text="● Disconnected", fg='#f85149')

    def clear_chart(self):
        if self.streaming:
            self.ib.cancelMktData(1)
            self.streaming = False
            self.stream_btn.config(text="▶ Start stream")
            self.status_lbl.config(text="● Connected", fg='#3fb950')
        self.ohlc_bars.clear()
        self.current_bar = None
        self.bar_start = None
        self.prices.clear()
        self.kalman_prices = []
        self.forecast_prices = []
        self.kalman = None
        self.phi = self.mu = self.sigma = None
        self.phi_lbl.config(text="—")
        self.mu_lbl.config(text="—")
        self.sigma_lbl.config(text="—")
        self.price_lbl.config(text="Last: ---")
        self.redraw_chart()

    def toggle_stream(self):
        if not self.connected:
            return
        if self.streaming:
            self.ib.cancelMktData(1)
            self.streaming = False
            self.stream_btn.config(text="▶ Start stream")
            self.status_lbl.config(text="● Connected", fg='#3fb950')
        else:
            self.refresh_30m()
            self.ib.reqMktData(1, self.contract(), "", False, False, [])
            self.streaming = True
            self.stream_btn.config(text="■ Stop stream")
            self.status_lbl.config(text="● Streaming", fg='#58a6ff')

    def on_tick(self, price, ts):
        self.prices.append(price)
        if self.current_bar is None:
            self.current_bar = [ts, price, price, price, price]
            self.bar_start = ts
        else:
            self.current_bar[2] = max(self.current_bar[2], price)
            self.current_bar[3] = min(self.current_bar[3], price)
            self.current_bar[4] = price
        if self.kalman is not None:
            self.kalman.update(price)
            self.forecast_prices = self.kalman.forecast(5)
        delta = (ts - self.bar_start).total_seconds() if self.bar_start else 0
        if delta >= self._bar_sec:
            self.ohlc_bars.append({
                't': self.bar_start, 'o': self.current_bar[1], 'h': self.current_bar[2],
                'l': self.current_bar[3], 'c': self.current_bar[4]
            })
            if self.online_params_var.get() and self.kalman is not None:
                scale = noise_lever_to_scale(self.noise_lever_var.get())
                result = self._recalibrate_from_bars(list(self.ohlc_bars), scale)
                if result is not None:
                    self.kalman = result[3]
                    self.kalman_prices = result[4]
            elif self.kalman is not None:
                self.kalman_prices.append(self.kalman.x)
            self.current_bar = [ts, price, price, price, price]
            self.bar_start = ts
            if self.kalman is not None:
                self.forecast_prices = self.kalman.forecast(5)
            if not self._chart_update_scheduled:
                self._chart_update_scheduled = True
                self.root.after(0, self._deferred_chart_update)
            return
        now = time.time()
        if not self._chart_update_scheduled and (now - self._last_redraw_time) >= 0.08:
            self._chart_update_scheduled = True
            self.root.after(0, self._deferred_chart_update)

    def _deferred_chart_update(self):
        self._chart_update_scheduled = False
        self._last_redraw_time = time.time()
        if self.ib.last_price is not None:
            self.price_lbl.config(text=f"Last: {self.ib.last_price:.2f}")
        self.redraw_chart()

    def _bar_size_to_sec_and_ib(self):
        bs = self.bar_size_var.get().strip()
        if bs == "30 s":
            return 30, "30 secs"
        if bs == "5 m":
            return 300, "5 mins"
        return 60, "1 min"

    def _get_calib_window(self):
        try:
            return max(5, min(500, int(self.calib_window_var.get())))
        except ValueError:
            return 60

    def _recalibrate_from_bars(self, bars_list, obs_scale):
        n = len(bars_list)
        if n < 5:
            return None
        w = min(self._get_calib_window(), n)

        def close(b):
            return b.get('c', b.get('close'))

        closes = [close(b) for b in bars_list[-w:]]
        params = estimate_ar1(closes)
        if params is None:
            return None
        phi, mu, sigma = params
        self.phi, self.mu, self.sigma = phi, mu, sigma
        kalman = KalmanOU(phi, mu, sigma, obs_noise_scale=obs_scale)
        kalman_prices = []
        for b in bars_list:
            kalman.update(close(b))
            kalman_prices.append(kalman.x)
        return phi, mu, sigma, kalman, kalman_prices

    def _update_ou_labels(self):
        if self.phi is not None and np.isfinite(self.phi):
            self.phi_lbl.config(text=f"{self.phi:.4f}")
        else:
            self.phi_lbl.config(text="—")
        if self.mu is not None and np.isfinite(self.mu):
            self.mu_lbl.config(text=f"{self.mu:.2f}")
        else:
            self.mu_lbl.config(text="—")
        if self.sigma is not None and np.isfinite(self.sigma):
            self.sigma_lbl.config(text=f"{self.sigma:.4f}")
        else:
            self.sigma_lbl.config(text="—")

    def refresh_30m(self):
        if not self.connected:
            return
        num_bars = self._get_calib_window()
        self.calib_window_var.set(str(num_bars))
        bar_sec, bar_size_setting = self._bar_size_to_sec_and_ib()
        self._bar_sec = bar_sec
        total_sec = num_bars * bar_sec
        if total_sec >= 86400:
            duration_str = f"{max(1, total_sec // 86400)} D"
        else:
            duration_str = f"{max(60, total_sec)} S"
        scale = noise_lever_to_scale(self.noise_lever_var.get())

        self.ib.historical_data.clear()
        self.ib.hist_done.clear()
        self.ib.reqHistoricalData(2, self.contract(), "", duration_str, bar_size_setting, "TRADES", 1, 1, False, [])
        self.ib.hist_done.wait(timeout=20)
        bars = self.ib.historical_data.get(2, [])
        if not bars:
            messagebox.showwarning("Data", "No historical bars received. Check symbol and market hours.")
            return
        bar_list = []
        for b in bars:
            try:
                t = datetime.strptime(b['date'][:16], '%Y%m%d  %H:%M') if len(b['date']) >= 16 else datetime.now()
            except Exception:
                t = datetime.now()
            try:
                c = float(b['close'])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(c) or c <= 0:
                continue
            bar_list.append({'t': t, 'o': float(b['open']), 'h': float(b['high']), 'l': float(b['low']), 'c': c})
        if len(bar_list) < 5:
            messagebox.showwarning("Fit", "Need at least 5 valid bars for AR(1). Increase calib window or try different bar size.")
            return
        result = self._recalibrate_from_bars(bar_list, scale)
        if result is None:
            messagebox.showwarning("Fit", "Could not estimate AR(1) from bars. Increase calib window or check data.")
            return
        self.kalman = result[3]
        self.kalman_prices = result[4]
        self.ohlc_bars = deque(bar_list, maxlen=self.max_bars)
        self.forecast_prices = self.kalman.forecast(5)
        self._update_ou_labels()
        self.redraw_chart()

    def place_trade(self, side):
        if not self.connected or self.ib.next_order_id is None:
            messagebox.showerror("Trade", "Not connected or no order ID.")
            return
        contract = self.contract()
        qty = 100
        if side == 0:
            if self.position == 0:
                messagebox.showinfo("Close", "No position to close.")
                return
            qty = abs(self.position)
            order = Order()
            order.action = "SELL" if self.position > 0 else "BUY"
            order.orderType = "MKT"
            order.totalQuantity = qty
        else:
            order = Order()
            order.action = "BUY" if side == 1 else "SELL"
            order.orderType = "MKT"
            order.totalQuantity = qty
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        self.ib.placeOrder(self.ib.next_order_id, contract, order)
        self.ib.next_order_id += 1
        self.position = 0 if side == 0 else (self.position + (qty if side == 1 else -qty))
        self.update_portfolio_display()

    def update_portfolio_display(self):
        self.pos_lbl.config(text=str(self.position))
        val = self.ib.account_value if self.ib.account_value is not None else self.cash
        self.port_lbl.config(text=f"{val:,.2f}")

    def redraw_chart(self):
        self.ax.clear()
        self.ax.set_facecolor('#161b22')
        self.ax.tick_params(colors='#8b949e')
        has_current = self.current_bar is not None and (self.ohlc_bars or self.streaming)
        n = len(self.ohlc_bars)
        if not self.ohlc_bars and not has_current:
            self.ax.set_ylabel("Price", color='#c9d1d9')
            self.ax.set_xlabel("Bar index", color='#8b949e')
            self.canvas.draw_idle()
            return
        if self.ohlc_bars:
            t = [b['t'] for b in self.ohlc_bars]
            o = [b['o'] for b in self.ohlc_bars]
            h = [b['h'] for b in self.ohlc_bars]
            l = [b['l'] for b in self.ohlc_bars]
            c = [b['c'] for b in self.ohlc_bars]
            for i in range(len(t)):
                color = '#3fb950' if c[i] >= o[i] else '#f85149'
                body_bottom, body_top = min(o[i], c[i]), max(o[i], c[i])
                height = body_top - body_bottom or 0.01
                self.ax.plot([i, i], [l[i], body_bottom], color=color, linewidth=1)
                self.ax.plot([i, i], [body_top, h[i]], color=color, linewidth=1)
                self.ax.add_patch(Rectangle((i - 0.35, body_bottom), 0.7, height, facecolor=color, edgecolor=color))
            self.ax.plot(range(len(t)), c, color='#58a6ff', alpha=0.5, linewidth=0.8, label='Close')
        if has_current:
            i = n
            o_cur = self.current_bar[1]
            h_cur = self.current_bar[2]
            l_cur = self.current_bar[3]
            c_cur = self.current_bar[4]
            color = '#3fb950' if c_cur >= o_cur else '#f85149'
            body_bottom, body_top = min(o_cur, c_cur), max(o_cur, c_cur)
            height = body_top - body_bottom or 0.01
            self.ax.plot([i, i], [l_cur, body_bottom], color=color, linewidth=1)
            self.ax.plot([i, i], [body_top, h_cur], color=color, linewidth=1)
            self.ax.add_patch(Rectangle((i - 0.35, body_bottom), 0.7, height, facecolor=color, edgecolor='#8b949e', linewidth=1.5))
        n_draw = n + (1 if has_current else 0)
        if self.kalman_prices and n > 0:
            k_len = min(len(self.kalman_prices), n)
            idx = list(range(n - k_len, n))
            self.ax.scatter(idx, self.kalman_prices[-k_len:], color='#f0883e', s=18, zorder=5, label='Kalman mean (history)')
        if self.kalman is not None and np.isfinite(self.kalman.x):
            self.ax.axhline(y=self.kalman.x, color='#7ee787', linestyle='-', linewidth=2, alpha=0.95, zorder=4, label='Live mean level (Kalman x)')
        if self.mu is not None and np.isfinite(self.mu):
            self.ax.axhline(y=self.mu, color='#8b949e', linestyle='--', linewidth=1, alpha=0.7, zorder=3, label='Long-run mean (μ)')
        if self.forecast_prices:
            fc_x = list(range(n_draw, n_draw + len(self.forecast_prices)))
            self.ax.scatter(fc_x, self.forecast_prices, color='#a371f7', s=22, zorder=5, label='OU forecast')
        self.ax.set_ylabel("Price", color='#c9d1d9')
        self.ax.set_xlabel("Bar index", color='#8b949e')
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc='upper left', fontsize=8)
        self.canvas.draw_idle()

    def refresh_timer(self):
        self.update_portfolio_display()
        self.root.after(2000, self.refresh_timer)


def main():
    root = tk.Tk()
    app = KalmanTradingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
