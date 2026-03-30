"""
Combined Tkinter + Plotly (Dash) app
Multivariate NN (single-row features -> power_output) + univariate Chen FTS
Default dataset path uses uploaded file: /mnt/data/synthetic_wind_energy.csv
"""

import os
import threading
import webbrowser
import math
import datetime
import traceback

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Plotly / Dash
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

# --------------------
# Color scheme for modern UI
# --------------------
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Pink/Magenta
    'success': '#28A745',      # Green
    'warning': '#FFC107',      # Yellow
    'danger': '#DC3545',       # Red
    'info': '#17A2B8',         # Cyan
    'dark': '#343A40',         # Dark gray
    'light': '#F8F9FA',        # Light gray
    'white': '#FFFFFF',        # White
    'bg': '#F0F4F8',           # Light blue background
    'card_bg': '#FFFFFF',      # Card background
    'fts_color': '#E74C3C',    # FTS model color
    'nn_color': '#3498DB',     # NN model color
    'hybrid_color': '#9B59B6', # Hybrid model color
}

# --------------------
# Default uploaded dataset path (provided by user/session)
# --------------------
DEFAULT_DATA_PATH = "/mnt/data/wind_energy.csv"

# --------------------
# Required feature columns and label
# --------------------
FEATURE_COLS = [
    "wind_speed", "wind_direction", "air_density", "temperature",
    "pressure", "humidity", "rotor_speed", "blade_pitch",
    "turbine_yaw", "vibration"
]
TIMESTAMP_COL = "timestamp"
LABEL_COL = "power_output"

# --------------------
# Utility functions
# --------------------
def load_table(path):
    """Load CSV or Excel and validate required columns."""
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path, parse_dates=[TIMESTAMP_COL])
    else:
        df = pd.read_csv(path, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    # Validate
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    # Fill or drop NaNs (simple strategy - drop)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    return df

# FTS helpers (univariate)
def build_intervals(series, n_intervals=8):
    mn, mx = series.min(), series.max()
    if mx == mn:
        edges = np.array([mn, mn + 1.0])
        midpoints = np.array([mn + 0.5])
        return edges, midpoints
    width = (mx - mn) / n_intervals
    edges = np.array([mn + i * width for i in range(n_intervals + 1)])
    midpoints = np.array([(edges[i] + edges[i+1]) / 2 for i in range(n_intervals)])
    return edges, midpoints

def fuzzify(value, edges):
    for i in range(len(edges)-1):
        if edges[i] <= value <= edges[i+1]:
            return i
    return len(edges)-2

def build_flr(train_series, edges):
    flr = {}
    for i in range(len(train_series)-1):
        a = fuzzify(train_series[i], edges)
        b = fuzzify(train_series[i+1], edges)
        flr.setdefault(a, []).append(b)
    return flr

def forecast_fts_single(last_value, edges, midpoints, flr, default=None):
    a = fuzzify(last_value, edges)
    consequents = flr.get(a, None)
    if not consequents:
        return default if default is not None else float(np.mean(midpoints))
    return float(np.mean([midpoints[c] for c in consequents]))

def rmse(a, b):
    return math.sqrt(mean_squared_error(a, b))

# --------------------
# Forecast pipeline (multivariate NN + univariate FTS)
# --------------------
class ForecastPipeline:
    def __init__(self):
        self.df = None
        self.train_df = None
        self.test_df = None

        # FTS
        self.edges = None
        self.midpoints = None
        self.flr = None
        self.fts_preds = None
        self.default_pred = None
        self.n_intervals = 8

        # NN
        self.nn_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.nn_preds = None
        
        # Hybrid FTS+MLP
        self.hybrid_model = None
        self.hybrid_scaler_X = None
        self.hybrid_scaler_y = None
        self.hybrid_preds = None
        self.fts_train_preds = None
        self.fts_test_preds = None

        # metrics
        self.metrics = {}

        # split ratio
        self.test_frac = 0.10

    def load(self, path=DEFAULT_DATA_PATH):
        self.df = load_table(path)
        # simple train/test split by last fraction
        n = len(self.df)
        test_size = max(1, int(self.test_frac * n))
        self.train_df = self.df.iloc[:-test_size].reset_index(drop=True)
        self.test_df = self.df.iloc[-test_size:].reset_index(drop=True)
        return self.df

    def train_fts(self, n_intervals=8):
        self.n_intervals = n_intervals
        series = self.train_df[LABEL_COL].values.astype(float)
        edges, midpoints = build_intervals(series, n_intervals=n_intervals)
        flr = build_flr(series, edges)
        default_pred = float(series.mean())
        # one-step one-step predictions on test set (use last train actual for first step and actual previous for subsequent)
        preds = []
        for i in range(len(self.test_df)):
            last = series[-1] if i == 0 else float(self.test_df.iloc[i-1][LABEL_COL])
            preds.append(forecast_fts_single(last, edges, midpoints, flr, default=default_pred))
        self.edges, self.midpoints, self.flr, self.default_pred = edges, midpoints, flr, default_pred
        self.fts_preds = np.array(preds)
        return self.fts_preds

    def train_nn(self, hidden=(128,64), max_iter=500):
        # train NN on single-row features -> label mapping (no windowing), i.e., sample-wise regression
        X = self.train_df[FEATURE_COLS].values.astype(float)
        y = self.train_df[LABEL_COL].values.astype(float).reshape(-1, 1)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y).ravel()
        model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter, random_state=42)
        model.fit(Xs, ys)
        self.nn_model = model
        # predict on test set (use actual features from test rows)
        X_test = self.test_df[FEATURE_COLS].values.astype(float)
        Xs_test = self.scaler_X.transform(X_test)
        preds_scaled = model.predict(Xs_test)
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
        self.nn_preds = np.array(preds)
        return self.nn_preds

    def train_hybrid(self, hidden=(64,32), max_iter=600):
        """
        Train hybrid FTS+MLP model.
        This model concatenates FTS predictions with original features
        and trains an MLP on the combined input.
        """
        if self.flr is None or self.nn_model is None:
            raise RuntimeError("Train FTS and NN models first.")
        
        # Get training features and labels
        X_train = self.train_df[FEATURE_COLS].values.astype(float)
        y_train = self.train_df[LABEL_COL].values.astype(float)
        X_test = self.test_df[FEATURE_COLS].values.astype(float)
        
        # Generate FTS predictions for training set (using previous actual value)
        fts_train_preds = []
        for i in range(len(y_train)):
            last = y_train[i-1] if i > 0 else y_train[0]
            fts_val = forecast_fts_single(last, self.edges, self.midpoints, self.flr, default=self.default_pred)
            fts_train_preds.append(fts_val)
        self.fts_train_preds = np.array(fts_train_preds).reshape(-1, 1)
        
        # Generate FTS predictions for test set (recursive forecasting)
        fts_test_preds = []
        last_val = y_train[-1]
        for i in range(len(self.test_df)):
            pred = forecast_fts_single(last_val, self.edges, self.midpoints, self.flr, default=self.default_pred)
            fts_test_preds.append(pred)
            last_val = float(self.test_df.iloc[i][LABEL_COL])
        self.fts_test_preds = np.array(fts_test_preds).reshape(-1, 1)
        
        # Concatenate FTS predictions with original features
        X_train_h = np.hstack([X_train, self.fts_train_preds])
        X_test_h = np.hstack([X_test, self.fts_test_preds])
        
        # Scale features and labels
        self.hybrid_scaler_X = MinMaxScaler()
        self.hybrid_scaler_y = MinMaxScaler()
        Xs_train = self.hybrid_scaler_X.fit_transform(X_train_h)
        Xs_test = self.hybrid_scaler_X.transform(X_test_h)
        ys_train = self.hybrid_scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Train hybrid MLP model
        model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter, random_state=42)
        model.fit(Xs_train, ys_train)
        self.hybrid_model = model
        
        # Predict on test set
        preds_scaled = model.predict(Xs_test)
        preds = self.hybrid_scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        self.hybrid_preds = np.array(preds)
        return self.hybrid_preds

    def evaluate(self):
        if self.fts_preds is None or self.nn_preds is None:
            raise RuntimeError("Train both models first.")
        actual = self.test_df[LABEL_COL].values.astype(float)
        fts_mae = mean_absolute_error(actual, self.fts_preds)
        fts_rmse = rmse(actual, self.fts_preds)
        fts_mape = float(np.mean(np.abs((actual - self.fts_preds) / (actual + 1e-8))) * 100)
        nn_mae = mean_absolute_error(actual, self.nn_preds)
        nn_rmse = rmse(actual, self.nn_preds)
        nn_mape = float(np.mean(np.abs((actual - self.nn_preds) / (actual + 1e-8))) * 100)
        
        # Hybrid metrics (if trained)
        if self.hybrid_preds is not None:
            hybrid_mae = mean_absolute_error(actual, self.hybrid_preds)
            hybrid_rmse = rmse(actual, self.hybrid_preds)
            hybrid_mape = float(np.mean(np.abs((actual - self.hybrid_preds) / (actual + 1e-8))) * 100)
            self.metrics = {
                "fts": {"mae": fts_mae, "rmse": fts_rmse, "mape": fts_mape},
                "nn": {"mae": nn_mae, "rmse": nn_rmse, "mape": nn_mape},
                "hybrid": {"mae": hybrid_mae, "rmse": hybrid_rmse, "mape": hybrid_mape},
            }
        else:
            self.metrics = {
                "fts": {"mae": fts_mae, "rmse": fts_rmse, "mape": fts_mape},
                "nn": {"mae": nn_mae, "rmse": nn_rmse, "mape": nn_mape},
            }
        return self.metrics

    def save_results(self, out_path="/mnt/data/fts_nn_multifeature_results.csv"):
        out = pd.DataFrame({
            TIMESTAMP_COL: self.test_df[TIMESTAMP_COL].values,
            "actual": self.test_df[LABEL_COL].values,
            "fts_pred": self.fts_preds,
            "nn_pred": self.nn_preds,
            "hybrid_pred": self.hybrid_preds if self.hybrid_preds is not None else [None] * len(self.test_df)
        })
        out.to_csv(out_path, index=False)
        return out_path

    def predict_row(self, row_dict):
        """
        row_dict: mapping for FEATURE_COLS -> values (numbers).
        Returns (fts_pred, nn_pred)
        - fts_pred uses last available actual power_output from training as 'last' for single-step.
        - nn_pred uses the trained scaler and model on provided features.
        """
        # FTS prediction: we use last train actual as "last_value" for one-step ahead
        if self.flr is None:
            fts_pred = None
        else:
            last_train_actual = float(self.train_df[LABEL_COL].values[-1])
            # But if row_dict includes a power_output value (e.g., same timestamp), ignore for FTS - FTS is univariate
            fts_pred = forecast_fts_single(last_train_actual, self.edges, self.midpoints, self.flr, default=self.default_pred)

        # NN prediction
        if self.nn_model is None:
            nn_pred = None
        else:
            x = np.array([row_dict[col] for col in FEATURE_COLS], dtype=float).reshape(1, -1)
            xs = self.scaler_X.transform(x)
            pred_scaled = self.nn_model.predict(xs)[0]
            nn_pred = float(self.scaler_y.inverse_transform([[pred_scaled]])[0,0])
        
        # Hybrid prediction
        if self.hybrid_model is None:
            hybrid_pred = None
        else:
            # Get FTS value for the last known power output
            if self.flr is not None:
                last_train_actual = float(self.train_df[LABEL_COL].values[-1])
                fts_val = forecast_fts_single(last_train_actual, self.edges, self.midpoints, self.flr, default=self.default_pred)
            else:
                fts_val = 0.0
            # Combine features with FTS prediction
            x = np.array([row_dict[col] for col in FEATURE_COLS], dtype=float).reshape(1, -1)
            x_hybrid = np.hstack([x, [[fts_val]]])
            xs_hybrid = self.hybrid_scaler_X.transform(x_hybrid)
            pred_scaled = self.hybrid_model.predict(xs_hybrid)[0]
            hybrid_pred = float(self.hybrid_scaler_y.inverse_transform([[pred_scaled]])[0,0])
        
        return fts_pred, nn_pred, hybrid_pred

# --------------------
# Tkinter GUI + Dash integration
# --------------------
class ModernCard(ttk.Frame):
    """A styled card container with title and content."""
    def __init__(self, parent, title="", **kwargs):
        style = ttk.Style()
        style.configure("Card.TFrame", background=COLORS['card_bg'])
        super().__init__(parent, style="Card.TFrame", **kwargs)
        self.configure(borderwidth=2, relief="raised")
        
        # Title bar
        if title:
            title_frame = tk.Frame(self, bg=COLORS['primary'])
            title_frame.pack(fill="x", padx=10, pady=5)
            tk.Label(title_frame, text=title, bg=COLORS['primary'], 
                    fg=COLORS['white'], font=("Segoe UI", 10, "bold")).pack()
        
        # Content area
        self.content = ttk.Frame(self, padding=10)
        self.content.pack(fill="both", expand=True)

class MetricGauge(ttk.Frame):
    """A metric display with value and progress bar."""
    def __init__(self, parent, title="", value=0, max_value=100, color=COLORS['primary'], **kwargs):
        super().__init__(parent, **kwargs)
        self.max_value = max_value
        self.color = color
        
        # Title
        tk.Label(self, text=title, font=("Segoe UI", 9), 
                fg=COLORS['dark'], bg=COLORS['card_bg']).pack(anchor="w")
        
        # Value label
        self.value_label = tk.Label(self, text=f"{value:.4f}", 
                                   font=("Segoe UI", 14, "bold"), 
                                   fg=color, bg=COLORS['card_bg'])
        self.value_label.pack(anchor="w", pady=(2, 5))
        
        # Progress bar
        self.progress = ttk.Progressbar(self, maximum=max_value, length=150, mode='determinate')
        self.progress.pack(fill="x")
        self.progress['value'] = min(value, max_value)

class MetricsDashboard(ttk.Frame):
    """A graphical metrics dashboard with charts."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.figures = []
        self._create_widgets()
    
    def _create_widgets(self):
        """Create dashboard widgets."""
        # Main container with scrolling
        container = tk.Frame(self, bg=COLORS['bg'])
        container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title = tk.Label(container, text="📊 Model Performance Dashboard", 
                        font=("Segoe UI", 16, "bold"), 
                        fg=COLORS['primary'], bg=COLORS['bg'])
        title.pack(pady=(0, 15))
        
        # Metrics cards row
        cards_frame = tk.Frame(container, bg=COLORS['bg'])
        cards_frame.pack(fill="x", pady=(0, 15))
        
        # FTS metrics card
        fts_card = ModernCard(cards_frame, title="FTS Model (Univariate)")
        fts_card.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.fts_mae = MetricGauge(fts_card.content, "MAE", value=0, color=COLORS['fts_color'])
        self.fts_mae.pack(fill="x", pady=5)
        self.fts_rmse = MetricGauge(fts_card.content, "RMSE", value=0, color=COLORS['fts_color'])
        self.fts_rmse.pack(fill="x", pady=5)
        self.fts_mape = MetricGauge(fts_card.content, "MAPE (%)", value=0, max_value=100, color=COLORS['fts_color'])
        self.fts_mape.pack(fill="x", pady=5)
        
        # NN metrics card
        nn_card = ModernCard(cards_frame, title="NN Model (Multivariate)")
        nn_card.pack(side="left", fill="both", expand=True, padx=5)
        
        self.nn_mae = MetricGauge(nn_card.content, "MAE", value=0, color=COLORS['nn_color'])
        self.nn_mae.pack(fill="x", pady=5)
        self.nn_rmse = MetricGauge(nn_card.content, "RMSE", value=0, color=COLORS['nn_color'])
        self.nn_rmse.pack(fill="x", pady=5)
        self.nn_mape = MetricGauge(nn_card.content, "MAPE (%)", value=0, max_value=100, color=COLORS['nn_color'])
        self.nn_mape.pack(fill="x", pady=5)
        
        # Hybrid metrics card
        hybrid_card = ModernCard(cards_frame, title="Hybrid Model (FTS+MLP)")
        hybrid_card.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        self.hybrid_mae = MetricGauge(hybrid_card.content, "MAE", value=0, color=COLORS['hybrid_color'])
        self.hybrid_mae.pack(fill="x", pady=5)
        self.hybrid_rmse = MetricGauge(hybrid_card.content, "RMSE", value=0, color=COLORS['hybrid_color'])
        self.hybrid_rmse.pack(fill="x", pady=5)
        self.hybrid_mape = MetricGauge(hybrid_card.content, "MAPE (%)", value=0, max_value=100, color=COLORS['hybrid_color'])
        self.hybrid_mape.pack(fill="x", pady=5)
        
        # Charts row
        charts_frame = tk.Frame(container, bg=COLORS['bg'])
        charts_frame.pack(fill="both", expand=True)
        
        # Bar chart for comparison
        bar_frame = ModernCard(charts_frame, title="Model Comparison (Lower is Better)")
        bar_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.fig_bar, self.ax_bar = plt.subplots(figsize=(5, 3), dpi=80)
        self.fig_bar.patch.set_facecolor(COLORS['card_bg'])
        self.bar_canvas = FigureCanvasTkAgg(self.fig_bar, bar_frame.content)
        self.bar_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Error distribution
        error_frame = ModernCard(charts_frame, title="Error Distribution")
        error_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        self.fig_error, self.ax_error = plt.subplots(figsize=(5, 3), dpi=80)
        self.fig_error.patch.set_facecolor(COLORS['card_bg'])
        self.error_canvas = FigureCanvasTkAgg(self.fig_error, error_frame.content)
        self.error_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Prediction comparison chart
        pred_frame = ModernCard(container, title="Actual vs Predictions (Sample)")
        pred_frame.pack(fill="both", expand=True, pady=(15, 0))
        
        self.fig_pred, self.ax_pred = plt.subplots(figsize=(10, 3), dpi=80)
        self.fig_pred.patch.set_facecolor(COLORS['card_bg'])
        self.pred_canvas = FigureCanvasTkAgg(self.fig_pred, pred_frame.content)
        self.pred_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def update_metrics(self, metrics):
        """Update dashboard with new metrics."""
        if not metrics:
            return
        
        m = metrics
        
        # Update FTS gauges
        self.fts_mae.value_label.config(text=f"{m['fts']['mae']:.4f}")
        self.fts_mae.progress['value'] = min(m['fts']['mae'], self.fts_mae.max_value)
        
        self.fts_rmse.value_label.config(text=f"{m['fts']['rmse']:.4f}")
        self.fts_rmse.progress['value'] = min(m['fts']['rmse'], self.fts_rmse.max_value)
        
        self.fts_mape.value_label.config(text=f"{m['fts']['mape']:.2f}%")
        self.fts_mape.progress['value'] = min(m['fts']['mape'], self.fts_mape.max_value)
        
        # Update NN gauges
        self.nn_mae.value_label.config(text=f"{m['nn']['mae']:.4f}")
        self.nn_mae.progress['value'] = min(m['nn']['mae'], self.nn_mae.max_value)
        
        self.nn_rmse.value_label.config(text=f"{m['nn']['rmse']:.4f}")
        self.nn_rmse.progress['value'] = min(m['nn']['rmse'], self.nn_rmse.max_value)
        
        self.nn_mape.value_label.config(text=f"{m['nn']['mape']:.2f}%")
        self.nn_mape.progress['value'] = min(m['nn']['mape'], self.nn_mape.max_value)
        
        # Update Hybrid gauges (if available)
        if 'hybrid' in m:
            self.hybrid_mae.value_label.config(text=f"{m['hybrid']['mae']:.4f}")
            self.hybrid_mae.progress['value'] = min(m['hybrid']['mae'], self.hybrid_mae.max_value)
            
            self.hybrid_rmse.value_label.config(text=f"{m['hybrid']['rmse']:.4f}")
            self.hybrid_rmse.progress['value'] = min(m['hybrid']['rmse'], self.hybrid_rmse.max_value)
            
            self.hybrid_mape.value_label.config(text=f"{m['hybrid']['mape']:.2f}%")
            self.hybrid_mape.progress['value'] = min(m['hybrid']['mape'], self.hybrid_mape.max_value)
        
        # Update bar chart
        self.ax_bar.clear()
        metrics_names = ['MAE', 'RMSE', 'MAPE (%)']
        fts_values = [m['fts']['mae'], m['fts']['rmse'], m['fts']['mape']]
        nn_values = [m['nn']['mae'], m['nn']['rmse'], m['nn']['mape']]
        
        x = np.arange(len(metrics_names))
        width = 0.25
        
        bars1 = self.ax_bar.bar(x - width, fts_values, width, label='FTS', color=COLORS['fts_color'], alpha=0.8)
        bars2 = self.ax_bar.bar(x, nn_values, width, label='NN', color=COLORS['nn_color'], alpha=0.8)
        
        # Add hybrid bars if available
        if 'hybrid' in m:
            hybrid_values = [m['hybrid']['mae'], m['hybrid']['rmse'], m['hybrid']['mape']]
            bars3 = self.ax_bar.bar(x + width, hybrid_values, width, label='Hybrid', color=COLORS['hybrid_color'], alpha=0.8)
            bars_all = bars1 + bars2 + bars3
        else:
            bars_all = bars1 + bars2
        
        self.ax_bar.set_ylabel('Value')
        self.ax_bar.set_title('Model Performance Comparison (Lower is Better)')
        self.ax_bar.set_xticks(x)
        self.ax_bar.set_xticklabels(metrics_names)
        self.ax_bar.legend()
        self.ax_bar.grid(axis='y', alpha=0.3)
        self.ax_bar.set_facecolor(COLORS['card_bg'])
        
        for bar in bars_all:
            height = bar.get_height()
            self.ax_bar.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)
        
        self.bar_canvas.draw()
        
        # Update error distribution (placeholder - will be updated with actual data)
        self.ax_error.clear()
        self.ax_error.text(0.5, 0.5, 'Train models to see\nerror distribution', 
                          ha='center', va='center', transform=self.ax_error.transAxes,
                          fontsize=12, color=COLORS['dark'])
        self.ax_error.set_facecolor(COLORS['card_bg'])
        self.ax_error.axis('off')
        self.error_canvas.draw()
        
        # Update prediction comparison
        self.ax_pred.clear()
        self.ax_pred.text(0.5, 0.5, 'Train models to see\nprediction comparison', 
                         ha='center', va='center', transform=self.ax_pred.transAxes,
                         fontsize=12, color=COLORS['dark'])
        self.ax_pred.set_facecolor(COLORS['card_bg'])
        self.ax_pred.axis('off')
        self.pred_canvas.draw()
    
    def update_charts(self, ts, actual, fts_preds, nn_preds, hybrid_preds=None, window=50):
        """Update charts with prediction data."""
        if ts is None or len(ts) == 0:
            return
        
        # Use last N samples for clarity
        n = min(window, len(ts))
        idx_start = len(ts) - n
        
        ts_subset = ts[idx_start:]
        actual_subset = actual[idx_start:]
        fts_subset = fts_preds[idx_start:]
        nn_subset = nn_preds[idx_start:]
        hybrid_subset = hybrid_preds[idx_start:] if hybrid_preds is not None else None
        
        # Update prediction comparison
        self.ax_pred.clear()
        self.ax_pred.plot(range(n), actual_subset, 'o-', label='Actual', 
                         color=COLORS['success'], linewidth=2, markersize=4)
        self.ax_pred.plot(range(n), fts_subset, 's--', label='FTS Pred', 
                         color=COLORS['fts_color'], linewidth=2, markersize=4, alpha=0.8)
        self.ax_pred.plot(range(n), nn_subset, '^--', label='NN Pred', 
                         color=COLORS['nn_color'], linewidth=2, markersize=4, alpha=0.8)
        if hybrid_subset is not None:
            self.ax_pred.plot(range(n), hybrid_subset, 'd--', label='Hybrid Pred', 
                             color=COLORS['hybrid_color'], linewidth=2, markersize=4, alpha=0.8)
        
        self.ax_pred.set_xlabel('Sample Index')
        self.ax_pred.set_ylabel('Power Output')
        self.ax_pred.set_title('Actual vs Predictions (Last {} Samples)'.format(n))
        self.ax_pred.legend(loc='upper right')
        self.ax_pred.grid(True, alpha=0.3)
        self.ax_pred.set_facecolor(COLORS['card_bg'])
        self.pred_canvas.draw()
        
        # Update error distribution
        fts_errors = np.abs(actual_subset - fts_subset)
        nn_errors = np.abs(actual_subset - nn_subset)
        
        self.ax_error.clear()
        self.ax_error.hist(fts_errors, bins=15, alpha=0.6, label='FTS Errors', 
                          color=COLORS['fts_color'], edgecolor='white')
        self.ax_error.hist(nn_errors, bins=15, alpha=0.6, label='NN Errors', 
                          color=COLORS['nn_color'], edgecolor='white')
        if hybrid_subset is not None:
            hybrid_errors = np.abs(actual_subset - hybrid_subset)
            self.ax_error.hist(hybrid_errors, bins=15, alpha=0.6, label='Hybrid Errors', 
                              color=COLORS['hybrid_color'], edgecolor='white')
        self.ax_error.set_xlabel('Absolute Error')
        self.ax_error.set_ylabel('Frequency')
        self.ax_error.set_title('Error Distribution')
        self.ax_error.legend()
        self.ax_error.grid(True, alpha=0.3)
        self.ax_error.set_facecolor(COLORS['card_bg'])
        self.error_canvas.draw()

class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multivariate Forecast GUI (FTS univariate + NN multivariate + Hybrid FTS+MLP)")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Set modern style
        self._setup_styles()
        
        self.pipeline = ForecastPipeline()
        self.data_path = DEFAULT_DATA_PATH
        
        # Main container with scrollbar
        main_container = tk.Frame(root, bg=COLORS['bg'])
        main_container.pack(fill="both", expand=True)
        
        # Header
        header_frame = tk.Frame(main_container, bg=COLORS['primary'])
        header_frame.pack(fill="x", padx=20, pady=15)
        
        title_label = tk.Label(header_frame, text="🌬️ Integration of fuzzing reasoning and neural network for non-stationary time series data", 
                              font=("Segoe UI", 20, "bold"),
                              fg=COLORS['white'], bg=COLORS['primary'])
        title_label.pack(side="left")
        
        subtitle_label = tk.Label(header_frame, 
                                 text="FTS (Univariate) + Neural Network (Multivariate) + Hybrid (FTS+MLP)",
                                 font=("Segoe UI", 10),
                                 fg=COLORS['light'], bg=COLORS['primary'])
        subtitle_label.pack(side="left", padx=20)
        
        # Content area with notebook (tabs)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        
        # Tab 1: Dashboard
        dashboard_frame = tk.Frame(self.notebook, bg=COLORS['bg'])
        self.notebook.add(dashboard_frame, text="📊 Dashboard")
        
        # Metrics dashboard
        self.metrics_dashboard = MetricsDashboard(dashboard_frame)
        self.metrics_dashboard.pack(fill="both", expand=True)
        
        # Tab 2: Data & Training
        data_frame = tk.Frame(self.notebook, bg=COLORS['bg'])
        self.notebook.add(data_frame, text="📁 Data & Training")
        self._build_data_tab(data_frame)
        
        # Tab 3: Predictions
        predict_frame = tk.Frame(self.notebook, bg=COLORS['bg'])
        self.notebook.add(predict_frame, text="🔮 Predictions")
        self._build_predict_tab(predict_frame)
        
        # Tab 4: Dashboard (Plotly)
        dash_frame = tk.Frame(self.notebook, bg=COLORS['bg'])
        self.notebook.add(dash_frame, text="📈 Plotly Dashboard")
        self._build_dash_tab(dash_frame)
        
        # Status bar
        status_frame = tk.Frame(main_container, bg=COLORS['dark'])
        status_frame.pack(fill="x", side="bottom", padx=10, pady=5)
        self.status_label = tk.Label(status_frame, text="Ready", 
                                     font=("Segoe UI", 9),
                                     fg=COLORS['light'], bg=COLORS['dark'])
        self.status_label.pack(side="left")
        
        # Log console
        log_frame = tk.Frame(main_container, bg=COLORS['bg'])
        log_frame.pack(fill="x", side="bottom", padx=10, pady=(5, 10))
        
        log_label = tk.Label(log_frame, text="📋 Console Log", 
                            font=("Segoe UI", 10, "bold"),
                            fg=COLORS['dark'], bg=COLORS['bg'])
        log_label.pack(anchor="w")
        
        self.logbox = tk.Text(log_frame, height=8, state="disabled", 
                             font=("Consolas", 9), bg=COLORS['white'])
        self.logbox.pack(fill="x", pady=(5, 0))
    
    def _setup_styles(self):
        """Setup custom ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button styles
        style.configure("TButton", font=("Segoe UI", 9), padding=6)
        style.map("TButton", 
                 background=[('active', COLORS['primary']),
                            ('!disabled', COLORS['light'])],
                 foreground=[('active', COLORS['white']),
                            ('!disabled', COLORS['dark'])])
        
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"), 
                       background=COLORS['primary'], foreground=COLORS['white'])
        style.map("Accent.TButton", 
                 background=[('active', COLORS['dark']),
                            ('!disabled', COLORS['primary'])])
        
        # Configure label styles
        style.configure("TLabel", font=("Segoe UI", 9))
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        
        # Configure notebook styles
        style.configure("TNotebook", background=COLORS['bg'])
        style.configure("TNotebook.Tab", font=("Segoe UI", 9), padding=[10, 5])
    
    def _build_data_tab(self, parent):
        """Build the data and training tab."""
        container = tk.Frame(parent, bg=COLORS['bg'])
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Data loading section
        data_card = ModernCard(container, title="📂 Load Dataset")
        data_card.pack(fill="x", pady=(0, 15))
        
        pathrow = tk.Frame(data_card.content, bg=COLORS['card_bg'])
        pathrow.pack(fill="x", pady=5)
        
        tk.Label(pathrow, text="Data path:", font=("Segoe UI", 9, "bold"),
                fg=COLORS['dark'], bg=COLORS['card_bg']).pack(side="left")
        self.path_var = tk.StringVar(value=self.data_path)
        path_entry = ttk.Entry(pathrow, textvariable=self.path_var, width=60)
        path_entry.pack(side="left", padx=10, fill="x", expand=True)
        
        btn_frame = tk.Frame(pathrow, bg=COLORS['card_bg'])
        btn_frame.pack(side="left", padx=10)
        
        ttk.Button(btn_frame, text="Browse", command=self.browse_file).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Load", command=self.load_data).pack(side="left", padx=2)
        
        # Training section
        train_card = ModernCard(container, title="⚙️ Model Training")
        train_card.pack(fill="x", pady=(0, 15))
        
        train_frame = tk.Frame(train_card.content, bg=COLORS['card_bg'])
        train_frame.pack(fill="x", pady=10)
        
        ttk.Button(train_frame, text="🚀 Train FTS + NN", 
                  command=self.train_models, style="Accent.TButton").pack(side="left", padx=5)
        ttk.Button(train_frame, text="📊 Show Plots", 
                  command=self.show_plots).pack(side="left", padx=5)
        ttk.Button(train_frame, text="💾 Save Results", 
                  command=self.save_results).pack(side="left", padx=5)
        
        # Progress indicator
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(train_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(side="left", padx=20, fill="x", expand=True)
        
        # Info label
        info_frame = tk.Frame(container, bg=COLORS['bg'])
        info_frame.pack(fill="x")
        self.data_info_label = tk.Label(info_frame, text="No data loaded", 
                                        font=("Segoe UI", 9),
                                        fg=COLORS['dark'], bg=COLORS['bg'])
        self.data_info_label.pack(anchor="w")
    
    def _build_predict_tab(self, parent):
        """Build the predictions tab."""
        container = tk.Frame(parent, bg=COLORS['bg'])
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Row prediction section
        row_card = ModernCard(container, title="🔮 Predict from Dataset Row")
        row_card.pack(fill="x", pady=(0, 15))
        
        row_frame = tk.Frame(row_card.content, bg=COLORS['card_bg'])
        row_frame.pack(fill="x", pady=10)
        
        tk.Label(row_frame, text="Row index (0-based):", 
                font=("Segoe UI", 9), bg=COLORS['card_bg']).pack(side="left")
        self.row_index_var = tk.IntVar(value=0)
        ttk.Entry(row_frame, textvariable=self.row_index_var, width=8).pack(side="left", padx=10)
        ttk.Button(row_frame, text="Predict", 
                  command=self.predict_from_row, style="Accent.TButton").pack(side="left", padx=10)
        
        # Result display
        self.row_result_var = tk.StringVar(value="FTS: -, NN: -, Hybrid: -")
        self.row_result_label = tk.Label(row_frame, textvariable=self.row_result_var,
                                        font=("Segoe UI", 12, "bold"),
                                        fg=COLORS['primary'], bg=COLORS['card_bg'])
        self.row_result_label.pack(side="left", padx=20)
        
        # Manual prediction section
        manual_card = ModernCard(container, title="⌨️ Manual Prediction")
        manual_card.pack(fill="both", expand=True)
        
        manual_frame = tk.Frame(manual_card.content, bg=COLORS['card_bg'])
        manual_frame.pack(fill="both", expand=True, pady=10)
        
        tk.Label(manual_frame, text="Feature values (comma-separated, in order):", 
                font=("Segoe UI", 9), bg=COLORS['card_bg']).pack(anchor="w")
        
        feat_order = ", ".join(FEATURE_COLS)
        tk.Label(manual_frame, text=feat_order, 
                font=("Segoe UI", 8), fg=COLORS['info'], bg=COLORS['card_bg']).pack(anchor="w", pady=(5, 0))
        
        self.manual_var = tk.StringVar()
        ttk.Entry(manual_frame, textvariable=self.manual_var, width=100).pack(fill="x", pady=10)
        
        btn_frame = tk.Frame(manual_frame, bg=COLORS['card_bg'])
        btn_frame.pack(fill="x")
        
        ttk.Button(btn_frame, text="Predict", 
                  command=self.predict_manual, style="Accent.TButton").pack(side="left")
        
        # Manual result
        self.manual_result_var = tk.StringVar(value="FTS: -, NN: -, Hybrid: -")
        self.manual_result_label = tk.Label(manual_frame, textvariable=self.manual_result_var,
                                           font=("Segoe UI", 12, "bold"),
                                           fg=COLORS['secondary'], bg=COLORS['card_bg'])
        self.manual_result_label.pack(side="left", padx=20)
    
    def _build_dash_tab(self, parent):
        """Build the Plotly dashboard tab."""
        container = tk.Frame(parent, bg=COLORS['bg'])
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        dash_card = ModernCard(container, title="🌐 Plotly Interactive Dashboard")
        dash_card.pack(fill="both", expand=True)
        
        btn_frame = tk.Frame(dash_card.content, bg=COLORS['card_bg'])
        btn_frame.pack(fill="x", pady=10)
        
        ttk.Button(btn_frame, text="🚀 Launch Dashboard", 
                  command=self.launch_dashboard, style="Accent.TButton").pack(side="left", padx=10)
        
        self.dash_status = tk.Label(btn_frame, text="Dashboard: not running", 
                                   font=("Segoe UI", 9),
                                   fg=COLORS['warning'], bg=COLORS['card_bg'])
        self.dash_status.pack(side="left", padx=20)
        
        dash_info = tk.Label(dash_card.content, 
                            text="The Plotly dashboard provides interactive charts and detailed metrics.\n"
                                 "Click 'Launch Dashboard' to open it in your browser.",
                            font=("Segoe UI", 9), fg=COLORS['dark'], bg=COLORS['card_bg'])
        dash_info.pack(anchor="w", pady=10)

    def log(self, msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logbox.configure(state="normal")
        self.logbox.insert("end", f"[{ts}] {msg}\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")
        
    def set_status(self, msg):
        """Update status bar message."""
        self.status_label.config(text=msg)

    def browse_file(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Excel", "*.xls;*.xlsx"), ("All", "*.*")])
        if p:
            self.path_var.set(p)
            self.data_path = p

    def load_data(self):
        path = self.path_var.get().strip() or DEFAULT_DATA_PATH
        try:
            df = self.pipeline.load(path)
            self.data_path = path
            self.log(f"Loaded dataset {path} rows={len(df)}")
            self.set_status(f"Loaded: {len(df)} rows from {os.path.basename(path)}")
            self.data_info_label.config(text=f"✅ Loaded: {len(df)} rows | Features: {len(FEATURE_COLS)} | {LABEL_COL}")
            # set row index default to last test row start
            self.row_index_var.set(0)
        except Exception as e:
            self.log(f"Load error: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def train_models(self):
        try:
            if self.pipeline.df is None:
                self.load_data()
            self.set_status("Training FTS model...")
            self.progress_var.set(20)
            self.log("Training FTS...")
            self.pipeline.train_fts(n_intervals=8)
            
            self.set_status("Training Neural Network...")
            self.progress_var.set(40)
            self.log("Training NN (multivariate MLP)...")
            self.pipeline.train_nn(hidden=(128,64), max_iter=500)
            
            self.set_status("Training Hybrid FTS+MLP model...")
            self.progress_var.set(60)
            self.log("Training Hybrid (FTS+MLP)...")
            self.pipeline.train_hybrid(hidden=(64,32), max_iter=600)
            
            self.pipeline.evaluate()
            m = self.pipeline.metrics
            
            self.progress_var.set(100)
            self.set_status("Training complete!")
            self.log(f"Trained. FTS MAE={m['fts']['mae']:.4f}, NN MAE={m['nn']['mae']:.4f}, Hybrid MAE={m.get('hybrid', {}).get('mae', 'N/A'):.4f}")
            
            # Update dashboard metrics
            self.metrics_dashboard.update_metrics(m)
            
            # Update prediction charts
            ts = pd.to_datetime(self.pipeline.test_df[TIMESTAMP_COL])
            actual = self.pipeline.test_df[LABEL_COL].values.astype(float)
            fts = self.pipeline.fts_preds
            nn = self.pipeline.nn_preds
            hybrid = self.pipeline.hybrid_preds
            self.metrics_dashboard.update_charts(ts, actual, fts, nn, hybrid)
            
            # Switch to dashboard tab to show results
            self.notebook.select(0)
            
            self.log("✅ Training complete! Check the Dashboard tab for metrics and charts.")
        except Exception as e:
            self.log("Training error: " + str(e))
            traceback.print_exc()
            self.set_status("Training failed")
            messagebox.showerror("Training error", str(e))

    def show_metrics(self):
        """Switch to dashboard tab to show graphical metrics."""
        if not self.pipeline.metrics:
            messagebox.showwarning("No metrics", "Train models first.")
            return
        self.notebook.select(0)  # Switch to dashboard tab

    def show_plots(self):
        if self.pipeline.fts_preds is None or self.pipeline.nn_preds is None:
            messagebox.showwarning("No predictions", "Train models first.")
            return
        try:
            ts = pd.to_datetime(self.pipeline.test_df[TIMESTAMP_COL])
            actual = self.pipeline.test_df[LABEL_COL].values.astype(float)
            fts = self.pipeline.fts_preds
            nn = self.pipeline.nn_preds
            hybrid = self.pipeline.hybrid_preds

            plt.figure(figsize=(12,5))
            plt.plot(ts, actual, label="Actual")
            plt.plot(ts, fts, label="FTS Pred")
            plt.plot(ts, nn, label="NN Pred")
            if hybrid is not None:
                plt.plot(ts, hybrid, label="Hybrid Pred")
            plt.title("Actual vs Predictions")
            plt.xlabel("Timestamp")
            plt.ylabel(LABEL_COL)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10,4))
            plt.plot(ts, np.abs(actual - fts))
            plt.title("FTS Absolute Error")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10,4))
            plt.plot(ts, np.abs(actual - nn))
            plt.title("NN Absolute Error")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            if hybrid is not None:
                plt.figure(figsize=(10,4))
                plt.plot(ts, np.abs(actual - hybrid))
                plt.title("Hybrid Absolute Error")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
        except Exception as e:
            self.log("Plot error: " + str(e))
            traceback.print_exc()
            messagebox.showerror("Plot error", str(e))

    def save_results(self):
        if self.pipeline.fts_preds is None or self.pipeline.nn_preds is None:
            messagebox.showwarning("No results", "Train models first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv"),("All","*.*")])
        if not out:
            return
        try:
            path = self.pipeline.save_results(out_path=out)
            self.log(f"Saved results to {path}")
            messagebox.showinfo("Saved", f"Saved to {path}")
        except Exception as e:
            self.log("Save error: " + str(e))
            traceback.print_exc()
            messagebox.showerror("Save error", str(e))

    def predict_from_row(self):
        if self.pipeline.df is None:
            messagebox.showwarning("No data", "Load dataset first.")
            return
        idx = int(self.row_index_var.get())
        if idx < 0 or idx >= len(self.pipeline.df):
            messagebox.showerror("Index error", f"Row index out of range [0, {len(self.pipeline.df)-1}]")
            return
        row = self.pipeline.df.iloc[idx]
        row_dict = {col: float(row[col]) for col in FEATURE_COLS}
        try:
            fts_p, nn_p, hybrid_p = self.pipeline.predict_row(row_dict)
            fts_text = f"{fts_p:.4f}" if fts_p is not None else "n/a"
            nn_text = f"{nn_p:.4f}" if nn_p is not None else "n/a"
            hybrid_text = f"{hybrid_p:.4f}" if hybrid_p is not None else "n/a"
            self.row_result_var.set(f"FTS: {fts_text} | NN: {nn_text} | Hybrid: {hybrid_text}")
            self.log(f"Predicted row {idx}: FTS={fts_text}, NN={nn_text}, Hybrid={hybrid_text}")
            self.set_status(f"Row {idx}: FTS={fts_text}, NN={nn_text}, Hybrid={hybrid_text}")
        except Exception as e:
            self.log("Predict row error: " + str(e))
            traceback.print_exc()
            messagebox.showerror("Predict error", str(e))

    def predict_manual(self):
        s = self.manual_var.get().strip()
        if not s:
            messagebox.showwarning("No input", "Enter comma-separated feature values.")
            return
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != len(FEATURE_COLS):
            messagebox.showerror("Input error", f"Expected {len(FEATURE_COLS)} values.")
            return
        try:
            row_dict = {FEATURE_COLS[i]: float(parts[i]) for i in range(len(FEATURE_COLS))}
            fts_p, nn_p, hybrid_p = self.pipeline.predict_row(row_dict)
            fts_text = f"{fts_p:.4f}" if fts_p is not None else "n/a"
            nn_text = f"{nn_p:.4f}" if nn_p is not None else "n/a"
            hybrid_text = f"{hybrid_p:.4f}" if hybrid_p is not None else "n/a"
            self.manual_result_var.set(f"FTS: {fts_text} | NN: {nn_text} | Hybrid: {hybrid_text}")
            self.log(f"Manual predict: FTS={fts_text}, NN={nn_text}, Hybrid={hybrid_text}")
            self.set_status(f"Manual: FTS={fts_text}, NN={nn_text}, Hybrid={hybrid_text}")
        except Exception as e:
            self.log("Manual predict error: " + str(e))
            traceback.print_exc()
            messagebox.showerror("Parse error", str(e))

    # --------------------
    # Dash app creation and run
    # --------------------
    def create_dash(self):
        # prepare preds dataframe if available
        preds_df = None
        if self.pipeline.test_df is not None and self.pipeline.fts_preds is not None and self.pipeline.nn_preds is not None:
            preds_df = pd.DataFrame({
                TIMESTAMP_COL: pd.to_datetime(self.pipeline.test_df[TIMESTAMP_COL]),
                "actual": self.pipeline.test_df[LABEL_COL].values,
                "fts_pred": self.pipeline.fts_preds,
                "nn_pred": self.pipeline.nn_preds,
                "hybrid_pred": self.pipeline.hybrid_preds if self.pipeline.hybrid_preds is not None else [None] * len(self.pipeline.test_df)
            })
        df_all = self.pipeline.df.copy() if self.pipeline.df is not None else None

        app = Dash(__name__)
        app.layout = html.Div([
            html.H3("FTS (univariate) vs NN (multivariate) vs Hybrid (FTS+MLP) Forecast Dashboard"),
            dcc.Graph(id="main_graph"),
            html.Div(id="metrics_html")
        ], style={"width":"95%", "margin":"auto"})

        @app.callback(
            Output("main_graph", "figure"),
            Output("metrics_html", "children"),
            Input("main_graph", "id")  # dummy input to trigger initial render
        )
        def render(_):
            if preds_df is None:
                if df_all is None:
                    return go.Figure(), "No data loaded. Use GUI to load & train."
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_all[TIMESTAMP_COL], y=df_all[LABEL_COL], mode="lines", name=LABEL_COL))
                return fig, "No predictions yet - train in GUI."
            dff = preds_df
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dff[TIMESTAMP_COL], y=dff["actual"], mode="lines+markers", name="actual"))
            fig.add_trace(go.Scatter(x=dff[TIMESTAMP_COL], y=dff["fts_pred"], mode="lines+markers", name="fts_pred"))
            fig.add_trace(go.Scatter(x=dff[TIMESTAMP_COL], y=dff["nn_pred"], mode="lines+markers", name="nn_pred"))
            if dff["hybrid_pred"].notna().any():
                fig.add_trace(go.Scatter(x=dff[TIMESTAMP_COL], y=dff["hybrid_pred"], mode="lines+markers", name="hybrid_pred"))
            fig.update_layout(title="Actual vs Predictions (test set)", xaxis_title="timestamp", yaxis_title=LABEL_COL)
            if self.pipeline.metrics:
                m = self.pipeline.metrics
                metrics_div = html.Div([
                    html.P(f"FTS -> MAE: {m['fts']['mae']:.4f}, RMSE: {m['fts']['rmse']:.4f}, MAPE: {m['fts']['mape']:.2f}%"),
                    html.P(f"NN  -> MAE: {m['nn']['mae']:.4f}, RMSE: {m['nn']['rmse']:.4f}, MAPE: {m['nn']['mape']:.2f}%"),
                    html.P(f"Hybrid -> MAE: {m.get('hybrid', {}).get('mae', 'N/A'):.4f}, RMSE: {m.get('hybrid', {}).get('rmse', 'N/A'):.4f}, MAPE: {m.get('hybrid', {}).get('mape', 'N/A'):.2f}%")
                ])
            else:
                metrics_div = "No metrics yet."
            return fig, metrics_div

        return app

    def _run_dash(self, port=8050):
        try:
            app = self.create_dash()
            self.log(f"Starting Dash on http://127.0.0.1:{port}")
            app.run_server(port=port, debug=False, use_reloader=False)
        except OSError as e:
            self.log(f"Dash error (port issue): {e}")
        except Exception as e:
            self.log("Dash runtime error: " + str(e))
            traceback.print_exc()

    def launch_dashboard(self):
        t = threading.Thread(target=self._run_dash, daemon=True)
        t.start()
        url = "http://127.0.0.1:8050"
        try:
            webbrowser.open_new_tab(url)
            self.dash_status.configure(text=f"Dashboard running at {url}")
            self.log("Opened Plotly Dash in browser.")
        except Exception as e:
            self.dash_status.configure(text=f"Dashboard thread started (open {url})")
            self.log("Failed to auto-open browser: " + str(e))

# --------------------
# Run application
# --------------------
def main():
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
