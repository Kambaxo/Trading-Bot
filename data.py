"""
data.py — Daten-Modul für Tester und Demo-Daten
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_ohlcv_csv(filepath: str) -> pd.DataFrame:
    """Lade OHLCV-Daten aus CSV."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def generate_synthetic_ohlcv(
    start_date: str = "2024-01-01",
    n_days: int = 250,
    seed: int = 42,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """Generiere synthetische OHLCV-Daten für Demo."""
    np.random.seed(seed)
    
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    
    # Random Walk für Preise
    returns = np.random.normal(0.0005, 0.015, n_days)  # 0.05% mean, 1.5% std
    close_prices = initial_price * np.exp(np.cumsum(returns))
    
    # OHLC
    opens = close_prices * (1 + np.random.normal(0, 0.005, n_days))
    highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    volumes = np.random.randint(1_000_000, 5_000_000, n_days)
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes,
    }, index=dates)
    
    return df
