"""
strategy.py — Signal-Generierung für Trading-Strategie
Basiert auf: 150-Tage SMA + 126-Tage Momentum
"""

import pandas as pd
import numpy as np
from decimal import Decimal as D


def generate_signals(
    df: pd.DataFrame,
    trend_sma_period: int = 150,
    momentum_lookback: int = 126,
    atr_period: int = 14,
    stop_atr_mult: float = 3.0,
) -> pd.DataFrame:
    """
    Generiere Trading-Signale basierend auf:
    1. Trend: Close > SMA(150)
    2. Momentum: Return(126 Tage) > 0
    3. ATR für Stop/Target
    
    Returns: DataFrame mit Signalen und Levels
    """
    df = df.copy()
    
    # 1. Trend-Filter
    df['sma_150'] = df['close'].rolling(window=trend_sma_period).mean()
    df['trend_signal'] = (df['close'] > df['sma_150']).astype(int)
    
    # 2. Momentum-Filter (126-Tage Return)
    df['momentum_return'] = df['close'].pct_change(momentum_lookback)
    df['momentum_signal'] = (df['momentum_return'] > 0).astype(int)
    
    # 3. Combined Entry Signal
    df['signal'] = (df['trend_signal'] == 1) & (df['momentum_signal'] == 1)
    df['signal'] = df['signal'].astype(int)
    
    # 4. ATR für Stop/Target Berechnung
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=atr_period).mean()
    
    # 5. Stop und Target (wenn Signal == 1)
    df['stop_price'] = df['close'] - (df['atr'] * stop_atr_mult)
    df['target_price'] = df['close'] + (df['atr'] * (stop_atr_mult + 1))  # Größer als Stop
    
    # 6. Exit-Signal (vereinfacht: wenn Close unter SMA fällt)
    df['exit_signal'] = ((df['close'] < df['sma_150']) & (df['trend_signal'] == 0)).astype(int)
    
    # 7. Regime (für Journal)
    df['regime'] = 'neutral'
    df.loc[df['trend_signal'] == 1, 'regime'] = 'uptrend'
    df.loc[df['trend_signal'] == 0, 'regime'] = 'downtrend'
    
    return df
