# Autonom synthetisierter Code von ADAS Meta-Architect
import numpy as np
import pandas as pd

def calculate_choppiness_index(df_daily, period=14):
    if df_daily is None or len(df_daily) < period + 3:
        return {"chop": 50.0, "regime": "NEUTRAL"}
    high = df_daily["High"].astype(float)
    low = df_daily["Low"].astype(float)
    close = df_daily["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    sum_tr = tr.rolling(period).sum()
    max_hi = high.rolling(period).max()
    min_lo = low.rolling(period).min()
    range_hl = (max_hi - min_lo).replace(0, 0.0001)
    chop = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
    val = round(float(chop.iloc[-1]), 2)
    regime = "STRONG_TREND" if val < 38.2 else ("CHOPPY_SIDEWAYS" if val > 61.8 else "TRANSITION")
    return {"chop": val, "regime": regime}
