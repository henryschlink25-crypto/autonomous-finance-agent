# Autonom synthetisierter Code von ADAS Meta-Architect
import numpy as np
import pandas as pd

def calculate_hull_ma(df_daily, period=16):
    if df_daily is None or len(df_daily) < period + 5:
        return {"hma_val": 0.0, "hma_slope": 0.0, "hma_trend": "NEUTRAL"}
    close = df_daily["Close"].astype(float)
    half_p = max(2, int(period / 2))
    sqrt_p = max(2, int(np.sqrt(period)))
    wma_half = close.rolling(half_p).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
    wma_full = close.rolling(period).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(sqrt_p).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)
    val = round(float(hma.iloc[-1]), 2)
    prev = round(float(hma.iloc[-2]), 2) if len(hma) >= 2 else val
    slope = round(float(val - prev), 3)
    trend = "BULLISH" if slope > 0 else ("BEARISH" if slope < 0 else "NEUTRAL")
    return {"hma_val": val, "hma_slope": slope, "hma_trend": trend}
