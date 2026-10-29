# Autonom synthetisierter Code von ADAS Meta-Architect
import numpy as np
import pandas as pd

def calculate_rvi(df_daily, period=14):
    if df_daily is None or len(df_daily) < period + 5:
        return {"rvi": 50.0, "status": "NEUTRAL"}
    close = df_daily["Close"].astype(float)
    std = close.rolling(period).std().bfill()
    chg = close.diff()
    up = (chg > 0).astype(float) * std
    down = (chg < 0).astype(float) * std
    up_avg = up.rolling(period).mean()
    down_avg = down.rolling(period).mean().replace(0, 0.0001)
    rvi = 100.0 - (100.0 / (1.0 + (up_avg / down_avg)))
    val = round(float(rvi.iloc[-1]), 2)
    stat = "VOLATILITY_EXPANSION_UP" if val > 65.0 else ("VOLATILITY_EXPANSION_DOWN" if val < 35.0 else "NORMAL")
    return {"rvi": val, "status": stat}
