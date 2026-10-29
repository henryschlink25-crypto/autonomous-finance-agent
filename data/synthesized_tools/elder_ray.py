# Autonom synthetisierter Code von ADAS Meta-Architect
import numpy as np
import pandas as pd

def calculate_elder_ray(df_daily, period=13):
    if df_daily is None or len(df_daily) < period + 2:
        return {"bull_power": 0.0, "status": "NEUTRAL"}
    high = df_daily["High"].astype(float)
    close = df_daily["Close"].astype(float)
    ema = close.ewm(span=period, adjust=False).mean()
    bull_power = high - ema
    val = round(float(bull_power.iloc[-1]), 3)
    status = "STRONG_BULL_DOMINANCE" if val > 0 else "BEAR_PRESSURE"
    return {"bull_power": val, "status": status}
