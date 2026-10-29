# Autonom synthetisierter Code von ADAS Meta-Architect
import numpy as np
import pandas as pd

def calculate_chaikin_money_flow(df_daily, period=20):
    if df_daily is None or len(df_daily) < period + 2:
        return {"cmf": 0.0, "status": "NEUTRAL", "accumulation": False}
    high = df_daily["High"].astype(float)
    low = df_daily["Low"].astype(float)
    close = df_daily["Close"].astype(float)
    vol = df_daily["Volume"].astype(float)
    hl_diff = high - low
    hl_diff = hl_diff.replace(0, 0.0001)
    mfv = (((close - low) - (high - close)) / hl_diff) * vol
    cmf = mfv.rolling(period).sum() / vol.rolling(period).sum().replace(0, 1)
    val = round(float(cmf.iloc[-1]), 4)
    status = "STRONG_ACCUMULATION" if val > 0.15 else ("DISTRIBUTION" if val < -0.15 else "NEUTRAL")
    return {"cmf": val, "status": status, "accumulation": val > 0.10}
