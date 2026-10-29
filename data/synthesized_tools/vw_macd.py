# Autonom synthetisierter Code von ADAS Meta-Architect
import numpy as np
import pandas as pd

def calculate_vw_macd(df_daily, fast=12, slow=26, signal_p=9):
    if df_daily is None or len(df_daily) < slow + signal_p:
        return {"vw_macd_hist": 0.0, "signal": "NEUTRAL"}
    close = df_daily["Close"].astype(float)
    vol = df_daily["Volume"].astype(float)
    pv = close * vol
    vw_fast = pv.rolling(fast).sum() / vol.rolling(fast).sum().replace(0, 1)
    vw_slow = pv.rolling(slow).sum() / vol.rolling(slow).sum().replace(0, 1)
    vw_macd = vw_fast - vw_slow
    vw_signal = vw_macd.rolling(signal_p).mean()
    hist = vw_macd - vw_signal
    val = round(float(hist.iloc[-1]), 4)
    sig = "BULLISH_SURGE" if val > 0 else "BEARISH_FADE"
    return {"vw_macd_hist": val, "signal": sig}
