# Autonom synthetisierter Code von ADAS Meta-Architect
import numpy as np
import pandas as pd

def calculate_twiggs_mf(df_daily, period=21):
    if df_daily is None or len(df_daily) < period + 3:
        return {"tmf": 0.0, "status": "NEUTRAL"}
    high = df_daily["High"].astype(float)
    low = df_daily["Low"].astype(float)
    close = df_daily["Close"].astype(float)
    vol = df_daily["Volume"].astype(float)
    prev_close = close.shift(1)
    true_low = pd.concat([low, prev_close], axis=1).min(axis=1)
    true_high = pd.concat([high, prev_close], axis=1).max(axis=1)
    tr = (true_high - true_low).replace(0, 0.0001)
    adv = ((2 * close - true_high - true_low) / tr) * vol
    alpha = 2.0 / (period + 1)
    ema_adv = adv.ewm(alpha=alpha, adjust=False).mean()
    ema_vol = vol.ewm(alpha=alpha, adjust=False).mean().replace(0, 1)
    tmf = round(float((ema_adv / ema_vol).iloc[-1]), 4)
    stat = "BULLISH_FLOW" if tmf > 0.05 else ("BEARISH_OUTFLOW" if tmf < -0.05 else "NEUTRAL")
    return {"tmf": tmf, "status": stat}
