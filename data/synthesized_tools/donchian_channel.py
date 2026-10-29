# Autonom synthetisierter Code von ADAS Meta-Architect
import numpy as np
import pandas as pd

def calculate_donchian_channel(df_daily, period=20):
    if df_daily is None or len(df_daily) < period + 2:
        return {"donchian_breakout": "NONE", "channel_width_pct": 0.0}
    high = df_daily["High"].astype(float)
    low = df_daily["Low"].astype(float)
    close = df_daily["Close"].astype(float)
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    cur_c = float(close.iloc[-1])
    cur_u = float(upper.iloc[-2]) if len(upper) >= 2 else cur_c
    cur_l = float(lower.iloc[-2]) if len(lower) >= 2 else cur_c
    width_pct = round(((cur_u - cur_l) / cur_c) * 100.0, 2) if cur_c > 0 else 0.0
    status = "UPPER_BREAKOUT" if cur_c >= cur_u else ("LOWER_BREAKDOWN" if cur_c <= cur_l else "INSIDE_CHANNEL")
    return {"donchian_breakout": status, "channel_width_pct": width_pct, "upper_barrier": round(cur_u, 2), "lower_barrier": round(cur_l, 2)}
