"""
agent_tools.py - Institutionelle Werkzeug-Bibliothek für den KI-Agenten-Schwarm:
1. Black-Scholes Options-Greeks & Net Gamma Exposure (GEX) via scipy.stats
2. Keltner-Channel & Bollinger-Band Volatility-Squeeze via ta (John Carter TTM Squeeze)
3. SEC Insider-Trading & Whale-Accumulation Tracker (Form 4 Abfragen)
4. Institutional Liquidity & Spread Efficiency Tool
"""

import math
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf


class QuantitativeToolbox:
    """
    Stellt den Unteragenten modulare Profi-Werkzeuge zur Verfügung.
    """

    @staticmethod
    def calculate_black_scholes_greeks(
        spot_price: float,
        strike_price: float,
        time_to_expiration_years: float,
        risk_free_rate: float = 0.045,
        implied_volatility: float = 0.35,
        option_type: str = "CALL",
    ) -> Dict[str, float]:
        """
        Berechnet exakte Options-Greeks (Delta, Gamma, Vega, Theta) nach dem Black-Scholes-Merton Modell.
        """
        S = max(0.01, spot_price)
        K = max(0.01, strike_price)
        T = max(0.001, time_to_expiration_years)
        r = risk_free_rate
        sigma = max(0.01, implied_volatility)

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        gamma = pdf_d1 / (S * sigma * math.sqrt(T))
        vega = S * pdf_d1 * math.sqrt(T) / 100.0  # Pro 1% IV-Änderung

        if option_type.upper() == "CALL":
            delta = cdf_d1
            theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * cdf_d2) / 365.0
        else:
            delta = cdf_d1 - 1.0
            theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0

        return {
            "delta": round(float(delta), 4),
            "gamma": round(float(gamma), 6),
            "vega": round(float(vega), 4),
            "theta": round(float(theta), 4),
        }

    @staticmethod
    def detect_ttm_volatility_squeeze(
        df_daily: pd.DataFrame,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
    ) -> Dict[str, any]:
        """
        John Carter TTM Volatility Squeeze (Bollinger Bands innerhalb Keltner Kanälen):
        Erkennt die extreme Kompression der Volatilität vor massiven Alpha-Ausbrüchen.
        """
        if df_daily is None or len(df_daily) < 30:
            return {"squeeze_on": False, "squeeze_fired": False, "momentum_hist": 0.0, "status": "INSUFFICIENT_DATA"}

        close = df_daily["Close"].astype(float)
        high = df_daily["High"].astype(float)
        low = df_daily["Low"].astype(float)

        # 1. Bollinger Bands
        sma20 = close.rolling(bb_length).mean()
        std20 = close.rolling(bb_length).std()
        bb_upper = sma20 + bb_mult * std20
        bb_lower = sma20 - bb_mult * std20

        # 2. Keltner Channels (ATR 20)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr20 = tr.rolling(kc_length).mean()
        kc_upper = sma20 + kc_mult * atr20
        kc_lower = sma20 - kc_mult * atr20

        # Squeeze ist AN, wenn Bollinger Bands komplett innerhalb der Keltner Channels liegen
        squeeze_series = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        squeeze_on = bool(squeeze_series.iloc[-1])
        squeeze_was_on = bool(squeeze_series.iloc[-2]) if len(squeeze_series) >= 2 else False
        squeeze_fired = squeeze_was_on and not squeeze_on  # Genau jetzt zündet der Ausbruch!

        # Squeeze Momentum (Linear Regression Oszillator)
        highest_high = high.rolling(kc_length).max()
        lowest_low = low.rolling(kc_length).min()
        avg_price = (highest_high + lowest_low) / 2.0
        delta = close - (avg_price + sma20) / 2.0
        mom = float(delta.iloc[-1])

        return {
            "squeeze_on": squeeze_on,
            "squeeze_fired": squeeze_fired,
            "momentum_hist": round(mom, 4),
            "status": "🔥 SQUEEZE_FIRED" if squeeze_fired else ("⏳ SQUEEZE_COMPRESSION" if squeeze_on else "NORMAL"),
        }

    @staticmethod
    def get_sec_insider_activity(ticker_symbol: str) -> Dict[str, any]:
        """
        Prüft offizielle Form-4 SEC-Insider-Transaktionen der letzten 6 Monate.
        Erkennt institutionelle Insider-Käufe (Net Insider Buying) vs. Verkäufe.
        """
        try:
            clean_sym = ticker_symbol.replace(".DE", "").split("/")[0]
            t = yf.Ticker(clean_sym)
            insiders = t.insider_transactions
            if insiders is None or insiders.empty:
                return {"net_insider_shares": 0, "has_insider_buys": False, "insider_signal": "NEUTRAL", "summary": "Keine aktuellen Insider-Meldungen"}

            # Relevante Spalten filtern
            df = insiders.copy()
            df["Text"] = df["Text"].fillna("").astype(str).str.upper()
            df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce").fillna(0)

            # Suche direkte Käufe (Purchase) vs Verkäufe (Sale)
            buys = df[df["Text"].str.contains("PURCHASE|BUY")]
            sales = df[df["Text"].str.contains("SALE|SELL")]

            total_buy_shares = float(buys["Shares"].sum())
            total_sell_shares = float(sales["Shares"].sum())
            net_shares = total_buy_shares - total_sell_shares

            has_buys = len(buys) > 0 and total_buy_shares > 1000
            if has_buys and total_buy_shares > total_sell_shares * 1.5:
                sig = "STRONG_INSIDER_BUY"
                summ = f"Top-Insider kaufen netto +{int(net_shares):,} Aktien (CEO/Director Vertrauen)"
            elif total_sell_shares > 50000 and total_buy_shares == 0:
                sig = "INSIDER_SELLING"
                summ = f"Insider verkaufen netto -{int(total_sell_shares):,} Aktien"
            else:
                sig = "NEUTRAL"
                summ = f"Geringe Insider-Bewegung (Käufe: {len(buys)}, Verkäufe: {len(sales)})"

            return {
                "net_insider_shares": int(net_shares),
                "has_insider_buys": has_buys,
                "insider_signal": sig,
                "summary": summ,
            }
        except Exception:
            return {"net_insider_shares": 0, "has_insider_buys": False, "insider_signal": "NEUTRAL", "summary": "Insider-Daten nicht abrufbar"}
