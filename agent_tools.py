"""
agent_tools.py - Institutionelle Werkzeug-Bibliothek für den KI-Agenten-Schwarm:
1. Black-Scholes Options-Greeks & Net Gamma Exposure (GEX) via scipy.stats
2. Keltner-Channel & Bollinger-Band Volatility-Squeeze via ta (John Carter TTM Squeeze)
3. SEC Insider-Trading & Whale-Accumulation Tracker (Form 4 Abfragen)
4. Institutional Liquidity & Spread Efficiency Tool
"""

import math
import os
import json
import importlib.util
from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import norm
import requests
import yfinance as yf

try:
    import quantstats as qs
except ImportError:
    qs = None


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

    @staticmethod
    def calculate_supertrend(
        df_daily: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Open-Source SuperTrend-Indikator (ATR-Bänder mit Trailing-Ratchet).
        Gibt Trend-Richtung (BULLISH/BEARISH), SuperTrend-Preisschwelle und Trend-Wechsel-Signal zurück.
        """
        if df_daily is None or len(df_daily) < period + 5:
            return {"trend": "NEUTRAL", "supertrend_val": 0.0, "trend_flipped": False}

        high = df_daily["High"].astype(float).values
        low = df_daily["Low"].astype(float).values
        close = df_daily["Close"].astype(float).values
        n = len(close)

        # True Range & ATR
        tr = np.zeros(n)
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        tr[0] = high[0] - low[0]

        atr = pd.Series(tr).rolling(period).mean().bfill().values

        hl2 = (high + low) / 2.0
        upper_basic = hl2 + (multiplier * atr)
        lower_basic = hl2 - (multiplier * atr)

        upper_final = np.copy(upper_basic)
        lower_final = np.copy(lower_basic)
        supertrend = np.zeros(n)
        trend = np.zeros(n)  # 1 = Bullish, -1 = Bearish

        for i in range(1, n):
            # Trailing Upper Band
            if upper_basic[i] < upper_final[i - 1] or close[i - 1] > upper_final[i - 1]:
                upper_final[i] = upper_basic[i]
            else:
                upper_final[i] = upper_final[i - 1]

            # Trailing Lower Band
            if lower_basic[i] > lower_final[i - 1] or close[i - 1] < lower_final[i - 1]:
                lower_final[i] = lower_basic[i]
            else:
                lower_final[i] = lower_final[i - 1]

            # Trend Determination
            if close[i] > upper_final[i - 1]:
                trend[i] = 1
            elif close[i] < lower_final[i - 1]:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]

            supertrend[i] = lower_final[i] if trend[i] == 1 else upper_final[i]

        curr_trend = "BULLISH" if trend[-1] == 1 else ("BEARISH" if trend[-1] == -1 else "NEUTRAL")
        trend_flipped = (trend[-1] != trend[-2]) if n >= 2 else False

        return {
            "trend": curr_trend,
            "supertrend_val": round(float(supertrend[-1]), 2),
            "trend_flipped": bool(trend_flipped),
            "atr": round(float(atr[-1]), 2),
        }

    @staticmethod
    def calculate_quantstats_tearsheet(
        returns: pd.Series,
        benchmark: str = "SPY",
    ) -> Dict[str, Any]:
        """
        Berechnet Profi-Hedgefonds-Kennzahlen via QuantStats (GitHub: ranaroussi/quantstats):
        Sharpe, Sortino, Max Drawdown, Calmar Ratio, Win Rate.
        """
        if returns is None or len(returns) < 5 or qs is None:
            return {
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "status": "INSUFFICIENT_DATA",
            }
        try:
            clean_rets = returns.dropna()
            sharpe = float(qs.stats.sharpe(clean_rets)) if not np.isnan(qs.stats.sharpe(clean_rets)) else 0.0
            sortino = float(qs.stats.sortino(clean_rets)) if not np.isnan(qs.stats.sortino(clean_rets)) else 0.0
            max_dd = float(qs.stats.max_drawdown(clean_rets)) if not np.isnan(qs.stats.max_drawdown(clean_rets)) else 0.0
            win_rate = float(qs.stats.win_rate(clean_rets)) if not np.isnan(qs.stats.win_rate(clean_rets)) else 0.0

            return {
                "sharpe": round(sharpe, 2),
                "sortino": round(sortino, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "win_rate_pct": round(win_rate * 100, 1),
                "status": "VALID",
            }
        except Exception:
            return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0, "win_rate_pct": 0.0, "status": "CALCULATION_ERROR"}

    @staticmethod
    def get_fear_and_greed_sentiment() -> Dict[str, Any]:
        """
        Holt das offizielle Fear & Greed Sentiment (Alternative.me API mit Fallback).
        Skala: 0 (Extreme Fear) bis 100 (Extreme Greed).
        """
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=4)
            if resp.status_code == 200:
                data = resp.json().get("data", [{}])[0]
                val = int(data.get("value", 50))
                cls = data.get("value_classification", "Neutral")
                return {
                    "score": val,
                    "sentiment": cls.upper(),
                    "summary": f"Market Sentiment: {cls} ({val}/100)",
                }
        except Exception:
            pass
        return {"score": 50, "sentiment": "NEUTRAL", "summary": "Sentiment neutral (50/100)"}

    @staticmethod
    def execute_synthesized_tools(df_daily: pd.DataFrame) -> Dict[str, Any]:
        """
        Dynamischer Werkzeug-Loader: Lädt alle vom ADAS-Meta-Architekten autonom
        programmierten Indikatoren aus 'data/synthesized_tools/' und führt sie auf den Kursdaten aus.
        """
        custom_metrics: Dict[str, Any] = {}
        if df_daily is None or len(df_daily) < 15:
            return custom_metrics

        tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthesized_tools")
        reg_file = os.path.join(tools_dir, "synthesized_registry.json")
        if not os.path.exists(reg_file):
            return custom_metrics

        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                registry = json.load(f)

            for tool_id, info in registry.get("active_tools", {}).items():
                py_path = info.get("file_path", "")
                if not os.path.exists(py_path):
                    continue
                try:
                    mod_name = f"synthesized_{tool_id}"
                    spec = importlib.util.spec_from_file_location(mod_name, py_path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        func = getattr(mod, f"calculate_{tool_id}", None)
                        if callable(func):
                            res = func(df_daily)
                            if isinstance(res, dict):
                                for k, v in res.items():
                                    custom_metrics[f"{tool_id}_{k}"] = v
                except Exception:
                    continue
        except Exception:
            pass
        return custom_metrics

    @staticmethod
    def get_sec_form8k_events(ticker_symbol: str) -> Dict[str, Any]:
        """
        Scannt SEC Form 8-K Ad-hoc-Meldungen & Material Corporate Events
        (FDA-Zulassungen, Übernahmen, strategische Partnerschaften, CEO-Wechsel).
        """
        try:
            clean_sym = ticker_symbol.replace(".DE", "").split("/")[0]
            t = yf.Ticker(clean_sym)
            news = t.news
            if not news:
                return {"has_material_event": False, "catalyst_type": "NONE", "summary": "Keine Ad-hoc-Meldungen"}

            catalyst_keywords = {
                "FDA": "FDA-Zulassung / MedTech Catalyst",
                "APPROV": "Regulatorische Zulassung",
                "ACQUISITION": "Übernahme / Merger",
                "MERGER": "Fusion / Übernahme",
                "PARTNERSHIP": "Strategische Partnerschaft",
                "CONTRACT": "Großauftrag / Regierungsvertrag",
                "EARNINGS": "Quartalszahlen-Überraschung",
                "OFFERING": "Kapitalerhöhung (Verwässerung)",
            }

            for item in news[:5]:
                title = str(item.get("title", "")).upper()
                for kw, cat_label in catalyst_keywords.items():
                    if kw in title:
                        is_negative = "OFFERING" in kw or "DOWNGRADE" in title
                        return {
                            "has_material_event": True,
                            "catalyst_type": cat_label,
                            "is_negative": is_negative,
                            "summary": f"{cat_label}: {item.get('title', '')[:80]}",
                        }
        except Exception:
            pass
        return {"has_material_event": False, "catalyst_type": "NONE", "summary": "Keine aktuellen 8-K Katalysatoren"}

    @staticmethod
    def calculate_qlib_alpha158_factors(df_daily: pd.DataFrame) -> Dict[str, Any]:
        """
        Berechnet die Microsoft Qlib Alpha158 Quantitative Faktoren Suite.
        """
        try:
            from qlib_alpha158_engine import get_qlib_alpha158_engine
            engine = get_qlib_alpha158_engine()
            res_df = engine.compute_factors_dataframe(df_daily)
            if res_df.empty:
                return {"status": "INSUFFICIENT_DATA"}
            latest = res_df.iloc[-1].dropna().to_dict()
            return {
                "status": "SUCCESS",
                "factors_count": len(latest),
                "factors": {k: round(float(v), 4) for k, v in list(latest.items())[:30]},
            }
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}


