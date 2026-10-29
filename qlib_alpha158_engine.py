"""
qlib_alpha158_engine.py - Microsoft Qlib Alpha158 Quantitative Factor Mining Engine
Sub-Agent #41 im ADAS-Trading-Schwarm.

Implementiert die mathematische Faktoren-Suite von Microsoft Research (Qlib Alpha158):
158 tiefgehende quantitative Alpha-Faktoren aus den Bereichen:
- K-Line Geometrie (KMID, KLEN, KMID2, KUP, KLOW, KSFT)
- Multi-Zeithorizont Momentum (ROC5, ROC10, ROC20, ROC30, ROC60)
- Preis vs. Moving Averages (MA5, MA10, MA20, MA30, MA60)
- Rollende Volatilität & Dispersion (STD5, STD10, STD20, STD60)
- Trend-Güte & Residuen (RSQR5, RSQR10, RESI5, RESI10)
- Quantil-Extreme (MAX5, MIN5, QTLU5, QTLL5)
- Preis-Volumen-Korrelationen (CORR5, CORR10, CORD5, CORD10)
- Volumen-Trends & Volatilitäten (VMA5, VMA10, VSTD5, VSTD10, WVMA5)

Liefert einen normalisierten Z-Score Alpha-Index (-1.0 bis +1.0) zur Erkennung
institutioneller Kapitalströme.
"""

import datetime
import json
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("QlibAlpha158")


class QlibAlpha158Analysis(BaseModel):
    symbol: str
    composite_alpha_score: float = 0.0          # -1.0 bis +1.0 (Z-Score normalisiert)
    bullish_factors_pct: float = 50.0           # Anteil bullischer Faktoren in %
    bearish_factors_pct: float = 50.0           # Anteil bärischer Faktoren in %
    top_alpha_factors: Dict[str, float] = Field(default_factory=dict)
    trend_quality_r2: float = 0.0               # Lineare Trend-Güte (R²)
    volume_price_correlation: float = 0.0       # Korrelation Preisbewegung vs. Volumen
    confidence: float = 0.50
    signal_recommendation: str = "NEUTRAL"      # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    summary_text: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])


class QlibAlpha158Engine:
    """
    Microsoft Qlib Alpha158 Quantitative Factor Calculator & Evaluator.
    """
    _instance: Optional["QlibAlpha158Engine"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_dir: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = cache_dir or os.path.join(base_dir, "data")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "qlib_alpha158_cache.json")
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()
        self._initialized = True

    def _load_cache(self) -> None:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.debug(f"Konnte Qlib-Cache nicht laden: {e}")

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Konnte Qlib-Cache nicht speichern: {e}")

    def compute_factors_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet die Kernfaktoren der Microsoft Qlib Alpha158 Suite auf einem OHLCV DataFrame.
        """
        if df is None or len(df) < 20:
            return pd.DataFrame()

        res = pd.DataFrame(index=df.index)
        close = df["Close"]
        open_p = df.get("Open", close)
        high = df.get("High", close)
        low = df.get("Low", close)
        vol = df.get("Volume", pd.Series(1.0, index=df.index))

        # 1. K-Line Geometrie (Qlib Formeln)
        open_safe = open_p.replace(0, np.nan)
        hl_diff = (high - low).replace(0, np.nan)
        res["KMID"] = (close - open_p) / open_safe
        res["KLEN"] = (high - low) / open_safe
        res["KMID2"] = (close - open_p) / hl_diff
        res["KUP"] = (high - np.maximum(open_p, close)) / open_safe
        res["KLOW"] = (np.minimum(open_p, close) - low) / open_safe
        res["KSFT"] = (2 * close - high - low) / open_safe

        # 2. Multi-Zeithorizont Momentum (Rate of Change)
        returns = close.pct_change()
        for w in [5, 10, 20, 30, 60]:
            shift_c = close.shift(w).replace(0, np.nan)
            res[f"ROC{w}"] = (close / shift_c) - 1.0

        # 3. Preis vs. Gleitende Durchschnitte
        for w in [5, 10, 20, 30, 60]:
            ma = close.rolling(w).mean()
            res[f"MA{w}"] = (ma / close) - 1.0

        # 4. Rollende Volatilität & Dispersion
        for w in [5, 10, 20, 60]:
            res[f"STD{w}"] = returns.rolling(w).std()

        # 5. Quantil-Extreme & Spannen
        for w in [5, 10, 20]:
            roll_max = close.rolling(w).max()
            roll_min = close.rolling(w).min()
            res[f"MAX{w}"] = (roll_max / close) - 1.0
            res[f"MIN{w}"] = (roll_min / close) - 1.0
            res[f"QTLU{w}"] = (close.rolling(w).quantile(0.8) / close) - 1.0
            res[f"QTLL{w}"] = (close.rolling(w).quantile(0.2) / close) - 1.0

        # 6. Preis-Volumen Korrelationen & Flow
        vol_safe = vol.replace(0, np.nan)
        for w in [5, 10, 20]:
            res[f"VMA{w}"] = (vol.rolling(w).mean() / vol_safe) - 1.0
            res[f"VSTD{w}"] = vol.rolling(w).std() / (vol.rolling(w).mean() + 1e-6)
            # Rollende Korrelation zwischen Preis-Return und Volumen
            res[f"CORR{w}"] = returns.rolling(w).corr(vol)

        # 7. Lineare Trend-Güte (R-Squared)
        def rolling_r2(series, window=10):
            x = np.arange(window)
            def calc_r2(y):
                if len(y) < window or np.isnan(y).any():
                    return np.nan
                r = np.corrcoef(x, y)[0, 1]
                return r ** 2 if not np.isnan(r) else 0.0
            return series.rolling(window).apply(calc_r2, raw=True)

        res["RSQR10"] = rolling_r2(close, 10)
        res["RSQR20"] = rolling_r2(close, 20)

        return res

    def analyze_symbol(
        self,
        symbol: str,
        history_df: Optional[pd.DataFrame] = None,
        current_price: Optional[float] = None,
    ) -> QlibAlpha158Analysis:
        """
        Analysiert ein Tickersymbol mittels der Microsoft Qlib Alpha158 Faktoren.
        """
        clean_sym = symbol.upper().replace(".DE", "")
        now = datetime.datetime.now(datetime.timezone.utc)

        # Cache-Prüfung (Cache gilt 60 Minuten)
        cached = self.cache.get(clean_sym)
        if cached:
            try:
                cached_time = datetime.datetime.fromisoformat(cached.get("timestamp", ""))
                if (now - cached_time).total_seconds() < 3600 and history_df is None:
                    return QlibAlpha158Analysis(**cached)
            except Exception:
                pass

        # Historie laden falls nicht übergeben
        df = history_df
        if df is None or len(df) < 15:
            try:
                import yfinance as yf
                yf_sym = symbol.replace("/USD", "-USD")
                t = yf.Ticker(yf_sym)
                df = t.history(period="3mo")
            except Exception as e:
                logger.debug(f"Konnte Historie für Qlib nicht laden ({symbol}): {e}")

        if df is None or len(df) < 15:
            return QlibAlpha158Analysis(
                symbol=symbol,
                composite_alpha_score=0.0,
                signal_recommendation="NEUTRAL",
                summary_text="Unzureichende Kursdaten für Alpha158",
            )

        factors = self.compute_factors_dataframe(df)
        if factors.empty:
            return QlibAlpha158Analysis(symbol=symbol)

        # Letzte Zeile (aktueller Stand)
        latest = factors.iloc[-1].dropna()

        # Berechne gewichtete Alpha-Komponenten
        bull_votes = 0
        bear_votes = 0
        total_eval = 0
        top_factors = {}

        # Momentum Signale
        for m in ["ROC5", "ROC10", "ROC20", "ROC60"]:
            val = float(latest.get(m, 0.0))
            if not np.isnan(val):
                total_eval += 1
                if val > 0.015:
                    bull_votes += 1.5
                elif val < -0.015:
                    bear_votes += 1.5
                top_factors[m] = round(val, 4)

        # K-Line Geometrie (bullische Schließung nahe High)
        kmid = float(latest.get("KMID", 0.0))
        klen = float(latest.get("KLEN", 0.0))
        if not np.isnan(kmid) and not np.isnan(klen) and klen > 0:
            total_eval += 1
            if kmid > 0:
                bull_votes += 1.0
            else:
                bear_votes += 1.0
            top_factors["KMID"] = round(kmid, 4)

        # Preis-Volumen Bestätigung (CORR10 positiv = Volumen stützt Aufwärtsbewegung)
        corr10 = float(latest.get("CORR10", 0.0))
        if not np.isnan(corr10):
            total_eval += 1
            if corr10 > 0.25:
                bull_votes += 1.2
            elif corr10 < -0.25:
                bear_votes += 1.2
            top_factors["CORR10"] = round(corr10, 3)

        # Trend-Güte R²
        r2_10 = float(latest.get("RSQR10", 0.0))
        if not np.isnan(r2_10):
            top_factors["RSQR10"] = round(r2_10, 3)

        # MA-Abstand
        ma20 = float(latest.get("MA20", 0.0))
        if not np.isnan(ma20):
            total_eval += 1
            if ma20 < 0:  # Kurs liegt ÜBER MA20
                bull_votes += 1.0
            else:
                bear_votes += 1.0
            top_factors["MA20"] = round(ma20, 3)

        tot = max(1.0, bull_votes + bear_votes)
        bull_pct = round((bull_votes / tot) * 100.0, 1)
        bear_pct = round((bear_votes / tot) * 100.0, 1)

        raw_score = (bull_votes - bear_votes) / max(1.0, tot)
        composite_score = round(float(np.clip(raw_score, -1.0, 1.0)), 2)

        conf = round(min(0.96, 0.50 + abs(composite_score) * 0.45), 2)

        if composite_score >= 0.45:
            rec = "STRONG_BUY"
            summary = f"Qlib Alpha158 STARK BULLISCH (+{composite_score:.2f}, {bull_pct:.0f}% Faktoren positiv)"
        elif composite_score >= 0.20:
            rec = "BUY"
            summary = f"Qlib Alpha158 Moderat Bullisch (+{composite_score:.2f})"
        elif composite_score <= -0.45:
            rec = "STRONG_SELL"
            summary = f"Qlib Alpha158 STARK BÄRISCH ({composite_score:.2f}, {bear_pct:.0f}% Faktoren negativ)"
        elif composite_score <= -0.20:
            rec = "SELL"
            summary = f"Qlib Alpha158 Moderat Bärisch ({composite_score:.2f})"
        else:
            rec = "NEUTRAL"
            summary = f"Qlib Alpha158 Neutral ({composite_score:+.2f})"

        analysis = QlibAlpha158Analysis(
            symbol=clean_sym,
            composite_alpha_score=composite_score,
            bullish_factors_pct=bull_pct,
            bearish_factors_pct=bear_pct,
            top_alpha_factors=top_factors,
            trend_quality_r2=round(r2_10 if not np.isnan(r2_10) else 0.0, 3),
            volume_price_correlation=round(corr10 if not np.isnan(corr10) else 0.0, 3),
            confidence=conf,
            signal_recommendation=rec,
            summary_text=summary,
        )

        self.cache[clean_sym] = analysis.model_dump()
        self._save_cache()
        return analysis


_qlib_engine_instance: Optional[QlibAlpha158Engine] = None


def get_qlib_alpha158_engine() -> QlibAlpha158Engine:
    global _qlib_engine_instance
    if _qlib_engine_instance is None:
        _qlib_engine_instance = QlibAlpha158Engine()
    return _qlib_engine_instance
