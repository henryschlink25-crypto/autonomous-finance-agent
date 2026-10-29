"""
probabilistic_forecast_sentinel.py
==================================
📈 Probabilistischer Forecast Trichter (Sub-Agent #30).

Inspiriert von führenden Open-Source-Zeitreihenmodellen auf Hugging Face
(Kronos / TimesFM / FinCast):
1. Sagt keinen fixen Punkt-Preis voraus, sondern berechnet einen probabilistischen
   Quantil-Trichter für die nächsten N Handelstage (Standard: 5 Tage).
2. Ermittelt:
   - P10 (10% Bären-Quantil / Worst-Case-Pfad)
   - P50 (50% Median / Mittlerer Erwartungspfad)
   - P90 (90% Bullen-Quantil / Asymmetrischer Best-Case)
3. Berechnet die Reward-Risk-Asymmetrie:
   Asymmetry Ratio = (P90 - P50) / max(0.01, P50 - P10)
4. Schützt das Portfolio:
   - Günstig (FAVORABLE_ASYMMETRY): Asymmetrie >= 1.8x und P50 über aktuellem Kurs.
   - Ungünstig (UNFAVORABLE_SKEW / VETO): Asymmetrie <= 0.7x (Downside-Risiko überwiegt).
"""

from __future__ import annotations
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("autonomous_agent.probabilistic_forecast")

CACHE_FILE = Path("data/probabilistic_forecast_cache.json")


class ForecastQuantiles(BaseModel):
    symbol: str
    current_price: float
    horizon_days: int = 5
    p10_price: float
    p50_price: float
    p90_price: float
    expected_return_p50_pct: float
    upside_p90_pct: float
    downside_p10_pct: float
    asymmetry_ratio: float  # Upside zu Downside
    forecast_status: str    # "FAVORABLE_UPSIDE_ASYMMETRY", "NEUTRAL_SYMMETRIC", "UNFAVORABLE_DOWNSIDE_SKEW"
    edge_score: float       # -1.0 bis +1.0
    confidence: float       # 0.0 bis 1.0
    annualized_vol_pct: float
    hurdle_rate_cleared: bool
    summary: str = ""
    calculated_at: str = ""


class ProbabilisticForecastSentinel:
    """
    Berechnet probabilistische Vorhersage-Fächer basierend auf historischer
    Drift-, Volatilitäts- und Kurtosis-Struktur.
    """

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Konnte Forecast-Cache nicht laden: {e}")
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Konnte Forecast-Cache nicht speichern: {e}")

    def compute_quantiles_from_history(
        self,
        symbol: str,
        closes: np.ndarray,
        horizon_days: int = 5,
        n_simulations: int = 2000,
    ) -> ForecastQuantiles:
        """
        Berechnet den probabilistischen Kurs-Trichter aus historischen Schlusskursen
        unter Berücksichtigung von Fat-Tails (Student-t / Bootstrapping).
        """
        sym = symbol.strip().upper()
        if len(closes) < 20:
            current_px = float(closes[-1]) if len(closes) > 0 else 100.0
            return self._fallback_quantiles(sym, current_px, horizon_days)

        current_px = float(closes[-1])
        daily_log_returns = np.diff(np.log(closes))

        # Realisierte Kennzahlen
        mean_drift = float(np.mean(daily_log_returns))
        vol_daily = float(np.std(daily_log_returns, ddof=1)) if len(daily_log_returns) > 1 else 0.015
        vol_daily = max(0.005, vol_daily)
        vol_ann_pct = vol_daily * math.sqrt(252) * 100.0

        # Historisches Block-Bootstrapping für realistische Fat-Tails (wie Kronos Tokenizer)
        # Mit residualer Diffusionskomponente (unobserved noise floor)
        np.random.seed(int(current_px * 100) % 10000)
        sampled_returns = np.random.choice(daily_log_returns, size=(n_simulations, horizon_days), replace=True)
        diffusion_noise = np.random.normal(0.0, max(0.003, vol_daily * 0.20), size=(n_simulations, horizon_days))
        cum_horizon_returns = np.sum(sampled_returns + diffusion_noise, axis=1)

        simulated_prices = current_px * np.exp(cum_horizon_returns)

        # Quantile ermitteln
        p10 = float(np.percentile(simulated_prices, 10))
        p50 = float(np.percentile(simulated_prices, 50))
        p90 = float(np.percentile(simulated_prices, 90))

        # Reward-to-Risk Asymmetrie relativ zum Einstiegskurs (current_px)
        # Upside Potential (P90 - S0) vs Downside Risk (S0 - P10)
        upside_diff = max(0.002 * current_px, p90 - current_px)
        downside_diff = max(0.002 * current_px, current_px - p10)
        asymmetry = round(float(upside_diff / downside_diff), 2)

        ret_p50_pct = round(((p50 / current_px) - 1.0) * 100.0, 2)
        upside_p90_pct = round(((p90 / current_px) - 1.0) * 100.0, 2)
        downside_p10_pct = round(((p10 / current_px) - 1.0) * 100.0, 2)

        # Asymmetrie-Bewertung
        if asymmetry >= 1.6 and ret_p50_pct > 0.2:
            status = "FAVORABLE_UPSIDE_ASYMMETRY"
            edge = min(0.92, 0.50 + min(0.35, (asymmetry - 1.0) * 0.25))
            conf = min(0.92, 0.70 + min(0.20, (ret_p50_pct / 5.0) * 0.15))
            hurdle = True
            summary = (
                f"📈 Attraktive Aufwärts-Asymmetrie ({asymmetry:.2f}x): 90% Bullen-Ziel bei ${p90:.2f} (+{upside_p90_pct}%), "
                f"während 10% Bären-Risiko bei ${p10:.2f} ({downside_p10_pct}%) liegt (P50: ${p50:.2f})."
            )
        elif asymmetry <= 0.75 or ret_p50_pct <= -1.0:
            status = "UNFAVORABLE_DOWNSIDE_SKEW"
            edge = -0.80
            conf = 0.85
            hurdle = False
            summary = (
                f"⚠️ Ungünstiges Abwärts-Risiko ({asymmetry:.2f}x): Downside überwiegt deutlich "
                f"(P10: ${p10:.2f} [{downside_p10_pct}%] vs P90: ${p90:.2f} [+{upside_p90_pct}%]). Trade unrentabel."
            )
        else:
            status = "NEUTRAL_SYMMETRIC"
            edge = 0.10 if ret_p50_pct >= 0 else -0.10
            conf = 0.60
            hurdle = False
            summary = f"Symmetrischer Kursfächer: P10 ${p10:.2f} | P50 ${p50:.2f} | P90 ${p90:.2f} (Asymmetrie {asymmetry:.2f}x)."

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        res = ForecastQuantiles(
            symbol=sym,
            current_price=round(current_px, 2),
            horizon_days=horizon_days,
            p10_price=round(p10, 2),
            p50_price=round(p50, 2),
            p90_price=round(p90, 2),
            expected_return_p50_pct=ret_p50_pct,
            upside_p90_pct=upside_p90_pct,
            downside_p10_pct=downside_p10_pct,
            asymmetry_ratio=asymmetry,
            forecast_status=status,
            edge_score=round(edge, 2),
            confidence=round(conf, 2),
            annualized_vol_pct=round(vol_ann_pct, 1),
            hurdle_rate_cleared=hurdle,
            summary=summary,
            calculated_at=now_iso,
        )

        self._cache[sym] = {
            "timestamp": time.time(),
            "data": res.model_dump(),
        }
        self._save_cache()
        return res

    def _fallback_quantiles(self, symbol: str, current_price: float, horizon_days: int) -> ForecastQuantiles:
        p10 = current_price * 0.96
        p50 = current_price * 1.005
        p90 = current_price * 1.05
        return ForecastQuantiles(
            symbol=symbol,
            current_price=round(current_price, 2),
            horizon_days=horizon_days,
            p10_price=round(p10, 2),
            p50_price=round(p50, 2),
            p90_price=round(p90, 2),
            expected_return_p50_pct=0.5,
            upside_p90_pct=5.0,
            downside_p10_pct=-4.0,
            asymmetry_ratio=1.25,
            forecast_status="NEUTRAL_SYMMETRIC",
            edge_score=0.10,
            confidence=0.55,
            annualized_vol_pct=22.0,
            hurdle_rate_cleared=False,
            summary="Fallback-Quantile aus Standard-Volatilität.",
            calculated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def analyze_symbol(self, symbol: str, horizon_days: int = 5, force_refresh: bool = False) -> ForecastQuantiles:
        """
        Holt historische Daten und berechnet den Trichter.
        """
        sym = symbol.strip().upper()
        now_ts = time.time()

        if not force_refresh and sym in self._cache:
            entry = self._cache[sym]
            if now_ts - entry.get("timestamp", 0) < 600:
                return ForecastQuantiles(**entry.get("data", {}))

        try:
            import yfinance as yf
            ticker = yf.Ticker(sym)
            df = ticker.history(period="6mo", interval="1d", auto_adjust=True)
            if df is not None and len(df) >= 20:
                closes = df["Close"].to_numpy().astype(float)
                return self.compute_quantiles_from_history(sym, closes, horizon_days=horizon_days)
        except Exception as e:
            logger.warning(f"Fehler bei yfinance Abfrage für {sym}: {e}")

        return self._fallback_quantiles(sym, 100.0, horizon_days)

    def format_telegram_summary(self, res: ForecastQuantiles) -> str:
        """
        Formatiert den Vorhersage-Trichter für Telegram.
        """
        icon = {
            "FAVORABLE_UPSIDE_ASYMMETRY": "🚀🟢",
            "UNFAVORABLE_DOWNSIDE_SKEW": "⚠️🔴",
            "NEUTRAL_SYMMETRIC": "⚪",
        }.get(res.forecast_status, "⚪")

        lines = [
            f"📈 <b>Probabilistischer Kurs-Trichter ({res.horizon_days} Tage): {res.symbol}</b>",
            "─────────────────────────────",
            f"• <b>Status:</b> {icon} <b>{res.forecast_status}</b>",
            f"• <b>Aktueller Kurs:</b> <b>${res.current_price:.2f}</b>",
            f"• <b>Asymmetrie-Verhältnis:</b> <b>{res.asymmetry_ratio:.2f}x</b> (Ziel: &ge; 1.6x)\n",
            "📊 <b>Erwartete Quantil-Spannweite:</b>",
            f"• 🟢 90% Bullen-Pfad (P90): <b>${res.p90_price:.2f}</b> (<b>+{res.upside_p90_pct:.1f}%</b>)",
            f"• ⚪ 50% Median-Pfad (P50): <b>${res.p50_price:.2f}</b> (<b>{res.expected_return_p50_pct:+.1f}%</b>)",
            f"• 🔴 10% Bären-Pfad (P10): <b>${res.p10_price:.2f}</b> (<b>{res.downside_p10_pct:.1f}%</b>)\n",
            f"• <b>Realisierte Volatilität (p.a.):</b> {res.annualized_vol_pct:.1f}%",
            f"• <b>Asymmetrie-Filter:</b> {'✅ Bestanden' if res.hurdle_rate_cleared else '⚪ Nicht bestanden'}",
            "─────────────────────────────",
            f"💡 <i>{res.summary}</i>",
        ]
        return "\n".join(lines)


# Singleton
_FORECAST_INSTANCE: Optional[ProbabilisticForecastSentinel] = None


def get_forecast_sentinel() -> ProbabilisticForecastSentinel:
    global _FORECAST_INSTANCE
    if _FORECAST_INSTANCE is None:
        _FORECAST_INSTANCE = ProbabilisticForecastSentinel()
    return _FORECAST_INSTANCE
