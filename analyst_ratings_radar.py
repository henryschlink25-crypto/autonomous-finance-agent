"""
analyst_ratings_radar.py
========================
Wall-Street-Analysten-Konsens & Kursziel-Potenzial Radar.

Erfasst institutionelle Expertenmeinungen:
1. Analysten-Empfehlungs-Konsens (recommendationKey):
   - "strong_buy", "buy", "hold", "underperform", "sell".
2. Kursziel-Potenzial (Target Price Upside):
   - Mittleres Kursziel (targetMeanPrice), Median & High-Target.
   - Upside-Potenzial % = (targetMeanPrice - currentPrice) / currentPrice * 100.
3. Analysten-Abdeckung (numberOfAnalystOpinions):
   - Je mehr Analysten (z. B. 40-60 Wall Street Banken bei NVDA/AAPL), desto verlässlicher
     ist das Kursziel-Cluster.
4. Konsens-Score (-1.0 bis +1.0) zur Einbindung in den ADAS-Schwarm und das Red-Team.
"""

from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional
import yfinance as yf

logger = logging.getLogger("autonomous_agent.analyst_ratings")

CACHE_DIR = Path("data/analyst_ratings_cache")
CACHE_EXPIRY_SEC = 3600 * 6  # 6 Stunden Cache


@dataclass
class AnalystRatingSnapshot:
    symbol: str
    timestamp: float
    current_price: float
    consensus_rating: str  # STRONG_BUY | BUY | HOLD | UNDERPERFORM | SELL | NO_COVERAGE
    target_mean_price: Optional[float]
    target_high_price: Optional[float]
    target_low_price: Optional[float]
    target_upside_pct: Optional[float]  # z. B. +32.5%
    analyst_count: int
    consensus_score: float  # -1.0 bis +1.0
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


class AnalystRatingsRadar:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}_ratings.json"

    def get_ratings(self, symbol: str, current_price: Optional[float] = None) -> AnalystRatingSnapshot:
        sym = symbol.upper().strip()
        cache_file = self._get_cache_path(sym)

        # 1. Cache prüfen
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if time.time() - data.get("timestamp", 0) < CACHE_EXPIRY_SEC:
                    return AnalystRatingSnapshot(**data)
            except Exception as e:
                logger.debug(f"Ratings Cache Error für {sym}: {e}")

        # 2. Live-Abfrage via yfinance
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}

            # Krypto oder ETFs haben keine klassischen Aktien-Analysten
            quote_type = info.get("quoteType", "").upper()
            if quote_type in ["ETF", "MUTUALFUND", "CRYPTOCURRENCY"] or "USD" in sym:
                snap = AnalystRatingSnapshot(
                    symbol=sym,
                    timestamp=time.time(),
                    current_price=current_price or 0.0,
                    consensus_rating="NO_COVERAGE",
                    target_mean_price=None,
                    target_high_price=None,
                    target_low_price=None,
                    target_upside_pct=None,
                    analyst_count=0,
                    consensus_score=0.0,
                    summary=f"{quote_type} Basket / Index - Keine Wall-Street-Einzelkursziele",
                )
                self._save_cache(snap)
                return snap

            rec_raw = str(info.get("recommendationKey", "none")).lower()
            t_mean = info.get("targetMeanPrice")
            t_high = info.get("targetHighPrice")
            t_low = info.get("targetLowPrice")
            n_analysts = int(info.get("numberOfAnalystOpinions", 0) or 0)

            c_price = current_price or float(info.get("currentPrice", 0.0) or info.get("regularMarketPrice", 0.0) or 0.0)

            # Upside berechnen
            upside_pct = None
            if t_mean and c_price > 0:
                upside_pct = round(((float(t_mean) - c_price) / c_price) * 100.0, 1)

            # Consensus Score & Rating
            score = 0.0
            if "strong_buy" in rec_raw:
                rating = "STRONG_BUY"
                score = 0.85
            elif "buy" in rec_raw:
                rating = "BUY"
                score = 0.55
            elif "hold" in rec_raw:
                rating = "HOLD"
                score = 0.0
            elif "underperform" in rec_raw:
                rating = "UNDERPERFORM"
                score = -0.55
            elif "sell" in rec_raw:
                rating = "SELL"
                score = -0.85
            else:
                rating = "HOLD"
                score = 0.0

            # Kursziel-Bonus/Malus
            if upside_pct is not None:
                if upside_pct >= 25.0:
                    score = min(1.0, score + 0.15)
                elif upside_pct <= -5.0:
                    score = max(-1.0, score - 0.25)

            # Gewichtung nach Analysten-Abdeckung (mehr Analysten = höheres Vertrauen)
            if n_analysts >= 20:
                score *= 1.10
            elif n_analysts < 3:
                score *= 0.60
            score = max(-1.0, min(1.0, round(score, 2)))

            summary = (
                f"Konsens: {rating} ({n_analysts} Analysten) | "
                f"Kursziel: ${t_mean:.2f} ({f'{upside_pct:+.1f}%' if upside_pct is not None else 'N/A'} Upside) | "
                f"Range: ${t_low or 0:.0f} - ${t_high or 0:.0f}"
            )

            snap = AnalystRatingSnapshot(
                symbol=sym,
                timestamp=time.time(),
                current_price=round(c_price, 2),
                consensus_rating=rating,
                target_mean_price=round(float(t_mean), 2) if t_mean else None,
                target_high_price=round(float(t_high), 2) if t_high else None,
                target_low_price=round(float(t_low), 2) if t_low else None,
                target_upside_pct=upside_pct,
                analyst_count=n_analysts,
                consensus_score=score,
                summary=summary,
            )

            self._save_cache(snap)
            return snap

        except Exception as e:
            logger.warning(f"Fehler bei Analysten-Ratings für {sym}: {e}")
            return AnalystRatingSnapshot(
                symbol=sym,
                timestamp=time.time(),
                current_price=current_price or 0.0,
                consensus_rating="HOLD",
                target_mean_price=None,
                target_high_price=None,
                target_low_price=None,
                target_upside_pct=None,
                analyst_count=0,
                consensus_score=0.0,
                summary=f"Analysten-Daten nicht verfügbar: {str(e)[:40]}",
            )

    def _save_cache(self, snap: AnalystRatingSnapshot) -> None:
        try:
            cache_file = self._get_cache_path(snap.symbol)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(snap.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Konnte Ratings-Cache nicht speichern: {e}")
