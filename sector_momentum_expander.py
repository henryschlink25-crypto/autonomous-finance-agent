"""
sector_momentum_expander.py
===========================
Autonomer Sektor-Rotations- & Relative-Stärke-Radar.

Funktionsweise:
1. Überwacht alle 11 offiziellen US-Sektor-ETFs (SPDR):
   - XLK (Technologie), XLF (Finanzen), XLV (Gesundheit), XLE (Energie),
   - XLI (Industrie), XLY (Konsum Zyklisch), XLP (Konsum Defensiv),
   - XLU (Versorger), XLB (Materialien), XLRE (Immobilien), XLC (Kommunikation).
2. Berechnet die relative Stärke (Alpha) jedes Sektors gegenüber dem S&P 500 (SPY):
   - 5-Tage & 20-Tage Outperformance.
3. Identifiziert die führenden Sektoren (Leader) und schwächsten Sektoren (Laggards).
4. Reicht die Top-Aktien der Gewinner-Sektoren direkt an den Schwarm weiter,
   damit das Kapital immer dorthin fließt, wo institutionelle Liquidität hineinflutet.
"""

from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf

logger = logging.getLogger("autonomous_agent.sector_rotation")

CACHE_DIR = Path("data/sector_momentum_cache")
CACHE_FILE = CACHE_DIR / "sector_rotation_state.json"
CACHE_EXPIRY_SEC = 3600  # 1 Stunde

SECTOR_MAP: Dict[str, Tuple[str, List[str]]] = {
    "XLK": ("Technologie & KI", ["NVDA", "MSFT", "AAPL", "AVGO", "ORCL"]),
    "XLE": ("Energie & Öl", ["XOM", "CVX", "OXY", "SLB"]),
    "XLF": ("Finanzen & Banken", ["JPM", "GS", "MS", "BAC"]),
    "XLI": ("Industrie & Luftfahrt", ["GE", "CAT", "BA", "LMT"]),
    "XLC": ("Kommunikation & Media", ["GOOGL", "META", "NFLX"]),
    "XLY": ("Konsumgüter Zyklisch", ["AMZN", "TSLA", "HD"]),
    "XLV": ("Gesundheit & Biotech", ["LLY", "JNJ", "PFE", "ABBV"]),
    "XLU": ("Versorger (Defensiv)", ["NEE", "SO", "DUK"]),
    "XLB": ("Materialien & Rohstoffe", ["LIN", "APD", "FCX"]),
    "XLRE": ("Immobilien (REITs)", ["PLD", "AMT", "EQIX"]),
    "XLP": ("Konsumgüter Defensiv", ["PG", "KO", "PEP", "COST"]),
}


@dataclass
class SectorPerformance:
    ticker: str
    name: str
    return_5d_pct: float
    relative_strength_vs_spy: float
    rank: int
    top_stocks: List[str]


@dataclass
class SectorRotationSnapshot:
    timestamp: float
    leading_sectors: List[SectorPerformance]
    lagging_sectors: List[SectorPerformance]
    top_alpha_sector: str
    recommended_hot_stocks: List[str]
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


class SectorMomentumExpander:
    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def scan_sector_rotation(self, force_refresh: bool = False) -> SectorRotationSnapshot:
        """
        Scannt die 11 US-Sektoren relativ zum SPY und liefert die Momentum-Leader.
        """
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if time.time() - data.get("timestamp", 0) < CACHE_EXPIRY_SEC:
                    return SectorRotationSnapshot(
                        timestamp=data["timestamp"],
                        leading_sectors=[SectorPerformance(**s) for s in data["leading_sectors"]],
                        lagging_sectors=[SectorPerformance(**s) for s in data["lagging_sectors"]],
                        top_alpha_sector=data["top_alpha_sector"],
                        recommended_hot_stocks=data["recommended_hot_stocks"],
                        summary=data["summary"],
                    )
            except Exception as e:
                logger.debug(f"Sector Cache Read Error: {e}")

        symbols = list(SECTOR_MAP.keys()) + ["SPY"]
        try:
            df = yf.download(symbols, period="1mo", interval="1d", progress=False)["Close"]
            if df is None or df.empty or len(df) < 6:
                return self._create_fallback_snapshot("Keine Sektordaten geladen")

            # 5-Tage Rendite von SPY
            spy_close = df["SPY"].dropna().astype(float) if "SPY" in df else pd.Series(dtype=float)
            spy_5d = float((spy_close.iloc[-1] - spy_close.iloc[-6]) / spy_close.iloc[-6] * 100.0) if len(spy_close) >= 6 else 0.0

            sector_list: List[SectorPerformance] = []
            for sec_sym, (name, stocks) in SECTOR_MAP.items():
                if sec_sym not in df.columns:
                    continue
                s_c = df[sec_sym].dropna().astype(float)
                if len(s_c) < 6:
                    continue
                ret_5d = float((s_c.iloc[-1] - s_c.iloc[-6]) / s_c.iloc[-6] * 100.0)
                rs = round(ret_5d - spy_5d, 2)
                sector_list.append(SectorPerformance(
                    ticker=sec_sym,
                    name=name,
                    return_5d_pct=round(ret_5d, 2),
                    relative_strength_vs_spy=rs,
                    rank=0,
                    top_stocks=stocks,
                ))

            # Nach relativer Stärke sortieren
            sector_list.sort(key=lambda s: s.relative_strength_vs_spy, reverse=True)
            for i, s in enumerate(sector_list):
                s.rank = i + 1

            leaders = sector_list[:3]
            laggards = sector_list[-3:]
            top_sec = leaders[0].name if leaders else "Technologie"

            hot_stocks = []
            for lead in leaders:
                hot_stocks.extend(lead.top_stocks[:2])
            hot_stocks = list(dict.fromkeys(hot_stocks))[:6]

            summary = (
                f"Sektor-Leader: {leaders[0].ticker} ({leaders[0].name}, RS {leaders[0].relative_strength_vs_spy:+.1f}%) | "
                f"Laggard: {laggards[-1].ticker} ({laggards[-1].name}, RS {laggards[-1].relative_strength_vs_spy:+.1f}%) | "
                f"Top-Aktien: {', '.join(hot_stocks[:4])}"
            )

            snap = SectorRotationSnapshot(
                timestamp=time.time(),
                leading_sectors=leaders,
                lagging_sectors=laggards,
                top_alpha_sector=top_sec,
                recommended_hot_stocks=hot_stocks,
                summary=summary,
            )

            self._save_cache(snap)
            return snap

        except Exception as e:
            logger.warning(f"Fehler bei Sektor-Rotations-Scan: {e}")
            return self._create_fallback_snapshot(f"Fehler: {str(e)[:40]}")

    def _create_fallback_snapshot(self, reason: str) -> SectorRotationSnapshot:
        return SectorRotationSnapshot(
            timestamp=time.time(),
            leading_sectors=[],
            lagging_sectors=[],
            top_alpha_sector="Technologie & KI",
            recommended_hot_stocks=["NVDA", "MSFT", "AAPL"],
            summary=f"Fallback: {reason}",
        )

    def _save_cache(self, snap: SectorRotationSnapshot) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(snap.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Konnte Sektor-Cache nicht schreiben: {e}")
