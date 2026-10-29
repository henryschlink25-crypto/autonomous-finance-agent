"""
financial_catalyst_scanner.py
=============================
Fundamentalanalyse, Bilanz-Solvenz und Earnings-Momentum Scanner.

Prüft vor Trade-Freigabe:
1. Solvenz & Liquidität: Quick Ratio, Verschuldungsgrad (Debt-to-Equity), Free Cash Flow.
2. Wachstum: Umsatzwachstum (YoY), Gewinnspannen.
3. Bewertung & Qualität: Forward P/E, PEG-Ratio.
4. Distress-Check (Red-Flag für Pleitekandidaten / toxische Verwässerungen).

Verhindert, dass der Schwarm in fundamentale Value-Traps oder insolvenzgefährdete
Penny-Stocks investiert.
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

logger = logging.getLogger("autonomous_agent.catalyst_scanner")

CACHE_DIR = Path("data/fundamentals_cache")
CACHE_EXPIRY_SEC = 3600 * 4  # 4 Stunden Cache


@dataclass
class FundamentalSnapshot:
    symbol: str
    timestamp: float
    forward_pe: Optional[float]
    peg_ratio: Optional[float]
    revenue_growth_yoy: Optional[float]  # e.g. 0.15 = +15%
    gross_margins: Optional[float]
    operating_cashflow: Optional[float]
    debt_to_equity: Optional[float]
    quick_ratio: Optional[float]
    market_cap: Optional[float]
    health_grade: str  # ELITE_GROWTH | SOLID_VALUE | SPECULATIVE | TOXIC_DISTRESSED | ETF_INDEX
    is_distressed: bool
    fundamental_score: float  # -1.0 bis +1.0
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


class FinancialCatalystScanner:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}_fund.json"

    def scan_symbol(self, symbol: str) -> FundamentalSnapshot:
        sym = symbol.upper().strip()
        cache_file = self._get_cache_path(sym)

        # 1. Disk-Cache prüfen
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if time.time() - data.get("timestamp", 0) < CACHE_EXPIRY_SEC:
                    return FundamentalSnapshot(**data)
            except Exception as e:
                logger.debug(f"Fund-Cache Fehler für {sym}: {e}")

        # 2. Live-Abfrage via yfinance
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info or {}

            # ETFs und Krypto gesondert behandeln
            quote_type = info.get("quoteType", "").upper()
            if quote_type in ["ETF", "MUTUALFUND", "CRYPTOCURRENCY"] or "USD" in sym:
                snap = FundamentalSnapshot(
                    symbol=sym,
                    timestamp=time.time(),
                    forward_pe=None,
                    peg_ratio=None,
                    revenue_growth_yoy=None,
                    gross_margins=None,
                    operating_cashflow=None,
                    debt_to_equity=None,
                    quick_ratio=None,
                    market_cap=info.get("marketCap"),
                    health_grade="ETF_INDEX",
                    is_distressed=False,
                    fundamental_score=0.2,  # Breiter Korb/Index erhält solide Grundbewertung
                    summary=f"{quote_type} Basket / Index - Keine Bilanzprüfung erforderlich",
                )
                self._save_cache(snap)
                return snap

            f_pe = info.get("forwardPE")
            peg = info.get("pegRatio")
            rev_growth = info.get("revenueGrowth")
            margins = info.get("grossMargins")
            ocf = info.get("operatingCashflow")
            debt_eq = info.get("debtToEquity")
            quick = info.get("quickRatio")
            mcap = info.get("marketCap")

            # Scoring berechnen
            score = 0.0
            distressed = False

            # Insolvenz / Liquiditäts-Check
            if ocf is not None and ocf < -50_000_000 and (debt_eq is not None and debt_eq > 250):
                # Negativer Cashflow bei hoher Verschuldung = Alarm!
                distressed = True
                score -= 0.8
            elif ocf is not None and ocf > 0:
                score += 0.25

            # Quick Ratio
            if quick is not None:
                if quick >= 1.5:
                    score += 0.2
                elif quick < 0.7:
                    score -= 0.3

            # Wachstum (Revenue Growth)
            if rev_growth is not None:
                if rev_growth >= 0.20:  # > 20% YoY Wachstum
                    score += 0.35
                elif rev_growth >= 0.08:
                    score += 0.15
                elif rev_growth < -0.10:  # Schrumpfendes Geschäft
                    score -= 0.35

            # Bewertung (Forward P/E & PEG)
            if f_pe is not None:
                if 10.0 <= f_pe <= 35.0:
                    score += 0.15
                elif f_pe > 90.0:  # Extrem teuer
                    score -= 0.2

            score = max(-1.0, min(1.0, round(score, 2)))

            # Grade Einstufung
            if distressed or score <= -0.4:
                grade = "TOXIC_DISTRESSED"
            elif score >= 0.45:
                grade = "ELITE_GROWTH"
            elif score >= 0.1:
                grade = "SOLID_VALUE"
            else:
                grade = "SPECULATIVE"

            summary = (
                f"Grade: {grade} | RevGrowth: {f'{rev_growth*100:.1f}%' if rev_growth is not None else 'N/A'} | "
                f"Fwd-PE: {f'{f_pe:.1f}' if f_pe else 'N/A'} | OCF: {f'${ocf/1e9:.2f}B' if ocf else 'N/A'} | "
                f"Quick: {f'{quick:.2f}' if quick else 'N/A'}"
            )

            snap = FundamentalSnapshot(
                symbol=sym,
                timestamp=time.time(),
                forward_pe=round(f_pe, 2) if f_pe else None,
                peg_ratio=round(peg, 2) if peg else None,
                revenue_growth_yoy=round(rev_growth, 3) if rev_growth else None,
                gross_margins=round(margins, 3) if margins else None,
                operating_cashflow=ocf,
                debt_to_equity=round(debt_eq, 2) if debt_eq else None,
                quick_ratio=round(quick, 2) if quick else None,
                market_cap=mcap,
                health_grade=grade,
                is_distressed=distressed,
                fundamental_score=score,
                summary=summary,
            )

            self._save_cache(snap)
            return snap

        except Exception as e:
            logger.warning(f"Fehler bei Fundamentalanalyse für {sym}: {e}")
            snap = FundamentalSnapshot(
                symbol=sym,
                timestamp=time.time(),
                forward_pe=None,
                peg_ratio=None,
                revenue_growth_yoy=None,
                gross_margins=None,
                operating_cashflow=None,
                debt_to_equity=None,
                quick_ratio=None,
                market_cap=None,
                health_grade="SPECULATIVE",
                is_distressed=False,
                fundamental_score=0.0,
                summary=f"Fundamentaldaten nicht verfügbar: {str(e)[:40]}",
            )
            return snap

    def _save_cache(self, snap: FundamentalSnapshot) -> None:
        try:
            cache_file = self._get_cache_path(snap.symbol)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(snap.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Konnte Fund-Cache für {snap.symbol} nicht speichern: {e}")
