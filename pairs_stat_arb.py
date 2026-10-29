"""
pairs_stat_arb.py
=================
Statistische Arbitrage & Kointegrations-Pairs-Trading-Modul.

Mathematische Funktionsweise:
1. Analysiert historisch hochkorrelierte Paare (z. B. NVDA/AMD, SPY/QQQ, GLD/SLV, BTC/ETH, GOOGL/MSFT).
2. Berechnet das rollierende Preisverhältnis: Ratio_t = Price_A / Price_B.
3. Berechnet den 30-Tage Mean und Standardabweichung zur Bestimmung des Z-Scores:
       Z_t = (Ratio_t - Mean_30) / Std_30
4. Schätzt die Mean-Reversion Halbwertszeit (Ornstein-Uhlenbeck-Prozess):
       Halflife = -ln(2) / theta
5. Generiert statistische Arbitrage-Signale bei extremen Verwerfungen (|Z| >= 2.0).
   - Bei Z <= -2.0: Asset A ist relativ zu Asset B statistisch stark unterbewertet -> KAUF Asset A.
   - Bei Z >= +2.0: Asset B ist relativ zu Asset A statistisch stark unterbewertet -> KAUF Asset B.
6. Hält 15-Minuten-Disk-Cache zur Ratenbegrenzung.
"""

from __future__ import annotations
import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("autonomous_agent.pairs_stat_arb")

CACHE_DIR = Path("data/pairs_arb_cache")
CACHE_FILE = CACHE_DIR / "pairs_opportunities.json"
CACHE_EXPIRY_SEC = 900  # 15 Minuten

DEFAULT_PAIRS: List[Tuple[str, str, str]] = [
    ("NVDA", "AMD", "Semiconductor Leaders (KI-Chips)"),
    ("SPY", "QQQ", "US Broad Market vs. Tech-100"),
    ("GLD", "SLV", "Precious Metals (Gold / Silber Ratio)"),
    ("BTC-USD", "ETH-USD", "Crypto Bluechip Majors"),
    ("GOOGL", "MSFT", "Mega-Cap AI & Cloud Monopolies"),
    ("XOM", "CVX", "US Oil & Energy Majors"),
]


@dataclass
class PairArbitrageOpportunity:
    pair_name: str
    asset_a: str
    asset_b: str
    description: str
    price_a: float
    price_b: float
    current_ratio: float
    mean_ratio_30d: float
    std_ratio_30d: float
    z_score: float
    correlation_60d: float
    half_life_days: float
    divergence_pct: float
    signal_action: str  # BUY_A_UNDERVALUED | BUY_B_UNDERVALUED | NEUTRAL
    target_asset: Optional[str]  # Welches Asset gekauft werden soll
    expected_edge_score: float
    thesis: str

    def to_dict(self) -> dict:
        return asdict(self)


class StatisticalArbitrageScanner:
    def __init__(self, pairs: Optional[List[Tuple[str, str, str]]] = None, cache_dir: Path = CACHE_DIR):
        self.pairs = pairs or DEFAULT_PAIRS
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "pairs_opportunities.json"

    def scan_all_pairs(self, force_refresh: bool = False) -> List[PairArbitrageOpportunity]:
        """
        Scannt alle definierten Paare nach statistischen Arbitrage-Gelegenheiten.
        Nutzt 15-Minuten-Cache.
        """
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if time.time() - data.get("timestamp", 0) < CACHE_EXPIRY_SEC:
                    return [PairArbitrageOpportunity(**item) for item in data.get("opportunities", [])]
            except Exception as e:
                logger.debug(f"Cache-Fehler in Pairs-Scanner: {e}")

        # Live-Download der Kurse für alle Paare
        all_symbols = list(set([p[0] for p in self.pairs] + [p[1] for p in self.pairs]))
        opportunities: List[PairArbitrageOpportunity] = []

        try:
            df = yf.download(all_symbols, period="6mo", interval="1d", progress=False)["Close"]
            if df is None or df.empty:
                return []

            for sym_a, sym_b, desc in self.pairs:
                if sym_a not in df.columns or sym_b not in df.columns:
                    continue

                series_a = df[sym_a].dropna().astype(float)
                series_b = df[sym_b].dropna().astype(float)

                common_idx = series_a.index.intersection(series_b.index)
                if len(common_idx) < 35:
                    continue

                s_a = series_a.loc[common_idx]
                s_b = series_b.loc[common_idx]

                p_a = float(s_a.iloc[-1])
                p_b = float(s_b.iloc[-1])
                if p_b <= 0:
                    continue

                ratio = s_a / s_b
                roll_mean = float(ratio.rolling(30).mean().iloc[-1])
                roll_std = float(ratio.rolling(30).std().iloc[-1])

                if roll_std <= 1e-7:
                    continue

                curr_r = float(ratio.iloc[-1])
                z_score = round((curr_r - roll_mean) / roll_std, 2)
                div_pct = round(((curr_r - roll_mean) / roll_mean) * 100.0, 2)

                # 60-Tage Korrelation berechnen
                ret_a = s_a.pct_change(fill_method=None).dropna().tail(60)
                ret_b = s_b.pct_change(fill_method=None).dropna().tail(60)
                corr = round(float(ret_a.corr(ret_b)), 2) if len(ret_a) > 20 else 0.80

                # Ornstein-Uhlenbeck Halbwertszeit (Mean Reversion Speed)
                half_life = self._calculate_half_life(ratio.tail(60))

                # Signal-Logik
                signal = "NEUTRAL"
                target_sym = None
                edge_score = 0.0
                thesis = f"Ratio im normalen Schwankungsbereich (Z: {z_score:+.2f})."

                if corr >= 0.65:
                    if z_score <= -2.0:
                        signal = "BUY_A_UNDERVALUED"
                        target_sym = sym_a
                        edge_score = round(min(2.5, abs(z_score) * 0.9), 2)
                        thesis = (
                            f"StatArb Signal: {sym_a} ist statistisch extrem unterbewertet gegenüber {sym_b} "
                            f"(Z-Score: {z_score:.2f}, Abweichung: {div_pct:+.1f}%, Halbwertszeit: {half_life:.1f}d). "
                            f"Hohe Mean-Reversion-Wahrscheinlichkeit zu Gunsten von {sym_a}."
                        )
                    elif z_score >= 2.0:
                        signal = "BUY_B_UNDERVALUED"
                        target_sym = sym_b
                        edge_score = round(min(2.5, abs(z_score) * 0.9), 2)
                        thesis = (
                            f"StatArb Signal: {sym_b} ist statistisch extrem unterbewertet gegenüber {sym_a} "
                            f"(Z-Score: {z_score:.2f}, Abweichung: {div_pct:+.1f}%, Halbwertszeit: {half_life:.1f}d). "
                            f"Hohe Mean-Reversion-Wahrscheinlichkeit zu Gunsten von {sym_b}."
                        )

                opp = PairArbitrageOpportunity(
                    pair_name=f"{sym_a}_{sym_b}",
                    asset_a=sym_a,
                    asset_b=sym_b,
                    description=desc,
                    price_a=round(p_a, 2),
                    price_b=round(p_b, 2),
                    current_ratio=round(curr_r, 4),
                    mean_ratio_30d=round(roll_mean, 4),
                    std_ratio_30d=round(roll_std, 4),
                    z_score=z_score,
                    correlation_60d=corr,
                    half_life_days=round(half_life, 1),
                    divergence_pct=div_pct,
                    signal_action=signal,
                    target_asset=target_sym,
                    expected_edge_score=edge_score,
                    thesis=thesis,
                )
                opportunities.append(opp)

            # Im Cache speichern
            self._save_cache(opportunities)
            return opportunities

        except Exception as e:
            logger.error(f"Fehler bei Pairs StatArb Scan: {e}")
            return []

    def get_pair_opportunity_for_symbol(self, symbol: str) -> Optional[PairArbitrageOpportunity]:
        """Gibt eine aktive Arbitrage-Chance für das gegebene Symbol zurück (falls vorhanden)."""
        opps = self.scan_all_pairs()
        sym = symbol.upper().strip()
        for opp in opps:
            if opp.signal_action != "NEUTRAL" and opp.target_asset == sym:
                return opp
        return None

    def _calculate_half_life(self, ratio_series: pd.Series) -> float:
        """
        Schätzt die Halbwertszeit des Mean-Reversion-Prozesses via linearer Regression:
        delta_y = theta * (y_{t-1} - mean) + epsilon
        Halflife = -ln(2) / theta
        """
        try:
            y = ratio_series.values
            delta_y = np.diff(y)
            y_lag = y[:-1] - np.mean(y[:-1])
            if len(y_lag) < 10:
                return 5.0
            # Steigung theta berechnen
            theta = np.dot(y_lag, delta_y) / max(1e-9, np.dot(y_lag, y_lag))
            if theta < 0:
                hl = -math.log(2) / theta
                return max(1.0, min(30.0, float(hl)))
            return 8.0  # Fallback
        except Exception:
            return 5.0

    def _save_cache(self, opps: List[PairArbitrageOpportunity]) -> None:
        try:
            payload = {
                "timestamp": time.time(),
                "opportunities": [o.to_dict() for o in opps],
            }
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.debug(f"Konnte Pairs-Cache nicht schreiben: {e}")
