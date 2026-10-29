"""
intermarket_lead_lag.py
=======================
Institutionelle Intermarket-Frühwarn- & Kredit-Spillover-Engine.

Quantitativer Vorsprung:
1. High-Yield Kredit-Spread (HYG / LQD):
   - Der Anleihen- & Junk-Bond-Markt (HYG) reagiert 2 bis 5 Handelstage schneller auf
     Liquiditätsengpässe als der Aktienmarkt (SPY/QQQ).
   - Ein plötzliches Einbrechen des Verhältnisses HYG/LQD ist das zuverlässigste Frühwarnsignal
     für systemischen Stress & Aktien-Abverkäufe.
2. Dr. Copper / Gold Ratio (CPER / GLD):
   - Kupfer spiegelt die reale weltweite Industrie- & Bau-Nachfrage wider.
   - Steigt das Verhältnis Kupfer/Gold, expandiert die Weltwirtschaft (Risk-On / Tech-Rückenwind).
   - Fällt das Verhältnis stark ab, droht Konjunktur-Abkühlung (Risk-Off).
3. Kredit-Ampel & Red-Team Veto:
   - Bei akutem Kreditstress (HYG/LQD Einbruch > -1.2% in 5 Tagen) warnt der Sentinel den Schwarm,
     drosselt Kauf-Positionen und schützt das Kapital.
"""

from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("autonomous_agent.intermarket")

CACHE_DIR = Path("data/intermarket_cache")
CACHE_FILE = CACHE_DIR / "credit_intermarket_state.json"
CACHE_EXPIRY_SEC = 1800  # 30 Minuten


@dataclass
class IntermarketLeadLagSnapshot:
    timestamp: float
    credit_ratio_hyg_lqd: float
    credit_ratio_5d_change_pct: float
    copper_gold_ratio: float
    copper_gold_5d_change_pct: float
    credit_stress_active: bool
    macro_economic_expansion: bool
    intermarket_bias: str  # BULLISH_CREDIT_TAILWIND | NEUTRAL_STABLE | BEARISH_CREDIT_WARNING | SEVERE_CRUNCH
    risk_budget_multiplier: float  # 0.40x bis 1.25x
    lead_lag_signals: List[str]
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


class IntermarketLeadLagEngine:
    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def analyze_intermarket_dynamics(self, force_refresh: bool = False) -> IntermarketLeadLagSnapshot:
        """
        Analysiert Kredit-Spreads (HYG/LQD) und Rohstoff-Leitindikatoren (CPER/GLD).
        """
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if time.time() - data.get("timestamp", 0) < CACHE_EXPIRY_SEC:
                    return IntermarketLeadLagSnapshot(**data)
            except Exception as e:
                logger.debug(f"Intermarket Cache Read Error: {e}")

        # Live-Daten laden
        tickers = ["HYG", "LQD", "CPER", "GLD"]
        try:
            df = yf.download(tickers, period="1mo", interval="1d", progress=False)["Close"]
            if df is None or df.empty:
                return self._create_fallback_snapshot("Keine Daten von Yahoo Finance erhalten")

            # 1. Kredit-Spread HYG / LQD
            s_hyg = df["HYG"].dropna().astype(float) if "HYG" in df else pd.Series(dtype=float)
            s_lqd = df["LQD"].dropna().astype(float) if "LQD" in df else pd.Series(dtype=float)

            credit_ratio = 0.75
            credit_chg_5d = 0.0
            if len(s_hyg) >= 6 and len(s_lqd) >= 6:
                c_idx = s_hyg.index.intersection(s_lqd.index)
                r_series = s_hyg.loc[c_idx] / s_lqd.loc[c_idx]
                credit_ratio = round(float(r_series.iloc[-1]), 4)
                prev_5 = float(r_series.iloc[-6])
                credit_chg_5d = round(((credit_ratio - prev_5) / prev_5) * 100.0, 2)

            # 2. Dr. Copper / Gold Ratio (CPER / GLD)
            s_cper = df["CPER"].dropna().astype(float) if "CPER" in df else pd.Series(dtype=float)
            s_gld = df["GLD"].dropna().astype(float) if "GLD" in df else pd.Series(dtype=float)

            cg_ratio = 0.12
            cg_chg_5d = 0.0
            if len(s_cper) >= 6 and len(s_gld) >= 6:
                cg_idx = s_cper.index.intersection(s_gld.index)
                cg_series = s_cper.loc[cg_idx] / s_gld.loc[cg_idx]
                cg_ratio = round(float(cg_series.iloc[-1]), 4)
                cg_prev_5 = float(cg_series.iloc[-6])
                cg_chg_5d = round(((cg_ratio - cg_prev_5) / cg_prev_5) * 100.0, 2)

            # 3. Intermarket Auswertung & Stress-Erkennung
            signals: List[str] = []
            credit_stress = False
            expansion = cg_chg_5d >= 0.50

            if credit_chg_5d <= -1.25:
                credit_stress = True
                signals.append(f"🚨 KREDIT-WARNUNG: Junk-Bonds kollabieren relativ zu Investment Grade ({credit_chg_5d:+.2f}% in 5d)")
                bias = "BEARISH_CREDIT_WARNING"
                risk_mult = 0.50
            elif credit_chg_5d <= -2.50:
                credit_stress = True
                signals.append(f"🛑 AKUTER KREDIT-CRUNCH: Massive Spread-Ausweitung ({credit_chg_5d:+.2f}% in 5d) – Aktien-Abverkauf droht!")
                bias = "SEVERE_CRUNCH"
                risk_mult = 0.30
            elif credit_chg_5d >= 0.50:
                signals.append(f"🟢 Kreditmärkte stabil & entspannt: HYG/LQD steigt ({credit_chg_5d:+.2f}% in 5d)")
                bias = "BULLISH_CREDIT_TAILWIND"
                risk_mult = 1.15
            else:
                signals.append(f"⚖️ Kreditmärkte neutral (HYG/LQD {credit_chg_5d:+.2f}% in 5d)")
                bias = "NEUTRAL_STABLE"
                risk_mult = 1.0

            if cg_chg_5d >= 1.50:
                signals.append(f"🏭 Dr. Copper / Gold expandiert stark ({cg_chg_5d:+.1f}% in 5d) ➜ Konjunktur- & Industrie-Rückenwind")
            elif cg_chg_5d <= -1.50:
                signals.append(f"⚠️ Dr. Copper / Gold fällt ({cg_chg_5d:+.1f}% in 5d) ➜ Verlangsamung des globalen Wachstums")

            summary = (
                f"Kredit: {bias} (HYG/LQD: {credit_ratio:.3f}, {credit_chg_5d:+.2f}%) | "
                f"Kupfer/Gold: {cg_ratio:.3f} ({cg_chg_5d:+.1f}%) | Multiplikator: {risk_mult:.2f}x"
            )

            snap = IntermarketLeadLagSnapshot(
                timestamp=time.time(),
                credit_ratio_hyg_lqd=credit_ratio,
                credit_ratio_5d_change_pct=credit_chg_5d,
                copper_gold_ratio=cg_ratio,
                copper_gold_5d_change_pct=cg_chg_5d,
                credit_stress_active=credit_stress,
                macro_economic_expansion=expansion,
                intermarket_bias=bias,
                risk_budget_multiplier=risk_mult,
                lead_lag_signals=signals,
                summary=summary,
            )

            self._save_cache(snap)
            return snap

        except Exception as e:
            logger.warning(f"Fehler bei Intermarket Lead-Lag Analyse: {e}")
            return self._create_fallback_snapshot(f"Fehler: {str(e)[:50]}")

    def _create_fallback_snapshot(self, reason: str) -> IntermarketLeadLagSnapshot:
        return IntermarketLeadLagSnapshot(
            timestamp=time.time(),
            credit_ratio_hyg_lqd=0.74,
            credit_ratio_5d_change_pct=0.0,
            copper_gold_ratio=0.12,
            copper_gold_5d_change_pct=0.0,
            credit_stress_active=False,
            macro_economic_expansion=True,
            intermarket_bias="NEUTRAL_STABLE",
            risk_budget_multiplier=1.0,
            lead_lag_signals=[reason],
            summary=f"Fallback: {reason}",
        )

    def _save_cache(self, snap: IntermarketLeadLagSnapshot) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(snap.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Konnte Intermarket-Cache nicht schreiben: {e}")
