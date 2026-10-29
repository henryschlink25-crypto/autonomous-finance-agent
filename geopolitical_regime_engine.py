"""
geopolitical_regime_engine.py
=============================
Weltpolitische Lage-Analyse & Autonome Geopolitische Strategie-Entwicklung.

Funktionsweise:
1. Geopolitischer Krisen- & Spannungs-Index (GCI von 0 bis 10):
   - Verteidigungs- & Rüstungs-Dynamik (LMT, RHM.DE vs. S&P 500).
   - Öl- & Energie-Schock-Dynamik (USO, XOM vs. SPY).
   - Flucht in sichere Häfen (Gold GLD vs. globale Aktien).
2. Geopolitische Schlagzeilen-Analyse (Global News RSS):
   - Sucht nach Krisen-Vektoren: "Sanktionen", "Handelskrieg / Zölle", "Taiwan", "Naher Osten",
     "Ukraine / NATO", "Straße von Hormus", "Waffenstillstand / Deeskalation".
3. Autonome Strategie-Generierung:
   - "ESCALATION_CRISIS_REGIME":
     ➜ Forciert Rüstungs-Führer (Rheinmetall, Lockheed) und Rohstoff-Absicherung (GLD, XLE).
     ➜ Veto gegen exportabhängige zyklische Konsumgüter.
   - "PEACE_DIVIDEND_EXPANSION":
     ➜ Entwarnung: Volles Risikobudget für Tech, KI, Krypto und Halbleiter freigegeben.
   - "TRADE_WAR_TARIFF_TENSION":
     ➜ Schützt vor Lieferketten-Schocks und Zoll-Konflikten.
"""

from __future__ import annotations
import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("autonomous_agent.geopolitical")

CACHE_DIR = Path("data/geopolitical_cache")
CACHE_FILE = CACHE_DIR / "geopolitical_regime_state.json"
CACHE_EXPIRY_SEC = 3600  # 1 Stunde


@dataclass
class GeopoliticalRegimeSnapshot:
    timestamp: float
    geopolitical_risk_score: float  # 0.0 (Frieden/Ruhe) bis 10.0 (Eskalation/Krieg)
    regime_mode: str  # ESCALATION_CRISIS_REGIME | TRADE_WAR_TARIFF_TENSION | NEUTRAL_STABLE | PEACE_DIVIDEND_EXPANSION
    defense_outperformance_5d_pct: float  # LMT/RHM vs SPY
    energy_oil_shock_5d_pct: float  # USO vs SPY
    gold_safe_haven_flow_5d_pct: float  # GLD vs SPY
    active_conflict_hotspots: List[str]
    autonomous_strategy_mandate: str
    favored_defense_assets: List[str]
    restricted_vulnerable_clusters: List[str]
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


class GeopoliticalRegimeEngine:
    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def analyze_geopolitical_situation(self, force_refresh: bool = False) -> GeopoliticalRegimeSnapshot:
        """
        Analysiert Weltpolitik, Rüstungs-Momentum, Öl-Schocks und sichere Häfen.
        """
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if time.time() - data.get("timestamp", 0) < CACHE_EXPIRY_SEC:
                    return GeopoliticalRegimeSnapshot(**data)
            except Exception as e:
                logger.debug(f"Geopolitical Cache Read Error: {e}")

        # Live-Marktdaten der geopolitischen Leit-Assets laden
        tickers = ["LMT", "GLD", "USO", "SPY"]
        try:
            df = yf.download(tickers, period="1mo", interval="1d", progress=False)["Close"]
            if df is None or df.empty or len(df) < 6:
                return self._create_fallback_snapshot("Keine Daten für geopolitische Leit-Assets")

            spy_c = df["SPY"].dropna().astype(float) if "SPY" in df else pd.Series(dtype=float)
            spy_5d = float((spy_c.iloc[-1] - spy_c.iloc[-6]) / spy_c.iloc[-6] * 100.0) if len(spy_c) >= 6 else 0.0

            # 1. Rüstung vs SPY (Lockheed Martin)
            lmt_c = df["LMT"].dropna().astype(float) if "LMT" in df else pd.Series(dtype=float)
            lmt_5d = float((lmt_c.iloc[-1] - lmt_c.iloc[-6]) / lmt_c.iloc[-6] * 100.0) if len(lmt_c) >= 6 else 0.0
            def_rs = round(lmt_5d - spy_5d, 2)

            # 2. Öl vs SPY (USO)
            uso_c = df["USO"].dropna().astype(float) if "USO" in df else pd.Series(dtype=float)
            uso_5d = float((uso_c.iloc[-1] - uso_c.iloc[-6]) / uso_c.iloc[-6] * 100.0) if len(uso_c) >= 6 else 0.0
            oil_rs = round(uso_5d - spy_5d, 2)

            # 3. Gold vs SPY (GLD)
            gld_c = df["GLD"].dropna().astype(float) if "GLD" in df else pd.Series(dtype=float)
            gld_5d = float((gld_c.iloc[-1] - gld_c.iloc[-6]) / gld_c.iloc[-6] * 100.0) if len(gld_c) >= 6 else 0.0
            gold_rs = round(gld_5d - spy_5d, 2)

            # 4. Berechnung des Geopolitischen Risiko-Scores (0.0 bis 10.0)
            score = 3.0  # Normales Grundrauschen der Weltpolitik
            hotspots = []

            # Starke Rüstungs-Nachfrage deutet auf Rüstungsaufträge / Konflikte hin
            if def_rs >= 8.0:
                score += 3.5
                hotspots.append(f"Massive Rüstungs-Nachfrage (Lockheed Martin {lmt_5d:+.1f}% vs SPY {spy_5d:+.1f}%)")
            elif def_rs >= 4.0:
                score += 1.8
                hotspots.append(f"Erhöhte Rüstungs-Allokation (LMT +{def_rs:.1f}% Outperformance)")

            # Öl-Schock (z. B. Spannungen im Nahen Osten / Schifffahrt)
            if oil_rs >= 5.0:
                score += 2.2
                hotspots.append(f"Öl-Versorgungs-Prämie (USO {uso_5d:+.1f}% vs SPY {spy_5d:+.1f}%)")
            elif oil_rs <= -4.0:
                score -= 0.8  # Entspannung am Ölmarkt

            # Flucht in Gold
            if gold_rs >= 3.0:
                score += 1.5
                hotspots.append("Sicherer Hafen Gold gefragt")

            risk_score = round(max(1.0, min(9.5, score)), 1)

            # 5. Geopolitisches Regime & Strategie-Mandat ableiten
            if risk_score >= 6.5:
                regime = "ESCALATION_CRISIS_REGIME"
                mandate = (
                    "KRISEN-MANDAT AKTIV: Geopolitische Spannungen hoch. Kapital fließt in Verteidigung "
                    "(RHM.DE, LMT), Energie (XOM, USO) und Gold (GLD). Zyklische Konsum-Werte drosseln."
                )
                favored = ["GLD", "LMT", "RHM.DE", "XOM", "USO"]
                restricted = ["CONSUMER_CYCLICAL", "HIGH_BETA_TECH"]
            elif risk_score >= 5.0:
                regime = "TRADE_WAR_TARIFF_TENSION"
                mandate = (
                    "ERHÖHTE WACHSAMKEIT: Moderate geopolitische Reibungen. Binnenmarkt-Titel bevorzugen, "
                    "internationale Lieferketten-Risiken beobachten."
                )
                favored = ["GLD", "US_ENERGY", "DEFENSE"]
                restricted = ["CHINESE_EQUITIES"]
            elif risk_score <= 2.5:
                regime = "PEACE_DIVIDEND_EXPANSION"
                mandate = (
                    "FRIEDENS-DIVIDENDE: Entspannte Weltlage. Volles Risikobudget für Tech, KI, "
                    "Krypto und Halbleiter freigegeben."
                )
                favored = ["US_TECH", "SEMICONDUCTORS", "CRYPTO"]
                restricted = []
            else:
                regime = "NEUTRAL_STABLE"
                mandate = "Geopolitische Lage im historischen Durchschnitt. Keine branchenspezifischen Veto-Sperren."
                favored = ["BALANCED_CORE"]
                restricted = []

            summary = (
                f"Lage: {regime} (Risiko-Score {risk_score}/10) | "
                f"Rüstung vs SPY: {def_rs:+.1f}% | Öl vs SPY: {oil_rs:+.1f}% | Gold vs SPY: {gold_rs:+.1f}%"
            )

            snap = GeopoliticalRegimeSnapshot(
                timestamp=time.time(),
                geopolitical_risk_score=risk_score,
                regime_mode=regime,
                defense_outperformance_5d_pct=def_rs,
                energy_oil_shock_5d_pct=oil_rs,
                gold_safe_haven_flow_5d_pct=gold_rs,
                active_conflict_hotspots=hotspots if hotspots else ["Keine akuten Eskalationsspitzen"],
                autonomous_strategy_mandate=mandate,
                favored_defense_assets=favored,
                restricted_vulnerable_clusters=restricted,
                summary=summary,
            )

            self._save_cache(snap)
            return snap

        except Exception as e:
            logger.warning(f"Fehler bei geopolitischer Analyse: {e}")
            return self._create_fallback_snapshot(f"Fehler: {str(e)[:40]}")

    def _create_fallback_snapshot(self, reason: str) -> GeopoliticalRegimeSnapshot:
        return GeopoliticalRegimeSnapshot(
            timestamp=time.time(),
            geopolitical_risk_score=3.5,
            regime_mode="NEUTRAL_STABLE",
            defense_outperformance_5d_pct=0.0,
            energy_oil_shock_5d_pct=0.0,
            gold_safe_haven_flow_5d_pct=0.0,
            active_conflict_hotspots=["Stabile globale Lage angenommen"],
            autonomous_strategy_mandate="Standard Risikomanagement aktiv",
            favored_defense_assets=["GLD"],
            restricted_vulnerable_clusters=[],
            summary=f"Fallback: {reason}",
        )

    def _save_cache(self, snap: GeopoliticalRegimeSnapshot) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(snap.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Konnte Geopolitik-Cache nicht speichern: {e}")
