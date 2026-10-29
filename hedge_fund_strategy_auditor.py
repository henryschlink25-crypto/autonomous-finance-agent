"""
hedge_fund_strategy_auditor.py
==============================
Hedge-Fonds-Strategie-Auditor & Situational Awareness Sentinel (Sub-Agent #50).

Inspiriert von den Lehren & Fehlern moderner Elite-Hedgefonds:
1. Leopold Aschenbrenner ("Situational Awareness: The Decade Ahead"):
   - Erkennung des säkularen KI-Superzyklus & des physischen Flaschenhalses:
     Pfeiler A: Compute & Halbleiter (NVDA, TSM, AMD, AVGO)
     Pfeiler B: Energie, Stromnetze & Kernkraft (VRT, CEG, CCJ, GEV, NEE)
     Pfeiler C: Datenzentren, Neocloud & optische Netzwerke (ANET, MRVL)
   - Säkularer Rückenwind: Erkennt die Multi-Milliarden-Dollar Capex-Wellen.

2. Der "Citadel / Ken Griffin" Anti-Blowup-Schutz (Fehler-Vermeidung):
   - Der klassische Hedgefonds-Fehler von Aschenbrenners "Situational Awareness LP" (2026):
     "Recht haben mit der 5-Jahres-These, aber liquidiert werden durch kurzfristige Hebel & Volatilität."
   - Blockiert hohe Hebel bei stark schwankenden KI- & Energie-Themen.
   - "Right Thesis, Wrong Timing" Schutz: Verhindert das Kaufen in fallende Messer,
     nur weil die fundamentale Story genial klingt.
   - Erzwingt "Wait for Citadel Dip": Einstieg erst, wenn die institutionelle Bereinigung
     abgeschlossen ist und die technische Trendwende steht.
"""

from __future__ import annotations
import datetime
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from models import MacroRegime, MarketSnapshot, Position

logger = logging.getLogger("autonomous_agent.hedge_fund_auditor")

# Aschenbrenner Situational Awareness Universum: Die 3 physischen Engpässe
AI_INFRASTRUCTURE_PILLARS = {
    # Pfeiler A: Compute & Custom Silicon
    "NVDA": {"pillar": "COMPUTE_SILICON", "name": "NVIDIA", "supercycle_weight": 1.45},
    "TSM": {"pillar": "COMPUTE_SILICON", "name": "TSMC (CoWoS Packaging)", "supercycle_weight": 1.40},
    "AVGO": {"pillar": "COMPUTE_SILICON", "name": "Broadcom (Custom AI Silicon)", "supercycle_weight": 1.35},
    "AMD": {"pillar": "COMPUTE_SILICON", "name": "Advanced Micro Devices", "supercycle_weight": 1.25},
    "ARM": {"pillar": "COMPUTE_SILICON", "name": "Arm Holdings", "supercycle_weight": 1.20},
    "MU": {"pillar": "COMPUTE_SILICON", "name": "Micron (HBM Memory)", "supercycle_weight": 1.30},

    # Pfeiler B: Energie, Stromnetz-Infrastruktur & Nuklearkraft
    "VRT": {"pillar": "ENERGY_COOLING_GRID", "name": "Vertiv (Liquid Cooling & Datacenter Power)", "supercycle_weight": 1.45},
    "CCJ": {"pillar": "NUCLEAR_ENERGY", "name": "Cameco (Uranium Fuel)", "supercycle_weight": 1.35},
    "CEG": {"pillar": "NUCLEAR_ENERGY", "name": "Constellation Energy", "supercycle_weight": 1.40},
    "GEV": {"pillar": "GRID_EQUIPMENT", "name": "GE Vernova (Power Turbines & Transformers)", "supercycle_weight": 1.35},
    "XOM": {"pillar": "BASELOAD_POWER", "name": "ExxonMobil (Natural Gas Baseload)", "supercycle_weight": 1.15},

    # Pfeiler C: Netzwerke, Optik & Datenzentren
    "ANET": {"pillar": "OPTICAL_INTERCONNECT", "name": "Arista Networks (Ethernet & Optical)", "supercycle_weight": 1.35},
    "MRVL": {"pillar": "OPTICAL_INTERCONNECT", "name": "Marvell Technology (Optical DSPs)", "supercycle_weight": 1.25},
    "PLTR": {"pillar": "SOVEREIGN_AI", "name": "Palantir (National Security AI Defense)", "supercycle_weight": 1.35},
    "MSFT": {"pillar": "HYPERSCALER_CAPEX", "name": "Microsoft Azure", "supercycle_weight": 1.20},
    "GOOGL": {"pillar": "HYPERSCALER_CAPEX", "name": "Alphabet TPU Cluster", "supercycle_weight": 1.20},
    "AMZN": {"pillar": "HYPERSCALER_CAPEX", "name": "AWS Trainium Cluster", "supercycle_weight": 1.20},
}


@dataclass
class HedgeFundAuditResult:
    symbol: str
    is_situational_awareness_asset: bool
    infrastructure_pillar: str
    supercycle_score: float         # 0.0 bis 1.0 (Makro-Rückenwind)
    timing_risk_score: float        # 0.0 bis 1.0 (Gefahr von kurzfristigem Liquidations-Druck)
    leverage_penalty: float         # Drosselung bei zu hohem Hebel
    citadel_dip_status: str         # READY_ACCUMULATE | WAIT_FOR_CLEANSE | OVERHEATED_LEVERAGE_VETO
    approved: bool
    recommended_leverage_cap: float # Max erlaubter Hebel (z.B. 1.0x statt 3.0x bei hoher Volatilität)
    rationale: str

    @property
    def citadel_leverage_veto(self) -> bool:
        return self.citadel_dip_status == "OVERHEATED_LEVERAGE_VETO"

    def to_dict(self) -> dict:
        return asdict(self)


class HedgeFundStrategyAuditor:
    """
    Sub-Agent #50: Situational Awareness & Hedge Fund Risk Auditor.
    Schützt das Portfolio vor den typischen Fehlern von Makro-Fonds (Aschenbrenner-Falle)
    und integriert Ken Griffins / Citadels marktneutrale Risikodisziplin.
    """

    def __init__(self):
        pass

    def audit_trade_setup(
        self,
        snapshot: MarketSnapshot,
        macro: Optional[MacroRegime] = None,
        planned_leverage: float = 1.0,
    ) -> HedgeFundAuditResult:
        sym = snapshot.symbol.upper().replace(".DE", "").strip()
        asset_meta = AI_INFRASTRUCTURE_PILLARS.get(sym)

        is_sa = asset_meta is not None
        pillar = asset_meta["pillar"] if is_sa else "GENERAL_MARKET"
        base_supercycle = asset_meta["supercycle_weight"] if is_sa else 1.0
        vix = macro.vix_level if macro else 18.0

        current_p = snapshot.current_price
        sma20 = getattr(snapshot, "sma_20", current_p)
        sma50 = getattr(snapshot, "sma_50", current_p)
        macd_h = getattr(snapshot, "macd_hist", 0.0)
        rsi = getattr(snapshot, "rsi_14", 50.0)
        vol_ann = getattr(snapshot, "volatility_annualized_pct", 30.0)

        # 1. FEHLER #1 VERMEIDEN: "The Leopold Aschenbrenner Leverage Trap"
        # Bei hoher Volatilität (>45% p.a.) oder schwankendem Tech-Markt (VIX >= 25)
        # oder Hebel > 2.0x greift das strikte Citadel Anti-Blowup Veto.
        is_excessive_leverage = (
            planned_leverage > 2.0 or
            ((planned_leverage > 1.5 or snapshot.intrinsic_leverage > 1.5) and (vol_ann >= 45.0 or vix >= 25.0))
        )
        if is_excessive_leverage:
            return HedgeFundAuditResult(
                symbol=sym,
                is_situational_awareness_asset=is_sa,
                infrastructure_pillar=pillar,
                supercycle_score=0.85,
                timing_risk_score=0.92,
                leverage_penalty=0.50,
                citadel_dip_status="OVERHEATED_LEVERAGE_VETO",
                approved=False,
                recommended_leverage_cap=1.0,
                rationale=(
                    f"⚠️ CITADEL ANTI-BLOWUP VETO (Aschenbrenner-Falle): Geplanter Hebel ({planned_leverage:.1f}x) "
                    f"bei hoher Volatilität ({vol_ann:.1f}% p.a., VIX {vix:.1f}) blockiert! Selbst bei intakter "
                    f"Superzyklus-These führen Zwischenkorrekturen zur Zwangsliquidation. Hebel strikt auf 1.0x begrenzt."
                ),
            )

        safe_lev_cap = 2.0 if (vix < 20.0 and vol_ann < 45.0) else (1.0 if (vix >= 25.0 or vol_ann >= 50.0) else 1.5)

        # 2. FEHLER #2 VERMEIDEN: "Right Thesis, Wrong Timing" (Falling Knife Syndrom)
        # Aktie notiert unter SMA50 und MACD fällt steil -> Großinvestoren liquidieren gerade Positionen.
        # Hier darf man nicht reingreifen, sondern muss warten ("Wait for Citadel Dip").
        is_in_liquidation = (current_p < sma50 and macd_h < -0.3 and rsi < 42.0)
        if is_sa and is_in_liquidation:
            return HedgeFundAuditResult(
                symbol=sym,
                is_situational_awareness_asset=True,
                infrastructure_pillar=pillar,
                supercycle_score=0.88,
                timing_risk_score=0.85,
                leverage_penalty=0.70,
                citadel_dip_status="WAIT_FOR_CLEANSE",
                approved=False,
                recommended_leverage_cap=1.0,
                rationale=(
                    f"⏳ WAIT FOR CITADEL DIP (Aschenbrenner-Falle): {sym} gehört zum KI-Energie-/Compute-Engpass ({pillar}), "
                    f"befindet sich aber in einer institutionellen Abverkaufswelle (unter SMA50, MACD {macd_h:+.2f}). "
                    f"Nicht ins fallende Messer greifen – warte auf technische Bodenbildung & Konvergenz!"
                ),
            )

        # 3. DAS POSITIVE SETUP: "Situational Awareness Super-Cycle Accumulation"
        # Säkularer Rückenwind + gesunde Konsolidierung oder Rebound über SMA20
        is_healthy_trend = (current_p >= sma20 and macd_h >= -0.05 and 42.0 <= rsi <= 74.0)
        if is_sa and is_healthy_trend:
            return HedgeFundAuditResult(
                symbol=sym,
                is_situational_awareness_asset=True,
                infrastructure_pillar=pillar,
                supercycle_score=round(base_supercycle * 0.70, 2),
                timing_risk_score=0.25,
                leverage_penalty=1.0,
                citadel_dip_status="READY_ACCUMULATE",
                approved=True,
                recommended_leverage_cap=safe_lev_cap,
                rationale=(
                    f"🏛️ SITUATIONAL AWARENESS ACCUMULATION: {sym} ({pillar}) profitiert direkt vom säkularen "
                    f"KI-Infrastruktur- & Energie-Engpass. Technisch stabil über SMA20 bei kontrolliertem Risiko (RSI {rsi:.1f})."
                ),
            )

        # 4. ALLGEMEINER MARKT-FALLBACK
        return HedgeFundAuditResult(
            symbol=sym,
            is_situational_awareness_asset=is_sa,
            infrastructure_pillar=pillar,
            supercycle_score=0.50,
            timing_risk_score=0.40,
            leverage_penalty=1.0,
            citadel_dip_status="READY_ACCUMULATE",
            approved=True,
            recommended_leverage_cap=safe_lev_cap,
            rationale=f"Hedge-Fonds Audit neutral: Standard-Marktexposition ohne asymmetrische Engpass-Risiken.",
        )
