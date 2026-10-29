"""
cio_conviction_synthesizer.py
=============================
🏛️ Chief Investment Officer (CIO) Conviction Synthesizer.

Inspiriert von den führenden Multi-Agenten-Architekturen auf GitHub
(virattt/ai-hedge-fund Portfolio Manager & TauricResearch/TradingAgents):

Anstatt 30 Unteragenten nur linear und unstrukturiert abzustimmen,
organisiert der CIO-Synthesizer den gesamten Schwarm in 4 professionelle
thematische Komitees (Committees):

1. FUNDAMENTAL & VALUATION COMMITTEE:
   - Fundamental Health & Distress
   - SEC Form 4 Whale & Cluster-Insider
   - Wall Street Consensus & Analyst Price Targets
   - Earnings Drift & Guidance

2. MACRO, REGIME & GEOPOLITICS COMMITTEE:
   - Intermarket Credit Spreads (HYG/LQD)
   - Macro Event Horizon (FOMC, CPI, NFP Whipsaw-Radar)
   - Geopolitical Risk Matrix (GCI)
   - Sector Rotation & Relative Strength vs SPY
   - VIX Dynamic Volatility Guard

3. QUANTITATIVE ALPHA & FLOW COMMITTEE:
   - Probabilistic Forecast Quantile Trichter (Kronos/TimesFM)
   - Institutional VWAP & Volume-Profile Liquidity Sweeper
   - Options Flow & Max-Pain Smart Money Sentinel
   - Hurst Exponent & Fractal Trend Persistence
   - StatArb Pairs Mean-Reversion Sentinel
   - Hugging Face FinBERT NLP Sentiment

4. RISK, EXECUTION & MICROSTRUCTURE COMMITTEE:
   - Corwin-Schultz Bid-Ask & Amihud Illiquidity
   - Penny Stock Cap & Volatility Sentinel
   - Smart Leverage Allocator
   - Hyperparameter AutoTuner Stop-Loss Calibration

Der CIO:
- Gewichtet die Komitees regime-abhängig (im Krisen-/Kreditstress wiegen Macro & Risk bis zu 2.5x).
- Löst Inter-Komitee-Konflikte auf (z. B. wenn Quantitative bullisch ist, aber Risk/Macro ein Veto einlegen).
- Errechnet den CIO Conviction Index (0.0 bis 1.0) und weist eine präzise Überzeugungsstufe zu
  (TIER_1_ALPHA_HIGH_CONVICTION, TIER_2_SOLID_OPPORTUNITY, TIER_3_TACTICAL_SMALL, CONVICTION_UNCONVINCED_HOLD).
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from models import MacroRegime, MarketSnapshot, TradingSignal

logger = logging.getLogger("autonomous_agent.cio_synthesizer")


class CommitteeResult(BaseModel):
    committee_name: str
    score: float = 0.0          # -1.0 bis +1.0
    confidence: float = 0.50     # 0.0 bis 1.0
    weight: float = 0.25        # Regime-Gewichtung
    bull_count: int = 0
    bear_count: int = 0
    veto_count: int = 0
    dominant_rationale: str = ""
    member_agents: List[str] = Field(default_factory=list)


class CIOSynthesisResult(BaseModel):
    symbol: str
    overall_conviction_score: float = 0.50  # 0.0 bis 1.0
    synthesized_action: str = "HOLD"        # "BUY", "HOLD", "SELL", "VETO_REJECT"
    conviction_tier: str = "TIER_2_SOLID_OPPORTUNITY"
    size_multiplier: float = 1.00           # 0.0 bis 1.25
    committee_conflict_detected: bool = False
    conflict_resolution_note: str = ""
    committees: Dict[str, CommitteeResult] = Field(default_factory=dict)
    executive_summary: str = ""


# Zuordnung der Agenten-Spezialisierungen zu Komitees
COMMITTEE_MAPPING = {
    # 1. Fundamental
    "FUNDAMENTAL_HEALTH": "FUNDAMENTAL",
    "FUNDAMENTALS_AND_NEWS": "FUNDAMENTAL",
    "SEC_INSIDER_TRACKER": "FUNDAMENTAL",
    "INSIDER_ACCUMULATION": "FUNDAMENTAL",
    "WALL_STREET_ANALYST": "FUNDAMENTAL",
    "WALL_STREET_CONSENSUS": "FUNDAMENTAL",
    "EARNINGS_DRIFT": "FUNDAMENTAL",
    "PEAD_DRIFT": "FUNDAMENTAL",
    "VALUE_INVESTING": "FUNDAMENTAL",

    # 2. Macro & Regime
    "MACRO_INTERMARKET": "MACRO_REGIME",
    "INTERMARKET_CREDIT": "MACRO_REGIME",
    "MACRO_CREDIT_INTERMARKET": "MACRO_REGIME",
    "MACRO_EVENT_HORIZON": "MACRO_REGIME",
    "GEOPOLITICAL_RISK": "MACRO_REGIME",
    "GEOPOLITICAL_STRATEGY": "MACRO_REGIME",
    "SECTOR_ROTATION": "MACRO_REGIME",
    "VIX_DYNAMIC": "MACRO_REGIME",

    # 3. Quantitative Alpha & Flow
    "PROBABILISTIC_FORECAST": "QUANTITATIVE_ALPHA",
    "VWAP_VOLUME_PROFILE": "QUANTITATIVE_ALPHA",
    "INTRADAY_MICROSTRUCTURE": "QUANTITATIVE_ALPHA",
    "OPTIONS_FLOW_SENTINEL": "QUANTITATIVE_ALPHA",
    "OPTIONS_WHALE_FLOW": "QUANTITATIVE_ALPHA",
    "OPTIONS_SMART_MONEY": "QUANTITATIVE_ALPHA",
    "GAMMA_SQUEEZE": "QUANTITATIVE_ALPHA",
    "HURST_FRACTAL": "QUANTITATIVE_ALPHA",
    "FRACTAL_REGIME": "QUANTITATIVE_ALPHA",
    "STAT_ARB_PAIRS": "QUANTITATIVE_ALPHA",
    "STATISTICAL_ARBITRAGE": "QUANTITATIVE_ALPHA",
    "HUGGINGFACE_FINBERT": "QUANTITATIVE_ALPHA",
    "MOMENTUM_BREAKOUT": "QUANTITATIVE_ALPHA",
    "MOMENTUM_RVOL_RS": "QUANTITATIVE_ALPHA",
    "SUPERTREND_MOMENTUM": "QUANTITATIVE_ALPHA",
    "SUPERTREND_REGIME_RIDER": "QUANTITATIVE_ALPHA",
    "TTM_SQUEEZE_EXPLOSION": "QUANTITATIVE_ALPHA",
    "QUALITY_MEAN_REVERSION": "QUANTITATIVE_ALPHA",

    # 4. Risk & Microstructure
    "MICROSTRUCTURE_CORWIN": "RISK_EXECUTION",
    "MARKET_MICROSTRUCTURE": "RISK_EXECUTION",
    "PENNY_STOCK_RISK": "RISK_EXECUTION",
    "PENNY_STOCK_MOMENTUM": "RISK_EXECUTION",
    "SMART_LEVERAGE": "RISK_EXECUTION",
    "STOP_LOSS_CALIBRATOR": "RISK_EXECUTION",
    "RISK_VETO_GUARD": "RISK_EXECUTION",
    "SHORT_SELL_SPECIALIST": "RISK_EXECUTION",
    "PORTFOLIO_HEAT": "RISK_EXECUTION",
}


class CIOConvictionSynthesizer:
    """
    Synthetisiert die Voten des ADAS-Schwarms über hierarchische Komitees
    und steuert die finale Überzeugung und Positionsgröße.
    """

    def synthesize_conviction(
        self,
        symbol: str,
        votes: List[SubAgentVote],
        active_subagents: Dict[str, Any],
        baseline_signal: TradingSignal,
        macro: MacroRegime,
        snapshot: MarketSnapshot,
    ) -> CIOSynthesisResult:
        sym = symbol.strip().upper()

        # 1. Komitee-Sammlung initialisieren
        committee_buckets: Dict[str, List[SubAgentVote]] = {
            "FUNDAMENTAL": [],
            "MACRO_REGIME": [],
            "QUANTITATIVE_ALPHA": [],
            "RISK_EXECUTION": [],
        }

        for v in votes:
            ag = active_subagents.get(v.agent_id)
            focus = getattr(ag, "specialty_focus", "") if ag else ""
            c_name = COMMITTEE_MAPPING.get(focus, "QUANTITATIVE_ALPHA")
            committee_buckets[c_name].append(v)

        # 2. Regime-Gewichte dynamisch ermitteln
        weights = self._compute_regime_weights(macro)

        # 3. Komitee-Ergebnisse aggregieren
        committee_results: Dict[str, CommitteeResult] = {}
        for c_name, c_votes in committee_buckets.items():
            committee_results[c_name] = self._aggregate_committee(c_name, c_votes, active_subagents, weights[c_name])

        # 4. Übergeordnete CIO-Synthese & Konflikt-Erkennung
        quant_score = committee_results["QUANTITATIVE_ALPHA"].score
        risk_score = committee_results["RISK_EXECUTION"].score
        macro_score = committee_results["MACRO_REGIME"].score
        fund_score = committee_results["FUNDAMENTAL"].score

        weighted_sum = sum(res.score * res.weight for res in committee_results.values())
        weighted_conf = sum(res.confidence * res.weight for res in committee_results.values())

        conflict_detected = False
        conflict_note = ""
        action = baseline_signal.action

        # Konflikt 1: Quant-Bulle vs. Risk-Bär (Gefährliche Bullenfalle)
        if quant_score >= 0.35 and risk_score <= -0.45:
            conflict_detected = True
            conflict_note = (
                f"Konflikt erkannt: Quantitative Alpha bullisch (+{quant_score:.2f}), "
                f"aber Risk/Mikrostruktur schlägt Alarm ({risk_score:+.2f}). Größe wird halbiert."
            )

        # Konflikt 2: Quant/Fund Bullisch vs. Makro-Kollaps / Event Horizon
        elif quant_score >= 0.25 and macro_score <= -0.55:
            conflict_detected = True
            conflict_note = (
                f"Makro-Veto: Starkes Einzeltitel-Signal (+{quant_score:.2f}), "
                f"aber Makro-Regime/Zinsentscheid blockiert ({macro_score:+.2f}). Kauf pausiert."
            )

        # 5. Conviction Tier und Größen-Multiplikator bestimmen
        # Normalisierter Conviction Score (0.0 bis 1.0)
        conviction_raw = (weighted_sum + 1.0) / 2.0  # -1..+1 -> 0..1
        conviction_score = round(max(0.05, min(0.98, conviction_raw * weighted_conf + 0.15 * (1.0 - weighted_conf))), 2)

        # Veto-Prüfung
        total_vetoes = sum(res.veto_count for res in committee_results.values())
        if total_vetoes >= 1 or macro_score <= -0.80 or (conflict_detected and macro_score <= -0.50):
            tier = "CONVICTION_UNCONVINCED_HOLD"
            size_mult = 0.0
            action = "HOLD"
            exec_summary = (
                f"🏛️ CIO VETO: Handel für {sym} blockiert ({total_vetoes} Veto(s) im Komitee). "
                f"Makro-Score: {macro_score:+.2f}, Risk-Score: {risk_score:+.2f}."
            )
        elif weighted_sum >= 0.35 and not conflict_detected and all(res.score >= -0.10 for res in committee_results.values()):
            tier = "TIER_1_ALPHA_HIGH_CONVICTION"
            size_mult = 1.25
            action = "BUY"
            exec_summary = (
                f"🏛️ CIO TIER 1 ALPHA ({conviction_score*100:.0f}% Überzeugung): "
                f"Vollständiger Komitee-Konsens (Quant {quant_score:+.2f}, Fund {fund_score:+.2f}, Makro {macro_score:+.2f}). "
                f"Allokation um +25% verstärkt."
            )
        elif weighted_sum >= 0.18 and not (conflict_detected and risk_score <= -0.60):
            tier = "TIER_2_SOLID_OPPORTUNITY"
            size_mult = 1.00 if not conflict_detected else 0.50
            action = "BUY" if baseline_signal.action == "BUY" else baseline_signal.action
            exec_summary = (
                f"🏛️ CIO TIER 2 SOLID ({conviction_score*100:.0f}% Überzeugung): Solides Setup bei kontrolliertem Risiko. "
                f"Größe: {int(size_mult*100)}%."
            )
        elif weighted_sum >= 0.05:
            tier = "TIER_3_TACTICAL_SMALL"
            size_mult = 0.50
            action = baseline_signal.action
            exec_summary = (
                f"🏛️ CIO TIER 3 TAKTISCH ({conviction_score*100:.0f}% Überzeugung): Mäßige Überzeugung / Komitees indifferent. "
                f"Halbe Standard-Größe."
            )
        else:
            tier = "CONVICTION_UNCONVINCED_HOLD"
            size_mult = 0.0
            action = "HOLD" if baseline_signal.action == "BUY" else baseline_signal.action
            exec_summary = (
                f"🏛️ CIO UNCONVINCED ({conviction_score*100:.0f}%): Komitees sehen keinen ausreichenden Edge "
                f"(Summe {weighted_sum:+.2f}). Kapital wird geschützt."
            )

        if conflict_note:
            exec_summary += f" [{conflict_note}]"

        return CIOSynthesisResult(
            symbol=sym,
            overall_conviction_score=conviction_score,
            synthesized_action=action,
            conviction_tier=tier,
            size_multiplier=size_mult,
            committee_conflict_detected=conflict_detected,
            conflict_resolution_note=conflict_note,
            committees=committee_results,
            executive_summary=exec_summary,
        )

    def _compute_regime_weights(self, macro: MacroRegime) -> Dict[str, float]:
        """Ermittelt regime-abhängige Komitee-Gewichtungen."""
        base = {
            "FUNDAMENTAL": 0.25,
            "MACRO_REGIME": 0.25,
            "QUANTITATIVE_ALPHA": 0.30,
            "RISK_EXECUTION": 0.20,
        }

        # Krisen- / Kredit-Stress-Regime: Macro & Risk wiegen dramatisch schwerer
        if (
            macro.credit_stress_active
            or getattr(macro, "geopolitical_risk_score", 0.0) >= 6.5
            or macro.regime_state == "RED_DEFENSIVE"
            or getattr(macro, "macro_event_horizon_active", False)
            or getattr(macro, "correlation_breakdown_active", False)
        ):
            base["MACRO_REGIME"] = 0.40
            base["RISK_EXECUTION"] = 0.35
            base["QUANTITATIVE_ALPHA"] = 0.15
            base["FUNDAMENTAL"] = 0.10

        # Bullen-Regime (Green Risk On): Alpha & Fundamentals gewichten höher
        elif macro.regime_state == "GREEN_RISK_ON":
            base["QUANTITATIVE_ALPHA"] = 0.40
            base["FUNDAMENTAL"] = 0.30
            base["MACRO_REGIME"] = 0.15
            base["RISK_EXECUTION"] = 0.15

        # Normalisierung auf 1.0
        tot = sum(base.values())
        return {k: round(v / tot, 3) for k, v in base.items()}

    def _aggregate_committee(
        self,
        committee_name: str,
        votes: List[SubAgentVote],
        active_subagents: Dict[str, Any],
        weight: float,
    ) -> CommitteeResult:
        if not votes:
            return CommitteeResult(
                committee_name=committee_name,
                score=0.0,
                confidence=0.50,
                weight=weight,
                dominant_rationale="Keine Agenten aktiv",
            )

        tot_w = 0.0
        w_score_sum = 0.0
        conf_sum = 0.0
        bulls = bears = vetoes = 0
        rationales: List[str] = []
        member_ids: List[str] = []

        for v in votes:
            member_ids.append(v.agent_id)
            ag = active_subagents.get(v.agent_id)
            t_w = getattr(ag, "trust_weight", 1.0) * max(0.40, v.confidence)
            tot_w += t_w
            w_score_sum += v.score * t_w
            conf_sum += v.confidence

            if v.vote_action == "VETO_BUY" or v.score <= -0.85:
                vetoes += 1
            if v.score >= 0.25:
                bulls += 1
            elif v.score <= -0.25:
                bears += 1

            if abs(v.score) >= 0.35 and v.rationale:
                rationales.append(f"{ag.name.split()[0] if ag else ''}: {v.rationale}")

        avg_score = round(w_score_sum / max(1e-6, tot_w), 2)
        avg_conf = round(conf_sum / len(votes), 2)
        dom_rat = " | ".join(rationales[:2]) if rationales else "Neutrale Komitee-Haltung"

        return CommitteeResult(
            committee_name=committee_name,
            score=avg_score,
            confidence=avg_conf,
            weight=weight,
            bull_count=bulls,
            bear_count=bears,
            veto_count=vetoes,
            dominant_rationale=dom_rat,
            member_agents=member_ids,
        )


_CIO_SYNTHESIZER_INSTANCE: Optional[CIOConvictionSynthesizer] = None


def get_cio_synthesizer() -> CIOConvictionSynthesizer:
    global _CIO_SYNTHESIZER_INSTANCE
    if _CIO_SYNTHESIZER_INSTANCE is None:
        _CIO_SYNTHESIZER_INSTANCE = CIOConvictionSynthesizer()
    return _CIO_SYNTHESIZER_INSTANCE
