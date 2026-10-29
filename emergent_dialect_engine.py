"""
emergent_dialect_engine.py
==========================
Emergent Language & Multi-Agent Deal-Negotiation Engine.
Inspiriert von Hugging Face Tokenizer-Codebooks & Multi-Agent Emergent Communication.

Ermöglicht den 26+ Handels-Spezialisten eine eigene, hochkompakte Symbol-Sprache
(Emergent Dialect Protocol), um in Mikrosekunden die besten Deals auszuhandeln:

1. Tokenizer-Codebook (Emergentes Vokabular):
   - Mathematische & Marktstruktur-Tokens:
     * Ω:PERSISTENT_HURST (Fraktale Persistenz)
     * Ψ:WHALE_FLOW (Options-Whales & Call-Accumulation)
     * Δ:SECTOR_ALPHA (Sektor-Rotation Geldzufluss)
     * ⚡:TTM_COMPRESSION_FIRED (Squeeze-Ausbruch)
     * 👔:INSIDER_CLUSTER_BUY (SEC Form 4 Cluster-Kauf)
     * 📊:WALL_ST_UPSIDE_EXPANSION (Analysten-Konsens)
     * 🛡️:DEFENSE_ENERGY_HEDGE (Geopolitisches Krisen-Alpha)
     * 🛑:CREDIT_CRUNCH_VETO (HYG/LQD Junk-Bond Warnung)
     * Σ:CONFLUENCE_QUAD / Σ:CONFLUENCE_TRIPLE (Multivariate Setup-Güte)

2. Deal-Pitch & Negotiation Handshake:
   - Pitch: Ein Spezialist formuliert einen kompakten Deal-Vektor.
   - Counter: Das Red-Team & der Risiko-Manager erwidern im Dialekt.
   - Consensus: Kelly-Sizer vergibt Allokation und genehmigt den Deal.

3. Memetische Vokabular-Evolution (Hugging Face Byte-Pair / Reinforcement):
   - Häufige Gewinner-Token-Kombinationen verschmelzen zu höherwertigen Alpha-Makros.
   - Verlustbringende Token-Kombinationen erhalten einen Malus und sterben aus.

4. Human-Decoder (Agenten-Dolmetscher):
   - Übersetzt den Dialekt live in verständliches Deutsch für Telegram (/dialect & /deals).
"""

from __future__ import annotations
import datetime
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("autonomous_agent.emergent_dialect")

DATA_DIR = Path("data")
DIALECT_FILE = DATA_DIR / "emergent_dialect_state.json"


@dataclass
class DialectToken:
    token_str: str  # z. B. "Ω:PERSISTENT_HURST"
    category: str   # "TREND", "FLOW", "CATALYST", "RISK", "CONFLUENCE"
    meaning_de: str # Bedeutung auf Deutsch
    weight: float = 1.0  # Semantisches Gewicht (steigt bei Gewinn-Deals)
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    @property
    def win_rate_pct(self) -> float:
        total = self.success_count + self.failure_count
        return round((self.success_count / total * 100.0) if total > 0 else 50.0, 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["win_rate_pct"] = self.win_rate_pct
        return d


@dataclass
class EvolvedIdiom:
    """Evolviertes N-Gramm-Sprachkonzept: Wenn 2+ Tokens hochgradig profitabel konvergieren."""
    idiom_token: str               # z. B. "💎:QUANT_SWEEP_CONVERGENCE"
    component_tokens: List[str]    # z. B. ["🎲:QUANTILE_ASYMMETRY", "🎯:LIQUIDITY_SWEEP_RECLAIM"]
    co_occurrence_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    meaning_de: str = ""
    edge_multiplier: float = 1.20  # Boost für alle Agenten bei diesem Muster
    discovered_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])

    @property
    def win_rate_pct(self) -> float:
        total = self.success_count + self.failure_count
        return round((self.success_count / total * 100.0) if total > 0 else 50.0, 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["win_rate_pct"] = self.win_rate_pct
        return d


@dataclass
class NegotiatedDealPacket:
    deal_id: str
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    dialect_sentence: str  # Kompakter String in der Agenten-Sprache
    pitching_agent: str
    countering_agent: str
    consensus_verdict: str  # "SEALED_APPROVED", "VETOED_REJECTED", "SIZE_REDUCED"
    edge_score: float
    recommended_allocation_pct: float
    plain_german_summary: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])

    def to_dict(self) -> dict:
        return asdict(self)


class EmergentDialectEngine:
    _instance: Optional[EmergentDialectEngine] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.vocabulary: Dict[str, DialectToken] = {}
        self.recent_deals: List[NegotiatedDealPacket] = []
        self.evolved_idioms: Dict[str, EvolvedIdiom] = {}
        self.pair_success: Dict[str, int] = {}
        self.pair_failure: Dict[str, int] = {}
        self.composite_macros: Dict[str, float] = {}  # Evolvierte Token-N-Gramme
        self._load()
        self._bootstrap_foundational_vocabulary()
        self._initialized = True

    # =========================================================================
    # ÖFFENTLICHE SCHNITTSTELLEN: DEAL-SYNTHESE & VERHANDLUNG
    # =========================================================================

    def compose_and_negotiate_deal(
        self,
        snapshot: Any,
        signal: Any,
        macro: Any,
        risk_evaluation: Any,
    ) -> NegotiatedDealPacket:
        """
        Formuliert das Setup in der emergenten Agenten-Sprache, führt den
        Handshake zwischen Pitcher, Counter und Risk-Manager durch und
        dokumentiert den Deal.
        """
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]
        tokens_used: List[str] = []

        # 1. Feature-Extraktion & Token-Zuweisung
        # A. Hurst Trend-Persistenz
        hurst = getattr(snapshot, "hurst_exponent", 0.50)
        if hurst >= 0.58:
            t = f"Ω:HURST_PERSIST[{hurst:.2f}]"
            self._ensure_token(t, "TREND", f"Persistenter Trendmodus (Hurst {hurst:.2f} >= 0.58)")
            tokens_used.append(t)
        elif hurst <= 0.44:
            t = f"Ω:HURST_MEAN_REV[{hurst:.2f}]"
            self._ensure_token(t, "TREND", f"Anti-persistente Mean-Reversion (Hurst {hurst:.2f})")
            tokens_used.append(t)

        # B. TTM Squeeze
        if "FIRED" in getattr(snapshot, "ttm_squeeze_status", ""):
            t = "⚡:SQUEEZE_FIRED"
            self._ensure_token(t, "MOMENTUM", "Bollinger-Band Ausbruch aus Keltner-Kompression")
            tokens_used.append(t)
        elif "COMPRESSION" in getattr(snapshot, "ttm_squeeze_status", ""):
            t = "⏳:SQUEEZE_COILING"
            self._ensure_token(t, "MOMENTUM", "Volatilitäts-Kompression vor dem Ausbruch")
            tokens_used.append(t)

        # C. Options Smart Money & Call Flow
        opt_flow = getattr(snapshot, "options_flow_sentiment", "NEUTRAL")
        opt_score = getattr(snapshot, "options_flow_score", 0.0)
        if opt_flow == "BULLISH_CALL_FLOW" or opt_score >= 0.4:
            t = f"Ψ:CALL_WHALE_BURST[+{opt_score:.2f}]"
            self._ensure_token(t, "FLOW", f"Ungewöhnlicher institutioneller Call-Flow (Score {opt_score:+.2f})")
            tokens_used.append(t)
        elif opt_score <= -0.4:
            t = f"Ψ:PUT_HEDGE_HEAVY[{opt_score:.2f}]"
            self._ensure_token(t, "FLOW", "Bärische Put-Absicherung dominiert Optionsmarkt")
            tokens_used.append(t)

        # D. Insider Form 4
        if getattr(snapshot, "insider_sentiment", "") == "STRONG_INSIDER_BUY" or getattr(snapshot, "form4_cluster_buy", False):
            t = "👔:INSIDER_CLUSTER_BUY"
            self._ensure_token(t, "CATALYST", "Vorstände & Großaktionäre kaufen gemeinsam auf freiem Markt")
            tokens_used.append(t)

        # E. Wall Street Analysten-Konsens
        analyst_cons = getattr(snapshot, "analyst_consensus", "NEUTRAL")
        upside = getattr(snapshot, "analyst_target_upside_pct", None)
        if analyst_cons in ("STRONG_BUY", "BUY") and upside and upside >= 15.0:
            t = f"📊:WALL_ST_UPSIDE[+{upside:.0f}%]"
            self._ensure_token(t, "CATALYST", f"Wall Street Konsens {analyst_cons} mit {upside:+.0f}% Kursziel-Upside")
            tokens_used.append(t)

        # F. Sektor-Rotation
        rel_str = getattr(snapshot, "relative_strength_vs_spy", 0.0)
        if rel_str >= 2.5:
            t = f"Δ:SECTOR_ALPHA_LEAD[+{rel_str:.1f}%]"
            self._ensure_token(t, "TREND", f"Sektor-Rückenwind mit relativer Stärke +{rel_str:.1f}% vs SPY")
            tokens_used.append(t)

        # G. Geopolitik & Makro
        geo_mode = getattr(macro, "geopolitical_regime_mode", "NEUTRAL_STABLE")
        geo_score = getattr(macro, "geopolitical_risk_score", 3.0)
        if geo_score >= 6.5:
            t = f"🌐:GEO_CRISIS_ESCALATION[{geo_mode}]"
            self._ensure_token(t, "RISK", f"Geopolitische Alarmstufe ({geo_mode}, Score {geo_score:.1f}/10)")
            tokens_used.append(t)
        elif geo_mode == "PEACE_DIVIDEND_EXPANSION":
            t = "🕊️:PEACE_DIVIDEND_EXPANSION"
            self._ensure_token(t, "RISK", "Globale Friedensdividende begünstigt Risiko-Assets")
            tokens_used.append(t)

        # H. Kredit-Stress
        if getattr(macro, "credit_stress_active", False):
            t = "🛑:CREDIT_CRUNCH_WARN"
            self._ensure_token(t, "RISK", "Junk-Bonds kollabieren relativ zu Investment Grade (HYG/LQD)")
            tokens_used.append(t)

        # I. VWAP & Volume-Profile Liquidity Sweep
        sweep_st = getattr(snapshot, "liquidity_sweep_status", "")
        z_vwap = getattr(snapshot, "vwap_z_score", 0.0)
        if sweep_st == "BULLISH_SWEEP_RECLAIM":
            t = f"🎯:LIQUIDITY_SWEEP_RECLAIM[{z_vwap:+.1f}σ]"
            self._ensure_token(t, "FLOW", f"Institutioneller Liquidity-Sweep & Stop-Absorption am VWAP ({z_vwap:+.1f}σ) mit Reclaim")
            tokens_used.append(t)
        elif sweep_st == "BEARISH_CLIMAX_SWEEP" or z_vwap >= 2.5:
            t = f"⚠️:VWAP_BUYING_EXHAUSTION[{z_vwap:+.1f}σ]"
            self._ensure_token(t, "RISK", f"Kauf-Erschöpfung am oberen VWAP-Band ({z_vwap:+.1f}σ)")
            tokens_used.append(t)

        # J. Probabilistischer Forecast Trichter (Kronos/TimesFM)
        asym = getattr(snapshot, "forecast_asymmetry_ratio", 1.0)
        if asym >= 1.6:
            t = f"🎲:QUANTILE_ASYMMETRY[{asym:.1f}x]"
            self._ensure_token(t, "FLOW", f"Attraktive Aufwärts-Asymmetrie im probabilistischen Trichter ({asym:.1f}x)")
            tokens_used.append(t)
        elif asym <= 0.75 and asym > 0:
            t = f"⚠️:DOWNSIDE_SKEW_WARN[{asym:.1f}x]"
            self._ensure_token(t, "RISK", f"Ungünstiges Downside-Risiko im Quantil-Trichter ({asym:.1f}x)")
            tokens_used.append(t)

        # K. CIO Conviction Tier & Correlation Breakdown Shield
        if getattr(signal, "cio_conviction_tier", "") == "TIER_1_ALPHA_HIGH_CONVICTION":
            t = "🏛️:CIO_TIER1_ALPHA"
            self._ensure_token(t, "CONFLUENCE", "Chief Investment Officer: Vollständiger Komitee-Konsens & Höchste Conviction")
            tokens_used.append(t)

        if macro is not None and getattr(macro, "correlation_breakdown_active", False):
            avg_c = getattr(macro, "average_portfolio_correlation", 0.8)
            t = f"🛑:CORRELATION_BREAKDOWN_PANIC[ρ={avg_c:.2f}]"
            self._ensure_token(t, "RISK", f"Systemischer Korrelations-Kollaps (ρ={avg_c:.2f}): Alle Assets stürzen synchron ab")
            tokens_used.append(t)

        # L. Multi-Timeframe Alignment & Proactive Risk Shield
        tf_st = getattr(snapshot, "timeframe_alignment_status", "")
        if tf_st == "TRIPLE_BULLISH_ALIGNMENT":
            t = "📐:TRIPLE_TIMEFRAME_ALIGN"
            self._ensure_token(t, "TREND", "Perfekte fraktale Resonanz: Daily, 4H und 1H synchron im Aufwärtstrend")
            tokens_used.append(t)

        proactive_risk = getattr(snapshot, "proactive_risk_alert", "NORMAL")
        void_detected = getattr(snapshot, "liquidity_void_detected", False)
        if void_detected or proactive_risk == "LIQUIDITY_VOID_EVACUATE":
            t = "🚨:LIQUIDITY_VOID_EVACUATE"
            self._ensure_token(t, "RISK", "Orderbuch-Vakuum & Liquiditäts-Schwund: Notausstieg zum Kapitalschutz")
            tokens_used.append(t)
        elif proactive_risk == "SURGE_VOLATILITY_HAZARD" or getattr(snapshot, "volatility_spike_ratio", 1.0) >= 2.0:
            t = "⚡:PROACTIVE_VOLATILITY_SHOCK"
            self._ensure_token(t, "RISK", "Volatilitäts-Explosion erkannt: Proaktive Breakeven-Stop-Nachziehung")
            tokens_used.append(t)

        # M. Taktische Einstiegs-Abwägung (Sofort-Ausführung vs. Geduldiges Abwarten auf Pullback/Retest)
        exec_timing = getattr(signal, "execution_timing", "IMMEDIATE")
        opt_entry = getattr(signal, "optimal_entry_price", None)
        if exec_timing == "WAIT_FOR_PULLBACK":
            t = f"⏳:ENTRY_PULLBACK_WAIT[{opt_entry:.2f}]" if opt_entry else "⏳:ENTRY_PULLBACK_WAIT"
            self._ensure_token("⏳:ENTRY_PULLBACK_WAIT", "TIMING", "Taktisches Abwarten auf Pullback/Retest zur Renditemaximierung")
            tokens_used.append(t)
        elif exec_timing == "IMMEDIATE" and signal.action in ("BUY", "SELL"):
            t = "⚡:ENTRY_IMMEDIATE"
            self._ensure_token("⚡:ENTRY_IMMEDIATE", "TIMING", "Sofortiger Markteinstieg bei direktem Ausbruchs-Momentum")
            tokens_used.append(t)
        elif exec_timing == "WAIT_FOR_RELIEF_BOUNCE":
            t = "⏳:WAIT_RELIEF_BOUNCE"
            self._ensure_token("⏳:WAIT_RELIEF_BOUNCE", "TIMING", "Überverkauft: Abwarten auf Entlastungs-Bounce für besseren Exit")
            tokens_used.append(t)

        # N. Confluence-Grad (Güteklasse des Setups)
        bullish_tokens = [t for t in tokens_used if not t.startswith("🛑") and not t.startswith("Ψ:PUT") and not t.startswith("🌐")]
        conf_len = len(bullish_tokens)
        if conf_len >= 4:
            c_tag = "Σ:CONFLUENCE_QUAD_A_PLUS"
        elif conf_len >= 3:
            c_tag = "Σ:CONFLUENCE_TRIPLE_STRONG"
        elif conf_len >= 2:
            c_tag = "Σ:CONFLUENCE_DUAL_SOLID"
        else:
            c_tag = "Σ:SINGLE_VECTOR"
        self._ensure_token(c_tag, "CONFLUENCE", f"{conf_len}-fache Signal-Bestätigung über unabhängige Domänen")
        tokens_used.append(c_tag)

        # M. Evolvierte Schwarm-Idiome prüfen (Eigene Agenten-Sprache)
        base_tokens_present = {t.split("[")[0] for t in tokens_used}
        for idiom in self.evolved_idioms.values():
            if all(comp in base_tokens_present for comp in idiom.component_tokens):
                t_str = f"{idiom.idiom_token}[+{idiom.edge_multiplier:.2f}x]"
                self._ensure_token(idiom.idiom_token, "EVOLVED_IDIOM", idiom.meaning_de)
                tokens_used.append(t_str)
                signal.expected_edge_score = round(signal.expected_edge_score * idiom.edge_multiplier, 3)

        # 2. Urteils-Synthese
        verdict = "SEALED_APPROVED" if risk_evaluation.approved else "VETOED_REJECTED"
        if risk_evaluation.approved and getattr(signal, "debate_result", None) and signal.debate_result.verdict == "REDUCED_SIZE":
            verdict = "SIZE_REDUCED"

        # 3. Dialekt-Satz formen
        tokens_str = " · ".join(tokens_used)
        action_sym = "🟢 KAUF" if signal.action == "BUY" else ("🔴 SHORT/VERKAUF" if signal.action == "SELL" else "⏸️ CASH")
        deal_dialect = f"⟪PITCH::{snapshot.symbol} | ACT:{signal.action} | {tokens_str} | VERDICT:{verdict} | EDGE:{signal.expected_edge_score:.2f}⟫"

        # 4. Plain-German Übersetzung für Henry generieren
        plain_text = self._decode_sentence_to_german(
            symbol=snapshot.symbol,
            action=action_sym,
            tokens=tokens_used,
            verdict=verdict,
            edge=signal.expected_edge_score,
            alloc=getattr(risk_evaluation, "position_size_usd", 0.0),
        )

        packet = NegotiatedDealPacket(
            deal_id=f"DEAL_{snapshot.symbol}_{now_str.replace(':', '').replace('-', '')}",
            symbol=snapshot.symbol,
            action=signal.action,
            dialect_sentence=deal_dialect,
            pitching_agent=getattr(signal, "analysis_source", "Swarm_Pitcher"),
            countering_agent="Red_Team_Tribunal",
            consensus_verdict=verdict,
            edge_score=round(signal.expected_edge_score, 2),
            recommended_allocation_pct=round((getattr(risk_evaluation, "position_size_usd", 0.0) / 1000.0), 1),
            plain_german_summary=plain_text,
            timestamp=now_str,
        )

        # Aktualisiere Token-Nutzung
        for t in tokens_used:
            tok = self.vocabulary.get(t.split("[")[0]) or self.vocabulary.get(t)
            if tok:
                tok.usage_count += 1

        self.recent_deals.append(packet)
        self.recent_deals = self.recent_deals[-25:]
        self._save()
        return packet

    def record_deal_profitability(self, symbol: str, move_pct: float) -> None:
        """
        Feedback-Schleife (Reinforcement & Idiom-Evolution): Belohnt Tokens erfolgreicher Deals,
        senkt das Gewicht unprofitabler Tokens und synthetisiert bei wiederholtem gemeinsamen Erfolg
        autonom evolvierte Schwarm-Idiome (Eigene Agenten-Sprache), die per SwarmKnowledgeBus
        an alle 30 Unteragenten zur kollektiven Konfidenz-Steigerung gebroadcastet werden!
        """
        is_success = move_pct >= 1.5
        is_failure = move_pct <= -1.5

        if not is_success and not is_failure:
            return

        # Finde jüngste Deals für dieses Symbol
        matching = [d for d in self.recent_deals if d.symbol == symbol]
        for deal in matching:
            # Parse alle aktiven Basis-Tokens aus dem Dialekt-Satz
            deal_tokens = []
            for t_raw in self.vocabulary.values():
                base_key = t_raw.token_str.split("[")[0]
                if base_key in deal.dialect_sentence:
                    deal_tokens.append(base_key)
                    if is_success:
                        t_raw.success_count += 1
                        t_raw.weight = round(min(2.5, t_raw.weight + 0.08), 3)
                    else:
                        t_raw.failure_count += 1
                        t_raw.weight = round(max(0.3, t_raw.weight - 0.10), 3)

            # Filter relevante Tokens für Paar-Konvergenz (schließe reine Confluence-Summen aus)
            clean_tokens = [t for t in set(deal_tokens) if not t.startswith("Σ:") and not t.startswith("🧬:") and not t.startswith("💎:") and not t.startswith("🐋:") and not t.startswith("🚀:") and not t.startswith("👑:")]

            # Paarweise Erfolgs- und Misserfolgsanalyse für emergente Schwarm-Sprache
            for i in range(len(clean_tokens)):
                for j in range(i + 1, len(clean_tokens)):
                    t1, t2 = sorted([clean_tokens[i], clean_tokens[j]])
                    pair_key = f"{t1} + {t2}"
                    if is_success:
                        self.pair_success[pair_key] = self.pair_success.get(pair_key, 0) + 1
                        # Wenn ein Paar mindestens 2 erfolgreiche Deals gemeinsam feiert -> Evolvierte Sprache!
                        if self.pair_success[pair_key] >= 2:
                            self._evolve_and_broadcast_idiom(
                                t1=t1,
                                t2=t2,
                                succ=self.pair_success[pair_key],
                                fail=self.pair_failure.get(pair_key, 0),
                                symbol=symbol,
                            )
                    else:
                        self.pair_failure[pair_key] = self.pair_failure.get(pair_key, 0) + 1

        self._save()

    def _evolve_and_broadcast_idiom(
        self, t1: str, t2: str, succ: int, fail: int, symbol: str
    ) -> Optional[EvolvedIdiom]:
        """
        Synthetisiert ein neues Idiom (Schwarm-Sprache) und sendet es über den
        SwarmKnowledgeBus an alle 30 Unteragenten für kollektiven Edge-Zuwachs.
        """
        pair_set = {t1, t2}

        # Semantischer Idiom-Name basierend auf Konvergenz-Kombination
        if any("QUANTILE" in t for t in pair_set) and any("LIQUIDITY" in t for t in pair_set):
            idiom_name = "💎:QUANT_SWEEP_CONVERGENCE"
            meaning = "Asymmetrische Quantil-Verteilung trifft auf institutionellen Liquiditäts-Reclaim"
        elif any("WHALE" in t for t in pair_set) and any("INSIDER" in t for t in pair_set):
            idiom_name = "🐋:WHALE_INSIDER_APEX"
            meaning = "Whale-Call-Burst konvergiert synchron mit Insider-Cluster-Käufen"
        elif any("SQUEEZE" in t for t in pair_set) and any("HURST" in t for t in pair_set):
            idiom_name = "🚀:FRACTAL_MOMENTUM_IGNITION"
            meaning = "Persistenter Hurst-Trend bricht aus extremer TTM-Volatilitäts-Kompression aus"
        elif any("WALL_ST" in t for t in pair_set) and any("SECTOR" in t for t in pair_set):
            idiom_name = "👑:SOVEREIGN_ALPHA_HARVEST"
            meaning = "Relative Branchenstärke vereint mit Wall Street Konsens-Kursziel"
        else:
            c1 = t1.split(":")[-1][:8]
            c2 = t2.split(":")[-1][:8]
            idiom_name = f"🧬:{c1}_{c2}"
            meaning = f"Synthetische Schwarm-Konvergenz aus {t1} und {t2}"

        edge_mult = round(min(1.35, 1.15 + (succ / max(1, succ + fail)) * 0.20), 2)

        idiom = self.evolved_idioms.get(idiom_name)
        if idiom:
            idiom.co_occurrence_count += 1
            idiom.success_count = succ
            idiom.failure_count = fail
            idiom.edge_multiplier = edge_mult
        else:
            idiom = EvolvedIdiom(
                idiom_token=idiom_name,
                component_tokens=sorted([t1, t2]),
                co_occurrence_count=succ + fail,
                success_count=succ,
                failure_count=fail,
                meaning_de=meaning,
                edge_multiplier=edge_mult,
            )
            self.evolved_idioms[idiom_name] = idiom
            self._ensure_token(idiom_name, "EVOLVED_IDIOM", meaning)

        # Broadcast über SwarmKnowledgeBus für alle 30 Unteragenten
        try:
            from swarm_knowledge_bus import get_swarm_knowledge_bus, KnowledgeInsight
            bus = get_swarm_knowledge_bus()
            ins_id = f"IDIOM_{idiom_name}"
            bus.insights[ins_id] = KnowledgeInsight(
                insight_id=ins_id,
                category="ALPHA_SYNERGY",
                author_agent_id="emergent_dialect_engine",
                author_agent_name="🗣️ Dialekt-Synthesizer",
                title=f"Evolviertes Schwarm-Idiom: {idiom_name}",
                lesson=f"Sprachkonvergenz {idiom_name} ({meaning}) erzeugt +{(edge_mult-1.0)*100:.0f}% Edge.",
                symbol_origin=symbol,
                confidence_impact=round(edge_mult - 1.0, 2),
                target_focuses=["*"],
                verifications_count=succ,
                accuracy_pct=round(succ / max(1, succ + fail) * 100.0, 1),
            )
            bus._save()
        except Exception as e:
            logger.debug("SwarmKnowledgeBus-Broadcast fehlgeschlagen: %s", e)

        return idiom

    def format_telegram_deals_summary(self) -> str:
        """Erstellt eine hochgradig lesbare Telegram-Übersicht des Agenten-Dialekts."""
        lines = ["🗣️ <b>Emergent Agent Dialect & Deal-Negotiator:</b>\n─────────────────────────────"]
        lines.append(f"• <b>Aktiver Wortschatz:</b> <b>{len(self.vocabulary)} Deal-Tokens</b>")
        lines.append(f"• <b>Verhandelte Deals (historisch):</b> <b>{len(self.recent_deals)}</b>\n")

        if not self.recent_deals:
            lines.append("<i>Aktuell noch keine verhandelten Deals im Speicher. Der Schwarm formuliert Deals beim nächsten Markt-Scan!</i>")
            return "\n".join(lines)

        latest = self.recent_deals[-1]
        lines.append(f"🔥 <b>Jüngster Schwarm-Deal: <code>{latest.symbol}</code></b>")
        lines.append(f"📜 <b>Agenten-Dialekt:</b>\n<code>{latest.dialect_sentence}</code>\n")
        lines.append(f"🗣️ <b>Dolmetscher-Übersetzung:</b>\n{latest.plain_german_summary}\n")

        # Evolvierte Schwarm-Idiome (Eigene Agenten-Sprache)
        if self.evolved_idioms:
            lines.append("🧬 <b>Evolvierte Schwarm-Idiome (Eigene Agenten-Sprache):</b>")
            for idiom in list(self.evolved_idioms.values())[:3]:
                lines.append(f"• <code>{idiom.idiom_token}</code> (+{idiom.edge_multiplier:.2f}x | {idiom.win_rate_pct:.0f}% Win)")
                lines.append(f"   <i>{idiom.meaning_de}</i>")
            lines.append("")

        # Top 3 wertvollste Evolved Tokens
        top_tokens = sorted(self.vocabulary.values(), key=lambda t: t.weight * max(1, t.usage_count), reverse=True)[:3]
        lines.append("💎 <b>Top-bewertete Dialekt-Tokens (Höchste Win-Rate):</b>")
        for t in top_tokens:
            lines.append(f"• <code>{t.token_str}</code> (Gewicht {t.weight:.2f}x | {t.win_rate_pct:.0f}% Win)")
            lines.append(f"   <i>{t.meaning_de}</i>")

        lines.append("\n<i>Die Agenten handeln Deals in Mikrosekunden aus, bevor eine Order erteilt wird!</i>")
        return "\n".join(lines)

    # =========================================================================
    # INTERNE DOLMETSCHER- & PARSING-LOGIK
    # =========================================================================

    def _decode_sentence_to_german(
        self,
        symbol: str,
        action: str,
        tokens: List[str],
        verdict: str,
        edge: float,
        alloc: float,
    ) -> str:
        explanations = []
        for t in tokens:
            base = t.split("[")[0]
            tok = self.vocabulary.get(base) or self.vocabulary.get(t)
            if tok and not base.startswith("Σ:"):
                explanations.append(f"• {tok.meaning_de}")

        exp_str = "\n".join(explanations)
        verdict_icon = "✅ <b>GENEHMIGT</b>" if verdict == "SEALED_APPROVED" else ("🛡️ <b>VETO ABGELEHNT</b>" if verdict == "VETOED_REJECTED" else "🟡 <b>HALBE GRÖSSE</b>")

        return (
            f"<b>{action} {symbol}</b> ({verdict_icon} · Edge: <b>{edge:.2f}</b>)\n"
            f"<b>Begründung der Agenten:</b>\n{exp_str}\n"
            f"<b>Einsatz:</b> ${alloc:,.0f} Margin"
        )

    def _ensure_token(self, token_str: str, category: str, meaning_de: str) -> None:
        base = token_str.split("[")[0]
        if base not in self.vocabulary:
            self.vocabulary[base] = DialectToken(
                token_str=base,
                category=category,
                meaning_de=meaning_de,
                weight=1.0,
            )

    def _bootstrap_foundational_vocabulary(self) -> None:
        if self.vocabulary:
            return

        core = [
            ("Ω:HURST_PERSIST", "TREND", "Persistenter Trendmodus (Hurst >= 0.58) – Kein Random-Walk"),
            ("Ω:HURST_MEAN_REV", "TREND", "Anti-persistente Mean-Reversion – Rückkehr zum Mittelwert"),
            ("⚡:SQUEEZE_FIRED", "MOMENTUM", "TTM Volatilitäts-Ausbruch zündet mit dynamischem Impuls"),
            ("⏳:SQUEEZE_COILING", "MOMENTUM", "Bollinger-Bänder komprimieren in Keltner-Kanälen"),
            ("Ψ:CALL_WHALE_BURST", "FLOW", "Ungewöhnlicher institutioneller Call-Flow (Whale-Akkumulation)"),
            ("Ψ:PUT_HEDGE_HEAVY", "FLOW", "Bärische Put-Absicherung dominiert"),
            ("👔:INSIDER_CLUSTER_BUY", "CATALYST", "Vorstände & CEOs kaufen zeitgleich Aktien im freien Markt"),
            ("📊:WALL_ST_UPSIDE", "CATALYST", "Wall Street Konsens (Buy) mit starkem Kursziel-Aufwärtspotenzial"),
            ("Δ:SECTOR_ALPHA_LEAD", "TREND", "Führende relative Branchen-Stärke vs. S&P 500"),
            ("🌐:GEO_CRISIS_ESCALATION", "RISK", "Geopolitische Kriseneskalation – Rüstung/Energie bevorzugt"),
            ("🕊️:PEACE_DIVIDEND_EXPANSION", "RISK", "Geopolitische Entspannung begünstigt Tech & Growth"),
            ("🛑:CREDIT_CRUNCH_WARN", "RISK", "Junk-Bond Kreditspreads kollabieren – Systemrisiko"),
            ("Σ:CONFLUENCE_QUAD_A_PLUS", "CONFLUENCE", "A+ Setup: 4 voneinander unabhängige Domänen bestätigen den Einstieg"),
            ("Σ:CONFLUENCE_TRIPLE_STRONG", "CONFLUENCE", "Starkes Setup: 3 voneinander unabhängige Domänen bestätigen"),
            ("Σ:CONFLUENCE_DUAL_SOLID", "CONFLUENCE", "Solides Setup: 2 Domänen bestätigen den Trade"),
        ]
        for t, cat, m in core:
            self.vocabulary[t] = DialectToken(token_str=t, category=cat, meaning_de=m)
        self._save()

    def _load(self) -> None:
        try:
            if DIALECT_FILE.exists():
                with open(DIALECT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.get("vocabulary", {}).items():
                        v_clean = dict(v)
                        v_clean.pop("win_rate_pct", None)
                        self.vocabulary[k] = DialectToken(**v_clean)
                    for d in data.get("recent_deals", []):
                        self.recent_deals.append(NegotiatedDealPacket(**d))
                    for k, v in data.get("evolved_idioms", {}).items():
                        v_clean = dict(v)
                        v_clean.pop("win_rate_pct", None)
                        self.evolved_idioms[k] = EvolvedIdiom(**v_clean)
                    self.pair_success = data.get("pair_success", {})
                    self.pair_failure = data.get("pair_failure", {})
                    self.composite_macros = data.get("composite_macros", {})
        except Exception as e:
            logger.warning("Konnte Emergent Dialect State nicht laden: %s", e)

    def _save(self) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(DIALECT_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "vocabulary": {k: v.to_dict() for k, v in self.vocabulary.items()},
                        "recent_deals": [d.to_dict() for d in self.recent_deals],
                        "evolved_idioms": {k: v.to_dict() for k, v in self.evolved_idioms.items()},
                        "pair_success": self.pair_success,
                        "pair_failure": self.pair_failure,
                        "composite_macros": self.composite_macros,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass


def get_dialect_engine() -> EmergentDialectEngine:
    return EmergentDialectEngine()
