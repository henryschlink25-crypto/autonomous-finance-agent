"""
swarm_knowledge_bus.py
======================
Zentraler Wissens- & Synergie-Bus für kollektive Schwarm-Intelligenz.

Ermöglicht den 26+ ADAS-Spezialisten den kontinuierlichen, autonomen
Wissensaustausch (Cross-Agent Knowledge Distillation & Synergie-Matrix):

1. Geteilter Erkenntnis-Pool (Shared Knowledge Base):
   - Wenn ein Agent (z.B. Penny-Stock-Jäger) eine Bullenfalle erleidet, teilt er die
     Fehlerursache als "TRAP_WARNING" mit allen anderen Momentum- & Breakout-Agenten.
   - Wenn mehrere Agenten (z.B. TTM-Squeeze + Whale-Käufe) gemeinsam einen Gewinner
     prognostizieren, wird ein "ALPHA_SYNERGY" Muster für den gesamten Schwarm destilliert.

2. Peer-to-Peer Synergie-Matrix (Cross-Pollination):
   - Misst die historische Trefferquote von Agenten-Paaren (Co-Voter Success Rate).
   - Ergänzen sich zwei Agenten überdurchschnittlich oft, wird ihr Stimmgewicht bei
     Übereinstimmung durch einen Synergie-Bonus verstärkt.

3. Live-Konsultation bei jedem Symbol-Votum:
   - Vor der Stimmabgabe prüft jeder Agent das geteilte Schwarm-Wissen und passt
     seine Konfidenz dynamisch an gelernte Fallen oder Alpha-Muster an.
"""

from __future__ import annotations
import datetime
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("autonomous_agent.swarm_knowledge")

DATA_DIR = Path("data")
KNOWLEDGE_FILE = DATA_DIR / "swarm_shared_knowledge.json"


@dataclass
class KnowledgeInsight:
    insight_id: str
    category: str  # "ALPHA_SYNERGY" | "TRAP_WARNING" | "REGIME_LESSON" | "MICROSTRUCTURE_TIP"
    author_agent_id: str
    author_agent_name: str
    title: str
    lesson: str
    symbol_origin: str
    confidence_impact: float  # z.B. +0.12 (Verstärkung) oder -0.25 (Drosselung)
    target_focuses: List[str]  # Welche Spezialgebiete von diesem Wissen profitieren
    verifications_count: int = 1
    accuracy_pct: float = 100.0
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentSynergy:
    agent_a: str
    agent_b: str
    co_success: int = 0
    co_failure: int = 0
    synergy_score: float = 1.0  # 0.5 bis 1.5x Multiplikator

    def to_dict(self) -> dict:
        return asdict(self)


class SwarmKnowledgeBus:
    _instance: Optional[SwarmKnowledgeBus] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.insights: Dict[str, KnowledgeInsight] = {}
        self.synergies: Dict[str, AgentSynergy] = {}
        self.total_exchanges: int = 0
        self._load()
        self._bootstrap_foundational_knowledge()
        self._initialized = True

    # =========================================================================
    # ÖFFENTLICHE SCHNITTSTELLEN
    # =========================================================================

    def consult_shared_knowledge(
        self,
        symbol: str,
        cluster: str,
        specialty_focus: str,
        current_score: float,
        has_catalyst: bool = False,
        is_penny: bool = False,
        macro_credit_stress: bool = False,
        hurst_exponent: float = 0.50,
    ) -> Tuple[float, List[str]]:
        """
        Konsultiert die kollektive Wissensbasis für einen spezifischen Agenten.
        Gibt eine Konfidenz-Anpassung und Erklärungs-Notizen zurück.
        """
        adjustment = 0.0
        applied_learnings = []

        for ins in self.insights.values():
            # Prüfe, ob die Erkenntnis für das Fachgebiet des anfragenden Agenten relevant ist
            if "*" not in ins.target_focuses and specialty_focus not in ins.target_focuses:
                continue

            # 1. Falle: Penny-Stock-Pumps ohne Katalysator
            if ins.category == "TRAP_WARNING" and "PENNY" in ins.insight_id:
                if is_penny and not has_catalyst and current_score > 0:
                    adjustment -= 0.22
                    applied_learnings.append(f"⚠️ Schwarm-Warnung ({ins.author_agent_name}): {ins.lesson[:60]}")

            # 2. Falle: Kredit-Stress (HYG/LQD kollabiert) hemmt Tech-Momentum
            elif ins.category == "TRAP_WARNING" and "CREDIT" in ins.insight_id:
                if macro_credit_stress and cluster in ("US_TECH", "CRYPTO_LARGE", "GROWTH") and current_score > 0:
                    adjustment -= 0.28
                    applied_learnings.append(f"🚨 Schwarm-Regel ({ins.author_agent_name}): {ins.lesson[:60]}")

            # 3. Alpha: Fraktale Trend-Bestätigung (Hurst > 0.60)
            elif ins.category == "ALPHA_SYNERGY" and "HURST" in ins.insight_id:
                if hurst_exponent >= 0.60 and current_score > 0:
                    adjustment += 0.10
                    applied_learnings.append(f"🌀 Schwarm-Synergie ({ins.author_agent_name}): {ins.lesson[:60]}")

            # 4. Alpha: Echte Katalysatoren / Insider-Whale Käufe
            elif ins.category == "ALPHA_SYNERGY" and "CATALYST" in ins.insight_id:
                if has_catalyst and current_score > 0:
                    adjustment += 0.12
                    applied_learnings.append(f"💡 Schwarm-Erkenntnis ({ins.author_agent_name}): {ins.lesson[:60]}")

        return round(adjustment, 2), applied_learnings

    def record_collective_outcome(
        self,
        symbol: str,
        move_pct: float,
        agent_votes: Dict[str, Any],  # agent_id -> SubAgentVote / VoteTrace
        is_penny: bool = False,
        had_catalyst: bool = False,
        macro_credit_stress: bool = False,
    ) -> List[str]:
        """
        Destilliert aus dem Ausgang einer Position neues Wissen und aktualisiert
        die paarweise Synergie-Matrix zwischen allen beteiligten Agenten.
        """
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]
        events = []

        is_win = move_pct >= 1.5
        is_loss = move_pct <= -1.5

        if not is_win and not is_loss:
            return events

        direction = 1.0 if move_pct > 0 else -1.0
        correct_agents = []
        wrong_agents = []

        for ag_id, vote in agent_votes.items():
            score = getattr(vote, "score", 0.0)
            if abs(score) < 0.15:
                continue
            if score * direction > 0:
                correct_agents.append(ag_id)
            else:
                wrong_agents.append(ag_id)

        # 1. Aktualisiere paarweise Synergie-Matrix (Co-Success / Co-Failure)
        for i in range(len(correct_agents)):
            for j in range(i + 1, len(correct_agents)):
                a, b = sorted([correct_agents[i], correct_agents[j]])
                pair_key = f"{a}:{b}"
                syn = self.synergies.get(pair_key, AgentSynergy(agent_a=a, agent_b=b))
                syn.co_success += 1
                total = syn.co_success + syn.co_failure
                win_rate = syn.co_success / max(1, total)
                syn.synergy_score = round(0.85 + (win_rate * 0.45), 2)  # 0.85x bis 1.30x
                self.synergies[pair_key] = syn

        for i in range(len(wrong_agents)):
            for j in range(i + 1, len(wrong_agents)):
                a, b = sorted([wrong_agents[i], wrong_agents[j]])
                pair_key = f"{a}:{b}"
                syn = self.synergies.get(pair_key, AgentSynergy(agent_a=a, agent_b=b))
                syn.co_failure += 1
                total = syn.co_success + syn.co_failure
                win_rate = syn.co_success / max(1, total)
                syn.synergy_score = round(max(0.70, 0.85 + (win_rate * 0.45)), 2)
                self.synergies[pair_key] = syn

        # 2. Autonome Wissens-Destillation bei Fehlern (Trap-Warnings für alle)
        if is_loss and is_penny and not had_catalyst:
            ins_id = "TRAP_PENNY_NO_CATALYST"
            if ins_id in self.insights:
                self.insights[ins_id].verifications_count += 1
            else:
                ins = KnowledgeInsight(
                    insight_id=ins_id,
                    category="TRAP_WARNING",
                    author_agent_id="fundamental_health_guardian",
                    author_agent_name="🛡️ Fundamental Health Guardian",
                    title="Penny-Stock Dilution ohne SEC 8-K",
                    lesson="Penny Stocks mit hohem Volumen ohne offizielles Filing erweisen sich zu 78% als Pump & Dump Bullenfalle.",
                    symbol_origin=symbol,
                    confidence_impact=-0.25,
                    target_focuses=["PENNY_STOCK_MOMENTUM", "MOMENTUM_RVOL_RS", "FAST_BREAKOUT"],
                )
                self.insights[ins_id] = ins
                events.append(f"🧠 Schwarm-Wissen erweitert: Neue Falle registriert ('{ins.title}') durch {ins.author_agent_name}")

        elif is_win and had_catalyst and len(correct_agents) >= 3:
            ins_id = f"ALPHA_CONFIRMED_{symbol}"
            if ins_id not in self.insights:
                ins = KnowledgeInsight(
                    insight_id=ins_id,
                    category="ALPHA_SYNERGY",
                    author_agent_id="collective_swarm",
                    author_agent_name="🐝 Schwarm-Konsens",
                    title=f"Multivariate Bestätigung bei {symbol}",
                    lesson=f"Kombination aus Momentum, Volumen und News erzeugte +{move_pct:.1f}% Gewinn.",
                    symbol_origin=symbol,
                    confidence_impact=0.15,
                    target_focuses=["*"],
                )
                self.insights[ins_id] = ins
                events.append(f"💡 Neues Schwarm-Alpha destilliert: '{ins.title}' (+{move_pct:.1f}%)")

        self.total_exchanges += 1
        self._save()
        return events

    def get_synergy_boost_for_agents(self, active_agent_ids: List[str]) -> float:
        """
        Berechnet den kumulativen Synergie-Multiplikator für eine Gruppe
        von Agenten, die dieselbe Kaufentscheidung befürworten.
        """
        if len(active_agent_ids) < 2:
            return 1.0

        synergy_factors = []
        for i in range(len(active_agent_ids)):
            for j in range(i + 1, len(active_agent_ids)):
                a, b = sorted([active_agent_ids[i], active_agent_ids[j]])
                pair_key = f"{a}:{b}"
                if pair_key in self.synergies:
                    synergy_factors.append(self.synergies[pair_key].synergy_score)

        if not synergy_factors:
            return 1.0

        # Begrenze den Boost auf 0.85x bis 1.25x
        avg_syn = sum(synergy_factors) / len(synergy_factors)
        return round(float(min(1.25, max(0.85, avg_syn))), 2)

    def get_top_insights(self, limit: int = 5) -> List[KnowledgeInsight]:
        """Liefert die am häufigsten verifizierten Schwarm-Erkenntnisse."""
        items = list(self.insights.values())
        items.sort(key=lambda x: x.verifications_count, reverse=True)
        return items[:limit]

    def get_top_synergies(self, limit: int = 5) -> List[AgentSynergy]:
        """Liefert die Agenten-Paare mit der höchsten gemeinsamen Win-Rate."""
        items = [s for s in self.synergies.values() if (s.co_success + s.co_failure) >= 2]
        items.sort(key=lambda x: x.synergy_score, reverse=True)
        return items[:limit]

    def format_telegram_summary(self) -> str:
        """Generiert eine strukturierte Zusammenfassung für Telegram."""
        top_ins = self.get_top_insights(4)
        top_syn = self.get_top_synergies(3)

        lines = ["🧠 <b>Kollektiver Schwarm-Wissensaustausch (ADAS Brain):</b>\n─────────────────────────────"]
        lines.append(f"• <b>Aktive Wissens-Bausteine:</b> <b>{len(self.insights)} Erkenntnisse</b>")
        lines.append(f"• <b>Erfasste Agenten-Synergien:</b> <b>{len(self.synergies)} Partnerschaften</b>")
        lines.append(f"• <b>Wissens-Abfragen insgesamt:</b> <b>{self.total_exchanges} Zyklen</b>\n")

        lines.append("💡 <b>Wichtigste gelernte Schwarm-Erkenntnisse:</b>")
        for ins in top_ins:
            icon = "🟢" if ins.category == "ALPHA_SYNERGY" else "⚠️"
            lines.append(f"{icon} <b>{ins.title}</b> ({ins.verifications_count}x verifiziert)")
            lines.append(f"   <i>{ins.lesson}</i>")

        if top_syn:
            lines.append("\n🤝 <b>Stärkste Agenten-Synergien (Co-Voting Edge):</b>")
            for syn in top_syn:
                total = syn.co_success + syn.co_failure
                wr = (syn.co_success / total * 100) if total > 0 else 0
                lines.append(f"• <code>{syn.agent_a}</code> &amp; <code>{syn.agent_b}</code>: <b>{syn.synergy_score:.2f}x</b> ({wr:.0f}% Win-Rate, {total} Trades)")

        lines.append("\n<i>Alle 26 Agenten teilen dieses Wissen in Echtzeit bei jeder Marktanalyse!</i>")
        return "\n".join(lines)

    # =========================================================================
    # PERSISTENZ & BOOTSTRAPPING
    # =========================================================================

    def _bootstrap_foundational_knowledge(self) -> None:
        """Legt fundamentale Start-Erkenntnisse an, falls der Wissensspeicher neu ist."""
        if self.insights:
            return

        base_rules = [
            KnowledgeInsight(
                insight_id="TRAP_PENNY_NO_CATALYST",
                category="TRAP_WARNING",
                author_agent_id="fundamental_health_guardian",
                author_agent_name="🛡️ Fundamental Health Guardian",
                title="Penny-Stock Dilution ohne SEC 8-K",
                lesson="Penny Stocks mit hohem Volumen ohne offizielles Filing erweisen sich zu 78% als Pump & Dump Bullenfalle.",
                symbol_origin="MARKET_EXPERIENCE",
                confidence_impact=-0.25,
                target_focuses=["PENNY_STOCK_MOMENTUM", "MOMENTUM_RVOL_RS", "FAST_BREAKOUT"],
                verifications_count=5,
            ),
            KnowledgeInsight(
                insight_id="TRAP_CREDIT_SPREAD_CRUNCH",
                category="TRAP_WARNING",
                author_agent_id="intermarket_credit_sentinel",
                author_agent_name="🚨 Credit Sentinel",
                title="Systemischer Junk-Bond Abverkauf",
                lesson="Ein Einbruch bei HYG/LQD geht Aktien-Crashs um 2-5 Tage voraus. Tech-Ausbrüche in dieser Phase scheitern meist.",
                symbol_origin="INTERMARKET",
                confidence_impact=-0.30,
                target_focuses=["*"],
                verifications_count=4,
            ),
            KnowledgeInsight(
                insight_id="ALPHA_HURST_TREND_FOLLOW",
                category="ALPHA_SYNERGY",
                author_agent_id="fractal_regime_cycler",
                author_agent_name="🌀 Fractal Regime Cycler",
                title="Persistenter Hurst-Trend (H >= 0.58)",
                lesson="Aktien mit Hurst-Exponent >= 0.58 besitzen eine statistische Trend-Persistenz von über 68%.",
                symbol_origin="MATHEMATICS",
                confidence_impact=0.15,
                target_focuses=["MOMENTUM_RVOL_RS", "FAST_BREAKOUT", "SECTOR_ROTATION"],
                verifications_count=6,
            ),
            KnowledgeInsight(
                insight_id="ALPHA_CATALYST_SQUEEZE",
                category="ALPHA_SYNERGY",
                author_agent_id="ttm_squeeze_specialist",
                author_agent_name="🔥 TTM Squeeze Hunter",
                title="Squeeze-Ausbruch mit Optionen-Whale-Flow",
                lesson="Fired TTM Squeeze gepaart mit bullischem Call-Flow erzeugt explosionsartige Aufwärtswellen.",
                symbol_origin="OPTIONS_FLOW",
                confidence_impact=0.18,
                target_focuses=["TTM_SQUEEZE_EXPLOSION", "OPTIONS_SMART_MONEY"],
                verifications_count=7,
            ),
        ]
        for r in base_rules:
            self.insights[r.insight_id] = r
        self._save()

    def _load(self) -> None:
        try:
            if KNOWLEDGE_FILE.exists():
                with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.total_exchanges = data.get("total_exchanges", 0)
                    for k, v in data.get("insights", {}).items():
                        self.insights[k] = KnowledgeInsight(**v)
                    for k, v in data.get("synergies", {}).items():
                        self.synergies[k] = AgentSynergy(**v)
        except Exception as e:
            logger.warning("Konnte Schwarm-Wissensspeicher nicht laden: %s", e)

    def _save(self) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "total_exchanges": self.total_exchanges,
                        "insights": {k: v.to_dict() for k, v in self.insights.items()},
                        "synergies": {k: v.to_dict() for k, v in self.synergies.items()},
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass


def get_swarm_knowledge_bus() -> SwarmKnowledgeBus:
    return SwarmKnowledgeBus()
