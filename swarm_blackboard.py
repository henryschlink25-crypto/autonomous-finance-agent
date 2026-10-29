"""
swarm_blackboard.py - Blackboard-Architektur & A2A (Agent-to-Agent) Envelope Protocol
State-of-the-Art Multi-Agenten Kommunikations-Infrastruktur für den Trading-Schwarm.
Ermöglicht dezentrale, strukturierte Zusammenarbeit:
- Geteiltes Gedächtnis (The Blackboard) für Hypothesen, Marktfakten und Kritiken.
- A2A-Envelope-Standard: Strukturierte Nachrichtenpakete mit Sender, Absicht (Intent), Payload und Konfidenz.
- Stigmergische Koordination: Agenten reagieren autonom auf Einträge anderer Spezialisten auf dem Blackboard.
"""

import datetime
import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("SwarmBlackboard")


@dataclass
class A2AMessageEnvelope:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str = ""  # z.B. "ALPHA::NVDA", "MACRO_SHIFT", "RISK_VETO::AAPL"
    from_agent: str = ""
    to_agent: str = "ALL"  # "ALL" (Broadcast) oder spezifische agent_id
    intent: Literal["PROPOSE_TRADE", "CHALLENGE_CRITIQUE", "PROVIDE_EVIDENCE", "CONFIRM_CONSENSUS", "EMERGENCY_VETO"] = "PROPOSE_TRADE"
    symbol: Optional[str] = None
    confidence: float = 0.50
    payload: Dict[str, Any] = field(default_factory=dict)
    summary_de: str = ""
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BlackboardTopicThread:
    topic_id: str
    symbol: Optional[str]
    created_at: str
    updated_at: str
    status: Literal["OPEN_DISCUSSION", "CONSENSUS_REACHED", "VETOED_CLOSED", "EXPIRED"] = "OPEN_DISCUSSION"
    messages: List[A2AMessageEnvelope] = field(default_factory=list)
    consensus_score: float = 0.0
    leading_action: str = "HOLD"


class SwarmBlackboard:
    """
    Zentrales Blackboard für den autonomen Multi-Agenten-Handelsschwarm.
    Speichert laufende Debatten, Hypothesen und Fakten transparent auf der Festplatte.
    """
    _instance: Optional["SwarmBlackboard"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, storage_path: str = None):
        if getattr(self, "_initialized", False):
            return
        if storage_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            storage_path = os.path.join(base_dir, "data", "swarm_blackboard.json")
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.threads: Dict[str, BlackboardTopicThread] = {}
        self._load()
        self._initialized = True

    def _load(self) -> None:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for t_id, t_dict in data.get("threads", {}).items():
                    msgs = [A2AMessageEnvelope(**m) for m in t_dict.get("messages", [])]
                    self.threads[t_id] = BlackboardTopicThread(
                        topic_id=t_dict["topic_id"],
                        symbol=t_dict.get("symbol"),
                        created_at=t_dict["created_at"],
                        updated_at=t_dict["updated_at"],
                        status=t_dict.get("status", "OPEN_DISCUSSION"),
                        messages=msgs,
                        consensus_score=t_dict.get("consensus_score", 0.0),
                        leading_action=t_dict.get("leading_action", "HOLD"),
                    )
            except Exception as e:
                logger.debug(f"Fehler beim Laden des Blackboards: {e}")

    def _save(self) -> None:
        try:
            # Begrenze Threads auf die letzten 50 aktivsten
            cleaned = {}
            sorted_threads = sorted(self.threads.items(), key=lambda x: x[1].updated_at, reverse=True)[:50]
            for t_id, thread in sorted_threads:
                t_data = {
                    "topic_id": thread.topic_id,
                    "symbol": thread.symbol,
                    "created_at": thread.created_at,
                    "updated_at": thread.updated_at,
                    "status": thread.status,
                    "consensus_score": thread.consensus_score,
                    "leading_action": thread.leading_action,
                    "messages": [m.to_dict() for m in thread.messages[-15:]],  # Letzte 15 Nachrichten pro Thread
                }
                cleaned[t_id] = t_data

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({"updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()[:19], "threads": cleaned}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Fehler beim Speichern des Blackboards: {e}")

    def post_envelope(self, envelope: A2AMessageEnvelope) -> str:
        """Postet eine neue strukturierte A2A-Nachricht auf das Blackboard."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]
        topic = envelope.topic or f"GENERAL::{envelope.symbol or 'MARKET'}"

        if topic not in self.threads:
            self.threads[topic] = BlackboardTopicThread(
                topic_id=topic,
                symbol=envelope.symbol,
                created_at=now_iso,
                updated_at=now_iso,
                status="OPEN_DISCUSSION",
                messages=[envelope],
            )
        else:
            thread = self.threads[topic]
            thread.updated_at = now_iso
            thread.messages.append(envelope)

        self._update_thread_consensus(self.threads[topic])
        self._save()
        return envelope.message_id

    def _update_thread_consensus(self, thread: BlackboardTopicThread) -> None:
        """Berechnet die gewichtete Meinungsbildung auf dem Blackboard."""
        buy_weight = 0.0
        sell_weight = 0.0
        for m in thread.messages:
            if m.intent == "PROPOSE_TRADE":
                act = m.payload.get("action", "")
                conf = m.confidence
                if act == "BUY":
                    buy_weight += conf
                elif act == "SELL":
                    sell_weight += conf
            elif m.intent == "EMERGENCY_VETO":
                sell_weight += 2.5
                thread.status = "VETOED_CLOSED"
                thread.leading_action = "VETO"
                thread.consensus_score = -1.0
                return
            elif m.intent == "CONFIRM_CONSENSUS":
                buy_weight += m.confidence * 1.5

        net = buy_weight - sell_weight
        thread.consensus_score = round(max(-1.0, min(1.0, net / max(1.0, buy_weight + sell_weight))), 2)
        if thread.consensus_score >= 0.40:
            thread.leading_action = "BUY"
            thread.status = "CONSENSUS_REACHED"
        elif thread.consensus_score <= -0.40:
            thread.leading_action = "SELL"
            thread.status = "CONSENSUS_REACHED"
        else:
            thread.leading_action = "HOLD"
            thread.status = "OPEN_DISCUSSION"

    def get_topic_context(self, topic: str) -> Optional[BlackboardTopicThread]:
        """Gibt den aktuellen Diskussionsstand zu einem Topic zurück."""
        return self.threads.get(topic)

    def get_symbol_insights(self, symbol: str) -> List[A2AMessageEnvelope]:
        """Holt alle relevanten Fakten, Hypothesen und Voten für ein Symbol vom Blackboard."""
        insights: List[A2AMessageEnvelope] = []
        for thread in self.threads.values():
            if thread.symbol == symbol:
                insights.extend(thread.messages)
        return sorted(insights, key=lambda x: x.timestamp, reverse=True)


_blackboard_instance: Optional[SwarmBlackboard] = None


def get_swarm_blackboard() -> SwarmBlackboard:
    global _blackboard_instance
    if _blackboard_instance is None:
        _blackboard_instance = SwarmBlackboard()
    return _blackboard_instance
