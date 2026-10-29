"""
sec_edgar_scout.py - SEC Edgar Filing & Catalyst Scout
Sub-Agent #40 im ADAS-Trading-Schwarm.
Scannt SEC EDGAR Einreichungen (Form 4 Insider & Whale Buys, Form 8-K Unscheduled Material Events).
Postet strukturierte Evidenz-Pakete (A2AMessageEnvelope) direkt auf das Swarm-Blackboard.
"""

import datetime
import json
import logging
import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from sec_form4_tracker import SECForm4Tracker
from swarm_blackboard import get_swarm_blackboard, A2AMessageEnvelope

logger = logging.getLogger("SECEdgarScout")


class SECCatalystReport(BaseModel):
    symbol: str
    catalyst_type: str = "NEUTRAL"       # "INSIDER_CLUSTER_BUY", "WHALE_BUY", "FORM_8K_CATALYST", "DILUTION_WARNING", "NEUTRAL"
    sentiment: str = "NEUTRAL"           # "BULLISH", "BEARISH", "NEUTRAL"
    catalyst_score: float = 0.0          # -1.0 bis +1.0
    confidence: float = 0.50
    has_active_catalyst: bool = False
    net_insider_usd_90d: float = 0.0
    filing_summary: str = ""
    posted_to_blackboard: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])


class SECEdgarScout:
    """
    Scout-Agent zur Erkennung aufsichtsrechtlicher und insiderrelevanter Impulse aus SEC EDGAR.
    """
    _instance: Optional["SECEdgarScout"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.tracker = SECForm4Tracker()
        self.blackboard = get_swarm_blackboard()
        self._initialized = True

    def scan_symbol_filings(
        self,
        symbol: str,
        sec_form8k_text: Optional[str] = None,
        override_form4_data: Optional[Dict[str, Any]] = None,
        post_to_blackboard: bool = True,
    ) -> SECCatalystReport:
        """
        Scannt Form 4 und Form 8-K Meldungen und postet bei Relevanz auf das Swarm-Blackboard.
        """
        clean_sym = symbol.upper().replace(".DE", "")
        if "/" in symbol:
            return SECCatalystReport(
                symbol=symbol,
                catalyst_type="NOT_APPLICABLE",
                sentiment="NEUTRAL",
                catalyst_score=0.0,
                confidence=0.50,
                has_active_catalyst=False,
                filing_summary="Krypto: Nicht SEC-EDGAR meldepflichtig",
            )

        # 1. Form 4 Daten abrufen
        if override_form4_data:
            form4_info = override_form4_data
        else:
            try:
                form4_info = self.tracker.get_insider_activity(clean_sym)
            except Exception as e:
                form4_info = {"insider_sentiment": "NEUTRAL", "net_usd_90d": 0.0, "recent_whale_buys": [], "cluster_buy_detected": False}

        cluster = form4_info.get("cluster_buy_detected", False)
        whales = form4_info.get("recent_whale_buys", [])
        net_usd = float(form4_info.get("net_usd_90d", 0.0))
        insider_sent = form4_info.get("insider_sentiment", "NEUTRAL")

        # 2. Form 8-K Analyse
        has_8k = bool(sec_form8k_text and len(sec_form8k_text.strip()) > 5)
        text_8k_lower = (sec_form8k_text or "").lower()

        # Heuristiken
        is_dilution = any(k in text_8k_lower for k in ["offering", "dilution", "unregistered sales", "convertible note default"])
        is_positive_8k = any(k in text_8k_lower for k in ["fda approval", "merger agreement", "patent granted", "commercial contract", "earnings beat", "partnership with"])

        catalyst_type = "NEUTRAL"
        sentiment = "NEUTRAL"
        score = 0.0
        conf = 0.50
        summary_parts = []

        if is_dilution:
            catalyst_type = "DILUTION_WARNING"
            sentiment = "BEARISH"
            score = -0.75
            conf = 0.90
            summary_parts.append(f"SEC 8-K Verwässerungswarnung: {sec_form8k_text}")
        elif cluster:
            catalyst_type = "INSIDER_CLUSTER_BUY"
            sentiment = "BULLISH"
            score = 0.88
            conf = 0.92
            summary_parts.append(f"SEC Form 4 Cluster-Kauf: Mehrere Vorstände kauften im Wert von ${net_usd:,.0f}")
        elif len(whales) > 0 and net_usd >= 50000:
            catalyst_type = "WHALE_BUY"
            sentiment = "BULLISH"
            score = 0.80
            conf = 0.86
            summary_parts.append(f"SEC Form 4 Whale-Kauf: Netto ${net_usd:,.0f} Zukäufe")
        elif is_positive_8k:
            catalyst_type = "FORM_8K_CATALYST"
            sentiment = "BULLISH"
            score = 0.75
            conf = 0.84
            summary_parts.append(f"SEC 8-K Kurstreiber: {sec_form8k_text}")
        elif insider_sent == "BULLISH":
            catalyst_type = "INSIDER_NET_BUY"
            sentiment = "BULLISH"
            score = 0.65
            conf = 0.75
            summary_parts.append(f"SEC Form 4 Netto-Zukauf: ${net_usd:,.0f}")
        elif insider_sent == "BEARISH":
            catalyst_type = "INSIDER_SELLING"
            sentiment = "BEARISH"
            score = -0.60
            conf = 0.72
            summary_parts.append(f"SEC Form 4 Insider-Verkäufe: ${abs(net_usd):,.0f}")
        else:
            summary_parts.append("Keine signifikanten neuen SEC EDGAR Impulse")

        full_summary = " · ".join(summary_parts)
        has_active = catalyst_type != "NEUTRAL"

        # 3. Bei signifikantem Katalysator -> Aufs Blackboard posten!
        posted = False
        if post_to_blackboard and has_active:
            try:
                intent = "EMERGENCY_VETO" if is_dilution else "PROVIDE_EVIDENCE"
                envelope = A2AMessageEnvelope(
                    topic=f"ALPHA::{clean_sym}",
                    from_agent="sec_edgar_scout",
                    intent=intent,
                    symbol=clean_sym,
                    confidence=conf,
                    payload={
                        "catalyst_type": catalyst_type,
                        "sentiment": sentiment,
                        "score": score,
                        "net_usd": net_usd,
                        "detail": full_summary,
                    },
                    summary_de=f"📑 SEC EDGAR Katalysator: {full_summary}",
                )
                self.blackboard.post_envelope(envelope)
                posted = True
            except Exception as e:
                logger.debug(f"Konnte SEC Envelope nicht aufs Blackboard posten: {e}")

        return SECCatalystReport(
            symbol=clean_sym,
            catalyst_type=catalyst_type,
            sentiment=sentiment,
            catalyst_score=score,
            confidence=conf,
            has_active_catalyst=has_active,
            net_insider_usd_90d=net_usd,
            filing_summary=full_summary,
            posted_to_blackboard=posted,
        )


_sec_edgar_scout_instance: Optional[SECEdgarScout] = None


def get_sec_edgar_scout() -> SECEdgarScout:
    global _sec_edgar_scout_instance
    if _sec_edgar_scout_instance is None:
        _sec_edgar_scout_instance = SECEdgarScout()
    return _sec_edgar_scout_instance
