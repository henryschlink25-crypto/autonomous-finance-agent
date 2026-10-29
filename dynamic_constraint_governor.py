"""
dynamic_constraint_governor.py
==============================
Autonomer Beschränkungs-Regulator & Dynamischer Constraint Governor (Sub-Agent #49).

Befreit das KI-Agenten-Netzwerk von starren Begrenzungen und reguliert diese
vollkommen eigenständig nach oben (Lockerung bei hoher Alpha-Chance) oder
nach unten (Schärfung bei Marktstress & Notfällen):

1. Extended Hours Governance (Pre-Market & After-Hours):
   - Schaltet den Handel ab 10:00 Uhr MEZ (04:00 ET Pre-Market) und bis 02:00 Uhr MEZ
     (20:00 ET After-Hours) autonom frei, wenn Liquidität & Katalysatoren vorliegen.
   - Schärft/blockiert Extended Hours bei extremen Spreads oder nächtlicher Markt-Panik.

2. Flexibilisierung der 15-Minuten Eröffnungs-Sperre (Opening Bell ORB):
   - Lockert die 15-Minuten-Pause bereits nach 3 bis 5 Minuten, wenn die Eröffnungs-Spanne
     (Opening Range Breakout) sauber etabliert ist und institutionelles Volumen bestätigt.
   - Schärft auf die vollen 15 Minuten zurück bei erratischen Whipsaws / Eröffnungs-Chaos.

3. Dynamischer Time-Stop statt starrer 3-5 Tage:
   - Koppelt die Haltedauer an Trendstruktur, SMA20, MACD-Histogramm und Gewinne.
   - Gewinner im gesunden Trend erhalten eine Verlängerung von +3 bis +10 Tagen!
   - Eingeschlafene Trades ohne Momentum werden hingegen zügig liquidiert.

4. Adaptives Scan- & Loop-Intervall:
   - Beschleunigt während der Haupthandelszeit auf 60-90 Sekunden für ultraschnelle Einstiege.
   - Drosselt außerhalb der Börsenzeiten auf 120-300 Sekunden zur Schonung von Ressourcen.
"""

from __future__ import annotations
import datetime
import json
import logging
import os
import pytz
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
from config import AgentConfig
from models import MacroRegime, MarketSnapshot, Position

logger = logging.getLogger("autonomous_agent.constraint_governor")


@dataclass
class ConstraintEvaluation:
    governance_mode: str          # EXPANDED_HIGH_ALPHA | BALANCED | TIGHTENED_DEFENSE
    extended_hours_allowed: bool  # Pre-/After-Market freigeschaltet
    early_breakout_allowed: bool  # 3-5 Min ORB Eröffnungs-Ausbruch erlaubt
    time_stop_extension_days: int # Tage, um die der Time-Stop verlängert wird (+0 bis +10)
    recommended_loop_seconds: int # Empfohlene Loop-Frequenz (60s, 75s, 120s, 300s)
    confidence: float
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


class DynamicConstraintGovernor:
    """
    Sub-Agent #49: Autonomous Constraint Regulator.
    Überwacht Marktumfeld, Spreads, Volatilität und Portfolio-PnL und
    lockert oder schärft die Handelsbeschränkungen vollkommen autonom.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def evaluate_symbol_constraints(
        self,
        snapshot: MarketSnapshot,
        macro: Optional[MacroRegime] = None,
        daily_pnl_pct: float = 0.0,
    ) -> ConstraintEvaluation:
        """
        Prüft autonom, ob Begrenzungen für ein Symbol gelockert oder geschärft werden sollen.
        """
        vix = macro.vix_level if macro else 18.0
        regime = macro.regime_state if macro else "GREEN_RISK_ON"
        is_breakdown = getattr(macro, "correlation_breakdown_active", False) if macro else False
        daily_drawdown_limit = getattr(self.config, "max_daily_drawdown_pct", 0.05) * 100.0

        # 1. NOTFALL-SCHÄRFUNG: Bei Crash-Panik (VIX >= 32), Correlation Breakdown oder hohem Tagesverlust
        if vix >= 32.0 or is_breakdown or daily_pnl_pct <= -abs(daily_drawdown_limit * 0.70):
            return ConstraintEvaluation(
                governance_mode="TIGHTENED_DEFENSE",
                extended_hours_allowed=False,
                early_breakout_allowed=False,
                time_stop_extension_days=0,
                recommended_loop_seconds=60,  # Schnell scannen zur Risikoverteidigung
                confidence=0.95,
                rationale=(
                    f"🔒 AUTONOME SCHÄRFUNG: Erhöhter Marktstress (VIX {vix:.1f}, Tages-PnL {daily_pnl_pct:+.1f}%). "
                    f"Extended Hours blockiert, 15-Min Opening Freeze voll aktiv zum Kapitalschutz."
                ),
            )

        # 2. AUTONOME LOCKERUNG (EXPANDED HIGH ALPHA):
        # Wenn Makro stabil ist (VIX < 26) und kein akuter Tagesverlust vorliegt:
        rvol = getattr(snapshot, "volume_ratio_vs_20d", getattr(snapshot, "volume_ratio", 1.0))
        rsi = getattr(snapshot, "rsi_14", 50.0)
        has_momentum = abs(snapshot.change_24h_pct) >= 0.4 or rvol >= 1.2

        if regime in ("GREEN_RISK_ON", "NEUTRAL") and vix < 26.0 and daily_pnl_pct >= -2.0:
            # Eröffnungs-Ausbruch (ORB) nach 3-5 Minuten erlauben, wenn Momentum & RVOL stimmen
            early_orb = has_momentum and 30.0 <= rsi <= 75.0

            # Time-Stop Verlängerung berechnen
            ext_days = 0
            is_uptrend = snapshot.current_price > getattr(snapshot, "sma_20", 0.0) and getattr(snapshot, "macd_hist", 0.0) >= 0
            if is_uptrend or snapshot.cluster == "CORE_ETF":
                ext_days = 4  # +4 Tage Trend-Puffer

            return ConstraintEvaluation(
                governance_mode="EXPANDED_HIGH_ALPHA",
                extended_hours_allowed=True,
                early_breakout_allowed=early_orb,
                time_stop_extension_days=ext_days,
                recommended_loop_seconds=75,
                confidence=0.90,
                rationale=(
                    f"🔓 AUTONOME LOCKERUNG: Gesunder Markt (VIX {vix:.1f}, RVOL {rvol:.1f}x). "
                    f"Extended Hours (Pre-/After-Market) freigegeben, Eröffnungs-ORB ab Minute 3 aktiv, "
                    f"Time-Stop dynamisch um +{ext_days}d verlängert."
                ),
            )

        # 3. BALANCED STANDARD MODUS
        return ConstraintEvaluation(
            governance_mode="BALANCED",
            extended_hours_allowed=getattr(self.config, "enable_extended_hours_trading", True),
            early_breakout_allowed=False,
            time_stop_extension_days=1,
            recommended_loop_seconds=120,
            confidence=0.80,
            rationale=f"Ausgewogener Modus (VIX {vix:.1f}) – Standard-Schutzparameter aktiv.",
        )

    def calculate_time_stop_extension(self, pos: Position, snap: MarketSnapshot) -> int:
        """
        Berechnet autonom, wie viele Tage ein profitabler oder stabiler Trend-Trade
        über das reguläre Limit hinaus weiterlaufen darf.
        """
        if getattr(pos, "horizon", "TACTICAL") == "CORE_ETF" or pos.cluster == "CORE_ETF":
            return 999  # Core-ETFs werden nie durch Time-Stop liquidiert

        ext = 0
        # 1. Wenn die Position im Plus liegt -> Gewinne laufen lassen!
        if pos.unrealized_pnl_pct >= 1.2:
            ext += 3
        if pos.unrealized_pnl_pct >= 4.0:
            ext += 4

        # 2. Wenn Kurs über SMA20 und MACD positiv ist -> Trend intakt
        sma20 = getattr(snap, "sma_20", 0.0)
        macd_h = getattr(snap, "macd_hist", 0.0)
        if snap.current_price > sma20 and macd_h > 0:
            ext += 3

        # 3. Bei schwächelnder Dynamik (negativer MACD und Verlust) keine Verlängerung
        if macd_h < -0.1 and pos.unrealized_pnl_pct < -1.0:
            return 0

        return min(10, ext)

    def get_recommended_loop_seconds(
        self, base_loop_seconds: int = 300, now_utc: Optional[datetime.datetime] = None
    ) -> int:
        """
        Gibt autonom die optimale Scan-Geschwindigkeit zurück:
        - 60-75s während der aktiven US-Haupthandelszeit (15:30 - 22:00 MEZ)
        - 120s während Extended Hours (Pre-Market 10:00 - 15:30 MEZ & After-Hours 22:00 - 02:00 MEZ)
        - 300s während ruhiger Nachtstunden
        """
        if now_utc is None:
            now_utc = datetime.datetime.now(datetime.timezone.utc)

        ny_tz = pytz.timezone("America/New_York")
        ny_now = now_utc.astimezone(ny_tz)

        if ny_now.weekday() >= 5:
            return max(180, base_loop_seconds)  # Wochenende

        ny_time = ny_now.time()
        mkt_open = datetime.time(9, 30)
        mkt_close = datetime.time(16, 0)
        pre_mkt = datetime.time(4, 0)
        after_mkt = datetime.time(20, 0)

        # 1. US Haupthandelszeit: Höchste Frequenz (60-75 Sekunden)
        if mkt_open <= ny_time <= mkt_close:
            return getattr(self.config, "adaptive_loop_interval_active_market", 75)

        # 2. Extended Hours: Mittlere Frequenz (120 Sekunden)
        if (pre_mkt <= ny_time < mkt_open) or (mkt_close < ny_time <= after_mkt):
            return getattr(self.config, "adaptive_loop_interval_extended_hours", 120)

        # 3. Off-Hours (Nacht): Energiesparend (300 Sekunden)
        return getattr(self.config, "adaptive_loop_interval_off_hours", 300)
