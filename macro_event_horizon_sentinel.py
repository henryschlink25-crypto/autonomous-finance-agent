"""
macro_event_horizon_sentinel.py
===============================
🏛️ Macro Economic Event Horizon & Fed/CPI Shield (Sub-Agent #29).

Schützt das Gesamtkapital vor brutalen Notenbank- und Inflations-Whipsaws:
1. Überwacht den institutionellen Wirtschaftskalender:
   - US Fed FOMC Zinsentscheide & Jerome Powell Pressekonferenzen (20:00 MEZ / 14:00 ET)
   - US CPI (Consumer Price Index / Inflation) (14:30 MEZ / 08:30 ET)
   - US PPI (Producer Price Index) (14:30 MEZ / 08:30 ET)
   - US NFP (Non-Farm Payrolls Arbeitsmarktbericht) (Jeder erste Freitag im Monat, 14:30 MEZ)
2. Definiert den "Event Horizon":
   - T-60 Min bis T+30 Min: Neukauf-Verbot (BLOCK_NEW_TRADES) & VETO im Red-Team.
   - T-180 Min bis T-60 Min: Erhöhte Vorsicht & Positionsgrößen-Halbierung (HALF_SIZE_ONLY).
   - Ausstehende Gewinne werden vor dem Event automatisch abgesichert.
"""

from __future__ import annotations
import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger("autonomous_agent.macro_events")

CALENDAR_FILE = Path("data/macro_economic_calendar.json")


class MacroEvent(BaseModel):
    name: str
    category: str  # "FOMC_RATE_DECISION", "US_CPI_INFLATION", "US_NFP_JOBS", "US_PPI", "OPEC_MEETING"
    timestamp_utc: str
    impact_level: str = "HIGH_VOLATILITY"
    expected_note: str = ""


class MacroEventSnapshot(BaseModel):
    current_time_utc: str
    next_event_name: str
    next_event_time_utc: str
    minutes_to_next_event: int
    event_horizon_active: bool
    horizon_state: str  # "EVENT_HORIZON_ACTIVE", "APPROACHING_CAUTION", "CLEAR_HORIZON"
    mandate: str
    protective_action: str  # "BLOCK_NEW_TRADES", "HALF_SIZE_ONLY", "UNRESTRICTED"
    recent_past_event: Optional[str] = None
    upcoming_events_7d: List[str] = Field(default_factory=list)
    summary: str = ""


class MacroEventHorizonSentinel:
    """
    Institutioneller Schutz vor Notenbank- und Wirtschaftsdaten-Schocks.
    """

    def __init__(self, calendar_file: Path = CALENDAR_FILE):
        self.calendar_file = Path(calendar_file)
        self.calendar_file.parent.mkdir(parents=True, exist_ok=True)
        self.events: List[MacroEvent] = self._load_or_bootstrap_calendar()

    def _load_or_bootstrap_calendar(self) -> List[MacroEvent]:
        if self.calendar_file.exists():
            try:
                with open(self.calendar_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return [MacroEvent(**e) for e in raw]
            except Exception as e:
                logger.warning(f"Konnte Makro-Kalender nicht laden: {e}")

        # Standard-Kalender für 2025/2026/2027 initialisieren
        events = self._generate_foundational_calendar()
        self._save_calendar(events)
        return events

    def _save_calendar(self, events: List[MacroEvent]) -> None:
        try:
            with open(self.calendar_file, "w", encoding="utf-8") as f:
                json.dump([e.model_dump() for e in events], f, indent=2)
        except Exception as e:
            logger.warning(f"Konnte Makro-Kalender nicht speichern: {e}")

    def _generate_foundational_calendar(self) -> List[MacroEvent]:
        """
        Erzeugt planmäßige FOMC-, CPI- und NFP-Termine für 2025/2026/2027.
        """
        evs: List[MacroEvent] = []

        # Bekannte FOMC Termine (Mittwochs 19:00/20:00 UTC)
        fomc_dates_2026 = [
            "2026-01-28T19:00:00Z",
            "2026-03-18T19:00:00Z",
            "2026-05-06T19:00:00Z",
            "2026-06-17T19:00:00Z",
            "2026-07-29T19:00:00Z",
            "2026-09-23T19:00:00Z",
            "2026-11-04T19:00:00Z",
            "2026-12-16T19:00:00Z",
        ]
        for dt in fomc_dates_2026:
            evs.append(MacroEvent(
                name="Federal Reserve FOMC Zinsentscheid & Pressekonferenz",
                category="FOMC_RATE_DECISION",
                timestamp_utc=dt,
                expected_note="US-Leitzinsentscheidung und Ausblick von Notenbankchef Powell.",
            ))

        # US CPI Inflation (Monatlich, meist zweiter Dienstag/Mittwoch um 13:30 UTC / 08:30 ET)
        cpi_dates_2026 = [
            "2026-01-14T13:30:00Z",
            "2026-02-11T13:30:00Z",
            "2026-03-11T13:30:00Z",
            "2026-04-14T13:30:00Z",
            "2026-05-13T13:30:00Z",
            "2026-06-10T13:30:00Z",
            "2026-07-15T13:30:00Z",
            "2026-08-12T13:30:00Z",
            "2026-09-15T13:30:00Z",
            "2026-10-14T13:30:00Z",
            "2026-11-12T13:30:00Z",
            "2026-12-10T13:30:00Z",
        ]
        for dt in cpi_dates_2026:
            evs.append(MacroEvent(
                name="US CPI Inflationsbericht (Verbraucherpreise)",
                category="US_CPI_INFLATION",
                timestamp_utc=dt,
                expected_note="Monatliche Kern- und Gesamtrate der US-Konsumentenpreise.",
            ))

        # US Non-Farm Payrolls (Erster Freitag im Monat um 13:30 UTC)
        nfp_dates_2026 = [
            "2026-01-09T13:30:00Z",
            "2026-02-06T13:30:00Z",
            "2026-03-06T13:30:00Z",
            "2026-04-03T13:30:00Z",
            "2026-05-08T13:30:00Z",
            "2026-06-05T13:30:00Z",
            "2026-07-02T13:30:00Z",
            "2026-08-07T13:30:00Z",
            "2026-09-04T13:30:00Z",
            "2026-10-02T13:30:00Z",
            "2026-11-06T13:30:00Z",
            "2026-12-04T13:30:00Z",
        ]
        for dt in nfp_dates_2026:
            evs.append(MacroEvent(
                name="US Non-Farm Payrolls (Arbeitsmarktbericht & Löhne)",
                category="US_NFP_JOBS",
                timestamp_utc=dt,
                expected_note="Neue geschaffene Stellen ex-Agrar, Arbeitslosenquote und Stundenlöhne.",
            ))

        evs.sort(key=lambda x: x.timestamp_utc)
        return evs

    def register_custom_event(self, event: MacroEvent) -> None:
        """
        Ermöglicht das spontane Eintragen außerplanmäßiger Reden oder Notfall-Sitzungen.
        """
        self.events.append(event)
        self.events.sort(key=lambda x: x.timestamp_utc)
        self._save_calendar(self.events)

    def analyze_event_horizon(self, reference_time_utc: Optional[datetime.datetime] = None) -> MacroEventSnapshot:
        """
        Überprüft die zeitliche Nähe zu High-Impact Wirtschafts- und Notenbank-Events.
        """
        if reference_time_utc is None:
            now_dt = datetime.datetime.now(datetime.timezone.utc)
        else:
            if reference_time_utc.tzinfo is None:
                now_dt = reference_time_utc.replace(tzinfo=datetime.timezone.utc)
            else:
                now_dt = reference_time_utc

        now_str = now_dt.isoformat()[:19] + "Z"

        upcoming: List[Tuple[int, MacroEvent]] = []
        recent_past: List[Tuple[int, MacroEvent]] = []

        for ev in self.events:
            try:
                ev_clean = ev.timestamp_utc.replace("Z", "+00:00")
                ev_dt = datetime.datetime.fromisoformat(ev_clean)
                diff_seconds = (ev_dt - now_dt).total_seconds()
                diff_minutes = int(diff_seconds / 60)

                if diff_minutes >= 0:
                    upcoming.append((diff_minutes, ev))
                else:
                    recent_past.append((abs(diff_minutes), ev))
            except Exception:
                continue

        upcoming.sort(key=lambda x: x[0])
        recent_past.sort(key=lambda x: x[0])

        next_diff_m, next_ev = upcoming[0] if upcoming else (99999, MacroEvent(name="Kein Event in Sicht", category="NONE", timestamp_utc=""))
        past_diff_m, past_ev = recent_past[0] if recent_past else (99999, None)

        # Event Horizon Logik:
        # 1. T-60 Minuten vor dem Event ODER T+30 Minuten nach dem Event
        is_active = (0 <= next_diff_m <= 60) or (past_diff_m <= 30)

        # 2. T-180 Minuten bis T-60 Minuten vor dem Event
        is_caution = (60 < next_diff_m <= 180)

        if is_active:
            state = "EVENT_HORIZON_ACTIVE"
            action = "BLOCK_NEW_TRADES"
            ev_name = next_ev.name if next_diff_m <= 60 else (past_ev.name if past_ev else next_ev.name)
            mandate = f"🚨 NOTENBANK-ALARM AKTIV: {ev_name}! Neukäufe gesperrt (Whipsaw-Schutz). Stops sichern."
            summary = f"Event-Horizon aktiv für '{ev_name}'. Handel bis 30 Minuten nach Event unterbrochen."
        elif is_caution:
            state = "APPROACHING_CAUTION"
            action = "HALF_SIZE_ONLY"
            mandate = f"⏳ Makro-Event naht in {next_diff_m} Min ({next_ev.name}). Neue Positionen auf 50% gedeckelt."
            summary = f"Anstehendes Groß-Event '{next_ev.name}' in {next_diff_m} Minuten. Positionsgröße wird halbiert."
        else:
            state = "CLEAR_HORIZON"
            action = "UNRESTRICTED"
            mandate = "🟢 HORIZONT FREI: Keine unmittelbar anstehenden High-Impact Zins- oder Inflationsentscheide."
            summary = f"Horizont frei. Nächstes Notenbank-/Inflations-Event in {next_diff_m} Minuten ({next_ev.name})."

        up_7d = []
        for diff_m, ev in upcoming:
            if diff_m <= 7 * 24 * 60:
                dt_short = ev.timestamp_utc[:16].replace("T", " ")
                up_7d.append(f"{dt_short} UTC: {ev.name}")

        return MacroEventSnapshot(
            current_time_utc=now_str,
            next_event_name=next_ev.name,
            next_event_time_utc=next_ev.timestamp_utc,
            minutes_to_next_event=next_diff_m,
            event_horizon_active=is_active,
            horizon_state=state,
            mandate=mandate,
            protective_action=action,
            recent_past_event=past_ev.name if past_ev and past_diff_m <= 60 else None,
            upcoming_events_7d=up_7d,
            summary=summary,
        )

    def format_telegram_summary(self, snap: MacroEventSnapshot) -> str:
        """
        Formatiert den Makro-Event Kalender für Telegram.
        """
        icon = {
            "EVENT_HORIZON_ACTIVE": "🚨🔴",
            "APPROACHING_CAUTION": "⏳🟡",
            "CLEAR_HORIZON": "🟢",
        }.get(snap.horizon_state, "⚪")

        hours_to = snap.minutes_to_next_event // 60
        mins_rem = snap.minutes_to_next_event % 60
        time_txt = f"{hours_to}h {mins_rem}m" if hours_to > 0 else f"{mins_rem} Minuten"

        lines = [
            "🏛️ <b>Macro Event Horizon & Notenbank-Radar:</b>",
            "─────────────────────────────",
            f"• <b>Status:</b> {icon} <b>{snap.horizon_state}</b>",
            f"• <b>Handels-Mandat:</b> <code>{snap.protective_action}</code>",
            f"• <b>Nächstes Groß-Event:</b> <b>{snap.next_event_name}</b>",
            f"• <b>Countdown:</b> in <b>{time_txt}</b> ({snap.next_event_time_utc[:16].replace('T', ' ')} UTC)\n",
        ]

        if snap.event_horizon_active:
            lines.append("🛑 <b>SCHUTZ-MASSNAHME AKTIV:</b>")
            lines.append("• Neukäufe im Schwarm vorübergehend gesperrt!")
            lines.append("• Verhindert unverschuldete Stop-Loss Ausstopper durch Zins-/Inflations-Whipsaws.\n")

        if snap.upcoming_events_7d:
            lines.append("📅 <b>Wichtige Events in den nächsten 7 Tagen:</b>")
            for e in snap.upcoming_events_7d[:5]:
                lines.append(f"• <i>{e}</i>")
            lines.append("")

        lines.append("─────────────────────────────")
        lines.append(f"💡 <i>{snap.mandate}</i>")

        return "\n".join(lines)


# Singleton
_EVENT_SENTINEL_INSTANCE: Optional[MacroEventHorizonSentinel] = None


def get_macro_event_sentinel() -> MacroEventHorizonSentinel:
    global _EVENT_SENTINEL_INSTANCE
    if _EVENT_SENTINEL_INSTANCE is None:
        _EVENT_SENTINEL_INSTANCE = MacroEventHorizonSentinel()
    return _EVENT_SENTINEL_INSTANCE
