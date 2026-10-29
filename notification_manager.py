"""
notification_manager.py
=======================
Zentraler Benachrichtigungs- & Entprellungs-Manager (Telegram & Discord).

Features:
1. Automatische Entprellung (Deduplication): Verhindert mehrfaches Senden derselben Meldung
   für dasselbe Asset innerhalb eines Zeitfensters (Standard: 60 Minuten).
2. Stündlicher Nachrichtendeckel (Rate-Limiting): Verhindert Nachrichtenflut (z.B. max 8 Nachrichten/Stunde).
3. Einstellbare Alert-Level:
   - "TRADES_ONLY": Nur echte Order-Ausführungen (Buy, Sell, Stop-Loss, TP). Keine Vetos, keine Evolutionen.
   - "BALANCED" (Standard): Echte Trades + echte Whale-Käufe + seltene Red-Team Vetos (max 1x pro Symbol/2h).
   - "ALL": Alle System-Ereignisse (mit Drosselung).
4. Automatische Test-Erkennung: Verhindert Spam bei Unit-Tests (test_audit.py).
5. Instanz-Identifikation (z.B. [GCP-Cloud] vs. [Local-PC]).
"""

from __future__ import annotations
import hashlib
import html
import json
import logging
import os
import ssl
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("autonomous_agent.notifications")

CACHE_DIR = Path("data/notification_cache")
CACHE_FILE = CACHE_DIR / "sent_alerts_history.json"


class TelegramNotificationManager:
    _instance: Optional[TelegramNotificationManager] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        alert_level: Optional[str] = None,
        max_per_hour: Optional[int] = None,
        dedup_window_sec: Optional[int] = None,
        instance_tag: Optional[str] = None,
    ):
        if getattr(self, "_initialized", False):
            return

        self.bot_token = (bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        self.alert_level = (alert_level or os.getenv("TELEGRAM_ALERT_LEVEL", "BALANCED")).strip().upper()
        self.max_per_hour = int(max_per_hour or os.getenv("TELEGRAM_MAX_ALERTS_PER_HOUR", "8"))
        self.dedup_window_sec = int(dedup_window_sec or os.getenv("TELEGRAM_DEDUP_WINDOW_SECONDS", "3600"))
        self.instance_tag = (instance_tag or os.getenv("TELEGRAM_INSTANCE_TAG", "🤖 Autonomous Agent")).strip()

        # In-Memory Ringpuffer für Zeitstempel der gesendeten Nachrichten (letzte 60 Minuten)
        self._sent_timestamps: deque[float] = deque()
        # Cache für dedup_key -> timestamp
        self._sent_hashes: Dict[str, float] = {}
        self._cap_warning_sent = False

        self._load_cache()
        self._initialized = True

    # =========================================================================
    # ÖFFENTLICHE SCHNITTSTELLEN
    # =========================================================================

    def send_trade_execution(
        self,
        symbol: str,
        action: str,
        qty: float,
        price: float,
        margin_usd: float,
        stop_loss: float,
        tp1: float,
        tp2: Optional[float] = None,
        reason: str = "",
        leverage: float = 1.0,
    ) -> bool:
        """
        Sendet eine Benachrichtigung über eine ECHTE Order-Ausführung.
        Höchste Priorität – wird immer gesendet (außer es ist ein exaktes Duplikat innerhalb von 30s).
        """
        dedup_key = f"TRADE:{symbol}:{action}:{price:.2f}"
        if self._is_duplicate(dedup_key, cooldown_seconds=60):
            logger.info("Unterdrücke doppelten Trade-Alert für %s %s", action, symbol)
            return False

        icon = "🟢" if "BUY" in action else ("🔴" if "SELL" in action or "SHORT" in action else "⚡")
        lev_str = f" ({leverage:.1f}x Hebel)" if leverage > 1.0 else ""
        tp2_str = f" | TP2: <b>${tp2:.2f}</b>" if tp2 else ""

        body = (
            f"{icon} <b>{self.instance_tag}: ORDER AUSGEFÜHRT!</b>\n"
            f"─────────────────────────────\n"
            f"• <b>Asset:</b> <code>{html.escape(symbol)}</code>\n"
            f"• <b>Aktion:</b> <b>{html.escape(action)}</b>{lev_str}\n"
            f"• <b>Menge / Kurs:</b> {qty:,.4f} Stk. @ <b>${price:,.2f}</b>\n"
            f"• <b>Margin / Einsatz:</b> ${margin_usd:,.2f}\n"
            f"• <b>Stop-Loss:</b> 🛡️ <b>${stop_loss:.2f}</b>\n"
            f"• <b>Gewinnziel TP1:</b> 🎯 <b>${tp1:.2f}</b>{tp2_str}\n"
        )
        if reason:
            body += f"\n📖 <i>Grund: {html.escape(reason[:120])}</i>"

        return self._send_raw_telegram(body, priority_bypass_cap=True, dedup_key=dedup_key)

    def send_trade_exit(
        self,
        symbol: str,
        exit_type: str,
        qty: float,
        exit_price: float,
        realized_pnl: float,
        reason: str = "",
    ) -> bool:
        """Sendet Take-Profit, Stop-Loss oder Time-Stop Schließungs-Alerts."""
        dedup_key = f"EXIT:{symbol}:{exit_type}:{exit_price:.2f}"
        if self._is_duplicate(dedup_key, cooldown_seconds=120):
            return False

        icon = "💰" if realized_pnl >= 0 else "🛑"
        pnl_color = "🟢" if realized_pnl >= 0 else "🔴"

        body = (
            f"{icon} <b>{self.instance_tag}: POSITION GESCHLOSSEN</b>\n"
            f"─────────────────────────────\n"
            f"• <b>Asset:</b> <code>{html.escape(symbol)}</code> ({html.escape(exit_type)})\n"
            f"• <b>Verkaufskurs:</b> ${exit_price:,.2f}\n"
            f"• <b>Realisierter PnL:</b> {pnl_color} <b>${realized_pnl:+,.2f}</b>\n"
        )
        if reason:
            body += f"• <b>Begründung:</b> {html.escape(reason[:120])}\n"

        return self._send_raw_telegram(body, priority_bypass_cap=True, dedup_key=dedup_key)

    def send_red_team_veto(
        self,
        symbol: str,
        risk_score: float,
        bear_rebuttal: str,
        cio_reasoning: str,
    ) -> bool:
        """
        Sendet Red-Team VETO Meldungen nur dann, wenn:
        1. alert_level != 'TRADES_ONLY'
        2. Dasselbe Symbol nicht bereits in den letzten 2 Stunden (7200s) vetoed gemeldet wurde.
        3. Das stündliche Rate-Limit nicht überschritten ist.
        """
        if self.alert_level == "TRADES_ONLY":
            logger.debug("Red-Team VETO für %s unterdrückt (TRADES_ONLY aktiv)", symbol)
            return False

        dedup_key = f"VETO:{symbol}"
        if self._is_duplicate(dedup_key, cooldown_seconds=7200):  # 2 Stunden Cooldown pro Symbol
            logger.info("VETO für %s innerhalb Cooldown (2h) unterdrückt", symbol)
            return False

        if not self._check_rate_limit():
            logger.warning("Stündliches Nachrichten-Limit erreicht. VETO für %s gedrosselt.", symbol)
            return False

        safe_sym = html.escape(symbol)
        safe_bear = html.escape(bear_rebuttal[:150])
        safe_cio = html.escape(cio_reasoning[:150])

        body = (
            f"🛡️ <b>{self.instance_tag}: Red-Team VETO</b>\n"
            f"─────────────────────────────\n"
            f"• <b>Asset:</b> <code>{safe_sym}</code>\n"
            f"• <b>Risiko-Score:</b> ⚠️ <b>{risk_score:.1f}/10</b>\n"
            f"• <b>Gegenwind:</b> {safe_bear}\n\n"
            f"⚖️ <b>CIO:</b> {safe_cio}\n\n"
            f"<i>Dein Kapital bleibt geschützt.</i>"
        )
        return self._send_raw_telegram(body, priority_bypass_cap=False, dedup_key=dedup_key)

    def send_whale_alert(self, symbol: str, summary: str, net_usd: float = 0.0) -> bool:
        """Sendet Insider-/Whale-Käufe nur ab hoher Relevanz und mit 4h Cooldown pro Symbol."""
        if self.alert_level == "TRADES_ONLY":
            return False

        dedup_key = f"WHALE:{symbol}"
        if self._is_duplicate(dedup_key, cooldown_seconds=14400):  # 4 Stunden Cooldown
            return False

        if not self._check_rate_limit():
            return False

        body = (
            f"🐋 <b>{self.instance_tag}: Whale & Insider-Kauf</b>\n"
            f"─────────────────────────────\n"
            f"• <b>Asset:</b> <code>{html.escape(symbol)}</code>\n"
            f"• <b>Umfang:</b> <b>${net_usd:,.0f}</b>\n"
            f"• <b>Details:</b> {html.escape(summary[:140])}\n"
        )
        return self._send_raw_telegram(body, priority_bypass_cap=False, dedup_key=dedup_key)

    def send_agent_spawn_alert(self, name: str, role: str, gen: int, trust: float) -> bool:
        """Agenten-Geburts-Alerts nur im 'ALL' Modus senden, sonst leise im Hintergrund arbeiten."""
        if self.alert_level != "ALL":
            return False

        dedup_key = f"SPAWN:{name}"
        if self._is_duplicate(dedup_key, cooldown_seconds=86400):  # 1x pro Tag
            return False

        if not self._check_rate_limit():
            return False

        body = (
            f"🧬 <b>{self.instance_tag}: Neuer KI-Spezialist erschaffen</b>\n"
            f"─────────────────────────────\n"
            f"• <b>Agent:</b> {html.escape(name)} (Gen #{gen})\n"
            f"• <b>Rolle:</b> {html.escape(role)}\n"
            f"• <b>Gewicht:</b> {trust:.2f}x\n"
        )
        return self._send_raw_telegram(body, priority_bypass_cap=False, dedup_key=dedup_key)

    def send_system_warning(self, title: str, text: str) -> bool:
        """Kritische System-Warnungen (z.B. Circuit Breaker, Margin Call)."""
        dedup_key = f"SYSTEM:{title}"
        if self._is_duplicate(dedup_key, cooldown_seconds=1800):
            return False

        body = (
            f"⚠️ <b>{self.instance_tag}: {html.escape(title)}</b>\n"
            f"─────────────────────────────\n"
            f"{html.escape(text)}\n"
        )
        return self._send_raw_telegram(body, priority_bypass_cap=True, dedup_key=dedup_key)

    # =========================================================================
    # INTERNE ENTPRELLUNG & VERSAND
    # =========================================================================

    def _is_duplicate(self, dedup_key: str, cooldown_seconds: int) -> bool:
        """Prüft, ob für den Schlüssel innerhalb des Cooldowns bereits gesendet wurde."""
        now = time.time()
        last_sent = self._sent_hashes.get(dedup_key, 0.0)
        if (now - last_sent) < cooldown_seconds:
            return True
        return False

    def _check_rate_limit(self) -> bool:
        """
        Prüft, ob in den letzten 3600 Sekunden weniger als max_per_hour
        Nachrichten gesendet wurden.
        """
        now = time.time()
        one_hour_ago = now - 3600.0

        # Bereinige ältere Zeitstempel
        while self._sent_timestamps and self._sent_timestamps[0] < one_hour_ago:
            self._sent_timestamps.popleft()

        if len(self._sent_timestamps) >= self.max_per_hour:
            if not self._cap_warning_sent:
                self._send_cap_warning()
                self._cap_warning_sent = True
            return False

        self._cap_warning_sent = False
        return True

    def _send_cap_warning(self) -> None:
        """Sendet einmalig eine Warnung, dass das Limit erreicht wurde."""
        body = (
            f"🔇 <b>{self.instance_tag}: Benachrichtigungs-Bremse aktiv</b>\n"
            f"─────────────────────────────\n"
            f"Das stündliche Limit von <b>{self.max_per_hour} Nachrichten</b> wurde erreicht. "
            f"Routine-Meldungen (Vetos, Whale-Scans) werden vorübergehend gedrosselt, "
            f"um dein Telegram ruhig zu halten.\n\n"
            f"<i>Echte Trade-Ausführungen und Stop-Losses werden weiterhin garantiert sofort übertragen!</i>"
        )
        self._send_raw_telegram(body, priority_bypass_cap=True)

    def _send_raw_telegram(
        self,
        html_text: str,
        priority_bypass_cap: bool = False,
        dedup_key: Optional[str] = None,
    ) -> bool:
        """Führt den eigentlichen HTTP-Aufruf an die Telegram Bot API aus."""
        now = time.time()

        # 1. Test-Schutz: Niemals während Unit-Tests echte Nachrichten über das Internet absenden!
        if self._is_running_in_test():
            logger.debug("Telegram-Versand im Testmodus abgefangen")
            self._sent_timestamps.append(now)
            if dedup_key:
                self._sent_hashes[dedup_key] = now
            return True

        if not self.bot_token or not self.chat_id:
            return False

        # 2. Rate-Limiting für Non-Priority
        if not priority_bypass_cap and not self._check_rate_limit():
            return False

        try:
            ctx = ssl.create_default_context()
        except Exception:
            ctx = ssl._create_unverified_context()

        cid = int(self.chat_id) if self.chat_id.isdigit() else self.chat_id
        payload = json.dumps({"chat_id": cid, "text": html_text, "parse_mode": "HTML"}).encode("utf-8")
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, context=ctx, timeout=4.5) as resp:
                if resp.status == 200:
                    self._sent_timestamps.append(now)
                    if dedup_key:
                        self._sent_hashes[dedup_key] = now
                    self._save_cache()
                    return True
        except Exception as e:
            logger.warning("Telegram Senden fehlgeschlagen: %s", e)

        return False

    def _is_running_in_test(self) -> bool:
        """Erkennt automatisch, ob der Code gerade innerhalb eines Tests läuft."""
        if os.getenv("TESTING") == "1":
            return True
        for mod in sys.modules:
            if "pytest" in mod or "unittest" in mod:
                return True
        if sys.argv and any("test" in arg.lower() for arg in sys.argv):
            return True
        return False

    def _load_cache(self) -> None:
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    now = time.time()
                    # Nur Einträge jünger als 24 Stunden laden
                    self._sent_hashes = {
                        k: v for k, v in data.get("sent_hashes", {}).items() if (now - v) < 86400
                    }
        except Exception:
            self._sent_hashes = {}

    def _save_cache(self) -> None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({"sent_hashes": self._sent_hashes}, f)
        except Exception:
            pass


# Singleton Helper-Funktion
def get_notifier() -> TelegramNotificationManager:
    return TelegramNotificationManager()
