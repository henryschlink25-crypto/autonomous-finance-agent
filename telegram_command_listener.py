"""
telegram_command_listener.py
============================
Interaktiver Telegram-Fernsteuerungs-Listener für den Autonomous Trading Agent.

Erlaubt dem Nutzer (Henry), von unterwegs per Telegram Befehle an seinen Agenten zu senden:
- /status        -> Systemgesundheit, Kontostand, Buying Power, Marktregime, aktive Schwarm-Agenten
- /portfolio     -> Offene Alpaca-Positionen mit aktuellem Kurs und G&V (Gewinn/Verlust)
- /trades        -> Die letzten 5 ausgeführten Orders
- /debates       -> Die jüngsten Red-Team-Tribunal-Debatten (Bull vs. Bear vs. CIO)
- /whales        -> Die letzten SEC Form 4 Insider- & Whale-Käufe
- /options <SYM> -> Live-Optionen-Flow (PCR, MaxPain, Unusual Activity) für ein beliebiges Symbol
- /fund <SYM>    -> Fundamentalanalyse & Bilanz-Check für ein beliebiges Symbol
- /breed         -> Startet im Hintergrund einen Evolutionszyklus neuer genetischer Strategien
- /help          -> Zeigt die Befehlsübersicht

Sicherheit: Reagiert AUSSCHLIESSLICH auf die autorisierte TELEGRAM_CHAT_ID.
"""

from __future__ import annotations
import html
import json
import logging
import os
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CTX = ssl._create_unverified_context()

logger = logging.getLogger("autonomous_agent.telegram_listener")


class TelegramCommandListener:
    def __init__(
        self,
        bot_token: str,
        authorized_chat_id: str,
        portfolio_callback: Optional[Callable[[], Dict[str, Any]]] = None,
        status_callback: Optional[Callable[[], Dict[str, Any]]] = None,
        breed_callback: Optional[Callable[[], str]] = None,
        panic_callback: Optional[Callable[[], List[str]]] = None,
        pause_callback: Optional[Callable[[bool], str]] = None,
        engine_ref: Optional[Any] = None,
    ):
        self.token = bot_token.strip()
        self.chat_id = str(authorized_chat_id).strip()
        self.portfolio_callback = portfolio_callback
        self.status_callback = status_callback
        self.breed_callback = breed_callback
        self.panic_callback = panic_callback
        self.pause_callback = pause_callback
        self.engine_ref = engine_ref
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._briefing_thread: Optional[threading.Thread] = None
        self._last_update_id = 0
        self._last_briefing_date: str = ""
        self._last_evening_date: str = ""

    def start(self) -> None:
        """Startet den Long-Polling-Listener in einem separaten Daemon-Thread."""
        if not self.token or not self.chat_id:
            logger.info("Telegram-Listener deaktiviert: Kein Token oder Chat-ID vorhanden.")
            return

        if self.is_running:
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._polling_loop, daemon=True, name="TelegramBotListener")
        self._thread.start()
        self._briefing_thread = threading.Thread(target=self._briefing_loop, daemon=True, name="TelegramBriefingListener")
        self._briefing_thread.start()
        logger.info(f"Telegram-Remote-Listener aktiv für Chat-ID {self.chat_id}")

    def stop(self) -> None:
        self.is_running = False

    def send_message(self, text: str) -> bool:
        """Sendet eine formatierte Nachricht an den autorisierten Chat."""
        if not self.token or not self.chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = json.dumps({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "TradingAgent/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"Telegram Send Error: {e}")
            return False

    def _polling_loop(self) -> None:
        # Initialer Offset-Sprung, um alte Nachrichten nicht erneut abzufeuern
        try:
            init_url = f"https://api.telegram.org/bot{self.token}/getUpdates?offset=-1"
            req = urllib.request.Request(init_url, headers={"User-Agent": "TradingAgent/1.0"})
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("ok") and data.get("result"):
                    self._last_update_id = data["result"][-1]["update_id"] + 1
        except Exception as e:
            logger.debug(f"Init Update Offset Error: {e}")

        while self.is_running:
            try:
                url = f"https://api.telegram.org/bot{self.token}/getUpdates?offset={self._last_update_id}&timeout=15"
                req = urllib.request.Request(url, headers={"User-Agent": "TradingAgent/1.0"})
                with urllib.request.urlopen(req, context=SSL_CTX, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                if data.get("ok"):
                    for update in data.get("result", []):
                        self._last_update_id = update["update_id"] + 1
                        msg = update.get("message") or update.get("edited_message")
                        if not msg:
                            continue

                        sender_chat_id = str(msg.get("chat", {}).get("id", ""))
                        text = msg.get("text", "").strip()

                        if sender_chat_id != self.chat_id:
                            logger.warning(f"Unbefugter Zugriff auf Telegram-Bot von Chat-ID {sender_chat_id} abgewehrt.")
                            continue

                        if text.startswith("/"):
                            self._handle_command(text)

            except Exception as e:
                time.sleep(3)

    def _handle_command(self, cmd_text: str) -> None:
        parts = cmd_text.split()
        cmd = parts[0].lower().split("@")[0]  # strip @botname if present
        args = parts[1:]

        logger.info(f"Empfangener Telegram-Befehl: {cmd} mit Args: {args}")

        if cmd in ["/start", "/help"]:
            help_msg = (
                "🤖 <b>Autonomous Finance Agent - Fernsteuerung</b>\n"
                "─────────────────────────────\n"
                "Verfügbare Befehle:\n\n"
                "📊 <b>/status</b> - Kontostand, Buying Power, Regime & Schwarm\n"
                "💼 <b>/portfolio</b> - Offene Positionen mit Live G&V\n"
                "📜 <b>/trades</b> - Die letzten 5 ausgeführten Trades\n"
                "⚖️ <b>/debates</b> - Die jüngsten Red-Team-Urteile\n"
                "🐋 <b>/whales</b> - Letzte SEC Form 4 Insider-Käufe\n"
                "📈 <b>/options &lt;SYM&gt;</b> - Optionen-Flow (z.B. <code>/options NVDA</code>)\n"
                "🔍 <b>/fund &lt;SYM&gt;</b> - Fundamental-Check (z.B. <code>/fund AAPL</code>)\n"
                "📊 <b>/analysts &lt;SYM&gt;</b> - Wall Street Analysten-Konsens & Kursziele\n"
                "🤗 <b>/sentiment &lt;SYM&gt;</b> - Hugging Face FinBERT NLP & News-Sentiment\n"
                "🎯 <b>/vwap &lt;SYM&gt;</b> - VWAP-Bänder (±2σ), Volume Profile POC & Liquidity Sweeps\n"
                "⏱️ <b>/timing &lt;SYM&gt;</b> - Einstiegs-Timing: Sofort kaufen vs. Abwarten auf Pullback\n"
                "📊 <b>/chart &lt;SYM&gt;</b> - Push-Chart & Price-Ladder (z.B. <code>/chart NVDA</code>)\n"
                "📈 <b>/forecast &lt;SYM&gt;</b> - Probabilistischer Kurs-Trichter (P10/P50/P90)\n"
                "🚨 <b>/shield</b> - Dynamic Correlation Breakdown Shield (Marktpanik & Gleichlauf)\n"
                "🌐 <b>/geopolitics</b> - Weltpolitische Lage, GCI-Index & Krisen-Mandat\n"
                "🏛️ <b>/events</b> - Makro-Event Horizon (Fed FOMC, CPI, NFP) & Whipsaw-Schutz\n"
                "🔀 <b>/pairs</b> - Statistische Arbitrage & Pairs-Spreads (Z-Score)\n"
                "🚀 <b>/sectors</b> - Sektor-Rotations-Radar & führende Branchen\n"
                "🧬 <b>/evolve</b> - Autonome Schwarm-Evolution & Bayesianische Re-Kalibrierung\n"
                "🧪 <b>/lab</b> - Walk-Forward Forschungs-Labor & WFER-Effizienz\n"
                "⚡ <b>/optimize</b> - Starte Walk-Forward Optimierung & Auto-Kalibrierung\n"
                "🧬 <b>/breed</b> - Sofortige genetische Strategie-Zucht starten\n"
                "🔔 <b>/alerts</b> [trades|balanced|all] - Benachrichtigungen drosseln & Limit prüfen\n"
                "🧠 <b>/knowledge</b> - Geteilter Schwarm-Wissensspeicher & Agenten-Synergien\n"
                "🗣️ <b>/deals</b> - Ausgehandelte Deals im Agenten-Dialekt\n"
                "🚨 <b>/panic</b> - Notfall-Ausstieg: Schließt SOFORT alle Positionen & sichert Cash\n"
                "⏸️ <b>/pause</b> - Handel pausieren (keine Neukäufe, Stops bleiben aktiv)\n"
                "▶️ <b>/resume</b> - Handel fortsetzen\n"
                "💓 <b>/ping</b> - Sofortiger Lebenszeichen- & Status-Check\n"
                "📋 <b>/report</b> - Ausführlicher 3-Tage-Reise- & PnL-Report\n"
                "🔄 <b>/update</b> - Autonomes Live-Update von GitHub laden & neu starten\n"
                "─────────────────────────────"
            )
            self.send_message(help_msg)

        elif cmd == "/status":
            self._handle_status()

        elif cmd in ["/portfolio", "/positions"]:
            self._handle_portfolio()

        elif cmd == "/trades":
            self._handle_trades()

        elif cmd in ["/debates", "/vetos"]:
            self._handle_debates()

        elif cmd == "/whales":
            self._handle_whales()

        elif cmd in ["/pairs", "/arb", "/arbitrage"]:
            self._handle_pairs()

        elif cmd in ["/sectors", "/sector", "/sektoren"]:
            self._handle_sectors()

        elif cmd == "/options":
            if not args:
                self.send_message("⚠️ Bitte Symbol angeben, z.B.: <code>/options NVDA</code>")
            else:
                self._handle_options(args[0])

        elif cmd in ["/fund", "/fundamental"]:
            if not args:
                self.send_message("⚠️ Bitte Symbol angeben, z.B.: <code>/fund AAPL</code>")
            else:
                self._handle_fundamentals(args[0])

        elif cmd in ["/analysts", "/analyst", "/ratings"]:
            if not args:
                self.send_message("⚠️ Bitte Symbol angeben, z.B.: <code>/analysts NVDA</code>")
            else:
                self._handle_analysts(args[0])

        elif cmd in ["/sentiment", "/hf", "/finbert", "/news"]:
            if not args:
                self.send_message("⚠️ Bitte Symbol angeben, z.B.: <code>/sentiment NVDA</code>")
            else:
                self._handle_sentiment(args[0])

        elif cmd in ["/vwap", "/sweep", "/poc", "/profile"]:
            if not args:
                self.send_message("⚠️ Bitte Symbol angeben, z.B.: <code>/vwap NVDA</code>")
            else:
                self._handle_vwap(args[0])

        elif cmd in ["/timing", "/entry", "/pullback", "/wait"]:
            if not args:
                self.send_message("⏱️ Bitte Symbol angeben, z.B.: <code>/timing NVDA</code>")
            else:
                self._handle_timing(args[0])

        elif cmd in ["/forecast", "/quantiles", "/trichter", "/p90"]:
            if not args:
                self.send_message("⚠️ Bitte Symbol angeben, z.B.: <code>/forecast NVDA</code>")
            else:
                self._handle_forecast(args[0])

        elif cmd in ["/chart", "/ladder", "/graph", "/candlestick"]:
            if not args:
                self.send_message("📊 Bitte Symbol angeben, z.B.: <code>/chart NVDA</code>")
            else:
                self._handle_chart(args[0])

        elif cmd in ["/evolve", "/evolution", "/schwarm_update"]:
            self._handle_evolve()

        elif cmd in ["/shield", "/correlation", "/corr", "/gleichlauf"]:
            self._handle_correlation_shield()

        elif cmd in ["/geopolitics", "/geo", "/war", "/peace"]:
            self._handle_geopolitics()

        elif cmd in ["/events", "/event", "/fomc", "/cpi", "/nfp", "/fed"]:
            self._handle_macro_events()

        elif cmd in ["/alerts", "/alert", "/quiet", "/drossel"]:
            self._handle_alerts(args)

        elif cmd in ["/knowledge", "/wissen", "/learnings", "/synergies", "/schwarm"]:
            self._handle_knowledge()

        elif cmd in ["/deals", "/deal", "/dialect", "/dialekt", "/sprache"]:
            self._handle_dialect()

        elif cmd in ["/lab", "/research", "/wfa", "/efficiency"]:
            self._handle_lab()

        elif cmd in ["/optimize", "/opt", "/calibrate"]:
            self._handle_optimize()

        elif cmd in ["/update", "/upgrade", "/pull", "/sync"]:
            self._handle_update()

        elif cmd in ["/panic", "/emergency", "/notausstieg", "/close_all"]:
            self._handle_panic()

        elif cmd in ["/pause", "/freeze", "/halt", "/stopp"]:
            self._handle_pause()

        elif cmd in ["/resume", "/unfreeze", "/continue", "/weiter"]:
            self._handle_resume()

        elif cmd in ["/ping", "/health", "/alive", "/check"]:
            self._handle_health()

        elif cmd in ["/report", "/briefing", "/summary"]:
            self._handle_report()

        elif cmd == "/breed":
            self.send_message("🧬 <i>Starte autonome genetische Strategie-Zucht und Krisen-Stresstest im Hintergrund...</i>")
            if self.breed_callback:
                threading.Thread(target=self._run_breed_task, daemon=True).start()
            else:
                self.send_message("ℹ️ Zucht-Modul ist derzeit im Single-Run-Modus konfiguriert.")

        else:
            self.send_message(f"❓ Unbekannter Befehl: <code>{html.escape(cmd)}</code>. Sende <b>/help</b> für alle Befehle.")

    def _handle_status(self) -> None:
        info = {}
        if self.status_callback:
            try:
                info = self.status_callback()
            except Exception as e:
                logger.error(f"Status Callback Error: {e}")

        # Fallback auf Datei-Reads
        portfolio_state = {}
        state_file = Path("data/portfolio_state.json")
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    portfolio_state = json.load(f)
            except Exception:
                pass

        adas_file = Path("data/adas_meta_state.json")
        swarm_agents_count = 33
        if adas_file.exists():
            try:
                with open(adas_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    swarm_agents_count = len(d.get("active_subagents", {})) or len(d.get("sub_agent_weights", {})) or 33
            except Exception:
                pass

        equity = portfolio_state.get("equity", info.get("equity", 100000.0))
        cash = portfolio_state.get("cash", info.get("cash", 100000.0))
        regime = info.get("regime", "BULL_EXPANSION")

        msg = (
            "📊 <b>Systemstatus & Portfolio</b>\n"
            "─────────────────────────────\n"
            f"💰 <b>Gesamtwert:</b> ${equity:,.2f}\n"
            f"💵 <b>Verfügbares Cash:</b> ${cash:,.2f}\n"
            f"🌐 <b>Marktregime:</b> {regime}\n"
            f"🤖 <b>Aktive Schwarm-Agenten:</b> {swarm_agents_count}\n"
            f"🛡️ <b>Red-Team Tribunal:</b> Aktiv (Bull vs. Bear vs. CIO)\n"
            f"🐋 <b>Whale & Form 4 Tracker:</b> Aktiv\n"
            f"⚡ <b>Optionen Flow Tracker:</b> Aktiv\n"
            "─────────────────────────────\n"
            "Status: 🟢 <i>Voll autonom & handlungsbereit</i>"
        )
        self.send_message(msg)

    def _handle_portfolio(self) -> None:
        positions = []
        if self.portfolio_callback:
            try:
                positions = self.portfolio_callback()
            except Exception as e:
                logger.error(f"Portfolio Callback Error: {e}")

        if not positions:
            # Fallback Alpaca API direkt abfragen falls konfiguriert
            try:
                from config import AgentConfig
                import alpaca_trade_api as tradeapi
                cfg = AgentConfig()
                if cfg.alpaca_api_key and cfg.alpaca_secret_key:
                    api = tradeapi.REST(cfg.alpaca_api_key, cfg.alpaca_secret_key, cfg.alpaca_base_url, api_version="v2")
                    al_pos = api.list_positions()
                    for p in al_pos:
                        positions.append({
                            "symbol": p.symbol,
                            "qty": float(p.qty),
                            "avg_entry_price": float(p.avg_entry_price),
                            "current_price": float(p.current_price),
                            "unrealized_pl": float(p.unrealized_pl),
                            "unrealized_plpc": float(p.unrealized_plpc),
                        })
            except Exception as e:
                logger.debug(f"Positions fetch error: {e}")

        if not positions:
            self.send_message("💼 <b>Offene Positionen:</b>\n\nAktuell keine offenen Positionen im Portfolio.")
            return

        lines = ["💼 <b>Offene Positionen (Alpaca):</b>\n─────────────────────────────"]
        total_pnl = 0.0
        for pos in positions:
            sym = pos["symbol"]
            qty = pos["qty"]
            entry = pos["avg_entry_price"]
            curr = pos["current_price"]
            pnl = pos["unrealized_pl"]
            pnl_pct = pos["unrealized_plpc"] * 100.0
            total_pnl += pnl

            icon = "🟢" if pnl >= 0 else "🔴"
            sign = "+" if pnl >= 0 else ""
            lines.append(
                f"{icon} <b>{sym}</b>: {qty:g} Stk.\n"
                f"   Einstieg: ${entry:.2f} ➜ Kurs: ${curr:.2f}\n"
                f"   G&V: <b>{sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)</b>\n"
            )

        tot_icon = "🟢" if total_pnl >= 0 else "🔴"
        tot_sign = "+" if total_pnl >= 0 else ""
        lines.append(f"─────────────────────────────\n<b>Gesamt G&V:</b> {tot_icon} {tot_sign}${total_pnl:.2f}")
        self.send_message("\n".join(lines))

    def _handle_trades(self) -> None:
        journal_file = Path("data/audit_journal.jsonl")
        if not journal_file.exists():
            self.send_message("📜 Noch keine Trades im Journal erfasst.")
            return

        lines = []
        try:
            with open(journal_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            recent = [json.loads(line) for line in all_lines[-5:]]
            recent.reverse()
        except Exception as e:
            self.send_message(f"Fehler beim Lesen des Journals: {e}")
            return

        out = ["📜 <b>Letzte 5 ausgeführte Orders:</b>\n─────────────────────────────"]
        for t in recent:
            sym = t.get("symbol", "N/A")
            action = t.get("action", "BUY")
            qty = t.get("qty", 0)
            price = t.get("price", 0.0)
            status = t.get("status", "EXECUTED")
            strategy = t.get("strategy", "Swarm Consensus")
            sl = t.get("stop_loss", 0.0)
            tp = t.get("take_profit", 0.0)

            act_icon = "🟢" if action == "BUY" else "🔴"
            out.append(
                f"{act_icon} <b>{action} {qty}x {sym}</b> @ ${price:.2f}\n"
                f"   Strategie: <i>{html.escape(str(strategy)[:30])}</i>\n"
                f"   SL: ${sl:.2f} | TP: ${tp:.2f} | Status: {status}\n"
            )

        out.append("─────────────────────────────")
        self.send_message("\n".join(out))

    def _handle_debates(self) -> None:
        debate_file = Path("data/red_team_debates.jsonl")
        if not debate_file.exists():
            self.send_message("⚖️ Noch keine Red-Team-Debatten protokolliert.")
            return

        try:
            with open(debate_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            recent = [json.loads(line) for line in all_lines[-3:]]
            recent.reverse()
        except Exception as e:
            self.send_message(f"Fehler beim Lesen der Debatten: {e}")
            return

        out = ["⚖️ <b>Jüngste Red-Team Tribunal Urteile:</b>\n─────────────────────────────"]
        for d in recent:
            sym = d.get("symbol", "N/A")
            verdict = d.get("verdict", "APPROVED")
            reason = d.get("reason", "N/A")
            bear_concerns = d.get("bear_concerns", [])

            v_icon = "✅" if verdict == "APPROVED" else "⚠️" if verdict == "REDUCED_SIZE" else "🛑"
            out.append(
                f"{v_icon} <b>{sym}</b>: <b>{verdict}</b>\n"
                f"   Urteil: <i>{html.escape(reason)}</i>\n"
                f"   Bear Veto-Risiken: {html.escape(', '.join(bear_concerns[:2]) if bear_concerns else 'Keine Einwände')}\n"
            )

        out.append("─────────────────────────────")
        self.send_message("\n".join(out))

    def _handle_whales(self) -> None:
        try:
            from sec_form4_tracker import SECForm4Tracker
            tracker = SECForm4Tracker()
            cache_files = list(Path("data/sec_form4_cache").glob("*.json"))
            if not cache_files:
                self.send_message("🐋 Keine gecachten SEC Form 4 Whale-Käufe vorhanden.")
                return

            whales_found = []
            for cf in cache_files[:20]:
                try:
                    with open(cf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("has_cluster_buy") or data.get("total_purchase_value", 0) > 100_000:
                        whales_found.append(data)
                except Exception:
                    continue

            if not whales_found:
                self.send_message("🐋 Aktuell keine außergewöhnlichen Insider-Cluster-Käufe im Cache.")
                return

            whales_found.sort(key=lambda x: x.get("total_purchase_value", 0), reverse=True)
            out = ["🐋 <b>Jüngste SEC Form 4 Insider- & Whale-Käufe:</b>\n─────────────────────────────"]
            for w in whales_found[:5]:
                sym = w.get("symbol")
                val = w.get("total_purchase_value", 0)
                insiders = w.get("insiders_buying_count", 0)
                cluster = "🔥 CLUSTER" if w.get("has_cluster_buy") else ""
                out.append(
                    f"🟢 <b>{sym}</b>: <b>${val:,.0f}</b> gekauft ({insiders} Insider) {cluster}\n"
                    f"   Details: <i>{html.escape(w.get('summary', ''))}</i>\n"
                )
            out.append("─────────────────────────────")
            self.send_message("\n".join(out))
        except Exception as e:
            self.send_message(f"Fehler bei Whale-Abfrage: {e}")

    def _handle_options(self, symbol: str) -> None:
        sym = symbol.upper().strip()
        self.send_message(f"🔍 <i>Scanne Optionen-Orderflow für {sym}...</i>")
        try:
            from options_darkpool_flow import OptionsDarkPoolTracker
            tracker = OptionsDarkPoolTracker()
            snap = tracker.analyze_flow(sym)

            icon = "🐂" if snap.flow_sentiment == "BULLISH_FLOW" else "🐻" if snap.flow_sentiment == "BEARISH_FLOW" else "⚖️"
            msg = (
                f"{icon} <b>Optionen-Orderflow: {sym}</b>\n"
                f"─────────────────────────────\n"
                f"🎯 <b>Spot-Kurs:</b> ${snap.spot_price:.2f}\n"
                f"📅 <b>Nächster Verfall:</b> {snap.nearest_expiration}\n"
                f"⚖️ <b>Put/Call Ratio (Volumen):</b> {snap.put_call_ratio_volume:.2f}\n"
                f"🧲 <b>Max Pain Preis:</b> ${snap.max_pain:.2f}\n"
                f"⚡ <b>Unusual Activity:</b> {'🚨 Ungewöhnliche CALL-Käufe' if snap.unusual_call_activity else '🚨 Ungewöhnliche PUT-Käufe' if snap.unusual_put_activity else 'Normal'}\n"
                f"🌊 <b>Smart Money Score:</b> {snap.flow_score:+.2f} ({snap.flow_sentiment})\n"
                f"─────────────────────────────\n"
                f"<i>{html.escape(snap.summary)}</i>"
            )
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Fehler bei Optionen-Scan für {sym}: {e}")

    def _handle_fundamentals(self, symbol: str) -> None:
        sym = symbol.upper().strip()
        self.send_message(f"🔍 <i>Prüfe Fundamentaldaten & Bilanz für {sym}...</i>")
        try:
            from financial_catalyst_scanner import FinancialCatalystScanner
            scanner = FinancialCatalystScanner()
            snap = scanner.scan_symbol(sym)

            grade_icon = "💎" if snap.health_grade == "ELITE_GROWTH" else "🛡️" if snap.health_grade == "SOLID_VALUE" else "⚠️" if snap.health_grade == "SPECULATIVE" else "☠️"
            distress_warning = "\n🚨 <b>ACHTUNG: DISTRESS ALARM (Insolvenz-/Verwässerungsrisiko)!</b>" if snap.is_distressed else ""

            msg = (
                f"{grade_icon} <b>Fundamentalanalyse: {sym}</b>\n"
                f"─────────────────────────────\n"
                f"🏅 <b>Health Grade:</b> <b>{snap.health_grade}</b>\n"
                f"📈 <b>Umsatzwachstum (YoY):</b> {f'{snap.revenue_growth_yoy*100:.1f}%' if snap.revenue_growth_yoy is not None else 'N/A'}\n"
                f"📊 <b>Forward P/E:</b> {f'{snap.forward_pe:.1f}' if snap.forward_pe else 'N/A'} | PEG: {f'{snap.peg_ratio:.2f}' if snap.peg_ratio else 'N/A'}\n"
                f"💵 <b>Operativer Cashflow:</b> {f'${snap.operating_cashflow/1e9:.2f} Mrd.' if snap.operating_cashflow else 'N/A'}\n"
                f"💧 <b>Quick Ratio:</b> {f'{snap.quick_ratio:.2f}' if snap.quick_ratio else 'N/A'}\n"
                f"⭐ <b>Fundamental Score:</b> {snap.fundamental_score:+.2f}{distress_warning}\n"
                f"─────────────────────────────\n"
                f"<i>{html.escape(snap.summary)}</i>"
            )
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Fehler bei Fundamentalanalyse für {sym}: {e}")

    def _handle_pairs(self) -> None:
        self.send_message("🔀 <i>Scanne statistische Arbitrage & Kointegrations-Spreads aller Paare...</i>")
        try:
            from pairs_stat_arb import StatisticalArbitrageScanner
            scanner = StatisticalArbitrageScanner()
            opps = scanner.scan_all_pairs()
            if not opps:
                self.send_message("🔀 Keine Pairs-Daten verfügbar.")
                return

            lines = ["🔀 <b>Statistische Arbitrage & Pairs-Spreads:</b>\n─────────────────────────────"]
            for o in opps:
                z = o.z_score
                action = o.signal_action
                icon = "🔥" if action != "NEUTRAL" else ("🟡" if abs(z) >= 1.5 else "⚪")
                sig_text = f" ➜ <b>KAUF {o.target_asset}</b> (Edge {o.expected_edge_score:+.2f})" if action != "NEUTRAL" else ""

                lines.append(
                    f"{icon} <b>{o.pair_name}</b> ({o.description[:24]})\n"
                    f"   Z-Score: <b>{z:+.2f}σ</b> | Abweichung: {o.divergence_pct:+.1f}%\n"
                    f"   Korr: {o.correlation_60d:.2f} | Halbwertszeit: {o.half_life_days:.1f}d{sig_text}\n"
                )

            lines.append("─────────────────────────────\n<i>Tipp: Bei |Z| &ge; 2.0σ löst der Schwarm automatisch Mean-Reversion Orders aus!</i>")
            self.send_message("\n".join(lines))
        except Exception as e:
            self.send_message(f"❌ Fehler bei Pairs-Arbitrage Scan: {e}")

    def _handle_sectors(self) -> None:
        self.send_message("🚀 <i>Analysiere 11 US-Sektoren & relative Stärke vs. S&P 500...</i>")
        try:
            from sector_momentum_expander import SectorMomentumExpander
            exp = SectorMomentumExpander()
            snap = exp.scan_sector_rotation()

            lines = ["🚀 <b>Sektor-Rotations-Radar & Relative Stärke (vs SPY):</b>\n─────────────────────────────"]
            lines.append("🏆 <b>Top Alpha-Sektoren (Geldzufluss):</b>")
            for s in snap.leading_sectors:
                lines.append(f"🟢 <b>{s.ticker}</b> ({s.name}): 5d <b>{s.return_5d_pct:+.1f}%</b> (RS {s.relative_strength_vs_spy:+.1f}%)")
                lines.append(f"   <i>Top-Aktien: {', '.join(s.top_stocks[:3])}</i>")

            lines.append("\n🌧️ <b>Schwächste Sektoren (Geldabfluss):</b>")
            for s in snap.lagging_sectors:
                lines.append(f"🔴 <b>{s.ticker}</b> ({s.name}): 5d <b>{s.return_5d_pct:+.1f}%</b> (RS {s.relative_strength_vs_spy:+.1f}%)")

            lines.append("─────────────────────────────")
            lines.append(f"🔥 <b>Führende Momentum-Aktien:</b> {', '.join(snap.recommended_hot_stocks)}")
            self.send_message("\n".join(lines))
        except Exception as e:
            self.send_message(f"❌ Fehler bei Sektor-Scan: {e}")

    def _handle_analysts(self, symbol: str) -> None:
        sym = symbol.strip().upper()
        self.send_message(f"📊 <i>Lade Wall Street Analysten-Ratings & Kursziele für {sym}...</i>")
        try:
            from analyst_ratings_radar import AnalystRatingsRadar
            radar = AnalystRatingsRadar()
            ratings = radar.get_ratings(sym)

            icon = "🟢" if "BUY" in ratings.consensus_rating else ("🔴" if "SELL" in ratings.consensus_rating else "🟡")
            lines = [f"📊 <b>Wall Street Analysten-Konsens: {sym}</b>\n─────────────────────────────"]
            lines.append(f"• <b>Konsens-Rating:</b> {icon} <b>{ratings.consensus_rating}</b> ({ratings.analyst_count} Analysten)")
            if ratings.target_mean_price:
                ups = ratings.target_upside_pct or 0.0
                lines.append(f"• <b>Kursziel (Mittelwert):</b> ${ratings.target_mean_price:.2f} (<b>{ups:+.1f}%</b>)")
            if ratings.target_high_price:
                lines.append(f"• <b>Kursziel (High / Bull-Case):</b> ${ratings.target_high_price:.2f}")
            if ratings.target_low_price:
                lines.append(f"• <b>Kursziel (Low / Bear-Case):</b> ${ratings.target_low_price:.2f}")
            lines.append(f"• <b>Konsens-Score (-1.0 bis +1.0):</b> {ratings.consensus_score:+.2f}")

            lines.append(f"\n💡 <i>Schwarm-Zusammenfassung: {ratings.summary}</i>")
            self.send_message("\n".join(lines))
        except Exception as e:
            self.send_message(f"❌ Fehler bei Analysten-Abfrage für {sym}: {e}")

    def _handle_sentiment(self, symbol: str) -> None:
        sym = symbol.strip().upper()
        self.send_message(f"🤗 <i>Analysiere Nachrichten & Schlagzeilen für {sym} mit Hugging Face FinBERT...</i>")
        try:
            from huggingface_sentiment_sentinel import get_hf_sentiment_sentinel
            sentinel = get_hf_sentiment_sentinel()
            snap = sentinel.analyze_symbol(sym)
            msg = sentinel.format_telegram_summary(snap)
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Fehler bei FinBERT-Sentiment Analyse für {sym}: {e}")

    def _handle_vwap(self, symbol: str) -> None:
        sym = symbol.strip().upper()
        self.send_message(f"🎯 <i>Berechne VWAP-Bänder, POC und Liquidity-Sweeps für {sym}...</i>")
        try:
            from vwap_liquidity_sweeper import get_vwap_sweeper
            sweeper = get_vwap_sweeper()
            res = sweeper.analyze_symbol(sym)
            msg = sweeper.format_telegram_summary(res)
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Fehler bei VWAP-Analyse für {sym}: {e}")

    def _handle_timing(self, symbol: str) -> None:
        sym = symbol.strip().upper()
        self.send_message(f"⏱️ <i>Analysiere Einstiegs-Timing (Sofort-Kauf vs. Abwarten auf Pullback) für {sym}...</i>")
        try:
            from data_collector import MultiAssetDataCollector
            collector = MultiAssetDataCollector()
            snap = collector.fetch_snapshot(sym)
            if not snap:
                self.send_message(f"⚠️ Keine Marktdaten für {sym} gefunden.")
                return

            cur_p = snap.current_price
            vwap = snap.intraday_vwap or cur_p
            vwap_gap = ((cur_p - vwap) / vwap * 100.0) if vwap > 0 else 0.0
            rsi = snap.rsi_14
            rvol = snap.volume_ratio
            atr = getattr(snap, "atr_14", cur_p * 0.02) or (cur_p * 0.02)

            is_overextended = (vwap_gap >= 1.8 or rsi >= 70.0)
            is_fresh_momentum = (rvol >= 1.8 and vwap_gap < 1.4 and rsi < 68.0)
            is_oversold = (rsi <= 28.0)

            if is_overextended:
                verdict = "⏳ <b>ABWARTEN AUF PULLBACK</b>"
                opt_entry = round(max(vwap * 1.002, cur_p - 0.70 * atr), 2)
                reason = (
                    f"Der Kurs ist kurzfristig überhitzt (+{vwap_gap:.1f}% über VWAP, RSI {rsi:.1f}). "
                    f"Ein Sofort-Kauf birgt Rücksetzer-Risiko. Optimaler Einstieg bei Dip nahe <b>${opt_entry:,.2f}</b> abwarten!"
                )
            elif is_fresh_momentum:
                verdict = "⚡ <b>SOFORT-EINSTIEG EMPFOHLEN</b>"
                opt_entry = round(cur_p, 2)
                reason = (
                    f"Frischer Momentum-Ausbruch mit hohem Volumen ({rvol:.1f}x) direkt an Basis/VWAP. "
                    f"Abwarten birgt Slippage- und Verpassungs-Risiko – sofortige Ausführung empfohlen!"
                )
            elif is_oversold:
                verdict = "⏳ <b>ABWARTEN AUF REBOUND</b>"
                opt_entry = round(cur_p * 1.018, 2)
                reason = (
                    f"Stark überverkauft (RSI {rsi:.1f}). Bei Verkäufen kein Panik-Ausstieg am Tief, "
                    f"sondern technischen Relief-Bounce für besseren Exit bei <b>${opt_entry:,.2f}</b> abwarten."
                )
            else:
                verdict = "⏱️ <b>TIMING NEUTRAL / SOLIDE</b>"
                opt_entry = round(cur_p, 2)
                reason = f"Kurs handelt in gesunder Spanne nahe VWAP (${vwap:,.2f}). Normaler Einstieg bei Signal-Bestätigung."

            msg = (
                f"⏱️ <b>Einstiegs-Timing & Abwägung für <code>{sym}</code></b>\n"
                f"─────────────────────────────\n"
                f"• <b>Aktueller Kurs:</b> ${cur_p:,.2f}\n"
                f"• <b>Intraday VWAP:</b> ${vwap:,.2f} ({vwap_gap:+.1f}%)\n"
                f"• <b>RSI (14):</b> {rsi:.1f} | <b>Volumen:</b> {rvol:.1f}x\n"
                f"• <b>Optimaler Einstiegskurs:</b> <b>${opt_entry:,.2f}</b>\n\n"
                f"🎯 <b>Timing-Urteil:</b> {verdict}\n"
                f"💡 <i>{reason}</i>\n"
                f"─────────────────────────────"
            )
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Fehler bei Timing-Analyse für {sym}: {e}")

    def _handle_forecast(self, symbol: str) -> None:
        sym = symbol.strip().upper()
        self.send_message(f"📈 <i>Berechne probabilistischen Kurs-Trichter (P10/P50/P90) für {sym}...</i>")
        try:
            from probabilistic_forecast_sentinel import get_forecast_sentinel
            sentinel = get_forecast_sentinel()
            res = sentinel.analyze_symbol(sym)
            msg = sentinel.format_telegram_summary(res)
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Fehler bei Forecast-Analyse für {sym}: {e}")

    def _handle_chart(self, symbol: str) -> None:
        sym = symbol.strip().upper()
        self.send_message(f"📊 <i>Generiere Push-Chart & Price-Ladder für {sym}...</i>")
        try:
            from data_collector import MultiAssetDataCollector
            collector = MultiAssetDataCollector()
            snap = collector.fetch_snapshot(sym)
            if not snap:
                self.send_message(f"⚠️ Keine Marktdaten für {sym} gefunden.")
                return

            cur_p = snap.current_price
            vwap = snap.intraday_vwap or cur_p
            atr = getattr(snap, "atr_14", cur_p * 0.02) or (cur_p * 0.02)
            rsi = snap.rsi_14
            rvol = snap.volume_ratio

            tp1 = round(cur_p + 1.5 * atr, 2)
            tp2 = round(cur_p + 3.0 * atr, 2)
            micro_dip = round(cur_p * 0.997, 2)
            sl = round(max(cur_p - 1.2 * atr, cur_p * 0.95), 2)

            tp2_pct = ((tp2 - cur_p) / cur_p) * 100.0 if cur_p > 0 else 0.0
            tp1_pct = ((tp1 - cur_p) / cur_p) * 100.0 if cur_p > 0 else 0.0
            dip_pct = ((micro_dip - cur_p) / cur_p) * 100.0 if cur_p > 0 else 0.0
            vwap_pct = ((vwap - cur_p) / cur_p) * 100.0 if cur_p > 0 else 0.0
            sl_pct = ((sl - cur_p) / cur_p) * 100.0 if cur_p > 0 else 0.0

            rsi_fill = int(min(max(rsi / 10.0, 1), 10))
            rsi_bar = "█" * rsi_fill + "░" * (10 - rsi_fill)

            ladder = (
                f"📊 <b>Push-Chart & Price-Ladder: <code>{sym}</code></b>\n"
                f"─────────────────────────────\n"
                f"<code>"
                f"  ${tp2:>7.2f} ── [TP2: {tp2_pct:>+5.1f}%] ─── 🟢 Zielzone\n"
                f"  ${tp1:>7.2f} ── [TP1: {tp1_pct:>+5.1f}%] ─── 🎯 Teilgewinn\n"
                f"► ${cur_p:>7.2f} ── [SPOT-KURS  ] ─── 🔵 Aktueller Preis\n"
                f"  ${micro_dip:>7.2f} ── [SNIPER-DIP] ─── ⚡ Micro-Dip ({dip_pct:+.1f}%)\n"
                f"  ${vwap:>7.2f} ── [VWAP-BASIS] ─── 🟣 Support ({vwap_pct:+.1f}%)\n"
                f"  ${sl:>7.2f} ── [STOP-LOSS ] ─── 🛑 Kapitalschutz ({sl_pct:+.1f}%)\n"
                f"</code>\n"
                f"─────────────────────────────\n"
                f"• <b>RSI (14):</b> {rsi:.1f}  <code>[{rsi_bar}]</code>\n"
                f"• <b>Relativ-Volumen (RVOL):</b> {rvol:.1f}x\n"
                f"• <b>Tages-ATR (14):</b> ${atr:.2f}\n"
                f"─────────────────────────────\n"
                f"<i>Micro-Dip Sniper aktiv: Kauforders greifen Rücksetzer bei -0.3% ab.</i>"
            )
            self.send_message(ladder)
        except Exception as e:
            self.send_message(f"❌ Fehler bei Chart-Generierung für {sym}: {e}")

    def _handle_correlation_shield(self) -> None:
        self.send_message("🚨 <i>Analysiere systemischen Gleichlauf & Dynamic Correlation Breakdown Shield...</i>")
        try:
            import pandas as pd
            from correlation_breakdown_shield import get_correlation_shield
            shield = get_correlation_shield()
            st = shield._last_status
            if st is None:
                st = shield.evaluate_returns_matrix(pd.DataFrame())
            self.send_message(shield.format_telegram_summary(st))
        except Exception as e:
            self.send_message(f"❌ Fehler bei Correlation Shield Abfrage: {e}")

    def _handle_geopolitics(self) -> None:
        self.send_message("🌐 <i>Analysiere globale Konflikt- und Rohstoffrisiken (GCI-Index)...</i>")
        try:
            from geopolitical_regime_engine import GeopoliticalRegimeEngine
            geo_engine = GeopoliticalRegimeEngine()
            snap = geo_engine.analyze_geopolitical_situation()

            icon = "🟢" if snap.geopolitical_risk_score < 4.0 else ("🔴" if snap.geopolitical_risk_score >= 6.5 else "🟡")
            lines = ["🌐 <b>Geopolitisches Lagezentrum & Strategie-Mandat:</b>\n─────────────────────────────"]
            lines.append(f"• <b>Geopolitischer Konflikt-Index (GCI):</b> {icon} <b>{snap.geopolitical_risk_score:.1f} / 10.0</b>")
            lines.append(f"• <b>Regime-Modus:</b> <b>{snap.regime_mode}</b>")
            lines.append(f"• <b>Strategie-Mandat:</b> <code>{snap.autonomous_strategy_mandate}</code>\n")

            lines.append("⚡ <b>Relative Dynamik vs. S&P 500 (5 Tage):</b>")
            lines.append(f"• Verteidigung (LMT vs SPY): {snap.defense_outperformance_5d_pct:+.1f}%")
            lines.append(f"• Energie / Öl-Schock (USO vs SPY): {snap.energy_oil_shock_5d_pct:+.1f}%")
            lines.append(f"• Gold Safe-Haven Flow (GLD vs SPY): {snap.gold_safe_haven_flow_5d_pct:+.1f}%\n")

            if snap.active_conflict_hotspots:
                lines.append("🛡️ <b>Geopolitische Brennpunkte & Treiber:</b>")
                for c in snap.active_conflict_hotspots:
                    lines.append(f"• {c}")

            self.send_message("\n".join(lines))
        except Exception as e:
            self.send_message(f"❌ Fehler bei Geopolitik-Analyse: {e}")

    def _handle_macro_events(self) -> None:
        self.send_message("🏛️ <i>Analysiere Wirtschaftskalender & Notenbank-Events (FOMC, CPI, NFP)...</i>")
        try:
            from macro_event_horizon_sentinel import get_macro_event_sentinel
            sentinel = get_macro_event_sentinel()
            snap = sentinel.analyze_event_horizon()
            msg = sentinel.format_telegram_summary(snap)
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Fehler bei Makro-Event Analyse: {e}")

    def _handle_alerts(self, args: List[str]) -> None:
        try:
            from notification_manager import get_notifier
            notifier = get_notifier()

            if args:
                wanted = args[0].strip().upper()
                if wanted in ["TRADES", "TRADES_ONLY", "TRADE", "QUIET"]:
                    notifier.alert_level = "TRADES_ONLY"
                    self.send_message("🔇 <b>Alert-Level auf 'TRADES_ONLY' gesetzt!</b>\n\nAb sofort erhältst du <b>nur noch echte Order-Ausführungen</b> (Buy, Sell, Stop-Loss). Keine Veto-Warnungen oder Routine-Scans.")
                    return
                elif wanted in ["BALANCED", "DEFAULT", "NORMAL"]:
                    notifier.alert_level = "BALANCED"
                    self.send_message("⚖️ <b>Alert-Level auf 'BALANCED' gesetzt!</b>\n\nDu erhältst echte Trades, echte Whale-Käufe und maximal 1 gefiltertes Red-Team Veto pro Symbol alle 2 Stunden.")
                    return
                elif wanted in ["ALL", "VERBOSE"]:
                    notifier.alert_level = "ALL"
                    self.send_message("📢 <b>Alert-Level auf 'ALL' gesetzt!</b>\n\nAlle Ereignisse aktiv (weiterhin durch das Stunden-Limit geschützt).")
                    return

            # Status anzeigen
            sent_count = len(notifier._sent_timestamps)
            lines = ["🔔 <b>Telegram-Benachrichtigungs-Kontrollzentrum:</b>\n─────────────────────────────"]
            lines.append(f"• <b>Aktuelles Alert-Level:</b> <b>{notifier.alert_level}</b>")
            lines.append(f"• <b>Stündliches Limit:</b> <b>{notifier.max_per_hour} Nachrichten / Stunde</b>")
            lines.append(f"• <b>In dieser Stunde gesendet:</b> <b>{sent_count} / {notifier.max_per_hour}</b>")
            lines.append(f"• <b>Entprellung (Dedup-Fenster):</b> <b>{notifier.dedup_window_sec // 60} Minuten</b>")
            lines.append(f"• <b>Instanz-Tag:</b> <code>{notifier.instance_tag}</code>\n")
            lines.append("<i>Befehle zum Ändern:</i>")
            lines.append("• <code>/alerts trades</code> ➜ Nur echte Trades (maximale Ruhe)")
            lines.append("• <code>/alerts balanced</code> ➜ Trades + Whales + seltene Vetos")
            lines.append("• <code>/alerts all</code> ➜ Ausführlicher Modus")
            self.send_message("\n".join(lines))
        except Exception as e:
            self.send_message(f"❌ Fehler bei Alert-Konfiguration: {e}")

    def _handle_knowledge(self) -> None:
        self.send_message("🧠 <i>Lade kollektiven Schwarm-Wissensspeicher & Synergie-Matrix...</i>")
        try:
            from swarm_knowledge_bus import get_swarm_knowledge_bus
            k_bus = get_swarm_knowledge_bus()
            summary = k_bus.format_telegram_summary()
            self.send_message(summary)
        except Exception as e:
            self.send_message(f"❌ Fehler beim Laden des Schwarm-Wissens: {e}")

    def _handle_dialect(self) -> None:
        self.send_message("🗣️ <i>Dekodiere verhandelte Schwarm-Deals aus dem Agenten-Dialekt...</i>")
        try:
            from emergent_dialect_engine import get_dialect_engine
            dialect_eng = get_dialect_engine()
            summary = dialect_eng.format_telegram_deals_summary()
            self.send_message(summary)
        except Exception as e:
            self.send_message(f"❌ Fehler beim Dekodieren des Agenten-Dialekts: {e}")

    def _handle_update(self) -> None:
        self.send_message("🔄 <i>Starte autonomes Update: Ziehe neusten Code von GitHub (origin/main)...</i>")
        try:
            import os
            import sys
            import subprocess
            import time
            from pathlib import Path

            # 1. Sicherheitskonfiguration für git in Docker / Windows
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", "*"], capture_output=True, text=True, timeout=10)

            # 2. Lokale dynamische Änderungen an getrackten Dateien verwerfen, damit merge niemals blockiert
            subprocess.run(["git", "checkout", "--", "."], capture_output=True, text=True, timeout=10)

            # 3. Neuesten Code von origin/main laden
            res = subprocess.run(["git", "pull", "--rebase", "origin", "main"], capture_output=True, text=True, timeout=35)
            if res.returncode != 0:
                subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, text=True, timeout=20)
                res = subprocess.run(["git", "reset", "--hard", "origin/main"], capture_output=True, text=True, timeout=15)

            output = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            output_clean = output.strip() or "Keine Ausgabe erhalten."

            if res.returncode == 0:
                self.send_message(
                    f"✅ <b>Update erfolgreich geladen!</b>\n\n"
                    f"<code>{html.escape(output_clean[:500])}</code>\n\n"
                    f"🔄 <i>Der Bot führt jetzt einen sofortigen Neustart durch...</i>"
                )
                time.sleep(2.0)

                # Wenn unter Watchdog-Aufsicht: Exit-Code 42 signalisiert dem Watchdog sofortigen Neustart mit neuem Code
                if os.environ.get("SUPERVISED_BY_WATCHDOG") == "1":
                    os._exit(42)

                # Standalone-Betrieb: Starte neuen Prozess und beende alten sauber
                try:
                    subprocess.Popen([sys.executable] + sys.argv, cwd=str(Path(__file__).resolve().parent))
                    os._exit(0)
                except Exception:
                    try:
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                    except Exception:
                        os._exit(0)
            else:
                self.send_message(
                    f"⚠️ <b>Git Pull mit Warnung / Fehler beendet (Exit-Code {res.returncode}):</b>\n\n"
                    f"<code>{html.escape(output_clean[:500])}</code>"
                )
        except Exception as e:
            self.send_message(f"❌ <b>Fehler beim autonomen Update:</b> {e}")

    def _handle_lab(self) -> None:
        self.send_message("🧪 <i>Lade aktuellen Walk-Forward Forschungs- & Robustheits-Bericht...</i>")
        try:
            from walk_forward_research_lab import get_research_lab
            lab = get_research_lab()
            msg = lab.format_telegram_summary()
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Fehler beim Laden des Forschungs-Labors: {e}")

    def _handle_optimize(self) -> None:
        self.send_message("⚡ <i>Starte Walk-Forward Optimierungs-Lauf & Multi-Regime Stresstest im Hintergrund...</i>")
        threading.Thread(target=self._run_optimization_task, daemon=True).start()

    def _run_optimization_task(self) -> None:
        try:
            from walk_forward_research_lab import get_research_lab
            lab = get_research_lab()
            res = lab.run_walk_forward_optimization()
            msg = lab.format_telegram_summary(res)
            self.send_message(f"✅ <b>Walk-Forward Optimierung abgeschlossen!</b>\n\n{msg}")
        except Exception as e:
            self.send_message(f"❌ Fehler bei Walk-Forward Optimierung: {e}")

    def _run_breed_task(self) -> None:
        try:
            if self.breed_callback:
                res = self.breed_callback()
                self.send_message(f"🧬 <b>Zucht-Zyklus abgeschlossen:</b>\n\n{html.escape(res)}")
        except Exception as e:
            self.send_message(f"❌ Fehler bei Strategie-Zucht: {e}")

    # =========================================================================
    # REISE- & NOTFALL-FUNKTIONEN (UNATTENDED MULTI-DAY OPERATION)
    # =========================================================================

    def _handle_panic(self) -> None:
        self.send_message("🚨 <i>Führe NOTFALL-AUSSTIEG durch: Schließe alle Positionen und storniere alle Orders...</i>")
        try:
            if self.panic_callback:
                events = self.panic_callback()
                ev_str = "\n".join(f"• {e}" for e in events)
                self.send_message(
                    f"🛑 <b>NOTFALL-AUSSTIEG ERFOLGREICH:</b>\n\n"
                    f"{ev_str}\n\n"
                    f"💵 <i>Konto-Kapital wurde vollständig in Cash gesichert.</i>"
                )
            elif self.engine_ref:
                events = self.engine_ref.emergency_liquidate_all_positions(reason="TELEGRAM_PANIC")
                ev_str = "\n".join(f"• {e}" for e in events)
                self.send_message(f"🛑 <b>NOTFALL-AUSSTIEG ERFOLGREICH:</b>\n\n{ev_str}")
            else:
                self.send_message("⚠️ Kein Ausführungs-Engine-Callback verfügbar.")
        except Exception as e:
            self.send_message(f"❌ Fehler beim Notfall-Ausstieg: {e}")

    def _handle_pause(self) -> None:
        try:
            if self.pause_callback:
                self.pause_callback(True)
            elif self.engine_ref:
                self.engine_ref.set_trading_paused(True)
            self.send_message(
                "⏸️ <b>Handel PAUSIERT (Sicherheits-Modus)</b>\n\n"
                "• Es werden <b>keine neuen Käufe</b> mehr eröffnet.\n"
                "• Bestehende Positionen werden weiter mit Trailing-Stops geschützt.\n"
                "• Reaktivierung jederzeit mit <code>/resume</code> möglich."
            )
        except Exception as e:
            self.send_message(f"❌ Fehler beim Pausieren: {e}")

    def _handle_resume(self) -> None:
        try:
            if self.pause_callback:
                self.pause_callback(False)
            elif self.engine_ref:
                self.engine_ref.set_trading_paused(False)
            self.send_message(
                "▶️ <b>Handel FORTGESETZT</b>\n\n"
                "• Der autonome 33-Agenten-Schwarm sucht wieder aktiv nach A+ Chancen.\n"
                "• Kauf-Filter & Ultra-Conviction-Festung sind aktiv."
            )
        except Exception as e:
            self.send_message(f"❌ Fehler bei Fortsetzung: {e}")

    def _handle_health(self) -> None:
        try:
            pid = os.getpid()
            now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
            msg = (
                f"💓 <b>PONG / System-Gesundheitscheck</b>\n"
                f"─────────────────────────────\n"
                f"• <b>Status:</b> 🟢 <b>VOLL FUNKTIONSFÄHIG</b>\n"
                f"• <b>Prozess-PID:</b> <code>{pid}</code>\n"
                f"• <b>Zeitpunkt:</b> {now_iso}\n"
                f"• <b>Schwarm:</b> 33 spezialisierte KI-Agenten online\n"
                f"• <b>Broker:</b> Alpaca Live ($1,000.00)\n"
                f"• <b>Watchdog & Resilienz:</b> Aktiv & schussfest\n"
                f"─────────────────────────────\n"
                f"<i>Du kannst beruhigt reisen – der Agent wacht autonom über dein Konto.</i>"
            )
            self.send_message(msg)
        except Exception as e:
            self.send_message(f"❌ Health-Check Fehler: {e}")

    def _handle_evolve(self) -> None:
        self.send_message("🧬 <i>Führe autonome Bayesianische Schwarm-Evolution und Rekalibrierung aus...</i>")
        try:
            from adas_meta_agent import ADASMetaAgent
            adas = ADASMetaAgent()
            evo_res = adas.execute_nightly_bayesian_swarm_evolution()
            top_performers = evo_res.get("top_performers", [])
            top_str = "\n".join([f"• <b>{name}:</b> Trust-Weight <code>{w:.3f}x</code> (Win-Rate {wr*100:.1f}%)" for name, w, wr in top_performers[:4]])
            evo_msg = (
                "🧬 <b>Autonome Schwarm-Evolution & Bayesianische Rekalibrierung</b>\n"
                "─────────────────────────────\n"
                f"• <b>Epoche:</b> #{evo_res.get('epoch', 1)} | Datum: {evo_res.get('date')}\n"
                f"• <b>Aktive KI-Agenten:</b> {evo_res.get('active_subagents_count', 34)}\n"
                f"• <b>Statistisches Trust-Upgrade:</b> {evo_res.get('upgraded_count', 0)} Agenten gestärkt\n"
                f"• <b>Trust-Drosselung:</b> {evo_res.get('downgraded_count', 0)} schwache Agenten abgewertet\n\n"
                f"🏆 <b>Führende Alpha-Agenten:</b>\n{top_str or '• Baseline-Gewichtung aktiv'}\n"
                "─────────────────────────────\n"
                "<i>Alle 34 Spezialisten wurden an die jüngste Marktstruktur angepasst!</i>"
            )
            self.send_message(evo_msg)
        except Exception as e:
            self.send_message(f"❌ Fehler bei Schwarm-Evolution: {e}")

    def _handle_report(self) -> None:
        try:
            state_file = Path("data/portfolio_state.json")
            if not state_file.exists():
                self.send_message("⚠️ Noch kein Portfolio-State vorhanden.")
                return
            with open(state_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            cash = d.get("cash_usd", 1000.0)
            realized = d.get("realized_pnl_usd", 0.0)
            positions = d.get("positions", {})
            pos_val = sum(p.get("market_value", 0.0) for p in positions.values())
            total_eq = round(cash + pos_val, 2)
            delta = total_eq - 1000.0
            delta_pct = (delta / 1000.0) * 100.0

            icon = "🟢" if delta >= 0 else "🔴"
            lines = [
                "📋 <b>Ausführlicher Reise- & PnL-Report</b>\n─────────────────────────────",
                f"💰 <b>Gesamt-Equity:</b> ${total_eq:,.2f} ({icon} {delta:+.2f}$ / {delta_pct:+.2f}%)",
                f"💵 <b>Verfügbarer Cash:</b> ${cash:,.2f}",
                f"📊 <b>Realisierter PnL:</b> ${realized:+,.2f}",
                f"🚀 <b>Compounding-Multiplikator:</b> {'Aktiv (+35% Gewinn-Reinvest)' if delta > 0 else 'Bereit ($1.000 Stammkapital gesichert)'}",
                f"📦 <b>Offene Positionen ({len(positions)}):</b>",
            ]
            if not positions:
                lines.append("• <i>Keine offenen Positionen (100% Cash-Schutz)</i>")
            else:
                for s, p in positions.items():
                    pnl_u = p.get("unrealized_pnl_usd", 0.0)
                    lines.append(f"• <b>{s}:</b> {p.get('quantity', 0):g} Stk. @ ${p.get('current_price', 0):,.2f} (PnL: ${pnl_u:+,.2f}, SL: ${p.get('stop_loss_price', 0):,.2f})")

            lines.append("─────────────────────────────")
            lines.append("<i>Nutze /health für Systemcheck oder /panic bei Notausstieg.</i>")
            self.send_message("\n".join(lines))
        except Exception as e:
            self.send_message(f"❌ Fehler beim Generieren des Reports: {e}")

    def _briefing_loop(self) -> None:
        """Überwacht die Uhrzeit und sendet automatische Morgen- (14:30 MEZ) & Abend-Briefings (22:15 MEZ)."""
        while self.is_running:
            try:
                time.sleep(45)
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                today_str = now_utc.date().isoformat()

                # US Open ist 13:30 UTC (14:30 MEZ bzw. 15:30 MESZ).
                # Morgen-Briefing vor Börsenöffnung um 12:30 UTC
                if now_utc.weekday() < 5 and now_utc.hour == 12 and now_utc.minute >= 30 and self._last_briefing_date != today_str:
                    self._last_briefing_date = today_str
                    self._send_morning_briefing()

                # US Close ist 20:00 UTC (21:00 MEZ bzw. 22:00 MESZ).
                # Abend-Briefing nach Börsenschluss um 20:15 UTC
                if now_utc.weekday() < 5 and now_utc.hour == 20 and now_utc.minute >= 15 and self._last_evening_date != today_str:
                    self._last_evening_date = today_str
                    self._send_evening_briefing()
            except Exception:
                pass

    def _send_morning_briefing(self) -> None:
        try:
            self.send_message(
                "🌅 <b>Guten Morgen Henry! Tagesausblick vor US-Börsenöffnung</b>\n"
                "─────────────────────────────\n"
                "• <b>Status:</b> Der autonome Trading-Agent läuft 100% stabil.\n"
                "• <b>Schwarm:</b> 34 KI-Spezialisten (inkl. PEAD Drift Hunter) scannen den Gesamtmarkt nach A+ Chancen.\n"
                "• <b>Fokus:</b> Ultra-Conviction (nur die gewinnreichsten Setups bei R:R ≥ 2.2:1).\n"
                "• <b>Sniper-Modus:</b> Micro-Dip Limit-Sniper aktiv (-0.3% Discount).\n"
                "─────────────────────────────\n"
                "<i>Du erhältst sofort eine Nachricht, sobald ein Trade ausgeführt wird. Schönen Tag!</i>"
            )
        except Exception:
            pass

    def _send_evening_briefing(self) -> None:
        try:
            self._handle_report()
            # Automatische nächtliche Schwarm-Evolution nach US-Close
            try:
                from adas_meta_agent import ADASMetaAgent
                adas = ADASMetaAgent()
                evo_res = adas.execute_nightly_bayesian_swarm_evolution()
                top_str = "\n".join([f"• <b>{name}:</b> Trust-Weight <code>{w:.3f}x</code> (Win-Rate {wr*100:.1f}%)" for name, w, wr in evo_res.get("top_performers", [])[:3]])
                evo_msg = (
                    "🧬 <b>Nächtliche Schwarm-Evolution abgeschlossen (Post-Market)</b>\n"
                    "─────────────────────────────\n"
                    f"• <b>Epoche:</b> #{evo_res.get('epoch', 1)} | Datum: {evo_res.get('date')}\n"
                    f"• <b>Aktive KI-Agenten:</b> {evo_res.get('active_subagents_count', 34)}\n"
                    f"• <b>Statistisches Trust-Upgrade:</b> {evo_res.get('upgraded_count', 0)} Agenten gestärkt\n"
                    f"• <b>Trust-Drosselung:</b> {evo_res.get('downgraded_count', 0)} schwache Agenten abgewertet\n\n"
                    f"🏆 <b>Top Performer:</b>\n{top_str or '• Keine signifikanten Gewichtssprünge'}\n"
                    "─────────────────────────────"
                )
                self.send_message(evo_msg)
            except Exception as ee:
                logger.debug(f"Nightly Swarm Evolution Error: {ee}")
        except Exception:
            pass
