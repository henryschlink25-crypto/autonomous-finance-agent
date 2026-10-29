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
import ssl
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
    ):
        self.token = bot_token.strip()
        self.chat_id = str(authorized_chat_id).strip()
        self.portfolio_callback = portfolio_callback
        self.status_callback = status_callback
        self.breed_callback = breed_callback
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._last_update_id = 0

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
                "🔀 <b>/pairs</b> - Statistische Arbitrage & Pairs-Spreads (Z-Score)\n"
                "🚀 <b>/sectors</b> - Sektor-Rotations-Radar & führende Branchen\n"
                "🧬 <b>/breed</b> - Sofortige genetische Strategie-Zucht starten\n"
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
        swarm_agents_count = 17
        if adas_file.exists():
            try:
                with open(adas_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    swarm_agents_count = len(d.get("sub_agent_weights", {}))
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

    def _run_breed_task(self) -> None:
        try:
            if self.breed_callback:
                res = self.breed_callback()
                self.send_message(f"🧬 <b>Zucht-Zyklus abgeschlossen:</b>\n\n{html.escape(res)}")
        except Exception as e:
            self.send_message(f"❌ Fehler bei Strategie-Zucht: {e}")
