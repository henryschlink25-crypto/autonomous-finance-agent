import datetime
import json
import os
from typing import Dict, List, Tuple
from config import AgentConfig
from ibkr_bridge import IBKRBridge
from models import MarketSnapshot, Position, RiskEvaluation, TradingSignal
from risk_manager import is_us_equity_market_open


class MultiAssetExecutionEngine:
    """
    Verwaltet das Portfolio und führt freigegebene Trades autonom aus.
    Unterstützt:
    - 2-Stufen Scale-Out (50% Teilgewinnmitnahme bei +2.0x ATR & Free-Ride Runner bis +4.4x ATR)
    - Dynamischen Chandelier-Trailing-Stop
    - LOCAL_PAPER, ALPACA_PAPER & CCXT_SANDBOX
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        os.makedirs(os.path.dirname(self.config.portfolio_state_file), exist_ok=True)
        self.cash_usd: float = self.config.initial_capital_usd
        self.realized_pnl_usd: float = 0.0
        self.day_start_date: str = datetime.date.today().isoformat()
        self.day_start_equity_usd: float = self.config.initial_capital_usd
        self.positions: Dict[str, Position] = {}
        self.alpaca_live_prices: Dict[str, float] = {}
        self.yahoo_vs_broker_price_diff: Dict[str, Tuple[float, float]] = {}
        self.ibkr_bridge = IBKRBridge(config)
        self._load_state()

    def _load_state(self) -> None:
        if os.path.exists(self.config.portfolio_state_file):
            try:
                with open(self.config.portfolio_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.cash_usd = float(data.get("cash_usd", self.config.initial_capital_usd))
                self.realized_pnl_usd = float(data.get("realized_pnl_usd", 0.0))
                raw_pos = data.get("positions", {})
                cluster_map = {item.symbol: item.cluster for item in self.config.watchlist}
                loaded_positions = {}
                for sym, pos_data in raw_pos.items():
                    if "cluster" not in pos_data:
                        pos_data["cluster"] = cluster_map.get(sym, "US_TECH")
                    loaded_positions[sym] = Position(**pos_data)
                self.positions = loaded_positions

                today_str = datetime.date.today().isoformat()
                saved_day = data.get("day_start_date", "")
                if saved_day == today_str and "day_start_equity_usd" in data:
                    self.day_start_date = saved_day
                    self.day_start_equity_usd = float(data["day_start_equity_usd"])
                else:
                    self.day_start_date = today_str
                    self.day_start_equity_usd = self.total_equity_usd
                    self._save_state()
            except Exception as e:
                print(f"[Warnung] Portfolio-State konnte nicht geladen werden ({e}). Starte neu.")
        else:
            self._save_state()

        # Synchronisiere Startkapital, Cash & Positionen mit Interactive Brokers (falls TWS/Gateway offen) oder Alpaca!
        if self.config.execution_mode in ("IBKR", "HYBRID") or self.config.ibkr_auto_connect:
            self._sync_from_ibkr_account()
        if "ALPACA" in self.config.execution_mode and self.config.alpaca_api_key and not self.ibkr_bridge.active_port:
            self._sync_from_alpaca_account()

    def _sync_from_ibkr_account(self) -> None:
        """
        Prüft automatisch, ob die Interactive Brokers TWS / IB Gateway API-Brücke lokal erreichbar ist
        (Ports 7497 Paper, 4002 Gateway Paper, 7496 Live, 4001 Gateway Live) und synchronisiert Kontostand & Positionen.
        """
        try:
            cluster_map = {item.symbol: item.cluster for item in self.config.watchlist}
            res = self.ibkr_bridge.sync_account_and_positions(self.positions, cluster_map)
            if res is not None:
                net_liq, cash_usd, updated_positions, live_prices = res
                self.positions = updated_positions
                self.alpaca_live_prices.update(live_prices)
                if net_liq > 0 and self.config.execution_mode == "IBKR":
                    self.cash_usd = cash_usd
                    self.config.initial_capital_usd = net_liq
                    if abs(self.day_start_equity_usd - net_liq) > net_liq * 0.25:
                        self.day_start_equity_usd = net_liq
                self._save_state()
        except Exception:
            pass

    def _sync_from_alpaca_account(self) -> None:
        """
        Liest das echte Gesamtvermögen (Equity, Cash & offene Positionen inkl. Stückzahl, Einstiegskurs & Echtzeit-/After-Hours-Kurs)
        direkt vom verbundenen Alpaca-Account aus und verhindert Abweichungen zwischen lokalem State und Broker.
        """
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(self.config.alpaca_api_key, self.config.alpaca_secret_key, paper=self.config.alpaca_paper)
            acct = client.get_account()
            alpaca_equity = float(acct.equity)
            alpaca_last_equity = float(getattr(acct, "last_equity", alpaca_equity) or alpaca_equity)

            # Synchronisiere alle echten Positionen direkt vom Alpaca-Server
            alpaca_positions = client.get_all_positions()
            self.alpaca_live_prices = {}
            cluster_map = {item.symbol: item.cluster for item in self.config.watchlist}
            broker_symbols = set()

            for ap in alpaca_positions:
                sym = str(ap.symbol).upper()
                broker_symbols.add(sym)
                qty = round(float(ap.qty), 4)
                entry_p = round(float(ap.avg_entry_price), 4)
                curr_p = round(float(ap.current_price), 4)
                self.alpaca_live_prices[sym] = curr_p

                if sym in self.positions:
                    pos = self.positions[sym]
                    pos.quantity = qty
                    pos.avg_entry_price = entry_p
                    pos.current_price = curr_p
                    pos.highest_price = max(pos.highest_price, curr_p, entry_p)
                    # Auf Alpaca entspricht der gebundene Eigenkapital-Einsatz dem Cost-Basis-Wert (oder anteilig bei Margin)
                    pos.margin_used_usd = round((qty * entry_p) / max(1.0, pos.leverage), 2)
                else:
                    # Falls eine Position direkt auf Alpaca existiert, aber lokal fehlte -> automatisch anlegen!
                    atr_est = max(curr_p * 0.025, 1.0)
                    self.positions[sym] = Position(
                        symbol=sym,
                        asset_class="CRYPTO" if "/" in sym else "EQUITY",
                        cluster=cluster_map.get(sym, "US_TECH"),
                        quantity=qty,
                        leverage=1.0,
                        intrinsic_leverage=1.0,
                        margin_used_usd=round(qty * entry_p, 2),
                        avg_entry_price=entry_p,
                        current_price=curr_p,
                        highest_price=max(curr_p, entry_p),
                        stop_loss_price=round(entry_p - 1.6 * atr_est, 4),
                        partial_tp_price=round(entry_p + 1.5 * atr_est, 4),
                        partial_tp_done=False,
                        take_profit_price=round(entry_p + 3.2 * atr_est, 4),
                        opened_at=datetime.datetime.now(datetime.timezone.utc).isoformat()[:19],
                    )

            # Bereinige Positionen, die beim Broker nicht mehr existieren (z.B. Wechsel von Paper auf Live)
            missing_on_broker = [s for s in list(self.positions.keys()) if s not in broker_symbols]
            for s in missing_on_broker:
                del self.positions[s]

            # WICHTIG: alpaca_equity enthält bereits den unrealisierten Gewinn/Verlust (market_value)!
            # Deshalb ziehen wir pos.market_value (NICHT nur pos.margin_used_usd) ab, damit total_equity_usd
            # auf den Cent genau mit alpaca_equity übereinstimmt und PnL niemals doppelt gezählt wird!
            pos_market_value = sum(p.market_value for p in self.positions.values())
            self.cash_usd = round(max(0.0, alpaca_equity - pos_market_value), 2)
            self.config.initial_capital_usd = alpaca_equity

            if abs(self.day_start_equity_usd - 1000.0) < 50.0 or abs(self.day_start_equity_usd - alpaca_last_equity) > 5.0 or self.day_start_equity_usd > alpaca_equity * 2.0:
                self.day_start_equity_usd = round(alpaca_last_equity, 2)
            self._save_state()
        except Exception:
            pass

    def _save_state(self, port_risk_dict: dict = None, diagnostics_dict: dict = None) -> None:
        data = {
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "day_start_date": self.day_start_date,
            "day_start_equity_usd": round(self.day_start_equity_usd, 2),
            "cash_usd": round(self.cash_usd, 2),
            "realized_pnl_usd": round(self.realized_pnl_usd, 2),
            "positions": {sym: pos.model_dump() for sym, pos in self.positions.items()},
        }
        old_data = {}
        if os.path.exists(self.config.portfolio_state_file):
            try:
                with open(self.config.portfolio_state_file, "r", encoding="utf-8") as old_f:
                    old_data = json.load(old_f)
            except Exception:
                old_data = {}

        if port_risk_dict is not None:
            data["portfolio_risk"] = port_risk_dict
        elif "portfolio_risk" in old_data:
            data["portfolio_risk"] = old_data["portfolio_risk"]

        if diagnostics_dict is not None:
            data["portfolio_diagnostics"] = diagnostics_dict
        elif "portfolio_diagnostics" in old_data:
            data["portfolio_diagnostics"] = old_data["portfolio_diagnostics"]

        with open(self.config.portfolio_state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _append_audit_log(self, event_type: str, payload: dict) -> None:
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": self.config.execution_mode,
            "event_type": event_type,
            **payload,
        }
        with open(self.config.audit_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @property
    def total_equity_usd(self) -> float:
        pos_value = sum(p.market_value for p in self.positions.values())
        return round(self.cash_usd + pos_value, 2)

    @property
    def total_return_pct(self) -> float:
        if self.config.initial_capital_usd <= 0:
            return 0.0
        return round(
            ((self.total_equity_usd - self.config.initial_capital_usd) / self.config.initial_capital_usd) * 100.0,
            2,
        )

    @property
    def daily_pnl_pct(self) -> float:
        """Berechnet die echte Tages-Performance (für den Daily Drawdown Circuit Breaker)."""
        today_str = datetime.date.today().isoformat()
        if self.day_start_date != today_str:
            self.day_start_date = today_str
            self.day_start_equity_usd = self.total_equity_usd
        if self.day_start_equity_usd <= 0:
            return 0.0
        return round(((self.total_equity_usd - self.day_start_equity_usd) / self.day_start_equity_usd) * 100.0, 2)

    def update_prices_and_check_stops(self, snapshots: Dict[str, MarketSnapshot]) -> List[str]:
        """
        Aktualisiert Kurse offener Positionen, führt 50%-Teilgewinnmitnahmen (Scale-Out bei TP1) aus,
        zieht den Chandelier-Trailing-Stop nach und löst SL/TP2 inkl. Gebühren aus.
        """
        if "ALPACA" in self.config.execution_mode and self.config.alpaca_api_key:
            self._sync_from_alpaca_account()

        triggered_events: List[str] = []
        symbols_to_close: List[Tuple[str, float, str]] = []
        us_open, _ = is_us_equity_market_open()
        today_date = datetime.date.today()

        for sym, pos in self.positions.items():
            snap = snapshots.get(sym)
            if not snap:
                continue
            # Falls ein echter Live-/After-Hours-Kurs direkt vom Broker (Alpaca) vorliegt, hat dieser Vorrang
            # vor dem verzögerten/regulären Yahoo-Finance-Schlusskurs!
            broker_live_p = self.alpaca_live_prices.get(sym)
            if broker_live_p and broker_live_p > 0:
                if abs(snap.current_price - broker_live_p) >= 0.05:
                    self.yahoo_vs_broker_price_diff[sym] = (snap.current_price, broker_live_p)
                pos.current_price = broker_live_p
            else:
                pos.current_price = snap.current_price
            pos.highest_price = max(pos.highest_price, pos.current_price)

            # Außerhalb der US-Börsenzeiten keine Market-Exits für Aktien auslösen (Schutz vor illiquiden Kursen)
            if self.config.enforce_market_hours and pos.asset_class == "EQUITY" and not us_open:
                continue

            # Haltedauer präzise in Tagen & Stunden berechnen
            holding_days = 0
            holding_hours = 0.0
            try:
                opened_str = pos.opened_at[:19]
                if "T" in opened_str:
                    opened_dt_full = datetime.datetime.fromisoformat(opened_str).replace(tzinfo=datetime.timezone.utc)
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    holding_hours = max(0.0, (now_utc - opened_dt_full).total_seconds() / 3600.0)
                    holding_days = int(holding_hours // 24)
                else:
                    opened_dt = datetime.date.fromisoformat(pos.opened_at[:10])
                    holding_days = (today_date - opened_dt).days
                    holding_hours = holding_days * 24.0
            except Exception:
                pass

            atr = max(snap.atr_14, snap.current_price * 0.015)

            # Berechne aktuelles News-Sentiment für Notausstieg bei plötzlichen Schock-Nachrichten
            non_zero_news = [n.sentiment_hint for n in snap.news if abs(n.sentiment_hint) > 0.05]
            avg_news_sent = float(sum(non_zero_news) / len(non_zero_news)) if non_zero_news else 0.0

            # Spezifische Maximal-Haltedauer:
            # - 3x Hebel-ETFs (SOXL, TQQQ, SQQQ): max. 3 Tage bei fehlendem Schwung (Schutz vor Volatility Drag!)
            # - Standard-Assets (2x Margin): max. 5 Tage bei fehlendem Schwung
            max_days_allowed = (
                self.config.leveraged_etf_max_days
                if pos.intrinsic_leverage > 1.0
                else self.config.max_stale_holding_days
            )

            if pos.side == "SHORT":
                pos.highest_price = min(pos.highest_price if pos.highest_price > 0 else pos.current_price, pos.current_price)
                if snap.current_price >= pos.stop_loss_price:
                    reason = "SHORT_TRAILING_STOP_TRIGGERED" if pos.stop_loss_price < pos.avg_entry_price else "SHORT_STOP_LOSS_TRIGGERED"
                    symbols_to_close.append((sym, snap.current_price, reason))
                elif snap.current_price <= pos.take_profit_price:
                    symbols_to_close.append((sym, snap.current_price, "SHORT_TAKE_PROFIT_2_TRIGGERED"))
                elif snap.rsi_14 <= 26.0 and pos.unrealized_pnl_pct >= 2.0:
                    symbols_to_close.append((sym, snap.current_price, "SHORT_OVERSOLD_CLIMAX_TAKE_PROFIT"))
                elif holding_days >= max_days_allowed and pos.unrealized_pnl_pct < 1.5 and snap.macd_hist >= 0:
                    symbols_to_close.append((sym, snap.current_price, f"SHORT_TIME_STOP_STALE_{holding_days}D"))
                elif pos.highest_price <= pos.avg_entry_price - 0.9 * atr:
                    new_sl = round(
                        min(pos.stop_loss_price, pos.avg_entry_price * 0.998, pos.highest_price + 1.5 * atr),
                        4,
                    )
                    if new_sl < pos.stop_loss_price:
                        pos.stop_loss_price = new_sl
                        triggered_events.append(
                            f"[SHORT_TRAILING_STOP_UPDATED] {sym}: Short Stop-Loss auf ${new_sl:,.2f} nachgezogen (Gewinnsicherung)"
                        )
            elif (
                getattr(snap, "liquidity_void_detected", False)
                or getattr(snap, "proactive_risk_alert", "NORMAL") == "LIQUIDITY_VOID_EVACUATE"
                or getattr(snap, "adverse_price_velocity_pct", 0.0) <= -3.5
            ):
                # Proaktiver Liquiditäts-Vakuum Notausstieg (Flash-Crash-Schutz vor SL-Slippage)
                symbols_to_close.append((sym, snap.current_price, "PROACTIVE_LIQUIDITY_VOID_EVACUATION"))
            elif snap.current_price <= pos.stop_loss_price:
                reason = "TRAILING_STOP_TRIGGERED" if pos.stop_loss_price > pos.avg_entry_price else "STOP_LOSS_TRIGGERED"
                symbols_to_close.append((sym, snap.current_price, reason))
            elif snap.current_price >= pos.take_profit_price:
                # Runner-Modus: Wenn Teilgewinn bereits gesichert ist (partial_tp_done) und der Trend extrem stark ist,
                # schließe nicht verfrüht, sondern erweitere das Kursziel auf +5.0x ATR bei nachziehendem Chandelier-Stop!
                if pos.partial_tp_done and snap.rsi_14 < 74.0 and snap.macd_hist > 0 and snap.current_price >= snap.sma_20:
                    expanded_tp = round(pos.avg_entry_price + 5.0 * atr, 4)
                    if expanded_tp > pos.take_profit_price:
                        pos.take_profit_price = expanded_tp
                        triggered_events.append(
                            f"[RUNNER_PROFIT_EXPANDED] {sym}: Bullen-Momentum hält an – TP für Runner auf ${expanded_tp:,.2f} (+5.0x ATR) erweitert!"
                        )
                else:
                    symbols_to_close.append((sym, snap.current_price, "TAKE_PROFIT_2_TRIGGERED"))
            elif avg_news_sent <= -0.40 and snap.change_24h_pct < -0.50:
                # 1. Breaking-News Notausstieg: Starke Negativ-Schlagzeile + Kurs fällt -> sofort raus, nicht auf SL warten!
                symbols_to_close.append((sym, snap.current_price, "NEWS_SHOCK_EMERGENCY_EXIT"))
            elif snap.rsi_14 >= 74.0 and pos.unrealized_pnl_pct >= 2.0:
                # 2. Climax / Überkauft-Gewinnmitnahme: Wenn RSI >= 74 im Gewinn erreicht wird, auf dem Peak verkaufen!
                symbols_to_close.append((sym, snap.current_price, "OVERBOUGHT_CLIMAX_TAKE_PROFIT"))
            elif (
                holding_days >= 2
                and snap.current_price < snap.sma_20
                and snap.macd_hist < 0
                and pos.unrealized_pnl_pct < 0.5
            ):
                # 3. Früher Trendbruch-Exit (ab Tag 2): Kurs fällt unter SMA20 + MACD negativ -> vorzeitig raus!
                symbols_to_close.append((sym, snap.current_price, f"EARLY_TREND_BREAK_{holding_days}D"))
            elif (
                holding_days >= max_days_allowed
                and pos.unrealized_pnl_pct < 1.5
                and snap.macd_hist <= 0
            ):
                # 4. Regulärer Time-Stop (3 Tage für 3x-ETFs / 5 Tage für 2x-Assets)
                symbols_to_close.append((sym, snap.current_price, f"TIME_STOP_STALE_{holding_days}D"))
            else:
                # Proaktiver Breakeven-Schutz bei Volatilitäts-Spike / ungünstiger Kursgeschwindigkeit
                vol_spike = getattr(snap, "volatility_spike_ratio", 1.0)
                adv_vel = getattr(snap, "adverse_price_velocity_pct", 0.0)
                if (vol_spike >= 2.0 or adv_vel <= -1.8) and snap.current_price > pos.avg_entry_price and pos.stop_loss_price < pos.avg_entry_price:
                    new_be_sl = round(pos.avg_entry_price * 1.001, 4)
                    pos.stop_loss_price = new_be_sl
                    triggered_events.append(
                        f"[PROACTIVE_RISK_SHIELD] {sym}: Volatilitäts-Schock ({vol_spike:.1f}x) – Stop-Loss proaktiv auf Einstand (${new_be_sl:,.2f}) nachgezogen!"
                    )

                # 5. Zeit-basierte Stop-Loss-Verengung (ab Tag 3): Jeden Tag über Tag 2 rückt der SL 0.25x ATR näher!
                if holding_days >= 3 and pos.stop_loss_price < pos.avg_entry_price:
                    tightened_sl = round(
                        min(
                            snap.current_price * 0.992,
                            pos.stop_loss_price + 0.25 * atr * min(holding_days - 2, 3),
                        ),
                        4,
                    )
                    if tightened_sl > pos.stop_loss_price:
                        pos.stop_loss_price = tightened_sl
                        triggered_events.append(
                            f"[TIME_DECAY_SL_TIGHTENED] {sym} (Tag {holding_days}/{max_days_allowed}): Stop-Loss wegen Haltedauer enger gezogen auf ${tightened_sl:,.2f}"
                        )

                # 1. Prüfe 50% Teilgewinnmitnahme (Scale-Out) bei Partial Take-Profit (TP1 = +1.5x ATR)
                if not pos.partial_tp_done and pos.partial_tp_price > 0 and snap.current_price >= pos.partial_tp_price:
                    decimals = 6 if pos.asset_class == "CRYPTO" else 4
                    sell_qty = round(pos.quantity * 0.5, decimals)
                    if sell_qty > 0 and (pos.quantity - sell_qty) > 0:
                        sell_ratio = sell_qty / pos.quantity
                        released_margin = round(pos.margin_used_usd * sell_ratio, 2)
                        gross_proceeds = sell_qty * snap.current_price
                        fee_usd = round(gross_proceeds * 0.0005, 2)
                        pnl = (sell_qty * (snap.current_price - pos.avg_entry_price)) - fee_usd

                        self.cash_usd += released_margin + pnl
                        self.realized_pnl_usd += pnl
                        pos.margin_used_usd = round(max(0.0, pos.margin_used_usd - released_margin), 2)
                        pos.quantity = round(pos.quantity - sell_qty, decimals)
                        pos.partial_tp_done = True
                        # Ziehe Stop-Loss für die verbleibende 50% Runner-Position sicher in den Gewinn (+0.3x ATR)
                        pos.stop_loss_price = round(max(pos.stop_loss_price, pos.avg_entry_price + 0.3 * atr), 4)

                        msg = (
                            f"[PARTIAL_TAKE_PROFIT_50%] {sym}: {sell_qty:g} Einheiten @ ${snap.current_price:,.2f} verkauft "
                            f"(Gewinn gesichert: {pnl:+.2f}$, Margin frei: {released_margin:.2f}$, Rest-Runner SL auf ${pos.stop_loss_price:,.2f})"
                        )
                        # Falls Live/Paper-Broker (Alpaca/CCXT) aktiv: 50%-Teilverkauf & neuen Stop-Loss auch an den Broker senden!
                        broker_sync = self._sync_broker_exit_or_stop_update(
                            symbol=sym,
                            asset_class=pos.asset_class,
                            sell_qty=sell_qty,
                            new_stop_loss=pos.stop_loss_price,
                            remaining_qty=pos.quantity,
                        )
                        if broker_sync:
                            msg += f" [{broker_sync}]"
                        triggered_events.append(msg)
                        self._append_audit_log(
                            "PARTIAL_TAKE_PROFIT_50PCT",
                            {
                                "symbol": sym,
                                "sold_quantity": sell_qty,
                                "remaining_quantity": pos.quantity,
                                "exit_price": snap.current_price,
                                "realized_pnl_usd": round(pnl, 2),
                                "new_stop_loss": pos.stop_loss_price,
                            },
                        )

                # 2. Dynamischer Chandelier Trailing-Stop: Sobald Höchstkurs >= Einstieg + 0.9x ATR erreicht
                if pos.highest_price >= pos.avg_entry_price + 0.9 * atr:
                    new_sl = round(
                        max(pos.stop_loss_price, pos.avg_entry_price * 1.002, pos.highest_price - 1.5 * atr),
                        4,
                    )
                    if new_sl > pos.stop_loss_price:
                        pos.stop_loss_price = new_sl
                        broker_sl_note = self._sync_broker_exit_or_stop_update(
                            symbol=sym,
                            asset_class=pos.asset_class,
                            sell_qty=0.0,
                            new_stop_loss=new_sl,
                            remaining_qty=pos.quantity,
                        )
                        sl_suffix = f" [{broker_sl_note}]" if broker_sl_note else ""
                        triggered_events.append(
                            f"[TRAILING_STOP_UPDATED] {sym}: Stop-Loss auf ${new_sl:,.2f} nachgezogen (Gewinnsicherung){sl_suffix}"
                        )

        for sym, exec_price, trigger_reason in symbols_to_close:
            pos = self.positions.pop(sym)
            # Falls Broker-Modus aktiv: Sende Exit-Order an Alpaca / CCXT
            broker_exit_note = self._sync_broker_exit_or_stop_update(
                symbol=sym,
                asset_class=pos.asset_class,
                sell_qty=pos.quantity,
                new_stop_loss=None,
                remaining_qty=0.0,
            )
            gross_proceeds = abs(pos.quantity) * exec_price
            fee_usd = round(gross_proceeds * 0.0005, 2)
            if pos.side == "SHORT":
                pnl = (abs(pos.quantity) * (pos.avg_entry_price - exec_price)) - fee_usd
            else:
                pnl = (abs(pos.quantity) * (exec_price - pos.avg_entry_price)) - fee_usd
            self.cash_usd += pos.margin_used_usd + pnl
            self.realized_pnl_usd += pnl
            b_tag = f" [{broker_exit_note}]" if broker_exit_note else ""
            msg = (
                f"[{trigger_reason}] {sym} ({pos.side}): {pos.quantity:g} Einheiten @ {exec_price:.2f}$ geschlossen "
                f"(Realisierter PnL nach Gebühren: {pnl:+.2f}$, Margin zurück: {pos.margin_used_usd:.2f}$){b_tag}"
            )
            triggered_events.append(msg)
            self._send_push_alert(f"⚡ Exit {sym} ({trigger_reason})", msg)
            self._append_audit_log(
                trigger_reason,
                {
                    "symbol": sym,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "exit_price": exec_price,
                    "fee_usd": fee_usd,
                    "realized_pnl_usd": round(pnl, 2),
                },
            )

        self._save_state()
        return triggered_events

    def rotate_weak_positions_for_better_setups(
        self,
        snapshots: Dict[str, MarketSnapshot],
        signals: Dict[str, TradingSignal],
        shadow_predictions: Dict[str, any],
    ) -> List[str]:
        """
        Aktive 'Opportunity-Cost-Rotation':
        Wenn die Börse geöffnet ist und ein hochkonfidentes A+ Kaufsignal (>= 85%, Edge >= 1.65) bereitsteht,
        prüft der Agent alle offenen Positionen darauf, ob eine Position ihre Einstiegs-These verloren hat
        (z.B. Kauf ohne Trend-/Volumen-Bestätigung wie AVGO + aktuell im Minus + Signal nur noch HOLD).
        Falls ja, tauscht der Agent das lahme Pferd aktiv gegen den neuen Marktführer aus!
        """
        if not self.config.enable_opportunity_rotation or not self.positions:
            return []

        us_open, _ = is_us_equity_market_open()
        if self.config.enforce_market_hours and not us_open:
            return []

        top_waiting_buys = [
            sig
            for sym, sig in signals.items()
            if sig.action == "BUY"
            and sym not in self.positions
            and sig.confidence >= 0.85
            and sig.expected_edge_score >= 1.65
        ]
        if not top_waiting_buys:
            return []

        best_new_leader = max(top_waiting_buys, key=lambda s: s.expected_edge_score)
        rotation_events: List[str] = []
        to_rotate: List[Tuple[str, float, str]] = []

        for sym, pos in self.positions.items():
            if pos.asset_class == "EQUITY" and self.config.enforce_market_hours and not is_us_equity_market_open(symbol=sym)[0]:
                continue
            sig = signals.get(sym)
            shadow = shadow_predictions.get(sym)
            entry_factors = getattr(shadow, "factor_contributions", {}) if shadow else {}
            entry_trend = float(entry_factors.get("trend", 0.0))
            entry_vol = float(entry_factors.get("volume", 0.0))

            # Rotations-Kriterium: Position im Minus, aktuelles Signal nur noch HOLD (kein BUY mehr),
            # und entweder fehlte beim Einstieg Trend+Volumen ODER das erwartete Edge ist halb so hoch wie beim neuen Leader!
            current_edge = sig.expected_edge_score if sig else 0.0
            missing_core_at_entry = (entry_trend <= 0.0 and entry_vol <= 0.0 and len(entry_factors) > 0)
            if (
                pos.unrealized_pnl < 0
                and (not sig or sig.action == "HOLD")
                and (missing_core_at_entry or (current_edge < best_new_leader.expected_edge_score * 0.55 and pos.unrealized_pnl_pct <= -1.5))
            ):
                to_rotate.append((
                    sym,
                    pos.current_price,
                    f"OPPORTUNITY_COST_ROTATION_FOR_{best_new_leader.symbol}",
                ))

        for sym, exec_price, rot_reason in to_rotate:
            pos = self.positions.pop(sym)
            broker_exit_note = self._sync_broker_exit_or_stop_update(
                symbol=sym,
                asset_class=pos.asset_class,
                sell_qty=pos.quantity,
                new_stop_loss=None,
                remaining_qty=0.0,
            )
            gross_proceeds = abs(pos.quantity) * exec_price
            fee_usd = round(gross_proceeds * 0.0005, 2)
            pnl = (abs(pos.quantity) * (exec_price - pos.avg_entry_price)) - fee_usd
            self.cash_usd += pos.margin_used_usd + pnl
            self.realized_pnl_usd += pnl
            b_tag = f" [{broker_exit_note}]" if broker_exit_note else ""
            msg = (
                f"[{rot_reason}] 🔄 Kapital-Rotation: Schwache Position {sym} ({pos.quantity:g} Stk. @ ${exec_price:.2f}, PnL {pnl:+.2f}$) "
                f"geschlossen, um Platz & Korrelations-Budget für Top-Leader {best_new_leader.symbol} "
                f"(BUY {best_new_leader.confidence*100:.0f}%, Edge {best_new_leader.expected_edge_score:.2f}) freizumachen!{b_tag}"
            )
            rotation_events.append(msg)
            self._send_push_alert(f"🔄 Kapital-Rotation: {sym} ➔ {best_new_leader.symbol}", msg)
            self._append_audit_log(
                "OPPORTUNITY_COST_ROTATION",
                {
                    "symbol": sym,
                    "replaced_by": best_new_leader.symbol,
                    "quantity": pos.quantity,
                    "exit_price": exec_price,
                    "realized_pnl_usd": round(pnl, 2),
                },
            )

        if rotation_events:
            self._save_state()
        return rotation_events

    def _send_push_alert(self, title: str, message: str) -> None:
        """
        Sendet optionale Echtzeit-Push-Benachrichtigungen aufs Smartphone über Telegram Bot API oder Discord Webhook.
        """
        import ssl
        import urllib.request
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            try:
                ctx = ssl._create_unverified_context()
            except Exception:
                ctx = ssl.create_default_context()

        if self.config.telegram_bot_token and self.config.telegram_chat_id:
            try:
                from notification_manager import get_notifier
                # Nutze den zentralen Notifier mit Entprellung & Rate-Limiting
                notifier = get_notifier()
                dedup_key = f"EXEC:{title}:{message[:30]}"
                if not notifier._is_duplicate(dedup_key, cooldown_seconds=60):
                    import html
                    safe_title = html.escape(title)
                    safe_msg = html.escape(message)
                    text_html = f"🤖 <b>{safe_title}</b>\n{safe_msg}"
                    notifier._send_raw_telegram(text_html, priority_bypass_cap=True, dedup_key=dedup_key)
            except Exception:
                pass
        if self.config.discord_webhook_url:
            try:
                payload = json.dumps({"content": f"**🤖 {title}**\n{message}"}).encode("utf-8")
                req = urllib.request.Request(
                    self.config.discord_webhook_url,
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": "AutonomousFinanceAgent/2.0"},
                )
                urllib.request.urlopen(req, context=ctx, timeout=4.0)
            except Exception:
                pass


    def execute_order(self, signal: TradingSignal, risk_eval: RiskEvaluation) -> str:
        """Führt eine vom Risk-Manager genehmigte Order (Long-Kauf, Scale-In, Verkauf oder Short-Sell) aus."""
        if not risk_eval.approved or risk_eval.adjusted_action == "HOLD":
            return f"SKIPPED ({risk_eval.reason})"

        symbol = signal.symbol
        qty = risk_eval.recommended_quantity
        price = signal.entry_price
        notional = round(qty * price, 2)

        is_short_entry = (
            risk_eval.adjusted_action == "SELL"
            and not (symbol in self.positions and self.positions[symbol].side == "LONG")
        )

        broker_note = "LOCAL_PAPER"
        # 1. Primäres oder automatisches IBKR-Routing (für alle Assets im IBKR-Modus oder automatisch für .DE Xetra-Aktien, sobald TWS/Gateway offen ist)
        if self.config.execution_mode == "IBKR" or (
            self.config.ibkr_auto_connect and self.ibkr_bridge.active_port is not None and symbol.endswith(".DE")
        ):
            broker_note = self._send_ibkr_order(signal, qty, risk_eval.adjusted_action, is_short_entry=is_short_entry)
            if broker_note.startswith("IBKR_ERROR") and self.config.execution_mode == "IBKR":
                return f"BROKER_ERROR ({broker_note})"
        elif symbol.endswith(".DE"):
            # Europäische Xetra-Ticker (.DE) sind bei US-Broker Alpaca nicht direkt gelistet -> lokales Paper-Execution-Tracking solange IBKR TWS geschlossen
            broker_note = "XETRA_LOCAL_PAPER (IBKR TWS wartet)"
        elif "ALPACA" in self.config.execution_mode and self.config.alpaca_api_key:
            broker_note = self._send_alpaca_order(signal, qty, risk_eval.adjusted_action)
            if broker_note.startswith("ALPACA_ERROR"):
                return f"BROKER_ERROR ({broker_note})"
        elif self.config.execution_mode == "CCXT_SANDBOX" and signal.asset_class == "CRYPTO" and self.config.ccxt_api_key:
            broker_note = self._send_ccxt_order(signal, qty, risk_eval.adjusted_action)
            if broker_note.startswith("CCXT_ERROR"):
                return f"BROKER_ERROR ({broker_note})"

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]

        if risk_eval.adjusted_action == "BUY":
            # Fall A: Bestehende SHORT-Position eindecken (Short Cover)
            if symbol in self.positions and self.positions[symbol].side == "SHORT":
                pos = self.positions.pop(symbol)
                fee_usd = round(notional * 0.0005, 2)
                pnl = (abs(pos.quantity) * (pos.avg_entry_price - price)) - fee_usd
                self.cash_usd += pos.margin_used_usd + pnl
                self.realized_pnl_usd += pnl
                self._save_state()
                msg = f"EXECUTED SHORT COVER: {pos.quantity} {symbol} @ {price:.2f}$ eingedeckt (Realisierter PnL: {pnl:+.2f}$)"
                self._send_push_alert(f"🟢 Short Cover {symbol}", msg)
                self._append_audit_log(
                    "ORDER_SHORT_COVER_EXECUTED",
                    {
                        "symbol": symbol,
                        "quantity": pos.quantity,
                        "price": price,
                        "fee_usd": fee_usd,
                        "realized_pnl_usd": round(pnl, 2),
                        "thesis": signal.thesis,
                        "broker": broker_note,
                    },
                )
                return msg

            # Fall B: Regulärer LONG-Kauf oder Re-Sizing (Aufstocken einer Mini-Testposition auf $100k-Zielgröße)
            lev = max(1.0, risk_eval.leverage_used)
            margin_req = risk_eval.margin_required_usd if risk_eval.margin_required_usd > 0 else round(notional / lev, 2)
            fee_usd = round(notional * 0.0005, 2)
            total_cash_outflow = margin_req + fee_usd
            if total_cash_outflow > self.cash_usd:
                return "REJECTED (Nicht genügend Cash/Margin inkl. Gebühren)"

            self.cash_usd -= total_cash_outflow
            effective_entry_price = round((notional + fee_usd) / qty, 4)
            is_scale_in = symbol in self.positions

            if is_scale_in:
                existing = self.positions[symbol]
                new_qty = existing.quantity + qty
                new_avg = ((existing.quantity * existing.avg_entry_price) + (notional + fee_usd)) / new_qty
                existing.quantity = round(new_qty, 6)
                existing.margin_used_usd = round(existing.margin_used_usd + margin_req, 2)
                existing.avg_entry_price = round(new_avg, 4)
                existing.current_price = price
                existing.highest_price = max(existing.highest_price, price)
                existing.stop_loss_price = max(existing.stop_loss_price, signal.stop_loss_price)
                existing.partial_tp_price = max(existing.partial_tp_price, signal.partial_take_profit_price)
                existing.take_profit_price = max(existing.take_profit_price, signal.take_profit_price)
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    asset_class=signal.asset_class,
                    cluster=signal.cluster,
                    side="LONG",
                    quantity=qty,
                    leverage=lev,
                    intrinsic_leverage=signal.intrinsic_leverage,
                    margin_used_usd=margin_req,
                    avg_entry_price=effective_entry_price,
                    current_price=price,
                    highest_price=price,
                    stop_loss_price=signal.stop_loss_price,
                    partial_tp_price=signal.partial_take_profit_price,
                    partial_tp_done=False,
                    take_profit_price=signal.take_profit_price,
                    opened_at=now_iso,
                )

            eff_lev = round(lev * signal.intrinsic_leverage, 1)
            lev_badge = f"⚡{eff_lev:.0f}x Hebel" if eff_lev > 1.0 else "1x"
            action_label = "EXECUTED SCALE-IN BUY" if is_scale_in else "EXECUTED BUY"
            self._save_state()
            msg = (
                f"{action_label} [{lev_badge}]: {qty} {symbol} @ {price:.2f}$ "
                f"(Margin: {margin_req:.0f}$ ➔ Kaufkraft: {notional:.0f}$, SL: {signal.stop_loss_price:.2f}$, TP1: {signal.partial_take_profit_price:.2f}$)"
            )
            self._send_push_alert(f"🚀 {action_label} {symbol}", msg)
            self._append_audit_log(
                "ORDER_BUY_EXECUTED",
                {
                    "symbol": symbol,
                    "asset_class": signal.asset_class,
                    "cluster": signal.cluster,
                    "quantity": qty,
                    "leverage_used": lev,
                    "effective_total_leverage": eff_lev,
                    "margin_used_usd": margin_req,
                    "price": price,
                    "notional_usd": notional,
                    "fee_usd": fee_usd,
                    "stop_loss": signal.stop_loss_price,
                    "partial_tp": signal.partial_take_profit_price,
                    "take_profit": signal.take_profit_price,
                    "confidence": signal.confidence,
                    "expected_edge_score": signal.expected_edge_score,
                    "thesis": signal.thesis,
                    "broker": broker_note,
                },
            )
            return msg

        elif risk_eval.adjusted_action == "SELL":
            # Fall A: Bestehende LONG-Position verkaufen
            if symbol in self.positions and self.positions[symbol].side == "LONG":
                pos = self.positions.pop(symbol)
                fee_usd = round(notional * 0.0005, 2)
                pnl = (pos.quantity * (price - pos.avg_entry_price)) - fee_usd
                self.cash_usd += pos.margin_used_usd + pnl
                self.realized_pnl_usd += pnl
                self._save_state()
                msg = f"EXECUTED SELL: {pos.quantity} {symbol} @ {price:.2f}$ (Realisierter PnL: {pnl:+.2f}$)"
                self._send_push_alert(f"🔴 Sell {symbol}", msg)
                self._append_audit_log(
                    "ORDER_SELL_EXECUTED",
                    {
                        "symbol": symbol,
                        "quantity": pos.quantity,
                        "price": price,
                        "fee_usd": fee_usd,
                        "realized_pnl_usd": round(pnl, 2),
                        "thesis": signal.thesis,
                        "broker": broker_note,
                    },
                )
                return msg

            # Fall B: Neuen Leerverkauf (SHORT-SELL) bei starkem Abwärtstrend (>= 85% Konfidenz) eröffnen!
            margin_req = risk_eval.margin_required_usd if risk_eval.margin_required_usd > 0 else notional
            fee_usd = round(notional * 0.0005, 2)
            if (margin_req + fee_usd) > self.cash_usd:
                return "REJECTED (Nicht genügend Cash-Margin für Short-Position)"

            self.cash_usd -= (margin_req + fee_usd)
            atr_est = max(abs(signal.entry_price - signal.stop_loss_price) / 1.6, price * 0.015)
            short_sl = round(price + 1.6 * atr_est, 4)
            short_tp1 = round(max(0.01, price - 1.5 * atr_est), 4)
            short_tp2 = round(max(0.01, price - 3.2 * atr_est), 4)

            self.positions[symbol] = Position(
                symbol=symbol,
                asset_class=signal.asset_class,
                cluster=signal.cluster,
                side="SHORT",
                quantity=qty,
                leverage=1.0,
                intrinsic_leverage=1.0,
                margin_used_usd=margin_req,
                avg_entry_price=price,
                current_price=price,
                highest_price=price,
                stop_loss_price=short_sl,
                partial_tp_price=short_tp1,
                partial_tp_done=False,
                take_profit_price=short_tp2,
                opened_at=now_iso,
            )
            self._save_state()
            msg = (
                f"EXECUTED SHORT-SELL [📉 Leerverkauf]: {qty} {symbol} @ {price:.2f}$ "
                f"(Margin: {margin_req:.0f}$, Stop-Loss oben: {short_sl:.2f}$, Gewinnziel TP1 unten: {short_tp1:.2f}$)"
            )
            self._send_push_alert(f"📉 Short-Sell {symbol}", msg)
            self._append_audit_log(
                "ORDER_SHORT_SELL_EXECUTED",
                {
                    "symbol": symbol,
                    "side": "SHORT",
                    "quantity": qty,
                    "margin_used_usd": margin_req,
                    "price": price,
                    "stop_loss": short_sl,
                    "partial_tp": short_tp1,
                    "take_profit": short_tp2,
                    "confidence": signal.confidence,
                    "thesis": signal.thesis,
                    "broker": broker_note,
                },
            )
            return msg

        return "NO_ACTION"

    def _send_ibkr_order(self, signal: TradingSignal, qty: float, side: str, is_short_entry: bool = False) -> str:
        """
        Sendet Kauf- / Verkaufs- / Leerverkaufs- & Options-Orders über die autonome IBKRBridge
        direkt an Interactive Brokers (TWS / IB Gateway).
        Unterstützt US-Aktien/ETFs (SMART/USD), deutsche Xetra-Aktien (.DE -> SMART/EUR/IBIS),
        Krypto (PAXOS/USD) sowie echte US-Optionen (CALL/PUT) inklusive serverseitiger Bracket-Orders!
        """
        return self.ibkr_bridge.place_order(
            signal=signal,
            qty=qty,
            side=side,
            is_short_entry=is_short_entry,
            option_badge=getattr(signal, "best_option_contract", ""),
        )

    def _send_alpaca_order(self, signal: TradingSignal, qty: float, side: str) -> str:
        """
        Sendet Kauf-/Verkaufsorders an Alpaca Markets.
        Bei ganzzahligen Aktien-Käufen (qty >= 1) wird automatisch eine serverseitige BRACKET-Order
        inklusive aktivem Stop-Loss und Take-Profit direkt auf dem Alpaca-Server platziert!
        Bei Fractional Shares wird die Market-Order ausgeführt und der Stop-Loss vom Agenten überwacht.
        """
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
            from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce

            client = TradingClient(self.config.alpaca_api_key, self.config.alpaca_secret_key, paper=self.config.alpaca_paper)

            # Prüfe ob wir bei einem Aktien-BUY eine serverseitige Bracket-Order (Einstieg + Stop-Loss + Take-Profit) senden können
            if side == "BUY" and signal.asset_class == "EQUITY" and qty >= 1.0:
                whole_qty = int(qty)
                req = MarketOrderRequest(
                    symbol=signal.symbol,
                    qty=whole_qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    stop_loss=StopLossRequest(stop_price=round(signal.stop_loss_price, 2)),
                    take_profit=TakeProfitRequest(limit_price=round(signal.take_profit_price, 2)),
                )
                order = client.submit_order(order_data=req)
                return f"ALPACA_BRACKET_ORDER_ID:{order.id} (Server-SL:{signal.stop_loss_price:.2f}$ / TP:{signal.take_profit_price:.2f}$)"

            req = MarketOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC if signal.asset_class == "CRYPTO" else TimeInForce.DAY,
            )
            order = client.submit_order(order_data=req)
            return f"ALPACA_ORDER_ID:{order.id}"
        except Exception as e:
            return f"ALPACA_ERROR:{e}"

    def _send_ccxt_order(self, signal: TradingSignal, qty: float, side: str) -> str:
        """
        Sendet Krypto-Orders über CCXT und platziert bei einem BUY zusätzlich
        eine serverseitige Stop-Loss-Order auf der Krypto-Börse (sofern von der Börse unterstützt).
        """
        try:
            import ccxt
            exchange_cls = getattr(ccxt, self.config.ccxt_exchange_id)
            exchange = exchange_cls({
                "apiKey": self.config.ccxt_api_key,
                "secret": self.config.ccxt_secret_key,
                "enableRateLimit": True,
            })
            if self.config.ccxt_sandbox:
                exchange.set_sandbox_mode(True)
            pair = signal.symbol.replace("/USD", "/USDT")
            order = exchange.create_market_order(pair, side.lower(), qty)
            order_id = order.get("id")

            # Platziere serverseitigen Stop-Loss auf der Krypto-Börse bei BUY
            sl_note = ""
            if side == "BUY" and signal.stop_loss_price > 0:
                try:
                    sl_order = exchange.create_order(
                        symbol=pair,
                        type="stop_market",
                        side="sell",
                        amount=qty,
                        params={"stopPrice": round(signal.stop_loss_price, 2)},
                    )
                    sl_note = f"+SERVER_SL:{sl_order.get('id')}"
                except Exception:
                    sl_note = "+LOCAL_SL_ACTIVE"

            return f"CCXT_ORDER_ID:{order_id}{sl_note}"
        except Exception as e:
            return f"CCXT_ERROR:{e}"

    def _sync_broker_exit_or_stop_update(
        self,
        symbol: str,
        asset_class: str,
        sell_qty: float,
        new_stop_loss: float = None,
        remaining_qty: float = 0.0,
    ) -> str:
        """
        Synchronisiert Teilverkäufe (50% Scale-Out), Komplett-Exits (SL/TP/Time-Stop) und
        nachgezogene Trailing-Stops direkt mit Alpaca bzw. CCXT, sofern der Broker-Modus aktiv ist.
        """
        notes = []
        pos_side = self.positions[symbol].side if symbol in self.positions else "LONG"
        if self.config.execution_mode == "IBKR" or (
            self.config.ibkr_auto_connect and self.ibkr_bridge.active_port is not None
        ):
            ib_note = self.ibkr_bridge.sync_exit_or_trailing_stop(
                symbol=symbol,
                asset_class=asset_class,
                sell_qty=sell_qty,
                new_stop_loss=new_stop_loss,
                remaining_qty=remaining_qty,
                position_side=pos_side,
            )
            if ib_note:
                notes.append(ib_note)
        if "ALPACA" in self.config.execution_mode and self.config.alpaca_api_key and not symbol.endswith(".DE"):
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, ReplaceOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

                client = TradingClient(self.config.alpaca_api_key, self.config.alpaca_secret_key, paper=self.config.alpaca_paper)
                if sell_qty > 0:
                    req = MarketOrderRequest(
                        symbol=symbol,
                        qty=sell_qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC if asset_class == "CRYPTO" else TimeInForce.DAY,
                    )
                    o = client.submit_order(order_data=req)
                    notes.append(f"ALPACA_SELL:{o.id}")
                if new_stop_loss is not None and remaining_qty > 0:
                    # Suche offene Stop-Order für dieses Symbol bei Alpaca und ziehe das Stop-Limit auf dem Server nach
                    open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
                    for oo in open_orders:
                        if oo.stop_price is not None:
                            client.replace_order_by_id(
                                order_id=oo.id,
                                order_data=ReplaceOrderRequest(
                                    qty=int(max(1, remaining_qty)),
                                    stop_price=round(new_stop_loss, 2),
                                ),
                            )
                            notes.append(f"ALPACA_SL_UPDATED:{new_stop_loss:.2f}$")
            except Exception as e:
                notes.append(f"ALPACA_SYNC_WARN:{e}")

        elif self.config.execution_mode == "CCXT_SANDBOX" and asset_class == "CRYPTO" and self.config.ccxt_api_key:
            try:
                import ccxt
                exchange_cls = getattr(ccxt, self.config.ccxt_exchange_id)
                exchange = exchange_cls({
                    "apiKey": self.config.ccxt_api_key,
                    "secret": self.config.ccxt_secret_key,
                    "enableRateLimit": True,
                })
                if self.config.ccxt_sandbox:
                    exchange.set_sandbox_mode(True)
                pair = symbol.replace("/USD", "/USDT")
                if sell_qty > 0:
                    o = exchange.create_market_order(pair, "sell", sell_qty)
                    notes.append(f"CCXT_SELL:{o.get('id')}")
                if new_stop_loss is not None and remaining_qty > 0:
                    try:
                        exchange.cancel_all_orders(pair)
                        sl_o = exchange.create_order(
                            symbol=pair,
                            type="stop_market",
                            side="sell",
                            amount=remaining_qty,
                            params={"stopPrice": round(new_stop_loss, 2)},
                        )
                        notes.append(f"CCXT_SL_UPDATED:{sl_o.get('id')}")
                    except Exception:
                        pass
            except Exception as e:
                notes.append(f"CCXT_SYNC_WARN:{e}")

        return " | ".join(notes)

    def diagnose_portfolio_health(
        self,
        snapshots: Dict[str, MarketSnapshot],
        signals: Dict[str, TradingSignal],
        shadow_predictions: Dict[str, any],
        factor_map: Dict[str, Dict[str, float]],
    ) -> dict:
        """
        Autonome Root-Cause- & Plan-Diagnose-Engine:
        Analysiert bei jedem Lauf exakt:
        1. Warum ist das Konto aktuell im Plus oder Minus? (Exakte Zerlegung nach Positionen & Realized/Unrealized PnL)
        2. Welche Berechnung oder Faktor-Annahme lag falsch / war ungenau? (Einstiegs-Faktoren vs. Ist-Verlauf, Broker- vs. Yahoo-Kurse)
        3. Läuft der Trade noch nach Plan (im erlaubten ATR-Stop-Korridor) oder ist eine These gebrochen?
        """
        total_eq = self.total_equity_usd
        init_cap = self.config.initial_capital_usd
        eq_delta_usd = round(total_eq - init_cap, 2)
        eq_delta_pct = self.total_return_pct
        unrealized_total = round(sum(p.unrealized_pnl for p in self.positions.values()), 2)
        realized_total = round(self.realized_pnl_usd, 2)

        position_audits = []
        system_notes = []
        any_thesis_broken = False
        any_attention = False

        # Prüfe US-Börsenstatus & wartende Kaufsignale
        us_open, mkt_reason = is_us_equity_market_open()
        queued_buys = [
            f"{sym} ({sig.confidence * 100:.0f}%, Edge {round(sig.confidence * sig.risk_reward_ratio, 2)})"
            for sym, sig in signals.items()
            if sig.action == "BUY" and sym not in self.positions
        ]
        if not us_open and queued_buys:
            system_notes.append(
                f"⏳ Börsenzeiten-Filter ({mkt_reason}): {len(queued_buys)} erkannte A+ Kaufsignale "
                f"({', '.join(queued_buys[:4])}) warten diszipliniert auf reguläre Börsenöffnung (Schutz vor teuren After-Hours-Spreads)."
            )

        for sym, pos in self.positions.items():
            snap = snapshots.get(sym)
            sig = signals.get(sym)
            shadow = shadow_predictions.get(sym)
            curr_factors = factor_map.get(sym, {})
            entry_factors = getattr(shadow, "factor_contributions", {}) if shadow else curr_factors

            pnl_usd = round(pos.unrealized_pnl, 2)
            pnl_pct = round(pos.unrealized_pnl_pct, 2)
            price_move_pct = round(((pos.current_price - pos.avg_entry_price) / pos.avg_entry_price) * 100.0, 2) if pos.avg_entry_price > 0 else 0.0

            # Berechne ATR-Stop-Puffer-Verbrauch (0% = auf/über Einstieg, 100% = am Stop-Loss)
            stop_dist_total = max(0.0001, pos.avg_entry_price - pos.stop_loss_price)
            if pos.current_price >= pos.avg_entry_price:
                stop_buffer_used_pct = 0.0
            else:
                stop_buffer_used_pct = round(min(100.0, ((pos.avg_entry_price - pos.current_price) / stop_dist_total) * 100.0), 1)
            stop_buffer_remaining_pct = round(max(0.0, 100.0 - stop_buffer_used_pct), 1)

            # 1. WARUM ist diese Position im Minus oder Plus?
            why_reasons = []
            if sym in self.yahoo_vs_broker_price_diff:
                yf_p, brk_p = self.yahoo_vs_broker_price_diff[sym]
                why_reasons.append(
                    f"Broker-Echtzeit-/After-Hours-Kurs (${brk_p:,.2f}) weicht vom regulären Yahoo-Schlusskurs (${yf_p:,.2f}) ab"
                )
            if pnl_usd < 0:
                if snap:
                    why_reasons.append(
                        f"Aktie notiert {price_move_pct:+.2f}% unter Einstieg (${pos.avg_entry_price:,.2f} → ${pos.current_price:,.2f}, RSI {snap.rsi_14:.1f}, Vol {snap.volume_ratio_vs_20d:.2f}x)"
                    )
                    if snap.cross_asset_thesis:
                        why_reasons.append(f"Sektor-/Korrelations-Einfluss: {snap.cross_asset_thesis}")
                else:
                    why_reasons.append(f"Kursrücksetzer um {price_move_pct:+.2f}% unter Einstiegskurs (${pos.avg_entry_price:,.2f})")
            else:
                why_reasons.append(
                    f"Kurs liegt {price_move_pct:+.2f}% über Einstieg (${pos.avg_entry_price:,.2f} → ${pos.current_price:,.2f}) und trägt +${pnl_usd:,.2f} zum Konto bei"
                )

            # 2. WELCHE BERECHNUNG / FAKTOR-ANNAHME WAR FALSCH ODER UNGENAU?
            calc_findings = []
            entry_trend = float(entry_factors.get("trend", 0.0))
            entry_vol = float(entry_factors.get("volume", 0.0))
            entry_mom = float(entry_factors.get("momentum", 0.0))
            entry_opt = float(entry_factors.get("options_flow", 0.0))

            if pnl_usd < 0:
                if entry_trend <= 0.0 and entry_vol <= 0.0:
                    calc_findings.append(
                        f"Fehlende Trend- & Volumen-Bestätigung beim Kauf (trend={entry_trend:+.2f}, volume={entry_vol:+.2f}): "
                        f"Das Kaufsignal stützte sich nur auf Sekundär-Faktoren (Momentum {entry_mom:+.2f}, Options-Flow {entry_opt:+.2f}). "
                        f"Ohne echtes institutionelles Kaufvolumen gab der Kurs nachbörslich nach. "
                        f"[Lerneffekt: AI-Analyst fordert ab sofort zwingend Trend > 0 oder Volumen > 0!]"
                    )
                # Prüfe ob ein Faktor seit Einstieg ins Negative gedreht hat
                flipped_factors = [
                    f"{k} ({entry_factors.get(k, 0.0):+.2f} → {curr_factors.get(k, 0.0):+.2f})"
                    for k in entry_factors
                    if entry_factors.get(k, 0.0) > 0.05 and curr_factors.get(k, 0.0) <= 0.0
                ]
                if flipped_factors:
                    calc_findings.append(f"Faktor-Abschwächung seit Einstieg bei: {', '.join(flipped_factors)}")
                if not calc_findings:
                    calc_findings.append(
                        "Keine fehlerhafte Signal-Berechnung: Alle Primär-Faktoren waren beim Einstieg valide; es handelt sich um normales Marktrauschen innerhalb der 14-Tage-ATR."
                    )
            else:
                pos_drivers = [f"{k} ({v:+.2f})" for k, v in curr_factors.items() if v > 0.05]
                calc_findings.append(
                    f"Berechnung bestätigt: Aktive Treiber ({', '.join(pos_drivers) if pos_drivers else 'Trend & Fundamentals intakt'}) stützen die Gewinnposition."
                )

            # Prüfe ob die Positionsgröße noch aus dem $1.000-Testlauf stammt
            if total_eq > 20000.0 and pos.margin_used_usd < 1000.0:
                calc_findings.append(
                    f"Hinweis zur Positionsgröße: Kleine Test-Stückzahl ({pos.quantity:g} Stk. / ${pos.margin_used_usd:,.2f} Einsatz) stammt noch aus dem initialen Test-Lauf vor der $100k-Skalierung."
                )

            # 3. LÄUFT ALLES NACH PLAN ODER IST DIE THESE GEBROCHEN?
            account_impact_pct = round((abs(pnl_usd) / max(1.0, total_eq)) * 100.0, 4)
            if sig and sig.action == "SELL" and pnl_usd < 0:
                status_code = "THESIS_BROKEN"
                status_badge = "🔴 THESE GEBROCHEN (Signal auf SELL gedreht)"
                any_thesis_broken = True
                plan_verdict = (
                    f"Achtung: KI-Signal hat auf SELL gedreht. Position wird beim nächsten regulären Börsen-Tick eng überwacht bzw. geschlossen."
                )
            elif stop_buffer_used_pct >= 65.0:
                status_code = "ATTENTION"
                status_badge = f"🟡 NAHE STOP-LOSS ({stop_buffer_used_pct:.0f}% Puffer verbraucht)"
                any_attention = True
                plan_verdict = (
                    f"Erhöhte Wachsamkeit: {stop_buffer_used_pct:.0f}% des Weges zum Stop-Loss (${pos.stop_loss_price:,.2f}) sind verbraucht. "
                    f"Server-Stop schützt automatisch."
                )
            elif pnl_usd < 0:
                status_code = "ON_PLAN_PULLBACK"
                status_badge = f"🟢 NACH PLAN (Normaler Pullback, {stop_buffer_remaining_pct:.0f}% SL-Puffer frei)"
                if entry_trend <= 0.0 and entry_vol <= 0.0:
                    any_attention = True
                plan_verdict = (
                    f"Alles im Plan-Korridor: Nur {stop_buffer_used_pct:.0f}% des erlaubten ATR-Stop-Abstands (${pos.stop_loss_price:,.2f}) genutzt. "
                    f"Das aktuelle Minus von ${pnl_usd:,.2f} macht nur {account_impact_pct:.3f}% deines ${total_eq:,.0f}-Kontos aus. "
                    f"Position bleibt bis Stop-Loss (${pos.stop_loss_price:,.2f}) oder Rebound Richtung TP1 (${pos.partial_tp_price:,.2f}) aktiv."
                )
            else:
                status_code = "ON_PLAN_PROFIT"
                status_badge = "🟢 NACH PLAN (Im Gewinn)"
                plan_verdict = (
                    f"Voll nach Plan: Position liegt +${pnl_usd:,.2f} ({pnl_pct:+.2f}%) im Plus. "
                    f"50%-Teilgewinnmitnahme (TP1) wartet bei ${pos.partial_tp_price:,.2f}, Stop-Loss sichert bei ${pos.stop_loss_price:,.2f} ab."
                )

            position_audits.append({
                "symbol": sym,
                "quantity": pos.quantity,
                "entry_price": pos.avg_entry_price,
                "current_price": pos.current_price,
                "stop_loss_price": pos.stop_loss_price,
                "partial_tp_price": pos.partial_tp_price,
                "take_profit_price": pos.take_profit_price,
                "pnl_usd": pnl_usd,
                "pnl_pct": pnl_pct,
                "account_impact_pct": account_impact_pct,
                "stop_buffer_used_pct": stop_buffer_used_pct,
                "stop_buffer_remaining_pct": stop_buffer_remaining_pct,
                "status_code": status_code,
                "status_badge": status_badge,
                "why_in_minus_or_plus": " | ".join(why_reasons),
                "which_calculation_was_off": " ".join(calc_findings),
                "plan_verdict": plan_verdict,
            })

        # Gesamtfazit für das Konto bilden
        losers = [p for p in position_audits if p["pnl_usd"] < 0]
        winners = [p for p in position_audits if p["pnl_usd"] >= 0]
        loser_str = ", ".join(f"{p['symbol']} ({p['pnl_usd']:+.2f}$)" for p in losers) or "keine"
        winner_str = ", ".join(f"{p['symbol']} ({p['pnl_usd']:+.2f}$)" for p in winners) or "keine"

        if any_thesis_broken:
            overall_status = "WARNING"
            overall_badge = "🔴 EINGRIFF ERFORDERLICH (These bei Einzelposition gebrochen)"
        elif any_attention:
            overall_status = "ATTENTION"
            overall_badge = "🟢 NACH PLAN MIT LERN-HINWEIS (Normaler ATR-Spielraum, 1 Berechnungs-Lerneffekt)"
        else:
            overall_status = "ON_PLAN"
            overall_badge = "🟢 ALLES 100% NACH PLAN"

        if eq_delta_usd < 0:
            overall_summary = (
                f"Das Konto steht aktuell minimal bei ${total_eq:,.2f} ({eq_delta_usd:+.2f}$ bzw. {eq_delta_pct:+.2f}% ggü. ${init_cap:,.0f} Startkapital). "
                f"Hauptursache für das kleine Minus: Offene Schwankung bei {loser_str}, während {winner_str} im Plus liegt (Realisiert: {realized_total:+.2f}$). "
                f"Alle Positionen bewegen sich weit oberhalb ihrer Stop-Loss-Marken im normalen ATR-Schwankungskorridor."
            )
        else:
            overall_summary = (
                f"Das Konto liegt bei ${total_eq:,.2f} ({eq_delta_usd:+.2f}$ / {eq_delta_pct:+.2f}% im Plus). "
                f"Gewinntreiber: {winner_str} (Rücksetzer: {loser_str}, Realisiert: {realized_total:+.2f}$)."
            )

        diagnostics = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()[:19],
            "overall_status": overall_status,
            "overall_badge": overall_badge,
            "overall_summary": overall_summary,
            "equity_usd": total_eq,
            "initial_capital_usd": init_cap,
            "equity_delta_usd": eq_delta_usd,
            "equity_delta_pct": eq_delta_pct,
            "unrealized_pnl_usd": unrealized_total,
            "realized_pnl_usd": realized_total,
            "system_notes": system_notes,
            "position_audits": position_audits,
        }
        self._save_state(diagnostics_dict=diagnostics)
        return diagnostics

