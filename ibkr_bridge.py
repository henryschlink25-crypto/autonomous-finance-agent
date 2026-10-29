import datetime
import re
import socket
from typing import Dict, List, Optional, Tuple
from config import AgentConfig
from models import Position, TradingSignal


class IBKRBridge:
    """
    Autonome Multi-Asset API-Brücke zu Interactive Brokers (TWS & IB Gateway):
    - Auto-Discovery der offenen Ports (7497 TWS Paper, 4002 Gateway Paper, 7496 TWS Live, 4001 Gateway Live)
    - Multi-Exchange Smart Routing:
      * Deutsche Xetra/DAX/MDAX-Aktien (.DE) -> Stock(sym, 'SMART', 'EUR', primaryExchange='IBIS')
      * US-Aktien, ADRs & 3x-ETFs -> Stock(sym, 'SMART', 'USD')
      * Echte US-Optionen (CALL / PUT) -> Option(sym, expiry_yyyymmdd, strike, right, 'SMART')
      * Krypto -> Crypto(base, 'PAXOS', 'USD')
    - Vollständige Konto-, Cash- & Positions-Synchronisation (inkl. automatischer EUR->USD Umrechnung für IBIE Irland Konten)
    - Echte serverseitige Bracket-Orders (Market Entry + Take-Profit + Stop-Loss) für LONG und SHORT
    - Dynamisches Nachziehen des serverseitigen Stop-Loss (Chandelier Trailing-Stop) & Teilverkäufe (50% Scale-Out)
    """

    CANDIDATE_PORTS = [
        (7497, "TWS Paper-Trading"),
        (4002, "IB Gateway Paper-Trading"),
        (7496, "TWS Live-Trading"),
        (4001, "IB Gateway Live-Trading"),
    ]

    def __init__(self, config: AgentConfig):
        self.config = config
        self.active_port: Optional[int] = None
        self.active_mode_label: str = "OFFLINE"
        self.last_status_message: str = "Noch nicht verbunden"
        self.account_id: str = config.ibkr_account or ""
        self.base_currency: str = "USD"
        self.fx_rate_to_usd: float = 1.08  # Fallback EUR/USD falls Konto in EUR geführt wird

    def connect_session(self, timeout: float = 2.5):
        """
        Verbindet sich mit TWS oder IB Gateway. Prüft zuerst den konfigurierten Port (z.B. 7497)
        und scannt bei Bedarf automatisch die alternativen IBKR-Ports (4002, 7496, 4001).
        """
        from ib_insync import IB, util

        try:
            util.startLoop()
        except Exception:
            pass

        ports_to_try: List[Tuple[int, str]] = [(self.config.ibkr_port, f"Port {self.config.ibkr_port}")]
        for p, label in self.CANDIDATE_PORTS:
            if p != self.config.ibkr_port:
                ports_to_try.append((p, label))

        last_err = ""
        for port, label in ports_to_try:
            # Schneller TCP-Socket-Vorabcheck (0.2s), damit ib_insync keine Fehlermeldung loggt, solange TWS noch nicht gestartet ist
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.25)
            is_open = sock.connect_ex((self.config.ibkr_host, port)) == 0
            sock.close()
            if not is_open:
                last_err = f"Port {port} geschlossen"
                continue

            ib = IB()
            try:
                ib.connect(
                    self.config.ibkr_host,
                    port,
                    clientId=self.config.ibkr_client_id,
                    timeout=timeout,
                )
                if ib.isConnected():
                    self.active_port = port
                    self.active_mode_label = label
                    managed = ib.managedAccounts()
                    if managed and not self.account_id:
                        self.account_id = managed[0]
                    self.last_status_message = (
                        f"ONLINE ({label} @ {self.config.ibkr_host}:{port} · Konto: {self.account_id or 'Auto'})"
                    )
                    return ib
            except Exception as e:
                last_err = str(e)
                try:
                    ib.disconnect()
                except Exception:
                    pass

        self.active_port = None
        self.active_mode_label = "OFFLINE"
        self.last_status_message = (
            f"WARTET AUF TWS/GATEWAY ({self.config.ibkr_host}:{self.config.ibkr_port} nicht offen – {last_err[:60]})"
        )
        return None

    def is_reachable(self) -> bool:
        ib = self.connect_session(timeout=1.5)
        if ib is not None:
            ib.disconnect()
            return True
        return False

    def build_stock_or_crypto_contract(self, symbol: str, asset_class: str = "EQUITY"):
        """Erstellt den passenden IBKR-Contract für US-Aktien (USD), Xetra-Aktien (.DE -> EUR) oder Krypto."""
        from ib_insync import Stock, Crypto

        if asset_class == "CRYPTO" or "/" in symbol:
            base_sym = symbol.split("/")[0]
            return Crypto(base_sym, "PAXOS", "USD")
        if symbol.endswith(".DE"):
            clean_de = symbol.replace(".DE", "")
            return Stock(clean_de, "SMART", "EUR", primaryExchange="IBIS")
        return Stock(symbol, "SMART", "USD")

    def build_option_contract_from_badge(self, symbol: str, option_badge: str, ref_year: Optional[int] = None):
        """
        Parst den vom Options-Scanner ermittelten Top-Kontrakt (z.B. 'CALL $50 (11-20) @ $2.30 [V/OI:118.5x 🔥Whale]')
        in einen echten, ausführbaren IBKR Option-Contract!
        """
        from ib_insync import Option

        if not option_badge or symbol.endswith(".DE"):
            return None
        m = re.search(r"(CALL|PUT)\s+\$([0-9]+(?:\.[0-9]+)?)\s+\((\d{2})-(\d{2})\)", option_badge)
        if not m:
            return None
        right = "C" if m.group(1) == "CALL" else "P"
        strike = float(m.group(2))
        month = int(m.group(3))
        day = int(m.group(4))
        now = datetime.date.today()
        year = ref_year or now.year
        exp_date = datetime.date(year, month, day)
        if exp_date < now:
            exp_date = datetime.date(year + 1, month, day)
        expiry_str = exp_date.strftime("%Y%m%d")
        return Option(symbol, expiry_str, strike, right, "SMART", currency="USD")

    def sync_account_and_positions(
        self,
        existing_positions: Dict[str, Position],
        cluster_map: Dict[str, str],
    ) -> Optional[Tuple[float, float, Dict[str, Position], Dict[str, float]]]:
        """
        Liest NetLiquidation, Cash und alle offenen Positionen live aus Interactive Brokers (TWS/Gateway) aus.
        Gibt (equity_usd, cash_usd, updated_positions, live_prices) zurück oder None, falls TWS nicht läuft.
        """
        ib = self.connect_session(timeout=2.5)
        if ib is None:
            return None

        try:
            acct_values = ib.accountValues(self.account_id) if self.account_id else ib.accountValues()
            net_liq = 0.0
            base_curr = "USD"

            # Suche NetLiquidation (bevorzugt in USD, sonst in EUR/Basiswährung mit Umrechnung)
            for av in acct_values:
                if av.tag == "NetLiquidation" and av.currency == "USD":
                    net_liq = float(av.value)
                    base_curr = "USD"
                    break
            if net_liq <= 0:
                for av in acct_values:
                    if av.tag == "NetLiquidation" and av.currency in ("EUR", "BASE") and float(av.value) > 0:
                        raw_val = float(av.value)
                        base_curr = av.currency
                        net_liq = raw_val * (self.fx_rate_to_usd if av.currency == "EUR" else 1.0)
                        break

            # Prüfe Wechselkurs aus AccountValues falls vorhanden
            for av in acct_values:
                if av.tag == "ExchangeRate" and av.currency == "EUR":
                    try:
                        fx = float(av.value)
                        if 0.8 <= fx <= 1.5:
                            self.fx_rate_to_usd = fx
                    except Exception:
                        pass

            self.base_currency = base_curr
            portfolio_items = ib.portfolio(self.account_id) if self.account_id else ib.portfolio()
            live_prices: Dict[str, float] = {}

            for item in portfolio_items:
                qty_raw = float(item.position)
                if abs(qty_raw) < 1e-6:
                    continue
                c = item.contract
                sym = str(c.symbol).upper()
                if getattr(c, "currency", "USD") == "EUR" and not sym.endswith(".DE") and sym not in ("SAP", "ASML"):
                    sym = f"{sym}.DE"

                fx_mult = self.fx_rate_to_usd if getattr(c, "currency", "USD") == "EUR" else 1.0
                curr_p = round(float(item.marketPrice) * fx_mult, 4)
                entry_p = round(float(item.averageCost) * fx_mult, 4)
                side = "SHORT" if qty_raw < 0 else "LONG"
                abs_qty = round(abs(qty_raw), 4)
                live_prices[sym] = curr_p

                if sym in existing_positions:
                    pos = existing_positions[sym]
                    pos.side = side
                    pos.quantity = abs_qty
                    pos.avg_entry_price = entry_p
                    pos.current_price = curr_p
                    pos.highest_price = (
                        min(pos.highest_price, curr_p, entry_p)
                        if side == "SHORT"
                        else max(pos.highest_price, curr_p, entry_p)
                    )
                    pos.margin_used_usd = round((abs_qty * entry_p) / max(1.0, pos.leverage), 2)
                else:
                    atr_est = max(curr_p * 0.025, 1.0)
                    sl_p = round(entry_p + 1.6 * atr_est, 4) if side == "SHORT" else round(entry_p - 1.6 * atr_est, 4)
                    tp1_p = round(entry_p - 1.5 * atr_est, 4) if side == "SHORT" else round(entry_p + 1.5 * atr_est, 4)
                    tp2_p = round(entry_p - 3.2 * atr_est, 4) if side == "SHORT" else round(entry_p + 3.2 * atr_est, 4)
                    existing_positions[sym] = Position(
                        symbol=sym,
                        asset_class="CRYPTO" if getattr(c, "secType", "") == "CRYPTO" else "EQUITY",
                        cluster=cluster_map.get(sym, "EU_LEADERS" if sym.endswith(".DE") else "US_TECH"),
                        side=side,
                        quantity=abs_qty,
                        leverage=1.0,
                        intrinsic_leverage=1.0,
                        margin_used_usd=round(abs_qty * entry_p, 2),
                        avg_entry_price=entry_p,
                        current_price=curr_p,
                        highest_price=curr_p,
                        stop_loss_price=sl_p,
                        partial_tp_price=tp1_p,
                        partial_tp_done=False,
                        take_profit_price=tp2_p,
                        opened_at=datetime.datetime.now(datetime.timezone.utc).isoformat()[:19],
                    )

            pos_market_val = sum(p.market_value for p in existing_positions.values())
            cash_usd = round(max(0.0, net_liq - pos_market_val), 2) if net_liq > 0 else 0.0
            ib.disconnect()
            return round(net_liq, 2), cash_usd, existing_positions, live_prices
        except Exception as e:
            self.last_status_message = f"IBKR_SYNC_ERROR: {e}"
            try:
                ib.disconnect()
            except Exception:
                pass
            return None

    def place_order(
        self,
        signal: TradingSignal,
        qty: float,
        side: str,
        is_short_entry: bool = False,
        option_badge: str = "",
    ) -> str:
        """
        Sendet eine Aktien-, ETF-, Krypto- oder Options-Order an Interactive Brokers.
        Nutzt bei Aktien-Einstiegen (LONG & SHORT) serverseitige Bracket-Orders (Market Entry + Stop-Loss + Take-Profit).
        """
        from ib_insync import MarketOrder

        ib = self.connect_session(timeout=3.5)
        if ib is None:
            return f"IBKR_ERROR: {self.last_status_message}"

        try:
            contract = self.build_stock_or_crypto_contract(signal.symbol, signal.asset_class)
            ib.qualifyContracts(contract)
            order_qty = max(1, int(round(qty))) if signal.asset_class == "EQUITY" else round(qty, 4)

            # 1. LONG-Einstieg mit serverseitiger Bracket-Order (Einstieg + Stop-Loss + Take-Profit)
            if side == "BUY" and not is_short_entry and signal.asset_class == "EQUITY" and order_qty >= 1:
                bracket = ib.bracketOrder(
                    action="BUY",
                    quantity=order_qty,
                    limitPrice=round(signal.entry_price * 1.003, 2),
                    takeProfitPrice=round(signal.take_profit_price, 2),
                    stopLossPrice=round(signal.stop_loss_price, 2),
                )
                bracket[0].orderType = "MKT"
                for o in bracket:
                    if self.account_id:
                        o.account = self.account_id
                    ib.placeOrder(contract, o)
                ib.sleep(0.25)

                # Optional: Falls der Agent gleichzeitig den konkreten Top-Optionskontrakt (Whale-Call) handeln soll
                opt_note = ""
                if self.config.ibkr_enable_options_trading and option_badge:
                    opt_note = self._try_place_companion_option_order(ib, signal.symbol, option_badge)

                ib.disconnect()
                return (
                    f"IBKR_BRACKET_LONG ({self.active_mode_label} · Qty:{order_qty} {signal.symbol}, "
                    f"SL:{signal.stop_loss_price:.2f}, TP:{signal.take_profit_price:.2f}){opt_note}"
                )

            # 2. SHORT-SELL-Einstieg mit umgekehrter Bracket-Order (SELL MKT + BUY STP oben + BUY LMT unten)
            if side == "SELL" and is_short_entry and signal.asset_class == "EQUITY" and order_qty >= 1:
                atr_est = max(abs(signal.entry_price - signal.stop_loss_price) / 1.6, signal.entry_price * 0.015)
                short_sl = round(signal.entry_price + 1.6 * atr_est, 2)
                short_tp = round(max(0.05, signal.entry_price - 3.2 * atr_est), 2)
                bracket = ib.bracketOrder(
                    action="SELL",
                    quantity=order_qty,
                    limitPrice=round(signal.entry_price * 0.997, 2),
                    takeProfitPrice=short_tp,
                    stopLossPrice=short_sl,
                )
                bracket[0].orderType = "MKT"
                for o in bracket:
                    if self.account_id:
                        o.account = self.account_id
                    ib.placeOrder(contract, o)
                ib.sleep(0.25)
                ib.disconnect()
                return (
                    f"IBKR_BRACKET_SHORT ({self.active_mode_label} · Qty:{order_qty} {signal.symbol}, "
                    f"SL oben:{short_sl:.2f}, TP unten:{short_tp:.2f})"
                )

            # 3. Direkt-Order (Scale-Out, glattstellen oder Krypto)
            mkt_order = MarketOrder(side, order_qty)
            if self.account_id:
                mkt_order.account = self.account_id
            trade = ib.placeOrder(contract, mkt_order)
            ib.sleep(0.25)
            order_id = trade.order.orderId
            ib.disconnect()
            return f"IBKR_ORDER_ID:{order_id} ({self.active_mode_label})"
        except Exception as e:
            try:
                ib.disconnect()
            except Exception:
                pass
            return f"IBKR_ERROR:{e}"

    def _try_place_companion_option_order(self, ib, symbol: str, option_badge: str) -> str:
        """Platziert auf Wunsch automatisch 1 Kontrakt der vom Scanner identifizierten Whale-Option."""
        try:
            from ib_insync import MarketOrder

            opt_contract = self.build_option_contract_from_badge(symbol, option_badge)
            if opt_contract is None:
                return ""
            qualified = ib.qualifyContracts(opt_contract)
            if not qualified:
                return ""
            opt_order = MarketOrder("BUY", 1)
            if self.account_id:
                opt_order.account = self.account_id
            trade = ib.placeOrder(opt_contract, opt_order)
            ib.sleep(0.2)
            return f" + 🎫 IBKR_OPTION_1x ({opt_contract.right} ${opt_contract.strike} {opt_contract.lastTradeDateOrContractMonth} ID:{trade.order.orderId})"
        except Exception:
            return ""

    def sync_exit_or_trailing_stop(
        self,
        symbol: str,
        asset_class: str,
        sell_qty: float,
        new_stop_loss: Optional[float] = None,
        remaining_qty: float = 0.0,
        position_side: str = "LONG",
    ) -> str:
        """
        Synchronisiert 50%-Teilverkäufe, Komplett-Exits und Chandelier-Trailing-Stop-Updates direkt mit IBKR.
        """
        from ib_insync import MarketOrder

        ib = self.connect_session(timeout=2.5)
        if ib is None:
            return f"IBKR_OFFLINE_SKIP ({self.last_status_message})"

        notes = []
        try:
            contract = self.build_stock_or_crypto_contract(symbol, asset_class)
            ib.qualifyContracts(contract)

            if sell_qty > 0:
                exit_action = "BUY" if position_side == "SHORT" else "SELL"
                q = max(1, int(round(sell_qty))) if asset_class == "EQUITY" else round(sell_qty, 4)
                mkt = MarketOrder(exit_action, q)
                if self.account_id:
                    mkt.account = self.account_id
                t = ib.placeOrder(contract, mkt)
                ib.sleep(0.2)
                notes.append(f"IBKR_{exit_action}:{t.order.orderId}")

            if new_stop_loss is not None and remaining_qty > 0:
                open_trades = ib.openTrades()
                clean_sym = symbol.replace(".DE", "").split("/")[0]
                for tr in open_trades:
                    if tr.contract.symbol.upper() == clean_sym and tr.order.orderType in ("STP", "STP LMT"):
                        tr.order.auxPrice = round(new_stop_loss, 2)
                        tr.order.totalQuantity = max(1, int(round(remaining_qty)))
                        ib.placeOrder(tr.contract, tr.order)
                        ib.sleep(0.15)
                        notes.append(f"IBKR_SL_UPDATED:{new_stop_loss:.2f}")
            ib.disconnect()
        except Exception as e:
            try:
                ib.disconnect()
            except Exception:
                pass
            notes.append(f"IBKR_SYNC_WARN:{e}")

        return " | ".join(notes)
