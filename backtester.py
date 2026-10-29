import json
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from config import AgentConfig
from data_collector import compute_wilder_rsi

console = Console(width=140)


class WalkForwardBacktester:
    """
    Simuliert die Multi-Faktor-Strategie & deterministische Risiko-Engine über historische Kursdaten:
    - Cross-Sectional Signal-Ranking (Beste Signale erhalten zuerst Kapital)
    - Sektor-/Cluster-Korrelationsschutz (max. 45% pro Cluster) & 10% Cash-Reserve
    - 2-Stufen Scale-Out (50% Teilgewinnmitnahme bei +2.0x ATR & Rest-Runner bis +4.4x ATR)
    - Dynamischer Chandelier-Trailing-Stop & VIX-Makro-Ampel
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.results_file = os.path.join(os.path.dirname(self.config.portfolio_state_file), "backtest_results.json")

    def _prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        df["sma_20"] = close.rolling(20).mean()
        df["sma_50"] = close.rolling(50).mean()
        df["sma_50_slope"] = df["sma_50"] - df["sma_50"].shift(5)
        df["rsi_14"] = compute_wilder_rsi(close, period=14)

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_hist"] = macd_line - signal_line

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean().bfill()

        df["vol_ratio"] = (volume / volume.rolling(20).mean()).fillna(1.0)
        df["ret_1d"] = close.pct_change().fillna(0.0)
        return df.dropna(subset=["sma_50", "atr_14"])

    def run_backtest(self, period: str = "1y") -> dict:
        console.rule(f"[bold cyan]📈 Multi-Asset Walk-Forward Backtest (Zeitraum: {period})[/bold cyan]")

        asset_data: Dict[str, pd.DataFrame] = {}
        safe_haven_symbols = {item.symbol for item in self.config.watchlist if item.is_safe_haven}
        cluster_map = {item.symbol: item.cluster for item in self.config.watchlist}
        intrinsic_lev_map = {item.symbol: item.intrinsic_leverage for item in self.config.watchlist}
        vix_series = pd.Series(dtype=float)

        with console.status("[bold green]Lade historische Zeitreihen, VIX Makro-Daten & berechne Indikatoren..."):
            try:
                vix_raw = yf.Ticker("^VIX").history(period=period, interval="1d")
                if vix_raw is not None and not vix_raw.empty:
                    vix_raw.index = vix_raw.index.tz_localize(None).normalize()
                    vix_series = vix_raw["Close"].astype(float)
            except Exception:
                pass

            for item in self.config.watchlist:
                raw = yf.Ticker(item.yf_ticker).history(period=period, interval="1d")
                if raw is not None and len(raw) > 60:
                    df = self._prepare_indicators(raw)
                    df.index = df.index.tz_localize(None).normalize()
                    asset_data[item.symbol] = df

        if not asset_data:
            console.print("[red]Keine ausreichenden historischen Daten für den Backtest gefunden.[/red]")
            return {}

        all_dates = sorted(set.union(*(set(df.index) for df in asset_data.values())))
        cash = self.config.initial_capital_usd
        positions: Dict[str, dict] = {}
        closed_trades: List[dict] = []
        equity_curve: List[dict] = []

        initial_prices = {sym: float(df["Close"].iloc[0]) for sym, df in asset_data.items()}

        for current_date in all_dates:
            date_str = current_date.strftime("%Y-%m-%d")

            # 1. Offene Positionen aktualisieren, 50% Scale-Out (TP1) & Trailing Stop / TP2 prüfen
            for sym in list(positions.keys()):
                df = asset_data[sym]
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                low_p = float(row["Low"])
                high_p = float(row["High"])
                close_p = float(row["Close"])
                atr_now = max(float(row["atr_14"]), close_p * 0.015)
                pos = positions[sym]
                pos["current_price"] = close_p
                pos["days_held"] = pos.get("days_held", 0) + 1

                exit_price = None
                exit_reason = None
                if low_p <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "TRAILING_STOP" if pos["stop_loss"] > pos["entry_price"] else "STOP_LOSS"
                elif high_p >= pos["take_profit"]:
                    exit_price = pos["take_profit"]
                    exit_reason = "TAKE_PROFIT_2"
                elif (
                    pos["days_held"] >= self.config.max_stale_holding_days
                    and ((close_p - pos["entry_price"]) / pos["entry_price"]) * 100.0 < 1.0
                    and float(row["macd_hist"]) <= 0
                ):
                    exit_price = close_p
                    exit_reason = f"TIME_STOP_{pos['days_held']}D"
                else:
                    # 1a. Prüfe 50% Teilgewinnmitnahme (Scale-Out) bei TP1 (+1.5x ATR)
                    if not pos.get("partial_tp_done", False) and high_p >= pos["partial_tp"]:
                        sell_qty = pos["qty"] * 0.5
                        tp1_price = pos["partial_tp"]
                        released_margin = pos["margin_used"] * 0.5
                        fee = sell_qty * tp1_price * 0.0005
                        pnl = (sell_qty * (tp1_price - pos["entry_price"])) - fee
                        pnl_pct = (pnl / max(1.0, released_margin)) * 100.0
                        cash += released_margin + pnl
                        pos["margin_used"] -= released_margin
                        pos["qty"] -= sell_qty
                        pos["partial_tp_done"] = True
                        pos["stop_loss"] = max(pos["stop_loss"], pos["entry_price"] + 0.3 * atr_now)
                        closed_trades.append({
                            "symbol": f"{sym} ({pos['eff_lev']:.0f}x)",
                            "entry_date": pos["entry_date"],
                            "exit_date": date_str,
                            "entry_price": round(pos["entry_price"], 2),
                            "exit_price": round(tp1_price, 2),
                            "pnl_usd": round(pnl, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "reason": "PARTIAL_TP_50%",
                        })

                    # 1b. Dynamischer Chandelier Trailing-Stop (aktiviert ab +0.9x ATR)
                    pos["highest_price"] = max(pos.get("highest_price", pos["entry_price"]), high_p)
                    if pos["highest_price"] >= pos["entry_price"] + 0.9 * atr_now:
                        new_trail_sl = pos["highest_price"] - 1.5 * atr_now
                        pos["stop_loss"] = max(pos["stop_loss"], pos["entry_price"] * 1.002, new_trail_sl)

                if exit_price is not None:
                    fee = pos["qty"] * exit_price * 0.0005
                    pnl = (pos["qty"] * (exit_price - pos["entry_price"])) - fee
                    pnl_pct = (pnl / max(1.0, pos["margin_used"])) * 100.0
                    cash += pos["margin_used"] + pnl
                    closed_trades.append({
                        "symbol": f"{sym} ({pos['eff_lev']:.0f}x)",
                        "entry_date": pos["entry_date"],
                        "exit_date": date_str,
                        "entry_price": round(pos["entry_price"], 2),
                        "exit_price": round(exit_price, 2),
                        "pnl_usd": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "reason": exit_reason,
                    })
                    del positions[sym]

            # 2. Cross-Sectional Signal-Analyse über alle Assets
            sell_candidates: List[tuple] = []
            buy_candidates: List[tuple] = []

            for sym, df in asset_data.items():
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                price = float(row["Close"])
                sma20 = float(row["sma_20"])
                sma50 = float(row["sma_50"])
                sma50_slope = float(row["sma_50_slope"]) if not np.isnan(row["sma_50_slope"]) else 0.0
                rsi = float(row["rsi_14"])
                macd_h = float(row["macd_hist"])
                atr = max(float(row["atr_14"]), price * 0.015)
                vol_r = float(row["vol_ratio"])
                ret_1d = float(row["ret_1d"])

                score = 0.0
                if price > sma20 > sma50 and sma50_slope > 0:
                    score += 0.40
                elif price < sma20 < sma50:
                    score -= 0.40

                if 48 <= rsi <= 68 and macd_h > 0:
                    score += 0.25
                elif rsi < 30 and price > sma50:
                    score += 0.30
                elif rsi > 76:
                    score -= 0.35

                if vol_r > 1.15 and ret_1d > 0:
                    score += 0.15

                if score <= -0.35 and sym in positions:
                    sell_candidates.append((sym, price, score))
                elif score >= 0.55 and sym not in positions:
                    buy_candidates.append((sym, price, atr, score))

            # 2a. Zuerst alle SELL-Signale ausführen (setzt Cash & Cluster-Kapazität frei)
            for sym, price, _ in sell_candidates:
                pos = positions.pop(sym)
                fee = pos["qty"] * price * 0.0005
                pnl = (pos["qty"] * (price - pos["entry_price"])) - fee
                pnl_pct = (pnl / max(1.0, pos["margin_used"])) * 100.0
                cash += pos["margin_used"] + pnl
                closed_trades.append({
                    "symbol": f"{sym} ({pos['eff_lev']:.0f}x)",
                    "entry_date": pos["entry_date"],
                    "exit_date": date_str,
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(price, 2),
                    "pnl_usd": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "reason": "SIGNAL_SELL",
                })

            # 2b. BUY-Kandidaten absteigend nach Signal-Score sortieren (Beste Chancen zuerst!)
            buy_candidates.sort(key=lambda x: x[3], reverse=True)
            vix_val = float(vix_series.get(current_date, 18.0)) if not vix_series.empty else 18.0
            min_order = self.config.min_order_usd

            for sym, price, atr, score in buy_candidates:
                is_safe = sym in safe_haven_symbols
                if vix_val >= 30.0 and not is_safe:
                    continue
                is_green_macro = vix_val < 22.5
                macro_mult = 1.0 if (is_safe or is_green_macro) else 0.5

                pos_val = sum(
                    p["margin_used"] + (p["qty"] * (p["current_price"] - p["entry_price"]))
                    for p in positions.values()
                )
                total_equity = cash + pos_val

                # Cluster-Limit (60%) & Mindest-Cash-Reserve (8%) prüfen
                sym_cluster = cluster_map.get(sym, "US_TECH")
                cluster_val = sum(
                    p["margin_used"] + (p["qty"] * (p["current_price"] - p["entry_price"]))
                    for p in positions.values()
                    if p.get("cluster") == sym_cluster
                )
                rem_cluster_cap = (total_equity * self.config.max_cluster_weight_pct) - cluster_val
                deployable_cash = max(0.0, (cash - total_equity * self.config.min_cash_reserve_pct) * 0.99)

                if rem_cluster_cap < min_order or deployable_cash < min_order:
                    continue

                intr_lev = intrinsic_lev_map.get(sym, 1.0)
                lev_used = self.config.smart_leverage_max if (is_green_macro and intr_lev <= 1.0 and not is_safe) else 1.0
                eff_lev = lev_used * intr_lev

                stop_dist = 1.6 * atr
                partial_tp_dist = 1.5 * atr
                tp_dist = 3.2 * atr
                risk_dollars = total_equity * self.config.max_risk_per_trade_pct * macro_mult
                risk_qty = risk_dollars / stop_dist
                risk_margin = (risk_qty * price) / lev_used

                max_margin = min(
                    risk_margin,
                    total_equity * self.config.max_position_weight_pct,
                    rem_cluster_cap,
                    deployable_cash,
                )
                if max_margin >= min_order:
                    notional = max_margin * lev_used
                    qty = notional / price
                    fee = notional * 0.0005
                    cash -= (max_margin + fee)
                    positions[sym] = {
                        "qty": qty,
                        "margin_used": max_margin,
                        "lev_used": lev_used,
                        "eff_lev": eff_lev,
                        "cluster": sym_cluster,
                        "entry_price": price,
                        "highest_price": price,
                        "current_price": price,
                        "stop_loss": price - stop_dist,
                        "partial_tp": price + partial_tp_dist,
                        "partial_tp_done": False,
                        "take_profit": price + tp_dist,
                        "entry_date": date_str,
                        "days_held": 0,
                    }

            # Tages-Endstand für Equity Curve & Benchmark speichern
            pos_val = sum(
                p["margin_used"] + (p["qty"] * (p["current_price"] - p["entry_price"]))
                for p in positions.values()
            )
            total_equity = cash + pos_val

            bench_rels = []
            for sym, df in asset_data.items():
                if current_date in df.index:
                    bench_rels.append(float(df.loc[current_date, "Close"]) / initial_prices[sym])
            bench_equity = self.config.initial_capital_usd * (float(np.mean(bench_rels)) if bench_rels else 1.0)

            equity_curve.append({
                "date": date_str,
                "agent_equity": round(total_equity, 2),
                "benchmark_equity": round(bench_equity, 2),
                "cash": round(cash, 2),
            })

        eq_series = pd.Series([pt["agent_equity"] for pt in equity_curve])
        bench_series = pd.Series([pt["benchmark_equity"] for pt in equity_curve])
        daily_ret = eq_series.pct_change().dropna()

        total_return_pct = ((eq_series.iloc[-1] - self.config.initial_capital_usd) / self.config.initial_capital_usd) * 100.0
        bench_return_pct = ((bench_series.iloc[-1] - self.config.initial_capital_usd) / self.config.initial_capital_usd) * 100.0

        rolling_max = eq_series.cummax()
        drawdowns = (eq_series - rolling_max) / rolling_max
        max_drawdown_pct = float(drawdowns.min() * 100.0)

        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

        wins = [t for t in closed_trades if t["pnl_usd"] > 0]
        losses = [t for t in closed_trades if t["pnl_usd"] <= 0]
        win_rate_pct = (len(wins) / len(closed_trades) * 100.0) if closed_trades else 0.0
        gross_profit = sum(t["pnl_usd"] for t in wins)
        gross_loss = abs(sum(t["pnl_usd"] for t in losses))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (9.99 if gross_profit > 0 else 0.0)

        summary = {
            "period": period,
            "initial_capital_usd": self.config.initial_capital_usd,
            "final_equity_usd": round(float(eq_series.iloc[-1]), 2),
            "total_return_pct": round(float(total_return_pct), 2),
            "benchmark_return_pct": round(float(bench_return_pct), 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_trades": len(closed_trades),
            "win_rate_pct": round(win_rate_pct, 1),
            "profit_factor": profit_factor,
            "recent_trades": closed_trades[-15:],
            "equity_curve": equity_curve,
        }

        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._render_summary(summary)
        return summary

    def _render_summary(self, s: dict) -> None:
        ret_col = "green" if s["total_return_pct"] >= 0 else "red"
        console.print(
            Panel(
                f"[bold]Startkapital:[/bold] ${s['initial_capital_usd']:,.2f}  ➔  "
                f"[bold]Endkapital:[/bold] ${s['final_equity_usd']:,.2f} ([{ret_col} bold]{s['total_return_pct']:+.2f}%[/{ret_col} bold])\n"
                f"[bold]Sharpe Ratio:[/bold] {s['sharpe_ratio']:.2f}   |   "
                f"[bold]Max Drawdown:[/bold] [yellow]{s['max_drawdown_pct']:.2f}%[/yellow]   |   "
                f"[bold]Win-Rate:[/bold] {s['win_rate_pct']:.1f}% ({s['total_trades']} Teil-/Gesamt-Exits)   |   "
                f"[bold]Profit-Faktor:[/bold] {s['profit_factor']:.2f}",
                title="🏆 Walk-Forward Backtest Ergebnis (inkl. 50% Scale-Out, Cluster-Schutz & Trailing-Stop)",
                border_style="green",
            )
        )

        if s["recent_trades"]:
            t_table = Table(title="Letzte abgeschlossene Backtest-Trades", box=box.SIMPLE_HEAVY, header_style="bold cyan")
            t_table.add_column("Symbol", style="bold")
            t_table.add_column("Einstieg (Datum / Kurs)")
            t_table.add_column("Ausstieg (Datum / Kurs)")
            t_table.add_column("Exit-Grund")
            t_table.add_column("PnL ($ / %)", justify="right")

            for tr in s["recent_trades"][-10:]:
                c = "green" if tr["pnl_usd"] >= 0 else "red"
                t_table.add_row(
                    tr["symbol"],
                    f"{tr['entry_date']} @ ${tr['entry_price']:,.2f}",
                    f"{tr['exit_date']} @ ${tr['exit_price']:,.2f}",
                    tr["reason"],
                    f"[{c}]{tr['pnl_usd']:+,.2f}$ ({tr['pnl_pct']:+.2f}%)[/{c}]",
                )
            console.print(t_table)
