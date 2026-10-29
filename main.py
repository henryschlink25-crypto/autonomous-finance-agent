import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import AgentConfig
from data_collector import MultiAssetDataCollector
from ai_analyst import HybridAIAnalyst
from risk_manager import DeterministicRiskManager, is_us_equity_market_open
from execution_engine import MultiAssetExecutionEngine
from backtester import WalkForwardBacktester
from dashboard_generator import DashboardGenerator
from self_improver import SelfLearningShadowTracker
from adas_meta_agent import ADASMetaArchitect
from models import MarketSnapshot, PortfolioRiskMetrics, TradingSignal

console = Console(width=152, legacy_windows=False)


def render_portfolio_summary(
    engine: MultiAssetExecutionEngine,
    config: AgentConfig,
    port_risk: Optional[PortfolioRiskMetrics] = None,
) -> None:
    pos_table = Table(
        title=f"💼 Smart-Leverage Portfolio (Startkapital: {config.initial_capital_usd:,.0f}$ | Modus: {config.execution_mode})",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    pos_table.add_column("Symbol", style="bold")
    pos_table.add_column("Cluster / Hebel")
    pos_table.add_column("Haltedauer\n(Max-Limit)", justify="center")
    pos_table.add_column("Menge", justify="right")
    pos_table.add_column("Einstieg ➔ Aktuell", justify="right")
    pos_table.add_column("Margin ➔ Kaufkraft", justify="right")
    pos_table.add_column("SL / TP1 (50%) / TP2", justify="right")
    pos_table.add_column("Hebel-PnL (auf Margin)", justify="right")

    if not engine.positions:
        pos_table.add_row("-", "-", "-", "-", "-", "-", "-", "-")
    else:
        import datetime as dt
        now_utc = dt.datetime.now(dt.timezone.utc)
        for pos in engine.positions.values():
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
            tp1_tag = "✓" if pos.partial_tp_done else f"{pos.partial_tp_price:,.2f}"
            eff_lev = round(pos.leverage * pos.intrinsic_leverage, 1)
            lev_badge = f"[bold yellow]⚡{eff_lev:.0f}x[/bold yellow]" if eff_lev > 1.0 else "1x"
            max_d = config.leveraged_etf_max_days if pos.intrinsic_leverage > 1.0 else config.max_stale_holding_days
            hold_str = f"0d / max {max_d}d"
            try:
                if "T" in pos.opened_at:
                    op_dt = dt.datetime.fromisoformat(pos.opened_at[:19]).replace(tzinfo=dt.timezone.utc)
                    hrs = max(0.0, (now_utc - op_dt).total_seconds() / 3600.0)
                    days = int(hrs // 24)
                    rem_h = int(hrs % 24)
                    hold_str = f"{days}d {rem_h}h / {max_d}d"
            except Exception:
                pass
            pos_table.add_row(
                pos.symbol,
                f"[dim]{pos.cluster}[/dim] {lev_badge}",
                hold_str,
                f"{pos.quantity:g}",
                f"${pos.avg_entry_price:,.2f} ➔ ${pos.current_price:,.2f}",
                f"${pos.market_value:,.2f} ➔ ${pos.notional_value:,.0f}",
                f"{pos.stop_loss_price:,.2f} / {tp1_tag} / {pos.take_profit_price:,.2f}",
                f"[{pnl_color}]{pos.unrealized_pnl:+,.2f}$ ({pos.unrealized_pnl_pct:+.2f}%)[/{pnl_color}]",
            )

    console.print(pos_table)
    ret_color = "green" if engine.total_return_pct >= 0 else "red"
    day_color = "green" if engine.daily_pnl_pct >= 0 else "red"

    risk_line = ""
    if port_risk is not None:
        var_col = "green" if port_risk.daily_var_95_pct < 1.8 else ("yellow" if port_risk.daily_var_95_pct < 2.2 else "red")
        pairs_str = "  |  ".join(port_risk.top_correlated_pairs[:4])
        risk_line = (
            f"\n[bold]Portfolio-Beta (vs SPY):[/bold] {port_risk.portfolio_beta:.2f}   |   "
            f"[bold]95%-Tages-VaR (Kovarianz):[/bold] [{var_col}]${port_risk.daily_var_95_usd:,.2f} ({port_risk.daily_var_95_pct:.2f}%)[/{var_col}]   |   "
            f"[bold]Diversifikations-Vorteil:[/bold] [green]-{port_risk.diversification_benefit_pct:.1f}% Risiko[/green]\n"
            f"[dim]🔗 Stärkste Markt-Kopplungen (60d): {pairs_str}[/dim]"
        )

    ibkr_status = getattr(engine, "ibkr_bridge", None)
    ibkr_badge = (
        f"[bold green]🟢 IBKR {ibkr_status.active_mode_label} (Port {ibkr_status.active_port})[/bold green]"
        if (ibkr_status and ibkr_status.active_port)
        else "[yellow]⏳ IBKR Bridge bereit (wartet auf TWS/Gateway Port 7497/4002/7496)[/yellow]"
    )

    console.print(
        Panel(
            f"[bold]Gesamt-Equity:[/bold] ${engine.total_equity_usd:,.2f}   |   "
            f"[bold]Verfügbarer Cash:[/bold] ${engine.cash_usd:,.2f} (Min-Reserve: {config.min_cash_reserve_pct*100:.0f}%)   |   "
            f"[bold]Realisierter PnL:[/bold] ${engine.realized_pnl_usd:+,.2f}   |   "
            f"[bold]Tages-PnL:[/bold] [{day_color}]{engine.daily_pnl_pct:+.2f}%[/{day_color}]   |   "
            f"[bold]Gesamt-Rendite:[/bold] [{ret_color}]{engine.total_return_pct:+.2f}%[/{ret_color}]\n"
            f"[bold]Broker-Brücken:[/bold] Mode={config.execution_mode}   |   {ibkr_badge}"
            f"{risk_line}",
            border_style="cyan",
        )
    )


def render_shadow_learning_summary(tracker: SelfLearningShadowTracker) -> None:
    cal = tracker.calibration
    weights_str = "  |  ".join(f"{k}: [bold]{v:.2f}x[/bold]" for k, v in cal.weights.items())
    lessons_str = "\n".join(f"• {l}" for l in cal.recent_lessons[-4:]) if cal.recent_lessons else "Noch keine abgeschlossenen Auswertungen."

    # Zeige laufende hypothetische Shadow-Trades (auch für nicht gehaltene Assets!)
    shadow_table = Table(
        title="🔬 Shadow-Portfolio & Hypothetisches Signal-Tracking (Alle Assets – auch ohne eigenen Bestand)",
        box=box.SIMPLE,
        header_style="bold yellow",
    )
    shadow_table.add_column("Asset", style="bold")
    shadow_table.add_column("Prognose (Konf.)")
    shadow_table.add_column("Im Echt-Portfolio?")
    shadow_table.add_column("Startkurs ➔ Aktuell", justify="right")
    shadow_table.add_column("Hypoth. Ziel / Stop", justify="right")
    shadow_table.add_column("Prognose-Erfolg (Richtungs-PnL)", justify="right")

    for sym, pred in tracker.active_predictions.items():
        act_col = "green" if pred.predicted_action == "BUY" else ("red" if pred.predicted_action == "SELL" else "dim")
        dir_col = "green" if pred.directional_return_pct >= 0 else "red"
        exec_badge = "[green]✔ Ja (Gekauft)[/green]" if pred.was_executed else f"[dim]Nein ({pred.execution_note[:28]})[/dim]"
        shadow_table.add_row(
            sym,
            f"[{act_col} bold]{pred.predicted_action}[/{act_col} bold] ({pred.confidence*100:.0f}%)",
            exec_badge,
            f"${pred.entry_price:,.2f} ➔ ${pred.current_price:,.2f}",
            f"Ziel: ${pred.target_price:,.2f} / Inv: ${pred.invalidation_price:,.2f}",
            f"[{dir_col}]{pred.directional_return_pct:+.2f}%[/{dir_col}] (Max: {pred.max_favorable_pct:+.2f}%)",
        )

    console.print(shadow_table)
    acc_col = "green" if cal.accuracy_pct >= 55.0 else "yellow"
    console.print(
        Panel(
            f"[bold]Geprüfte historische Hypothesen:[/bold] {cal.total_evaluated}   |   "
            f"[bold]Trefferquote (BUY & SELL Calls):[/bold] [{acc_col}]{cal.accuracy_pct:.1f}%[/{acc_col}] ({cal.correct_calls} Treffer / {cal.false_calls} Fehlanalysen)   |   "
            f"[bold]Durch Risk-Guardrails verhinderte Fehlkäufe:[/bold] [green]{cal.guardrail_saved_losses}[/green]\n"
            f"⚖️ [bold]Selbst-kalibrierte Faktor-Gewichte:[/bold] {weights_str}\n"
            f"[dim]🧠 Letzte Lerneffekte & Fehler-Diagnosen:\n{lessons_str}[/dim]",
            title="🧠 Selbstlernendes Diagnose- & Kalibrierungs-Gedächtnis (Self-Improvement Loop)",
            border_style="yellow",
        )
    )


def render_portfolio_diagnostics(diagnostics: dict) -> None:
    if not diagnostics:
        return
    status = diagnostics.get("overall_status", "ON_PLAN")
    border_col = "green" if status == "ON_PLAN" else ("yellow" if status == "ATTENTION" else "red")
    lines = [
        f"[bold]{diagnostics.get('overall_badge', '')}[/bold]",
        f"{diagnostics.get('overall_summary', '')}",
    ]
    for note in diagnostics.get("system_notes", []):
        lines.append(f"[cyan]{note}[/cyan]")

    pos_audits = diagnostics.get("position_audits", [])
    if pos_audits:
        lines.append("\n[bold underline]Einzel-Positions-Audit (Warum im Plus/Minus? Welche Berechnung? Läuft es nach Plan?):[/bold underline]")
        for pa in pos_audits:
            p_col = "green" if pa["pnl_usd"] >= 0 else "red"
            lines.append(
                f"• [bold]{pa['symbol']}[/bold] ({pa['quantity']:g} Stk. | Einstieg ${pa['entry_price']:,.2f} → Aktuell ${pa['current_price']:,.2f} | "
                f"[{p_col}]{pa['pnl_usd']:+.2f}$ / {pa['pnl_pct']:+.2f}%[/{p_col}]) — {pa['status_badge']}\n"
                f"   ├─ [bold]Warum im Plus/Minus?[/bold] {pa['why_in_minus_or_plus']}\n"
                f"   ├─ [bold]Berechnungs- & Faktor-Prüfung:[/bold] {pa['which_calculation_was_off']}\n"
                f"   └─ [bold]Plan-Urteil:[/bold] [dim]{pa['plan_verdict']}[/dim]"
            )

    console.print(
        Panel(
            "\n".join(lines),
            title="🩺 Autonome Konto-Diagnose & Fehler-Ursachen-Analyse (Root-Cause & Plan-Audit)",
            border_style=border_col,
        )
    )


def render_adas_swarm_summary(adas: ADASMetaArchitect) -> None:
    st = adas.state
    ag_table = Table(
        title=(
            f"🧬 ADAS Meta-Agent & Autonomer Unteragenten-Schwarm "
            f"(Generation #{st.evolution_generation} · {len(st.active_subagents)} aktive Spezial-Agenten · {st.total_evolution_cycles} Evolutions-Zyklen)"
        ),
        box=box.SIMPLE,
        header_style="bold cyan",
    )
    ag_table.add_column("Unteragent (Spezialist)", style="bold")
    ag_table.add_column("Herkunft / Gen", justify="center")
    ag_table.add_column("Spezial-Aufgabe & Fokus", max_width=52)
    ag_table.add_column("Stimmgewicht\n(Trust)", justify="right")
    ag_table.add_column("Trefferquote\n(Votes)", justify="right")
    ag_table.add_column("Letztes Urteil", justify="right")

    for ag in st.active_subagents.values():
        origin_badge = f"[bold magenta]🧬 ADAS-Spawn (G{ag.generation})[/bold magenta]" if ag.is_spawned_by_adas else f"[dim]Kern (G{ag.generation})[/dim]"
        w_col = "green" if ag.trust_weight >= 1.15 else ("yellow" if ag.trust_weight >= 0.95 else "red")
        acc_str = f"{ag.accuracy_pct:.1f}% ({ag.correct_votes}/{ag.signals_evaluated})" if ag.signals_evaluated > 0 else "Kalibriert..."
        ag_table.add_row(
            ag.name,
            origin_badge,
            ag.role,
            f"[{w_col} bold]{ag.trust_weight:.2f}x[/{w_col} bold]",
            acc_str,
            ag.last_verdict_summary or "-",
        )

    console.print(ag_table)
    evo_logs = "\n".join(f"• {e}" for e in st.evolution_log[-4:]) if st.evolution_log else "Keine Evolutions-Events."
    recent_reflections = "\n".join(f"• {r}" for r in adas.recent_reflections(4)) if hasattr(adas, "recent_reflections") and adas.recent_reflections(4) else "Noch keine reflexiven Nachanalysen."
    console.print(
        Panel(
            f"[dim]{evo_logs}[/dim]\n\n"
            f"[bold cyan]🔍 Schritt-für-Schritt Selbst-Reflexion der Unteragenten:[/bold cyan]\n"
            f"[dim]{recent_reflections}[/dim]",
            title="🧬 ADAS Selbst-Evolutions- & Reflexions-Protokoll",
            border_style="magenta",
        )
    )


def format_fundamentals_and_corr_cell(snap: MarketSnapshot, blackout_days: int, config_lev: float = 2.0) -> str:
    lines = []
    # Beta & Korrelation zum aktuellen Bestand
    r_val = snap.max_corr_with_portfolio
    r_col = "red" if r_val >= 0.80 else ("yellow" if r_val >= 0.65 else "green")
    if snap.most_correlated_holding != "-":
        lines.append(f"β:{snap.beta_vs_spy:.2f} | [{r_col}]r:{r_val:+.2f}({snap.most_correlated_holding})[/{r_col}]")
    else:
        lines.append(f"β:{snap.beta_vs_spy:.2f} | [green]r:0.00[/green]")

    if snap.options_put_call_ratio is not None:
        pcr = snap.options_put_call_ratio
        iv_txt = f" IV:{snap.options_implied_vol_pct:.0f}%" if snap.options_implied_vol_pct else ""
        if snap.options_flow_sentiment == "BULLISH_CALL_FLOW":
            lines.append(f"[bold green]🎯 Opt P/C:{pcr:.2f} (Calls↑){iv_txt}[/bold green]")
        elif snap.options_flow_sentiment == "BEARISH_PUT_FLOW":
            lines.append(f"[bold red]🎯 Opt P/C:{pcr:.2f} (Puts↑){iv_txt}[/bold red]")
        else:
            lines.append(f"[dim]🎯 Opt P/C:{pcr:.2f}{iv_txt}[/dim]")
        if snap.best_option_contract:
            lines.append(f"[cyan]🎫 {snap.best_option_contract[:29]}[/cyan]")

    if snap.intraday_vwap is not None or abs(snap.relative_strength_vs_spy) >= 0.5:
        rs_col = "green" if snap.relative_strength_vs_spy >= 0 else "red"
        vwap_tag = (
            "[green]>VWAP[/green]"
            if snap.intraday_15m_trend == "BULLISH_ABOVE_VWAP"
            else ("[red]<VWAP[/red]" if snap.intraday_15m_trend in ("BEARISH_BELOW_VWAP", "FADING_FROM_HIGH") else "≈VWAP")
        )
        lines.append(f"{vwap_tag} | [{rs_col}]RS:{snap.relative_strength_vs_spy:+.1f}%[/{rs_col}]")

    if snap.asset_class == "CRYPTO":
        lines.append(f"[bold cyan]⚡{config_lev:.0f}x Margin[/bold cyan] | Vol:{snap.volatility_annualized_pct:.0f}%")
        return "\n".join(lines)
    if snap.is_safe_haven:
        lines.append("[bold yellow]🛡️ Gold-Hedge (1x)[/bold yellow]")
        return "\n".join(lines)
    if snap.intrinsic_leverage > 1.0:
        lines.append(f"[bold magenta]🚀 {snap.intrinsic_leverage:.0f}x Lev-ETF[/bold magenta]")
        return "\n".join(lines)
    if snap.forward_pe is None and snap.revenue_growth_pct is None and snap.days_to_earnings is None:
        lines.append("[dim]Index-ETF[/dim]")
        return "\n".join(lines)

    if snap.forward_pe is not None and snap.revenue_growth_pct is not None:
        g_col = "green" if snap.revenue_growth_pct >= 10 else ("red" if snap.revenue_growth_pct < 0 else "white")
        lines.append(f"KGV:{snap.forward_pe:.1f} | [{g_col}]Ums:{snap.revenue_growth_pct:+.0f}%[/{g_col}]")
    if snap.days_to_earnings is not None:
        if snap.days_to_earnings <= blackout_days:
            lines.append(f"[bold red]⚠️ Zahlen: {snap.days_to_earnings}d[/bold red]")
        else:
            lines.append(f"[dim]Zahlen: {snap.days_to_earnings}d[/dim]")
    return "\n".join(lines)


def run_autonomous_cycle(
    config: AgentConfig,
    collector: MultiAssetDataCollector,
    analyst: HybridAIAnalyst,
    risk_mgr: DeterministicRiskManager,
    engine: MultiAssetExecutionEngine,
) -> None:
    console.rule("[bold blue]🤖 Autonomer Multi-Asset Analyse-, Korrelations- & Ausführungs-Zyklus[/bold blue]")

    snapshots: Dict[str, MarketSnapshot] = {}
    with console.status("[bold green]Scanne 110+ globale Aktien & 9 Live-Screener, filtere Top-Action, lege Aufbaukandidaten auf 'Weiterbeobachten' & sortiere Langweiler aus..."):
        macro = collector.fetch_macro_regime()
        active_watchlist, hot_notes, pruned_boring = collector.filter_active_stocks_in_play(
            candidate_pool=config.watchlist,
            open_position_symbols=set(engine.positions.keys()),
            max_active=config.max_active_watchlist_size,
        )
        # Parallele, asynchrone Multi-Threaded Marktdaten-Erfassung (6x Geschwindigkeits-Boost)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_item = {executor.submit(collector.fetch_snapshot, item): item for item in active_watchlist}
            for future in as_completed(future_to_item):
                try:
                    snap = future.result()
                    if snap:
                        snapshots[snap.symbol] = snap
                except Exception as e:
                    logger.debug(f"Snapshot Fetch Error: {e}")

    watch_closely_str = (
        "  |  ".join(collector.last_watch_closely_notes[:8])
        if collector.last_watch_closely_notes
        else "Keine Kandidaten in Lauerstellung"
    )
    promo_line = (
        f"\n🚀 [bold cyan]Aus 'Weiterbeobachten' in den aktiven Handel gezündet:[/bold cyan] {'  |  '.join(collector.last_promoted_notes)}"
        if collector.last_promoted_notes
        else ""
    )
    console.print(
        Panel(
            f"🌍 [bold]Breitband-Markt-Sweep:[/bold] [bold white]{collector.last_total_scanned_count} Aktien & Options-Ketten geprüft[/bold white] (US, DAX/EU, China, Biotech, 9 Live-Screener)"
            + promo_line
            + f"\n🔥 [bold green]Zone 1 – Sofortige Tiefen- & Options-Analyse ({len(active_watchlist)} aktiv):[/bold green] "
            + ("  |  ".join(hot_notes[:7]) if hot_notes else "Standard-Radar")
            + f"\n👀 [bold yellow]Zone 2 – 'Weiterbeobachten / Auf der Lauer' ({len(collector.last_watch_closely_notes)} im Gedächtnis für nächste Zyklen):[/bold yellow] "
            + watch_closely_str
            + f"\n⏭️ [dim]Zone 3 – 'Weitergucken / Aktuell uninteressant' ({len(pruned_boring)} verworfen): "
            + (", ".join(pruned_boring[:22]) + (" ..." if len(pruned_boring) > 22 else "") if pruned_boring else "Keine")
            + "[/dim]",
            title="📡 Dynamischer 3-Zonen Markt- & Options-Trichter (Checken ➔ Handeln vs. Weiterbeobachten vs. Weitergucken)",
            border_style="magenta",
        )
    )

    # Makro- & Intermarket-Ampel + Börsenzeiten-Status anzeigen
    macro_border = "green" if macro.regime_state == "GREEN_RISK_ON" else ("yellow" if macro.regime_state == "YELLOW_CAUTION" else "red")
    intermarket_str = "  |  ".join(macro.intermarket_signals)
    us_open, mkt_status = is_us_equity_market_open()
    mkt_badge = f"[bold green]🕒 {mkt_status}[/bold green]" if us_open else f"[bold yellow]🕒 {mkt_status}[/bold yellow]"
    console.print(
        Panel(
            f"[bold]{macro.summary}[/bold]   |   {mkt_badge}   |   [bold cyan]Krypto: 24/7 OFFEN[/bold cyan]\n"
            f"🌐 [bold]Intermarket-Vorläufer:[/bold] {intermarket_str}\n"
            f"[dim]VIX: {macro.vix_level:.2f} | SPY: ${macro.spy_price:,.2f} (SMA50: ${macro.spy_sma_50:,.2f}) | "
            f"10Y US-Yield: {macro.us10y_yield:.2f}% ({macro.us10y_change_5d_bps:+.0f} bps 5d) | "
            f"US-Dollar (DXY): {macro.dxy_level:.2f} ({macro.dxy_change_5d_pct:+.2f}% 5d)[/dim]",
            title="🧭 Makro-, Intermarket- & Börsenzeiten-Radar (VIX / Zinsen / US-Dollar / NYSE)",
            border_style=macro_border,
        )
    )

    # 1. Aktualisiere offene Positionen und prüfe 50%-Scale-Out (TP1) / Trailing-Stop / Stop-Loss / TP2
    stop_events = engine.update_prices_and_check_stops(snapshots)
    for ev in stop_events:
        console.print(f"[bold yellow]⚡ {ev}[/bold yellow]")

    # 1b. Selbstlernendes Shadow-Tracking initialisieren, inaktive Aktien aussortieren & laufende Hypothesen prüfen
    shadow_tracker = SelfLearningShadowTracker(config)
    shadow_tracker.prune_inactive_symbols(set(snapshots.keys()), set(engine.positions.keys()))
    shadow_tracker.bootstrap_historical_calibration(snapshots, collector.return_histories)
    learning_events = shadow_tracker.evaluate_existing_predictions(snapshots)
    for lev in learning_events:
        console.print(f"[bold magenta]{lev}[/bold magenta]")

    # Berechne initiale 60-Tage-Korrelations-Matrix & Kovarianz-VaR
    port_risk = collector.compute_cross_asset_risk(snapshots, engine.positions, engine.total_equity_usd)

    # 1c. Walk-Forward Hyperparameter Auto-Tuning (Kalibriert SL, TP, CRV und Konfidenz dynamisch an VIX & Regime)
    try:
        from hyperparameter_autotuner import HyperparameterAutoTuner
        tuner = HyperparameterAutoTuner()
        calib = tuner.tune_parameters(macro)
        console.print(f"[bold cyan]🎯 Hyperparameter Auto-Tuner:[/bold cyan] {calib.tuning_rationale} (SL: {calib.atr_stop_loss_mult}x ATR, TP: {calib.atr_take_profit_mult}x ATR, CRV: {calib.target_risk_reward_ratio}:1)")
    except Exception as e:
        logger.debug(f"AutoTuner Fehler: {e}")

    # 1d. ADAS (Automated Design of Agentic Systems) Meta-Architekt & Unteragenten-Schwarm:
    #     Erschafft bei neuen Markt-Phänomenen autonom neue Spezial-Unteragenten und lässt den gesamten Schwarm abstimmen!
    adas = ADASMetaArchitect(config)
    adas_events = adas.run_evolution_and_spawning_cycle(snapshots, macro, engine.positions)
    for aev in adas_events:
        console.print(f"[bold magenta]{aev}[/bold magenta]")

    # 2. Cross-Sectional Signal-Analyse, ADAS-Schwarm-Abstimmung & Priorisierung
    analyzed_pairs: List[Tuple[MarketSnapshot, TradingSignal]] = []
    for sym, snap in snapshots.items():
        base_sig = analyst.analyze(
            snap,
            learned_weights=shadow_tracker.calibration.weights,
            recent_lessons=shadow_tracker.calibration.recent_lessons,
        )
        swarm_sig, _, _ = adas.evaluate_symbol_with_swarm(snap, base_sig, macro)
        analyzed_pairs.append((snap, swarm_sig))
    adas._save()

    def allocation_priority(pair: Tuple[MarketSnapshot, TradingSignal]) -> Tuple[int, float]:
        action = pair[1].action
        edge = pair[1].expected_edge_score
        if action == "SELL":
            return (0, -edge)
        elif action == "BUY":
            return (1, -edge)
        return (2, 0.0)

    sorted_pairs = sorted(analyzed_pairs, key=allocation_priority)

    analysis_table = Table(
        title="📊 Cross-Sectional Research-Ranking, 60d-Korrelation, Fundamentals & Risiko-Entscheidung",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
    )
    analysis_table.add_column("Asset /\nCluster", style="bold")
    analysis_table.add_column("Preis (24h)", justify="right")
    analysis_table.add_column("RSI / MACD", justify="right")
    analysis_table.add_column("Beta, Korrelation &\nFundamentals", justify="right")
    analysis_table.add_column("Signal\n(Edge-Score)", justify="center")
    analysis_table.add_column("SL / TP1 / TP2", justify="right")
    analysis_table.add_column("Research-These & Katalysatoren", max_width=34)
    analysis_table.add_column("Risk-Engine & Execution", max_width=34)

    if all(sig.action != "BUY" for _, sig in sorted_pairs):
        console.print(
            Panel(
                "[bold cyan]⏸️ SNIPER-GEDULD-MODUS AKTIV:[/bold cyan] Aktuell erfüllt keines der 11 Assets die strengen A+ Kaufsignale "
                f"(Min-Konfidenz {config.min_signal_confidence*100:.0f}%, Min-Edge {config.min_expected_edge_score:.2f}, mind. 2 bestätigte Faktoren ohne Gegenwind).\n"
                "👉 [bold green]Der Agent erzwingt keine mittelmäßigen Trades und hält dein freies Kapital sicher in Cash, bis eine glasklare Chance entsteht![/bold green]",
                border_style="cyan",
            )
        )

    entry_shadow_snapshot = dict(shadow_tracker.active_predictions)
    signal_map = {sig.symbol: sig for _, sig in sorted_pairs}

    # Aktive Opportunity-Cost-Rotation: Tausche schwache Alt-Positionen ohne Trend/Volumen gegen neue Top-A+-Leader
    rot_events = engine.rotate_weak_positions_for_better_setups(
        snapshots=snapshots,
        signals=signal_map,
        shadow_predictions=entry_shadow_snapshot,
    )
    for rev in rot_events:
        console.print(f"[bold cyan]🔄 {rev}[/bold cyan]")

    for snap, signal in sorted_pairs:
        # Aktualisiere paarweise Korrelation & Portfolio-VaR dynamisch nach jeder Bestandsänderung
        port_risk = collector.compute_cross_asset_risk(snapshots, engine.positions, engine.total_equity_usd)
        risk_eval = risk_mgr.evaluate_signal(
            signal=signal,
            snapshot=snap,
            macro=macro,
            cash_usd=engine.cash_usd,
            total_equity_usd=engine.total_equity_usd,
            daily_pnl_pct=engine.daily_pnl_pct,
            positions=engine.positions,
            portfolio_risk=port_risk,
        )

        # Autonome Deal-Verhandlung im emergenten Agenten-Dialekt
        try:
            from emergent_dialect_engine import get_dialect_engine
            dialect_eng = get_dialect_engine()
            dialect_eng.compose_and_negotiate_deal(
                snapshot=snap,
                signal=signal,
                macro=macro,
                risk_evaluation=risk_eval,
            )
        except Exception:
            pass

        exec_result = engine.execute_order(signal, risk_eval)
        # Zeichne JEDE Prognose (auch SELL ohne Bestand & abgelehnte BUY-Signale) im Shadow-Tracker auf!
        shadow_tracker.record_prediction(
            snap=snap,
            signal=signal,
            factor_contributions=analyst.last_factor_contributions.get(snap.symbol, {}),
            exec_result=exec_result,
            is_in_portfolio=(snap.symbol in engine.positions),
        )

        sig_color = "green" if signal.action == "BUY" else ("red" if signal.action == "SELL" else "dim")
        chg_color = "green" if snap.change_24h_pct >= 0 else "red"
        exec_color = (
            "bold green"
            if exec_result.startswith("EXECUTED")
            else (
                "bold magenta"
                if "RED-TEAM-VETO" in exec_result
                else (
                    "yellow"
                    if any(k in exec_result for k in ("EARNINGS", "MAKRO", "CLUSTER", "CASH-RESERVE", "KORRELATIONS", "VaR"))
                    else "dim"
                )
            )
        )

        eff_lev = round(
            (config.smart_leverage_max if (macro.regime_state == "GREEN_RISK_ON" and snap.intrinsic_leverage <= 1.0 and not snap.is_safe_haven) else 1.0)
            * snap.intrinsic_leverage,
            1,
        )
        lev_tag = f" [bold yellow]⚡{eff_lev:.0f}x[/bold yellow]" if eff_lev > 1.0 else ""
        rt_note = f"\n[dim cyan]⚖️ Red-Team: {signal.debate_result.verdict} (Risk: {signal.debate_result.risk_score:.1f}/10)[/dim cyan]" if signal.debate_result else ""
        analysis_table.add_row(
            f"{snap.symbol}{lev_tag}\n[dim]{snap.cluster}[/dim]",
            f"${snap.current_price:,.2f}\n[{chg_color}]{snap.change_24h_pct:+.2f}%[/{chg_color}]",
            f"RSI: {snap.rsi_14:.1f}\nMACD: {snap.macd_hist:+.2f}",
            format_fundamentals_and_corr_cell(snap, config.earnings_blackout_days, config_lev=eff_lev),
            f"[{sig_color} bold]{signal.action}[/{sig_color} bold]\n({signal.confidence*100:.0f}% | E:{signal.expected_edge_score:.2f})",
            f"SL: {signal.stop_loss_price:,.2f}\nTP1: {signal.partial_take_profit_price:,.2f}\nTP2: {signal.take_profit_price:,.2f}",
            f"{signal.thesis}\n[dim]📰 {signal.catalyst_summary}[/dim]{rt_note}",
            f"[{exec_color}]{exec_result}[/{exec_color}]",
        )

    signal_map = {sig.symbol: sig for _, sig in sorted_pairs}
    diagnostics = engine.diagnose_portfolio_health(
        snapshots=snapshots,
        signals=signal_map,
        shadow_predictions=entry_shadow_snapshot,
        factor_map=analyst.last_factor_contributions,
    )
    port_risk = collector.compute_cross_asset_risk(snapshots, engine.positions, engine.total_equity_usd)
    engine._save_state(port_risk.model_dump(), diagnostics_dict=diagnostics)
    console.print(analysis_table)
    render_portfolio_summary(engine, config, port_risk)
    render_portfolio_diagnostics(diagnostics)
    render_adas_swarm_summary(adas)
    render_shadow_learning_summary(shadow_tracker)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomer Multi-Asset Finance & Execution Agent")
    parser.add_argument("--once", action="store_true", help="Führt genau einen vollständigen Analyse- & Trading-Zyklus aus")
    parser.add_argument("--loop", type=int, default=0, metavar="SECONDS", help="Läuft dauerhaft im Intervall (z.B. --loop 900 für alle 15 Min)")
    parser.add_argument("--status", action="store_true", help="Zeigt nur den aktuellen Portfolio-Status an")
    parser.add_argument("--backtest", type=str, nargs="?", const="1y", metavar="PERIOD", help="Startet Walk-Forward Backtest (z.B. 6mo, 1y, 2y)")
    parser.add_argument("--breed", type=int, nargs="?", const=2, metavar="GENERATIONS", help="Startet genetische Strategie-Zucht (z.B. --breed 2)")
    parser.add_argument("--dashboard", action="store_true", help="Generiert das interaktive HTML Web-Dashboard (data/dashboard.html)")
    args = parser.parse_args()

    config = AgentConfig()
    collector = MultiAssetDataCollector()
    analyst = HybridAIAnalyst(config)
    risk_mgr = DeterministicRiskManager(config)
    engine = MultiAssetExecutionEngine(config)

    if args.backtest:
        bt = WalkForwardBacktester(config)
        bt.run_backtest(period=args.backtest)
        dash = DashboardGenerator(config)
        path = dash.generate(open_browser=args.dashboard)
        console.print(f"[bold green]✔ Interaktives Web-Dashboard aktualisiert:[/bold green] {path}")
        return

    if args.breed:
        from genetic_strategy_breeder import run_strategy_breeding_cycle
        run_strategy_breeding_cycle(num_generations=args.breed, pop_per_gen=6)
        dash = DashboardGenerator(config)
        path = dash.generate(open_browser=args.dashboard)
        console.print(f"[bold green]✔ Zucht abgeschlossen & Web-Dashboard aktualisiert:[/bold green] {path}")
        return

    if args.dashboard:
        dash = DashboardGenerator(config)
        path = dash.generate(open_browser=True)
        console.print(f"[bold green]✔ Interaktives Web-Dashboard erstellt & geöffnet:[/bold green] {path}")
        return

    if args.status:
        render_portfolio_summary(engine, config)
        return

    # Telegram-Fernsteuerungs-Listener starten (erlaubt /status, /portfolio, /debates, /breed per Telegram)
    if config.telegram_bot_token and config.telegram_chat_id:
        try:
            from telegram_command_listener import TelegramCommandListener
            from genetic_strategy_breeder import run_strategy_breeding_cycle

            def _get_live_portfolio():
                return [
                    {
                        "symbol": p.symbol,
                        "qty": p.quantity,
                        "avg_entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "unrealized_pl": p.unrealized_pnl_usd,
                        "unrealized_plpc": p.unrealized_pnl_pct / 100.0,
                    }
                    for p in engine.positions.values()
                ]

            def _get_live_status():
                return {
                    "equity": engine.total_equity_usd,
                    "cash": engine.cash_usd,
                    "regime": "AUTONOMOUS_SWARM_ACTIVE",
                }

            def _trigger_breed():
                run_strategy_breeding_cycle(num_generations=1, pop_per_gen=4)
                return "1 Generation erfolgreich gezüchtet & im Schwarm integriert!"

            tg_listener = TelegramCommandListener(
                bot_token=config.telegram_bot_token,
                authorized_chat_id=config.telegram_chat_id,
                portfolio_callback=_get_live_portfolio,
                status_callback=_get_live_status,
                breed_callback=_trigger_breed,
            )
            tg_listener.start()
            console.print("[bold green]✔ Telegram-Fernsteuerungs-Listener aktiv[/bold green] (Antwortet auf /status, /portfolio, /trades, /debates, /breed)")
        except Exception as e:
            console.print(f"[yellow]Warnung: Telegram-Listener konnte nicht gestartet werden: {e}[/yellow]")

    if args.loop > 0:
        console.print(f"[bold green]Starte autonomen Dauerbetrieb (Intervall: {args.loop} Sekunden)...[/bold green]")
        while True:
            run_autonomous_cycle(config, collector, analyst, risk_mgr, engine)
            DashboardGenerator(config).generate(open_browser=False)
            console.print(f"[dim]Nächster Zyklus in {args.loop} Sekunden (Strg+C zum Beenden)...[/dim]")
            time.sleep(args.loop)
    else:
        run_autonomous_cycle(config, collector, analyst, risk_mgr, engine)
        path = DashboardGenerator(config).generate(open_browser=False)
        console.print(f"[bold green]✔ Interaktives Web-Dashboard aktualisiert:[/bold green] {path}")


if __name__ == "__main__":
    main()
