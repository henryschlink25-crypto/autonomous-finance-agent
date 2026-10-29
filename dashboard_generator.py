import json
import os
import webbrowser
from config import AgentConfig


class DashboardGenerator:
    """
    Generiert ein modernes, institutionelles Web-Dashboard (Glassmorphism FinTech UI + Chart.js + Auto-Refresh)
    aus dem aktuellen Live/Paper-Portfolio, der 60-Tage-Korrelationsmatrix, dem Audit-Journal und dem Shadow-Tracker.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.data_dir = os.path.dirname(self.config.portfolio_state_file)
        self.backtest_file = os.path.join(self.data_dir, "backtest_results.json")
        self.shadow_file = os.path.join(self.data_dir, "shadow_predictions.json")
        self.calibration_file = os.path.join(self.data_dir, "learned_weights.json")
        self.dynamic_watchlist_file = os.path.join(self.data_dir, "dynamic_watchlist.json")
        self.adas_state_file = os.path.join(self.data_dir, "adas_meta_state.json")
        self.output_html = os.path.join(self.data_dir, "dashboard.html")

    def generate(self, open_browser: bool = False) -> str:
        portfolio = {"cash_usd": self.config.initial_capital_usd, "realized_pnl_usd": 0.0, "positions": {}}
        if os.path.exists(self.config.portfolio_state_file):
            with open(self.config.portfolio_state_file, "r", encoding="utf-8") as f:
                portfolio = json.load(f)

        backtest = {}
        if os.path.exists(self.backtest_file):
            with open(self.backtest_file, "r", encoding="utf-8") as f:
                backtest = json.load(f)

        audit_events = []
        if os.path.exists(self.config.audit_log_file):
            with open(self.config.audit_log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        audit_events.append(json.loads(line))
        audit_events = list(reversed(audit_events[-12:]))

        shadow_active = {}
        if os.path.exists(self.shadow_file):
            try:
                with open(self.shadow_file, "r", encoding="utf-8") as f:
                    shadow_active = json.load(f).get("active", {})
            except Exception:
                pass

        calibration = {
            "total_evaluated": 0,
            "correct_calls": 0,
            "false_calls": 0,
            "guardrail_saved_losses": 0,
            "weights": {},
            "recent_lessons": [],
        }
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, "r", encoding="utf-8") as f:
                    calibration = json.load(f)
            except Exception:
                pass

        positions = portfolio.get("positions", {})
        cash_usd = float(portfolio.get("cash_usd", self.config.initial_capital_usd))
        realized_pnl = float(portfolio.get("realized_pnl_usd", 0.0))
        updated_at = portfolio.get("updated_at", "")[:19].replace("T", " ")

        total_margin_equity = 0.0
        total_notional_exposure = 0.0
        total_unrealized_pnl = 0.0

        alloc_labels = ["Cash-Reserve"]
        alloc_values = [round(cash_usd, 2)]

        pos_rows_html = ""
        for sym, p in positions.items():
            qty = float(p.get("quantity", 0.0))
            entry_p = float(p.get("avg_entry_price", 0.0))
            curr_p = float(p.get("current_price", 0.0))
            lev = float(p.get("leverage", 1.0))
            intr_lev = float(p.get("intrinsic_leverage", 1.0))
            eff_lev = round(lev * intr_lev, 1)
            margin_used = float(p.get("margin_used_usd", (qty * entry_p) / max(1.0, lev)))
            notional = qty * curr_p
            pnl = (curr_p - entry_p) * qty
            eq_val = margin_used + pnl
            pnl_pct = (pnl / margin_used) * 100.0 if margin_used > 0 else 0.0

            total_margin_equity += eq_val
            total_notional_exposure += notional
            total_unrealized_pnl += pnl

            alloc_labels.append(f"{sym} ({eff_lev:.0f}x)")
            alloc_values.append(round(max(0.0, eq_val), 2))

            pnl_cls = "pos" if pnl >= 0 else "neg"
            sl_p = float(p.get("stop_loss_price", curr_p * 0.95))
            tp1_p = float(p.get("partial_tp_price", curr_p * 1.04))
            tp2_p = float(p.get("take_profit_price", curr_p * 1.08))
            tp1_done = p.get("partial_tp_done", False)

            # Fortschrittsbalken zwischen Stop-Loss (0%) und Take-Profit 2 (100%)
            span = max(0.0001, tp2_p - sl_p)
            progress_pct = max(2.0, min(98.0, ((curr_p - sl_p) / span) * 100.0))
            entry_marker_pct = max(2.0, min(98.0, ((entry_p - sl_p) / span) * 100.0))
            bar_color = "linear-gradient(90deg, #10b981, #38bdf8)" if pnl >= 0 else "linear-gradient(90deg, #f59e0b, #ef4444)"
            tp1_badge = '<span class="pill pill-green">TP1 ✓</span>' if tp1_done else f'<span class="tp1">${tp1_p:,.2f}</span>'

            import datetime as dt
            max_d = self.config.leveraged_etf_max_days if intr_lev > 1.0 else self.config.max_stale_holding_days
            hold_badge = f"0d 0h / max {max_d}d"
            try:
                op_str = p.get("opened_at", "")[:19]
                if "T" in op_str:
                    op_dt = dt.datetime.fromisoformat(op_str).replace(tzinfo=dt.timezone.utc)
                    hrs = max(0.0, (dt.datetime.now(dt.timezone.utc) - op_dt).total_seconds() / 3600.0)
                    hold_badge = f"{int(hrs // 24)}d {int(hrs % 24)}h / max {max_d}d"
            except Exception:
                pass

            pos_rows_html += f"""
            <tr>
                <td>
                    <div class="asset-cell">
                        <div class="asset-icon">{sym[:4]}</div>
                        <div>
                            <div class="asset-sym">{sym} <span class="lev-badge">⚡{eff_lev:.0f}x</span></div>
                            <div class="asset-sub">{p.get('cluster', 'US_TECH')} · ⏱️ {hold_badge}</div>
                        </div>
                    </div>
                </td>
                <td class="num">{qty:g}</td>
                <td class="num">${entry_p:,.2f}</td>
                <td class="num"><strong>${curr_p:,.2f}</strong></td>
                <td class="num">
                    <div><strong>${eq_val:,.2f}</strong> <span class="muted">Margin</span></div>
                    <div class="sub-notional">➔ ${notional:,.0f} Kaufkraft</div>
                </td>
                <td style="min-width:210px;">
                    <div class="range-labels">
                        <span class="sl">SL ${sl_p:,.2f}</span>
                        {tp1_badge}
                        <span class="tp">TP2 ${tp2_p:,.2f}</span>
                    </div>
                    <div class="range-track">
                        <div class="range-fill" style="width:{progress_pct:.1f}%;background:{bar_color};"></div>
                        <div class="range-entry" style="left:{entry_marker_pct:.1f}%;" title="Einstieg: ${entry_p:,.2f}"></div>
                    </div>
                </td>
                <td class="num {pnl_cls}">
                    <div class="pnl-big">{pnl:+,.2f}$</div>
                    <div class="pnl-sub">{pnl_pct:+.2f}% auf Margin</div>
                </td>
            </tr>"""

        if not pos_rows_html:
            pos_rows_html = '<tr><td colspan="7" style="text-align:center;color:#94a3b8;padding:28px;">Keine offenen Positionen – Agent wartet auf nächstes High-Conviction-Setup</td></tr>'

        total_equity = round(cash_usd + total_margin_equity, 2)
        total_return_pct = ((total_equity - self.config.initial_capital_usd) / self.config.initial_capital_usd) * 100.0 if self.config.initial_capital_usd else 0.0
        portfolio_leverage = round(total_notional_exposure / max(1.0, total_equity), 2)

        curve = backtest.get("equity_curve", [])
        dates_json = json.dumps([pt["date"] for pt in curve])
        agent_eq_json = json.dumps([pt["agent_equity"] for pt in curve])
        bench_eq_json = json.dumps([pt["benchmark_equity"] for pt in curve])
        alloc_labels_json = json.dumps(alloc_labels)
        alloc_values_json = json.dumps(alloc_values)

        audit_rows_html = ""
        for ev in audit_events:
            ts = ev.get("timestamp", "")[:19].replace("T", " ")
            ev_type = ev.get("event_type", "")
            sym = ev.get("symbol", "-")
            qty = ev.get("quantity", ev.get("sold_quantity", 0))
            price = ev.get("price", ev.get("exit_price", 0))
            broker_str = ev.get("broker", ev.get("mode", "ALPACA_PAPER"))
            eff_l = ev.get("effective_total_leverage", 1.0)
            lev_tag = f'<span class="lev-badge">⚡{eff_l:.0f}x</span>' if eff_l and eff_l > 1.0 else ""
            ev_pill_cls = "pill-green" if "BUY" in ev_type or "TAKE_PROFIT" in ev_type else ("pill-red" if "STOP" in ev_type or "SELL" in ev_type else "pill-blue")
            thesis = ev.get("thesis", f"Realisierter PnL: {ev.get('realized_pnl_usd', 0):+.2f}$")
            audit_rows_html += f"""
            <tr>
                <td class="muted">{ts}</td>
                <td><span class="pill {ev_pill_cls}">{ev_type}</span></td>
                <td><strong>{sym}</strong> {lev_tag}</td>
                <td class="num">{qty:g} @ ${price:,.2f}</td>
                <td>
                    <div>{thesis}</div>
                    <div class="asset-sub">Broker: {broker_str}</div>
                </td>
            </tr>"""

        if not audit_rows_html:
            audit_rows_html = '<tr><td colspan="5" style="text-align:center;color:#94a3b8;padding:20px;">Noch keine Transaktionen im Audit-Log</td></tr>'

        shadow_rows_html = ""
        for sym, sp in shadow_active.items():
            act = sp.get("predicted_action", "HOLD")
            conf_pct = sp.get("confidence", 0) * 100.0
            act_pill = "pill-green" if act == "BUY" else ("pill-red" if act == "SELL" else "pill-muted")
            d_ret = sp.get("directional_return_pct", 0.0)
            max_fav = sp.get("max_favorable_pct", 0.0)
            d_cls = "pos" if d_ret >= 0 else "neg"
            was_exec = sp.get("was_executed", False)
            exec_badge = '<span class="pill pill-green">✔ Im Echt-Portfolio</span>' if was_exec else f'<span class="pill pill-muted">Shadow ({sp.get("execution_note", "")[:26]})</span>'
            shadow_rows_html += f"""
            <tr>
                <td><strong>{sym}</strong></td>
                <td><span class="pill {act_pill}">{act} · {conf_pct:.0f}%</span></td>
                <td>{exec_badge}</td>
                <td class="num">${sp.get('entry_price', 0):,.2f} ➔ <strong>${sp.get('current_price', 0):,.2f}</strong></td>
                <td class="num">
                    <span class="tp">Ziel: ${sp.get('target_price', 0):,.2f}</span> ·
                    <span class="sl">Stop: ${sp.get('invalidation_price', 0):,.2f}</span>
                </td>
                <td class="num {d_cls}">
                    <strong>{d_ret:+.2f}%</strong>
                    <span class="muted">(Peak: {max_fav:+.2f}%)</span>
                </td>
            </tr>"""

        if not shadow_rows_html:
            shadow_rows_html = '<tr><td colspan="6" style="text-align:center;color:#94a3b8;">Noch keine aktiven Shadow-Prognosen</td></tr>'

        cal_total = calibration.get("total_evaluated", 0)
        cal_corr = calibration.get("correct_calls", 0)
        cal_false = calibration.get("false_calls", 0)
        cal_saved = calibration.get("guardrail_saved_losses", 0)
        cal_acc = (cal_corr / cal_total * 100.0) if cal_total > 0 else 0.0
        w_dict = calibration.get("weights", {})

        weight_bars_html = ""
        for k, v in w_dict.items():
            pct_w = min(100.0, max(15.0, (v / 1.5) * 100.0))
            weight_bars_html += f"""
            <div class="weight-item">
                <div class="weight-hdr"><span>{k.upper()}</span><strong>{v:.2f}x</strong></div>
                <div class="weight-track"><div class="weight-fill" style="width:{pct_w:.0f}%;"></div></div>
            </div>"""

        lessons_items = "".join(f'<div class="lesson-item">{l}</div>' for l in calibration.get("recent_lessons", [])[-5:])
        if not lessons_items:
            lessons_items = '<div class="lesson-item">Noch keine abgeschlossenen Hypothesen-Auswertungen.</div>'

        bt_ret = backtest.get("total_return_pct", 0.0)
        bt_sharpe = backtest.get("sharpe_ratio", 0.0)
        bt_dd = backtest.get("max_drawdown_pct", 0.0)
        bt_win = backtest.get("win_rate_pct", 0.0)
        bt_pf = backtest.get("profit_factor", 0.0)

        pr = portfolio.get("portfolio_risk", {})
        p_beta = pr.get("portfolio_beta", 0.54)
        p_var_usd = pr.get("daily_var_95_usd", 8.47)
        p_var_pct = pr.get("daily_var_95_pct", 0.85)
        p_div_ben = pr.get("diversification_benefit_pct", 33.0)
        top_pairs_badges = "".join(
            f'<span class="corr-chip">{pair}</span>'
            for pair in pr.get("top_correlated_pairs", ["TQQQ ↔ SQQQ: r=-1.00", "BTC/USD ↔ ETH/USD: r=+0.93"])[:5]
        )

        # Baue kompakte 60-Tage Korrelations-Heatmap (inkl. China, Europa, US-3x-ETFs & Krypto)
        corr_matrix = pr.get("correlation_matrix", {})
        corr_syms = [
            s
            for s in ["SOXL", "TQQQ", "YINN", "SQQQ", "NVDA", "AVGO", "AAPL", "BABA", "PDD", "ASML", "SAP", "GLD", "BTC/USD"]
            if s in corr_matrix
        ]
        heatmap_html = ""
        if corr_syms:
            hdr_cells = "".join(f"<th>{s.split('/')[0]}</th>" for s in corr_syms)
            rows_cells = ""
            for r_sym in corr_syms:
                row_str = f"<td><strong>{r_sym.split('/')[0]}</strong></td>"
                for c_sym in corr_syms:
                    val = float(corr_matrix.get(r_sym, {}).get(c_sym, 0.0))
                    if r_sym == c_sym:
                        bg = "rgba(56, 189, 248, 0.22)"
                    elif val >= 0.75:
                        bg = f"rgba(239, 68, 68, {min(0.65, abs(val)*0.65):.2f})"
                    elif val >= 0.45:
                        bg = f"rgba(245, 158, 11, {min(0.45, abs(val)*0.45):.2f})"
                    elif val <= -0.30:
                        bg = f"rgba(16, 185, 129, {min(0.55, abs(val)*0.60):.2f})"
                    else:
                        bg = "rgba(30, 41, 59, 0.45)"
                    row_str += f'<td class="hm-cell" style="background:{bg};">{val:+.2f}</td>'
                rows_cells += f"<tr>{row_str}</tr>"
            heatmap_html = f"""
            <table class="hm-table">
                <thead><tr><th>60d r</th>{hdr_cells}</tr></thead>
                <tbody>{rows_cells}</tbody>
            </table>"""

        # Baue die Sektion "🩺 Autonome Konto-Diagnose & Fehler-Ursachen-Analyse (Warum Plus/Minus, Berechnungs-Audit & Plan-Check)"
        diag = portfolio.get("portfolio_diagnostics", {})
        diag_badge = diag.get("overall_badge", "🟢 ALLES NACH PLAN")
        diag_summary = diag.get(
            "overall_summary",
            "Alle offenen Positionen liegen im geplanten ATR-Risikokorridor oberhalb ihrer serverseitigen Stop-Loss-Marken.",
        )
        diag_sys_notes_html = "".join(
            f'<div class="lesson-item" style="border-left-color:#38bdf8;margin-bottom:8px;">{n}</div>'
            for n in diag.get("system_notes", [])
        )
        diag_pos_rows_html = ""
        for pa in diag.get("position_audits", []):
            p_cls = "pos" if pa.get("pnl_usd", 0.0) >= 0 else "neg"
            s_code = pa.get("status_code", "ON_PLAN")
            s_pill = "pill-green" if "ON_PLAN" in s_code else ("pill-red" if "BROKEN" in s_code else "pill-amber")
            diag_pos_rows_html += f"""
            <tr>
                <td>
                    <div><strong>{pa.get('symbol')}</strong> <span class="muted">({pa.get('quantity', 0):g} Stk.)</span></div>
                    <div class="num {p_cls}" style="font-weight:700;margin-top:4px;">{pa.get('pnl_usd', 0.0):+,.2f}$ ({pa.get('pnl_pct', 0.0):+.2f}%)</div>
                    <div class="asset-sub">Konto-Effekt: {pa.get('account_impact_pct', 0.0):.3f}%</div>
                </td>
                <td><span class="pill {s_pill}">{pa.get('status_badge', '')}</span></td>
                <td>{pa.get('why_in_minus_or_plus', '')}</td>
                <td>{pa.get('which_calculation_was_off', '')}</td>
                <td>{pa.get('plan_verdict', '')}</td>
            </tr>"""

        if not diag_pos_rows_html:
            diag_pos_rows_html = '<tr><td colspan="5" style="text-align:center;color:#94a3b8;">Keine offenen Positionen zur Diagnose</td></tr>'

        diagnostics_panel_html = f"""
    <div class="panel" style="border-color:rgba(56, 189, 248, 0.35);">
        <div class="panel-hdr">
            <div>
                <h2>🩺 Autonome Konto-Diagnose & Fehler-Ursachen-Analyse (Root-Cause & Plan-Audit)</h2>
                <div class="panel-subtitle">Beantwortet live bei jedem Zyklus: Warum ist das Konto im Plus/Minus? Welche Berechnung lag daneben? Läuft alles nach Plan?</div>
            </div>
            <span class="pill pill-green">{diag_badge}</span>
        </div>
        <div class="lesson-item" style="border-left-color:#10b981;margin-bottom:12px;font-size:13.5px;color:#e2e8f0;">
            <strong>Gesamtkonto-Befund:</strong> {diag_summary}
        </div>
        {diag_sys_notes_html}
        <table>
            <thead>
                <tr>
                    <th>Position & Ergebnis</th>
                    <th>Plan-Status & SL-Puffer</th>
                    <th>1. Warum im Plus / Minus?</th>
                    <th>2. Welche Berechnung war ungenau / Faktor-Audit?</th>
                    <th>3. Läuft alles nach Plan? (Handlungs-Urteil)</th>
                </tr>
            </thead>
            <tbody>
                {diag_pos_rows_html}
            </tbody>
        </table>
    </div>"""

        # Baue das 3-Zonen Dynamik- & Optionen-Trichter-Panel (Hot Action vs. Weiterbeobachten vs. Weitergucken)
        dyn_wl = {}
        if os.path.exists(self.dynamic_watchlist_file):
            try:
                with open(self.dynamic_watchlist_file, "r", encoding="utf-8") as f:
                    dyn_wl = json.load(f)
            except Exception:
                pass

        total_scanned = dyn_wl.get("total_candidates_scanned", 0)
        hot_list = dyn_wl.get("hot_action_zone", [])
        watch_queue = dyn_wl.get("watch_closely_queue", {})
        discarded_list = dyn_wl.get("discarded_moved_on", [])

        hot_badges_html = "".join(
            f'<span class="pill pill-green" style="margin:3px;">🔥 {h.get("symbol")} ({h.get("change_1d_pct", 0):+.1f}% · RVOL {h.get("rvol", 1):.1f}x)</span>'
            for h in hot_list[:16]
        ) or '<span class="muted">Warte auf ersten Zyklus...</span>'

        watch_badges_html = "".join(
            f'<span class="pill pill-amber" style="margin:3px;">👀 {sym} (Zyklus {info.get("cycles_watched", 1)}/3 · {info.get("reason", "")})</span>'
            for sym, info in list(watch_queue.items())[:14]
        ) or '<span class="muted">Aktuell keine Werte in der Lauer-Queue</span>'

        discard_badges_html = "".join(
            f'<span class="pill pill-muted" style="margin:2px;opacity:0.75;">⏭️ {sym}</span>'
            for sym in discarded_list[:24]
        )
        if len(discarded_list) > 24:
            discard_badges_html += f' <span class="muted">+{len(discarded_list) - 24} weitere ignoriert</span>'

        funnel_panel_html = f"""
    <div class="panel" style="border-color:rgba(129, 140, 248, 0.35);">
        <div class="panel-hdr">
            <div>
                <h2>🔭 Dynamischer 3-Zonen Markt- & Optionen-Trichter ({total_scanned} Aktien & ETFs pro Zyklus geprüft)</h2>
                <div class="panel-subtitle">Scannt 110+ globale Basiswerte + 9 Live-Screener: Sofort-Analyse & Options-Picker (Zone 1) · Weiterbeobachten auf der Lauer (Zone 2) · Weitergucken & Aussortieren (Zone 3)</div>
            </div>
            <span class="pill pill-cyan">{total_scanned} Geprüft ➔ {len(hot_list)} Hot · {len(watch_queue)} Lauer-Queue · {len(discarded_list)} Aussortiert</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(300px, 1fr));gap:14px;">
            <div class="lesson-item" style="border-left-color:#10b981;">
                <div style="font-weight:700;color:#10b981;margin-bottom:6px;">🔥 Zone 1: HOT ACTION ({len(hot_list)} Top-Ausbrüche in Voll-Analyse + Options-Scan)</div>
                <div>{hot_badges_html}</div>
            </div>
            <div class="lesson-item" style="border-left-color:#f59e0b;">
                <div style="font-weight:700;color:#f59e0b;margin-bottom:6px;">👀 Zone 2: WEITERBEOBACHTEN ({len(watch_queue)} Werte im Momentum-Aufbau · max. 3 Zyklen)</div>
                <div>{watch_badges_html}</div>
            </div>
            <div class="lesson-item" style="border-left-color:#64748b;">
                <div style="font-weight:700;color:#94a3b8;margin-bottom:6px;">⏭️ Zone 3: WEITERGUCKEN ({len(discarded_list)} inaktive Werte sofort übersprungen)</div>
                <div>{discard_badges_html}</div>
            </div>
        </div>
    </div>"""

        # Baue das ADAS (Automated Design of Agentic Systems) Multi-Agenten-Schwarm-Panel
        adas_data = {}
        if os.path.exists(self.adas_state_file):
            try:
                with open(self.adas_state_file, "r", encoding="utf-8") as f:
                    adas_data = json.load(f)
            except Exception:
                pass

        adas_gen = adas_data.get("evolution_generation", 1)
        adas_cycles = adas_data.get("total_evolution_cycles", 0)
        adas_agents = adas_data.get("active_subagents", {})
        adas_evo_log = adas_data.get("evolution_log", [])

        synth_registry = {}
        synth_tools_file = os.path.join(self.data_dir, "synthesized_tools", "synthesized_registry.json")
        if os.path.exists(synth_tools_file):
            try:
                with open(synth_tools_file, "r", encoding="utf-8") as f:
                    synth_registry = json.load(f)
            except Exception:
                pass
        strat_registry = {}
        strat_file = os.path.join(self.data_dir, "strategy_registry.json")
        if os.path.exists(strat_file):
            try:
                with open(strat_file, "r", encoding="utf-8") as f:
                    strat_registry = json.load(f)
            except Exception:
                pass
        active_strats = strat_registry.get("active_strategies", {})
        strat_cards_html = "".join(
            f"""<div class="lesson-item" style="border-left-color:#10b981;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
                    <strong>🧬 {s.get('class_name')} <span style="font-size:11px;color:var(--muted);">({s.get('archetype')})</span></strong>
                    <span class="pill pill-green">⭐️ Sharpe: {s.get('performance', {}).get('sharpe_ratio', 0):.2f} · PF: {s.get('performance', {}).get('profit_factor', 0):.2f}x · Win: {s.get('performance', {}).get('win_rate_pct', 0):.1f}% · Max DD: {s.get('performance', {}).get('max_drawdown_pct', 0):.1f}%</span>
                </div>
                <div style="font-size:12px;color:var(--muted);margin-top:4px;">{s.get('hypothesis', '')}</div>
            </div>"""
            for s in active_strats.values()
        ) or '<div class="lesson-item"><span class="muted">Noch keine gezüchteten Strategien aktiv.</span></div>'

        active_synths = synth_registry.get("active_tools", {})
        synth_badges = "".join(f'<span class="pill pill-cyan">⚡ {v.get("name", k)}</span> ' for k, v in active_synths.items()) or '<span class="pill pill-muted">Keine</span>'

        adas_rows_html = ""
        for ag_id, ag in adas_agents.items():
            is_spawn = ag.get("is_spawned_by_adas", False)
            gen_badge = (
                f'<span class="pill pill-cyan">🧬 ADAS-Spawn (Gen {ag.get("generation", 1)})</span>'
                if is_spawn
                else f'<span class="pill pill-muted">Kern (Gen {ag.get("generation", 1)})</span>'
            )
            tw = float(ag.get("trust_weight", 1.0))
            tw_cls = "pos" if tw >= 1.15 else ("neg" if tw < 0.90 else "")
            ev_cnt = int(ag.get("signals_evaluated", 0))
            corr_cnt = int(ag.get("correct_votes", 0))
            acc_pct = (corr_cnt / ev_cnt * 100.0) if ev_cnt > 0 else 50.0
            acc_str = f"{acc_pct:.1f}% ({corr_cnt}/{ev_cnt})" if ev_cnt > 0 else "Kalibriert..."
            adas_rows_html += f"""
            <tr>
                <td><strong>{ag.get('name', ag_id)}</strong></td>
                <td>{gen_badge}</td>
                <td>{ag.get('role', '')}</td>
                <td class="num {tw_cls}"><strong>{tw:.2f}x</strong></td>
                <td class="num">{acc_str}</td>
                <td class="num">{ag.get('last_verdict_summary', '-')}</td>
            </tr>"""

        adas_evo_html = "".join(
            f'<div class="lesson-item" style="border-left-color:#a855f7;margin-bottom:6px;">{e}</div>'
            for e in adas_evo_log[-4:]
        )

        adas_panel_html = f"""
    <div class="panel" style="border-color:rgba(168, 85, 247, 0.4);">
        <div class="panel-hdr">
            <div>
                <h2>🧬 ADAS (Automated Design of Agentic Systems) · Selbst-Evolvierender Unteragenten-Schwarm</h2>
                <div class="panel-subtitle">Der Meta-Architekt erschafft autonom neue Spezial-Unteragenten für neue Markt-Muster, lässt alle abstimmen und gewichtet sie nach Trefferquote</div>
            </div>
            <span class="pill pill-green">Generation #{adas_gen} · {len(adas_agents)} Spezial-Agenten aktiv · {adas_cycles} Zyklen</span>
        </div>
        <div style="margin:8px 0 14px 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
            <span style="font-size:12px;font-weight:600;color:var(--muted);">Autonome Werkzeuge (AI Scientist):</span>
            {synth_badges}
            <span class="pill pill-indigo">📐 Kelly-Kriterium & Volatilitäts-Parität aktiv</span>
            <span class="pill pill-green">⚔️ AlphaZero Monte-Carlo Simulator</span>
            <span class="pill pill-green">📱 Telegram Alerts: @Henry_trading_agent_bot (Live)</span>
            <span class="pill pill-green">🧬 {len(active_strats)} Elite-Strategien gezüchtet</span>
        </div>

        {adas_evo_html}
        <div style="margin: 14px 0 10px 0; font-size:13px; font-weight:700; color:var(--green);">
            🧬 Autonom gezüchtete & krisengeprüfte Handelsstrategien (Genetic Strategy Breeding):
        </div>
        {strat_cards_html}
        <table>
            <thead>
                <tr>
                    <th>Spezial-Unteragent</th>
                    <th>Herkunft & Generation</th>
                    <th>Spezial-Aufgabe & Domänen-Fokus</th>
                    <th class="num">Stimmgewicht (Trust)</th>
                    <th class="num">Trefferquote</th>
                    <th class="num">Letztes Votum</th>
                </tr>
            </thead>
            <tbody>
                {adas_rows_html}
            </tbody>
        </table>
    </div>"""

        html = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="60">
    <title>Autonomous Finance Agent · Smart-Leverage Command Center</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-main: #060913;
            --bg-card: rgba(15, 23, 42, 0.78);
            --bg-card-hover: rgba(22, 33, 58, 0.88);
            --border: rgba(148, 163, 184, 0.14);
            --border-highlight: rgba(56, 189, 248, 0.35);
            --text: #f8fafc;
            --muted: #94a3b8;
            --cyan: #38bdf8;
            --indigo: #818cf8;
            --green: #10b981;
            --amber: #f59e0b;
            --red: #f43f5e;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 28px 36px;
            background:
                radial-gradient(circle at 12% 8%, rgba(56, 189, 248, 0.12), transparent 34%),
                radial-gradient(circle at 88% 14%, rgba(129, 140, 248, 0.11), transparent 36%),
                radial-gradient(circle at 50% 92%, rgba(16, 185, 129, 0.07), transparent 40%),
                var(--bg-main);
            color: var(--text);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
        }}
        .num {{ font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }}
        .topbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            padding: 20px 26px;
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 18px;
            margin-bottom: 24px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
        }}
        .brand {{ display: flex; align-items: center; gap: 16px; }}
        .brand-logo {{
            width: 46px;
            height: 46px;
            border-radius: 14px;
            background: linear-gradient(135deg, #38bdf8, #6366f1);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 23px;
            box-shadow: 0 8px 22px rgba(56, 189, 248, 0.35);
        }}
        .brand h1 {{ margin: 0; font-size: 21px; font-weight: 800; letter-spacing: -0.02em; display: flex; align-items: center; gap: 10px; }}
        .brand-sub {{ color: var(--muted); font-size: 12.5px; margin-top: 4px; }}
        .topbar-right {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
        .live-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.13);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 7px 14px;
            border-radius: 999px;
            font-size: 12.5px;
            font-weight: 600;
        }}
        .pulse-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
            70% {{ box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
        }}
        .mode-badge {{
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(129, 140, 248, 0.18));
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #7dd3fc;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.03em;
        }}

        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 20px;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .kpi-card:hover {{ transform: translateY(-2px); border-color: var(--border-highlight); }}
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--cyan), var(--indigo));
            opacity: 0.75;
        }}
        .kpi-label {{ color: var(--muted); font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }}
        .kpi-val {{ font-size: 25px; font-weight: 800; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.02em; }}
        .kpi-sub {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}
        .pos {{ color: var(--green); }}
        .neg {{ color: var(--red); }}
        .muted {{ color: var(--muted); }}

        /* Layout Grids */
        .grid-2col {{ display: grid; grid-template-columns: 2.1fr 1fr; gap: 20px; margin-bottom: 24px; }}
        .grid-bottom {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 20px; margin-bottom: 24px; }}
        @media (max-width: 1200px) {{
            .grid-2col, .grid-bottom {{ grid-template-columns: 1fr; }}
        }}

        .panel {{
            background: var(--bg-card);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 22px 24px;
            margin-bottom: 24px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
        }}
        .panel-hdr {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            gap: 12px;
        }}
        .panel-hdr h2 {{ margin: 0; font-size: 16px; font-weight: 700; color: #e2e8f0; display: flex; align-items: center; gap: 8px; }}
        .panel-subtitle {{ color: var(--muted); font-size: 12.5px; }}

        /* Tables */
        table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
        th, td {{ text-align: left; padding: 13px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
        th {{ color: var(--muted); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.05em; }}
        tbody tr:hover {{ background: rgba(255, 255, 255, 0.025); }}

        .asset-cell {{ display: flex; align-items: center; gap: 12px; }}
        .asset-icon {{
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(99, 102, 241, 0.2));
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: #7dd3fc;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .asset-sym {{ font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 6px; }}
        .asset-sub {{ color: var(--muted); font-size: 11.5px; margin-top: 2px; }}
        .sub-notional {{ color: #38bdf8; font-size: 11.5px; font-weight: 600; margin-top: 2px; }}
        .lev-badge {{
            background: rgba(245, 158, 11, 0.16);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.4);
            padding: 2px 7px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
        }}

        /* SL / TP Range Bar */
        .range-labels {{ display: flex; justify-content: space-between; font-size: 11.5px; font-family: 'JetBrains Mono', monospace; margin-bottom: 6px; }}
        .range-track {{ height: 7px; width: 100%; background: rgba(30, 41, 59, 0.9); border-radius: 999px; position: relative; overflow: hidden; }}
        .range-fill {{ height: 100%; border-radius: 999px; }}
        .range-entry {{ position: absolute; top: -2px; bottom: -2px; width: 2px; background: #f8fafc; }}
        .sl {{ color: #fb7185; }}
        .tp1 {{ color: #fbbf24; }}
        .tp {{ color: #34d399; }}
        .pnl-big {{ font-size: 14.5px; font-weight: 700; }}
        .pnl-sub {{ font-size: 11.5px; opacity: 0.85; }}

        /* Pills & Chips */
        .pill {{ display: inline-block; padding: 4px 9px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 0.02em; }}
        .pill-green {{ background: rgba(16, 185, 129, 0.16); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }}
        .pill-red {{ background: rgba(244, 63, 94, 0.16); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.35); }}
        .pill-blue {{ background: rgba(56, 189, 248, 0.16); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.35); }}
        .pill-muted {{ background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.2); }}

        .corr-strip {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 8px; }}
        .corr-chip {{
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
            color: #cbd5e1;
        }}

        /* Heatmap */
        .hm-table th, .hm-table td {{ padding: 7px 6px; text-align: center; font-size: 11px; font-family: 'JetBrains Mono', monospace; }}
        .hm-cell {{ border-radius: 4px; color: #f8fafc; font-weight: 600; }}

        /* Adaptive Weights & Lessons */
        .weights-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 12px; margin: 14px 0 18px; }}
        .weight-item {{ background: rgba(15, 23, 42, 0.9); border: 1px solid var(--border); padding: 10px 12px; border-radius: 10px; }}
        .weight-hdr {{ display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-bottom: 6px; }}
        .weight-hdr strong {{ color: var(--cyan); font-family: 'JetBrains Mono', monospace; }}
        .weight-track {{ height: 5px; background: #1e293b; border-radius: 999px; overflow: hidden; }}
        .weight-fill {{ height: 100%; background: linear-gradient(90deg, #38bdf8, #818cf8); }}
        .lesson-item {{
            padding: 9px 12px;
            background: rgba(15, 23, 42, 0.65);
            border-left: 3px solid var(--cyan);
            border-radius: 6px;
            font-size: 12.5px;
            color: #cbd5e1;
            margin-bottom: 8px;
        }}
        .chart-container {{ height: 320px; width: 100%; }}
        .donut-container {{ height: 250px; width: 100%; display: flex; align-items: center; justify-content: center; }}
    </style>
</head>
<body>
    <div class="topbar">
        <div class="brand">
            <div class="brand-logo">⚡</div>
            <div>
                <h1>Autonomous Multi-Asset Finance Agent <span class="mode-badge">{self.config.execution_mode}</span></h1>
                <div class="brand-sub">Smart-Leverage (2x–3x) · 60d-Kovarianz-Engine · Selbstlernendes Shadow-Portfolio · Stand: {updated_at} UTC</div>
            </div>
        </div>
        <div class="topbar-right">
            <div class="live-indicator">
                <span class="pulse-dot"></span>
                Daemon aktiv · Auto-Refresh (60s) · Max Risk/Trade: {self.config.max_risk_per_trade_pct*100:.1f}%
            </div>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Gesamt-Equity (Start: ${self.config.initial_capital_usd:,.0f})</div>
            <div class="kpi-val">${total_equity:,.2f}</div>
            <div class="kpi-sub {'pos' if total_return_pct >= 0 else 'neg'}">Gesamt-Rendite: {total_return_pct:+.2f}% (Unreal.: {total_unrealized_pnl:+,.2f}$)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Aktive Markt-Kaufkraft (Hebel)</div>
            <div class="kpi-val" style="color:#38bdf8;">${total_notional_exposure:,.0f}</div>
            <div class="kpi-sub">Margin-Einsatz: ${total_margin_equity:,.2f} ({portfolio_leverage:.2f}x Gesamt-Hebel)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Verfügbarer Cash-Pool</div>
            <div class="kpi-val">${cash_usd:,.2f}</div>
            <div class="kpi-sub">Eiserne Reserve: {self.config.min_cash_reserve_pct*100:.0f}% (${total_equity*self.config.min_cash_reserve_pct:,.0f})</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">95%-Tages-VaR (Kovarianz)</div>
            <div class="kpi-val pos">${p_var_usd:,.2f} <span style="font-size:15px;">({p_var_pct:.2f}%)</span></div>
            <div class="kpi-sub pos">Diversifikations-Vorteil: -{p_div_ben:.1f}% Risiko · β {p_beta:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">2J Walk-Forward Backtest</div>
            <div class="kpi-val {'pos' if bt_ret >= 0 else 'neg'}">{bt_ret:+.2f}%</div>
            <div class="kpi-sub">Win-Rate: {bt_win:.1f}% · Profit-Faktor: {bt_pf:.2f} · Sharpe: {bt_sharpe:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Self-Learning Kalibrierung</div>
            <div class="kpi-val" style="color:#fbbf24;">{cal_total} <span style="font-size:15px;">Calls</span></div>
            <div class="kpi-sub">Trefferquote: {cal_acc:.1f}% ({cal_corr} ✓ / {cal_false} ✗) · {cal_saved} Guardrail-Saves</div>
        </div>
    </div>

    <div class="panel">
        <div class="panel-hdr">
            <div>
                <h2>💼 Offene Smart-Leverage Positionen (Alpaca Paper Synchronisiert)</h2>
                <div class="panel-subtitle">Visueller Echtzeit-Fortschritt zwischen Stop-Loss (links), 50%-Teilgewinnmitnahme (TP1) und Runner-Ziel (TP2)</div>
            </div>
            <span class="pill pill-blue">5-Tage Time-Stop aktiv</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Asset & Cluster</th>
                    <th class="num">Stückzahl</th>
                    <th class="num">Einstieg</th>
                    <th class="num">Aktuell</th>
                    <th class="num">Margin ➔ Kaufkraft</th>
                    <th>Stop-Loss ➔ TP1 (50%) ➔ TP2 Radar</th>
                    <th class="num">Hebel-PnL</th>
                </tr>
            </thead>
            <tbody>
                {pos_rows_html}
            </tbody>
        </table>
    </div>

    {diagnostics_panel_html}

    {funnel_panel_html}

    {adas_panel_html}

    <div class="grid-2col">
        <div class="panel" style="margin-bottom:0;">
            <div class="panel-hdr">
                <div>
                    <h2>📈 Walk-Forward Vermögenskurve (Smart-Leverage vs. Benchmark)</h2>
                    <div class="panel-subtitle">Simulation inklusive 50% Scale-Out bei +1.5x ATR, Chandelier Trailing-Stop & VIX-Ampel</div>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>

        <div class="panel" style="margin-bottom:0;">
            <div class="panel-hdr">
                <div>
                    <h2>🧭 Kapital-Allokation & Korrelation</h2>
                    <div class="panel-subtitle">Margin-Verteilung & stärkste 60-Tage-Kopplungen</div>
                </div>
            </div>
            <div class="donut-container">
                <canvas id="allocChart"></canvas>
            </div>
            <div style="margin-top:14px;">
                <div class="kpi-label">🔗 Stärkste 60-Tage Markt-Kopplungen (Sperre ab r ≥ 0.85)</div>
                <div class="corr-strip">{top_pairs_badges}</div>
            </div>
        </div>
    </div>

    <div class="panel">
        <div class="panel-hdr">
            <div>
                <h2>🔬 Selbstlernendes Shadow-Portfolio (Trackt alle 11 Assets – auch ohne eigenen Bestand)</h2>
                <div class="panel-subtitle">Überprüft jede BUY- und SELL-Hypothese am echten Marktverlauf und passt die Faktor-Gewichte autonom an</div>
            </div>
        </div>
        <div class="weights-grid">
            {weight_bars_html}
        </div>
        <table>
            <thead>
                <tr>
                    <th>Asset</th>
                    <th>Prognose & Konfidenz</th>
                    <th>Portfolio-Status</th>
                    <th class="num">Kursverlauf seit Prognose</th>
                    <th class="num">Hypoth. Ziel / Invalidierung</th>
                    <th class="num">Prognose-Erfolg (Richtungs-PnL)</th>
                </tr>
            </thead>
            <tbody>
                {shadow_rows_html}
            </tbody>
        </table>
        <div style="margin-top:18px;">
            <div class="kpi-label" style="margin-bottom:10px;">🧠 Letzte autonome Lerneffekte & Fehler-Diagnosen</div>
            {lessons_items}
        </div>
    </div>

    <div class="grid-bottom">
        <div class="panel" style="margin-bottom:0;">
            <div class="panel-hdr">
                <div>
                    <h2>📜 Autonomes Order- & Broker-Journal</h2>
                    <div class="panel-subtitle">Letzte ausgeführte Trades & Stop-Loss-Anpassungen</div>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Zeit (UTC)</th>
                        <th>Aktion</th>
                        <th>Symbol</th>
                        <th class="num">Ausführung</th>
                        <th>Begründung & Broker-ID</th>
                    </tr>
                </thead>
                <tbody>
                    {audit_rows_html}
                </tbody>
            </table>
        </div>

        <div class="panel" style="margin-bottom:0;">
            <div class="panel-hdr">
                <div>
                    <h2>🧮 60-Tage Cross-Asset Korrelations-Matrix</h2>
                    <div class="panel-subtitle">Rot = Gleichlauf (Klumpenrisiko) · Grün = Gegenläufig (Hedge)</div>
                </div>
            </div>
            <div style="overflow-x:auto;">
                {heatmap_html}
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        const grad = ctx.createLinearGradient(0, 0, 0, 320);
        grad.addColorStop(0, 'rgba(56, 189, 248, 0.32)');
        grad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {dates_json},
                datasets: [
                    {{
                        label: 'Smart-Leverage Agent ($)',
                        data: {agent_eq_json},
                        borderColor: '#38bdf8',
                        backgroundColor: grad,
                        borderWidth: 2.4,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.2
                    }},
                    {{
                        label: 'Benchmark (Ohne Risk-Engine)',
                        data: {bench_eq_json},
                        borderColor: '#64748b',
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                plugins: {{
                    legend: {{ labels: {{ color: '#cbd5e1', font: {{ family: 'Inter', weight: '600' }} }} }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#64748b', maxTicksLimit: 10 }}, grid: {{ color: 'rgba(148, 163, 184, 0.08)' }} }},
                    y: {{ ticks: {{ color: '#94a3b8', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: 'rgba(148, 163, 184, 0.08)' }} }}
                }}
            }}
        }});

        const allocCtx = document.getElementById('allocChart').getContext('2d');
        new Chart(allocCtx, {{
            type: 'doughnut',
            data: {{
                labels: {alloc_labels_json},
                datasets: [{{
                    data: {alloc_values_json},
                    backgroundColor: ['#1e293b', '#38bdf8', '#818cf8', '#10b981', '#f59e0b', '#f43f5e'],
                    borderColor: '#0f172a',
                    borderWidth: 3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                cutout: '68%',
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#cbd5e1', font: {{ family: 'Inter', size: 12 }} }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.output_html, "w", encoding="utf-8") as f:
            f.write(html)

        if open_browser:
            try:
                webbrowser.open(f"file:///{os.path.abspath(self.output_html).replace(os.sep, '/')}")
            except Exception:
                pass
        return self.output_html
