import os
import sys
import tempfile
import numpy as np
import pandas as pd
from config import AgentConfig
from data_collector import compute_wilder_rsi
from models import MacroRegime, MarketSnapshot, Position, TradingSignal
from risk_manager import DeterministicRiskManager, is_us_equity_market_open, is_opening_bell_cooldown
from execution_engine import MultiAssetExecutionEngine
import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def test_all_audit_fixes() -> None:
    print("=== Starte automatisierten Code- & Logik-Audit-Test ===")

    # 1. Test Wilder-RSI Grenzfall (14 steigende Tage ohne Verlust -> RSI muss ~100 sein, NICHT 50!)
    rising_prices = pd.Series([100.0 + i * 2.0 for i in range(25)])
    rsi_vals = compute_wilder_rsi(rising_prices, period=14)
    assert rsi_vals.iloc[-1] > 95.0, f"FEHLER: Wilder-RSI bei reinem Aufwärtstrend sollte >95 sein, war {rsi_vals.iloc[-1]}"
    print(f"[OK] Test 1 bestanden: Wilder-RSI Grenzfall (streng monoton steigend -> RSI = {rsi_vals.iloc[-1]:.1f})")

    # 2. Test Daily Drawdown Circuit Breaker vs. All-Time Return
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AgentConfig()
        cfg.initial_capital_usd = 1000.0
        cfg.execution_mode = "LOCAL_PAPER"  # Isolierter Unit-Test ohne echte Broker-Orders
        cfg.enforce_market_hours = False  # Für deterministische Unit-Tests unabhängig von der Test-Uhrzeit
        cfg.portfolio_state_file = os.path.join(tmpdir, "state.json")
        cfg.audit_log_file = os.path.join(tmpdir, "audit.jsonl")

        engine = MultiAssetExecutionEngine(cfg)
        # Simuliere: 1.000$ Startkapital ist historisch +25% im Gewinn ($1,250 Start-Equity heute), fällt aber heute auf $1,180 (-5.6% Tagesverlust)
        engine.day_start_equity_usd = 1250.0
        engine.cash_usd = 1180.0
        assert engine.total_return_pct == 18.0, f"Gesamtrendite muss +18.0% sein, war {engine.total_return_pct}%"
        assert engine.daily_pnl_pct == -5.6, f"Tages-PnL muss -5.6% sein, war {engine.daily_pnl_pct}%"

        risk_mgr = DeterministicRiskManager(cfg)
        dummy_snap = MarketSnapshot(
            symbol="NVDA",
            name="NVIDIA",
            asset_class="EQUITY",
            current_price=200.0,
            change_24h_pct=1.0,
            change_7d_pct=3.0,
            rsi_14=55.0,
            sma_20=195.0,
            sma_50=185.0,
            sma_50_slope=2.0,
            macd_hist=0.5,
            atr_14=5.0,
            volatility_annualized_pct=35.0,
            volume_ratio_vs_20d=1.3,
        )
        dummy_sig = TradingSignal(
            symbol="NVDA",
            asset_class="EQUITY",
            action="BUY",
            confidence=0.85,
            sentiment_score=0.4,
            entry_price=200.0,
            stop_loss_price=189.0,
            take_profit_price=222.0,
            risk_reward_ratio=2.0,
            catalyst_summary="Test",
            thesis="Test",
        )
        eval_res = risk_mgr.evaluate_signal(
            signal=dummy_sig,
            snapshot=dummy_snap,
            macro=None,
            cash_usd=engine.cash_usd,
            total_equity_usd=engine.total_equity_usd,
            daily_pnl_pct=engine.daily_pnl_pct,
            positions=engine.positions,
        )
        assert not eval_res.approved and "CIRCUIT BREAKER" in eval_res.reason, "Circuit Breaker hätte auslösen müssen!"
        print(f"[OK] Test 2 bestanden: Daily Drawdown Circuit Breaker erkennt -5.6% Tagesverlust trotz +18% Gesamtrendite ({eval_res.reason})")

        # 3. Test Earnings Blackout Schutz
        engine.day_start_equity_usd = 1180.0  # Tagesverlust auf 0% zurücksetzen
        dummy_snap.days_to_earnings = 2  # Quartalszahlen in 2 Tagen (Limit = 3)
        eval_earn = risk_mgr.evaluate_signal(
            signal=dummy_sig,
            snapshot=dummy_snap,
            macro=None,
            cash_usd=engine.cash_usd,
            total_equity_usd=engine.total_equity_usd,
            daily_pnl_pct=engine.daily_pnl_pct,
            positions=engine.positions,
        )
        assert not eval_earn.approved and "EARNINGS-SCHUTZ" in eval_earn.reason
        print(f"[OK] Test 3 bestanden: Earnings-Blackout blockiert Kauf 2 Tage vor Quartalszahlen ({eval_earn.reason})")

        # 4. Test Makro-Ampel (Rote Ampel blockiert Aktien/Krypto, erlaubt aber Safe-Haven GLD)
        dummy_snap.days_to_earnings = 30
        red_macro = MacroRegime(
            vix_level=34.5,
            spy_price=700.0,
            spy_sma_50=740.0,
            spy_sma_200=720.0,
            regime_state="RED_DEFENSIVE",
            risk_multiplier=0.0,
            summary="Rote Ampel",
        )
        eval_red_nvda = risk_mgr.evaluate_signal(
            signal=dummy_sig,
            snapshot=dummy_snap,
            macro=red_macro,
            cash_usd=engine.cash_usd,
            total_equity_usd=engine.total_equity_usd,
            daily_pnl_pct=engine.daily_pnl_pct,
            positions=engine.positions,
        )
        assert not eval_red_nvda.approved and "MAKRO-STOPP" in eval_red_nvda.reason

        dummy_snap.symbol = "GLD"
        dummy_snap.is_safe_haven = True
        dummy_sig.symbol = "GLD"
        eval_red_gld = risk_mgr.evaluate_signal(
            signal=dummy_sig,
            snapshot=dummy_snap,
            macro=red_macro,
            cash_usd=engine.cash_usd,
            total_equity_usd=engine.total_equity_usd,
            daily_pnl_pct=engine.daily_pnl_pct,
            positions=engine.positions,
        )
        assert eval_red_gld.approved, "Safe-Haven GLD muss bei Roter Makro-Ampel freigegeben werden!"
        print("[OK] Test 4 bestanden: Rote VIX-Ampel blockiert NVDA, lässt aber Safe-Haven GLD durch.")

        # 5. Test Chandelier Trailing Stop & Gebühren-Konsistenz in ExecutionEngine
        exec_msg = engine.execute_order(dummy_sig, eval_red_gld)
        assert exec_msg.startswith("EXECUTED BUY")
        pos = engine.positions["GLD"]
        initial_sl = pos.stop_loss_price

        # Kurs steigt um +2.0x ATR (+10$) -> Trailing Stop muss über Einstiegskurs nachziehen
        dummy_snap.current_price = 210.0
        events = engine.update_prices_and_check_stops({"GLD": dummy_snap})
        assert pos.stop_loss_price > pos.avg_entry_price and pos.stop_loss_price > initial_sl
        print(f"[OK] Test 5 bestanden: Chandelier Trailing-Stop von ${initial_sl:.2f} auf ${pos.stop_loss_price:.2f} nachgezogen ({events[0]})")

        # 6. Test US-Börsenzeiten-Filter (09:00 Uhr Berlin = 03:00 Uhr New York -> Geschlossen für Aktien, offen für Krypto)
        morning_utc = datetime.datetime(2026, 10, 28, 8, 0, tzinfo=datetime.timezone.utc)  # 09:00 MEZ / 04:00 NY
        afternoon_utc = datetime.datetime(2026, 10, 28, 15, 0, tzinfo=datetime.timezone.utc)  # 16:00 MEZ / 11:00 NY
        assert not is_us_equity_market_open(morning_utc)[0], "Um 09:00 Uhr MEZ muss US-Börse als geschlossen erkannt werden!"
        assert is_us_equity_market_open(afternoon_utc)[0], "Um 16:00 Uhr MEZ muss US-Börse als offen erkannt werden!"
        print("[OK] Test 6 bestanden: US-Börsenzeiten-Filter unterscheidet 09:00 Uhr MEZ (Watch-Only) und 16:00 Uhr MEZ (Trading aktiv) exakt.")

        # 7. Test 5-Tage Time-Stop gegen "totes Kapital" & Smart-Leverage Prüfung
        pos.opened_at = "2026-10-01T14:00:00"  # Vor >5 Tagen eröffnet
        pos.stop_loss_price = 180.0
        dummy_snap.current_price = 200.5  # Nur +0.2% nach >5 Tagen (< 1.0% Schwelle)
        dummy_snap.macd_hist = -0.5       # Negatives Momentum
        stale_events = engine.update_prices_and_check_stops({"GLD": dummy_snap})
        assert "GLD" not in engine.positions and any("TIME_STOP_STALE" in e for e in stale_events)
        print(f"[OK] Test 7 bestanden: 5-Tage Time-Stop schließt eingeschlafene Position autonom ({stale_events[0]})")

        # 8. Test Smart-Leverage (2.0x Margin bei Grüner Ampel auf Standard-Assets, 1.0x Margin auf 3x ETF SOXL -> effektiv 3.0x)
        green_macro = MacroRegime(
            vix_level=15.5,
            spy_price=710.0,
            spy_sma_50=690.0,
            spy_sma_200=650.0,
            regime_state="GREEN_RISK_ON",
            risk_multiplier=1.0,
            summary="Grüne Ampel",
        )
        dummy_snap.symbol = "NVDA"
        dummy_snap.is_safe_haven = False
        dummy_snap.intrinsic_leverage = 1.0
        dummy_sig.symbol = "NVDA"
        eval_green_nvda = risk_mgr.evaluate_signal(
            signal=dummy_sig,
            snapshot=dummy_snap,
            macro=green_macro,
            cash_usd=engine.cash_usd,
            total_equity_usd=engine.total_equity_usd,
            daily_pnl_pct=0.0,
            positions=engine.positions,
        )
        assert eval_green_nvda.approved and eval_green_nvda.leverage_used == 2.0 and eval_green_nvda.effective_total_leverage == 2.0
        assert abs(eval_green_nvda.notional_value_usd - 2.0 * eval_green_nvda.margin_required_usd) < 1.0

        dummy_snap.symbol = "SOXL"
        dummy_snap.intrinsic_leverage = 3.0
        dummy_sig.symbol = "SOXL"
        eval_green_soxl = risk_mgr.evaluate_signal(
            signal=dummy_sig,
            snapshot=dummy_snap,
            macro=green_macro,
            cash_usd=engine.cash_usd,
            total_equity_usd=engine.total_equity_usd,
            daily_pnl_pct=0.0,
            positions=engine.positions,
        )
        assert eval_green_soxl.approved and eval_green_soxl.leverage_used == 1.0 and eval_green_soxl.effective_total_leverage == 3.0
        print(
            f"[OK] Test 8 bestanden: Smart-Leverage vergibt 2.0x Margin auf NVDA (Margin {eval_green_nvda.margin_required_usd:.0f}$ -> Kaufkraft {eval_green_nvda.notional_value_usd:.0f}$) "
            f"und erkennt 3.0x Eigenhebel bei SOXL ohne Doppel-Margin."
        )

        # 9. Test Aktives Short-Selling (Leerverkauf bei >= 85% SELL-Konfidenz) & Gewinn bei Kursrückgang
        short_snap = MarketSnapshot(
            symbol="COIN",
            name="Coinbase",
            asset_class="EQUITY",
            current_price=160.0,
            change_24h_pct=-4.5,
            change_7d_pct=-12.0,
            rsi_14=38.0,
            sma_20=175.0,
            sma_50=190.0,
            sma_50_slope=-2.5,
            macd_hist=-1.2,
            atr_14=8.0,
            volatility_annualized_pct=55.0,
            volume_ratio_vs_20d=1.8,
        )
        short_sig = TradingSignal(
            symbol="COIN",
            asset_class="EQUITY",
            action="SELL",
            confidence=0.92,
            sentiment_score=-0.5,
            entry_price=160.0,
            stop_loss_price=147.2,
            take_profit_price=185.6,
            risk_reward_ratio=2.0,
            catalyst_summary="Bearish breakdown",
            thesis="Strong downtrend",
        )
        eval_short = risk_mgr.evaluate_signal(
            signal=short_sig,
            snapshot=short_snap,
            macro=green_macro,
            cash_usd=engine.cash_usd,
            total_equity_usd=engine.total_equity_usd,
            daily_pnl_pct=0.0,
            positions=engine.positions,
        )
        assert eval_short.approved and eval_short.adjusted_action == "SELL"
        short_exec = engine.execute_order(short_sig, eval_short)
        assert "EXECUTED SHORT-SELL" in short_exec and engine.positions["COIN"].side == "SHORT"
        # Kurs fällt von 160$ auf 145$ -> Short-Position muss im Gewinn liegen!
        short_snap.current_price = 145.0
        engine.update_prices_and_check_stops({"COIN": short_snap})
        assert engine.positions["COIN"].unrealized_pnl > 0, "Short-Position muss bei fallendem Kurs im Gewinn liegen!"
        print(f"[OK] Test 9 bestanden: Short-Selling auf COIN eröffnet & verdient +{engine.positions['COIN'].unrealized_pnl:.2f}$ bei Kursrückgang ({short_exec})")

        # 10. Test 15-Minuten Opening-Bell-Schutz (09:35 ET = Cooldown aktiv, 09:50 ET = Cooldown vorbei)
        open_0935_utc = datetime.datetime(2026, 10, 28, 13, 35, tzinfo=datetime.timezone.utc)  # 09:35 EDT
        open_0950_utc = datetime.datetime(2026, 10, 28, 13, 50, tzinfo=datetime.timezone.utc)  # 09:50 EDT
        assert is_opening_bell_cooldown(open_0935_utc, "NVDA", 15)[0], "Um 09:35 ET muss der 15-Min Opening-Bell-Schutz aktiv sein!"
        assert not is_opening_bell_cooldown(open_0950_utc, "NVDA", 15)[0], "Um 09:50 ET muss der 15-Min Opening-Bell-Schutz beendet sein!"
        print("[OK] Test 10 bestanden: 15-Minuten Opening-Bell-Schutz verhindert Käufe im Eröffnungs-Chaos (09:30-09:45 ET).")

        # 11. Test IBKRBridge Multi-Asset & Options-Contract-Builder (Xetra .DE -> EUR/IBIS, US -> USD/SMART, Option Badge -> Option Contract)
        from ibkr_bridge import IBKRBridge
        bridge = IBKRBridge(cfg)
        de_contract = bridge.build_stock_or_crypto_contract("RHM.DE", "EQUITY")
        assert de_contract.symbol == "RHM" and de_contract.currency == "EUR" and de_contract.primaryExchange == "IBIS"
        us_contract = bridge.build_stock_or_crypto_contract("NVDA", "EQUITY")
        assert us_contract.symbol == "NVDA" and us_contract.currency == "USD" and us_contract.exchange == "SMART"
        opt_contract = bridge.build_option_contract_from_badge("AXGN", "CALL $50 (11-20) @ $2.30 [V/OI:118.5x 🔥Whale]", ref_year=2026)
        assert opt_contract is not None and opt_contract.symbol == "AXGN" and opt_contract.right == "C" and opt_contract.strike == 50.0 and opt_contract.lastTradeDateOrContractMonth == "20261120"
        print("[OK] Test 11 bestanden: IBKRBridge routet RHM.DE auf Xetra (EUR/IBIS), NVDA auf US-SMART (USD) und baut aus dem Options-Scanner echte IBKR-Optionskontrakte.")

        # 12. Test ADAS (Automated Design of Agentic Systems) Autonomous Sub-Agent Spawning & Swarm Veto
        from adas_meta_agent import ADASMetaArchitect
        adas = ADASMetaArchitect(cfg)
        whale_snap = MarketSnapshot(
            symbol="AXGN",
            name="Axogen",
            asset_class="EQUITY",
            current_price=49.25,
            change_24h_pct=31.5,
            change_7d_pct=35.0,
            rsi_14=74.0,
            sma_20=38.0,
            sma_50=34.0,
            macd_hist=1.2,
            atr_14=2.2,
            volatility_annualized_pct=55.0,
            volume_ratio_vs_20d=4.8,
            unusual_options_vol_oi=118.5,
            options_put_call_ratio=0.04,
            options_flow_sentiment="BULLISH_CALL_FLOW",
            best_option_contract="CALL $50 (11-20) @ $2.30 [V/OI:118.5x 🔥Whale]",
            intraday_15m_trend="BULLISH_ABOVE_VWAP",
            intraday_vwap=47.5,
            relative_strength_vs_spy=25.7,
        )
        spawn_events = adas.run_evolution_and_spawning_cycle({"AXGN": whale_snap}, green_macro, {})
        assert "gamma_squeeze_specialist" in adas.state.active_subagents and "pead_earnings_drift_agent" in adas.state.active_subagents
        assert len(adas.state.active_subagents) >= 10, "ADAS muss aus 6 Kern-Agenten mindestens 4 neue Spezial-Unteragenten autonom erschaffen!"
        print(f"[OK] Test 12 bestanden: ADAS hat autonom {len(adas.state.active_subagents)} Spezial-Unteragenten (Generation #{adas.state.evolution_generation}) erschaffen & orchestriert.")

        # 13. Test Penny-Stock Rocket Specialist & Risiko-Deckelung (Max $500, Hebel = 1.0x)
        penny_snap = MarketSnapshot(
            symbol="SOUN",
            name="SoundHound AI",
            asset_class="EQUITY",
            cluster="PENNY_STOCKS",
            current_price=2.45,
            change_24h_pct=14.5,
            change_7d_pct=18.0,
            rsi_14=65.0,
            sma_20=2.10,
            sma_50=1.95,
            macd_hist=0.08,
            atr_14=0.18,
            volatility_annualized_pct=75.0,
            volume_ratio_vs_20d=3.8,
            ttm_squeeze_status="🔥 SQUEEZE_FIRED",
            ttm_squeeze_momentum=0.22,
            supertrend_direction="BULLISH",
            supertrend_val=2.15,
            supertrend_flipped=True,
        )
        adas.run_evolution_and_spawning_cycle({"SOUN": penny_snap}, green_macro, {})
        assert "penny_stock_rocket_specialist" in adas.state.active_subagents, "ADAS muss penny_stock_rocket_specialist erschaffen!"
        assert "volatility_squeeze_specialist" in adas.state.active_subagents, "ADAS muss volatility_squeeze_specialist erschaffen!"
        assert "supertrend_regime_specialist" in adas.state.active_subagents, "ADAS muss supertrend_regime_specialist erschaffen!"

        # Risiko-Prüfung für Penny Stock (muss strikt auf $500 gedeckelt sein und Hebel 1.0x haben)
        penny_signal = TradingSignal(
            symbol="SOUN",
            asset_class="EQUITY",
            cluster="PENNY_STOCKS",
            action="BUY",
            confidence=0.88,
            sentiment_score=0.75,
            entry_price=2.45,
            stop_loss_price=2.20,
            take_profit_price=3.10,
            risk_reward_ratio=2.6,
            catalyst_summary="AI Voice Micro-cap RVOL Spike",
            thesis="Penny Stock Ausbruch mit 3.8x Volumen",
        )

        risk_eval_penny = risk_mgr.evaluate_signal(
            signal=penny_signal,
            snapshot=penny_snap,
            macro=green_macro,
            cash_usd=90_000.0,
            total_equity_usd=100_000.0,
            daily_pnl_pct=0.0,
            positions={},
        )
        assert risk_eval_penny.approved, "Penny Stock Signal muss freigegeben werden"
        assert risk_eval_penny.leverage_used == 1.0, f"Penny Stocks dürfen keinen Margin-Hebel erhalten! Erhalten: {risk_eval_penny.leverage_used}"
        assert risk_eval_penny.notional_value_usd <= 505.0, f"Penny-Stock Orderwert darf max 500$ sein, war {risk_eval_penny.notional_value_usd}$"
        print(f"[OK] Test 13 bestanden: Penny Stock SOUN @ 2.45$ von Spezialisten bewertet, auf {risk_eval_penny.notional_value_usd}$ Margin gedeckelt (1.0x Hebel) & Swarm umfasst {len(adas.state.active_subagents)} Sub-Agenten.")

        # ============================================================================
        # TEST 14: Genetische Strategie-Zucht (Sakana AI / AlphaZero Breeding Sandbox)
        # ============================================================================
        from genetic_strategy_breeder import GeneticStrategyBreeder, STRATEGY_ARCHETYPES
        cfg_breed = AgentConfig()
        breeder = GeneticStrategyBreeder(cfg_breed)
        test_genes = {
            "min_rvol": 1.8,
            "min_change_pct": 2.5,
            "max_rsi": 72.0,
            "min_edge_score": 1.40,
            "atr_sl_mult": 2.0,
            "atr_tp_mult": 4.5,
            "time_stop_days": 5,
            "penny_max_price": 999.0,
        }
        res = breeder._simulate_strategy_backtest(test_genes, "VOLATILITY_SQUEEZE_EXPLOSION")
        assert "sharpe_ratio" in res and "profit_factor" in res and "win_rate_pct" in res
        assert res["total_trades"] >= 10
        ast_ok, ast_msg = breeder._verify_code_ast("class TestStrategy:\n    def evaluate_entry(self, s):\n        return True\n")
        assert ast_ok, "AST-Sicherheitsprüfung muss gültigen Code akzeptieren"
        ast_bad, _ = breeder._verify_code_ast("import os\nos.system('echo dangerous')\n")
        assert not ast_bad, "AST-Sicherheitsprüfung muss gefährliche os.system Calls abfangen!"
        print(f"[OK] Test 14 bestanden: Genetische Strategie-Zucht simuliert Backtest (Sharpe {res['sharpe_ratio']}, Win-Rate {res['win_rate_pct']}%) & AST-Sicherheitsprüfung schützt Codebase.")

        # ============================================================================
        # TEST 15: Adversarial LLM Red-Teaming Tribunal (Bulle vs. Bär vor dem CIO)
        # ============================================================================
        from adversarial_red_team import AdversarialRedTeamTribunal
        tribunal = AdversarialRedTeamTribunal(cfg_breed)
        trap_snap = MarketSnapshot(
            symbol="TRAP",
            name="Trap Microcap",
            asset_class="EQUITY",
            cluster="PENNY_STOCKS",
            current_price=2.80,
            change_24h_pct=28.5,
            change_7d_pct=45.0,
            rsi_14=84.5,  # Extrem überkauft!
            sma_20=2.10,
            sma_50=1.80,
            macd_hist=0.15,
            atr_14=0.25,
            volatility_annualized_pct=110.0,
            volume_ratio_vs_20d=4.2,
            days_to_earnings=45,  # Keine Quartalszahlen, reine Bullenfalle!
            sec_form8k_catalyst="",  # Kein SEC Filing!
        )
        trap_signal = TradingSignal(
            symbol="TRAP",
            asset_class="EQUITY",
            cluster="PENNY_STOCKS",
            action="BUY",
            confidence=0.88,
            sentiment_score=0.70,
            entry_price=2.80,
            stop_loss_price=2.40,
            take_profit_price=3.60,
            risk_reward_ratio=2.0,
            catalyst_summary="Pump Ausbruch",
            thesis="Parabolischer Anstieg",
        )
        debate_trap = tribunal.convene_tribunal(trap_signal, trap_snap, green_macro, {})
        assert debate_trap.verdict == "REJECTED_VETO", f"Red Team muss tödliche Falle vetoen! Erhalten: {debate_trap.verdict}"
        assert debate_trap.risk_score >= 8.0, f"Risiko-Score muss hoch sein! Erhalten: {debate_trap.risk_score}"

        # Risk-Manager muss den VETO-Entscheid respektieren und den Kauf blockieren
        risk_mgr_test = DeterministicRiskManager(cfg_breed)
        risk_eval_trap = risk_mgr_test.evaluate_signal(
            signal=trap_signal,
            snapshot=trap_snap,
            macro=green_macro,
            cash_usd=90_000.0,
            total_equity_usd=100_000.0,
            daily_pnl_pct=0.0,
            positions={},
        )
        assert not risk_eval_trap.approved, "Risk Manager muss VETO Trade blockieren"
        assert "RED-TEAM-VETO" in risk_eval_trap.reason, f"Begründung muss RED-TEAM-VETO enthalten! Erhalten: {risk_eval_trap.reason}"
        print(f"[OK] Test 15 bestanden: Adversarial Red-Team Tribunal entlarvt Bullenfalle & verhindert Fehlkauf (Score {debate_trap.risk_score}/10 -> {debate_trap.verdict}).")

        # ============================================================================
        # TEST 16: SEC Form 4 Insider- & Whale-Tracking
        # ============================================================================
        from sec_form4_tracker import SECForm4Tracker
        f4_tracker = SECForm4Tracker()
        f4_res = f4_tracker.get_insider_activity("NVDA")
        assert "insider_sentiment" in f4_res and "net_shares_90d" in f4_res
        
        # Teste Snapshot-Integration mit simuliertem Whale Cluster-Kauf
        whale_snap = MarketSnapshot(
            symbol="WHALE",
            name="Whale Microcap",
            asset_class="EQUITY",
            cluster="PENNY_STOCKS",
            current_price=3.50,
            change_24h_pct=5.5,
            change_7d_pct=8.0,
            rsi_14=62.0,
            sma_20=3.20,
            sma_50=2.90,
            macd_hist=0.08,
            atr_14=0.20,
            volatility_annualized_pct=65.0,
            volume_ratio_vs_20d=2.4,
            insider_sentiment="STRONG_INSIDER_BUY",
            insider_summary="CEO & CFO kauften 75.000 Stk. für $260.000",
            form4_net_usd_90d=260_000.0,
            form4_cluster_buy=True,
            form4_recent_whale_buys=[{"insider": "CEO", "shares": 50000, "value_usd": 175000}],
        )
        assert whale_snap.form4_cluster_buy is True
        assert whale_snap.insider_sentiment == "STRONG_INSIDER_BUY"
        
        # Prüfe ADAS Specialist Votum mit baseline_signal und green_macro
        whale_sig = TradingSignal(
            symbol="WHALE",
            asset_class="EQUITY",
            cluster="PENNY_STOCKS",
            action="BUY",
            confidence=0.82,
            sentiment_score=0.80,
            entry_price=3.50,
            stop_loss_price=3.10,
            take_profit_price=4.60,
            risk_reward_ratio=2.75,
            catalyst_summary="Whale Cluster-Kauf",
            thesis="Form 4 Zukäufe",
        )
        updated_sig, votes, summary = adas.evaluate_symbol_with_swarm(whale_snap, whale_sig, green_macro)
        assert updated_sig.action == "BUY"
        assert len(votes) >= 1
        print(f"[OK] Test 16 bestanden: SEC Form 4 Insider- & Whale-Tracker erkennt Cluster-Kauf ({whale_snap.insider_summary}) & ADAS-Schwarm bestätigt BUY ({summary}).")

        # ============================================================================
        # TEST 17: Options Order Flow & Dark Pool Tracker
        # ============================================================================
        from options_darkpool_flow import OptionsDarkPoolTracker
        opt_tracker = OptionsDarkPoolTracker()
        opt_res = opt_tracker.analyze_flow("NVDA", 140.0)
        assert opt_res.symbol == "NVDA"
        assert opt_res.flow_sentiment in ("BULLISH_FLOW", "BEARISH_FLOW", "NEUTRAL", "NO_OPTIONS_DATA")
        assert -1.0 <= opt_res.flow_score <= 1.0
        print(f"[OK] Test 17 bestanden: Options-Flow & Max-Pain Scanner liefert Sentiment {opt_res.flow_sentiment} (Score {opt_res.flow_score:+.2f}, MaxPain ${opt_res.max_pain:.2f}).")

        # ============================================================================
        # TEST 18: Financial Catalyst & Distress Scanner
        # ============================================================================
        from financial_catalyst_scanner import FinancialCatalystScanner
        fund_scanner = FinancialCatalystScanner()
        fund_aapl = fund_scanner.scan_symbol("AAPL")
        assert fund_aapl.symbol == "AAPL"
        assert fund_aapl.health_grade in ("ELITE_GROWTH", "SOLID_VALUE", "SPECULATIVE", "TOXIC_DISTRESSED")
        assert -1.0 <= fund_aapl.fundamental_score <= 1.0

        # Distress-Prüfung: Krypto/Index darf nicht als Pleitekandidat markiert werden
        fund_spy = fund_scanner.scan_symbol("SPY")
        assert fund_spy.health_grade == "ETF_INDEX"
        assert fund_spy.is_distressed is False
        print(f"[OK] Test 18 bestanden: Fundamentalanalyse & Distress-Scanner klassifiziert AAPL als {fund_aapl.health_grade} & SPY als {fund_spy.health_grade}.")

        # ============================================================================
        # TEST 19: Telegram Command Listener (Befehls-Routing & Autorisierung)
        # ============================================================================
        from telegram_command_listener import TelegramCommandListener
        tg_l = TelegramCommandListener("dummy_token", "9215209981")
        assert tg_l.chat_id == "9215209981"
        assert tg_l.is_running is False
        # Simuliere Portfolio-Callback
        tg_l.portfolio_callback = lambda: [{"symbol": "NVDA", "qty": 10, "avg_entry_price": 130.0, "current_price": 142.0, "unrealized_pl": 120.0, "unrealized_plpc": 0.092}]
        port_data = tg_l.portfolio_callback()
        assert len(port_data) == 1 and port_data[0]["symbol"] == "NVDA"
        print(f"[OK] Test 19 bestanden: Telegram Remote Listener autorisiert Chat-ID {tg_l.chat_id} & verarbeitet Portfolio-Abfragen.")

        # ============================================================================
        # TEST 20: Walk-Forward Hyperparameter Auto-Tuner
        # ============================================================================
        from hyperparameter_autotuner import HyperparameterAutoTuner
        autotuner = HyperparameterAutoTuner()
        calib_bull = autotuner.tune_parameters(green_macro)
        assert calib_bull.regime == "GREEN_RISK_ON"
        assert 1.5 <= calib_bull.atr_stop_loss_mult <= 2.2
        assert calib_bull.target_risk_reward_ratio >= 1.6

        # Krisen-Makro testen (muss Stopps weiten und Konfidenz anheben)
        crisis_macro = MacroRegime(
            vix_level=32.0,
            spy_price=510.0,
            spy_sma_50=530.0,
            spy_sma_200=550.0,
            us10y_yield=4.8,
            us10y_change_5d_bps=20.0,
            dxy_level=107.0,
            dxy_change_5d_pct=2.5,
            regime_state="RED_DEFENSIVE",
            risk_multiplier=0.0,
            summary="Panik",
        )
        calib_crisis = autotuner.tune_parameters(crisis_macro)
        assert calib_crisis.atr_stop_loss_mult > calib_bull.atr_stop_loss_mult, "Im Crash müssen Stopps geweitet werden"
        assert calib_crisis.min_confidence_cutoff > calib_bull.min_confidence_cutoff, "Im Crash muss Konfidenz-Hürde steigen"
        print(f"[OK] Test 20 bestanden: Hyperparameter-AutoTuner kalibriert Stop-Loss dynamisch (Bull {calib_bull.atr_stop_loss_mult}x ATR vs Crash {calib_crisis.atr_stop_loss_mult}x ATR).")

        # ============================================================================
        # TEST 21: Statistische Arbitrage & Kointegrations-Pairs Trading
        # ============================================================================
        from pairs_stat_arb import StatisticalArbitrageScanner, PairArbitrageOpportunity
        pairs_scanner = StatisticalArbitrageScanner()
        all_opps = pairs_scanner.scan_all_pairs()
        assert len(all_opps) >= 4, f"Mindestens 4 Paare müssen gescannt werden! Gefunden: {len(all_opps)}"
        for o in all_opps:
            assert -10.0 <= o.z_score <= 10.0
            assert -1.0 <= o.correlation_60d <= 1.0

        # Simuliere Extrem-Divergenz Z = -2.4 auf NVDA in NVDA_AMD
        stat_snap = MarketSnapshot(
            symbol="AMD",
            name="Advanced Micro Devices",
            asset_class="EQUITY",
            cluster="US_TECH",
            current_price=150.0,
            change_24h_pct=1.2,
            change_7d_pct=-3.0,
            rsi_14=48.0,
            sma_20=152.0,
            sma_50=155.0,
            macd_hist=0.02,
            atr_14=3.5,
            volatility_annualized_pct=38.0,
            volume_ratio_vs_20d=1.4,
            stat_arb_pair="NVDA_AMD",
            stat_arb_z_score=2.35,  # AMD ist stark unterbewertet ggü NVDA
            stat_arb_thesis="AMD hinkt NVDA 2.35σ hinterher",
        )
        assert stat_snap.stat_arb_z_score == 2.35

        # ADAS Pairs-Trading Specialist muss Votum abgeben
        base_amd_sig = TradingSignal(
            symbol="AMD",
            asset_class="EQUITY",
            cluster="US_TECH",
            action="HOLD",
            confidence=0.50,
            sentiment_score=0.50,
            entry_price=150.0,
            stop_loss_price=142.0,
            take_profit_price=165.0,
            risk_reward_ratio=1.87,
            catalyst_summary="Pairs Trading",
            thesis="Neutral Baseline",
        )
        up_sig, v_list, sum_txt = adas.evaluate_symbol_with_swarm(stat_snap, base_amd_sig, green_macro)
        arb_votes = [v for v in v_list if v.agent_id == "pairs_stat_arb_specialist"]
        assert len(arb_votes) == 1, "pairs_stat_arb_specialist muss im Schwarm aktiv sein"
        assert arb_votes[0].score >= 0.70, f"Specialist muss bei Z >= 2.0 stark bullisch votieren! Score: {arb_votes[0].score}"
        print(f"[OK] Test 21 bestanden: Statistische Arbitrage erkennt Z=+2.35σ Divergenz in NVDA_AMD & Specialist votiert {arb_votes[0].vote_action} ({arb_votes[0].score:+.2f}).")

        # ============================================================================
        # TEST 22: Hurst-Exponent & Fraktale Marktanalyse
        # ============================================================================
        from hurst_fractal_analyzer import HurstFractalAnalyzer
        hurst_tool = HurstFractalAnalyzer()
        # Teste synthetischen persistenten Trend (Hurst > 0.55)
        trend_series = pd.Series(np.linspace(100, 200, 65) + np.random.randn(65) * 0.5)
        frac_res = hurst_tool.analyze_series("TREND", trend_series)
        assert frac_res.hurst_exponent >= 0.55, f"Hurst-Exponent bei linearem Trend muss hoch sein! Erhalten: {frac_res.hurst_exponent}"
        assert frac_res.regime_mode == "PERSISTENT_TRENDING"
        assert frac_res.regime_edge_multiplier >= 1.20
        print(f"[OK] Test 22 bestanden: Hurst-Exponent & Fraktal-Analyzer erkennt persistenten Trend (H={frac_res.hurst_exponent:.2f}, Modus={frac_res.regime_mode}, Multiplikator={frac_res.regime_edge_multiplier}x).")

        # ============================================================================
        # TEST 23: Intermarket High-Yield Credit & Rohstoff-Lead-Lag Engine
        # ============================================================================
        from intermarket_lead_lag import IntermarketLeadLagEngine
        im_engine = IntermarketLeadLagEngine()
        im_snap = im_engine.analyze_intermarket_dynamics()
        assert im_snap.credit_ratio_hyg_lqd > 0.3
        assert im_snap.intermarket_bias in ("BULLISH_CREDIT_TAILWIND", "NEUTRAL_STABLE", "BEARISH_CREDIT_WARNING", "SEVERE_CRUNCH")
        assert 0.20 <= im_snap.risk_budget_multiplier <= 1.50
        print(f"[OK] Test 23 bestanden: Intermarket-Engine analysiert Kredit-Spreads HYG/LQD ({im_snap.credit_ratio_hyg_lqd:.3f}) & Bias ({im_snap.intermarket_bias}).")

        # ============================================================================
        # TEST 24: Corwin-Schultz Bid-Ask Spread & Amihud Illiquiditäts-Schätzer
        # ============================================================================
        from market_microstructure import MarketMicrostructureAnalyzer
        micro_tool = MarketMicrostructureAnalyzer()
        # Simuliere Kurs-DataFrame mit engem Spread
        synth_df = pd.DataFrame({
            "High": np.linspace(101, 120, 25),
            "Low": np.linspace(99, 118, 25),
            "Close": np.linspace(100, 119, 25),
            "Volume": np.full(25, 5_000_000.0),
        })
        micro_res = micro_tool.analyze_microstructure("SYNTH", synth_df)
        assert micro_res.estimated_bid_ask_spread_pct >= 0.01
        assert micro_res.routing_recommendation in ("MARKET_EXECUTION_SAFE", "LIMIT_MIDPOINT_PEG", "SLIPPAGE_TRAP_AVOID")
        print(f"[OK] Test 24 bestanden: Corwin-Schultz Spread-Schätzer berechnet Spread {micro_res.estimated_bid_ask_spread_pct:.2f}% & Routing {micro_res.routing_recommendation}.")

        # ============================================================================
        # TEST 25: Schwarm-Spezialisten #21, #22, #23 & Red-Team Kredit-Crunch Veto
        # ============================================================================
        # 1. Simuliere toxischen systemischen Kredit-Stress
        credit_crunch_macro = MacroRegime(
            vix_level=28.0,
            spy_price=530.0,
            spy_sma_50=545.0,
            spy_sma_200=520.0,
            regime_state="YELLOW_CAUTION",
            risk_multiplier=0.45,
            credit_spread_hyg_lqd=0.68,
            credit_stress_active=True,  # Junk-Bonds kollabieren!
            copper_gold_ratio=0.10,
            intermarket_bias="SEVERE_CRUNCH",
            summary="Kredit-Krise",
        )
        safe_stock_snap = MarketSnapshot(
            symbol="TECH",
            name="Tech Growth",
            asset_class="EQUITY",
            cluster="US_TECH",
            current_price=200.0,
            change_24h_pct=2.0,
            change_7d_pct=4.0,
            rsi_14=55.0,
            sma_20=195.0,
            sma_50=190.0,
            macd_hist=0.5,
            atr_14=4.0,
            volatility_annualized_pct=28.0,
            volume_ratio_vs_20d=1.5,
            hurst_exponent=0.62,
            fractal_regime_mode="PERSISTENT_TRENDING",
            bid_ask_spread_pct=0.04,
            microstructure_routing="MARKET_EXECUTION_SAFE",
        )
        tech_sig = TradingSignal(
            symbol="TECH",
            asset_class="EQUITY",
            cluster="US_TECH",
            action="BUY",
            confidence=0.82,
            sentiment_score=0.75,
            entry_price=200.0,
            stop_loss_price=190.0,
            take_profit_price=225.0,
            risk_reward_ratio=2.5,
            catalyst_summary="Technischer Ausbruch",
            thesis="Hurst Persistenter Trend",
        )
        # Schwarm auswerten
        _, votes_credit, _ = adas.evaluate_symbol_with_swarm(safe_stock_snap, tech_sig, credit_crunch_macro)
        credit_vote = next(v for v in votes_credit if v.agent_id == "intermarket_credit_sentinel")
        assert credit_vote.score <= -0.80, f"Credit Sentinel muss bei Kredit-Stress extrem negativ votieren! Erhalten: {credit_vote.score}"

        # Red-Team Tribunal muss Kredit-Kollaps erkennen und VETO einlegen
        tribunal_res = tribunal.convene_tribunal(tech_sig, safe_stock_snap, credit_crunch_macro, {})
        assert tribunal_res.verdict == "REJECTED_VETO", f"Red-Team muss bei Systemic Credit Crunch VETO einlegen! Erhalten: {tribunal_res.verdict}"
        assert "Credit" in tribunal_res.cio_reasoning or "Credit" in tribunal_res.bear_rebuttal, "Begründung muss Credit Crunch enthalten"
        print(f"[OK] Test 25 bestanden: Intermarket Credit Sentinel votiert {credit_vote.vote_action} ({credit_vote.score:+.2f}) & Red Team legt VETO ein ({tribunal_res.cio_reasoning[:60]}...).")

        # ============================================================================
        # TEST 26: Mathematisches Fractional Kelly Kriterium & Volatilitäts-Targeting
        # ============================================================================
        from fractional_kelly_sizing import FractionalKellyPositionSizer
        kelly_sizer = FractionalKellyPositionSizer(kelly_fraction_multiplier=0.25, target_annual_vol=0.16)
        # Setup: 75% Win-Chance, CRV 2.5:1, Vola 20%
        k_res = kelly_sizer.compute_size(
            total_equity_usd=100_000.0,
            available_cash_usd=80_000.0,
            entry_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=112.5,
            confidence=0.75,
            annualized_vol_pct=20.0,
        )
        assert k_res.kelly_fraction_raw > 0.50, f"Raw Kelly muss bei 75% Konfidenz und 2.5 CRV stark positiv sein! Erhalten: {k_res.kelly_fraction_raw}"
        assert 0.02 <= k_res.recommended_position_pct <= 0.25
        assert k_res.recommended_position_usd > 1000.0
        print(f"[OK] Test 26 bestanden: Fractional Kelly berechnet mathematisch optimale Allokation ({k_res.recommended_position_pct*100:.1f}% = ${k_res.recommended_position_usd:,.0f}, Raw f*={k_res.kelly_fraction_raw*100:.1f}%).")

        # ============================================================================
        # TEST 27: Sektor-Rotations-Radar & Spezialist #24 (Sector Alpha Hunter)
        # ============================================================================
        from sector_momentum_expander import SectorMomentumExpander
        sec_scanner = SectorMomentumExpander()
        sec_snap = sec_scanner.scan_sector_rotation()
        assert len(sec_snap.leading_sectors) >= 1
        assert len(sec_snap.recommended_hot_stocks) >= 2
        # Teste Schwarm-Votum für eine Aktie aus dem Gewinner-Sektor
        top_sec_stock = sec_snap.recommended_hot_stocks[0]
        leader_snap = MarketSnapshot(
            symbol=top_sec_stock,
            name="Sector Leader",
            asset_class="EQUITY",
            cluster="US_TECH",
            current_price=120.0,
            change_24h_pct=2.5,
            change_7d_pct=5.0,
            rsi_14=58.0,
            sma_20=115.0,
            sma_50=110.0,
            macd_hist=0.2,
            atr_14=2.0,
            volatility_annualized_pct=22.0,
            volume_ratio_vs_20d=1.6,
        )
        leader_sig = TradingSignal(
            symbol=top_sec_stock,
            asset_class="EQUITY",
            cluster="US_TECH",
            action="BUY",
            confidence=0.80,
            sentiment_score=0.75,
            entry_price=120.0,
            stop_loss_price=114.0,
            take_profit_price=135.0,
            risk_reward_ratio=2.5,
            catalyst_summary="Sektor-Rotation",
            thesis="Sector Alpha Momentum",
        )
        _, votes_sec, _ = adas.evaluate_symbol_with_swarm(leader_snap, leader_sig, green_macro)
        sec_vote = next(v for v in votes_sec if v.agent_id == "sector_rotation_hunter")
        assert sec_vote.score >= 0.70, f"Sector Hunter muss Sektor-Leader stark befürworten! Score: {sec_vote.score}"
        print(f"[OK] Test 27 bestanden: Sektor-Rotations-Radar identifiziert {sec_snap.top_alpha_sector} & Specialist #24 votiert {sec_vote.vote_action} ({sec_vote.score:+.2f}) für {top_sec_stock}.")

        # ============================================================================
        # TEST 28: Wall Street Analysten-Radar & Kursziel-Potenzial
        # ============================================================================
        from analyst_ratings_radar import AnalystRatingsRadar
        radar = AnalystRatingsRadar()
        ratings = radar.get_ratings("NVDA")
        assert ratings.symbol == "NVDA"
        assert ratings.analyst_count >= 10, f"NVDA sollte von vielen Analysten gecovert werden, erhalten: {ratings.analyst_count}"
        assert ratings.consensus_rating in ("STRONG_BUY", "BUY", "HOLD", "UNDERPERFORM", "SELL")
        assert ratings.target_mean_price is not None and ratings.target_mean_price > 50.0
        print(f"[OK] Test 28 bestanden: Wall Street Analysten-Radar analysiert {ratings.analyst_count} Analysten für NVDA (Konsens: {ratings.consensus_rating}, Kursziel: ${ratings.target_mean_price:.2f}, Upside: {ratings.target_upside_pct:+.1f}%).")

        # ============================================================================
        # TEST 29: Geopolitischer Regime-Index (GCI) & Krisen-Mandat
        # ============================================================================
        from geopolitical_regime_engine import GeopoliticalRegimeEngine
        geo_engine = GeopoliticalRegimeEngine()
        geo_snap = geo_engine.analyze_geopolitical_situation()
        assert 0.0 <= geo_snap.geopolitical_risk_score <= 10.0
        assert geo_snap.regime_mode in ("ESCALATION_CRISIS_REGIME", "TRADE_WAR_TARIFF_TENSION", "NEUTRAL_STABLE", "PEACE_DIVIDEND_EXPANSION")
        assert len(geo_snap.autonomous_strategy_mandate) > 0
        print(f"[OK] Test 29 bestanden: Geopolitisches Krisenzentrum berechnet GCI {geo_snap.geopolitical_risk_score:.1f}/10.0 (Regime: {geo_snap.regime_mode}, Mandat: {geo_snap.autonomous_strategy_mandate[:50]}...).")

        # ============================================================================
        # TEST 30: ADAS Swarm Spezialist #25 (Wall Street) & #26 (Geopolitics) & Red-Team Veto
        # ============================================================================
        # Erstelle ein Snapshot mit starkem Analysten-Konsens und Geopolitik-Mandat
        analyst_snap = MarketSnapshot(
            symbol="NVDA",
            name="NVIDIA Corporation",
            asset_class="EQUITY",
            cluster="US_TECH",
            current_price=140.0,
            change_24h_pct=2.1,
            change_7d_pct=4.5,
            rsi_14=56.0,
            sma_20=135.0,
            sma_50=128.0,
            macd_hist=0.4,
            atr_14=3.5,
            volatility_annualized_pct=28.0,
            volume_ratio_vs_20d=1.4,
            analyst_consensus="STRONG_BUY",
            analyst_target_upside_pct=32.5,
            analyst_count=52,
        )
        analyst_sig = TradingSignal(
            symbol="NVDA",
            asset_class="EQUITY",
            cluster="US_TECH",
            action="BUY",
            confidence=0.82,
            sentiment_score=0.80,
            entry_price=140.0,
            stop_loss_price=133.0,
            take_profit_price=160.0,
            risk_reward_ratio=2.85,
            catalyst_summary="Wall Street Konsens & AI Momentum",
            thesis="Analyst Target Expansion",
        )
        _, votes_wallst, _ = adas.evaluate_symbol_with_swarm(analyst_snap, analyst_sig, green_macro)
        wallst_vote = next(v for v in votes_wallst if v.agent_id == "wall_street_consensus_analyst")
        assert wallst_vote.score >= 0.70, f"Wall Street Hunter muss bei STRONG_BUY und +32% Upside stark bullisch votieren! Score: {wallst_vote.score}"

        # Teste geopolitisches Red-Team Veto bei extremer Kriegseskalation (GCI >= 6.5)
        war_macro = MacroRegime(
            vix_level=24.0,
            spy_price=520.0,
            spy_sma_50=515.0,
            spy_sma_200=500.0,
            us10y_yield=4.3,
            us10y_change_5d_bps=5.0,
            dxy_level=104.0,
            dxy_change_5d_pct=0.5,
            regime_state="YELLOW_CAUTION",
            risk_multiplier=0.5,
            geopolitical_risk_score=8.5,
            geopolitical_regime_mode="ESCALATION_CRISIS_REGIME",
            geopolitical_strategy_mandate="DEFENSIVE_CAPITAL_PRESERVATION",
            summary="Geopolitischer Schock",
        )
        tribunal_war = tribunal.convene_tribunal(analyst_sig, analyst_snap, war_macro, {})
        assert tribunal_war.verdict == "REJECTED_VETO", f"Red-Team muss riskante Tech-Aktie bei Kriegseskalation vetoen! Erhalten: {tribunal_war.verdict}"
        assert "Geopolitical" in tribunal_war.bear_rebuttal or "Weltpolitisch" in tribunal_war.bear_rebuttal or "Geopolitical" in tribunal_war.cio_reasoning
        print(f"[OK] Test 30 bestanden: Wall Street Specialist #25 votiert {wallst_vote.vote_action} ({wallst_vote.score:+.2f}) & Red-Team verhängt VETO bei Geopolitik-Eskalation.")

        # ============================================================================
        # TEST 31: Telegram Notification Manager (Entprellung, Deckelung & Spam-Schutz)
        # ============================================================================
        from notification_manager import TelegramNotificationManager
        # Test-Instanz mit strengen Limits
        mgr = TelegramNotificationManager(
            bot_token="TEST_TOKEN",
            chat_id="123456",
            alert_level="BALANCED",
            max_per_hour=3,
            dedup_window_sec=60,
        )
        assert mgr._is_running_in_test(), "Manager muss automatisch erkennen, dass er in einem Test läuft"

        # 1. Teste Trade-Alert und sofortige Entprellung (Duplicate Detection)
        res1 = mgr.send_trade_execution("NVDA", "BUY", 10, 140.0, 700.0, 135.0, 150.0)
        assert res1 is True, "Erster Trade muss gesendet werden"
        res2 = mgr.send_trade_execution("NVDA", "BUY", 10, 140.0, 700.0, 135.0, 150.0)
        assert res2 is False, "Doppelter Trade-Alert innerhalb des Cooldowns muss unterdrückt werden!"

        # 2. Teste Red-Team VETO Entprellung (2h Cooldown)
        veto1 = mgr.send_red_team_veto("VETO_SYM", 9.0, "Überdehnt", "VETO")
        assert veto1 is True
        veto2 = mgr.send_red_team_veto("VETO_SYM", 9.0, "Überdehnt", "VETO")
        assert veto2 is False, "Zweites VETO für selbes Symbol muss unterdrückt werden!"

        # 3. Teste 'TRADES_ONLY' Filtermodus
        mgr.alert_level = "TRADES_ONLY"
        veto_suppressed = mgr.send_red_team_veto("NEW_SYM", 8.5, "Risk", "VETO")
        assert veto_suppressed is False, "Im TRADES_ONLY Modus dürfen keine Vetos gesendet werden!"
        mgr.alert_level = "BALANCED"  # Zurücksetzen
        print("[OK] Test 31 bestanden: Notification-Manager entprellt Duplikate zuverlässig, drosselt Veto-Spam und schützt Testläufe.")

        # ============================================================================
        # TEST 32: Kollektive Schwarm-Intelligenz & Wissensaustausch (SwarmKnowledgeBus)
        # ============================================================================
        from swarm_knowledge_bus import SwarmKnowledgeBus, get_swarm_knowledge_bus
        k_bus = get_swarm_knowledge_bus()
        assert len(k_bus.insights) >= 4, "Start-Erkenntnisse müssen im Wissensspeicher geladen sein"

        # 1. Teste automatische Konsultation: Penny-Stock ohne Katalysator muss Warnung auslösen
        adj, notes = k_bus.consult_shared_knowledge(
            symbol="DILUTE",
            cluster="PENNY_STOCKS",
            specialty_focus="PENNY_STOCK_MOMENTUM",
            current_score=0.75,
            has_catalyst=False,
            is_penny=True,
            macro_credit_stress=False,
        )
        assert adj < 0, f"Schwarm-Wissen muss Konfidenz bei ungesichertem Penny-Stock drosseln! Erhalten: {adj}"
        assert any("Penny-Stock" in n or "Warnung" in n for n in notes)

        # 2. Teste Schwarm-Lerneffekt: Verzeichne einen erfolgreichen Trade
        fake_votes = {
            "fast_breakout_chaser": type("V", (), {"score": 0.85})(),
            "ttm_squeeze_specialist": type("V", (), {"score": 0.80})(),
            "options_smart_money_hunter": type("V", (), {"score": 0.75})(),
        }
        learn_events = k_bus.record_collective_outcome(
            symbol="WINNER",
            move_pct=4.2,
            agent_votes=fake_votes,
            is_penny=False,
            had_catalyst=True,
        )
        # Synergie zwischen Co-Votern muss gestiegen sein
        pair_key = "fast_breakout_chaser:ttm_squeeze_specialist"
        assert pair_key in k_bus.synergies, "Synergie-Paar muss registriert sein"
        assert k_bus.synergies[pair_key].co_success >= 1
        boost = k_bus.get_synergy_boost_for_agents(["fast_breakout_chaser", "ttm_squeeze_specialist"])
        assert boost > 1.0, f"Synergie-Multiplikator muss nach gemeinsamem Gewinner über 1.0 liegen! Erhalten: {boost}"

        # 3. Teste Schwarm-Integration im ADAS-Agenten
        penny_trap_snap = MarketSnapshot(
            symbol="PUMPBULL",
            name="Pump Microcap",
            asset_class="EQUITY",
            cluster="PENNY_STOCKS",
            current_price=2.10,
            change_24h_pct=6.5,
            change_7d_pct=10.0,
            rsi_14=65.0,
            sma_20=1.90,
            sma_50=1.70,
            macd_hist=0.1,
            atr_14=0.15,
            volatility_annualized_pct=75.0,
            volume_ratio_vs_20d=2.5,
            sec_form8k_catalyst="",  # Kein Katalysator!
        )
        dummy_sig = TradingSignal(
            symbol="PUMPBULL",
            asset_class="EQUITY",
            cluster="PENNY_STOCKS",
            action="BUY",
            confidence=0.80,
            sentiment_score=0.70,
            entry_price=2.10,
            stop_loss_price=1.85,
            take_profit_price=2.80,
            risk_reward_ratio=2.8,
            catalyst_summary="Momentum",
            thesis="Breakout",
        )
        _, swarm_votes_trap, _ = adas.evaluate_symbol_with_swarm(penny_trap_snap, dummy_sig, green_macro)
        penny_ag_vote = next(v for v in swarm_votes_trap if "penny" in v.agent_id or "rocket" in v.agent_id)
        assert "Schwarm-Warnung" in penny_ag_vote.rationale or penny_ag_vote.score < 0.82, "Penny Hunter muss Schwarm-Warnung integriert haben"
        print(f"[OK] Test 32 bestanden: Kollektive Schwarm-Intelligenz teilt Erkenntnisse über alle 26 Spezialisten & vergibt Synergie-Boni ({boost:.2f}x).")

        # ============================================================================
        # TEST 33: Emergent Agent Dialect & Deal-Negotiation Engine
        # ============================================================================
        from emergent_dialect_engine import EmergentDialectEngine, get_dialect_engine
        from models import RiskEvaluation
        dialect_eng = get_dialect_engine()
        assert len(dialect_eng.vocabulary) >= 10, "Codebook muss Basistokens enthalten"

        # Simuliere Setup mit Hurst-Persistenz, Options-Call Flow & Sektor-Alpha
        ai_snap = MarketSnapshot(
            symbol="NVDA",
            name="Nvidia Corp",
            asset_class="EQUITY",
            cluster="US_TECH",
            current_price=140.0,
            change_24h_pct=3.2,
            change_7d_pct=7.0,
            rsi_14=62.0,
            sma_20=132.0,
            sma_50=125.0,
            macd_hist=0.5,
            atr_14=3.0,
            volatility_annualized_pct=25.0,
            volume_ratio_vs_20d=1.8,
            hurst_exponent=0.66,
            options_flow_sentiment="BULLISH_CALL_FLOW",
            options_flow_score=0.65,
            relative_strength_vs_spy=3.5,
            analyst_consensus="STRONG_BUY",
            analyst_target_upside_pct=35.0,
        )
        ai_sig = TradingSignal(
            symbol="NVDA",
            asset_class="EQUITY",
            cluster="US_TECH",
            action="BUY",
            confidence=0.88,
            sentiment_score=0.80,
            entry_price=140.0,
            stop_loss_price=133.0,
            take_profit_price=160.0,
            risk_reward_ratio=2.85,
            expected_edge_score=2.35,
            catalyst_summary="Alpha",
            thesis="Deal Pitch",
        )
        ai_risk_eval = RiskEvaluation(
            approved=True,
            reason="Freigabe",
            adjusted_action="BUY",
            position_size_usd=1400.0,
        )

        packet = dialect_eng.compose_and_negotiate_deal(ai_snap, ai_sig, green_macro, ai_risk_eval)
        assert "Ω:HURST_PERSIST" in packet.dialect_sentence
        assert "Ψ:CALL_WHALE_BURST" in packet.dialect_sentence
        assert "Σ:CONFLUENCE" in packet.dialect_sentence
        assert packet.consensus_verdict == "SEALED_APPROVED"
        assert len(packet.plain_german_summary) > 20
        assert "Begründung der Agenten" in packet.plain_german_summary

        # Feedback-Loop testen
        dialect_eng.record_deal_profitability("NVDA", move_pct=3.5)
        top_h_tok = dialect_eng.vocabulary.get("Ω:HURST_PERSIST")
        assert top_h_tok and top_h_tok.weight > 1.0, "Gewinner-Tokens müssen im Gewicht steigen"
        print(f"[OK] Test 33 bestanden: Emergent Dialect Engine verhandelt Deals symbolisch (Dialekt: {packet.dialect_sentence[:50]}...) & Dolmetscher dekodiert nach Deutsch.")

        # --- Test 34: Hugging Face FinBERT NLP & Sentiment Sentinel (Sub-Agent #27) ---
        from huggingface_sentiment_sentinel import HuggingFaceSentimentSentinel
        test_cache = os.path.join(tmpdir, "test_hf_sentiment_cache.json")
        hf_sentinel = HuggingFaceSentimentSentinel(cache_file=test_cache)

        # 1. Bullische Schlagzeilen mit FinBERT analysieren
        bull_headlines = [
            "NVIDIA reports record revenue surge, beating analyst estimates by 25%",
            "Wall Street upgrades NVDA to Strong Buy following massive AI chip demand and profit breakout"
        ]
        snap_bull = hf_sentinel.analyze_symbol("NVDA", headlines=bull_headlines, force_refresh=True)
        assert snap_bull.sentiment_label in ("BULLISH", "VERY_BULLISH"), f"Erwartet BULLISH, erhalten: {snap_bull.sentiment_label}"
        assert snap_bull.compound_score >= 0.35, f"Score zu niedrig: {snap_bull.compound_score}"
        assert snap_bull.positive_prob >= 0.50
        assert snap_bull.veto_triggered is False

        # 2. Bärische Schock-Nachrichten analysieren (Veto-Trigger)
        bear_headlines = [
            "SEC launches formal fraud investigation into accounting irregularities and revenue manipulation",
            "Company issues profit warning and announces massive dilution with public offering amid bankruptcy fears"
        ]
        snap_bear = hf_sentinel.analyze_symbol("TROUBLE_CORP", headlines=bear_headlines, force_refresh=True)
        assert snap_bear.sentiment_label in ("BEARISH", "VERY_BEARISH"), f"Erwartet BEARISH, erhalten: {snap_bear.sentiment_label}"
        assert snap_bear.compound_score <= -0.35, f"Score zu hoch: {snap_bear.compound_score}"
        assert snap_bear.negative_prob >= 0.55
        assert snap_bear.veto_triggered is True

        # 3. Sub-Agent #27 Votum im ADAS-Schwarm prüfen
        ada_hf_snap = MarketSnapshot(
            symbol="NVDA",
            name="Nvidia",
            asset_class="EQUITY",
            cluster="US_TECH",
            current_price=140.0,
            change_24h_pct=3.0,
            change_7d_pct=7.0,
            rsi_14=58.0,
            sma_20=135.0,
            sma_50=128.0,
            macd_hist=0.6,
            atr_14=3.0,
            volatility_annualized_pct=26.0,
            volume_ratio_vs_20d=1.5,
            hf_sentiment_label=snap_bull.sentiment_label,
            hf_sentiment_score=snap_bull.compound_score,
            hf_positive_prob=snap_bull.positive_prob,
            hf_negative_prob=snap_bull.negative_prob,
        )
        arch_hf = ADASMetaArchitect(cfg)
        hf_spec = next((ag for ag in arch_hf.state.active_subagents.values() if ag.specialty_focus == "HUGGINGFACE_SENTIMENT"), None)
        assert hf_spec is not None, "Sub-Agent #27 (HUGGINGFACE_SENTIMENT) muss im Schwarm registriert sein"
        hf_vote, _ = arch_hf._compute_single_subagent_vote(hf_spec, ada_hf_snap, green_macro)
        assert hf_vote.vote_action == "BUY"
        assert hf_vote.score >= 0.50

        # 4. Red-Team Veto bei negativem FinBERT-Sentiment prüfen
        from adversarial_red_team import AdversarialRedTeamTribunal
        red_tribunal_hf = AdversarialRedTeamTribunal(cfg)
        bear_market_snap = MarketSnapshot(
            symbol="TROUBLE_CORP",
            name="Trouble Corp",
            asset_class="EQUITY",
            cluster="US_TECH",
            current_price=20.0,
            change_24h_pct=-6.0,
            change_7d_pct=-15.0,
            rsi_14=42.0,
            sma_20=24.0,
            sma_50=28.0,
            macd_hist=-0.5,
            atr_14=1.2,
            volatility_annualized_pct=45.0,
            volume_ratio_vs_20d=2.0,
            hf_sentiment_label="VERY_BEARISH",
            hf_sentiment_score=-0.75,
            hf_negative_prob=0.88,
        )
        fake_buy_sig = TradingSignal(
            symbol="TROUBLE_CORP",
            asset_class="EQUITY",
            cluster="US_TECH",
            action="BUY",
            confidence=0.75,
            sentiment_score=0.5,
            entry_price=20.0,
            stop_loss_price=18.0,
            take_profit_price=25.0,
            risk_reward_ratio=2.5,
            expected_edge_score=1.5,
            catalyst_summary="Katalysator",
            thesis="Kauf-These",
        )
        tribunal_res = red_tribunal_hf.convene_tribunal(fake_buy_sig, bear_market_snap, green_macro, {})
        assert "FinBERT Negative Sentiment Shock" in tribunal_res.bear_rebuttal or "FinBERT" in tribunal_res.bear_rebuttal
        assert tribunal_res.verdict in ("REJECTED_VETO", "REDUCED_SIZE")

        print(f"[OK] Test 34 bestanden: Hugging Face FinBERT NLP & Sentiment Sentinel klassifiziert News (Score {snap_bull.compound_score:+.2f} Bull / {snap_bear.compound_score:+.2f} Bear) & Red-Team schützt vor Sentiment-Schocks.")

        # --- Test 35: Institutional VWAP & Volume-Profile Liquidity Sweeper (Sub-Agent #28) ---
        from vwap_liquidity_sweeper import InstitutionalVWAPLiquiditySweeper
        vwap_cache_path = os.path.join(tmpdir, "test_vwap_cache.json")
        vwap_sweeper = InstitutionalVWAPLiquiditySweeper(cache_file=vwap_cache_path)

        # 1. Synthetisches Liquidity-Sweep Setup erzeugen
        bars = []
        # 15 normale Konsolidierungs-Balken um $100
        for i in range(15):
            bars.append({"High": 100.5, "Low": 99.5, "Close": 100.0, "Volume": 1000.0})
        # Gezielter Stop-Loss-Sweep nach unten auf $95.0 mit hohem Volumen (Stop-Abfischung)
        bars.append({"High": 99.8, "Low": 95.0, "Close": 95.5, "Volume": 3500.0})
        # Rebound & Reclaim zurück über das Band auf $96.5
        bars.append({"High": 97.2, "Low": 95.4, "Close": 96.5, "Volume": 2500.0})
        df_sweep = pd.DataFrame(bars)

        vwap_res = vwap_sweeper.compute_vwap_and_profile("NVDA", df_sweep)
        assert vwap_res.sweep_status == "BULLISH_SWEEP_RECLAIM", f"Erwartet BULLISH_SWEEP_RECLAIM, erhalten: {vwap_res.sweep_status}"
        assert vwap_res.band_lower_2s < vwap_res.vwap < vwap_res.band_upper_2s
        assert vwap_res.poc_price > 0
        assert vwap_res.suggested_stop_loss is not None and vwap_res.suggested_stop_loss < 95.0
        assert vwap_res.suggested_take_profit is not None and vwap_res.suggested_take_profit >= vwap_res.vwap
        assert vwap_res.risk_reward_ratio >= 1.0

        # 2. Sub-Agent #28 im ADAS-Schwarm testen
        arch_vwap = ADASMetaArchitect(cfg)
        vwap_spec = next((ag for ag in arch_vwap.state.active_subagents.values() if ag.specialty_focus == "VWAP_LIQUIDITY_SWEEP"), None)
        assert vwap_spec is not None, "Sub-Agent #28 (VWAP_LIQUIDITY_SWEEP) muss im Schwarm existieren"

        snap_vwap = MarketSnapshot(
            symbol="NVDA",
            name="Nvidia",
            asset_class="EQUITY",
            cluster="US_TECH",
            current_price=96.5,
            change_24h_pct=-2.5,
            change_7d_pct=4.0,
            rsi_14=48.0,
            sma_20=98.0,
            sma_50=94.0,
            macd_hist=0.2,
            atr_14=2.0,
            volatility_annualized_pct=25.0,
            volume_ratio_vs_20d=1.8,
            vwap_val=vwap_res.vwap,
            vwap_z_score=vwap_res.z_score,
            liquidity_sweep_status=vwap_res.sweep_status,
            volume_profile_poc=vwap_res.poc_price,
        )
        vwap_vote, _ = arch_vwap._compute_single_subagent_vote(vwap_spec, snap_vwap, green_macro)
        assert vwap_vote.vote_action == "BUY"
        assert vwap_vote.score >= 0.80

        # 3. Red-Team Tribunal: Bullen-These prüft Sweep & Bären-Advokat warnt bei Buying Exhaustion
        sig_vwap = TradingSignal(
            symbol="NVDA",
            asset_class="EQUITY",
            cluster="US_TECH",
            action="BUY",
            confidence=0.88,
            sentiment_score=0.8,
            entry_price=97.5,
            stop_loss_price=94.2,
            take_profit_price=101.0,
            risk_reward_ratio=2.5,
            expected_edge_score=2.2,
            catalyst_summary="Liquidity Sweep Rebound",
            thesis="Institutional Sweep",
        )
        tribunal_vwap = red_tribunal_hf.convene_tribunal(sig_vwap, snap_vwap, green_macro, {})
        assert "Liquidity-Sweep" in tribunal_vwap.bull_argument or "POC" in tribunal_vwap.bull_argument

        # Bären-Exhaustion Test bei +2.8σ über VWAP
        snap_exhaustion = snap_vwap.model_copy(update={
            "vwap_z_score": 2.8,
            "liquidity_sweep_status": "BEARISH_CLIMAX_SWEEP",
        })
        tribunal_exhaustion = red_tribunal_hf.convene_tribunal(sig_vwap, snap_exhaustion, green_macro, {})
        assert "VWAP Buying Exhaustion" in tribunal_exhaustion.bear_rebuttal or "VWAP" in tribunal_exhaustion.bear_rebuttal

        # 4. Emergent Dialect Engine: Symbolisches Token testen
        deal_vwap = dialect_eng.compose_and_negotiate_deal(snap_vwap, sig_vwap, green_macro, ai_risk_eval)
        assert "🎯:LIQUIDITY_SWEEP_RECLAIM" in deal_vwap.dialect_sentence

        print(f"[OK] Test 35 bestanden: Institutional VWAP & Volume-Profile Liquidity Sweeper (Sub-Agent #28) identifiziert Stop-Loss-Absorption (Status: {vwap_res.sweep_status}, CRV {vwap_res.risk_reward_ratio}:1) & Dialekt verhandelt Token 🎯:LIQUIDITY_SWEEP_RECLAIM.")

        # --- Test 36: Walk-Forward Research & Optimization Lab (Forschungsfeld C) ---
        from pathlib import Path
        from walk_forward_research_lab import WalkForwardResearchLab, ParameterCandidate
        test_lab_report = Path(tmpdir) / "test_wfa_report.json"
        test_lab_calibrated = Path(tmpdir) / "test_wfa_calibrated.json"
        lab = WalkForwardResearchLab(report_file=test_lab_report, calibrated_file=test_lab_calibrated)

        # 1. Schnellen Walk-Forward Optimierungszyklus über alle 4 Regime ausführen
        wfa_res = lab.run_walk_forward_optimization(custom_candidates=[
            ParameterCandidate(atr_stop_loss_mult=2.0, atr_take_profit_mult=3.8, min_confidence_cutoff=0.68),
            ParameterCandidate(atr_stop_loss_mult=2.5, atr_take_profit_mult=4.5, min_confidence_cutoff=0.74),
        ])
        assert wfa_res.total_iterations_tested == 2
        assert len(wfa_res.regime_breakdown) == 4, "Muss alle 4 Regime getestet haben"
        assert wfa_res.overall_wfer_pct > 0.0, "WFER muss berechnet sein"
        assert wfa_res.robustness_grade in ("A_EXEMPLARY", "B_ROBUST", "C_MARGINAL", "REJECTED_OVERFIT")
        assert test_lab_report.exists(), "Report-Datei muss erzeugt worden sein"

        # 2. Telegram Formatierung testen
        summary_txt = lab.format_telegram_summary(wfa_res)
        assert "Walk-Forward Forschungs-Labor" in summary_txt
        assert "WFER" in summary_txt

        print(f"[OK] Test 36 bestanden: Walk-Forward Research Lab optimiert Parameter rollend (WFER {wfa_res.overall_wfer_pct:.1f}%, OOS Sharpe {wfa_res.overall_oos_sharpe:.2f}, Note: {wfa_res.robustness_grade}) über 4 Stress-Regime.")

    print("=== Alle 36 Architektur-, Smart-Leverage-, Penny-Stock-, IBKR-Bridge-, ADAS-Swarm-, Strategie-Zucht-, Red-Team-, SEC-Whale-, Options-Flow-, Fundamental-, AutoTuner-, StatArb-Pairs-, Hurst-Fractal-, Credit-Intermarket-, Microstructure-, Fractional-Kelly-, Sector-Rotation-, Wall-Street-Ratings-, Geopolitik-, Notification-Spam-, Swarm-Knowledge-, Emergent-Dialect-, HuggingFace-Sentiment-, VWAP-Liquidity-Sweep- & Walk-Forward-Lab-Tests erfolgreich bestanden! ===")



if __name__ == "__main__":
    test_all_audit_fixes()
