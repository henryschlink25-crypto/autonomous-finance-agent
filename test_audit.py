import os
import sys
import tempfile
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

    print("=== Alle 14 Architektur-, Smart-Leverage-, Penny-Stock-, IBKR-Bridge-, ADAS-Swarm- & Strategie-Zucht-Tests erfolgreich bestanden! ===")



if __name__ == "__main__":
    test_all_audit_fixes()
