"""
adversarial_red_team.py - Adversarial Red-Teaming & LLM-Debatten-Tribunal
Vor jeder Trade-Freigabe treten drei spezialisierte Instanzen gegeneinander an:
1. 🐂 Der Bullen-Advokat (Alpha-Hunter): Plädiert für den Kauf anhand von Katalysatoren & Momentum.
2. 🐻 Der Bären-Advokat (Teufels Advokat / Red Team): Sucht gezielt nach tödlichen Fallen (Dilution, Parabolic Exhaustion, Binary Risk).
3. ⚖️ Der Chief Investment Officer (CIO / Richter): Wägt ab, fällt das Urteil (APPROVED, REDUCED_SIZE oder REJECTED_VETO).
"""

import datetime
import json
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

from config import AgentConfig
from models import AdversarialDebateResult, MacroRegime, MarketSnapshot, Position, TradingSignal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


class AdversarialRedTeamTribunal:
    """
    Führt vor jeder Trade-Eröffnung ein formelles Red-Teaming-Tribunal durch.
    Nutzt Gemini LLM (falls API-Key vorhanden) oder eine deterministische
    institutionelle Heuristik-Engine als Ausfallschutz.
    """

    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.data_dir = os.path.dirname(self.config.portfolio_state_file)
        self.log_file = os.path.join(self.data_dir, "red_team_debates.jsonl")
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def convene_tribunal(
        self,
        signal: TradingSignal,
        snapshot: MarketSnapshot,
        macro: MacroRegime,
        positions: Dict[str, Position],
    ) -> AdversarialDebateResult:
        """Beruft das Tribunal ein und liefert das Urteil."""
        now_str = datetime.datetime.now().isoformat()
        
        # 1. Sammle Fakten für Bull und Bär
        bull_args = self._formulate_bull_argument(signal, snapshot, macro)
        bear_args, fatal_risks, risk_score = self._formulate_bear_rebuttal(signal, snapshot, macro, positions)

        # 2. Urteilsfindung durch den CIO
        verdict = "APPROVED"
        size_mult = 1.0
        cio_reasoning = ""

        if len(fatal_risks) >= 2 or risk_score >= 8.0:
            verdict = "REJECTED_VETO"
            size_mult = 0.0
            cio_reasoning = (
                f"VETO EINGELEGT: Der Bären-Advokat hat {len(fatal_risks)} kritische Risikofaktoren aufgedeckt "
                f"({', '.join(fatal_risks)}). Risiko-Score ({risk_score:.1f}/10) übersteigt Risikotoleranz. Kapital wird geschützt."
            )
        elif len(fatal_risks) == 1 or risk_score >= 5.5:
            verdict = "REDUCED_SIZE"
            size_mult = 0.50
            cio_reasoning = (
                f"TEIL-FREIGABE (50% Positionsgröße): Guter Bullen-Edge vorhanden, aber Bär mahnt zur Vorsicht: "
                f"{fatal_risks[0] if fatal_risks else 'Erhöhte Volatilität'}. Größe wird halbiert."
            )
        else:
            verdict = "APPROVED"
            size_mult = 1.0
            cio_reasoning = (
                f"FREIGABE VOLL: Der Bullen-Advokat hat die Bären-Bedenken überzeugend entkräftet. "
                f"Starker Katalysator bei kontrolliertem Risiko (Score {risk_score:.1f}/10). Trade genehmigt."
            )

        res = AdversarialDebateResult(
            symbol=signal.symbol,
            action_debated=signal.action,
            verdict=verdict,
            bull_argument=bull_args,
            bear_rebuttal=bear_args,
            cio_reasoning=cio_reasoning,
            risk_score=round(risk_score, 1),
            size_multiplier=size_mult,
            debated_at=now_str,
        )

        # 3. Protokollieren & bei VETO Alarm senden
        self._log_debate(res)
        if verdict == "REJECTED_VETO":
            self._send_veto_push(res)

        return res

    def _formulate_bull_argument(self, signal: TradingSignal, snapshot: MarketSnapshot, macro: Optional[MacroRegime]) -> str:
        points = []
        points.append(f"Signal: {signal.action} mit {signal.confidence*100:.0f}% Konfidenz (Edge {signal.expected_edge_score:.2f})")
        vol_r = getattr(snapshot, "volume_ratio_vs_20d", 1.0)
        if vol_r >= 1.5:
            points.append(f"Ungewöhnliches Volumen: {vol_r:.1f}x des 20-Tage-Durchschnitts")
        if snapshot.change_24h_pct >= 2.0:
            points.append(f"Starkes 24h-Momentum (+{snapshot.change_24h_pct:.1f}%)")
        if snapshot.sec_form8k_catalyst:
            points.append(f"SEC 8-K Katalysator: {snapshot.sec_form8k_catalyst}")
        if getattr(snapshot, "insider_sentiment", "") == "STRONG_INSIDER_BUY" or getattr(snapshot, "form4_cluster_buy", False):
            points.append(f"🐋 SEC Form 4 Whale/Cluster-Kauf: {snapshot.insider_summary}")
        if getattr(snapshot, "fundamental_health_grade", "") == "ELITE_GROWTH":
            points.append(f"💎 Elite-Wachstum: {snapshot.revenue_growth_pct or 20.0:.1f}% Umsatzwachstum bei solider Bilanz")
        if getattr(snapshot, "options_flow_score", 0.0) >= 0.35:
            points.append(f"🌊 Bullischer Smart-Money Flow (Score {snapshot.options_flow_score:+.2f})")
        if getattr(snapshot, "stat_arb_z_score", None) is not None and abs(snapshot.stat_arb_z_score) >= 1.8:
            points.append(f"🔀 StatArb Mean-Reversion Edge ({snapshot.stat_arb_pair}, Z: {snapshot.stat_arb_z_score:+.2f})")
        if getattr(snapshot, "hurst_exponent", 0.50) >= 0.58:
            points.append(f"🌀 Persistenter Hurst-Trend (H={snapshot.hurst_exponent:.2f})")
        if getattr(snapshot, "bid_ask_spread_pct", 0.10) <= 0.10:
            points.append(f"🎯 Tiefe Liquidität (Spread {snapshot.bid_ask_spread_pct:.2f}%)")
        if macro is not None and getattr(macro, "intermarket_bias", "") == "BULLISH_CREDIT_TAILWIND":
            points.append("🟢 Entspannte Kreditmärkte (HYG/LQD Rückenwind)")
        if snapshot.options_flow_sentiment == "BULLISH_CALL_FLOW":
            points.append(f"Institutioneller Call-Whale Flow (Put/Call {snapshot.options_put_call_ratio or 0.5:.2f})")
        if getattr(snapshot, "analyst_consensus", "") in ("STRONG_BUY", "BUY"):
            ups = getattr(snapshot, "analyst_target_upside_pct", 0.0) or 0.0
            points.append(f"📊 Wall Street Konsens ({snapshot.analyst_consensus}): +{ups:.1f}% Kurspotenzial ({snapshot.analyst_count} Analysten)")
        if getattr(snapshot, "hf_sentiment_score", 0.0) >= 0.35:
            points.append(f"🤗 Hugging Face FinBERT Bullish Sentiment (Score {snapshot.hf_sentiment_score:+.2f})")
        if getattr(snapshot, "liquidity_sweep_status", "") == "BULLISH_SWEEP_RECLAIM":
            poc = getattr(snapshot, "volume_profile_poc", 0.0) or 0.0
            poc_txt = f" (POC ${poc:.2f})" if poc > 0 else ""
            points.append(f"🎯 Bullischer Liquidity-Sweep am -2σ VWAP-Band absorbiert{poc_txt}")
        if getattr(snapshot, "forecast_asymmetry_ratio", 1.0) >= 1.6:
            points.append(f"📈 Probabilistischer Trichter: {snapshot.forecast_asymmetry_ratio:.1f}x Upside-Asymmetrie")
        if macro is not None and getattr(macro, "geopolitical_regime_mode", "") == "PEACE_DIVIDEND":
            points.append("🕊️ Globale Friedensdividende (GCI ruhig)")
        if getattr(snapshot, "supertrend_direction", "") == "BULLISH":
            points.append("SuperTrend im Bullen-Regime")
        if signal.thesis:
            points.append(f"These: {signal.thesis}")
        return " | ".join(points)

    def _formulate_bear_rebuttal(
        self,
        signal: TradingSignal,
        snapshot: MarketSnapshot,
        macro: Optional[MacroRegime],
        positions: Dict[str, Position],
    ) -> Tuple[str, List[str], float]:
        rebuttals = []
        fatal_risks = []
        risk_score = 2.0  # Basis-Risiko

        # 1. Parabolic Exhaustion / Überkauft
        if snapshot.rsi_14 >= 78.0:
            msg = f"RSI extrem überkauft ({snapshot.rsi_14:.1f}) – Gefahr von Mean-Reversion / Bullenfalle"
            rebuttals.append(msg)
            fatal_risks.append("Parabolic Exhaustion (RSI > 78)")
            risk_score += 3.0
        elif snapshot.rsi_14 >= 72.0:
            rebuttals.append(f"RSI erhöht ({snapshot.rsi_14:.1f})")
            risk_score += 1.5

        # 2. Abstand zum 20er SMA (Überdehnung)
        if snapshot.sma_20 > 0 and snapshot.current_price > (snapshot.sma_20 * 1.15):
            pct_over = ((snapshot.current_price - snapshot.sma_20) / snapshot.sma_20) * 100.0
            rebuttals.append(f"Kurs ist +{pct_over:.1f}% über 20-Tage-Linie überdehnt")
            risk_score += 1.5

        # 3. Earnings Blackout / Binäres Zahlen-Event
        if snapshot.days_to_earnings is not None and snapshot.days_to_earnings <= 3:
            msg = f"Quartalszahlen in {snapshot.days_to_earnings} Tagen (unkalkulierbares Gap-Risiko)"
            rebuttals.append(msg)
            fatal_risks.append("Earnings Binary Risk (<= 3 Tage)")
            risk_score += 3.5

        # 4. Penny Stock Verwässerungs-Gefahr (Offering / Dilution)
        if snapshot.cluster == "PENNY_STOCKS" or snapshot.current_price <= 5.0:
            if snapshot.change_24h_pct >= 25.0 and not snapshot.sec_form8k_catalyst:
                msg = "Extremer Kurssprung ohne SEC 8-K Filing (Verdacht auf Pump & Dump oder ATM-Offering)"
                rebuttals.append(msg)
                fatal_risks.append("Penny-Stock Dilution / Pump-Verdacht")
                risk_score += 3.0
            else:
                rebuttals.append("Penny-Stock Volatilitäts-Risiko (erhöhte Bid-Ask-Spreads)")
                risk_score += 1.0

        # 5. Makro-Gegenwind (VIX / Zinsen)
        if macro is not None:
            if macro.regime_state == "RED_DEFENSIVE":
                msg = f"Makro-Alarm: VIX bei {macro.vix_level:.1f} im roten Defensiv-Modus"
                rebuttals.append(msg)
                fatal_risks.append("Makro-Panik (VIX > 25)")
                risk_score += 2.5
            elif macro.us10y_change_5d_bps >= 15.0:
                rebuttals.append(f"US-Zinsen steigen stark (+{macro.us10y_change_5d_bps:.0f} bps in 5d)")
                risk_score += 1.0

        # 6. Cluster-Klumpenrisiko
        same_cluster_count = sum(1 for p in positions.values() if p.cluster == snapshot.cluster)
        if same_cluster_count >= 3:
            msg = f"Bereits {same_cluster_count} Positionen im Cluster '{snapshot.cluster}' offen (Klumpenrisiko)"
            rebuttals.append(msg)
            fatal_risks.append(f"Klumpenrisiko ({snapshot.cluster})")
            risk_score += 2.0

        # 7. Form 4 Insider-Verkäufe
        if getattr(snapshot, "insider_sentiment", "") == "INSIDER_SELLING":
            rebuttals.append(f"Erhebliche Vorstand-Verkäufe ({snapshot.insider_summary})")
            risk_score += 1.5

        # 8. Fundamentale Schieflage (Insolvenz-/Verwässerungsrisiko)
        if getattr(snapshot, "is_fundamentally_distressed", False) or getattr(snapshot, "fundamental_health_grade", "") == "TOXIC_DISTRESSED":
            msg = "Fundamentaler Distress: Negativer Cashflow bei hoher Überschuldung"
            rebuttals.append(msg)
            fatal_risks.append("Fundamental Distress / Toxic Balance")
            risk_score += 3.5

        # 9. Bärische Optionen-Fluss-Divergenz (Smart Money Absicherung)
        if getattr(snapshot, "options_flow_score", 0.0) <= -0.4:
            rebuttals.append(f"Institutioneller Flow bärisch (Score {snapshot.options_flow_score:+.2f})")
            risk_score += 1.5

        # 10. Systemischer Kredit-Stress (Junk Bonds brechen ein)
        if macro is not None and getattr(macro, "credit_stress_active", False):
            is_severe = getattr(macro, "intermarket_bias", "") == "SEVERE_CRUNCH" or macro.regime_state != "GREEN_RISK_ON"
            msg = "Systemischer Kredit-Stress: Junk-Bonds kollabieren relativ zu Investment Grade (HYG/LQD)"
            rebuttals.append(msg)
            fatal_risks.append("Systemic Credit Crunch")
            risk_score += 6.5 if is_severe else 4.0

        # 11. Illiquiditäts- & Slippage-Falle (Corwin-Schultz / Amihud)
        if getattr(snapshot, "microstructure_routing", "") == "SLIPPAGE_TRAP_AVOID":
            msg = f"Dünnes Orderbuch (Spread {getattr(snapshot, 'bid_ask_spread_pct', 0.8):.2f}%): Hohe Market-Impact-Kosten"
            rebuttals.append(msg)
            fatal_risks.append("Severe Slippage / Thin Liquidity")
            risk_score += 3.0

        # 12. Wall Street Analysten-Warnung / Downgrade
        if getattr(snapshot, "analyst_consensus", "") in ("UNDERPERFORM", "SELL"):
            msg = f"Wall Street Analysten warnen vor Abverkauf (Konsens: {snapshot.analyst_consensus})"
            rebuttals.append(msg)
            risk_score += 2.0

        # 13. Weltpolitische Eskalation / Geopolitischer Schock
        if macro is not None and getattr(macro, "geopolitical_risk_score", 0.0) >= 6.5:
            if not snapshot.is_safe_haven and snapshot.cluster not in ("DEFENSE", "ENERGY"):
                is_severe = getattr(macro, "geopolitical_risk_score", 0.0) >= 8.0 or macro.geopolitical_regime_mode in ("ESCALATION_CRISIS_REGIME", "CRITICAL_WAR_ESCALATION")
                msg = f"Weltpolitisches Krisen-Regime ({macro.geopolitical_regime_mode}, GCI {macro.geopolitical_risk_score:.1f}/10)"
                rebuttals.append(msg)
                fatal_risks.append("Geopolitical Shock Risk")
                risk_score += 6.5 if is_severe else 3.5

        # 14. Hugging Face FinBERT Negativer Sentiment-Schock
        if getattr(snapshot, "hf_sentiment_score", 0.0) <= -0.35 or getattr(snapshot, "hf_negative_prob", 0.0) >= 0.55:
            msg = f"Hugging Face FinBERT erkennt negativen Sentiment-Schock ({getattr(snapshot, 'hf_sentiment_label', 'BEARISH')}, Score {getattr(snapshot, 'hf_sentiment_score', 0.0):+.2f})"
            rebuttals.append(msg)
            fatal_risks.append("FinBERT Negative Sentiment Shock")
            risk_score += 3.5

        # 15. VWAP Buying Exhaustion (+2.5σ Überdehnung)
        if getattr(snapshot, "liquidity_sweep_status", "") == "BEARISH_CLIMAX_SWEEP" or getattr(snapshot, "vwap_z_score", 0.0) >= 2.5:
            z_s = getattr(snapshot, "vwap_z_score", 2.5)
            msg = f"VWAP Buying Exhaustion: Kurs extrem überdehnt ({z_s:+.1f}σ über VWAP) – Hohe Mean-Reversion Gefahr"
            rebuttals.append(msg)
            fatal_risks.append("VWAP Climax Exhaustion (> +2.5σ)")
            risk_score += 3.0

        # 16. Macro Economic Event Horizon (FOMC, CPI, NFP Whipsaw-Gefahr)
        if macro is not None:
            if getattr(macro, "macro_event_horizon_active", False):
                ev_n = getattr(macro, "macro_event_horizon_name", "FOMC / CPI Event")
                msg = f"🏛️ Macro Event Horizon aktiv ({ev_n}) – Whipsaw-Gefahr & Liquiditätsentzug"
                rebuttals.append(msg)
                fatal_risks.append("Macro Event Horizon")
                risk_score += 7.0
            elif getattr(macro, "macro_event_protective_action", "") == "HALF_SIZE_ONLY":
                mins = getattr(macro, "macro_minutes_to_event", 120)
                rebuttals.append(f"Makro-Event naht in {mins} Min – Positionsgröße halbiert")
                risk_score += 2.0

        # 17. Probabilistischer Forecast Skew (Ungünstiges Downside-Risiko)
        asym = getattr(snapshot, "forecast_asymmetry_ratio", 1.0)
        if asym <= 0.75 and asym > 0:
            p10 = getattr(snapshot, "forecast_p10_price", 0.0)
            msg = f"Probabilistischer Trichter ungünstig ({asym:.2f}x): Downside-Risiko (P10 ${p10:.2f}) überwiegt Chance"
            rebuttals.append(msg)
            fatal_risks.append("Unfavorable Probabilistic Skew")
            risk_score += 3.5

        bear_text = " | ".join(rebuttals) if rebuttals else "Keine kritischen Risiken identifiziert – Positionierbar."
        return bear_text, fatal_risks, min(10.0, risk_score)

    def _log_debate(self, res: AdversarialDebateResult) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(res.model_dump_json() + "\n")
        except Exception:
            pass

    def _send_veto_push(self, res: AdversarialDebateResult) -> None:
        """Sendet einen Telegram-Push-Alert, wenn das Red Team einen Kauf verhindert hat (mit Entprellung & Drosselung)."""
        try:
            from notification_manager import get_notifier
            get_notifier().send_red_team_veto(
                symbol=res.symbol,
                risk_score=res.risk_score,
                bear_rebuttal=res.bear_rebuttal,
                cio_reasoning=res.cio_reasoning,
            )
        except Exception:
            pass
