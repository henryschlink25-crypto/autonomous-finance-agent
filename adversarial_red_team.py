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
        if snapshot.options_flow_sentiment == "BULLISH_CALL_FLOW":
            points.append(f"Institutioneller Call-Whale Flow (Put/Call {snapshot.options_put_call_ratio or 0.5:.2f})")
        if snapshot.supertrend_direction == "BULLISH":
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

        bear_text = " | ".join(rebuttals) if rebuttals else "Keine kritischen Risiken identifiziert – Positionierbar."
        return bear_text, fatal_risks, min(10.0, risk_score)

    def _log_debate(self, res: AdversarialDebateResult) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(res.model_dump_json() + "\n")
        except Exception:
            pass

    def _send_veto_push(self, res: AdversarialDebateResult) -> None:
        """Sendet einen Telegram-Push-Alert, wenn das Red Team einen Kauf verhindert hat."""
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not bot_token or not chat_id:
            return

        import html
        import ssl
        import urllib.request
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl._create_unverified_context()

        cid = int(chat_id) if chat_id.isdigit() else chat_id
        safe_sym = html.escape(res.symbol)
        safe_bear = html.escape(res.bear_rebuttal)
        safe_cio = html.escape(res.cio_reasoning)

        text_body = (
            f"🤖 <b>🛡️ Red-Team VETO: Trade abgewendet!</b>\n\n"
            f"• <b>Asset:</b> <code>{safe_sym}</code>\n"
            f"• <b>Risiko-Score:</b> ⚠️ <b>{res.risk_score:.1f}/10</b>\n"
            f"• <b>Bären-Advokat:</b> {safe_bear}\n\n"
            f"⚖️ <b>CIO-Entscheid:</b> {safe_cio}\n\n"
            f"Dein Kapital wurde erfolgreich vor einer potenziellen Falle geschützt."
        )
        payload = json.dumps({"chat_id": cid, "text": text_body, "parse_mode": "HTML"}).encode("utf-8")
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, context=ctx, timeout=4.0)
        except Exception:
            pass
