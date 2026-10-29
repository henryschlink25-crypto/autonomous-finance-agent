"""
smart_recovery_evaluator.py
Intelligente Erholungspotenzial-Prüfung & These-Validierung (Smart Recovery Evaluator).
Verhindert voreiliges Ausstoppen (Stop-Hunting / Whipsaw-Verluste) bei temporärem Rauschen / Liquidity Dips,
solange Rebound-Faktoren (Oszillator-Erschöpfung, VWAP-Support, Qlib Alpha158, Dark Pool) aktiv sind
und der harte Notfall-Stopp nicht durchbrochen wurde.
"""

from typing import List, Optional
from pydantic import BaseModel
from models import Position, MarketSnapshot


class RecoveryEvaluationResult(BaseModel):
    symbol: str
    can_recover: bool
    recovery_score: float  # 0.0 bis 4.0+
    hard_emergency_exit: bool
    reasons: List[str]
    summary: str


class SmartRecoveryEvaluator:
    """
    Evaluiert, ob eine im Minus befindliche Position hohes Erholungspotenzial aufweist
    oder ob die Trading-These strukturell gebrochen ist.
    """

    @staticmethod
    def evaluate(
        pos: Position,
        snap: MarketSnapshot,
        news_sentiment: float = 0.0,
        dark_pool_bullish: bool = False,
    ) -> RecoveryEvaluationResult:
        reasons: List[str] = []

        # -------------------------------------------------------------
        # 1. ZUERST: Harte Notfallgrenzen (Katastrophenschutz)
        # -------------------------------------------------------------
        hard_emergency = False
        emergency_reason = ""

        # A) Hard Stop Loss Floor (wenn Notfallgrenze unterschritten wird)
        if pos.side == "LONG":
            hard_floor = pos.hard_stop_loss_price if pos.hard_stop_loss_price > 0 else pos.avg_entry_price * 0.955
            if snap.current_price <= hard_floor:
                hard_emergency = True
                emergency_reason = f"Harter Notfall-Stop unterschritten (${snap.current_price:,.2f} <= ${hard_floor:,.2f})"
        else:  # SHORT
            hard_ceiling = pos.hard_stop_loss_price if pos.hard_stop_loss_price > 0 else pos.avg_entry_price * 1.045
            if snap.current_price >= hard_ceiling:
                hard_emergency = True
                emergency_reason = f"Harte Notfall-Obergrenze überschritten (${snap.current_price:,.2f} >= ${hard_ceiling:,.2f})"

        # B) Fundamentaler News-Schock / Flash Crash
        if not hard_emergency:
            if news_sentiment <= -0.45 and snap.change_24h_pct < -2.0:
                hard_emergency = True
                emergency_reason = f"Fundamentaler News-Schock (Sent {news_sentiment:+.2f}, Kurs {snap.change_24h_pct:+.1f}%)"
            elif getattr(snap, "liquidity_void_detected", False):
                hard_emergency = True
                emergency_reason = "Liquiditäts-Vakuum erkannt (Flash-Crash-Gefahr)"

        if hard_emergency:
            return RecoveryEvaluationResult(
                symbol=pos.symbol,
                can_recover=False,
                recovery_score=0.0,
                hard_emergency_exit=True,
                reasons=[emergency_reason],
                summary=f"NOTFALL-REISSLEINE: {emergency_reason}",
            )

        # -------------------------------------------------------------
        # 2. Erholungspotenzial-Kriterien (Score-Akkumulation)
        # -------------------------------------------------------------
        score = 0.0

        if pos.side == "LONG":
            # A) Oszillator-Erschöpfung & Mean Reversion (bis 1.5 Pt)
            if snap.rsi_14 <= 30.0:
                score += 1.5
                reasons.append(f"RSI extrem überverkauft ({snap.rsi_14:.1f} <= 30) -> Starker Rebound wahrscheinlich")
            elif snap.rsi_14 <= 36.0:
                score += 0.8
                reasons.append(f"RSI im Bodenbildungs-Bereich ({snap.rsi_14:.1f})")

            lower_bb = getattr(snap, "bollinger_lower", 0.0)
            if lower_bb > 0 and snap.current_price <= lower_bb * 1.008:
                score += 0.7
                reasons.append("Kurs am unteren Bollinger-Band (Statistische Volatilitäts-Untergrenze)")

            # B) Institutionelle Unterstützungszonen (VWAP & SMA) (bis 1.0 Pt)
            vwap_level = getattr(snap, "vwap_val", getattr(snap, "intraday_vwap", 0.0)) or 0.0
            if vwap_level > 0 and abs(snap.current_price - vwap_level) / vwap_level <= 0.015:
                score += 0.8
                reasons.append(f"Kurs testet institutionellen VWAP-Boden (${vwap_level:,.2f})")
            elif snap.sma_50 > 0 and snap.current_price >= snap.sma_50 * 0.990 and snap.current_price <= snap.sma_50 * 1.015:
                score += 0.7
                reasons.append(f"Institutionelle 50-Tage-Linie SMA50 (${snap.sma_50:,.2f}) intakt")
            elif snap.sma_200 > 0 and snap.current_price >= snap.sma_200 * 0.990 and snap.current_price <= snap.sma_200 * 1.015:
                score += 0.7
                reasons.append(f"Strategischer 200-Tage-Support SMA200 (${snap.sma_200:,.2f}) intakt")

            # C) Quantitativer Faktor-Rückenwind (Qlib Alpha158 & Dark Pool) (bis 1.2 Pt)
            qlib_sc = getattr(snap, "qlib_alpha158_score", 0.0)
            if qlib_sc >= 0.10:
                score += 1.0
                reasons.append(f"Microsoft Qlib Alpha158 Matrix quantitativ positiv ({qlib_sc:+.2f})")

            dp_sent = getattr(snap, "dark_pool_sentiment", "")
            if dark_pool_bullish or dp_sent in ("BULLISH_FLOW", "ACCUMULATION"):
                score += 0.7
                reasons.append("Dark-Pool-Akkumulation der Großbanken stützt den Kurs")

            # D) Volumen-Qualität: Rücksetzer ohne Panik-Volumen (bis 0.5 Pt)
            vol_r = getattr(snap, "volume_ratio_vs_20d", getattr(snap, "volume_ratio", 1.0))
            if vol_r <= 1.25:
                score += 0.5
                reasons.append(f"Rücksetzer auf moderatem Volumen ({vol_r:.1f}x vs 20d Avg, kein Panik-Ausverkauf)")

        else:  # SHORT-Position
            if snap.rsi_14 >= 70.0:
                score += 1.5
                reasons.append(f"RSI extrem überkauft ({snap.rsi_14:.1f} >= 70) -> Rücksetzer nach unten wahrscheinlich")
            elif snap.rsi_14 >= 64.0:
                score += 0.8
                reasons.append(f"RSI im Überhitzungs-Bereich ({snap.rsi_14:.1f})")

            upper_bb = getattr(snap, "bollinger_upper", 0.0)
            if upper_bb > 0 and snap.current_price >= upper_bb * 0.992:
                score += 0.7
                reasons.append("Kurs am oberen Bollinger-Band (Widerstand)")

            qlib_sc = getattr(snap, "qlib_alpha158_score", 0.0)
            if qlib_sc <= -0.10:
                score += 1.0
                reasons.append(f"Microsoft Qlib Alpha158 Matrix quantitativ negativ ({qlib_sc:+.2f})")

        can_recover = (score >= 1.5)
        summary = "; ".join(reasons) if reasons else "Keine signifikanten Rebound-Faktoren aktiv"

        return RecoveryEvaluationResult(
            symbol=pos.symbol,
            can_recover=can_recover,
            recovery_score=round(score, 2),
            hard_emergency_exit=False,
            reasons=reasons,
            summary=summary,
        )
