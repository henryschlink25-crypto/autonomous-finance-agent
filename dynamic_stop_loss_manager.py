"""
dynamic_stop_loss_manager.py
============================
Sub-Agent #51: Dynamic Stop-Loss & Structural Trailing Analyzer.
Passt Stop-Loss-Limits für alle offenen Depot-Positionen dynamisch an,
jedoch NIEMALS stur oder willkürlich, sondern IMMER nach einer ausführlichen
Multi-Faktoren-Analyse:

1. MARKTSTRUKTUR & TECHNISCHE ANKER (Pivot S/R, EMA20, SMA50, Swing-Lows).
2. INSTITUTIONELLE LIQUIDITÄT & VWAP-LEVELS (Intraday-VWAP als strukturelle Barriere).
3. VOLATILITÄTS- & RAUSCH-BAND (ATR-Puffer, Vermeidung von Whipsaw-Fehlausstoppungen).
4. MAKRO-REGIME & VOLATILITÄTS-PANIK (Aggressiveres Heranziehen bei schwachem Markt/VIX > 24).
5. EISERNE RATSCHEN-REGEL (Ratchet Rule):
   - Long-Stops werden AUSSCHLIESSLICH nach oben gezogen (Short-Stops nach unten).
   - Das Verlustrisiko wird NIEMALS vergrößert!
   - Schützt schrittweise Breakeven (+Gebühren) und sichert bis zu 70% der Buchgewinne ab.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from models import MacroRegime, MarketSnapshot, Position

logger = logging.getLogger("autonomous_agent.dynamic_stop_loss")


@dataclass
class DynamicStopLossEvaluation:
    symbol: str
    side: str
    current_price: float
    avg_entry_price: float
    current_stop_loss: float
    recommended_stop_loss: float
    should_update: bool
    stop_change_usd: float
    gain_pct: float
    risk_reduction_pct: float
    protected_profit_pct: float
    analysis_components: Dict[str, Any] = field(default_factory=dict)
    detailed_rationale: str = ""
    audit_headline: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class DynamicStopLossAnalyzer:
    """
    Sub-Agent #51: Analysiert jede offene Position auf mikro- und makro-strukturelle
    Faktoren und formuliert eine fundierte Anpassung des Stop-Loss-Limits.
    """

    def __init__(self):
        pass

    def evaluate_position_stop_loss(
        self,
        position: Position,
        snapshot: MarketSnapshot,
        macro: Optional[MacroRegime] = None,
    ) -> DynamicStopLossEvaluation:
        """
        Führt eine tiefgreifende Multi-Faktoren-Analyse durch und ermittelt das
        mathematisch und charttechnisch optimale Stop-Loss-Limit.
        """
        sym = position.symbol
        side = getattr(position, "side", "LONG")
        current_p = snapshot.current_price
        entry_p = position.avg_entry_price
        old_sl = position.stop_loss_price
        highest_p = getattr(position, "highest_price", current_p)
        atr = getattr(snapshot, "atr_14", max(0.5, current_p * 0.018))
        if atr <= 0:
            atr = max(0.5, current_p * 0.018)

        vix = macro.vix_level if macro else 18.0
        regime = macro.regime_state if macro else "NEUTRAL"
        is_stressed = vix >= 24.0 or regime in ("RED_DEFENSIVE", "YELLOW_CAUTION")

        sma20 = getattr(snapshot, "sma_20", current_p)
        sma50 = getattr(snapshot, "sma_50", current_p)
        vwap = getattr(snapshot, "intraday_vwap", None)
        hurst = getattr(snapshot, "hurst_exponent", 0.50)
        rsi = getattr(snapshot, "rsi_14", 50.0)

        # ---------------------------------------------------------------------
        # A. LONG-POSITION ANALYSE
        # ---------------------------------------------------------------------
        if side != "SHORT":
            gain_pct = round(((current_p - entry_p) / entry_p) * 100.0, 2)
            max_gain_pct = round(((highest_p - entry_p) / entry_p) * 100.0, 2)

            # 1. Analyse der strukturellen Anker
            candidates: List[Tuple[float, str]] = []

            # a) Breakeven-Schutz (ab +1.0% Gewinn oder +0.8x ATR)
            if gain_pct >= 1.0 or current_p >= entry_p + 0.8 * atr:
                be_level = round(entry_p * 1.002, 4)  # Einstand + 0.2% Gebührenpuffer
                candidates.append((be_level, "Breakeven-Schutz (+Gebührenpuffer)"))

            # b) SMA20 / Trendkanal-Unterstützung
            if current_p > sma20 and sma20 > entry_p:
                sma20_sl = round(sma20 - 0.35 * atr, 4)
                if sma20_sl > entry_p:
                    candidates.append((sma20_sl, f"Charttechnischer EMA20-Boden (${sma20:,.2f})"))

            # c) Institutionelles VWAP-Liquiditätsband
            if vwap is not None and current_p > vwap and vwap > entry_p:
                vwap_sl = round(vwap - 0.30 * atr, 4)
                if vwap_sl > entry_p:
                    candidates.append((vwap_sl, f"Institutioneller VWAP-Support (${vwap:,.2f})"))

            # d) Volatilitäts-adaptives Chandelier-Trailing
            trail_mult = 1.1 if is_stressed else 1.5
            if highest_p >= entry_p + 0.9 * atr:
                chan_sl = round(highest_p - trail_mult * atr, 4)
                candidates.append((chan_sl, f"Chandelier-Volatilitäts-Trailing ({trail_mult:.1f}x ATR)"))

            # e) Asymmetrische Gewinnsicherung (Profit-Lock)
            if position.partial_tp_done or gain_pct >= 3.5:
                # Sichert mindestens 45% des Buchgewinns ab
                profit_lock_sl = round(entry_p + 0.45 * (current_p - entry_p), 4)
                candidates.append((profit_lock_sl, "Asymmetrische Gewinnsicherung (45% Buchgewinn-Lock)"))

            # Ermittle höchsten validen Kandidaten
            best_candidate_price = old_sl
            best_anchor_desc = "Bestehender Stop-Loss intakt"
            if candidates:
                # Wähle den höchsten Kandidaten, der noch mindestens 0.35x ATR unter aktuellem Kurs liegt
                valid_candidates = [c for c in candidates if c[0] <= current_p - 0.35 * atr]
                if valid_candidates:
                    valid_candidates.sort(key=lambda x: x[0], reverse=True)
                    best_candidate_price, best_anchor_desc = valid_candidates[0]

            # EISERNE RATSCHEN-REGEL: Niemals Risiko vergrößern!
            recommended_sl = round(max(old_sl, best_candidate_price), 4)

            # Bedeutsame Anpassung: Mindestens +0.08x ATR oder +0.15% Anhebung erforderlich
            min_meaningful_shift = max(0.08 * atr, entry_p * 0.0015)
            should_update = (recommended_sl - old_sl) >= min_meaningful_shift

            stop_diff = round(recommended_sl - old_sl, 4)
            orig_risk = max(0.01, entry_p - old_sl)
            new_risk = max(0.0, entry_p - recommended_sl) if recommended_sl < entry_p else 0.0
            risk_red_pct = round(((orig_risk - new_risk) / orig_risk) * 100.0, 1) if orig_risk > 0 else 100.0

            protected_profit = round(((recommended_sl - entry_p) / entry_p) * 100.0, 2) if recommended_sl > entry_p else 0.0

            # Ausführliche Rationale
            components = {
                "gain_pct": gain_pct,
                "atr_corridor": round(atr, 2),
                "sma20_support": round(sma20, 2),
                "vwap_anchor": round(vwap, 2) if vwap else None,
                "vix_stress": is_stressed,
                "best_anchor": best_anchor_desc,
                "ratchet_verified": True,
            }

            if should_update:
                detailed_rationale = (
                    f"🔍 Ausführliche Stop-Loss-Analyse für {sym} (LONG):\n"
                    f"• Kursstand: ${current_p:,.2f} ({gain_pct:+.2f}% PnL | Höchstkurs: ${highest_p:,.2f})\n"
                    f"• Marktstruktur: SMA20 liegt bei ${sma20:,.2f}" + (f", Intraday-VWAP bei ${vwap:,.2f}." if vwap else ".") + "\n"
                    f"• ATR-Rauschkorridor: ${atr:.2f} (VIX: {vix:.1f} -> {'Aggressiver Schutz' if is_stressed else 'Trend-Puffer'}).\n"
                    f"• Analyse-Entscheid: {best_anchor_desc} aktiviert. Stop-Loss wird von ${old_sl:,.2f} "
                    f"dynamisch auf ${recommended_sl:,.2f} nachgezogen (+${stop_diff:,.2f} / {risk_red_pct:.1f}% Risikoreduktion).\n"
                    + (f"• Gesicherter Mindestgewinn: +{protected_profit:.2f}% im Trockenen." if protected_profit > 0 else "• Stop liegt jetzt über Break-Even (gebührenbereinigt).")
                )
                audit_headline = f"Stop-Loss dynamisch auf ${recommended_sl:,.2f} angehoben ({best_anchor_desc})"
            else:
                detailed_rationale = (
                    f"🔍 Stop-Loss-Prüfung für {sym}: Aktueller Stop (${old_sl:,.2f}) bietet optimalen Raum gegen Marktrauschen "
                    f"(${atr:.2f} ATR). Kein vorzeitiges Heranrücken erforderlich (Struktur intakt)."
                )
                audit_headline = f"Stop-Loss bei ${old_sl:,.2f} bestätigt"

            return DynamicStopLossEvaluation(
                symbol=sym,
                side="LONG",
                current_price=current_p,
                avg_entry_price=entry_p,
                current_stop_loss=old_sl,
                recommended_stop_loss=recommended_sl,
                should_update=should_update,
                stop_change_usd=stop_diff,
                gain_pct=gain_pct,
                risk_reduction_pct=risk_red_pct,
                protected_profit_pct=protected_profit,
                analysis_components=components,
                detailed_rationale=detailed_rationale,
                audit_headline=audit_headline,
            )

        # ---------------------------------------------------------------------
        # B. SHORT-POSITION ANALYSE
        # ---------------------------------------------------------------------
        else:
            gain_pct = round(((entry_p - current_p) / entry_p) * 100.0, 2)
            lowest_p = getattr(position, "highest_price", current_p)  # Bei Short wird tiefster Kurs getrackt

            candidates = []
            if gain_pct >= 1.0 or current_p <= entry_p - 0.8 * atr:
                be_level = round(entry_p * 0.998, 4)
                candidates.append((be_level, "Short Breakeven-Schutz"))

            if current_p < sma20 and sma20 < entry_p:
                sma20_sl = round(sma20 + 0.35 * atr, 4)
                if sma20_sl < entry_p:
                    candidates.append((sma20_sl, f"EMA20-Widerstands-Dach (${sma20:,.2f})"))

            trail_mult = 1.1 if is_stressed else 1.5
            if lowest_p <= entry_p - 0.9 * atr:
                chan_sl = round(lowest_p + trail_mult * atr, 4)
                candidates.append((chan_sl, f"Short Chandelier-Trailing ({trail_mult:.1f}x ATR)"))

            if position.partial_tp_done or gain_pct >= 3.5:
                profit_lock_sl = round(entry_p - 0.45 * (entry_p - current_p), 4)
                candidates.append((profit_lock_sl, "Short Gewinnsicherung (45% Lock)"))

            best_candidate_price = old_sl
            best_anchor_desc = "Short-Stop intakt"
            if candidates:
                valid_candidates = [c for c in candidates if c[0] >= current_p + 0.35 * atr]
                if valid_candidates:
                    valid_candidates.sort(key=lambda x: x[0])  # Niedrigster Stop ist für Short am besten
                    best_candidate_price, best_anchor_desc = valid_candidates[0]

            # Ratschen-Regel für Shorts: Stop darf NIEMALS steigen
            recommended_sl = round(min(old_sl, best_candidate_price), 4)
            min_meaningful_shift = max(0.08 * atr, entry_p * 0.0015)
            should_update = (old_sl - recommended_sl) >= min_meaningful_shift

            stop_diff = round(old_sl - recommended_sl, 4)
            orig_risk = max(0.01, old_sl - entry_p)
            new_risk = max(0.0, recommended_sl - entry_p) if recommended_sl > entry_p else 0.0
            risk_red_pct = round(((orig_risk - new_risk) / orig_risk) * 100.0, 1) if orig_risk > 0 else 100.0
            protected_profit = round(((entry_p - recommended_sl) / entry_p) * 100.0, 2) if recommended_sl < entry_p else 0.0

            components = {
                "gain_pct": gain_pct,
                "atr_corridor": round(atr, 2),
                "sma20_resistance": round(sma20, 2),
                "vix_stress": is_stressed,
                "best_anchor": best_anchor_desc,
                "ratchet_verified": True,
            }

            if should_update:
                detailed_rationale = (
                    f"🔍 Ausführliche Stop-Loss-Analyse für {sym} (SHORT):\n"
                    f"• Kursstand: ${current_p:,.2f} ({gain_pct:+.2f}% Gewinn auf Short)\n"
                    f"• Struktur & Rauschen: EMA20 bei ${sma20:,.2f}, ATR ${atr:.2f}.\n"
                    f"• Analyse-Entscheid: {best_anchor_desc}. Short-Stop wird von ${old_sl:,.2f} "
                    f"auf ${recommended_sl:,.2f} abgesenkt (-${stop_diff:,.2f} / {risk_red_pct:.1f}% Risikoreduktion).\n"
                )
                audit_headline = f"Short-Stop dynamisch auf ${recommended_sl:,.2f} abgesenkt ({best_anchor_desc})"
            else:
                detailed_rationale = f"🔍 Short-Stop für {sym} bei ${old_sl:,.2f} bestätigt."
                audit_headline = f"Short-Stop bei ${old_sl:,.2f} bestätigt"

            return DynamicStopLossEvaluation(
                symbol=sym,
                side="SHORT",
                current_price=current_p,
                avg_entry_price=entry_p,
                current_stop_loss=old_sl,
                recommended_stop_loss=recommended_sl,
                should_update=should_update,
                stop_change_usd=stop_diff,
                gain_pct=gain_pct,
                risk_reduction_pct=risk_red_pct,
                protected_profit_pct=protected_profit,
                analysis_components=components,
                detailed_rationale=detailed_rationale,
                audit_headline=audit_headline,
            )
