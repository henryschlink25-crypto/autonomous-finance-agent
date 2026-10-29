import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from config import AgentConfig
from models import MarketSnapshot, TradingSignal


class HybridAIAnalyst:
    """
    Kombiniert quantitative Multi-Faktor-Signale (Trend, Momentum, Fundamentaldaten, News-Lexikon)
    mit adaptiven, selbstlernenden Faktor-Gewichten (aus dem Shadow-Tracker) und optionaler
    Google Gemini LLM-Analyse inkl. Fehler-Gedächtnis (Self-Reflection Memory).
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.last_factor_contributions: Dict[str, Dict[str, float]] = {}
        self.gemini_client = None
        if self.config.gemini_api_key and self.config.gemini_api_key != "your_gemini_api_key_here":
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.config.gemini_api_key)
            except Exception as e:
                print(f"[Info] Gemini SDK Initialisierung übersprungen ({e}). Nutze Quant-Hybrid-Engine.")

    def _compute_quant_baseline(
        self,
        snap: MarketSnapshot,
        learned_weights: Optional[Dict[str, float]] = None,
    ) -> TradingSignal:
        """Deterministisches Multi-Faktor-Scoring skaliert mit den selbstlernenden Faktor-Gewichten."""
        w = learned_weights or {
            "trend": 1.0,
            "momentum": 1.0,
            "volume": 1.0,
            "fundamentals": 1.0,
            "news_sentiment": 1.0,
            "options_flow": 1.0,
            "cross_asset_lead_lag": 1.0,
        }

        factors: Dict[str, float] = {
            "trend": 0.0,
            "momentum": 0.0,
            "volume": 0.0,
            "fundamentals": 0.0,
            "news_sentiment": 0.0,
            "options_flow": 0.0,
            "cross_asset_lead_lag": 0.0,
            "intraday_vwap_rs": 0.0,
        }
        reasons = []

        # 1. Trend- & Regime-Faktor (SMA 20 vs SMA 50 + steigende SMA50)
        if snap.current_price > snap.sma_20 > snap.sma_50 and snap.sma_50_slope >= 0:
            factors["trend"] = round(0.32 * w.get("trend", 1.0), 3)
            reasons.append("Aufwärtstrend (Kurs > SMA20 > SMA50↑)")
        elif snap.current_price > snap.sma_20 > snap.sma_50:
            factors["trend"] = round(0.22 * w.get("trend", 1.0), 3)
            reasons.append("Positiver Trend (Kurs > SMA20 > SMA50)")
        elif snap.current_price < snap.sma_20 < snap.sma_50:
            factors["trend"] = round(-0.35 * w.get("trend", 1.0), 3)
            reasons.append("Abwärtstrend (Kurs < SMA20 < SMA50)")

        # 2. Momentum & Mean-Reversion (Wilder RSI 14 + MACD Histogramm)
        if 46 <= snap.rsi_14 <= 67 and snap.macd_hist > 0:
            factors["momentum"] = round(0.24 * w.get("momentum", 1.0), 3)
            reasons.append(f"Positives Momentum (RSI {snap.rsi_14:.1f}, MACD+)")
        elif snap.rsi_14 < 32 and snap.current_price >= snap.sma_50 * 0.96:
            factors["momentum"] = round(0.28 * w.get("momentum", 1.0), 3)
            reasons.append(f"Überverkauft im Trend (RSI {snap.rsi_14:.1f})")
        elif snap.rsi_14 > 75:
            factors["momentum"] = round(-0.35 * w.get("momentum", 1.0), 3)
            reasons.append(f"Überkauft (RSI {snap.rsi_14:.1f})")
        elif snap.macd_hist < 0 and snap.rsi_14 < 45:
            factors["momentum"] = round(-0.20 * w.get("momentum", 1.0), 3)
            reasons.append("Negatives MACD-Momentum")

        # 3. Volumen-Bestätigung
        if snap.volume_ratio_vs_20d > 1.20 and snap.change_24h_pct > 0:
            factors["volume"] = round(0.14 * w.get("volume", 1.0), 3)
            reasons.append(f"Volumen-Schub ({snap.volume_ratio_vs_20d:.2f}x)")

        # 4. Fundamental-Qualitätsfaktor (für Einzelaktien)
        if snap.revenue_growth_pct is not None or snap.profit_margin_pct is not None:
            rg = snap.revenue_growth_pct or 0.0
            pm = snap.profit_margin_pct or 0.0
            fpe = snap.forward_pe or 25.0
            if rg >= 12.0 and pm >= 15.0:
                factors["fundamentals"] = round(0.16 * w.get("fundamentals", 1.0), 3)
                reasons.append(f"Starke Fundamentals (Umsatz +{rg:.0f}%, Marge {pm:.0f}%, KGV {fpe:.1f})")
            elif rg < 0.0 and pm < 5.0:
                factors["fundamentals"] = round(-0.18 * w.get("fundamentals", 1.0), 3)
                reasons.append(f"Schwache Fundamentals (Umsatz {rg:.0f}%)")

        # 5. News-Sentiment-Faktor & Breaking-News-Katalysator
        avg_sentiment = 0.0
        best_catalyst_headline = snap.news[0].title if snap.news else "Keine aktuellen Schlagzeilen"
        if snap.news:
            # Wähle die Schlagzeile mit dem stärksten Sentiment-Ausschlag als Haupt-Katalysator für die Anzeige
            strongest_news = max(snap.news, key=lambda n: abs(n.sentiment_hint))
            if abs(strongest_news.sentiment_hint) > 0:
                best_catalyst_headline = f"[{strongest_news.sentiment_hint:+.2f}] {strongest_news.title}"

            # Gewichte echte Breaking-News mit Sentiment stärker als neutrale Meldungen
            non_zero = [n.sentiment_hint for n in snap.news if abs(n.sentiment_hint) > 0.05]
            avg_sentiment = float(np.mean(non_zero)) if non_zero else 0.0
            if avg_sentiment >= 0.15:
                factors["news_sentiment"] = round(0.24 * w.get("news_sentiment", 1.0), 3)
                reasons.append(f"News-Katalysator positiv ({avg_sentiment:+.2f})")
            elif avg_sentiment <= -0.15:
                factors["news_sentiment"] = round(-0.26 * w.get("news_sentiment", 1.0), 3)
                reasons.append(f"News-Katalysator negativ ({avg_sentiment:+.2f})")

        # 6. Options-Flow, Smart-Money Put/Call-Radar & Top-Optionsschein-Picker
        if snap.options_put_call_ratio is not None:
            pcr = snap.options_put_call_ratio
            iv_str = f", IV {snap.options_implied_vol_pct:.0f}%" if snap.options_implied_vol_pct else ""
            opt_contract_note = f" ➔ [{snap.best_option_contract}]" if snap.best_option_contract else ""
            whale_boost = 0.06 if snap.unusual_options_vol_oi >= 1.8 else 0.0
            if snap.options_flow_sentiment == "BULLISH_CALL_FLOW":
                factors["options_flow"] = round((0.18 + whale_boost) * w.get("options_flow", 1.0), 3)
                reasons.append(f"Options-Call-Flow bullisch (P/C {pcr:.2f}{iv_str}){opt_contract_note}")
            elif snap.options_flow_sentiment == "BEARISH_PUT_FLOW":
                factors["options_flow"] = round(-(0.20 + whale_boost) * w.get("options_flow", 1.0), 3)
                reasons.append(f"Options-Put-Absicherung bärisch (P/C {pcr:.2f}{iv_str}){opt_contract_note}")
            elif snap.unusual_options_vol_oi >= 2.0 and snap.best_option_contract:
                reasons.append(f"Unusual Options Activity{opt_contract_note}")

        # 7. Cross-Asset Lead-Lag & Inverse-Rotation ("Aktie X fällt -> Aktie Y steigt" / Nachzügler-Arbitrage)
        if abs(snap.cross_asset_opportunity_score) > 0.01 and snap.cross_asset_thesis:
            factors["cross_asset_lead_lag"] = round(
                snap.cross_asset_opportunity_score * w.get("cross_asset_lead_lag", 1.0), 3
            )
            reasons.append(snap.cross_asset_thesis)

        # 8. Intraday 15m-VWAP Timing & Relative Strength (RS) vs. S&P 500 (Schutz vor Käufen am Tageshoch)
        if snap.intraday_15m_trend == "FADING_FROM_HIGH":
            factors["intraday_vwap_rs"] = round(-0.18 * w.get("intraday_vwap_rs", 1.0), 3)
            reasons.append("⚠️ Intraday-Abverkauf vom Tageshoch (< 15m-EMA9)")
        elif snap.intraday_15m_trend == "BULLISH_ABOVE_VWAP" and snap.relative_strength_vs_spy >= 1.5:
            factors["intraday_vwap_rs"] = round(0.16 * w.get("intraday_vwap_rs", 1.0), 3)
            vwap_str = f" > VWAP ${snap.intraday_vwap:,.2f}" if snap.intraday_vwap else ""
            reasons.append(f"Intraday-Trend stark ({vwap_str}, RS {snap.relative_strength_vs_spy:+.1f}% vs. SPY)")
        elif snap.relative_strength_vs_spy >= 4.5 and snap.intraday_15m_trend != "BEARISH_BELOW_VWAP":
            factors["intraday_vwap_rs"] = round(0.12 * w.get("intraday_vwap_rs", 1.0), 3)
            reasons.append(f"Relative-Strength-Leader (RS {snap.relative_strength_vs_spy:+.1f}% vs. SPY)")
        elif snap.intraday_15m_trend == "BEARISH_BELOW_VWAP" and snap.relative_strength_vs_spy <= -2.0:
            factors["intraday_vwap_rs"] = round(-0.16 * w.get("intraday_vwap_rs", 1.0), 3)
            reasons.append(f"Unter Intraday-VWAP & Relative Schwäche (RS {snap.relative_strength_vs_spy:+.1f}% vs. SPY)")

        self.last_factor_contributions[snap.symbol] = factors
        score = sum(factors.values())

        # Sniper-Filter gegen mittelmäßige / widersprüchliche Setups ("Im Zweifel Cash halten & abwarten"):
        # 1. Mindestens 2 unabhängige positive Faktoren müssen gleichzeitig grün sein (Confluence)
        # 2. Mindestens ein echter Primär-Treiber (Trend > 0, Volumen-Ausbruch > 0, starker News-Katalysator oder Cross-Asset-Rotation)
        #    muss aktiv sein! Verhindert Fehlkäufe wie bei AVGO (wo Trend=0 und Volumen=0 flach lagen).
        # 3. Wenn ein starker Gegen-Faktor (z.B. negatives MACD-Momentum oder schlechte News <= -0.15) vorliegt,
        #    wird ein mittelmäßiges Kaufsignal sofort auf HOLD gedrosselt.
        positive_factor_count = sum(1 for v in factors.values() if v > 0.08)
        has_core_driver = (
            factors.get("trend", 0.0) > 0.0
            or factors.get("volume", 0.0) > 0.0
            or factors.get("news_sentiment", 0.0) > 0.10
            or factors.get("cross_asset_lead_lag", 0.0) > 0.12
        )
        has_strong_headwind = any(v <= -0.15 for v in factors.values())
        has_intraday_weakness = snap.change_24h_pct < -0.40 and snap.macd_hist <= 0

        if (
            score >= 0.45
            and positive_factor_count >= 2
            and has_core_driver
            and not has_strong_headwind
            and not has_intraday_weakness
        ):
            action = "BUY"
            confidence = float(np.clip(0.58 + score * 0.42, 0.60, 0.95))
        elif score <= -0.35:
            action = "SELL"
            confidence = float(np.clip(0.56 + abs(score) * 0.44, 0.56, 0.94))
        else:
            action = "HOLD"
            confidence = round(0.50 + abs(score) * 0.15, 2)
            if score >= 0.30 and (has_strong_headwind or has_intraday_weakness):
                reasons.append("⏸️ ABWARTEN (Widersprüchliches Kurzfrist-Momentum – Kapital geschützt)")
            elif score >= 0.35 and not has_core_driver:
                reasons.append("⏸️ ABWARTEN (Fehlende Trend- & Volumen-Bestätigung – Schutz vor Fehlausbruch)")

        # Dynamische 2-Stufen Fast-Momentum Stop-Loss & Take-Profit Berechnung über 14-Tage ATR
        atr = max(snap.atr_14, snap.current_price * 0.015)
        stop_distance = 1.6 * atr
        partial_tp_distance = 1.5 * atr
        tp_distance = 3.2 * atr

        stop_loss_price = round(max(0.0001, snap.current_price - stop_distance), 4)
        partial_tp_price = round(snap.current_price + partial_tp_distance, 4)
        take_profit_price = round(snap.current_price + tp_distance, 4)
        rr_ratio = round(tp_distance / stop_distance, 2) if stop_distance > 0 else 2.0

        thesis = " | ".join(reasons) if reasons else "Neutrale Marktlage ohne klaren statistischen Edge."

        return TradingSignal(
            symbol=snap.symbol,
            asset_class=snap.asset_class,
            cluster=snap.cluster,
            intrinsic_leverage=snap.intrinsic_leverage,
            action=action,
            confidence=round(confidence, 2),
            sentiment_score=round(avg_sentiment, 2),
            entry_price=snap.current_price,
            stop_loss_price=stop_loss_price,
            partial_take_profit_price=partial_tp_price,
            take_profit_price=take_profit_price,
            risk_reward_ratio=rr_ratio,
            catalyst_summary=best_catalyst_headline[:120],
            thesis=thesis,
            best_option_contract=snap.best_option_contract,
            analysis_source="QUANT_ADAPTIVE_ENGINE",
        )

    def analyze(
        self,
        snap: MarketSnapshot,
        learned_weights: Optional[Dict[str, float]] = None,
        recent_lessons: Optional[List[str]] = None,
    ) -> TradingSignal:
        baseline = self._compute_quant_baseline(snap, learned_weights=learned_weights)
        if not self.gemini_client:
            return baseline

        news_text = "\n".join(
            [f"- [{n.publisher}] {n.title} (Lexikon-Hint: {n.sentiment_hint:+.2f})" for n in snap.news]
        ) or "Keine aktuellen News verfügbar."

        lessons_text = "\n".join(recent_lessons[-5:]) if recent_lessons else "Noch keine Fehler-Historie."
        fund_info = (
            f"Forward KGV: {snap.forward_pe}, Umsatzwachstum: {snap.revenue_growth_pct}%, "
            f"Gewinnmarge: {snap.profit_margin_pct}%, Tage bis Quartalszahlen: {snap.days_to_earnings}"
        )

        prompt = f"""Du bist ein institutioneller Multi-Asset Portfolio-Manager & selbstlernender Quant-Research-Agent.
Analysiere das folgende Asset ({snap.symbol} - {snap.name}, Klasse: {snap.asset_class}) und erstelle ein präzises TradingSignal.

MARKTDATEN, TECHNIK & FUNDAMENTALS:
- Aktueller Preis: {snap.current_price} USD (24h: {snap.change_24h_pct:+.2f}%, 7d: {snap.change_7d_pct:+.2f}%)
- SMA 20: {snap.sma_20} | SMA 50: {snap.sma_50} (Slope: {snap.sma_50_slope:+.2f})
- Wilder RSI (14): {snap.rsi_14} | MACD-Histogramm: {snap.macd_hist}
- ATR (14): {snap.atr_14} | Ann. Volatilität: {snap.volatility_annualized_pct}%
- Fundamentaldaten: {fund_info}

AKTUELLE SCHLAGZEILEN:
{news_text}

ERKENNTNISSE AUS EIGENEN FRÜHEREN FEHLANALYSEN (SELF-REFLECTION MEMORY):
{lessons_text}

QUANT-BASELINE-EINSCHÄTZUNG:
- Action: {baseline.action}, Confidence: {baseline.confidence}, Thesis: {baseline.thesis}

Antworte ausschließlich im JSON-Format mit den Feldern:
action ("BUY", "SELL", "HOLD"), confidence (0.0-1.0), sentiment_score (-1.0 bis 1.0),
stop_loss_price (float, muss unter aktuellem Preis liegen), take_profit_price (float, muss über aktuellem Preis liegen),
catalyst_summary (kurzer String auf Deutsch), thesis (prägnante Begründung auf Deutsch).
"""
        try:
            response = self.gemini_client.models.generate_content(
                model=self.config.gemini_model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            data = json.loads(response.text)

            raw_action = str(data.get("action", baseline.action)).strip().upper()
            action = raw_action if raw_action in ("BUY", "SELL", "HOLD") else baseline.action
            conf = float(np.clip(float(data.get("confidence", baseline.confidence)), 0.0, 1.0))
            sent = float(np.clip(float(data.get("sentiment_score", baseline.sentiment_score)), -1.0, 1.0))

            sl = float(data.get("stop_loss_price", baseline.stop_loss_price))
            tp = float(data.get("take_profit_price", baseline.take_profit_price))

            if sl >= snap.current_price or sl <= 0 or tp <= snap.current_price:
                sl = baseline.stop_loss_price
                tp = baseline.take_profit_price

            risk = max(snap.current_price - sl, snap.current_price * 0.005)
            reward = max(tp - snap.current_price, 0.0)
            rr = round(reward / risk, 2)

            return TradingSignal(
                symbol=snap.symbol,
                asset_class=snap.asset_class,
                cluster=snap.cluster,
                intrinsic_leverage=snap.intrinsic_leverage,
                action=action,
                confidence=round(conf, 2),
                sentiment_score=round(sent, 2),
                entry_price=snap.current_price,
                stop_loss_price=round(sl, 4),
                partial_take_profit_price=baseline.partial_take_profit_price,
                take_profit_price=round(tp, 4),
                risk_reward_ratio=rr,
                catalyst_summary=str(data.get("catalyst_summary", baseline.catalyst_summary))[:140],
                thesis=str(data.get("thesis", baseline.thesis)),
                best_option_contract=snap.best_option_contract,
                analysis_source=f"GEMINI ({self.config.gemini_model})",
            )
        except Exception as e:
            print(f"[Info] LLM-Aufruf für {snap.symbol} fehlgeschlagen ({e}) -> Nutze Quant-Baseline.")
            return baseline
