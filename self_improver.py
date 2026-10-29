import datetime
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from config import AgentConfig
from models import MarketSnapshot, RiskEvaluation, TradingSignal


class ShadowPrediction(BaseModel):
    prediction_id: str
    opened_at: str
    symbol: str
    asset_class: str
    predicted_action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    was_executed: bool
    execution_note: str    # z.B. "EXECUTED BUY", "SKIPPED (Kein Bestand)", "SKIPPED (EARNINGS-SCHUTZ)"
    entry_price: float
    current_price: float
    target_price: float        # Bei BUY: +2.0x ATR, Bei SELL: -2.0x ATR
    invalidation_price: float  # Bei BUY: -2.2x ATR, Bei SELL: +2.2x ATR
    directional_return_pct: float = 0.0  # Positiv wenn Prognose richtig liegt, negativ bei Fehlanalyse
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0
    status: str = "ACTIVE"     # "ACTIVE", "CORRECT_CALL", "FEHLANALYSE", "MISSED_BREAKOUT"
    lesson_learned: str = ""
    factor_contributions: Dict[str, float] = Field(default_factory=dict)


class LearnedCalibration(BaseModel):
    updated_at: str = ""
    total_evaluated: int = 0
    correct_calls: int = 0
    false_calls: int = 0
    guardrail_saved_losses: int = 0
    # Adaptive Faktor-Multiplikatoren (sicher begrenzt zwischen 0.65 und 1.35)
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "trend": 1.00,
            "momentum": 1.00,
            "volume": 1.00,
            "fundamentals": 1.00,
            "news_sentiment": 1.00,
            "options_flow": 1.00,
            "cross_asset_lead_lag": 1.00,
        }
    )
    asset_accuracy: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    recent_lessons: List[str] = Field(default_factory=list)

    @property
    def accuracy_pct(self) -> float:
        if self.total_evaluated <= 0:
            return 0.0
        return round((self.correct_calls / self.total_evaluated) * 100.0, 1)


class SelfLearningShadowTracker:
    """
    Verfolgt JEDE Prognose (BUY, SELL, HOLD) über alle Watchlist-Assets im Hintergrund ("Shadow Portfolio") –
    selbst wenn die Aktie/der Coin gar nicht im Portfolio liegt oder von der Risk-Engine blockiert wurde.
    Erkennt Fehlanalysen automatisch, diagnostiziert den schuldigen Faktor und kalibriert die Faktor-Gewichte nach.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        data_dir = os.path.dirname(self.config.portfolio_state_file)
        os.makedirs(data_dir, exist_ok=True)
        self.shadow_file = os.path.join(data_dir, "shadow_predictions.json")
        self.calibration_file = os.path.join(data_dir, "learned_weights.json")

        self.active_predictions: Dict[str, ShadowPrediction] = {}
        self.closed_predictions: List[ShadowPrediction] = []
        self.calibration: LearnedCalibration = LearnedCalibration()
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, "r", encoding="utf-8") as f:
                    self.calibration = LearnedCalibration(**json.load(f))
            except Exception:
                self.calibration = LearnedCalibration()

        if os.path.exists(self.shadow_file):
            try:
                with open(self.shadow_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.active_predictions = {
                    k: ShadowPrediction(**v) for k, v in raw.get("active", {}).items()
                }
                self.closed_predictions = [
                    ShadowPrediction(**v) for v in raw.get("closed", [])[-50:]
                ]
            except Exception:
                pass

    def _save(self) -> None:
        self.calibration.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]
        with open(self.calibration_file, "w", encoding="utf-8") as f:
            json.dump(self.calibration.model_dump(), f, indent=2, ensure_ascii=False)

        shadow_data = {
            "updated_at": self.calibration.updated_at,
            "active": {k: v.model_dump() for k, v in self.active_predictions.items()},
            "closed": [v.model_dump() for v in self.closed_predictions[-50:]],
        }
        with open(self.shadow_file, "w", encoding="utf-8") as f:
            json.dump(shadow_data, f, indent=2, ensure_ascii=False)

    def prune_inactive_symbols(self, active_symbols: set, open_position_symbols: set) -> None:
        """
        Entfernt langweilige / inaktive Aktien aus den aktiven Shadow-Prognosen, sobald sie nicht mehr
        zu den spannenden 'Stocks in Play' gehören (außer sie liegen im echten Portfolio).
        """
        keep = set(active_symbols) | set(open_position_symbols)
        to_drop = [s for s in self.active_predictions.keys() if s not in keep]
        if to_drop:
            for s in to_drop:
                self.active_predictions.pop(s, None)
            self._save()

    def evaluate_existing_predictions(self, snapshots: Dict[str, MarketSnapshot]) -> List[str]:
        """
        Prüft alle laufenden Shadow-Prognosen gegen die neuen Echtzeit-Kurse.
        Erkennt Treffer, Fehlanalysen und verpasste Ausbrüche und passt die Lern-Gewichte an.
        """
        learning_events: List[str] = []
        to_archive: List[str] = []

        for sym, pred in self.active_predictions.items():
            snap = snapshots.get(sym)
            if not snap:
                continue

            price = snap.current_price
            if np.isnan(price) or price <= 0:
                continue
            if np.isnan(pred.entry_price) or pred.entry_price <= 0:
                pred.entry_price = price
                pred.target_price = round(price * 1.04, 4)
                pred.invalidation_price = round(price * 0.96, 4)
            pred.current_price = price
            raw_move_pct = ((price - pred.entry_price) / pred.entry_price) * 100.0 if pred.entry_price > 0 else 0.0

            # Richtungs-Rendite aus Sicht der Prognose berechnen:
            # - Bei BUY ist steigender Kurs positiv (+)
            # - Bei SELL (auch ohne Bestand!) ist fallender Kurs positiv (+), steigender Kurs negativ (- Fehlanalyse!)
            if pred.predicted_action == "BUY":
                dir_ret = raw_move_pct
            elif pred.predicted_action == "SELL":
                dir_ret = -raw_move_pct
            else:
                dir_ret = -abs(raw_move_pct)  # Bei HOLD sollte der Kurs nicht stark ausbrechen

            pred.directional_return_pct = round(dir_ret, 2)
            pred.max_favorable_pct = round(max(pred.max_favorable_pct, dir_ret), 2)
            pred.max_adverse_pct = round(min(pred.max_adverse_pct, dir_ret), 2)

            # Prüfe ob Ziel (CORRECT_CALL) oder Invalidierung (FEHLANALYSE) erreicht wurde
            if pred.predicted_action == "BUY":
                if price >= pred.target_price:
                    pred.status = "CORRECT_CALL"
                    lesson = self._learn_from_outcome(pred, success=True)
                    pred.lesson_learned = lesson
                    learning_events.append(f"🧠 [SHADOW-TREFFER BUY] {sym} erreichte Ziel ${pred.target_price:,.2f} ({dir_ret:+.2f}%) -> {lesson}")
                    to_archive.append(sym)
                elif price <= pred.invalidation_price:
                    pred.status = "FEHLANALYSE"
                    lesson = self._learn_from_outcome(pred, success=False)
                    pred.lesson_learned = lesson
                    learning_events.append(f"🔍 [FEHLANALYSE ERKANNT BUY] {sym} fiel auf ${price:,.2f} ({dir_ret:+.2f}%) -> {lesson}")
                    to_archive.append(sym)

            elif pred.predicted_action == "SELL":
                # Bei SELL liegt das Ziel UNTER dem Einstiegskurs (Kurs fällt wie prognostiziert)
                if price <= pred.target_price:
                    pred.status = "CORRECT_CALL"
                    lesson = self._learn_from_outcome(pred, success=True)
                    pred.lesson_learned = lesson
                    learning_events.append(f"🧠 [SHADOW-TREFFER SELL] {sym} fiel wie prognostiziert auf ${price:,.2f} (Prognose-Gewinn: {dir_ret:+.2f}%) -> {lesson}")
                    to_archive.append(sym)
                elif price >= pred.invalidation_price:
                    pred.status = "FEHLANALYSE"
                    lesson = self._learn_from_outcome(pred, success=False)
                    pred.lesson_learned = lesson
                    learning_events.append(f"🔍 [FEHLANALYSE ERKANNT SELL] {sym} stieg trotz SELL-Prognose auf ${price:,.2f} ({dir_ret:+.2f}%) -> {lesson}")
                    to_archive.append(sym)

            elif pred.predicted_action == "HOLD":
                # Wenn der Agent HOLD sagt, das Asset aber um > 2.2x ATR ausbricht -> Verpasste Chance analysieren!
                atr_pct = (snap.atr_14 / pred.entry_price) * 100.0 if pred.entry_price > 0 else 2.0
                if abs(raw_move_pct) >= max(3.5, 2.2 * atr_pct):
                    pred.status = "MISSED_BREAKOUT"
                    direction_str = "nach oben (+BUY verpasst)" if raw_move_pct > 0 else "nach unten (-SELL verpasst)"
                    pred.lesson_learned = f"Ausbruch {direction_str} ({raw_move_pct:+.2f}%) während HOLD -> Momentum-Sensitivität leicht erhöht."
                    self.calibration.weights["momentum"] = round(
                        float(np.clip(self.calibration.weights.get("momentum", 1.0) + 0.02, 0.65, 1.35)), 3
                    )
                    learning_events.append(f"🔍 [HOLD-AUSBRUCH ERKANNT] {sym}: {pred.lesson_learned}")
                    to_archive.append(sym)

        for sym in to_archive:
            closed = self.active_predictions.pop(sym)
            self.closed_predictions.append(closed)

        self._save()
        return learning_events

    def _learn_from_outcome(self, pred: ShadowPrediction, success: bool) -> str:
        """
        Analysiert bei einem Treffer oder einer Fehlanalyse genau, welche Faktoren (Trend, Momentum,
        Volumen, Fundamentals, News-Sentiment) für das Signal verantwortlich waren, und passt deren Gewicht an.
        """
        self.calibration.total_evaluated += 1
        sym_stats = self.calibration.asset_accuracy.setdefault(pred.symbol, {"correct": 0, "false": 0})

        if success:
            self.calibration.correct_calls += 1
            sym_stats["correct"] += 1
        else:
            self.calibration.false_calls += 1
            sym_stats["false"] += 1

        direction_sign = 1.0 if pred.predicted_action == "BUY" else -1.0
        culprit_factors: List[str] = []
        helping_factors: List[str] = []

        for factor, raw_contrib in pred.factor_contributions.items():
            if abs(raw_contrib) < 0.01:
                continue
            # Hat dieser Faktor in die prognostizierte Richtung gedrückt?
            pushed_for_prediction = (raw_contrib * direction_sign) > 0
            current_w = self.calibration.weights.get(factor, 1.0)

            if success and pushed_for_prediction:
                # Faktor lag richtig -> leicht verstärken (+0.025)
                self.calibration.weights[factor] = round(float(np.clip(current_w + 0.025, 0.65, 1.35)), 3)
                helping_factors.append(factor)
            elif not success and pushed_for_prediction:
                # Faktor hat die Fehlanalyse verursacht -> Gewicht dämpfen (-0.035)
                self.calibration.weights[factor] = round(float(np.clip(current_w - 0.035, 0.65, 1.35)), 3)
                culprit_factors.append(factor)
            elif not success and not pushed_for_prediction:
                # Faktor hat vor dem Fehler gewarnt, wurde aber überstimmt -> Gewicht stärken (+0.025)
                self.calibration.weights[factor] = round(float(np.clip(current_w + 0.025, 0.65, 1.35)), 3)

        if success:
            lesson = (
                f"Prognose ({pred.predicted_action} {pred.symbol}) bestätigt. "
                f"Treiber {helping_factors} gestärkt."
            )
        else:
            was_guardrail_block = (
                not pred.was_executed
                and "SKIPPED" in pred.execution_note
                and "HOLD" not in pred.execution_note
                and pred.predicted_action == "BUY"
            )
            if was_guardrail_block:
                self.calibration.guardrail_saved_losses += 1
            guard_note = (
                f" (Risk-Guardrail '{pred.execution_note[:32]}' hat Verlust verhindert!)"
                if was_guardrail_block
                else ""
            )
            lesson = (
                f"Fehlanalyse bei {pred.predicted_action} {pred.symbol} ({pred.directional_return_pct:+.2f}%): "
                f"Irreführende Faktoren {culprit_factors} abgewertet{guard_note}."
            )

        self.calibration.recent_lessons.append(
            f"[{datetime.date.today().isoformat()}] {lesson}"
        )
        self.calibration.recent_lessons = self.calibration.recent_lessons[-12:]
        return lesson

    def record_prediction(
        self,
        snap: MarketSnapshot,
        signal: TradingSignal,
        factor_contributions: Dict[str, float],
        exec_result: str,
        is_in_portfolio: bool = False,
    ) -> None:
        """
        Legt für jedes Asset eine überprüfbare Shadow-Prognose an (oder aktualisiert sie bei Richtungswechsel),
        egal ob das Asset im Portfolio gehalten wird oder nicht.
        """
        existing = self.active_predictions.get(snap.symbol)
        # Behalte die ursprüngliche Kauf-Prognose (inkl. der Einstiegs-Faktoren für die Fehler-Diagnose!),
        # solange die Position noch offen im Portfolio gehalten wird und nicht auf SELL gedreht hat.
        if existing and (existing.predicted_action == signal.action or (is_in_portfolio and existing.predicted_action == "BUY" and signal.action == "HOLD")):
            if exec_result.startswith("EXECUTED") or is_in_portfolio:
                existing.was_executed = True
                if exec_result.startswith("EXECUTED"):
                    existing.execution_note = exec_result[:60]
            self._save()
            return

        # Wenn sich die Signal-Richtung geändert hat (z.B. von HOLD auf BUY oder von SELL auf BUY),
        # werte die bisherige Prognose sofort als Zwischenergebnis am Wendepunkt aus!
        if existing and existing.predicted_action in ("BUY", "SELL") and existing.predicted_action != signal.action:
            if abs(existing.directional_return_pct) >= 0.8:
                is_win = existing.directional_return_pct > 0
                existing.status = "CORRECT_CALL" if is_win else "FEHLANALYSE"
                existing.lesson_learned = self._learn_from_outcome(existing, success=is_win)
                self.closed_predictions.append(existing)

        atr = max(snap.atr_14, snap.current_price * 0.015)
        if signal.action == "BUY":
            target_p = round(snap.current_price + 2.0 * atr, 4)
            inval_p = round(max(0.0001, snap.current_price - 2.2 * atr), 4)
        elif signal.action == "SELL":
            # Hypothetischer Short / Abwärts-Check: Gewinnziel liegt -2.0x ATR tiefer, Invalidierung +2.2x ATR höher
            target_p = round(max(0.0001, snap.current_price - 2.0 * atr), 4)
            inval_p = round(snap.current_price + 2.2 * atr, 4)
        else:
            target_p = round(snap.current_price + 2.2 * atr, 4)
            inval_p = round(max(0.0001, snap.current_price - 2.2 * atr), 4)

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]
        self.active_predictions[snap.symbol] = ShadowPrediction(
            prediction_id=f"{snap.symbol}_{now_iso}",
            opened_at=now_iso,
            symbol=snap.symbol,
            asset_class=snap.asset_class,
            predicted_action=signal.action,
            confidence=signal.confidence,
            was_executed=exec_result.startswith("EXECUTED") or is_in_portfolio,
            execution_note=exec_result[:65],
            entry_price=snap.current_price,
            current_price=snap.current_price,
            target_price=target_p,
            invalidation_price=inval_p,
            directional_return_pct=0.0,
            factor_contributions=factor_contributions,
        )
        self._save()

    def bootstrap_historical_calibration(self, snapshots: Dict[str, MarketSnapshot], return_histories: Dict[str, object]) -> None:
        """
        Falls noch keine historischen Auswertungen vorliegen (erster Start), führt der Tracker
        automatisch eine Walk-Forward-Selbstprüfung auf den letzten 30 Handelstagen aller 10 Assets durch,
        um sofort erste Fehlanalysen aufzudecken und die Faktor-Gewichte vorzukalibrieren.
        """
        if self.calibration.total_evaluated > 0:
            return

        import yfinance as yf
        from data_collector import compute_wilder_rsi

        for item in self.config.watchlist:
            try:
                df = yf.Ticker(item.yf_ticker).history(period="3mo", interval="1d")
                if df is None or len(df) < 40:
                    continue
                close = df["Close"].astype(float)
                sma20 = close.rolling(20).mean()
                sma50 = close.rolling(50).mean()
                rsi = compute_wilder_rsi(close, 14)
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd_h = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

                # Prüfe 3 historische Stichtage (vor 25, 16 und 8 Tagen) und vergleiche mit dem echten Kurs 6 Tage später
                for offset in (-25, -16, -8):
                    if abs(offset) >= len(close) - 20:
                        continue
                    p_entry = float(close.iloc[offset])
                    p_exit = float(close.iloc[offset + 6])
                    fwd_ret_pct = ((p_exit - p_entry) / p_entry) * 100.0

                    f_trend = 0.32 if p_entry > float(sma20.iloc[offset]) > float(sma50.iloc[offset]) else (-0.35 if p_entry < float(sma20.iloc[offset]) < float(sma50.iloc[offset]) else 0.0)
                    r_val = float(rsi.iloc[offset])
                    m_val = float(macd_h.iloc[offset])
                    f_mom = 0.24 if (46 <= r_val <= 67 and m_val > 0) else (-0.25 if (r_val > 75 or (m_val < 0 and r_val < 45)) else 0.0)

                    tot = f_trend + f_mom
                    if abs(tot) < 0.30 or abs(fwd_ret_pct) < 0.8:
                        continue

                    pred_act = "BUY" if tot > 0 else "SELL"
                    dir_ret = fwd_ret_pct if pred_act == "BUY" else -fwd_ret_pct
                    is_success = dir_ret > 0

                    dummy_pred = ShadowPrediction(
                        prediction_id=f"HIST_{item.symbol}_{offset}",
                        opened_at=datetime.date.today().isoformat(),
                        symbol=item.symbol,
                        asset_class=item.asset_class,
                        predicted_action=pred_act,
                        confidence=0.72,
                        was_executed=False,
                        execution_note="HISTORICAL_SELF_AUDIT",
                        entry_price=round(p_entry, 2),
                        current_price=round(p_exit, 2),
                        target_price=round(p_entry * (1.03 if pred_act == "BUY" else 0.97), 2),
                        invalidation_price=round(p_entry * (0.97 if pred_act == "BUY" else 1.03), 2),
                        directional_return_pct=round(dir_ret, 2),
                        status="CORRECT_CALL" if is_success else "FEHLANALYSE",
                        factor_contributions={"trend": f_trend, "momentum": f_mom},
                    )
                    dummy_pred.lesson_learned = self._learn_from_outcome(dummy_pred, success=is_success)
                    self.closed_predictions.append(dummy_pred)
            except Exception:
                pass

        self._save()
