"""
walk_forward_research_lab.py
============================
🧪 Walk-Forward Research & Optimization Lab (Forschungsfeld C) – Version 2.

Korrigierte Fassung nach Fehleranalyse (siehe LESSONS_LEARNED.md, Eintrag #1):
- V1 annualisierte Pro-Trade-Renditen mit sqrt(252) -> unplausible Sharpe-Werte (z. B. -214).
- V1 wählte das beste Parameterset anhand der OUT-OF-SAMPLE-Ergebnisse (Data Snooping).
- V1 nutzte ausschließlich synthetische Zufallskurse.
- V1 setzte bei negativer In-Sample-Performance einen erfundenen WFER-Standardwert (60-75 %).
- V1 schrieb in eine Datei, die der Auto-Tuner alle 5 Minuten überschreibt.

V2:
1. Echte historische Tageskurse (yfinance, auto-adjusted). Synthetische Daten nur als
   klar gekennzeichneter Fallback – ohne Kalibrierung.
2. Rollierende Walk-Forward-Analyse: 252 Tage Training (IS) -> 63 Tage Blind-Test (OOS),
   Schrittweite 63 Tage. Parameterauswahl AUSSCHLIESSLICH anhand IS.
3. Tägliche Mark-to-Market-Equity-Kurve, Sharpe aus Tagesrenditen * sqrt(252),
   Transaktionskosten pro Seite, konservative Annahme bei SL+TP am selben Tag (SL zuerst).
4. WFER = annualisierte OOS-Rendite / annualisierte IS-Rendite. Kein Fallback-Wert.
5. Plausibilitäts-Prüfungen: unplausible Kennzahlen blockieren jede Kalibrierung.
6. Benchmark-Vergleich gegen Buy & Hold im selben OOS-Zeitraum.
7. Validierte Parameter landen in 'data/walk_forward_validated_params.json',
   die der HyperparameterAutoTuner als Basis verwendet.
"""

from __future__ import annotations
import itertools
import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("autonomous_agent.research_lab")

REPORT_FILE = Path("data/walk_forward_audit_report.json")
VALIDATED_FILE = Path("data/walk_forward_validated_params.json")

DEFAULT_SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMD", "GLD", "XLE"]
COST_PER_SIDE = 0.0005          # 0.05 % Gebühren + Slippage pro Seite
TRAIN_BARS = 252
TEST_BARS = 63
STEP_BARS = 63
MIN_IS_TRADES = 3

# Plausibilitätsgrenzen (Lehre aus Fehler #1)
MAX_PLAUSIBLE_ABS_SHARPE = 6.0
MIN_TRADES_FOR_SIGNIFICANCE = 30


class ParameterCandidate(BaseModel):
    atr_stop_loss_mult: float = 2.0
    atr_take_profit_mult: float = 3.5
    momentum_threshold: float = 0.0   # Mindest-5-Tage-Momentum für Einstieg

    def key(self) -> str:
        return f"SL{self.atr_stop_loss_mult:.1f}_TP{self.atr_take_profit_mult:.1f}_M{self.momentum_threshold:.3f}"


class SimStats(BaseModel):
    sharpe: float = 0.0
    ann_return_pct: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    trades: int = 0


class RegimePerformance(BaseModel):
    regime_name: str
    folds: int
    oos_sharpe: float
    oos_return_pct: float
    win_rate_pct: float
    trades: int
    is_robust: bool


class WalkForwardOptimizationResult(BaseModel):
    timestamp: str
    data_source: str                       # "REAL_YFINANCE" | "SYNTHETIC_FALLBACK" | "PROVIDED"
    symbols: List[str] = Field(default_factory=list)
    total_folds: int = 0
    candidates_tested: int = 0
    best_parameters: ParameterCandidate
    selection_frequency_pct: float = 0.0   # Wie oft wurde das Set in den Folds gewählt
    overall_is_sharpe: float = 0.0
    overall_oos_sharpe: float = 0.0
    overall_is_ann_return_pct: float = 0.0
    overall_oos_ann_return_pct: float = 0.0
    overall_wfer_pct: float = 0.0
    oos_max_drawdown_pct: float = 0.0
    oos_total_trades: int = 0
    buy_hold_oos_ann_return_pct: float = 0.0
    beats_buy_and_hold: bool = False
    robustness_grade: str = "REJECTED"     # A_EXEMPLARY | B_ROBUST | C_MARGINAL | REJECTED
    sanity_warnings: List[str] = Field(default_factory=list)
    regime_breakdown: List[RegimePerformance] = Field(default_factory=list)
    auto_calibrated_applied: bool = False
    executive_summary: str = ""


def _stats_from_daily(daily_rets: np.ndarray, trade_pnls: List[float]) -> SimStats:
    """Kennzahlen aus TÄGLICHEN Renditen (nicht aus Pro-Trade-Renditen!)."""
    if daily_rets.size == 0:
        return SimStats()
    equity = np.cumprod(1.0 + daily_rets)
    total = float(equity[-1] - 1.0)
    years = daily_rets.size / 252.0
    ann = float((equity[-1]) ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else -1.0
    std = float(np.std(daily_rets, ddof=1)) if daily_rets.size > 1 else 0.0
    sharpe = float(np.mean(daily_rets) / std * math.sqrt(252)) if std > 1e-12 else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / peak)) if equity.size else 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return SimStats(
        sharpe=round(sharpe, 3),
        ann_return_pct=round(ann * 100.0, 2),
        total_return_pct=round(total * 100.0, 2),
        max_drawdown_pct=round(max_dd * 100.0, 2),
        win_rate_pct=round(wins / len(trade_pnls) * 100.0, 1) if trade_pnls else 0.0,
        trades=len(trade_pnls),
    )


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Nachlaufende (trailing) Indikatoren – kein Blick in die Zukunft."""
    out = pd.DataFrame(index=df.index)
    out["open"] = df["Open"].astype(float)
    out["high"] = df["High"].astype(float)
    out["low"] = df["Low"].astype(float)
    out["close"] = df["Close"].astype(float)
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    out["sma20"] = out["close"].rolling(20).mean()
    out["mom5"] = out["close"] / out["close"].shift(5) - 1.0
    return out.dropna()


def simulate_strategy(ind: pd.DataFrame, params: ParameterCandidate) -> Tuple[np.ndarray, List[float]]:
    """
    Long-only Trendfolge mit ATR-Stop/-Target, voll investiert, tägliche Mark-to-Market-Bewertung.
    Einstieg zum Schlusskurs von Tag i, Prüfung von SL/TP ab Tag i+1.
    Gibt (tägliche Renditen, Liste der Trade-PnLs) zurück.
    """
    o = ind["open"].to_numpy()
    h = ind["high"].to_numpy()
    l = ind["low"].to_numpy()
    c = ind["close"].to_numpy()
    atr = ind["atr"].to_numpy()
    sma = ind["sma20"].to_numpy()
    mom = ind["mom5"].to_numpy()
    n = len(c)

    daily = np.zeros(n)
    trades: List[float] = []
    in_pos = False
    entry = sl = tp = 0.0

    for i in range(n):
        if in_pos and i > 0:
            exit_px = None
            # Konservativ: Gap unter SL -> Ausstieg zum Open; SL wird vor TP geprüft
            if o[i] <= sl:
                exit_px = o[i]
            elif l[i] <= sl:
                exit_px = sl
            elif o[i] >= tp:
                exit_px = o[i]
            elif h[i] >= tp:
                exit_px = tp

            if exit_px is not None:
                daily[i] = (exit_px * (1 - COST_PER_SIDE)) / c[i - 1] - 1.0
                trades.append((exit_px * (1 - COST_PER_SIDE)) / (entry * (1 + COST_PER_SIDE)) - 1.0)
                in_pos = False
                continue  # kein Wiedereinstieg am selben Tag
            daily[i] = c[i] / c[i - 1] - 1.0

        if not in_pos and i < n - 1:
            if c[i] > sma[i] and mom[i] >= params.momentum_threshold and atr[i] > 0:
                in_pos = True
                entry = c[i]
                sl = entry - params.atr_stop_loss_mult * atr[i]
                tp = entry + params.atr_take_profit_mult * atr[i]
                daily[i] += -COST_PER_SIDE  # Einstiegskosten

    # Offene Position am Ende zum letzten Schlusskurs bewerten
    if in_pos:
        trades.append((c[-1] * (1 - COST_PER_SIDE)) / (entry * (1 + COST_PER_SIDE)) - 1.0)
        daily[-1] += -COST_PER_SIDE
    return daily, trades


def default_candidates() -> List[ParameterCandidate]:
    grid = itertools.product([1.5, 2.0, 2.5, 3.0], [2.5, 3.5, 4.5], [0.0, 0.02])
    return [ParameterCandidate(atr_stop_loss_mult=s, atr_take_profit_mult=t, momentum_threshold=m) for s, t, m in grid]


class WalkForwardResearchLab:
    def __init__(self, report_file: Path = REPORT_FILE, validated_file: Path = VALIDATED_FILE):
        self.report_file = Path(report_file)
        self.validated_file = Path(validated_file)
        self.report_file.parent.mkdir(parents=True, exist_ok=True)
        self.validated_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Daten
    def load_real_history(self, symbols: List[str], period: str = "6y") -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        try:
            import yfinance as yf
        except Exception as e:
            logger.warning(f"yfinance nicht verfügbar: {e}")
            return data
        for sym in symbols:
            try:
                df = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
                if df is not None and len(df) >= TRAIN_BARS + TEST_BARS + 40:
                    data[sym] = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            except Exception as e:
                logger.warning(f"Historie für {sym} nicht ladbar: {e}")
        return data

    @staticmethod
    def generate_synthetic_series(seed: int, n_bars: int = 900, drift: float = 0.0004, vol: float = 0.015) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rets = rng.normal(drift, vol, n_bars)
        close = 100.0 * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, vol * 0.2, n_bars))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol * 0.4, n_bars)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol * 0.4, n_bars)))
        idx = pd.bdate_range("2020-01-01", periods=n_bars)
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1e6}, index=idx)

    # ------------------------------------------------------- Walk-Forward-Kern
    def run_walk_forward_optimization(
        self,
        symbols: Optional[List[str]] = None,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        candidates: Optional[List[ParameterCandidate]] = None,
        data_source_label: Optional[str] = None,
    ) -> WalkForwardOptimizationResult:
        symbols = symbols or DEFAULT_SYMBOLS
        candidates = candidates or default_candidates()

        if price_data is not None:
            data = price_data
            source = data_source_label or "PROVIDED"
        else:
            data = self.load_real_history(symbols)
            source = "REAL_YFINANCE"
            if not data:
                data = {f"SYN{i}": self.generate_synthetic_series(seed=100 + i) for i in range(4)}
                source = "SYNTHETIC_FALLBACK"

        is_rets_all: List[np.ndarray] = []
        oos_rets_all: List[np.ndarray] = []
        oos_trades_all: List[float] = []
        bh_rets_all: List[np.ndarray] = []
        chosen_keys: List[str] = []
        regime_buckets: Dict[str, Dict[str, list]] = {}
        total_folds = 0

        for sym, raw in data.items():
            ind = prepare_indicators(raw)
            n = len(ind)
            start = 0
            while start + TRAIN_BARS + TEST_BARS <= n:
                is_ind = ind.iloc[start:start + TRAIN_BARS]
                oos_ind = ind.iloc[start + TRAIN_BARS:start + TRAIN_BARS + TEST_BARS]

                # 1) Auswahl NUR anhand In-Sample
                best_cand, best_is_sharpe, best_is_rets = None, -1e9, None
                for cand in candidates:
                    d, t = simulate_strategy(is_ind, cand)
                    if len(t) < MIN_IS_TRADES:
                        continue
                    st = _stats_from_daily(d, t)
                    if st.sharpe > best_is_sharpe:
                        best_cand, best_is_sharpe, best_is_rets = cand, st.sharpe, d
                if best_cand is None:
                    start += STEP_BARS
                    continue

                # 2) Blind-Test des gewählten Sets auf OOS
                d_oos, t_oos = simulate_strategy(oos_ind, best_cand)
                close_oos = oos_ind["close"].to_numpy()
                bh = np.concatenate([[0.0], close_oos[1:] / close_oos[:-1] - 1.0])

                is_rets_all.append(best_is_rets)
                oos_rets_all.append(d_oos)
                oos_trades_all.extend(t_oos)
                bh_rets_all.append(bh)
                chosen_keys.append(best_cand.key())
                total_folds += 1

                # 3) Regime des OOS-Fensters (realisiert, nur zur Auswertung)
                bh_total = float(np.prod(1 + bh) - 1)
                bh_vol = float(np.std(bh, ddof=1) * math.sqrt(252)) if len(bh) > 1 else 0.0
                if bh_vol >= 0.40:
                    regime = "HIGH_VOLATILITY"
                elif bh_total >= 0.05:
                    regime = "BULL_TREND"
                elif bh_total <= -0.05:
                    regime = "BEAR_TREND"
                else:
                    regime = "SIDEWAYS_CHOP"
                b = regime_buckets.setdefault(regime, {"rets": [], "trades": []})
                b["rets"].append(d_oos)
                b["trades"].extend(t_oos)

                start += STEP_BARS

        warnings: List[str] = []
        if total_folds == 0:
            res = WalkForwardOptimizationResult(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                data_source=source,
                symbols=list(data.keys()),
                candidates_tested=len(candidates),
                best_parameters=ParameterCandidate(),
                robustness_grade="REJECTED",
                sanity_warnings=["Keine auswertbaren Folds (zu wenig Historie oder zu wenige Trades)."],
                executive_summary="Walk-Forward nicht durchführbar – zu wenig Daten.",
            )
            self._save_report(res)
            return res

        is_stats = _stats_from_daily(np.concatenate(is_rets_all), [])
        oos_stats = _stats_from_daily(np.concatenate(oos_rets_all), oos_trades_all)
        bh_stats = _stats_from_daily(np.concatenate(bh_rets_all), [])

        # WFER ehrlich: keine Ersatzwerte
        if is_stats.ann_return_pct > 0:
            wfer = oos_stats.ann_return_pct / is_stats.ann_return_pct * 100.0
        else:
            wfer = 0.0
            warnings.append("In-Sample-Rendite <= 0 – Strategie hat schon im Training nicht funktioniert.")

        # Stabilste Parameter = am häufigsten gewählt
        key_counts = Counter(chosen_keys)
        best_key, best_count = key_counts.most_common(1)[0]
        best_params = next(c for c in candidates if c.key() == best_key)
        sel_freq = best_count / total_folds * 100.0

        # Plausibilitäts-Prüfungen (Lehre aus Fehler #1)
        for label, val in (("IS-Sharpe", is_stats.sharpe), ("OOS-Sharpe", oos_stats.sharpe)):
            if not math.isfinite(val) or abs(val) > MAX_PLAUSIBLE_ABS_SHARPE:
                warnings.append(f"UNPLAUSIBEL: {label} = {val:.2f} (|Sharpe| > {MAX_PLAUSIBLE_ABS_SHARPE}). Rechenfehler wahrscheinlich.")
        if oos_stats.trades < MIN_TRADES_FOR_SIGNIFICANCE:
            warnings.append(f"Nur {oos_stats.trades} OOS-Trades – statistisch nicht aussagekräftig (Minimum {MIN_TRADES_FOR_SIGNIFICANCE}).")
        if source != "REAL_YFINANCE":
            warnings.append(f"Datenquelle '{source}' – keine echten Marktdaten, daher keine Kalibrierung.")
        if sel_freq < 25.0:
            warnings.append(f"Parameter instabil: bestes Set nur in {sel_freq:.0f}% der Folds gewählt.")

        implausible = any(w.startswith("UNPLAUSIBEL") for w in warnings)
        if implausible or is_stats.ann_return_pct <= 0 or oos_stats.sharpe < 0:
            grade = "REJECTED"
        elif wfer >= 60 and oos_stats.sharpe >= 1.0 and oos_stats.trades >= 50:
            grade = "A_EXEMPLARY"
        elif wfer >= 50 and oos_stats.sharpe >= 0.5 and oos_stats.trades >= MIN_TRADES_FOR_SIGNIFICANCE:
            grade = "B_ROBUST"
        else:
            grade = "C_MARGINAL"

        regimes: List[RegimePerformance] = []
        for name, b in sorted(regime_buckets.items()):
            st = _stats_from_daily(np.concatenate(b["rets"]), b["trades"])
            regimes.append(RegimePerformance(
                regime_name=name, folds=len(b["rets"]), oos_sharpe=st.sharpe,
                oos_return_pct=st.ann_return_pct, win_rate_pct=st.win_rate_pct,
                trades=st.trades, is_robust=st.sharpe > 0,
            ))

        beats_bh = oos_stats.ann_return_pct > bh_stats.ann_return_pct
        applied = False
        if grade in ("A_EXEMPLARY", "B_ROBUST") and source == "REAL_YFINANCE" and not implausible:
            self._write_validated_params(best_params, oos_stats, wfer, grade)
            applied = True

        summary = (
            f"{total_folds} Walk-Forward-Folds über {len(data)} Symbole ({source}). "
            f"OOS: Sharpe {oos_stats.sharpe:.2f}, {oos_stats.ann_return_pct:+.1f}% p.a., Max-DD {oos_stats.max_drawdown_pct:.1f}%, "
            f"{oos_stats.trades} Trades. Buy & Hold im selben Zeitraum: {bh_stats.ann_return_pct:+.1f}% p.a. "
            f"({'geschlagen' if beats_bh else 'NICHT geschlagen'}). WFER {wfer:.0f}%. Note: {grade}."
        )

        res = WalkForwardOptimizationResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            data_source=source,
            symbols=list(data.keys()),
            total_folds=total_folds,
            candidates_tested=len(candidates),
            best_parameters=best_params,
            selection_frequency_pct=round(sel_freq, 1),
            overall_is_sharpe=is_stats.sharpe,
            overall_oos_sharpe=oos_stats.sharpe,
            overall_is_ann_return_pct=is_stats.ann_return_pct,
            overall_oos_ann_return_pct=oos_stats.ann_return_pct,
            overall_wfer_pct=round(wfer, 1),
            oos_max_drawdown_pct=oos_stats.max_drawdown_pct,
            oos_total_trades=oos_stats.trades,
            buy_hold_oos_ann_return_pct=bh_stats.ann_return_pct,
            beats_buy_and_hold=beats_bh,
            robustness_grade=grade,
            sanity_warnings=warnings,
            regime_breakdown=regimes,
            auto_calibrated_applied=applied,
            executive_summary=summary,
        )
        self._save_report(res)
        return res

    # ---------------------------------------------------------- Persistenz
    def _save_report(self, res: WalkForwardOptimizationResult) -> None:
        try:
            with open(self.report_file, "w", encoding="utf-8") as f:
                json.dump(res.model_dump(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Konnte WFA-Report nicht speichern: {e}")

    def _write_validated_params(self, params: ParameterCandidate, oos: SimStats, wfer: float, grade: str) -> None:
        payload = {
            "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "grade": grade,
            "data_source": "REAL_YFINANCE",
            "atr_stop_loss_mult": params.atr_stop_loss_mult,
            "atr_take_profit_mult": params.atr_take_profit_mult,
            "momentum_threshold": params.momentum_threshold,
            "oos_sharpe": oos.sharpe,
            "oos_ann_return_pct": oos.ann_return_pct,
            "wfer_pct": round(wfer, 1),
        }
        try:
            with open(self.validated_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Konnte validierte Parameter nicht schreiben: {e}")

    def load_latest_report(self) -> Optional[WalkForwardOptimizationResult]:
        if self.report_file.exists():
            try:
                with open(self.report_file, "r", encoding="utf-8") as f:
                    return WalkForwardOptimizationResult(**json.load(f))
            except Exception as e:
                logger.warning(f"Konnte WFA-Report nicht laden (evtl. altes V1-Format): {e}")
        return None

    # ------------------------------------------------------------ Telegram
    def format_telegram_summary(self, res: Optional[WalkForwardOptimizationResult] = None) -> str:
        if res is None:
            res = self.load_latest_report()
        if res is None:
            return (
                "🧪 <b>Walk-Forward Forschungs-Labor:</b>\n"
                "─────────────────────────────\n"
                "Noch kein gültiger Bericht vorhanden.\n\n"
                "<i>Sende <code>/optimize</code>, um einen Lauf mit echten Kursdaten zu starten.</i>"
            )

        badge = {
            "A_EXEMPLARY": "🏆 A (robust & stark)",
            "B_ROBUST": "✅ B (robust)",
            "C_MARGINAL": "🟡 C (grenzwertig – keine Kalibrierung)",
            "REJECTED": "❌ Abgelehnt",
        }.get(res.robustness_grade, res.robustness_grade)
        p = res.best_parameters

        lines = [
            "🧪 <b>Walk-Forward Forschungs-Labor (V2):</b>",
            "─────────────────────────────",
            f"• <b>Note:</b> <b>{badge}</b>",
            f"• <b>Datenquelle:</b> <code>{res.data_source}</code> ({len(res.symbols)} Symbole, {res.total_folds} Folds)",
            f"• <b>OOS (Blind-Test):</b> Sharpe <b>{res.overall_oos_sharpe:.2f}</b> | {res.overall_oos_ann_return_pct:+.1f}% p.a. | Max-DD {res.oos_max_drawdown_pct:.1f}%",
            f"• <b>IS (Training):</b> Sharpe {res.overall_is_sharpe:.2f} | {res.overall_is_ann_return_pct:+.1f}% p.a.",
            f"• <b>WFER:</b> {res.overall_wfer_pct:.0f}% (Ziel ≥ 50–60%)",
            f"• <b>Buy &amp; Hold OOS:</b> {res.buy_hold_oos_ann_return_pct:+.1f}% p.a. ➜ {'✅ geschlagen' if res.beats_buy_and_hold else '❌ nicht geschlagen'}",
            f"• <b>OOS-Trades:</b> {res.oos_total_trades}\n",
            "⚙️ <b>Stabilstes Parameterset:</b>",
            f"• SL {p.atr_stop_loss_mult:.1f}x ATR | TP {p.atr_take_profit_mult:.1f}x ATR | Momentum ≥ {p.momentum_threshold*100:.0f}%",
            f"• In {res.selection_frequency_pct:.0f}% der Folds gewählt",
            f"• Übernahme in Live-Parameter: <b>{'✅ ja' if res.auto_calibrated_applied else '⚪ nein'}</b>\n",
        ]
        if res.regime_breakdown:
            lines.append("📊 <b>OOS nach Marktregime:</b>")
            for r in res.regime_breakdown:
                icon = "🟢" if r.is_robust else "🔴"
                lines.append(f"{icon} {r.regime_name}: Sharpe {r.oos_sharpe:.2f} | {r.oos_return_pct:+.1f}% p.a. | {r.trades} Trades ({r.folds} Folds)")
        if res.sanity_warnings:
            lines.append("\n⚠️ <b>Plausibilitäts-Hinweise:</b>")
            for w in res.sanity_warnings:
                lines.append(f"• {w}")
        lines.append("─────────────────────────────")
        lines.append(f"💡 <i>{res.executive_summary}</i>")
        return "\n".join(lines)


_LAB_INSTANCE: Optional[WalkForwardResearchLab] = None


def get_research_lab() -> WalkForwardResearchLab:
    global _LAB_INSTANCE
    if _LAB_INSTANCE is None:
        _LAB_INSTANCE = WalkForwardResearchLab()
    return _LAB_INSTANCE
