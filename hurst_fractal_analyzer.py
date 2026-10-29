"""
hurst_fractal_analyzer.py
=========================
Fraktale Marktanalyse & Hurst-Exponent Regime-Klassifizierung.

Mathematischer Hintergrund:
1. Hurst-Exponent (H) via Rescaled-Variance / Increment-Scaling:
   - H > 0.58: Persistenter Trend (Trendfolge-Strategien haben massiven statistischen Edge).
   - 0.45 <= H <= 0.58: Random Walk / Brownsche Bewegung (Häcksler-Markt; Fehlausbrüche häufen sich).
   - H < 0.45: Anti-persistente Mean-Reversion (Oszillatoren, Bollinger-Fades & Dip-Buying überlegen).
2. Fraktale Dimension: D = 2 - H
   - Hohes D (> 1.55) = Raue, unruhige Kurve (starkes Rauschen).
   - Niedriges D (< 1.42) = Glatte, gerichtete Trendbewegung.
3. Kaufman-Effizienz-Ratio (ER):
   - ER = |Close_t - Close_{t-n}| / Sum(|Close_i - Close_{i-1}|)
   - ER > 0.40 = Hocheffiziente, saubere Trend-Autobahn.

Verhindert Fehlausbruchs-Käufe in chaotischen Random-Walk-Phasen und maximiert die
Positionsgröße in echten persistenten Trend-Phasen.
"""

from __future__ import annotations
import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("autonomous_agent.hurst_fractal")


@dataclass
class FractalRegimeSnapshot:
    symbol: str
    hurst_exponent: float
    fractal_dimension: float
    efficiency_ratio: float
    regime_mode: str  # PERSISTENT_TRENDING | MEAN_REVERTING | RANDOM_WALK_CHOP
    favored_strategy_type: str  # TREND_FOLLOWING | MEAN_REVERSION | CASH_STANDBY
    regime_edge_multiplier: float  # 0.5x (Chop) bis 1.35x (Reiner Trend)
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


class HurstFractalAnalyzer:
    def __init__(self, min_periods: int = 40):
        self.min_periods = min_periods

    def analyze_series(self, symbol: str, prices: pd.Series) -> FractalRegimeSnapshot:
        """
        Analysiert eine Schlusskurs-Reihe (Close Prices) und liefert den fraktalen Regime-Status.
        """
        clean_p = prices.dropna().astype(float)
        if len(clean_p) < self.min_periods:
            return FractalRegimeSnapshot(
                symbol=symbol,
                hurst_exponent=0.50,
                fractal_dimension=1.50,
                efficiency_ratio=0.25,
                regime_mode="RANDOM_WALK_CHOP",
                favored_strategy_type="CASH_STANDBY",
                regime_edge_multiplier=0.80,
                summary="Zu wenig Kurshistorie für fraktale Analyse",
            )

        p_vals = clean_p.values[-64:]  # Letzte 64 Handelstage
        n = len(p_vals)

        # 1. Hurst-Exponent Berechnung via Generalized Structure Function:
        # E[(p_{t+tau} - p_t)^2]^(1/2) ~ tau^H
        lags = [4, 8, 16, 24]
        vals = []
        for lag in lags:
            diffs = p_vals[lag:] - p_vals[:-lag]
            rms = float(np.sqrt(np.mean(diffs ** 2)))
            vals.append(max(1e-6, rms))

        # Lineare Regression: log(rms) ~ H * log(lag)
        log_lags = np.log(lags)
        log_vals = np.log(vals)
        try:
            poly = np.polyfit(log_lags, log_vals, 1)
            hurst = float(np.clip(poly[0], 0.15, 0.95))
        except Exception:
            hurst = 0.50

        # 2. Fraktale Dimension: D = 2 - H
        fractal_dim = round(2.0 - hurst, 3)

        # 3. Kaufman-Effizienz-Ratio (ER) über die letzten 20 Tage
        p_20 = p_vals[-20:]
        net_move = abs(p_20[-1] - p_20[0])
        total_path = np.sum(np.abs(np.diff(p_20)))
        efficiency = float(net_move / max(1e-8, total_path))
        efficiency = round(np.clip(efficiency, 0.0, 1.0), 3)

        # 4. Regime-Einstufung
        if hurst >= 0.58 and efficiency >= 0.32:
            mode = "PERSISTENT_TRENDING"
            favored = "TREND_FOLLOWING"
            mult = 1.30
            desc = f"Persistenter Trend (H: {hurst:.2f}, Effizienz: {efficiency:.2f}) – Ausbrüche haben hohen Edge"
        elif hurst <= 0.44:
            mode = "MEAN_REVERTING"
            favored = "MEAN_REVERSION"
            mult = 1.15
            desc = f"Anti-persistente Mean-Reversion (H: {hurst:.2f}) – Überverkaufte Dips bevorzugen"
        else:
            mode = "RANDOM_WALK_CHOP"
            favored = "CASH_STANDBY"
            mult = 0.65
            desc = f"Random-Walk Rauschen (H: {hurst:.2f}, Effizienz: {efficiency:.2f}) – Fehlausbruchs-Gefahr"

        return FractalRegimeSnapshot(
            symbol=symbol,
            hurst_exponent=round(hurst, 3),
            fractal_dimension=fractal_dim,
            efficiency_ratio=efficiency,
            regime_mode=mode,
            favored_strategy_type=favored,
            regime_edge_multiplier=mult,
            summary=desc,
        )
