"""
market_microstructure.py
========================
Markt-Mikrostruktur, Liquiditäts-Tiefe & Corwin-Schultz Spread-Schätzer.

Quantitatives Werkzeug zur Slippage-Minimierung & Order-Routing-Optimierung:
1. Amihud Illiquiditäts-Ratio:
   - Misst den Preisimpuls pro gehandeltem Dollarvolumen:
     Illiq = (|Return| / DollarVolume) * 10^6
   - Hohe Werte signalisieren dünne Orderbücher, in denen Market-Orders massiven Spread-Verlust erzeugen.
2. Corwin-Schultz Bid-Ask Spread-Schätzer (Journal of Finance):
   - Schätzt den tatsächlichen effektiven Bid-Ask-Spread aus High- und Low-Preisen ohne teure Level-2-Tick-Feeds.
   - Verhindert, dass der Agent bei illiquiden Nebenwerten 1-3% Slippage an Market-Maker verschenkt.
3. Intelligente Routing-Empfehlung:
   - "MARKET_EXECUTION_SAFE": Spread < 0.12% ➜ Direkte Ausführung.
   - "LIMIT_MIDPOINT_PEG": Spread 0.12% - 0.60% ➜ Limit-Order auf den Spread-Mittelpunkt legen.
   - "SLIPPAGE_TRAP_AVOID": Spread > 0.60% ➜ Größe drosseln oder nur mit passivem Limit einsteigen.
"""

from __future__ import annotations
import json
import logging
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("autonomous_agent.microstructure")


@dataclass
class MicrostructureSnapshot:
    symbol: str
    amihud_illiquidity: float  # Niedrig (< 0.10) = Hochliquide, Hoch (> 1.0) = Dünnes Orderbuch
    estimated_bid_ask_spread_pct: float  # z.B. 0.08%
    dollar_volume_20d_avg: float
    routing_recommendation: str  # MARKET_EXECUTION_SAFE | LIMIT_MIDPOINT_PEG | SLIPPAGE_TRAP_AVOID
    max_slippage_slippage_bps: float
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


class MarketMicrostructureAnalyzer:
    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days

    def analyze_microstructure(self, symbol: str, ohlcv_df: pd.DataFrame) -> MicrostructureSnapshot:
        """
        Analysiert OHLCV-Daten und berechnet Illiquidität & Corwin-Schultz Spread.
        """
        if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 5:
            return MicrostructureSnapshot(
                symbol=symbol,
                amihud_illiquidity=0.05,
                estimated_bid_ask_spread_pct=0.10,
                dollar_volume_20d_avg=50_000_000.0,
                routing_recommendation="MARKET_EXECUTION_SAFE",
                max_slippage_slippage_bps=5.0,
                summary="Standard-Liquidität angenommen (unzureichende Historie)",
            )

        df = ohlcv_df.tail(self.lookback_days).copy()
        high = df["High"].astype(float).values.flatten()
        low = df["Low"].astype(float).values.flatten()
        close = df["Close"].astype(float).values.flatten()
        volume = df["Volume"].astype(float).values.flatten()

        # 1. 20-Tage Durchschnitts-Dollar-Volumen
        dollar_vol = close * volume
        avg_dollar_vol = float(np.mean(dollar_vol)) if len(dollar_vol) > 0 else 10_000_000.0

        # 2. Amihud Illiquiditäts-Ratio
        returns = np.abs(np.diff(close) / close[:-1])
        d_vol = dollar_vol[1:]
        amihud_vals = returns / np.maximum(100_000.0, d_vol) * 1e6
        amihud = round(float(np.median(amihud_vals)) if len(amihud_vals) > 0 else 0.05, 4)

        # 3. Corwin-Schultz Bid-Ask Spread Schätzer
        spread_pct = self._calculate_corwin_schultz_spread(high, low)

        # 4. Routing-Empfehlung
        if spread_pct <= 0.15:
            routing = "MARKET_EXECUTION_SAFE"
            slip_bps = 5.0
            rec_text = f"Hohe Liquidität (Spread ~{spread_pct:.2f}%) ➜ Unbedenkliche Sofort-Ausführung"
        elif spread_pct <= 0.60:
            routing = "LIMIT_MIDPOINT_PEG"
            slip_bps = 18.0
            rec_text = f"Mittlerer Spread (~{spread_pct:.2f}%) ➜ Midpoint-Limit-Order empfohlen"
        else:
            routing = "SLIPPAGE_TRAP_AVOID"
            slip_bps = 45.0
            rec_text = f"🚨 Dünnes Orderbuch (Spread ~{spread_pct:.2f}%, Amihud {amihud:.3f}) ➜ Vorsicht vor Slippage!"

        summary = (
            f"Spread: {spread_pct:.2f}% | Amihud: {amihud:.3f} | "
            f"Dollar-Vol: ${avg_dollar_vol/1e6:.1f}M/Tag ➜ {routing}"
        )

        return MicrostructureSnapshot(
            symbol=symbol,
            amihud_illiquidity=amihud,
            estimated_bid_ask_spread_pct=round(spread_pct, 2),
            dollar_volume_20d_avg=round(avg_dollar_vol, 2),
            routing_recommendation=routing,
            max_slippage_slippage_bps=slip_bps,
            summary=summary,
        )

    def _calculate_corwin_schultz_spread(self, high: np.ndarray, low: np.ndarray) -> float:
        """
        Berechnet den 2-Tage High/Low Spread nach Corwin & Schultz (2012).
        """
        try:
            if len(high) < 4:
                return 0.12

            # Gamma = [ln(H_{t, t+1} / L_{t, t+1})]^2
            # Beta = [ln(H_t / L_t)]^2 + [ln(H_{t+1} / L_{t+1})]^2
            hl_ratio_daily = np.log(np.maximum(1.0001, high / np.maximum(1e-4, low)))
            beta = hl_ratio_daily[:-1] ** 2 + hl_ratio_daily[1:] ** 2

            h_2d = np.maximum(high[:-1], high[1:])
            l_2d = np.minimum(low[:-1], low[1:])
            gamma = np.log(np.maximum(1.0001, h_2d / np.maximum(1e-4, l_2d))) ** 2

            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
            alpha = np.maximum(0.0, alpha)  # Negative Spreads auf 0 clippen

            # Spread = 2 * (e^alpha - 1) / (1 + e^alpha)
            spreads = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha)) * 100.0
            med_spread = float(np.median(spreads))
            return max(0.02, min(5.0, med_spread))
        except Exception:
            return 0.15
