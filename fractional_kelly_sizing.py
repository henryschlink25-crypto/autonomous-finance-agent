"""
fractional_kelly_sizing.py
==========================
Mathematisch optimale Positionsgrößenberechnung via Fractional Kelly Criterion
und Volatilitäts-Targeting.

Formel:
1. Kelly-Fraktion (reines Kelly):
       f* = (p * b - q) / b
   - p = Statistische Gewinnwahrscheinlichkeit (abgeleitet aus Schwarm-Konfidenz)
   - q = 1 - p (Verlustwahrscheinlichkeit)
   - b = Auszahlungsverhältnis (Chance-Risiko-Verhältnis / CRV)
2. Quarter-Kelly Dämpfung (lambda = 0.25):
       f_damped = lambda * f*
   - Verhindert das "Gambler's Ruin" Risiko bei realen Marktschwankungen.
3. Volatilitäts-Skalierung (Target Volatility 15% p.a.):
       vol_scale = target_vol / max(0.08, asset_annual_vol)
   - Hohe Volatilität (z. B. 65% bei Penny Stocks / Krypto) ➜ kleinere Position.
   - Niedrige Volatilität (z. B. 16% bei Bluechips) ➜ größere, sichere Position.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class KellySizingResult:
    kelly_fraction_raw: float  # Ungefiltertes Kelly
    kelly_fraction_damped: float  # Quarter-Kelly (25%)
    volatility_scale_factor: float
    recommended_position_pct: float  # Empfohlener Portfolio-Anteil (z.B. 0.08 = 8%)
    recommended_position_usd: float
    units_to_buy: float
    sizing_rationale: str


class FractionalKellyPositionSizer:
    def __init__(
        self,
        kelly_fraction_multiplier: float = 0.25,  # Quarter-Kelly
        target_annual_vol: float = 0.16,  # 16% annualisierte Zielvolatilität
        max_position_weight_pct: float = 0.25,  # Max. 25% des Portfolios in 1 Trade
        min_position_weight_pct: float = 0.02,  # Mind. 2% für spürbaren Ertrag
    ):
        self.kelly_lambda = kelly_fraction_multiplier
        self.target_annual_vol = target_annual_vol
        self.max_position_weight = max_position_weight_pct
        self.min_position_weight = min_position_weight_pct

    def compute_size(
        self,
        total_equity_usd: float,
        available_cash_usd: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        confidence: float,  # 0.50 bis 0.95
        annualized_vol_pct: float,  # z. B. 35.0 für 35% Vola
    ) -> KellySizingResult:
        if entry_price <= 0 or total_equity_usd <= 0:
            return KellySizingResult(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, "Ungültige Parameter")

        risk_dist = abs(entry_price - stop_loss_price)
        reward_dist = abs(take_profit_price - entry_price)

        if risk_dist <= 0:
            return KellySizingResult(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, "Stop-Loss Abstand ist 0")

        # 1. Chance-Risiko-Verhältnis b
        b = max(1.1, reward_dist / risk_dist)

        # 2. Gewinnwahrscheinlichkeit p (Schwarm-Konfidenz auf 52% - 82% mappen)
        p = max(0.51, min(0.82, confidence))
        q = 1.0 - p

        # 3. Kelly Formel
        f_raw = (p * b - q) / b
        if f_raw <= 0:
            return KellySizingResult(
                round(f_raw, 3), 0.0, 1.0, 0.0, 0.0, 0.0,
                f"Negativer Kelly Edge ({f_raw:.2f}) – Trade hat keinen statistischen Vorteil"
            )

        f_damped = f_raw * self.kelly_lambda

        # 4. Volatilitäts-Targeting
        asset_vol = max(0.10, annualized_vol_pct / 100.0)
        vol_scalar = max(0.40, min(1.80, self.target_annual_vol / asset_vol))

        final_pct = f_damped * vol_scalar
        final_pct = max(self.min_position_weight, min(self.max_position_weight, final_pct))

        # Verfügbaren Cash berücksichtigen
        max_usd_equity = total_equity_usd * final_pct
        alloc_usd = min(max_usd_equity, available_cash_usd * 0.95)
        alloc_usd = max(0.0, alloc_usd)

        units = round(alloc_usd / entry_price, 4) if entry_price > 0 else 0.0

        rationale = (
            f"Fractional Kelly (f*={f_raw*100:.1f}%, 1/4-Kelly: {f_damped*100:.1f}%) | "
            f"Vola-Skalierung: {vol_scalar:.2f}x (Asset-Vola {annualized_vol_pct:.0f}%) ➜ "
            f"Allokation: {final_pct*100:.1f}% (${alloc_usd:,.0f})"
        )

        return KellySizingResult(
            kelly_fraction_raw=round(f_raw, 4),
            kelly_fraction_damped=round(f_damped, 4),
            volatility_scale_factor=round(vol_scalar, 3),
            recommended_position_pct=round(final_pct, 4),
            recommended_position_usd=round(alloc_usd, 2),
            units_to_buy=units,
            sizing_rationale=rationale,
        )
