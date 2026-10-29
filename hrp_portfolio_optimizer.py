"""
hrp_portfolio_optimizer.py - Riskfolio-Lib Hierarchical Risk Parity (HRP) Optimizer
Nutzt institutionelle Hierarchische Risiko-Parität (HRP) via Riskfolio-Lib & Dendrogramm-Clustering,
um Portfoliogewichte mathematisch optimal zu verteilen. Verhindert Klumpenrisiken in korrelierten
Tech-Werten und allokiert freies Kapital entlang der inversen Cluster-Varianz.
"""

import datetime
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

logger = logging.getLogger("HRPOptimizer")


@dataclass
class HRPAllocationResult:
    weights: Dict[str, float]  # Symbol -> Gewicht (Summe = 1.0)
    risk_measure: str  # "MV" (Volatilität) oder "CVaR"
    model_used: str  # "RISKFOLIO_HRP" oder "INVERSE_VOLATILITY_PARITY"
    asset_count: int
    portfolio_equity_usd: float
    max_position_caps_usd: Dict[str, float]
    summary_text: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])


class HRPPortfolioOptimizer:
    """
    Institutioneller Portfolio-Optimierer basierend auf Riskfolio-Lib (HRP).
    Gruppiert Assets hierarchisch nach Korrelationsbäumen und verteilt das Risiko paritätisch.
    """
    _instance: Optional["HRPPortfolioOptimizer"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

    def optimize_hrp_weights(
        self,
        returns_df: pd.DataFrame,
        total_equity_usd: float = 1000.0,
        risk_measure: str = "MV",  # "MV" oder "CVaR"
        min_history_days: int = 15,
    ) -> HRPAllocationResult:
        """
        Berechnet optimale HRP-Gewichte über Riskfolio-Lib.
        Falls Riskfolio-Lib nicht verfügbar oder Daten unvollständig sind,
        erfolgt ein robuster Inverse-Volatility-Parity Fallback.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]

        if returns_df is None or returns_df.empty or len(returns_df.columns) == 0:
            return HRPAllocationResult(
                weights={},
                risk_measure=risk_measure,
                model_used="EMPTY_FALLBACK",
                asset_count=0,
                portfolio_equity_usd=total_equity_usd,
                max_position_caps_usd={},
                summary_text="Keine Assets für HRP-Optimierung übergeben",
                timestamp=now_iso,
            )

        clean_returns = returns_df.dropna(axis=1, thresh=min_history_days).fillna(0.0)
        assets = list(clean_returns.columns)
        n_assets = len(assets)

        if n_assets == 1:
            sym = assets[0]
            return HRPAllocationResult(
                weights={sym: 1.0},
                risk_measure=risk_measure,
                model_used="SINGLE_ASSET",
                asset_count=1,
                portfolio_equity_usd=total_equity_usd,
                max_position_caps_usd={sym: round(total_equity_usd * 0.25, 2)},
                summary_text=f"HRP Single Asset: 100% {sym}",
                timestamp=now_iso,
            )

        # 1. Versuch: Riskfolio-Lib Hierarchical Risk Parity
        try:
            import riskfolio as rp
            port = rp.HCPortfolio(returns=clean_returns)
            w_df = port.optimization(
                model="HRP",
                codependence="pearson",
                rm=risk_measure,
                rf=0.04 / 252.0,
            )
            raw_weights = w_df.to_dict()["weights"]
            weights = {str(k): round(float(v), 4) for k, v in raw_weights.items()}

            # Position-Caps berechnen (z.B. max 25% pro Asset bei 1.000$ Konto = max $250)
            caps = {s: round(total_equity_usd * min(0.30, max(0.08, w * 1.5)), 2) for s, w in weights.items()}
            top_alloc = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = " | ".join([f"{k}: {v*100:.1f}%" for k, v in top_alloc])

            return HRPAllocationResult(
                weights=weights,
                risk_measure=risk_measure,
                model_used="RISKFOLIO_HRP",
                asset_count=n_assets,
                portfolio_equity_usd=total_equity_usd,
                max_position_caps_usd=caps,
                summary_text=f"Riskfolio HRP ({n_assets} Assets): {top_str}",
                timestamp=now_iso,
            )

        except Exception as e:
            logger.debug(f"Riskfolio HRP Optimierung fehlgeschlagen ({e}), weiche auf Inverse-Vol-Parity aus.")

        # 2. Fallback: Inverse-Volatility Parity
        try:
            stds = clean_returns.std()
            inv_stds = 1.0 / np.where(stds > 1e-6, stds, 1e-4)
            sum_inv = np.sum(inv_stds)
            norm_w = inv_stds / sum_inv
            weights = {assets[i]: round(float(norm_w[i]), 4) for i in range(n_assets)}
            caps = {s: round(total_equity_usd * min(0.30, max(0.08, w * 1.5)), 2) for s, w in weights.items()}
            top_alloc = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = " | ".join([f"{k}: {v*100:.1f}%" for k, v in top_alloc])

            return HRPAllocationResult(
                weights=weights,
                risk_measure=risk_measure,
                model_used="INVERSE_VOLATILITY_PARITY",
                asset_count=n_assets,
                portfolio_equity_usd=total_equity_usd,
                max_position_caps_usd=caps,
                summary_text=f"Inverse-Vol Parity ({n_assets} Assets): {top_str}",
                timestamp=now_iso,
            )
        except Exception:
            equal_w = round(1.0 / n_assets, 4)
            weights = {s: equal_w for s in assets}
            caps = {s: round(total_equity_usd * 0.20, 2) for s in assets}
            return HRPAllocationResult(
                weights=weights,
                risk_measure=risk_measure,
                model_used="EQUAL_WEIGHT_FALLBACK",
                asset_count=n_assets,
                portfolio_equity_usd=total_equity_usd,
                max_position_caps_usd=caps,
                summary_text=f"Equal-Weight Fallback ({n_assets} Assets)",
                timestamp=now_iso,
            )


_hrp_optimizer: Optional[HRPPortfolioOptimizer] = None


def get_hrp_optimizer() -> HRPPortfolioOptimizer:
    global _hrp_optimizer
    if _hrp_optimizer is None:
        _hrp_optimizer = HRPPortfolioOptimizer()
    return _hrp_optimizer
