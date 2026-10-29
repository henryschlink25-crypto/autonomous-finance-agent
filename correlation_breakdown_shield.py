"""
correlation_breakdown_shield.py
===============================
🚨 Dynamic Correlation Breakdown Shield (Citadel / Risk Parity Crisis Alpha).

Inspiriert von institutionellen Risikomodellen (Bridgewater All-Weather, Citadel Crisis Alpha):
- Wall-Street-Axiom: "In a crash, all correlations go to 1.0."
- Wenn Panik-Abverkäufe und Liquiditätskollapse einsetzen, verlieren traditionelle
  Diversifikationsvorteile ihre Wirkung (Aktien, Krypto und High-Yield-Bonds stürzen synchron ab).
- Der Shield misst die rollierende paarweise Korrelation über alle gehaltenen Positionen
  sowie über das Gesamt-Universum.
- Schwellenwerte:
  • Durchschnittliche Korrelation < 0.55: HEALTHY_DIVERSIFICATION (Normaler Handel)
  • Durchschnittliche Korrelation >= 0.60: ELEVATED_CO_MOVEMENT (Erhöhte Wachsamkeit, halbe Neukäufe)
  • Durchschnittliche Korrelation >= 0.75: SYSTEMIC_CORRELATION_BREAKDOWN:
    -> Sofortiger Stopp für neue Long-Positionen (BLOCK_NEW_LONGS).
    -> Hebung der Mindest-Cash-Quote auf 40-45%.
    -> Bevorzugung von Safe-Haven-Absicherungen (z. B. GLD).
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("autonomous_agent.correlation_shield")

CACHE_FILE = Path("data/correlation_shield_cache.json")


class CorrelationShieldStatus(BaseModel):
    shield_active: bool = False
    average_pairwise_correlation: float = 0.0
    max_pairwise_correlation: float = 0.0
    market_regime: str = "HEALTHY_DIVERSIFICATION"  # "HEALTHY_DIVERSIFICATION", "ELEVATED_CO_MOVEMENT", "SYSTEMIC_CORRELATION_BREAKDOWN"
    protective_mandate: str = "NORMAL_ALLOCATION"   # "NORMAL_ALLOCATION", "RESTRICT_NEW_LONGS", "EMERGENCY_CASH_PRESERVATION"
    required_cash_buffer_pct: float = 15.0          # 15%, 30% oder 45%
    high_correlation_pairs: List[str] = Field(default_factory=list)
    monitored_assets_count: int = 0
    calculated_at: str = ""
    summary: str = ""


class CorrelationBreakdownShield:
    """
    Überwacht die systemische Kopplung der Portfoliopositionen und verhindert
    Gleichlauf-Fallen in volatilen Abverkäufen.
    """

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_status: Optional[CorrelationShieldStatus] = None

    def evaluate_returns_matrix(
        self,
        returns_df: pd.DataFrame,
        held_symbols: Optional[List[str]] = None,
    ) -> CorrelationShieldStatus:
        """
        Berechnet den Gleichlauf aus einem Return-DataFrame.
        Falls Portfolio-Positionen übergeben wurden und >= 2 im Datensatz vorhanden sind,
        wird primär das gehaltene Portfolio bewertet, andernfalls das Gesamtuniversum.
        """
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if returns_df is None or returns_df.empty or len(returns_df.columns) < 2:
            return CorrelationShieldStatus(
                shield_active=False,
                average_pairwise_correlation=0.20,
                max_pairwise_correlation=0.35,
                market_regime="HEALTHY_DIVERSIFICATION",
                protective_mandate="NORMAL_ALLOCATION",
                required_cash_buffer_pct=15.0,
                monitored_assets_count=len(returns_df.columns) if returns_df is not None else 0,
                calculated_at=now_iso,
                summary="Zu wenige Daten für Korrelationsanalyse (Standard-Allokation aktiv).",
            )

        # Relevante Spalten auswählen
        valid_cols = [c for c in returns_df.columns if returns_df[c].std() > 1e-6]
        if held_symbols:
            held_in_data = [s for s in held_symbols if s in valid_cols]
            if len(held_in_data) >= 2:
                eval_cols = held_in_data
            else:
                eval_cols = valid_cols
        else:
            eval_cols = valid_cols

        if len(eval_cols) < 2:
            return CorrelationShieldStatus(
                shield_active=False,
                average_pairwise_correlation=0.25,
                max_pairwise_correlation=0.40,
                market_regime="HEALTHY_DIVERSIFICATION",
                protective_mandate="NORMAL_ALLOCATION",
                required_cash_buffer_pct=15.0,
                monitored_assets_count=len(eval_cols),
                calculated_at=now_iso,
                summary="Einzelposition / Nicht genügend Paare für Korrelationsmatrix.",
            )

        sub_df = returns_df[eval_cols].tail(30).fillna(0.0)
        corr_matrix = sub_df.corr().fillna(0.0)

        # Paarweise Korrelationen (obere Dreiecksmatrix ohne Diagonale)
        pairs_corr: List[Tuple[str, str, float]] = []
        n = len(eval_cols)
        for i in range(n):
            for j in range(i + 1, n):
                r = float(corr_matrix.iloc[i, j])
                pairs_corr.append((eval_cols[i], eval_cols[j], r))

        if not pairs_corr:
            avg_corr = 0.0
            max_corr = 0.0
            high_pairs = []
        else:
            corrs_only = [p[2] for p in pairs_corr]
            avg_corr = round(float(np.mean(corrs_only)), 2)
            max_corr = round(float(np.max(corrs_only)), 2)
            sorted_pairs = sorted(pairs_corr, key=lambda x: x[2], reverse=True)
            high_pairs = [f"{a} ↔ {b} (r={r:+.2f})" for a, b, r in sorted_pairs if r >= 0.70][:5]

        # Schwellenwert-Logik
        if avg_corr >= 0.75:
            regime = "SYSTEMIC_CORRELATION_BREAKDOWN"
            mandate = "EMERGENCY_CASH_PRESERVATION"
            shield_active = True
            cash_buffer = 45.0
            summary = (
                f"🚨 SYSTEMIC CORRELATION BREAKDOWN: Mittlere Korrelation liegt bei {avg_corr:.2f} (>= 0.75)! "
                f"Diversifikation versagt – 'All correlations go to 1.0'. Alle Neukäufe blockiert, Cash auf 45% erhöht."
            )
        elif avg_corr >= 0.60:
            regime = "ELEVATED_CO_MOVEMENT"
            mandate = "RESTRICT_NEW_LONGS"
            shield_active = False  # Noch kein vollständiger Notfall-Shield, aber Warnung
            cash_buffer = 30.0
            summary = (
                f"⚠️ Erhöhter systemischer Gleichlauf ({avg_corr:.2f}): Assets koppeln sich enger. "
                f"Neukauf-Größen werden um 50% gedrosselt, Cash-Ziel 30%."
            )
        else:
            regime = "HEALTHY_DIVERSIFICATION"
            mandate = "NORMAL_ALLOCATION"
            shield_active = False
            cash_buffer = 15.0
            summary = (
                f"🟢 Gesunde Diversifikation: Mittlere Korrelation {avg_corr:.2f} im sicheren Bereich (< 0.60). "
                f"Volle Allokationsfreiheit aktiv."
            )

        res = CorrelationShieldStatus(
            shield_active=shield_active,
            average_pairwise_correlation=avg_corr,
            max_pairwise_correlation=max_corr,
            market_regime=regime,
            protective_mandate=mandate,
            required_cash_buffer_pct=cash_buffer,
            high_correlation_pairs=high_pairs,
            monitored_assets_count=len(eval_cols),
            calculated_at=now_iso,
            summary=summary,
        )

        self._last_status = res
        self._save_cache(res)
        return res

    def _save_cache(self, status: CorrelationShieldStatus) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(status.model_dump(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Konnte Correlation-Shield-Cache nicht speichern: {e}")

    def format_telegram_summary(self, status: CorrelationShieldStatus) -> str:
        icon = {
            "SYSTEMIC_CORRELATION_BREAKDOWN": "🚨🔴",
            "ELEVATED_CO_MOVEMENT": "⚠️🟡",
            "HEALTHY_DIVERSIFICATION": "🛡️🟢",
        }.get(status.market_regime, "🛡️")

        lines = [
            f"{icon} <b>Dynamic Correlation Breakdown Shield</b>",
            "─────────────────────────────",
            f"• <b>Status:</b> <b>{status.market_regime}</b>",
            f"• <b>Notfall-Sperre:</b> {'🚨 AKTIV (Kaufstopp)' if status.shield_active else '✅ Inaktiv (Freigabe)'}",
            f"• <b>Mittlere Korrelation (ρ):</b> <b>{status.average_pairwise_correlation:.2f}</b> (Alarm ab &ge; 0.75)",
            f"• <b>Maximale Kopplung:</b> <b>{status.max_pairwise_correlation:.2f}</b>",
            f"• <b>Geforderte Cash-Quote:</b> <b>{status.required_cash_buffer_pct:.0f}%</b>",
            f"• <b>Überwachte Assets:</b> {status.monitored_assets_count}\n",
        ]

        if status.high_correlation_pairs:
            lines.append("🔗 <b>Eng gekoppelte Paare:</b>")
            for p in status.high_correlation_pairs:
                lines.append(f"• {p}")
            lines.append("")

        lines.extend([
            "─────────────────────────────",
            f"💡 <i>{status.summary}</i>",
        ])
        return "\n".join(lines)


_SHIELD_INSTANCE: Optional[CorrelationBreakdownShield] = None


def get_correlation_shield() -> CorrelationBreakdownShield:
    global _SHIELD_INSTANCE
    if _SHIELD_INSTANCE is None:
        _SHIELD_INSTANCE = CorrelationBreakdownShield()
    return _SHIELD_INSTANCE
