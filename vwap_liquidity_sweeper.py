"""
vwap_liquidity_sweeper.py
=========================
🎯 Institutional VWAP & Volume-Profile Liquidity Sweeper (Sub-Agent #28).

Berechnet quantitative institutionelle Orderflow-Metriken:
1. Intraday & Multi-Day Volume-Weighted Average Price (VWAP).
2. VWAP Standardabweichungs-Bänder (±1σ, ±2σ, ±3σ).
3. Volume Profile: Point of Control (POC), Value Area High (VAH) & Value Area Low (VAL, 70% Volumenbereich).
4. Erkennt gezielte "Liquidity Sweeps":
   - Wenn Market Maker Retail-Stop-Losses unter dem -2σ Band oder VAL abfischen
   - Wenn institutionelles Smart Money die Verkaufsorders absorbiert (Absorption Volume)
   - Wenn der Kurs zügig zurück über das Band schließt (Reclaim Reversal)
"""

from __future__ import annotations
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("autonomous_agent.vwap_sweeper")


class VWAPVolumeProfileResult(BaseModel):
    """Vollständige quantitative VWAP- und Volume-Profile Analyse."""
    symbol: str
    current_price: float
    vwap: float
    vwap_stdev: float
    z_score: float  # z. B. -2.1σ
    band_upper_1s: float
    band_upper_2s: float
    band_upper_3s: float
    band_lower_1s: float
    band_lower_2s: float
    band_lower_3s: float
    poc_price: float        # Point of Control (Höchstes gehandeltes Volumen)
    value_area_high: float  # VAH (70% Obergrenze)
    value_area_low: float   # VAL (70% Untergrenze)
    sweep_status: str       # "BULLISH_SWEEP_RECLAIM", "BEARISH_CLIMAX_SWEEP", "IN_VALUE_RANGE", "OVERSOLD_EXPANSION", "OVERBOUGHT_EXPANSION"
    sweep_edge_score: float # -1.0 bis +1.0
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    risk_reward_ratio: float = 0.0
    volume_absorption_ratio: float = 1.0
    summary: str = ""
    calculated_at: str = ""


class InstitutionalVWAPLiquiditySweeper:
    """
    Analysiert Kurscharts auf institutionelle Stop-Loss-Abfischungen (Liquidity Sweeps)
    und berechnet präzise Mean-Reversion Einstiege zum POC/VWAP.
    """

    def __init__(self, cache_file: Optional[str] = None):
        if cache_file:
            self.cache_file = Path(cache_file)
        else:
            self.cache_file = Path("data") / "vwap_liquidity_cache.json"
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Konnte VWAP Cache nicht laden: {e}")
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Konnte VWAP Cache nicht speichern: {e}")

    def compute_vwap_and_profile(
        self,
        symbol: str,
        df: pd.DataFrame,
        num_bins: int = 30
    ) -> VWAPVolumeProfileResult:
        """
        Berechnet VWAP, Standardabweichungs-Bänder und Volume Profile aus OHLCV-Daten.
        Erwartet DataFrame mit Spalten: ['High', 'Low', 'Close', 'Volume'] oder klein geschrieben.
        """
        sym = symbol.strip().upper()
        if df.empty or len(df) < 5:
            return self._create_fallback_result(sym, df)

        # Spalten normalisieren
        cols = {c.lower(): c for c in df.columns}
        h_col = cols.get("high", "High")
        l_col = cols.get("low", "Low")
        c_col = cols.get("close", "Close")
        v_col = cols.get("volume", "Volume")

        highs = df[h_col].values.astype(float)
        lows = df[l_col].values.astype(float)
        closes = df[c_col].values.astype(float)
        volumes = df[v_col].values.astype(float)

        current_price = float(closes[-1])

        # 1. Typical Price & VWAP Berechnung
        typical_prices = (highs + lows + closes) / 3.0
        cum_vol = np.sum(volumes)
        if cum_vol <= 0:
            cum_vol = 1.0

        vwap = float(np.sum(typical_prices * volumes) / cum_vol)

        # 2. VWAP Standardabweichung
        variance = float(np.sum(volumes * ((typical_prices - vwap) ** 2)) / cum_vol)
        stdev = math.sqrt(max(0.0001, variance))

        z_score = float((current_price - vwap) / stdev) if stdev > 0 else 0.0

        # Bänder berechnen
        b_up_1s = vwap + 1.0 * stdev
        b_up_2s = vwap + 2.0 * stdev
        b_up_3s = vwap + 3.0 * stdev

        b_lo_1s = vwap - 1.0 * stdev
        b_lo_2s = vwap - 2.0 * stdev
        b_lo_3s = vwap - 3.0 * stdev

        # 3. Volume Profile & Point of Control (POC)
        price_min = float(np.min(lows))
        price_max = float(np.max(highs))
        if price_max <= price_min:
            price_max = price_min + 1.0

        bin_edges = np.linspace(price_min, price_max, num_bins + 1)
        bin_volumes = np.zeros(num_bins)

        for tp, vol in zip(typical_prices, volumes):
            idx = int(np.digitize(tp, bin_edges) - 1)
            idx = max(0, min(num_bins - 1, idx))
            bin_volumes[idx] += vol

        poc_idx = int(np.argmax(bin_volumes))
        poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)

        # Value Area (70% des Gesamtvolumens um den POC)
        target_va_vol = 0.70 * cum_vol
        current_va_vol = bin_volumes[poc_idx]
        left_idx = poc_idx
        right_idx = poc_idx

        while current_va_vol < target_va_vol and (left_idx > 0 or right_idx < num_bins - 1):
            next_left_vol = bin_volumes[left_idx - 1] if left_idx > 0 else 0.0
            next_right_vol = bin_volumes[right_idx + 1] if right_idx < num_bins - 1 else 0.0

            if next_left_vol >= next_right_vol and left_idx > 0:
                left_idx -= 1
                current_va_vol += next_left_vol
            elif right_idx < num_bins - 1:
                right_idx += 1
                current_va_vol += next_right_vol
            else:
                break

        val_price = float(bin_edges[left_idx])
        vah_price = float(bin_edges[right_idx + 1])

        # 4. Liquidity Sweep & Absorption Erkennung
        recent_low = float(np.min(lows[-5:])) if len(lows) >= 5 else float(lows[-1])
        recent_high = float(np.max(highs[-5:])) if len(highs) >= 5 else float(highs[-1])
        recent_vol_avg = float(np.mean(volumes[-5:])) if len(volumes) >= 5 else float(volumes[-1])
        base_vol_avg = float(np.mean(volumes)) if len(volumes) > 0 else 1.0
        absorption_ratio = float(recent_vol_avg / max(1.0, base_vol_avg))

        sweep_status = "IN_VALUE_RANGE"
        edge_score = 0.0
        sl_price = None
        tp_price = None
        rr_ratio = 0.0

        # Bullischer Liquidity Sweep:
        # Tief tauchte unter -2σ oder VAL ein, aber aktueller Kurs schließt zurück darüber mit Bestätigung!
        pierced_lower = (recent_low <= b_lo_2s) or (recent_low < val_price)
        reclaimed_up = (
            (recent_low <= b_lo_2s and current_price >= b_lo_2s and current_price > recent_low + 0.3 * stdev) or
            (recent_low < val_price and current_price >= val_price and current_price > recent_low + 0.3 * stdev)
        )

        # Bärische Erschöpfung / Climax Exhaustion am oberen Band (+2σ / VAH):
        pierced_upper = (recent_high >= b_up_2s) or (recent_high > vah_price)
        rejected_down = (
            (recent_high >= b_up_2s and current_price <= b_up_2s and current_price < recent_high - 0.3 * stdev) or
            (recent_high > vah_price and current_price <= vah_price and current_price < recent_high - 0.3 * stdev)
        )

        if pierced_lower and reclaimed_up and z_score < 0.5:
            sweep_status = "BULLISH_SWEEP_RECLAIM"
            edge_score = min(0.92, 0.65 + min(0.20, (absorption_ratio - 1.0) * 0.15))
            sl_price = round(recent_low * 0.995, 2)  # Knapper Stopp knapp unter dem Sweep-Tief
            tp_price = round(max(poc_price, vwap, b_up_1s), 2)  # Rebound-Ziel: POC, VWAP oder +1σ Band
            risk = max(0.05, current_price - sl_price)
            reward = max(0.05, tp_price - current_price)
            rr_ratio = round(reward / risk, 2)
            summary = (
                f"🎯 Bullischer Liquidity-Sweep bestätigt! Kurs tauchte auf ${recent_low:.2f} (unter -2σ / VAL) "
                f"mit {absorption_ratio:.1f}x Absorptions-Volumen und schloss zurück auf ${current_price:.2f}. "
                f"Ziel: POC ${poc_price:.2f} (CRV {rr_ratio}:1)."
            )

        # Bärische Erschöpfung / Exhaustion Sweep am oberen Band (+2σ / VAH):
        elif pierced_upper and rejected_down and z_score >= 1.2:
            sweep_status = "BEARISH_CLIMAX_SWEEP"
            edge_score = -0.75
            sl_price = round(recent_high * 1.008, 2)
            tp_price = round(poc_price, 2)
            summary = (
                f"⚠️ Bärische Buying Exhaustion! Kurs prallte am +2σ Band (${b_up_2s:.2f}) ab. "
                f"Hohes Risiko für Mean-Reversion zurück zum POC ${poc_price:.2f}."
            )
        elif z_score <= -2.0:
            sweep_status = "OVERSOLD_EXPANSION"
            edge_score = 0.40
            summary = f"Stark überverkauft am -2σ VWAP-Band ({z_score:.2f}σ). Warte auf Rebound-Bestätigung."
        elif z_score >= 2.0:
            sweep_status = "OVERBOUGHT_EXPANSION"
            edge_score = -0.40
            summary = f"Überkauft am +2σ VWAP-Band ({z_score:.2f}σ). Kein Neukauf empfohlen."
        else:
            sweep_status = "IN_VALUE_RANGE"
            edge_score = 0.05 if z_score < 0 else -0.05
            summary = f"Kurs pendelt in der Value Area zwischen VAL (${val_price:.2f}) und VAH (${vah_price:.2f}) um den POC (${poc_price:.2f})."

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        res = VWAPVolumeProfileResult(
            symbol=sym,
            current_price=round(current_price, 2),
            vwap=round(vwap, 2),
            vwap_stdev=round(stdev, 2),
            z_score=round(z_score, 2),
            band_upper_1s=round(b_up_1s, 2),
            band_upper_2s=round(b_up_2s, 2),
            band_upper_3s=round(b_up_3s, 2),
            band_lower_1s=round(b_lo_1s, 2),
            band_lower_2s=round(b_lo_2s, 2),
            band_lower_3s=round(b_lo_3s, 2),
            poc_price=round(poc_price, 2),
            value_area_high=round(vah_price, 2),
            value_area_low=round(val_price, 2),
            sweep_status=sweep_status,
            sweep_edge_score=round(edge_score, 2),
            suggested_stop_loss=sl_price,
            suggested_take_profit=tp_price,
            risk_reward_ratio=rr_ratio,
            volume_absorption_ratio=round(absorption_ratio, 2),
            summary=summary,
            calculated_at=now_iso,
        )

        self._cache[sym] = {
            "timestamp": time.time(),
            "data": res.model_dump()
        }
        self._save_cache()
        return res

    def _create_fallback_result(self, symbol: str, df: pd.DataFrame) -> VWAPVolumeProfileResult:
        px = 100.0 if df.empty else float(df.iloc[-1].get("Close", 100.0))
        return VWAPVolumeProfileResult(
            symbol=symbol,
            current_price=px,
            vwap=px,
            vwap_stdev=px * 0.02,
            z_score=0.0,
            band_upper_1s=round(px * 1.02, 2),
            band_upper_2s=round(px * 1.04, 2),
            band_upper_3s=round(px * 1.06, 2),
            band_lower_1s=round(px * 0.98, 2),
            band_lower_2s=round(px * 0.96, 2),
            band_lower_3s=round(px * 0.94, 2),
            poc_price=px,
            value_area_high=round(px * 1.03, 2),
            value_area_low=round(px * 0.97, 2),
            sweep_status="IN_VALUE_RANGE",
            sweep_edge_score=0.0,
            summary="Nicht genügend historische Balken für vollständiges Volume Profile.",
            calculated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    def analyze_symbol(self, symbol: str, force_refresh: bool = False) -> VWAPVolumeProfileResult:
        """
        Holt historische Marktdaten (z.B. 5-Minuten / Stundendaten via yfinance) und analysiert das Profil.
        """
        sym = symbol.strip().upper()
        now_ts = time.time()

        if not force_refresh and sym in self._cache:
            entry = self._cache[sym]
            if now_ts - entry.get("timestamp", 0) < 600:  # 10 Min Cache
                return VWAPVolumeProfileResult(**entry.get("data", {}))

        # Marktdaten laden
        try:
            import yfinance as yf
            ticker = yf.Ticker(sym)
            df = ticker.history(period="5d", interval="15m")
            if df.empty or len(df) < 10:
                df = ticker.history(period="1mo", interval="1d")
            return self.compute_vwap_and_profile(sym, df)
        except Exception as e:
            logger.warning(f"Fehler beim Laden der Marktdaten für {sym}: {e}")
            return self._create_fallback_result(sym, pd.DataFrame())

    def format_telegram_summary(self, res: VWAPVolumeProfileResult) -> str:
        """
        Formatiert die Analyse für Telegram.
        """
        icon = {
            "BULLISH_SWEEP_RECLAIM": "🎯🟢",
            "BEARISH_CLIMAX_SWEEP": "⚠️🔴",
            "OVERSOLD_EXPANSION": "🟡",
            "OVERBOUGHT_EXPANSION": "🟠",
            "IN_VALUE_RANGE": "⚪",
        }.get(res.sweep_status, "⚪")

        lines = [
            f"🎯 <b>Institutional VWAP & Volume Profile: {res.symbol}</b>",
            "─────────────────────────────",
            f"• <b>Status:</b> {icon} <b>{res.sweep_status}</b> (Edge: <b>{res.sweep_edge_score:+.2f}</b>)",
            f"• <b>Aktueller Kurs:</b> <b>${res.current_price:.2f}</b>",
            f"• <b>VWAP:</b> <b>${res.vwap:.2f}</b> (Abweichung: <b>{res.z_score:+.2f}σ</b>)",
            f"• <b>Point of Control (POC):</b> <b>${res.poc_price:.2f}</b> (Max. Volumen)",
            f"• <b>Value Area (70%):</b> ${res.value_area_low:.2f} (VAL) bis ${res.value_area_high:.2f} (VAH)\n",
            "📊 <b>VWAP Standardabweichungs-Bänder:</b>",
            f"• +2σ Band (Kauf-Erschöpfung): ${res.band_upper_2s:.2f}",
            f"• +1σ Band: ${res.band_upper_1s:.2f}",
            f"• -1σ Band: ${res.band_lower_1s:.2f}",
            f"• -2σ Band (Liquiditäts-Tasche): ${res.band_lower_2s:.2f}\n",
        ]

        if res.sweep_status == "BULLISH_SWEEP_RECLAIM":
            lines.append("🔥 <b>Institutionelles Setup aktiv:</b>")
            lines.append(f"• Stop-Loss (Sweep Low): <b>${res.suggested_stop_loss:.2f}</b>")
            lines.append(f"• Gewinnziel (POC / VWAP): <b>${res.suggested_take_profit:.2f}</b>")
            lines.append(f"• Chance-Risiko-Verhältnis (CRV): <b>{res.risk_reward_ratio}:1</b>")
            lines.append(f"• Absorptions-Volumen: <b>{res.volume_absorption_ratio:.1f}x</b>\n")

        lines.append("─────────────────────────────")
        lines.append(f"💡 <i>{res.summary}</i>")
        return "\n".join(lines)


# Singleton
_SWEEPER_INSTANCE: Optional[InstitutionalVWAPLiquiditySweeper] = None


def get_vwap_sweeper() -> InstitutionalVWAPLiquiditySweeper:
    global _SWEEPER_INSTANCE
    if _SWEEPER_INSTANCE is None:
        _SWEEPER_INSTANCE = InstitutionalVWAPLiquiditySweeper()
    return _SWEEPER_INSTANCE
