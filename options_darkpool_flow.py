"""
options_darkpool_flow.py
========================
Institutionelles Smart-Money- & Optionen-Orderflow-Tracking.

Analysiert:
1. Put/Call Ratio (PCR) auf Volumen- und Open-Interest-Basis.
2. Ungewöhnliche Options-Aktivität (Unusual Call Activity: Volumen > Open Interest bei OTM-Strikes = Aggressive institutionelle Call-Käufe).
3. Max-Pain-Preis: Der Strike-Preis, an dem Optionen-Schreiber am wenigsten auszahlen müssen (wirkt wie ein Magnet zum Verfall).
4. Dark-Pool / Smart-Money-Flow Score (-1.0 bis +1.0) zur Swarm- und Red-Team-Validierung.
"""

from __future__ import annotations
import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yfinance as yf

logger = logging.getLogger("autonomous_agent.options_flow")

CACHE_DIR = Path("data/options_flow_cache")
CACHE_EXPIRY_SEC = 900  # 15 Minuten Cache


@dataclass
class OptionsFlowSnapshot:
    symbol: str
    timestamp: float
    spot_price: float
    nearest_expiration: str
    total_call_volume: int
    total_put_volume: int
    put_call_ratio_volume: float
    put_call_ratio_oi: float
    max_pain: float
    unusual_call_activity: bool
    unusual_put_activity: bool
    top_bullish_strike: Optional[float]
    top_bearish_strike: Optional[float]
    flow_sentiment: str  # BULLISH_FLOW | BEARISH_FLOW | NEUTRAL | NO_OPTIONS_DATA
    flow_score: float  # -1.0 bis +1.0
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


class OptionsDarkPoolTracker:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}_flow.json"

    def analyze_flow(self, symbol: str, current_price: Optional[float] = None) -> OptionsFlowSnapshot:
        """
        Analysiert den Optionen-Flow für das gegebene Symbol.
        Nutzt 15-Minuten Disk-Cache zur Ratenbegrenzung.
        """
        sym = symbol.upper().strip()
        cache_file = self._get_cache_path(sym)

        # 1. Cache prüfen
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if time.time() - data.get("timestamp", 0) < CACHE_EXPIRY_SEC:
                    return OptionsFlowSnapshot(**data)
            except Exception as e:
                logger.debug(f"Cache-Fehler für {sym}: {e}")

        # 2. Live-Abfrage via yfinance
        try:
            ticker = yf.Ticker(sym)
            expirations = ticker.options
            if not expirations:
                snap = self._create_empty_snapshot(sym, current_price or 0.0, "Keine Optionen für Symbol gelistet")
                self._save_cache(snap)
                return snap

            # Wähle nächste Verfallsserie (vorzugsweise nächste 1-3 Wochen)
            target_exp = expirations[0]
            chain = ticker.option_chain(target_exp)

            calls = chain.calls
            puts = chain.puts

            if calls.empty and puts.empty:
                snap = self._create_empty_snapshot(sym, current_price or 0.0, "Leere Options-Chain")
                self._save_cache(snap)
                return snap

            # Spot-Preis ermitteln
            spot = current_price
            if spot is None or spot <= 0:
                try:
                    spot = float(ticker.fast_info.last_price or 0.0)
                except Exception:
                    spot = 100.0  # Fallback

            # Volumen & Open Interest Aggregation
            call_vol = int(calls["volume"].fillna(0).sum()) if "volume" in calls else 0
            put_vol = int(puts["volume"].fillna(0).sum()) if "volume" in puts else 0

            call_oi = int(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls else 0
            put_oi = int(puts["openInterest"].fillna(0).sum()) if "openInterest" in puts else 0

            # Put/Call Ratios berechnen
            pcr_vol = round(put_vol / max(1, call_vol), 3)
            pcr_oi = round(put_oi / max(1, call_oi), 3)

            # Ungewöhnliche Aktivitäten scannen (Whale Accumulation: Volume > OI bei OTM Strikes)
            unusual_call = False
            top_bull_strike = None
            if not calls.empty and spot > 0:
                otm_calls = calls[calls["strike"] >= spot * 0.98]
                for _, row in otm_calls.iterrows():
                    v = row.get("volume", 0) or 0
                    oi = row.get("openInterest", 0) or 0
                    if v >= 250 and v > oi * 1.5:
                        unusual_call = True
                        top_bull_strike = float(row.get("strike", 0.0))
                        break

            unusual_put = False
            top_bear_strike = None
            if not puts.empty and spot > 0:
                otm_puts = puts[puts["strike"] <= spot * 1.02]
                for _, row in otm_puts.iterrows():
                    v = row.get("volume", 0) or 0
                    oi = row.get("openInterest", 0) or 0
                    if v >= 250 and v > oi * 1.5:
                        unusual_put = True
                        top_bear_strike = float(row.get("strike", 0.0))
                        break

            # Max Pain berechnen
            max_pain = self._calculate_max_pain(calls, puts, spot)

            # Flow-Score und Sentiment
            # PCR < 0.6 = Stark bullisch, PCR > 1.2 = Bärisch
            score = 0.0
            if pcr_vol < 0.6:
                score += 0.4
            elif pcr_vol > 1.2:
                score -= 0.4

            if unusual_call:
                score += 0.4
            if unusual_put:
                score -= 0.4

            # Wenn Spot signifikant unter Max Pain liegt, Tendenz nach oben (Magnet)
            if max_pain > 0 and spot > 0:
                pain_diff_pct = (max_pain - spot) / spot
                if pain_diff_pct > 0.03:
                    score += 0.2
                elif pain_diff_pct < -0.03:
                    score -= 0.2

            score = max(-1.0, min(1.0, round(score, 2)))

            if score >= 0.35:
                sentiment = "BULLISH_FLOW"
            elif score <= -0.35:
                sentiment = "BEARISH_FLOW"
            else:
                sentiment = "NEUTRAL"

            summary = (
                f"Exp: {target_exp} | PCR-Vol: {pcr_vol} (Calls: {call_vol:,}, Puts: {put_vol:,}) | "
                f"MaxPain: ${max_pain:.2f} | Unusual: {'CALLS' if unusual_call else 'PUTS' if unusual_put else 'None'}"
            )

            snap = OptionsFlowSnapshot(
                symbol=sym,
                timestamp=time.time(),
                spot_price=round(spot, 2),
                nearest_expiration=target_exp,
                total_call_volume=call_vol,
                total_put_volume=put_vol,
                put_call_ratio_volume=pcr_vol,
                put_call_ratio_oi=pcr_oi,
                max_pain=round(max_pain, 2),
                unusual_call_activity=unusual_call,
                unusual_put_activity=unusual_put,
                top_bullish_strike=top_bull_strike,
                top_bearish_strike=top_bear_strike,
                flow_sentiment=sentiment,
                flow_score=score,
                summary=summary,
            )

            self._save_cache(snap)
            return snap

        except Exception as e:
            logger.warning(f"Fehler bei Options-Analyse für {sym}: {e}")
            return self._create_empty_snapshot(sym, current_price or 0.0, f"Fehler: {str(e)[:50]}")

    def _calculate_max_pain(self, calls, puts, spot: float) -> float:
        """
        Berechnet den Max-Pain-Strike für die Options-Kette.
        """
        try:
            strikes = sorted(list(set(calls["strike"].tolist() + puts["strike"].tolist())))
            if not strikes:
                return spot

            min_loss = float("inf")
            max_pain_strike = spot

            for strike in strikes:
                # Verlust der Call-Schreiber: Wenn Spot > Strike
                call_loss = calls.apply(
                    lambda row: max(0.0, strike - row["strike"]) * (row.get("openInterest", 0) or 0), axis=1
                ).sum()
                # Verlust der Put-Schreiber: Wenn Spot < Strike
                put_loss = puts.apply(
                    lambda row: max(0.0, row["strike"] - strike) * (row.get("openInterest", 0) or 0), axis=1
                ).sum()

                total_loss = call_loss + put_loss
                if total_loss < min_loss:
                    min_loss = total_loss
                    max_pain_strike = strike

            return float(max_pain_strike)
        except Exception:
            return spot

    def _create_empty_snapshot(self, symbol: str, spot: float, reason: str) -> OptionsFlowSnapshot:
        return OptionsFlowSnapshot(
            symbol=symbol,
            timestamp=time.time(),
            spot_price=round(spot, 2),
            nearest_expiration="N/A",
            total_call_volume=0,
            total_put_volume=0,
            put_call_ratio_volume=1.0,
            put_call_ratio_oi=1.0,
            max_pain=round(spot, 2),
            unusual_call_activity=False,
            unusual_put_activity=False,
            top_bullish_strike=None,
            top_bearish_strike=None,
            flow_sentiment="NO_OPTIONS_DATA",
            flow_score=0.0,
            summary=reason,
        )

    def _save_cache(self, snap: OptionsFlowSnapshot) -> None:
        try:
            cache_file = self._get_cache_path(snap.symbol)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(snap.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Konnte Options-Cache für {snap.symbol} nicht speichern: {e}")
