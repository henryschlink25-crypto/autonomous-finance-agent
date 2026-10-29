"""
tradingview_sentinel.py - TradingView Consensus & 26-Indicator Matrix Sentinel (Sub-Agent #38)
Fragt das offizielle TradingView Technical Analysis Rating in Echtzeit ab:
- 26 Technische Indikatoren (RSI, Stoch, MACD, ADX, CCI, Ichimoku, EMA10..200, SMA10..200)
- Gibt Empfehlungen (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL)
- Schützt vor Käufen gegen schweren technischen Gegenwind und verleiht Conviction-Boost bei Konsens.
"""

import datetime
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger("TradingViewSentinel")


@dataclass
class TradingViewAnalysis:
    symbol: str
    recommendation: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    buy_count: int
    sell_count: int
    neutral_count: int
    oscillators_recommendation: str
    moving_averages_recommendation: str
    confidence_score: float  # 0.0 bis 1.0
    summary_text: str
    cached: bool = False
    fetched_at: str = ""


class TradingViewSentinel:
    _instance: Optional["TradingViewSentinel"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_ttl_seconds: int = 180):
        if getattr(self, "_initialized", False):
            return
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Tuple[datetime.datetime, TradingViewAnalysis]] = {}
        self._initialized = True

    def _resolve_exchange_and_screener(self, symbol: str) -> Tuple[str, str, str]:
        """Bestimmt Exchange, Screener und TV-Symbol."""
        sym_clean = symbol.upper().strip()

        # Krypto
        if sym_clean in ("BTC-USD", "BTCUSD", "BTC"):
            return "BTCUSD", "BINANCE", "crypto"
        if sym_clean in ("ETH-USD", "ETHUSD", "ETH"):
            return "ETHUSD", "BINANCE", "crypto"
        if sym_clean in ("SOL-USD", "SOLUSD", "SOL"):
            return "SOLUSD", "BINANCE", "crypto"

        # Europa / Xetra
        if sym_clean.endswith(".DE"):
            ticker = sym_clean.replace(".DE", "")
            return ticker, "XETR", "germany"

        # US Standard
        # Typische NASDAQ vs NYSE Zuordnungen
        nasdaq_symbols = {
            "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST",
            "AMD", "INTC", "QCOM", "PLTR", "SOUN", "BBAI", "CRWD", "PANW", "COIN",
            "MSTR", "MARA", "RIOT", "CLSK", "ASTS", "HIMS", "DKNG", "CELH", "RDDT",
            "NET", "SOFI", "IONQ", "AXGN", "PURR", "MESO", "SOXL", "QQQ"
        }
        if sym_clean in nasdaq_symbols:
            return sym_clean, "NASDAQ", "america"
        
        # Default America (NYSE / NASDAQ)
        return sym_clean, "NYSE", "america"

    def analyze_symbol(self, symbol: str) -> TradingViewAnalysis:
        """Fragt das TradingView Technical Rating mit 180s Cache ab."""
        now = datetime.datetime.now(datetime.timezone.utc)
        if symbol in self._cache:
            ts, cached_analysis = self._cache[symbol]
            if (now - ts).total_seconds() < self.cache_ttl:
                cached_analysis.cached = True
                return cached_analysis

        tv_sym, exch, screener = self._resolve_exchange_and_screener(symbol)
        try:
            from tradingview_ta import TA_Handler, Interval
            handler = TA_Handler(
                symbol=tv_sym,
                exchange=exch,
                screener=screener,
                interval=Interval.INTERVAL_1_DAY,
                timeout=4.5,
            )
            analysis = handler.get_analysis()
            summary = analysis.summary or {}
            rec = summary.get("RECOMMENDATION", "NEUTRAL")
            buy = int(summary.get("BUY", 0))
            sell = int(summary.get("SELL", 0))
            neutral = int(summary.get("NEUTRAL", 0))
            osc_rec = (analysis.oscillators or {}).get("RECOMMENDATION", "NEUTRAL")
            ma_rec = (analysis.moving_averages or {}).get("RECOMMENDATION", "NEUTRAL")

            total_valid = max(1, buy + sell + neutral)
            conf = round(float(buy / total_valid), 2) if rec in ("STRONG_BUY", "BUY") else (
                round(float(sell / total_valid), 2) if rec in ("STRONG_SELL", "SELL") else 0.50
            )

            res = TradingViewAnalysis(
                symbol=symbol,
                recommendation=rec,
                buy_count=buy,
                sell_count=sell,
                neutral_count=neutral,
                oscillators_recommendation=osc_rec,
                moving_averages_recommendation=ma_rec,
                confidence_score=conf,
                summary_text=f"TV {rec} ({buy} Buy / {sell} Sell / {neutral} Neutral | MA: {ma_rec})",
                cached=False,
                fetched_at=now.isoformat()[:19],
            )
            self._cache[symbol] = (now, res)
            return res

        except Exception as e:
            logger.debug(f"TradingView TA Fetch Fehler für {symbol}: {e}")
            # Neutraler Fallback bei Timeout oder ungelistetem Ticker
            fb = TradingViewAnalysis(
                symbol=symbol,
                recommendation="NEUTRAL",
                buy_count=0,
                sell_count=0,
                neutral_count=0,
                oscillators_recommendation="NEUTRAL",
                moving_averages_recommendation="NEUTRAL",
                confidence_score=0.50,
                summary_text=f"TV Konsens neutral (Fallback: {e})",
                cached=False,
                fetched_at=now.isoformat()[:19],
            )
            return fb


_tv_sentinel: Optional[TradingViewSentinel] = None


def get_tradingview_sentinel() -> TradingViewSentinel:
    global _tv_sentinel
    if _tv_sentinel is None:
        _tv_sentinel = TradingViewSentinel()
    return _tv_sentinel
