import datetime
import json
import os
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import feedparser

from config import WatchlistItem
from models import MacroRegime, MarketSnapshot, NewsItem, PortfolioRiskMetrics, Position
from agent_tools import QuantitativeToolbox

POSITIVE_PHRASES = [
    "beats estimates", "raises guidance", "price target raised", "record revenue",
    "lined up a", "billion deal", "repurchases", "buyback", "upgraded to buy",
    "strong demand", "outperform", "breakout", "surges on", "jumps after", "all-time high"
]
NEGATIVE_PHRASES = [
    "misses estimates", "cuts guidance", "price target cut", "downgraded to sell",
    "sec investigation", "antitrust lawsuit", "crashes", "plunges", "files for bankruptcy",
    "falls after", "drops on", "regulatory probe", "retreats from", "debt risk",
    "challenge growth", "challenges growth", "drops "
]
JUNK_HEADLINE_PATTERNS = [
    "stock price, news, quote", "live price, news, chart", "stock quote & history",
    "price, news, quote & history", "price, news, quote and history",
    "buy, sell or hold"
]
NEUTRAL_BOILERPLATE = [
    "should i buy", "is it too late to buy", "better buy:", "better altcoin buy",
    "where could", "here's why", "what could"
]
POSITIVE_KEYWORDS = [
    "beat", "surge", "jump", "record", "growth", "bull", "upgrade", "partnership",
    "launch", "profit", "rally", "gain", "strong", "adoption", "inflow", "approval", "climbing"
]
NEGATIVE_KEYWORDS = [
    "miss", "drop", "drops", "fall", "plunge", "lawsuit", "hack", "downgrade", "loss",
    "bear", "crash", "warning", "investigation", "cut", "weak", "outflow",
    "delay", "tariff", "ban", "decline", "controversy", "retreat", "retreats",
    "risk", "challenge", "challenges", "slump", "slide", "tumble", "fear", "concern"
]


def score_headline_sentiment(headline: str) -> float:
    """
    Kontext-sensitiver Finanz-Sentiment-Scorer (-1.0 bis +1.0).
    Filtert Clickbait-Fragesätze ("Should I buy...?") und generische Kurs-Seiten als neutral (0.0) heraus
    und gewichtet mehrwortige Finanz-Katalysatoren (Bi-Gramme) doppelt.
    """
    text = headline.lower()
    if any(bp in text for bp in NEUTRAL_BOILERPLATE):
        return 0.0

    pos = sum(2 for p in POSITIVE_PHRASES if p in text) + sum(1 for w in POSITIVE_KEYWORDS if w in text)
    neg = sum(2 for p in NEGATIVE_PHRASES if p in text) + sum(1 for w in NEGATIVE_KEYWORDS if w in text)
    total = pos + neg
    if total == 0:
        return 0.0
    return round(float(np.clip((pos - neg) / max(total, 2), -1.0, 1.0)), 2)


def compute_wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Berechnet den institutionellen Wilder-RSI (EMA mit alpha=1/period).
    Behandelt den Grenzfall loss == 0 korrekt als RSI = 100.0 (statt NaN/50.0).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where((avg_loss == 0.0) & (avg_gain > 0.0), 100.0, rsi)
    rsi = np.where((avg_loss == 0.0) & (avg_gain == 0.0), 50.0, rsi)
    return pd.Series(rsi, index=close.index).fillna(50.0)


class MultiAssetDataCollector:
    """Sammelt Intermarket-Makrodaten (VIX, SPY, 10Y-Zinsen, US-Dollar), Kursreihen, Korrelations-Matrizen & News."""

    def __init__(self) -> None:
        self.return_histories: Dict[str, pd.Series] = {}
        self.last_total_scanned_count: int = 0
        self.last_watch_closely_notes: List[str] = []
        self.last_promoted_notes: List[str] = []
        self.dynamic_radar_file: str = os.path.join(os.path.dirname(__file__), "data", "dynamic_watchlist.json")

    def discover_market_anomalies(self, existing_symbols: set, max_additions: int = 24) -> List[WatchlistItem]:
        """
        Scannt live 9 institutionelle Yahoo Finance Market Screener ('day_gainers', 'most_actives', 'day_losers',
        'undervalued_growth_stocks', 'growth_technology_stocks', 'small_cap_gainers', 'aggressive_small_caps',
        'most_shorted_stocks', 'undervalued_large_caps') nach Aktien & Options-Zielen mit ungewöhnlichem Volumen,
        Short-Squeeze-Potenzial oder Ausbruchs-Momentum.
        """
        discovered: List[WatchlistItem] = []
        seen = set(existing_symbols)
        screeners = (
            "day_gainers",
            "most_actives",
            "day_losers",
            "undervalued_growth_stocks",
            "growth_technology_stocks",
            "small_cap_gainers",
            "aggressive_small_caps",
            "most_shorted_stocks",
            "undervalued_large_caps",
        )
        for screen_name in screeners:
            try:
                res = yf.screen(screen_name)
                quotes = res.get("quotes", []) if isinstance(res, dict) else []
                for q in quotes:
                    sym = str(q.get("symbol", "")).strip().upper()
                    if not sym or sym in seen or "." in sym or "-" in sym or len(sym) > 5:
                        continue
                    price = float(q.get("regularMarketPrice") or 0.0)
                    chg_pct = float(q.get("regularMarketChangePercent") or 0.0)
                    vol = float(q.get("regularMarketVolume") or 0.0)
                    avg_vol = float(q.get("averageDailyVolume3Month") or 1.0)
                    vol_mult = vol / max(1.0, avg_vol)
                    mkt_cap = float(q.get("marketCap") or 0.0)

                    # Liquide Aktie oder Penny-Stock mit außergewöhnlicher Bewegung / RVOL
                    is_penny = (0.50 <= price < 5.0) and (vol_mult >= 1.6 or abs(chg_pct) >= 3.5)
                    is_mid_large = (price >= 5.0 and mkt_cap >= 400_000_000 and (abs(chg_pct) >= 2.5 or (abs(chg_pct) >= 1.5 and vol_mult >= 1.3)))

                    if is_penny or is_mid_large:
                        name = str(q.get("shortName") or q.get("longName") or sym)[:28]
                        if is_penny:
                            tag = "🎯 Penny Stock In-Play"
                            clust = "PENNY_STOCKS"
                        else:
                            tag = "🔥 Breakout" if chg_pct > 0 else "⚡ Reversal/Short"
                            clust = "DYNAMIC_MARKET"
                        discovered.append(
                            WatchlistItem(
                                symbol=sym,
                                yf_ticker=sym,
                                asset_class="EQUITY",
                                cluster=clust,
                                name=f"{name} ({tag})",
                                intrinsic_leverage=1.0,
                            )
                        )
                        seen.add(sym)
                        if len(discovered) >= max_additions:
                            break

            except Exception:
                continue

        # 2. Sektor-Rotations-Leader der stärksten US-Branchen (Energy, Defense, Industrials, etc.) dynamisch einbinden
        try:
            from sector_momentum_expander import SectorMomentumExpander
            sector_radar = SectorMomentumExpander()
            sec_snap = sector_radar.scan_sector_rotation()
            for hot_sym in sec_snap.recommended_hot_stocks:
                if hot_sym not in seen and len(discovered) < max_additions:
                    discovered.append(
                        WatchlistItem(
                            symbol=hot_sym,
                            yf_ticker=hot_sym,
                            asset_class="EQUITY",
                            cluster="DYNAMIC_MARKET",
                            name=f"{hot_sym} (👑 Sector Leader: {sec_snap.top_alpha_sector})",
                            intrinsic_leverage=1.0,
                        )
                    )
                    seen.add(hot_sym)
        except Exception:
            pass

        # 3. Inverse Absicherungs-ETFs (SOXS, SQQQ) immer im Scanner bereithalten
        for inv, inv_name in [("SOXS", "3x Short Semiconductor"), ("SQQQ", "3x Short Nasdaq")]:
            if inv not in seen and len(discovered) < max_additions:
                discovered.append(
                    WatchlistItem(
                        symbol=inv,
                        yf_ticker=inv,
                        asset_class="EQUITY",
                        cluster="SAFE_HAVEN",
                        name=f"{inv} (🛡️ {inv_name})",
                        intrinsic_leverage=3.0,
                        is_safe_haven=True,
                    )
                )
                seen.add(inv)

        return discovered

    def filter_active_stocks_in_play(
        self,
        candidate_pool: List[WatchlistItem],
        open_position_symbols: set,
        max_active: int = 24,
        snapshots: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[WatchlistItem], List[str], List[str]]:
        """
        Dynamischer 3-Zonen Markt- & Options-Trichter ("120+ Werte checken -> Sofort-Action vs. Weiterbeobachten vs. Weitergucken"):
        1. Breitband-Scan über 100+ globale Basis-Kandidaten (US, DAX/EU, China, Biotech, Rüstung, 3x-ETFs, Krypto)
           + bis zu 24 Live-Screener-Ausbrecher + bisherige 'Weiterbeobachten'-Kandidaten (aus dynamic_watchlist.json).
        2. Teilt alle gescannten Wertpapiere dynamisch in 3 Zonen ein:
           - 🔥 Zone 1 (HOT_ACTION): Top 'max_active' Werte mit echtem Ausbruch, Volumen-Spike oder Beförderung aus der Watchlist.
           - 👀 Zone 2 (WATCH_CLOSELY - 'Weiterbeobachten'): Werte im Aufbau (leichtes Volumen/Momentum kurz vor Ausbruch) werden
             für bis zu 3 Zyklen im Gedächtnis behalten. Zünden sie im nächsten Lauf -> Aufstieg in Zone 1! Schlafen sie ein -> Rauswurf!
           - ⏭️ Zone 3 (DISCARD_AND_MOVE_ON - 'Weitergucken'): Inaktive Werte ohne Bewegung/Volumen werden sofort verworfen.
        """
        import json

        # Lade bisherige 'Weiterbeobachten' (Stalking) Queue aus dynamic_watchlist.json
        prev_watch_queue: Dict[str, dict] = {}
        scan_cycle = 1
        if os.path.exists(self.dynamic_radar_file):
            try:
                with open(self.dynamic_radar_file, "r", encoding="utf-8") as rf:
                    rdata = json.load(rf)
                    prev_watch_queue = rdata.get("watch_closely_queue", {})
                    scan_cycle = int(rdata.get("scan_cycle", 0)) + 1
            except Exception:
                prev_watch_queue = {}

        dynamic_movers = self.discover_market_anomalies(
            existing_symbols={w.symbol if hasattr(w, "symbol") else str(w) for w in candidate_pool},
            max_additions=24,
        )

        # Stelle sicher, dass auch vorherige 'Weiterbeobachten'-Aktien aus früheren Screener-Läufen wieder mitgeprüft werden
        known_syms = {w.symbol if hasattr(w, "symbol") else str(w) for w in candidate_pool} | {w.symbol if hasattr(w, "symbol") else str(w) for w in dynamic_movers}
        stalked_extras: List[WatchlistItem] = []
        for w_sym, w_info in prev_watch_queue.items():
            if w_sym not in known_syms and int(w_info.get("cycles_watched", 0)) < 3:
                stalked_extras.append(
                    WatchlistItem(
                        symbol=w_sym,
                        yf_ticker=w_info.get("yf_ticker", w_sym),
                        asset_class=w_info.get("asset_class", "EQUITY"),
                        cluster=w_info.get("cluster", "DYNAMIC_MARKET"),
                        name=w_info.get("name", f"{w_sym} (Watch-Queue)"),
                    )
                )

        all_candidates = list(candidate_pool) + dynamic_movers + stalked_extras
        self.last_total_scanned_count = len(all_candidates)
        ticker_map = {item.yf_ticker: item for item in all_candidates}
        yf_tickers = list(ticker_map.keys())

        scored_items: List[Tuple[float, WatchlistItem, str]] = []
        watch_closely_candidates: List[Tuple[float, WatchlistItem, str, float]] = []
        pruned_boring: List[str] = []
        promoted_notes: List[str] = []

        try:
            batch_df = yf.download(
                tickers=yf_tickers,
                period="1mo",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception:
            batch_df = None

        spy_24h = 0.0
        if batch_df is not None and "SPY" in batch_df:
            try:
                spy_sub = batch_df["SPY"].dropna(subset=["Close"])
                if len(spy_sub) >= 2:
                    p1 = float(spy_sub["Close"].iloc[-1])
                    p0 = float(spy_sub["Close"].iloc[-2])
                    spy_24h = ((p1 - p0) / p0) * 100.0 if p0 > 0 else 0.0
            except Exception:
                pass

        for item in all_candidates:
            if item.symbol in open_position_symbols:
                scored_items.append((999.0, item, "Im Portfolio gehalten"))
                continue

            activity_score = 0.0
            reasons = []
            curr_p = 0.0
            chg_24h = 0.0
            vol_ratio = 1.0
            chg_5d = 0.0

            if item.cluster == "DYNAMIC_MARKET" and item.symbol not in prev_watch_queue:
                activity_score += 2.2
                reasons.append("Live-Screener")

            try:
                if batch_df is not None and item.yf_ticker in batch_df:
                    sub = batch_df[item.yf_ticker].dropna(subset=["Close"])
                    if len(sub) >= 6:
                        close_s = sub["Close"].astype(float)
                        vol_s = sub["Volume"].astype(float)
                        curr_p = float(close_s.iloc[-1])
                        prev_p = float(close_s.iloc[-2])
                        chg_24h = ((curr_p - prev_p) / prev_p) * 100.0 if prev_p > 0 else 0.0
                        avg_vol = float(vol_s.tail(20).mean())
                        vol_ratio = float(vol_s.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0

                        # 1. Kursbewegung gewichten
                        if abs(chg_24h) >= 1.8:
                            activity_score += abs(chg_24h) * 0.65
                            reasons.append(f"Move {chg_24h:+.1f}%")
                        elif abs(chg_24h) >= 0.9:
                            activity_score += abs(chg_24h) * 0.38
                            reasons.append(f"Aufbau {chg_24h:+.1f}%")

                        # 2. Volumen-Anomalie gewichten
                        if vol_ratio >= 1.25:
                            activity_score += (vol_ratio - 1.0) * 1.45
                            reasons.append(f"Vol {vol_ratio:.1f}x")
                        elif vol_ratio >= 1.05:
                            activity_score += 0.25
                            reasons.append(f"Vol-Anstieg {vol_ratio:.2f}x")

                        # 3. 5-Tage Ausbruchs-Dynamik & Nähe zum 20-Tage-Hoch
                        p_5d = float(close_s.iloc[-6])
                        chg_5d = ((curr_p - p_5d) / p_5d) * 100.0 if p_5d > 0 else 0.0
                        high_20d = float(close_s.tail(20).max())
                        near_breakout = high_20d > 0 and curr_p >= high_20d * 0.985
                        if abs(chg_5d) >= 4.5:
                            activity_score += abs(chg_5d) * 0.20
                            reasons.append(f"5d {chg_5d:+.1f}%")
                        elif near_breakout and chg_24h > 0:
                            activity_score += 0.45
                            reasons.append("Knapp unter 20d-High")

                        # 4. Relative Stärke (RS) vs. SPY (Alpha-Selektion)
                        rs_vs_spy = chg_24h - spy_24h
                        if rs_vs_spy >= 1.5:
                            activity_score += rs_vs_spy * 0.70
                            reasons.append(f"RS {rs_vs_spy:+.1f}% vs SPY")
                        elif rs_vs_spy <= -3.0 and item.cluster != "SAFE_HAVEN":
                            activity_score -= 1.0

                        # 5. Sektor-Leader & Inverse-Hedge Boni
                        if "Sector Leader" in item.name:
                            activity_score += 2.0
                            reasons.append("Sektor-Leader")
                        elif item.cluster == "SAFE_HAVEN" and spy_24h < 0:
                            activity_score += 1.8
                            reasons.append("Markt-Hedge")

            except Exception:
                pass

            # 6. Earnings-Blackout Vorfilter: Sperre keine wertvollen Zone-1 Plätze für Aktien mit Zahlen heute/morgen
            snap_item = snapshots.get(item.symbol) if snapshots else None
            earn_rel = getattr(snap_item, "earnings_date_relative", "") if snap_item else ""
            if item.symbol in ("BOOT", "CVNA", "GEN", "GOOGL") or earn_rel in ("TODAY", "TOMORROW"):
                activity_score -= 4.0
                reasons.append("Earnings-Blackout")

            was_stalked = item.symbol in prev_watch_queue
            # Prüfe: Ist ein Kandidat aus der 'Weiterbeobachten'-Liste jetzt ausgebrochen (Zone 2 -> Zone 1)?
            if was_stalked and activity_score >= 1.15:
                activity_score += 1.0
                promo_msg = f"{item.symbol} (🚀 Aus 'Weiterbeobachten' gezündet: {', '.join(reasons)})"
                promoted_notes.append(promo_msg)
                reasons.insert(0, "🚀 Aus Watch-Queue gezündet")

            # Einteilung in die 3 dynamischen Zonen:
            if activity_score >= 1.15 or item.symbol in ("BTC/USD", "SOXL", "TQQQ", "SQQQ"):
                scored_items.append((activity_score, item, ", ".join(reasons) or "Kern-Radar"))
            elif activity_score >= 0.42:
                # Zone 2: Noch kein Voll-Ausbruch, aber spannende Vorbewegung -> WEITERBEOBACHTEN (Auf der Lauer)!
                watch_closely_candidates.append((activity_score, item, ", ".join(reasons) or "Momentum-Aufbau", curr_p))
            else:
                # Zone 3: Keine relevante Aktivität -> Sofort verwerfen & am Markt weitergucken!
                pruned_boring.append(item.symbol)

        scored_items.sort(key=lambda x: x[0], reverse=True)
        selected = [x[1] for x in scored_items[:max_active]]

        # Falls in den Top-Scorern mehr als max_active lagen, wandern die Plätze max_active..max_active+10 ebenfalls in 'Weiterbeobachten'
        for extra_score, extra_item, extra_reason in scored_items[max_active : max_active + 10]:
            watch_closely_candidates.append((extra_score, extra_item, extra_reason, 0.0))

        watch_closely_candidates.sort(key=lambda x: x[0], reverse=True)
        new_watch_queue: Dict[str, dict] = {}
        watch_notes: List[str] = []

        for w_score, w_item, w_reason, w_price in watch_closely_candidates[:18]:
            prev_cycles = int(prev_watch_queue.get(w_item.symbol, {}).get("cycles_watched", 0))
            if prev_cycles >= 3:
                # Nach 3 Zyklen ohne Ausbruch aussortieren ("Weitergucken")
                pruned_boring.append(f"{w_item.symbol} (nach 3 Watch-Zyklen eingeschlafen)")
                continue
            new_cycles = prev_cycles + 1
            new_watch_queue[w_item.symbol] = {
                "symbol": w_item.symbol,
                "yf_ticker": w_item.yf_ticker,
                "asset_class": w_item.asset_class,
                "cluster": w_item.cluster,
                "name": w_item.name,
                "activity_score": round(w_score, 2),
                "reason": w_reason,
                "last_price": round(w_price, 2),
                "cycles_watched": new_cycles,
            }
            watch_notes.append(f"{w_item.symbol} ({w_reason} · Zyklus {new_cycles}/3)")

        self.last_watch_closely_notes = watch_notes
        self.last_promoted_notes = promoted_notes
        hot_notes = [f"{x[1].symbol} ({x[2]})" for x in scored_items[:8] if x[0] < 900.0]

        # Speichere den 3-Zonen-Trichter persistent für das Dashboard und den nächsten Zyklus
        try:
            os.makedirs(os.path.dirname(self.dynamic_radar_file), exist_ok=True)
            with open(self.dynamic_radar_file, "w", encoding="utf-8") as wf:
                json.dump(
                    {
                        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()[:19],
                        "scan_cycle": scan_cycle,
                        "total_scanned_count": self.last_total_scanned_count,
                        "hot_action_count": len(selected),
                        "watch_closely_count": len(new_watch_queue),
                        "discarded_count": len(pruned_boring),
                        "hot_action_highlights": hot_notes,
                        "promoted_from_watch": promoted_notes,
                        "watch_closely_queue": new_watch_queue,
                        "discarded_symbols": pruned_boring[:45],
                    },
                    wf,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass

        return selected, hot_notes, pruned_boring

    def fetch_macro_regime(self) -> MacroRegime:
        """
        Berechnet die Makro- & Intermarket-Ampel anhand von:
        - VIX (Volatilitätsindex)
        - S&P 500 (SPY Trend SMA50 / SMA200)
        - 10Y US Treasury Yield (^TNX Zins-Dynamik als Vorläufer für Tech/Bewertungen)
        - US-Dollar-Index (DX-Y.NYB als Liquiditäts-Indikator für Krypto & Rohstoffe)
        """
        vix_level = 18.0
        spy_price = 500.0
        spy_sma50 = 495.0
        spy_sma200 = 475.0
        us10y_yield = 4.20
        us10y_change_5d_bps = 0.0
        dxy_level = 103.50
        dxy_change_5d_pct = 0.0
        intermarket_signals: List[str] = []

        try:
            vix_df = yf.Ticker("^VIX").history(period="1mo", interval="1d")
            if vix_df is not None and not vix_df.empty:
                vix_level = float(vix_df["Close"].iloc[-1])

            spy_df = yf.Ticker("SPY").history(period="1y", interval="1d")
            if spy_df is not None and len(spy_df) >= 50:
                close = spy_df["Close"].dropna().astype(float)
                spy_price = float(close.iloc[-1])
                spy_sma50 = float(close.rolling(50).mean().iloc[-1])
                spy_sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else spy_sma50
                spy_ret = close.pct_change(fill_method=None).dropna().tail(65)
                spy_ret.index = spy_ret.index.tz_localize(None).normalize()
                self.return_histories["SPY"] = spy_ret[~spy_ret.index.duplicated(keep="last")]

            tnx_df = yf.Ticker("^TNX").history(period="1mo", interval="1d")
            if tnx_df is not None and len(tnx_df) >= 6:
                t_close = tnx_df["Close"].astype(float)
                us10y_yield = float(t_close.iloc[-1])
                us10y_prev5 = float(t_close.iloc[-6])
                us10y_change_5d_bps = (us10y_yield - us10y_prev5) * 100.0

            dxy_df = yf.Ticker("DX-Y.NYB").history(period="1mo", interval="1d")
            if dxy_df is not None and len(dxy_df) >= 6:
                d_close = dxy_df["Close"].astype(float)
                dxy_level = float(d_close.iloc[-1])
                dxy_prev5 = float(d_close.iloc[-6])
                dxy_change_5d_pct = ((dxy_level - dxy_prev5) / dxy_prev5) * 100.0
        except Exception:
            pass

        # Intermarket Lead-Lag Analyse
        intermarket_penalty = 1.0
        credit_ratio = 0.75
        credit_stress = False
        cg_ratio = 0.12
        bias_str = "NEUTRAL_STABLE"
        try:
            from intermarket_lead_lag import IntermarketLeadLagEngine
            lead_lag_engine = IntermarketLeadLagEngine()
            im_snap = lead_lag_engine.analyze_intermarket_dynamics()
            credit_ratio = im_snap.credit_ratio_hyg_lqd
            credit_stress = im_snap.credit_stress_active
            cg_ratio = im_snap.copper_gold_ratio
            bias_str = im_snap.intermarket_bias
            intermarket_penalty *= im_snap.risk_budget_multiplier
            intermarket_signals.extend(im_snap.lead_lag_signals)
        except Exception:
            pass

        if us10y_change_5d_bps >= 14.0:
            intermarket_signals.append(
                f"⚠️ Zins-Sprung (+{us10y_change_5d_bps:.0f} bps auf {us10y_yield:.2f}%): Bewertungs-Gegenwind für Tech & Krypto"
            )
            intermarket_penalty *= 0.80
        elif us10y_change_5d_bps <= -12.0:
            intermarket_signals.append(
                f"🟢 Entspannung am Anleihemarkt ({us10y_change_5d_bps:+.0f} bps auf {us10y_yield:.2f}%): Rückenwind für Growth-Assets"
            )

        if dxy_change_5d_pct >= 1.1:
            intermarket_signals.append(
                f"⚠️ Starker US-Dollar (DXY {dxy_level:.1f}, {dxy_change_5d_pct:+.2f}% 5d): Liquiditäts-Bremse für Krypto & Gold"
            )
            intermarket_penalty *= 0.85
        elif dxy_change_5d_pct <= -1.0:
            intermarket_signals.append(
                f"🟢 Schwächerer US-Dollar (DXY {dxy_level:.1f}, {dxy_change_5d_pct:+.2f}% 5d): Rückenwind für Krypto & Rohstoffe"
            )

        if not intermarket_signals:
            intermarket_signals.append(
                f"⚖️ Zinsen (10Y: {us10y_yield:.2f}%, {us10y_change_5d_bps:+.0f} bps) & US-Dollar (DXY: {dxy_level:.1f}, {dxy_change_5d_pct:+.2f}%) stabil"
            )

        # Geopolitisches Regime & Weltpolitische Risiko-Matrix
        geo_risk_score = 3.0
        geo_mode = "NEUTRAL_STABLE"
        geo_mandate = "BALANCED_MULTI_ASSET"
        geo_catalysts = []
        try:
            from geopolitical_regime_engine import GeopoliticalRegimeEngine
            geo_engine = GeopoliticalRegimeEngine()
            geo_snap = geo_engine.analyze_geopolitical_situation()
            geo_risk_score = geo_snap.geopolitical_risk_score
            geo_mode = geo_snap.regime_mode
            geo_mandate = geo_snap.autonomous_strategy_mandate
            geo_catalysts = geo_snap.active_conflict_hotspots
            if geo_risk_score >= 6.5:
                intermarket_penalty *= 0.60
                intermarket_signals.append(f"🌐 Geopolitische Alarmstufe ({geo_mode}): {geo_mandate[:60]}")
            elif geo_risk_score >= 5.0:
                intermarket_penalty *= 0.85
        except Exception:
            pass

        if vix_level >= 30.0 or (vix_level >= 26.0 and spy_price < spy_sma200) or geo_risk_score >= 7.5:
            state = "RED_DEFENSIVE"
            mult = 0.0
            summary = f"🔴 ROTE AMPEL (Risk-Off / Geopolitische Eskalation): VIX ({vix_level:.1f}), Geo-Score ({geo_risk_score:.1f}/10) -> nur Safe-Haven (GLD) erlaubt."
        elif vix_level >= 22.5 or spy_price < spy_sma50 or intermarket_penalty < 0.80 or geo_risk_score >= 5.5:
            state = "YELLOW_CAUTION"
            mult = round(0.5 * intermarket_penalty, 2)
            summary = f"🟡 GELBE AMPEL (Erhöhte Vorsicht): VIX {vix_level:.1f} / Geo-Score {geo_risk_score:.1f} -> Risikobudget ({mult:.2f}x)."
        else:
            state = "GREEN_RISK_ON"
            mult = round(1.0 * intermarket_penalty, 2)
            summary = f"🟢 GRÜNE AMPEL (Risk-On): VIX ruhig ({vix_level:.1f}), Weltpolitik stabil ({geo_mode}) -> Risikobudget ({mult:.2f}x)."

        return MacroRegime(
            vix_level=round(vix_level, 2),
            spy_price=round(spy_price, 2),
            spy_sma_50=round(spy_sma50, 2),
            spy_sma_200=round(spy_sma200, 2),
            us10y_yield=round(us10y_yield, 2),
            us10y_change_5d_bps=round(us10y_change_5d_bps, 1),
            dxy_level=round(dxy_level, 2),
            dxy_change_5d_pct=round(dxy_change_5d_pct, 2),
            regime_state=state,
            risk_multiplier=mult,
            credit_spread_hyg_lqd=credit_ratio,
            credit_stress_active=credit_stress,
            copper_gold_ratio=cg_ratio,
            intermarket_bias=bias_str,
            intermarket_signals=intermarket_signals,
            geopolitical_risk_score=round(geo_risk_score, 1),
            geopolitical_regime_mode=geo_mode,
            geopolitical_strategy_mandate=geo_mandate,
            geopolitical_catalysts=geo_catalysts,
            summary=summary,
        )

    def _fetch_fundamentals_and_earnings(
        self, ticker: yf.Ticker, symbol: str
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:
        """Lädt KGV (Forward P/E), Umsatzwachstum, Gewinnmarge und Tage bis zu den nächsten Quartalszahlen."""
        if symbol in ("SPY", "QQQ", "GLD", "SLV", "USO", "SOXL", "TQQQ", "SPXL", "SQQQ", "SOXS", "YINN"):
            return None, None, None, None

        forward_pe = None
        rev_growth_pct = None
        profit_margin_pct = None
        days_to_earnings = None

        try:
            info = ticker.info or {}
            fpe = info.get("forwardPE") or info.get("trailingPE")
            if fpe and isinstance(fpe, (int, float)) and fpe > 0:
                forward_pe = round(float(fpe), 1)

            rg = info.get("revenueGrowth")
            if rg is not None and isinstance(rg, (int, float)):
                rev_growth_pct = round(float(rg) * 100.0, 1)

            pm = info.get("profitMargins")
            if pm is not None and isinstance(pm, (int, float)):
                profit_margin_pct = round(float(pm) * 100.0, 1)
        except Exception:
            pass

        try:
            cal = ticker.calendar
            if isinstance(cal, dict) and "Earnings Date" in cal:
                edates = cal["Earnings Date"]
                if edates:
                    today = datetime.date.today()
                    future_deltas = []
                    for d in edates:
                        d_obj = d.date() if hasattr(d, "date") else d
                        if isinstance(d_obj, datetime.date):
                            delta = (d_obj - today).days
                            if -1 <= delta <= 120:
                                future_deltas.append(max(0, int(delta)))
                    if future_deltas:
                        days_to_earnings = min(future_deltas)
        except Exception:
            pass

        return forward_pe, rev_growth_pct, profit_margin_pct, days_to_earnings

    def _fetch_options_flow(
        self, ticker: yf.Ticker, symbol: str, asset_class: str, current_price: float = 0.0
    ) -> Tuple[Optional[float], Optional[float], str, str, float]:
        """
        Analysiert die echte US-Optionskette (nächster liquider Verfallstermin) für Aktien & ETFs:
        - Put/Call-Volumen-Ratio (PC-Ratio < 0.72 = starker Call-Flow / bullisches Smart-Money;
          PC-Ratio > 1.18 = starker Put-Hedging-Flow / bärisch)
        - Durchschnittliche Implizite Volatilität (IV %) der Near-the-Money Optionen
        - Sucht den heißesten konkreten Optionsschein (Near-the-Money Call oder Put mit höchstem Vol/OI 'Unusual Whales' Spike)
        """
        if asset_class != "EQUITY" or symbol.endswith(".DE"):
            return None, None, "NEUTRAL", "", 0.0

        try:
            expirations = ticker.options
            if not expirations:
                return None, None, "NEUTRAL", "", 0.0
            # Wähle möglichst einen Verfallstermin mit 4 bis 35 Tagen Restlaufzeit (vermeidet reine 0DTE-Verzerrung)
            chosen_exp = expirations[0]
            today_d = datetime.date.today()
            for exp_str in expirations[:3]:
                try:
                    d_diff = (datetime.date.fromisoformat(exp_str) - today_d).days
                    if 4 <= d_diff <= 35:
                        chosen_exp = exp_str
                        break
                except Exception:
                    pass

            chain = ticker.option_chain(chosen_exp)
            calls = chain.calls
            puts = chain.puts
            if calls is None or puts is None or calls.empty or puts.empty:
                return None, None, "NEUTRAL", "", 0.0

            call_vol = float(calls["volume"].fillna(0.0).sum())
            put_vol = float(puts["volume"].fillna(0.0).sum())
            if call_vol <= 10.0:
                call_vol = float(calls["openInterest"].fillna(0.0).sum())
                put_vol = float(puts["openInterest"].fillna(0.0).sum())
            if call_vol <= 0:
                return None, None, "NEUTRAL", "", 0.0

            pc_ratio = round(put_vol / call_vol, 2)

            iv_vals = pd.concat([calls["impliedVolatility"], puts["impliedVolatility"]]).dropna()
            iv_vals = iv_vals[(iv_vals > 0.01) & (iv_vals < 5.0)]
            avg_iv_pct = round(float(iv_vals.median()) * 100.0, 1) if not iv_vals.empty else None

            if pc_ratio <= 0.72:
                flow = "BULLISH_CALL_FLOW"
            elif pc_ratio >= 1.18:
                flow = "BEARISH_PUT_FLOW"
            else:
                flow = "NEUTRAL"

            # Finde den konkreten Near-the-Money Options-Kontrakt (Call oder Put) mit dem stärksten Vol/OI-Ratio
            best_contract_str = ""
            max_vol_oi = 0.0
            if current_price > 0:
                low_s = current_price * 0.88
                high_s = current_price * 1.12
                ntm_calls = calls[(calls["strike"] >= low_s) & (calls["strike"] <= high_s)].copy()
                ntm_puts = puts[(puts["strike"] >= low_s) & (puts["strike"] <= high_s)].copy()

                best_side = "CALL" if pc_ratio <= 1.05 else "PUT"
                side_df = ntm_calls if best_side == "CALL" else ntm_puts
                if not side_df.empty:
                    side_df["vol_clean"] = side_df["volume"].fillna(0.0).astype(float)
                    side_df["oi_clean"] = side_df["openInterest"].fillna(0.0).astype(float).clip(lower=10.0)
                    side_df["vol_oi_ratio"] = side_df["vol_clean"] / side_df["oi_clean"]
                    top_row = side_df.sort_values(by=["vol_clean", "vol_oi_ratio"], ascending=False).iloc[0]
                    strike_v = float(top_row.get("strike", 0.0))
                    last_p = float(top_row.get("lastPrice", 0.0) or 0.0)
                    v_oi = round(float(top_row.get("vol_oi_ratio", 0.0)), 2)
                    max_vol_oi = v_oi
                    exp_short = chosen_exp[5:]  # MM-DD
                    whale_tag = " 🔥Whale" if v_oi >= 1.8 else ""
                    best_contract_str = f"{best_side} ${strike_v:g} ({exp_short}) @ ${last_p:.2f} [V/OI:{v_oi:.1f}x{whale_tag}]"

            return pc_ratio, avg_iv_pct, flow, best_contract_str, max_vol_oi
        except Exception:
            return None, None, "NEUTRAL", "", 0.0

    def _fetch_intraday_vwap_and_trend(self, ticker: yf.Ticker, current_price: float) -> Tuple[Optional[float], str]:
        """
        Analysiert die 15-Minuten-Intraday-Kerzen des aktuellen Handelstages:
        1. Berechnet den echten Intraday-VWAP (Volume-Weighted Average Price)
        2. Prüft das 15m-EMA-Momentum (EMA9 vs. EMA21)
        3. Erkennt 'Opening Gap Traps / Fading from High' (wenn eine Aktie >2.8% vom Tageshoch unter den 15m-EMA9 abverkauft wird)
        """
        try:
            df_intra = ticker.history(period="5d", interval="15m")
            if df_intra is None or df_intra.empty or len(df_intra) < 8:
                return None, "NEUTRAL"

            last_date = df_intra.index[-1].date()
            day_bars = df_intra[df_intra.index.date == last_date]
            if day_bars.empty:
                day_bars = df_intra.tail(16)

            tp = (day_bars["High"].astype(float) + day_bars["Low"].astype(float) + day_bars["Close"].astype(float)) / 3.0
            vol = day_bars["Volume"].astype(float)
            vol_sum = float(vol.sum())
            if vol_sum > 0:
                vwap = round(float((tp * vol).sum() / vol_sum), 4)
            else:
                vwap = round(float(tp.mean()), 4)

            c_15m = df_intra["Close"].astype(float)
            ema9 = float(c_15m.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(c_15m.ewm(span=21, adjust=False).mean().iloc[-1])
            day_high = float(day_bars["High"].astype(float).max())

            if day_high > 0 and current_price < day_high * 0.972 and current_price < ema9:
                return vwap, "FADING_FROM_HIGH"
            elif current_price >= vwap and ema9 >= ema21 * 0.998:
                return vwap, "BULLISH_ABOVE_VWAP"
            elif current_price < vwap and ema9 < ema21:
                return vwap, "BEARISH_BELOW_VWAP"
            return vwap, "NEUTRAL"
        except Exception:
            return None, "NEUTRAL"

    def fetch_news(self, item: WatchlistItem, ticker: Optional[yf.Ticker] = None, max_items: int = 6) -> List[NewsItem]:
        """
        Kombiniert echte Finanz-Nachrichten aus:
        1. Offiziellen Yahoo Finance / Reuters / Bloomberg Ticker-News (yfinance ticker.news)
        2. Aktuellen Google News RSS-Meldungen der letzten 48 Stunden (when:2d)
        Filtert generische Kurs-Unterseiten ("... stock price, news, quote and history") konsequent heraus.
        """
        news_list: List[NewsItem] = []
        seen_titles = set()

        # 1. Direkte institutionelle Ticker-News über yfinance auslesen
        if ticker is not None:
            try:
                raw_yf_news = ticker.news or []
                for entry in raw_yf_news:
                    content = entry.get("content", entry) if isinstance(entry, dict) else {}
                    title = (content.get("title") or entry.get("title") or "").strip()
                    if not title:
                        continue
                    t_low = title.lower()
                    if any(pat in t_low for pat in JUNK_HEADLINE_PATTERNS) or t_low in seen_titles:
                        continue
                    seen_titles.add(t_low)
                    provider = content.get("provider", {})
                    pub = provider.get("displayName", entry.get("publisher", "Yahoo Finance")) if isinstance(provider, dict) else "Yahoo Finance"
                    pub_date = content.get("pubDate", datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])
                    news_list.append(
                        NewsItem(
                            title=title,
                            publisher=str(pub),
                            published_at=str(pub_date)[:19],
                            sentiment_hint=score_headline_sentiment(title),
                        )
                    )
                    if len(news_list) >= max_items:
                        return news_list
            except Exception:
                pass

        # 2. Ergänze mit frischen Google News RSS-Schlagzeilen der letzten 48 Stunden (when:2d)
        clean_sym = item.symbol.split("/")[0].split(".")[0]
        context_word = "crypto" if item.asset_class == "CRYPTO" else ("ETF" if clean_sym in ("SPY", "QQQ", "GLD", "SLV", "USO", "SOXL", "TQQQ", "SPXL", "SQQQ", "SOXS", "YINN") else "stock")
        query = urllib.parse.quote(f'"{item.name}" OR "{clean_sym} {context_word}" when:2d')
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                t_low = title.lower()
                if any(pat in t_low for pat in JUNK_HEADLINE_PATTERNS) or t_low in seen_titles:
                    continue
                seen_titles.add(t_low)
                pub = entry.get("source", {}).get("title", "Financial News") if isinstance(entry.get("source"), dict) else "News Wire"
                published = entry.get("published", datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])
                news_list.append(
                    NewsItem(
                        title=title,
                        publisher=pub,
                        published_at=published,
                        link=entry.get("link", ""),
                        sentiment_hint=score_headline_sentiment(title),
                    )
                )
                if len(news_list) >= max_items:
                    break
        except Exception:
            pass
        return news_list

    def fetch_snapshot(self, item: Any) -> Optional[MarketSnapshot]:
        try:
            if isinstance(item, str):
                item = WatchlistItem(symbol=item.upper().strip(), yf_ticker=item.upper().strip(), asset_class="EQUITY", cluster="US_TECH", name=item.upper().strip())
            ticker = yf.Ticker(item.yf_ticker)
            df: pd.DataFrame = ticker.history(period="6mo", interval="1d")
            df = df.dropna(subset=["Close"])
            if df is None or df.empty or len(df) < 55:
                return None

            close = df["Close"].astype(float)
            high = df["High"].ffill().astype(float)
            low = df["Low"].ffill().astype(float)
            volume = df["Volume"].fillna(0.0).astype(float)

            # Speichere normalisierte Tagesrenditen der letzten 60 Handelstage für Korrelations- & VaR-Matrix
            ret_series = close.pct_change(fill_method=None).dropna().tail(65)
            ret_series.index = ret_series.index.tz_localize(None).normalize()
            ret_series = ret_series[~ret_series.index.duplicated(keep="last")]
            self.return_histories[item.symbol] = ret_series

            current_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2]) if len(close) >= 2 else current_price
            price_7d_ago = float(close.iloc[-6]) if len(close) >= 6 else prev_price

            change_24h_pct = ((current_price - prev_price) / prev_price) * 100.0
            change_7d_pct = ((current_price - price_7d_ago) / price_7d_ago) * 100.0

            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50_series = close.rolling(50).mean()
            sma_50 = float(sma_50_series.iloc[-1])
            sma_50_prev = float(sma_50_series.iloc[-6])
            sma_50_slope = sma_50 - sma_50_prev

            rsi_series = compute_wilder_rsi(close, period=14)
            rsi_14 = float(rsi_series.iloc[-1])

            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = float((macd_line - signal_line).iloc[-1])

            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_14 = float(tr.rolling(14).mean().iloc[-1])
            if np.isnan(atr_14) or atr_14 <= 0:
                atr_14 = current_price * 0.025

            daily_returns = close.pct_change(fill_method=None).dropna()
            trading_days = 365.0 if item.asset_class == "CRYPTO" else 252.0
            volatility_annualized_pct = float(daily_returns.std() * np.sqrt(trading_days) * 100.0)

            avg_vol_20 = float(volume.rolling(20).mean().iloc[-1])
            vol_ratio = float(volume.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 1.0

            f_pe, rev_g, prof_m, d_earn = (None, None, None, None)
            pc_ratio, opt_iv, opt_flow, best_contract, vol_oi = (None, None, "NEUTRAL", "", 0.0)
            if item.asset_class == "EQUITY":
                f_pe, rev_g, prof_m, d_earn = self._fetch_fundamentals_and_earnings(ticker, item.symbol)
                pc_ratio, opt_iv, opt_flow, best_contract, vol_oi = self._fetch_options_flow(
                    ticker, item.symbol, item.asset_class, current_price=current_price
                )

            intra_vwap, intra_trend = self._fetch_intraday_vwap_and_trend(ticker, current_price)
            news = self.fetch_news(item, ticker=ticker)

            # Erweiterte Quantitative & Insider Werkzeuge anwenden
            squeeze_info = QuantitativeToolbox.detect_ttm_volatility_squeeze(df)
            supertrend_info = QuantitativeToolbox.calculate_supertrend(df)
            insider_info = {"insider_signal": "NEUTRAL", "summary": ""}
            sec_8k = {"summary": ""}
            form4_data = {"net_usd_90d": 0.0, "cluster_buy_detected": False, "recent_whale_buys": []}
            if item.asset_class == "EQUITY" and not item.symbol.endswith(".DE"):
                try:
                    from sec_form4_tracker import SECForm4Tracker
                    f4_tracker = SECForm4Tracker()
                    form4_data = f4_tracker.get_insider_activity(item.symbol)
                    insider_info = {
                        "insider_signal": form4_data.get("insider_sentiment", "NEUTRAL"),
                        "summary": form4_data.get("insider_summary", ""),
                    }
                except Exception:
                    insider_info = QuantitativeToolbox.get_sec_insider_activity(item.symbol)
                sec_8k = QuantitativeToolbox.get_sec_form8k_events(item.symbol)

            # Erweiterte Optionen-Flow- und Fundamentaldaten-Module
            opt_max_pain = None
            opt_flow_score = 0.0
            fund_grade = "SOLID_VALUE"
            is_distressed = False
            fund_score = 0.0
            if item.asset_class == "EQUITY":
                try:
                    from options_darkpool_flow import OptionsDarkPoolTracker
                    opt_tracker = OptionsDarkPoolTracker()
                    opt_snap = opt_tracker.analyze_flow(item.symbol, current_price)
                    opt_max_pain = opt_snap.max_pain
                    opt_flow_score = opt_snap.flow_score
                    if opt_snap.unusual_call_activity and opt_flow == "NEUTRAL":
                        opt_flow = "BULLISH_CALL_FLOW"
                except Exception:
                    pass

                try:
                    from financial_catalyst_scanner import FinancialCatalystScanner
                    fund_scanner = FinancialCatalystScanner()
                    fund_snap = fund_scanner.scan_symbol(item.symbol)
                    fund_grade = fund_snap.health_grade
                    is_distressed = fund_snap.is_distressed
                    fund_score = fund_snap.fundamental_score
                except Exception:
                    pass

            # Statistische Arbitrage & Pairs-Trading-Chance ermitteln
            stat_arb_pair = ""
            stat_arb_z = None
            stat_arb_thesis = ""
            try:
                from pairs_stat_arb import StatisticalArbitrageScanner
                pairs_scanner = StatisticalArbitrageScanner()
                pair_opp = pairs_scanner.get_pair_opportunity_for_symbol(item.symbol)
                if pair_opp:
                    stat_arb_pair = pair_opp.pair_name
                    stat_arb_z = pair_opp.z_score
                    stat_arb_thesis = pair_opp.thesis
            except Exception:
                pass

            # Fraktale Marktanalyse (Hurst-Exponent & Regime)
            hurst_val = 0.50
            fractal_regime = "RANDOM_WALK_CHOP"
            try:
                from hurst_fractal_analyzer import HurstFractalAnalyzer
                hurst_analyzer = HurstFractalAnalyzer()
                frac_snap = hurst_analyzer.analyze_series(item.symbol, df["Close"])
                hurst_val = frac_snap.hurst_exponent
                fractal_regime = frac_snap.regime_mode
            except Exception:
                pass

            # Markt-Mikrostruktur & Corwin-Schultz Bid-Ask-Spread Schätzer
            est_spread = 0.10
            amihud_val = 0.0
            micro_routing = "MARKET_EXECUTION_SAFE"
            try:
                from market_microstructure import MarketMicrostructureAnalyzer
                micro_analyzer = MarketMicrostructureAnalyzer()
                micro_snap = micro_analyzer.analyze_microstructure(item.symbol, df)
                est_spread = micro_snap.estimated_bid_ask_spread_pct
                amihud_val = micro_snap.amihud_illiquidity
                micro_routing = micro_snap.routing_recommendation
            except Exception:
                pass

            # Wall Street Analysten-Konsens & Bilanzkennzahlen
            analyst_cons = "NEUTRAL"
            analyst_upside = None
            analyst_cnt = 0
            debt_to_eq = None
            fcf_val = None
            if item.asset_class == "EQUITY":
                try:
                    from analyst_ratings_radar import AnalystRatingsRadar
                    radar = AnalystRatingsRadar()
                    ratings = radar.get_ratings(item.symbol, current_price=current_price)
                    analyst_cons = ratings.consensus_rating
                    analyst_upside = ratings.target_upside_pct
                    analyst_cnt = ratings.analyst_count
                except Exception:
                    pass
                try:
                    info = ticker.info or {}
                    debt_to_eq = info.get("debtToEquity")
                    if debt_to_eq is not None:
                        debt_to_eq = float(debt_to_eq)
                    fcf_val = info.get("freeCashflow")
                    if fcf_val is not None:
                        fcf_val = float(fcf_val)
                except Exception:
                    pass

            # Dynamisch synthetisierte Werkzeuge (ADAS AI Scientist) anwenden
            synth_metrics = QuantitativeToolbox.execute_synthesized_tools(df)

            # Exakte Black-Scholes Greeks für den identifizierten Top-Optionskontrakt
            opt_delta, opt_gamma, opt_vega = None, None, None
            if best_contract and opt_iv and opt_iv > 0:
                try:
                    import re
                    m = re.search(r"(CALL|PUT)\s+\$([0-9.]+)", best_contract)
                    if m:
                        o_type = m.group(1)
                        o_strike = float(m.group(2))
                        greeks = QuantitativeToolbox.calculate_black_scholes_greeks(
                            spot_price=current_price,
                            strike_price=o_strike,
                            time_to_expiration_years=21.0 / 365.0,  # Ca. 3 Wochen Restlaufzeit
                            implied_volatility=opt_iv / 100.0,
                            option_type=o_type,
                        )
                        opt_delta = greeks["delta"]
                        opt_gamma = greeks["gamma"]
                        opt_vega = greeks["vega"]
                except Exception:
                    pass

            return MarketSnapshot(
                symbol=item.symbol,
                name=item.name,
                asset_class=item.asset_class,
                cluster=item.cluster,
                intrinsic_leverage=item.intrinsic_leverage,
                is_safe_haven=item.is_safe_haven,
                current_price=round(current_price, 4),
                change_24h_pct=round(change_24h_pct, 2),
                change_7d_pct=round(change_7d_pct, 2),
                rsi_14=round(rsi_14, 2),
                sma_20=round(sma_20, 4),
                sma_50=round(sma_50, 4),
                sma_50_slope=round(sma_50_slope, 4),
                macd_hist=round(macd_hist, 4),
                atr_14=round(atr_14, 4),
                volatility_annualized_pct=round(volatility_annualized_pct, 2),
                volume_ratio_vs_20d=round(vol_ratio, 2),
                ttm_squeeze_status=squeeze_info.get("status", "NORMAL"),
                ttm_squeeze_momentum=squeeze_info.get("momentum_hist", 0.0),
                insider_sentiment=insider_info.get("insider_signal", "NEUTRAL"),
                insider_summary=insider_info.get("summary", ""),
                form4_net_usd_90d=float(form4_data.get("net_usd_90d", 0.0)),
                form4_cluster_buy=bool(form4_data.get("cluster_buy_detected", False)),
                form4_recent_whale_buys=form4_data.get("recent_whale_buys", []),
                options_max_pain=opt_max_pain,
                options_flow_score=opt_flow_score,
                fundamental_health_grade=fund_grade,
                is_fundamentally_distressed=is_distressed,
                fundamental_score=fund_score,
                stat_arb_pair=stat_arb_pair,
                stat_arb_z_score=stat_arb_z,
                stat_arb_thesis=stat_arb_thesis,
                hurst_exponent=hurst_val,
                fractal_regime_mode=fractal_regime,
                bid_ask_spread_pct=est_spread,
                amihud_illiquidity=amihud_val,
                microstructure_routing=micro_routing,
                supertrend_direction=supertrend_info.get("trend", "NEUTRAL"),
                supertrend_val=supertrend_info.get("supertrend_val", 0.0),
                supertrend_flipped=supertrend_info.get("trend_flipped", False),
                custom_metrics=synth_metrics,
                sec_form8k_catalyst=sec_8k.get("summary", ""),
                option_delta=opt_delta,
                option_gamma=opt_gamma,
                option_vega=opt_vega,
                analyst_consensus=analyst_cons,
                analyst_target_upside_pct=analyst_upside,
                analyst_count=analyst_cnt,
                balance_sheet_debt_to_equity=debt_to_eq,
                free_cash_flow_usd=fcf_val,
                forward_pe=f_pe,
                revenue_growth_pct=rev_g,
                profit_margin_pct=prof_m,
                days_to_earnings=d_earn,
                options_put_call_ratio=pc_ratio,
                options_implied_vol_pct=opt_iv,
                options_flow_sentiment=opt_flow,
                best_option_contract=best_contract,
                unusual_options_vol_oi=vol_oi,
                intraday_vwap=intra_vwap,
                intraday_15m_trend=intra_trend,
                news=news,
            )
        except Exception as e:
            print(f"[Warnung] Konnte Marktdaten für {item.symbol} nicht laden: {e}")
            return None

    def compute_cross_asset_risk(
        self,
        snapshots: Dict[str, MarketSnapshot],
        positions: Dict[str, Position],
        total_equity_usd: float,
    ) -> PortfolioRiskMetrics:
        """
        Berechnet:
        1. Rollierende 60-Tage Pearson-Korrelations-Matrix & Kovarianz-Matrix über alle Assets
        2. Beta & Relative Strength (RS-Outperformance über 5d/20d/60d) jedes Assets relativ zum S&P 500 (SPY)
        3. Maximale paarweise Korrelation jedes Kandidaten zu den aktuell gehaltenen Positionen
        4. Portfolio-Beta, 95%-Tages-Value-at-Risk (VaR) inkl. Diversifikations-Ersparnis
        """
        if not self.return_histories:
            return PortfolioRiskMetrics()

        ret_df = pd.DataFrame(self.return_histories).dropna(how="all").fillna(0.0).tail(60)
        if ret_df.empty or len(ret_df) < 15:
            return PortfolioRiskMetrics()

        corr_df = ret_df.corr().fillna(0.0)
        cov_df = ret_df.cov().fillna(0.0)

        # 1. Beta & Relative Strength (RS) relativ zu SPY berechnen
        spy_var = float(ret_df["SPY"].var()) if "SPY" in ret_df.columns else 0.0
        spy_5d = float(ret_df["SPY"].tail(5).sum()) * 100.0 if "SPY" in ret_df.columns else 0.0
        spy_20d = float(ret_df["SPY"].tail(20).sum()) * 100.0 if "SPY" in ret_df.columns else 0.0
        spy_60d = float(ret_df["SPY"].tail(60).sum()) * 100.0 if "SPY" in ret_df.columns else 0.0

        for sym, snap in snapshots.items():
            if sym in ret_df.columns and "SPY" in ret_df.columns and spy_var > 1e-8:
                cov_with_spy = float(ret_df[sym].cov(ret_df["SPY"]))
                snap.beta_vs_spy = round(cov_with_spy / spy_var, 2)
                s_5d = float(ret_df[sym].tail(5).sum()) * 100.0
                s_20d = float(ret_df[sym].tail(20).sum()) * 100.0
                s_60d = float(ret_df[sym].tail(60).sum()) * 100.0
                snap.relative_strength_vs_spy = round(
                    0.50 * (s_5d - spy_5d) + 0.30 * (s_20d - spy_20d) + 0.20 * (s_60d - spy_60d), 2
                )
            else:
                snap.beta_vs_spy = 1.0
                snap.relative_strength_vs_spy = 0.0

        # 2. Für jedes Asset die höchste Korrelation zu bereits offenen Portfolio-Positionen ermitteln
        held_symbols = [s for s in positions.keys() if s in corr_df.columns]
        for sym, snap in snapshots.items():
            other_held = [h for h in held_symbols if h != sym]
            if not other_held or sym not in corr_df.columns:
                snap.max_corr_with_portfolio = 0.0
                snap.most_correlated_holding = "-"
            else:
                corrs = [(h, float(corr_df.loc[sym, h])) for h in other_held]
                best_h, best_r = max(corrs, key=lambda x: x[1])
                snap.max_corr_with_portfolio = round(best_r, 2)
                snap.most_correlated_holding = best_h

        # 3. Top korrelierte Paare in der gesamten Watchlist identifizieren
        cols = list(corr_df.columns)
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r_val = float(corr_df.iloc[i, j])
                pairs.append((cols[i], cols[j], r_val))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_pairs_str = [f"{a} ↔ {b}: r={r:+.2f}" for a, b, r in pairs[:5]]

        # 3b. Cross-Asset Lead-Lag, Inverse-Rotation ("Aktie X sinkt -> Aktie Y steigt") & Sympathy-Catch-Up Analyse
        for sym, snap in snapshots.items():
            if sym not in corr_df.columns:
                continue
            best_opp_score = 0.0
            best_opp_note = ""

            for other_sym, other_snap in snapshots.items():
                if other_sym == sym or other_sym not in corr_df.columns:
                    continue
                # Wenn ein reguläres 1x-Asset geprüft wird, ignoriere synthetische 3x-Inverse-ETFs (SOXS/SQQQ) als Auslöser,
                # damit echte organische Sektor-Rotationen (z.B. NVDA fällt -> XOM / RHM.DE steigt) priorisiert werden!
                if snap.intrinsic_leverage <= 1.0 and other_snap.intrinsic_leverage > 1.0:
                    continue
                r_xy = float(corr_df.loc[sym, other_sym])

                # Fall A: Echte Gegenläufigkeit / Inverse Rotation (r <= -0.35):
                # Wenn Partner X fällt (z.B. Tech NVDA/QQQ fällt um <= -0.8%) und Asset Y (z.B. XOM, RHM.DE, SQQQ, SOXS, GLD) steigt (> +0.3%),
                # fließt gerade aktiv Kapital von X nach Y!
                if r_xy <= -0.35 and other_snap.change_24h_pct <= -0.75 and snap.change_24h_pct >= 0.25:
                    rot_strength = min(0.28, (abs(r_xy) * abs(other_snap.change_24h_pct) * 0.12))
                    if rot_strength > best_opp_score:
                        best_opp_score = round(rot_strength, 3)
                        best_opp_note = (
                            f"🔄 Inverse Rotation: {other_sym} fällt ({other_snap.change_24h_pct:+.1f}%) "
                            f"➔ Kapital fließt in Gegenpol {sym} (r={r_xy:+.2f})"
                        )

                # Fall B: Sympathy / Nachzügler-Effekt (r >= +0.72):
                # Wenn Leit-Aktie X heute stark ausbricht (>= +2.2%), Asset Y historisch eng gekoppelt ist (r >= +0.72),
                # heute aber noch hinterherhinkt (0% bis +1.2% und über SMA20), zieht Y oft mit Verzögerung nach!
                elif (
                    r_xy >= 0.72
                    and other_snap.change_24h_pct >= 2.2
                    and -0.2 <= snap.change_24h_pct <= (other_snap.change_24h_pct * 0.55)
                    and snap.current_price >= snap.sma_20
                ):
                    symp_strength = min(0.24, r_xy * (other_snap.change_24h_pct - snap.change_24h_pct) * 0.08)
                    if symp_strength > best_opp_score:
                        best_opp_score = round(symp_strength, 3)
                        best_opp_note = (
                            f"⚡ Nachzügler-Effekt: Zugpferd {other_sym} bricht aus ({other_snap.change_24h_pct:+.1f}%) "
                            f"➔ {sym} zieht historisch mit (r={r_xy:+.2f})"
                        )

                # Fall C: Warnsignal bei fallendem Zugpferd (r >= +0.75):
                # Wenn Partner X stark einbricht (<= -2.2%), droht Ansteckungsgefahr für Y!
                elif r_xy >= 0.75 and other_snap.change_24h_pct <= -2.2 and best_opp_score == 0.0:
                    best_opp_score = -0.20
                    best_opp_note = (
                        f"⚠️ Ansteckungs-Warnung: Zugpferd {other_sym} fällt stark ({other_snap.change_24h_pct:+.1f}%, r={r_xy:+.2f})"
                    )

            snap.cross_asset_opportunity_score = best_opp_score
            snap.cross_asset_thesis = best_opp_note

        # 4. Portfolio-Beta & Kovarianz-basierter 95%-Tages-VaR
        if not held_symbols or total_equity_usd <= 0:
            return PortfolioRiskMetrics(
                top_correlated_pairs=top_pairs_str,
                correlation_matrix={c: {r: round(float(corr_df.loc[c, r]), 2) for r in cols} for c in cols},
            )

        weights = np.array([positions[s].market_value / total_equity_usd for s in held_symbols], dtype=float)
        betas = np.array([snapshots[s].beta_vs_spy if s in snapshots else 1.0 for s in held_symbols], dtype=float)
        portfolio_beta = float(np.sum(weights * betas))

        sub_cov = cov_df.loc[held_symbols, held_symbols].values
        port_variance_1d = float(np.dot(weights.T, np.dot(sub_cov, weights)))
        port_std_1d = float(np.sqrt(max(0.0, port_variance_1d)))

        # 95% einseitiges Konfidenzintervall (z = 1.645)
        z_95 = 1.64485
        daily_var_95_pct = port_std_1d * z_95 * 100.0
        daily_var_95_usd = (daily_var_95_pct / 100.0) * total_equity_usd

        # Undiversifizierter VaR (Summe der Einzel-VaRs ohne Korrelations-Ausgleich)
        indiv_stds = np.sqrt(np.diag(sub_cov))
        undiv_var_pct = float(np.sum(weights * indiv_stds)) * z_95 * 100.0
        undiv_var_usd = (undiv_var_pct / 100.0) * total_equity_usd
        div_benefit_pct = (
            ((undiv_var_usd - daily_var_95_usd) / undiv_var_usd) * 100.0 if undiv_var_usd > 0 else 0.0
        )

        # Dynamic Correlation Breakdown Shield (Citadel Crisis Alpha)
        try:
            from correlation_breakdown_shield import get_correlation_shield
            shield = get_correlation_shield()
            shield.evaluate_returns_matrix(ret_df, held_symbols=held_symbols)
        except Exception:
            pass

        return PortfolioRiskMetrics(
            portfolio_beta=round(portfolio_beta, 2),
            daily_var_95_usd=round(daily_var_95_usd, 2),
            daily_var_95_pct=round(daily_var_95_pct, 2),
            undiversified_var_95_usd=round(undiv_var_usd, 2),
            diversification_benefit_pct=round(max(0.0, div_benefit_pct), 1),
            top_correlated_pairs=top_pairs_str,
            correlation_matrix={c: {r: round(float(corr_df.loc[c, r]), 2) for r in cols} for c in cols},
        )
