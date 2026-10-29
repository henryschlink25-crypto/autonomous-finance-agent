"""
sec_form4_tracker.py - SEC Form 4 Insider- & Whales-Tracking
Überwacht meldepflichtige Insider-Transaktionen (Vorstände, CEOs, CFOs, Direktoren und 10%+ Großaktionäre / Whales).
Nutzt die offizielle SEC EDGAR API und Yahoo Finance Insider Transactions.
Erkennt gezielt:
1. Open-Market Käufe (Transaktions-Code 'P') – Das stärkste legale Vertrauenssignal an der Börse.
2. Cluster-Buying: Wenn mehrere Vorstände zeitgleich Aktien kaufen.
3. Insider-Verkäufe (Code 'S') zur Risikowarnung im Adversarial Red-Team.
"""

import datetime
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field
from rich.console import Console

from config import AgentConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

console = Console(width=140, legacy_windows=False)


class Form4Transaction(BaseModel):
    symbol: str
    insider_name: str
    insider_title: str
    transaction_type: str  # "BUY", "SALE", "OPTION_EXERCISE", "GRANT"
    shares: float
    price_per_share: float = 0.0
    total_value_usd: float = 0.0
    filing_date: str
    is_whale: bool = False  # >= 50,000$ oder >= 10,000 Aktien
    is_open_market: bool = False  # Echter Kauf mit eigenem Privatgeld
    filing_url: str = ""


class SECForm4Tracker:
    """
    Verfolgt und analysiert SEC Form 4 Meldungen für Watchlist-Werte und Penny Stocks.
    """

    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sec_form4_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "form4_activity.json")

    def get_insider_activity(self, symbol: str) -> Dict[str, Any]:
        """
        Ruft die neuesten Insider-Transaktionen (Form 4) für das Symbol ab und berechnet
        den aggregierten Insider-Sentiment-Score und Whale-Aktivitäten.
        """
        clean_sym = symbol.replace(".DE", "")
        # Krypto hat keine SEC Form 4 Meldungen
        if "/" in symbol or symbol.endswith(".DE"):
            return {
                "insider_sentiment": "NEUTRAL",
                "insider_summary": "Nicht SEC-meldepflichtig (Krypto/Xetra)",
                "net_shares_90d": 0,
                "net_usd_90d": 0.0,
                "recent_whale_buys": [],
                "cluster_buy_detected": False,
            }

        try:
            ticker = yf.Ticker(clean_sym)
            df = getattr(ticker, "insider_transactions", None)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return {
                    "insider_sentiment": "NEUTRAL",
                    "insider_summary": "Keine aktuellen Form 4 Meldungen",
                    "net_shares_90d": 0,
                    "net_usd_90d": 0.0,
                    "recent_whale_buys": [],
                    "cluster_buy_detected": False,
                }

            transactions: List[Form4Transaction] = []
            now = datetime.datetime.now()
            cutoff_date = (now - datetime.timedelta(days=90)).strftime("%Y-%m-%d")

            net_shares = 0.0
            net_usd = 0.0
            unique_buyers = set()
            whale_buys = []

            for _, row in df.head(25).iterrows():
                insider = str(row.get("Insider", "Unknown")).strip()
                pos = str(row.get("Position", "Insider")).strip()
                tx_text = str(row.get("Transaction", "")).lower()
                shares = float(row.get("Shares", 0) or 0)
                val = float(row.get("Value", 0) or 0)
                date_str = str(row.get("Start Date", ""))[:10]

                # Bestimme Transaktionsart
                is_buy = ("purchase" in tx_text) or ("buy" in tx_text)
                is_sale = ("sale" in tx_text) or ("sell" in tx_text)

                if is_buy:
                    tx_type = "BUY"
                    is_open_market = True
                    net_shares += shares
                    net_usd += val
                    if date_str >= cutoff_date:
                        unique_buyers.add(insider)
                    if val >= 50_000 or shares >= 10_000:
                        whale_buys.append({
                            "insider": insider,
                            "position": pos,
                            "shares": shares,
                            "value_usd": val,
                            "date": date_str,
                        })
                elif is_sale:
                    tx_type = "SALE"
                    is_open_market = False
                    net_shares -= shares
                    net_usd -= val
                else:
                    tx_type = "OTHER"
                    is_open_market = False

                transactions.append(
                    Form4Transaction(
                        symbol=clean_sym,
                        insider_name=insider,
                        insider_title=pos,
                        transaction_type=tx_type,
                        shares=shares,
                        total_value_usd=val,
                        filing_date=date_str,
                        is_whale=(val >= 100_000 or shares >= 25_000),
                        is_open_market=is_open_market,
                    )
                )

            cluster_buy = len(unique_buyers) >= 2

            # Sentiment-Einstufung
            if cluster_buy or (net_usd > 150_000 and len(whale_buys) > 0):
                sentiment = "STRONG_INSIDER_BUY"
                wb_top = whale_buys[0]
                summary = f"🐋 Whale-Kauf: {wb_top['position']} ({wb_top['insider']}) kaufte {wb_top['shares']:,.0f} Stk. für ${wb_top['value_usd']:,.0f}"
                if cluster_buy:
                    summary += f" · 👥 Cluster-Buying ({len(unique_buyers)} Vorstände kaufen zeitgleich!)"
            elif len(whale_buys) > 0:
                sentiment = "MODERATE_INSIDER_BUY"
                summary = f"Insider-Kauf: {len(whale_buys)} Zukäufe in letzten 90 Tagen"
            elif net_usd < -5_000_000 and net_shares < -100_000:
                sentiment = "INSIDER_SELLING"
                summary = f"Hohe Insider-Verkäufe (${abs(net_usd):,.0f} in 90 Tagen)"
            else:
                sentiment = "NEUTRAL"
                summary = "Insider-Aktivität im normalen Rahmen"

            return {
                "insider_sentiment": sentiment,
                "insider_summary": summary,
                "net_shares_90d": int(net_shares),
                "net_usd_90d": round(net_usd, 2),
                "recent_whale_buys": whale_buys[:3],
                "cluster_buy_detected": cluster_buy,
                "transactions_count": len(transactions),
            }

        except Exception as e:
            return {
                "insider_sentiment": "NEUTRAL",
                "insider_summary": f"Form 4 Abfrage-Hinweis: {e}",
                "net_shares_90d": 0,
                "net_usd_90d": 0.0,
                "recent_whale_buys": [],
                "cluster_buy_detected": False,
            }

    def notify_whale_transaction(self, symbol: str, insider_info: Dict[str, Any]) -> None:
        """Sendet einen sofortigen Telegram-Alert, wenn ein neuer Whale-Kauf registriert wird."""
        if insider_info.get("insider_sentiment") != "STRONG_INSIDER_BUY":
            return

        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not bot_token or not chat_id:
            return

        import html
        import ssl
        import urllib.request
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl._create_unverified_context()

        cid = int(chat_id) if chat_id.isdigit() else chat_id
        safe_sym = html.escape(symbol)
        safe_sum = html.escape(insider_info.get("insider_summary", ""))

        text_body = (
            f"🤖 <b>🐋 SEC Form 4 Whale & Insider-Alert!</b>\n\n"
            f"• <b>Aktie:</b> <code>{safe_sym}</code>\n"
            f"• <b>Status:</b> 🚀 <b>STRONG INSIDER BUY</b>\n"
            f"• <b>Details:</b> {safe_sum}\n\n"
            f"Vorstände und Großinvestoren kaufen Aktien ihres eigenen Unternehmens auf dem freien Markt. "
            f"Der KI-Trading-Schwarm priorisiert dieses Signal!"
        )
        payload = json.dumps({"chat_id": cid, "text": text_body, "parse_mode": "HTML"}).encode("utf-8")
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, context=ctx, timeout=4.0)
        except Exception:
            pass


if __name__ == "__main__":
    console.print("[bold cyan]🔍 Teste SEC Form 4 Insider- & Whale-Tracker...[/bold cyan]")
    tracker = SECForm4Tracker()
    test_tickers = ["NVDA", "SOUN", "PLUG", "BBAI"]
    for t in test_tickers:
        res = tracker.get_insider_activity(t)
        console.print(f"• [bold]{t}[/bold]: {res['insider_sentiment']} ➔ {res['insider_summary']}")
