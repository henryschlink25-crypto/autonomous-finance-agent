import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class WatchlistItem:
    symbol: str
    yf_ticker: str
    asset_class: str  # "EQUITY" oder "CRYPTO"
    cluster: str      # "LEVERAGED_3X", "US_TECH", "SAFE_HAVEN", "CRYPTO"
    name: str
    intrinsic_leverage: float = 1.0  # 3.0 bei TQQQ/SOXL/SQQQ, 1.0 bei normalen Assets
    is_safe_haven: bool = False


@dataclass
class AgentConfig:
    # API & Mode Settings (EXECUTION_MODE: LOCAL_PAPER, ALPACA_PAPER, IBKR, CCXT_SANDBOX)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    execution_mode: str = os.getenv("EXECUTION_MODE", "LOCAL_PAPER")

    # Interactive Brokers (IBKR TWS / IB Gateway) Settings
    ibkr_host: str = os.getenv("IBKR_HOST", "127.0.0.1")
    ibkr_port: int = int(os.getenv("IBKR_PORT", "7497"))  # 7497 = TWS Paper, 7496 = TWS Live, 4002 = Gateway Paper, 4001 = Gateway Live
    ibkr_client_id: int = int(os.getenv("IBKR_CLIENT_ID", "1"))
    ibkr_account: str = os.getenv("IBKR_ACCOUNT", "")
    ibkr_auto_connect: bool = os.getenv("IBKR_AUTO_CONNECT", "True").lower() == "true"
    ibkr_enable_options_trading: bool = os.getenv("IBKR_ENABLE_OPTIONS_TRADING", "False").lower() == "true"

    # Alpaca Settings
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_paper: bool = os.getenv("ALPACA_PAPER", "True").lower() == "true"

    # CCXT Settings
    ccxt_exchange_id: str = os.getenv("CCXT_EXCHANGE_ID", "binance")
    ccxt_api_key: str = os.getenv("CCXT_API_KEY", "")
    ccxt_secret_key: str = os.getenv("CCXT_SECRET_KEY", "")
    ccxt_sandbox: bool = os.getenv("CCXT_SANDBOX", "True").lower() == "true"

    # Kapital- & Smart-Leverage-Parameter (Synchronisiert mit den vollen $100.000 / 4x Buying Power auf Alpaca Paper!)
    initial_capital_usd: float = float(os.getenv("INITIAL_CAPITAL_USD", "100000.0"))
    smart_leverage_max: float = float(os.getenv("SMART_LEVERAGE_MAX", "2.0"))
    max_risk_per_trade_pct: float = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "0.025"))
    max_position_weight_pct: float = float(os.getenv("MAX_POSITION_WEIGHT_PCT", "0.30"))
    max_cluster_weight_pct: float = float(os.getenv("MAX_CLUSTER_WEIGHT_PCT", "0.60"))
    min_cash_reserve_pct: float = float(os.getenv("MIN_CASH_RESERVE_PCT", "0.10"))
    max_daily_drawdown_pct: float = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.05"))
    max_pairwise_correlation: float = float(os.getenv("MAX_PAIRWISE_CORRELATION", "0.84"))
    max_portfolio_var_95_pct: float = float(os.getenv("MAX_PORTFOLIO_VAR_95_PCT", "0.035"))
    min_signal_confidence: float = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.74"))
    min_risk_reward_ratio: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.8"))
    min_expected_edge_score: float = float(os.getenv("MIN_EXPECTED_EDGE_SCORE", "1.48"))
    earnings_blackout_days: int = int(os.getenv("EARNINGS_BLACKOUT_DAYS", "3"))
    enforce_market_hours: bool = os.getenv("ENFORCE_MARKET_HOURS", "True").lower() == "true"
    opening_bell_cooldown_minutes: int = int(os.getenv("OPENING_BELL_COOLDOWN_MINUTES", "15"))
    enable_short_selling: bool = os.getenv("ENABLE_SHORT_SELLING", "True").lower() == "true"
    min_short_confidence: float = float(os.getenv("MIN_SHORT_CONFIDENCE", "0.85"))
    enable_opportunity_rotation: bool = os.getenv("ENABLE_OPPORTUNITY_ROTATION", "True").lower() == "true"
    max_stale_holding_days: int = int(os.getenv("MAX_STALE_HOLDING_DAYS", "5"))
    leveraged_etf_max_days: int = int(os.getenv("LEVERAGED_ETF_MAX_DAYS", "3"))
    enable_adversarial_red_team: bool = os.getenv("ENABLE_ADVERSARIAL_RED_TEAM", "True").lower() == "true"
    min_order_usd: float = 25.0

    # 1.000 € Multiplikator & Asymmetrisches Compounding-Modell
    enable_asymmetric_compounding: bool = os.getenv("ENABLE_ASYMMETRIC_COMPOUNDING", "True").lower() == "true"
    base_principal_capital_usd: float = float(os.getenv("BASE_PRINCIPAL_CAPITAL_USD", "1000.0"))
    profit_reinvestment_rate: float = float(os.getenv("PROFIT_REINVESTMENT_RATE", "0.35"))  # 35% der Reingewinne für vergrößerte Spitzen-Trades reinvestieren

    # Micro-Dip Limit-Sniper (Slippage-Eliminierung & Rabatt-Einstiege)
    enable_micro_dip_sniper: bool = os.getenv("ENABLE_MICRO_DIP_SNIPER", "True").lower() == "true"

    # Optionale Echtzeit-Push-Alerts (Telegram Bot, Discord Webhook oder WhatsApp)
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    telegram_alert_level: str = os.getenv("TELEGRAM_ALERT_LEVEL", "BALANCED")  # "TRADES_ONLY" | "BALANCED" | "ALL"
    telegram_max_alerts_per_hour: int = int(os.getenv("TELEGRAM_MAX_ALERTS_PER_HOUR", "8"))
    telegram_dedup_window_seconds: int = int(os.getenv("TELEGRAM_DEDUP_WINDOW_SECONDS", "3600"))
    telegram_instance_tag: str = os.getenv("TELEGRAM_INSTANCE_TAG", "🤖 Autonomous Agent")
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    whatsapp_phone_number: str = os.getenv("WHATSAPP_PHONE_NUMBER", "")
    whatsapp_api_key: str = os.getenv("WHATSAPP_API_KEY", "")

    # Storage paths
    portfolio_state_file: str = os.path.join(os.path.dirname(__file__), "data", "portfolio_state.json")
    audit_log_file: str = os.path.join(os.path.dirname(__file__), "data", "audit_journal.jsonl")

    # Dynamische, selbstbereinigende Watchlist & Markt-Scanner-Parameter:
    # - Aktien ohne interessante Aktivität (kein Volumen-Schub, kein Momentum, keine News) werden automatisch aussortiert!
    # - Dafür scannt der Agent live den Gesamtmarkt nach aktuellen Ausbrechern (" Stocks in Play").
    enable_dynamic_anomaly_scanner: bool = True
    max_dynamic_anomaly_tickers: int = 75
    max_active_watchlist_size: int = int(os.getenv("MAX_ACTIVE_WATCHLIST_SIZE", "45"))
    min_activity_score_to_keep: float = 0.25
    dynamic_watchlist_file: str = os.path.join(os.path.dirname(__file__), "data", "dynamic_watchlist.json")
    watchlist: List[WatchlistItem] = field(default_factory=lambda: [
        # 0. Langfristige Core-ETFs & Makro-Anker (Diversifikation & stetiger Vermögensaufbau)
        WatchlistItem(symbol="SPY", yf_ticker="SPY", asset_class="EQUITY", cluster="CORE_ETF", name="SPDR S&P 500 ETF Trust (Core Anchor)"),
        WatchlistItem(symbol="QQQ", yf_ticker="QQQ", asset_class="EQUITY", cluster="CORE_ETF", name="Invesco QQQ Trust (Nasdaq-100 Core Anchor)"),
        WatchlistItem(symbol="TLT", yf_ticker="TLT", asset_class="EQUITY", cluster="SAFE_HAVEN", name="iShares 20+ Year Treasury Bond ETF", is_safe_haven=True),
        # 1. 3x Gehebelte Long- & Short-ETFs (US, Halbleiter, S&P 500, Biotech, China & Crash-Hedges)
        WatchlistItem(symbol="SOXL", yf_ticker="SOXL", asset_class="EQUITY", cluster="LEVERAGED_3X", name="3x Long Semiconductor ETF", intrinsic_leverage=3.0),
        WatchlistItem(symbol="TQQQ", yf_ticker="TQQQ", asset_class="EQUITY", cluster="LEVERAGED_3X", name="3x Long Nasdaq-100 ETF", intrinsic_leverage=3.0),
        WatchlistItem(symbol="SPXL", yf_ticker="SPXL", asset_class="EQUITY", cluster="LEVERAGED_3X", name="3x Long S&P 500 ETF", intrinsic_leverage=3.0),
        WatchlistItem(symbol="LABU", yf_ticker="LABU", asset_class="EQUITY", cluster="LEVERAGED_3X", name="3x Long S&P Biotech ETF", intrinsic_leverage=3.0),
        WatchlistItem(symbol="YINN", yf_ticker="YINN", asset_class="EQUITY", cluster="CHINA_TECH", name="3x Long FTSE China Bull ETF", intrinsic_leverage=3.0),
        WatchlistItem(symbol="SQQQ", yf_ticker="SQQQ", asset_class="EQUITY", cluster="SAFE_HAVEN", name="3x Short Nasdaq Crash-Hedge ETF", intrinsic_leverage=3.0, is_safe_haven=True),
        WatchlistItem(symbol="SOXS", yf_ticker="SOXS", asset_class="EQUITY", cluster="SAFE_HAVEN", name="3x Short Semiconductor Crash-Hedge ETF", intrinsic_leverage=3.0, is_safe_haven=True),
        # 2. US-Tech, KI, Cloud, Cyber & Halbleiter-Universum
        WatchlistItem(symbol="NVDA", yf_ticker="NVDA", asset_class="EQUITY", cluster="US_TECH", name="NVIDIA Corp."),
        WatchlistItem(symbol="AVGO", yf_ticker="AVGO", asset_class="EQUITY", cluster="US_TECH", name="Broadcom Inc."),
        WatchlistItem(symbol="AMD", yf_ticker="AMD", asset_class="EQUITY", cluster="US_TECH", name="Advanced Micro Devices"),
        WatchlistItem(symbol="TSM", yf_ticker="TSM", asset_class="EQUITY", cluster="US_TECH", name="Taiwan Semiconductor (TSMC)"),
        WatchlistItem(symbol="MU", yf_ticker="MU", asset_class="EQUITY", cluster="US_TECH", name="Micron Technology (HBM Memory)"),
        WatchlistItem(symbol="MRVL", yf_ticker="MRVL", asset_class="EQUITY", cluster="US_TECH", name="Marvell Technology (Custom AI Silicon)"),
        WatchlistItem(symbol="ARM", yf_ticker="ARM", asset_class="EQUITY", cluster="US_TECH", name="Arm Holdings plc"),
        WatchlistItem(symbol="SMCI", yf_ticker="SMCI", asset_class="EQUITY", cluster="US_TECH", name="Super Micro Computer"),
        WatchlistItem(symbol="VRT", yf_ticker="VRT", asset_class="EQUITY", cluster="US_TECH", name="Vertiv Holdings (AI Data Center Cooling)"),
        WatchlistItem(symbol="ANET", yf_ticker="ANET", asset_class="EQUITY", cluster="US_TECH", name="Arista Networks"),
        WatchlistItem(symbol="LRCX", yf_ticker="LRCX", asset_class="EQUITY", cluster="US_TECH", name="Lam Research"),
        WatchlistItem(symbol="KLAC", yf_ticker="KLAC", asset_class="EQUITY", cluster="US_TECH", name="KLA Corp."),
        WatchlistItem(symbol="AMAT", yf_ticker="AMAT", asset_class="EQUITY", cluster="US_TECH", name="Applied Materials"),
        WatchlistItem(symbol="CRWD", yf_ticker="CRWD", asset_class="EQUITY", cluster="US_TECH", name="CrowdStrike Holdings (Cybersecurity)"),
        WatchlistItem(symbol="PANW", yf_ticker="PANW", asset_class="EQUITY", cluster="US_TECH", name="Palo Alto Networks"),
        WatchlistItem(symbol="SNOW", yf_ticker="SNOW", asset_class="EQUITY", cluster="US_TECH", name="Snowflake Inc."),
        WatchlistItem(symbol="NOW", yf_ticker="NOW", asset_class="EQUITY", cluster="US_TECH", name="ServiceNow Inc."),
        WatchlistItem(symbol="CRM", yf_ticker="CRM", asset_class="EQUITY", cluster="US_TECH", name="Salesforce Inc."),
        WatchlistItem(symbol="APP", yf_ticker="APP", asset_class="EQUITY", cluster="US_TECH", name="AppLovin Corp. (AI AdTech)"),
        WatchlistItem(symbol="AAPL", yf_ticker="AAPL", asset_class="EQUITY", cluster="US_TECH", name="Apple Inc."),
        WatchlistItem(symbol="MSFT", yf_ticker="MSFT", asset_class="EQUITY", cluster="US_TECH", name="Microsoft Corp."),
        WatchlistItem(symbol="META", yf_ticker="META", asset_class="EQUITY", cluster="US_TECH", name="Meta Platforms Inc."),
        WatchlistItem(symbol="GOOGL", yf_ticker="GOOGL", asset_class="EQUITY", cluster="US_TECH", name="Alphabet Inc. (Google)"),
        WatchlistItem(symbol="AMZN", yf_ticker="AMZN", asset_class="EQUITY", cluster="US_TECH", name="Amazon.com Inc."),
        WatchlistItem(symbol="PLTR", yf_ticker="PLTR", asset_class="EQUITY", cluster="US_TECH", name="Palantir Technologies (AI Defense)"),
        # 3. High-Beta Momentum & Krypto-Proxies (4)
        WatchlistItem(symbol="TSLA", yf_ticker="TSLA", asset_class="EQUITY", cluster="HIGH_BETA", name="Tesla Inc."),
        WatchlistItem(symbol="COIN", yf_ticker="COIN", asset_class="EQUITY", cluster="HIGH_BETA", name="Coinbase Global"),
        WatchlistItem(symbol="MSTR", yf_ticker="MSTR", asset_class="EQUITY", cluster="HIGH_BETA", name="Strategy (MicroStrategy Bitcoin Proxy)"),
        WatchlistItem(symbol="HOOD", yf_ticker="HOOD", asset_class="EQUITY", cluster="HIGH_BETA", name="Robinhood Markets"),
        WatchlistItem(symbol="SOFI", yf_ticker="SOFI", asset_class="EQUITY", cluster="HIGH_BETA", name="SoFi Technologies"),
        WatchlistItem(symbol="SHOP", yf_ticker="SHOP", asset_class="EQUITY", cluster="HIGH_BETA", name="Shopify Inc."),
        WatchlistItem(symbol="UBER", yf_ticker="UBER", asset_class="EQUITY", cluster="HIGH_BETA", name="Uber Technologies"),
        WatchlistItem(symbol="IONQ", yf_ticker="IONQ", asset_class="EQUITY", cluster="HIGH_BETA", name="IonQ Inc. (Quantum Computing)"),
        WatchlistItem(symbol="RGTI", yf_ticker="RGTI", asset_class="EQUITY", cluster="HIGH_BETA", name="Rigetti Computing"),
        WatchlistItem(symbol="RKLB", yf_ticker="RKLB", asset_class="EQUITY", cluster="HIGH_BETA", name="Rocket Lab USA (Space)"),
        WatchlistItem(symbol="ASTS", yf_ticker="ASTS", asset_class="EQUITY", cluster="HIGH_BETA", name="AST SpaceMobile (Direct-to-Cell)"),
        WatchlistItem(symbol="HIMS", yf_ticker="HIMS", asset_class="EQUITY", cluster="HIGH_BETA", name="Hims & Hers Health"),
        WatchlistItem(symbol="CAVA", yf_ticker="CAVA", asset_class="EQUITY", cluster="HIGH_BETA", name="CAVA Group Inc."),
        WatchlistItem(symbol="DKNG", yf_ticker="DKNG", asset_class="EQUITY", cluster="HIGH_BETA", name="DraftKings Inc."),
        WatchlistItem(symbol="CELH", yf_ticker="CELH", asset_class="EQUITY", cluster="HIGH_BETA", name="Celsius Holdings"),
        WatchlistItem(symbol="RDDT", yf_ticker="RDDT", asset_class="EQUITY", cluster="HIGH_BETA", name="Reddit Inc."),
        WatchlistItem(symbol="NET", yf_ticker="NET", asset_class="EQUITY", cluster="US_TECH", name="Cloudflare Inc."),
        WatchlistItem(symbol="NFLX", yf_ticker="NFLX", asset_class="EQUITY", cluster="US_TECH", name="Netflix Inc."),
        # 4. Chinesische High-Growth & E-Mobility Leader
        WatchlistItem(symbol="BABA", yf_ticker="BABA", asset_class="EQUITY", cluster="CHINA_TECH", name="Alibaba Group (China Cloud/AI)"),
        WatchlistItem(symbol="PDD", yf_ticker="PDD", asset_class="EQUITY", cluster="CHINA_TECH", name="PDD Holdings / Temu"),
        WatchlistItem(symbol="BIDU", yf_ticker="BIDU", asset_class="EQUITY", cluster="CHINA_TECH", name="Baidu Inc. (China AI & Robotaxi)"),
        WatchlistItem(symbol="JD", yf_ticker="JD", asset_class="EQUITY", cluster="CHINA_TECH", name="JD.com Inc."),
        WatchlistItem(symbol="NIO", yf_ticker="NIO", asset_class="EQUITY", cluster="CHINA_TECH", name="NIO Inc. (China EV)"),
        WatchlistItem(symbol="XPEV", yf_ticker="XPEV", asset_class="EQUITY", cluster="CHINA_TECH", name="XPeng Inc. (China Smart EV)"),
        WatchlistItem(symbol="LI", yf_ticker="LI", asset_class="EQUITY", cluster="CHINA_TECH", name="Li Auto Inc."),
        WatchlistItem(symbol="FUTU", yf_ticker="FUTU", asset_class="EQUITY", cluster="CHINA_TECH", name="Futu Holdings (Asia FinTech)"),
        WatchlistItem(symbol="TCOM", yf_ticker="TCOM", asset_class="EQUITY", cluster="CHINA_TECH", name="Trip.com Group"),
        WatchlistItem(symbol="SE", yf_ticker="SE", asset_class="EQUITY", cluster="CHINA_TECH", name="Sea Limited (Southeast Asia Tech)"),
        # 5. Europäische & Deutsche DAX-40 / MDAX Marktführer
        WatchlistItem(symbol="ASML", yf_ticker="ASML", asset_class="EQUITY", cluster="EU_LEADERS", name="ASML Holding (EU Chip-Monopol)"),
        WatchlistItem(symbol="SAP", yf_ticker="SAP", asset_class="EQUITY", cluster="EU_LEADERS", name="SAP SE (EU Enterprise Cloud)"),
        WatchlistItem(symbol="NVO", yf_ticker="NVO", asset_class="EQUITY", cluster="EU_LEADERS", name="Novo Nordisk (EU Biotech Leader)"),
        WatchlistItem(symbol="RHM.DE", yf_ticker="RHM.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="Rheinmetall AG (Xetra Defense)"),
        WatchlistItem(symbol="SIE.DE", yf_ticker="SIE.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="Siemens AG (Xetra Industrial Tech)"),
        WatchlistItem(symbol="ENR.DE", yf_ticker="ENR.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="Siemens Energy AG (Xetra Power Grid)"),
        WatchlistItem(symbol="IFX.DE", yf_ticker="IFX.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="Infineon Technologies AG (Xetra Semis)"),
        WatchlistItem(symbol="ALV.DE", yf_ticker="ALV.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="Allianz SE (Xetra Insurance)"),
        WatchlistItem(symbol="DTE.DE", yf_ticker="DTE.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="Deutsche Telekom AG"),
        WatchlistItem(symbol="AIR.DE", yf_ticker="AIR.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="Airbus SE (Xetra Aerospace)"),
        WatchlistItem(symbol="MTX.DE", yf_ticker="MTX.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="MTU Aero Engines AG"),
        WatchlistItem(symbol="CBK.DE", yf_ticker="CBK.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="Commerzbank AG"),
        WatchlistItem(symbol="DBK.DE", yf_ticker="DBK.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="Deutsche Bank AG"),
        WatchlistItem(symbol="ADS.DE", yf_ticker="ADS.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="adidas AG"),
        WatchlistItem(symbol="MBG.DE", yf_ticker="MBG.DE", asset_class="EQUITY", cluster="EU_LEADERS", name="Mercedes-Benz Group AG"),
        # 6. US-Biotech, Pharma, Finanzen, Rüstung & Energie-Rotation
        WatchlistItem(symbol="JPM", yf_ticker="JPM", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="JPMorgan Chase & Co."),
        WatchlistItem(symbol="GS", yf_ticker="GS", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Goldman Sachs Group"),
        WatchlistItem(symbol="LLY", yf_ticker="LLY", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Eli Lilly & Co. (Pharma)"),
        WatchlistItem(symbol="VRTX", yf_ticker="VRTX", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Vertex Pharmaceuticals"),
        WatchlistItem(symbol="REGN", yf_ticker="REGN", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Regeneron Pharmaceuticals"),
        WatchlistItem(symbol="ISRG", yf_ticker="ISRG", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Intuitive Surgical (MedTech Robotics)"),
        WatchlistItem(symbol="MRNA", yf_ticker="MRNA", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Moderna Inc."),
        WatchlistItem(symbol="LMT", yf_ticker="LMT", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Lockheed Martin (Defense)"),
        WatchlistItem(symbol="RTX", yf_ticker="RTX", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="RTX Corp. (Aerospace & Defense)"),
        WatchlistItem(symbol="GE", yf_ticker="GE", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="GE Aerospace"),
        WatchlistItem(symbol="CAT", yf_ticker="CAT", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Caterpillar Inc."),
        WatchlistItem(symbol="CCJ", yf_ticker="CCJ", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Cameco Corp. (Uranium & Nuclear)"),
        WatchlistItem(symbol="FCX", yf_ticker="FCX", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Freeport-McMoRan (Copper)"),
        WatchlistItem(symbol="XOM", yf_ticker="XOM", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Exxon Mobil Corp. (Energy)"),
        WatchlistItem(symbol="CVX", yf_ticker="CVX", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Chevron Corp. (Energy)"),
        WatchlistItem(symbol="XLE", yf_ticker="XLE", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="Energy Select Sector SPDR Fund"),
        # 7. Rohstoffe & Safe-Haven Hedges (3)
        WatchlistItem(symbol="GLD", yf_ticker="GLD", asset_class="EQUITY", cluster="SAFE_HAVEN", name="SPDR Gold Trust (Safe Haven)", is_safe_haven=True),
        WatchlistItem(symbol="SLV", yf_ticker="SLV", asset_class="EQUITY", cluster="SAFE_HAVEN", name="iShares Silver Trust", is_safe_haven=True),
        WatchlistItem(symbol="USO", yf_ticker="USO", asset_class="EQUITY", cluster="VALUE_CYCLICAL", name="United States Oil Fund (WTI Crude)"),
        # 8. 24/7 Krypto-Leader (5)
        WatchlistItem(symbol="BTC/USD", yf_ticker="BTC-USD", asset_class="CRYPTO", cluster="CRYPTO", name="Bitcoin"),
        WatchlistItem(symbol="ETH/USD", yf_ticker="ETH-USD", asset_class="CRYPTO", cluster="CRYPTO", name="Ethereum"),
        WatchlistItem(symbol="SOL/USD", yf_ticker="SOL-USD", asset_class="CRYPTO", cluster="CRYPTO", name="Solana"),
        WatchlistItem(symbol="XRP/USD", yf_ticker="XRP-USD", asset_class="CRYPTO", cluster="CRYPTO", name="Ripple XRP"),
        WatchlistItem(symbol="DOGE/USD", yf_ticker="DOGE-USD", asset_class="CRYPTO", cluster="CRYPTO", name="Dogecoin"),
        # 9. Penny Stocks & High-Momentum Micro-Caps (Alpaca Paper Trading)
        WatchlistItem(symbol="SOUN", yf_ticker="SOUN", asset_class="EQUITY", cluster="PENNY_STOCKS", name="SoundHound AI (AI Voice Microcap)"),
        WatchlistItem(symbol="BBAI", yf_ticker="BBAI", asset_class="EQUITY", cluster="PENNY_STOCKS", name="BigBear.ai Holdings (Defense AI)"),
        WatchlistItem(symbol="LUNR", yf_ticker="LUNR", asset_class="EQUITY", cluster="PENNY_STOCKS", name="Intuitive Machines (Space Exploration)"),
        WatchlistItem(symbol="ACHR", yf_ticker="ACHR", asset_class="EQUITY", cluster="PENNY_STOCKS", name="Archer Aviation (eVTOL Air Taxi)"),
        WatchlistItem(symbol="OPEN", yf_ticker="OPEN", asset_class="EQUITY", cluster="PENNY_STOCKS", name="Opendoor Technologies (Real Estate Tech)"),
        WatchlistItem(symbol="CLOV", yf_ticker="CLOV", asset_class="EQUITY", cluster="PENNY_STOCKS", name="Clover Health Investments"),
        WatchlistItem(symbol="PLUG", yf_ticker="PLUG", asset_class="EQUITY", cluster="PENNY_STOCKS", name="Plug Power (Hydrogen Energy)"),
    ])

    @property
    def is_live_trading(self) -> bool:
        """
        Prüft, ob der Agent mit echtem Geld (Live Brokerage) handelt.
        - True: Live-Echtgeld-Konto (Alpaca Live, IBKR Live Port 7496/4001, CCXT Live)
        - False: Paper-Trading / Sandbox / Simulation (Experimentier-Modus)
        """
        if self.execution_mode == "IBKR" and self.ibkr_port in (7496, 4001):
            return True
        if "ALPACA" in self.execution_mode and not self.alpaca_paper:
            return True
        if self.execution_mode == "CCXT" and not self.ccxt_sandbox:
            return True
        return False


