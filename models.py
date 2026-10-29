from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    title: str
    publisher: str
    published_at: str
    link: str = ""
    sentiment_hint: float = 0.0


class MacroRegime(BaseModel):
    vix_level: float
    spy_price: float
    spy_sma_50: float
    spy_sma_200: float
    us10y_yield: float = 4.20
    us10y_change_5d_bps: float = 0.0
    dxy_level: float = 103.50
    dxy_change_5d_pct: float = 0.0
    regime_state: Literal["GREEN_RISK_ON", "YELLOW_CAUTION", "RED_DEFENSIVE"]
    risk_multiplier: float = Field(description="Skaliertes Risikobudget (0.0 bis 1.0)")
    credit_spread_hyg_lqd: float = 0.75
    credit_stress_active: bool = False
    copper_gold_ratio: float = 0.12
    intermarket_bias: str = "NEUTRAL_STABLE"
    intermarket_signals: List[str] = Field(default_factory=list)
    summary: str


class PortfolioRiskMetrics(BaseModel):
    portfolio_beta: float = 0.0
    daily_var_95_usd: float = 0.0
    daily_var_95_pct: float = 0.0
    undiversified_var_95_usd: float = 0.0
    diversification_benefit_pct: float = 0.0
    top_correlated_pairs: List[str] = Field(default_factory=list)
    correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class MarketSnapshot(BaseModel):
    symbol: str
    name: str
    asset_class: Literal["EQUITY", "CRYPTO"]
    cluster: str = "US_TECH"
    intrinsic_leverage: float = 1.0
    is_safe_haven: bool = False
    current_price: float
    change_24h_pct: float
    change_7d_pct: float
    rsi_14: float
    sma_20: float
    sma_50: float
    sma_50_slope: float = 0.0
    macd_hist: float
    atr_14: float
    volatility_annualized_pct: float
    volume_ratio_vs_20d: float
    # Cross-Asset Korrelation, Beta & Lead-Lag / Inverse-Arbitrage
    beta_vs_spy: float = 1.0
    max_corr_with_portfolio: float = 0.0
    most_correlated_holding: str = "-"
    cross_asset_opportunity_score: float = 0.0
    cross_asset_thesis: str = ""
    # Fundamental- & Earnings-Daten (für Aktien)
    forward_pe: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    profit_margin_pct: Optional[float] = None
    days_to_earnings: Optional[int] = None
    # Options-Markt-Radar (Put/Call-Volumen-Ratio, Implizite Volatilität, Top-Kontrakt & Smart-Money Options-Bias)
    options_put_call_ratio: Optional[float] = None
    options_implied_vol_pct: Optional[float] = None
    options_flow_sentiment: str = "NEUTRAL"  # "BULLISH_CALL_FLOW", "BEARISH_PUT_FLOW", "NEUTRAL"
    best_option_contract: str = ""
    unusual_options_vol_oi: float = 0.0
    radar_zone: str = "HOT_ACTION"  # "HOT_ACTION", "PROMOTED_FROM_WATCH", "WATCH_CLOSELY"
    # Intraday Multi-Timeframe (15m VWAP & Trend) + Relative Strength vs. SPY
    intraday_vwap: Optional[float] = None
    intraday_15m_trend: str = "NEUTRAL"  # "BULLISH_ABOVE_VWAP", "BEARISH_BELOW_VWAP", "FADING_FROM_HIGH", "NEUTRAL"
    relative_strength_vs_spy: float = 0.0
    # Erweiterte Quantitative & Insider Werkzeuge (agent_tools.py)
    ttm_squeeze_status: str = "NORMAL"  # "🔥 SQUEEZE_FIRED", "⏳ SQUEEZE_COMPRESSION", "NORMAL"
    ttm_squeeze_momentum: float = 0.0
    insider_sentiment: str = "NEUTRAL"  # "STRONG_INSIDER_BUY", "INSIDER_SELLING", "NEUTRAL"
    insider_summary: str = ""
    form4_net_usd_90d: float = 0.0
    form4_cluster_buy: bool = False
    form4_recent_whale_buys: List[Dict[str, Any]] = Field(default_factory=list)
    # Erweiterte Optionen- und Fundamental-Felder
    options_max_pain: Optional[float] = None
    options_flow_score: float = 0.0
    fundamental_health_grade: str = "SOLID_VALUE"
    is_fundamentally_distressed: bool = False
    fundamental_score: float = 0.0
    # Statistische Arbitrage & Pairs Trading Felder
    stat_arb_pair: str = ""
    stat_arb_z_score: Optional[float] = None
    stat_arb_thesis: str = ""
    # Fraktale Marktanalyse & Mikrostruktur-Felder
    hurst_exponent: float = 0.50
    fractal_regime_mode: str = "RANDOM_WALK_CHOP"
    bid_ask_spread_pct: float = 0.10
    amihud_illiquidity: float = 0.0
    microstructure_routing: str = "MARKET_EXECUTION_SAFE"
    supertrend_direction: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    supertrend_val: float = 0.0
    supertrend_flipped: bool = False
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    sec_form8k_catalyst: str = ""
    option_delta: Optional[float] = None
    option_gamma: Optional[float] = None
    option_vega: Optional[float] = None
    news: List[NewsItem] = Field(default_factory=list)


class AdversarialDebateResult(BaseModel):
    """
    Ergebnis der Adversarial LLM-Debatte vor Trade-Freigabe (Red-Teaming Tribunal).
    Bulle (Alpha-Hunter) vs. Bär (Teufels Advokat / Red Team) vor dem CIO-Richter.
    """
    symbol: str
    action_debated: str = "BUY"
    verdict: Literal["APPROVED", "REDUCED_SIZE", "REJECTED_VETO"] = "APPROVED"
    bull_argument: str = ""
    bear_rebuttal: str = ""
    cio_reasoning: str = ""
    risk_score: float = Field(default=3.0, ge=0.0, le=10.0, description="Risiko-Score von 0 (harmlos) bis 10 (Falle)")
    size_multiplier: float = 1.0
    debated_at: str = ""


class TradingSignal(BaseModel):
    symbol: str
    asset_class: Literal["EQUITY", "CRYPTO"]
    cluster: str = "US_TECH"
    intrinsic_leverage: float = 1.0
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0, description="Konfidenz zwischen 0.0 und 1.0")
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Aggregiertes News/Markt-Sentiment")
    entry_price: float
    stop_loss_price: float
    partial_take_profit_price: float = 0.0
    take_profit_price: float
    risk_reward_ratio: float
    expected_edge_score: float = 0.0
    catalyst_summary: str = Field(description="Wichtigste News- oder Technik-Treiber")
    thesis: str = Field(description="Begründung für die Handelsentscheidung")
    best_option_contract: str = ""
    analysis_source: str = "AI_HYBRID"
    debate_result: Optional[AdversarialDebateResult] = None

    def model_post_init(self, __context) -> None:
        if self.partial_take_profit_price <= 0 and self.take_profit_price > self.entry_price:
            self.partial_take_profit_price = round(
                self.entry_price + 0.45 * (self.take_profit_price - self.entry_price), 4
            )
        if self.expected_edge_score <= 0:
            self.expected_edge_score = round(self.confidence * self.risk_reward_ratio, 3)


class RiskEvaluation(BaseModel):
    approved: bool
    reason: str
    adjusted_action: Literal["BUY", "SELL", "HOLD"]
    leverage_used: float = 1.0
    effective_total_leverage: float = 1.0
    recommended_quantity: float = 0.0
    margin_required_usd: float = 0.0
    notional_value_usd: float = 0.0
    max_loss_usd: float = 0.0
    portfolio_risk_pct: float = 0.0
    debate_result: Optional[AdversarialDebateResult] = None


class Position(BaseModel):
    symbol: str
    asset_class: Literal["EQUITY", "CRYPTO"]
    cluster: str = "US_TECH"
    side: Literal["LONG", "SHORT"] = "LONG"
    quantity: float
    leverage: float = 1.0
    intrinsic_leverage: float = 1.0
    margin_used_usd: float = 0.0
    avg_entry_price: float
    current_price: float
    highest_price: float = 0.0
    stop_loss_price: float
    partial_tp_price: float = 0.0
    partial_tp_done: bool = False
    take_profit_price: float
    opened_at: str

    def model_post_init(self, __context) -> None:
        if self.highest_price <= 0:
            self.highest_price = (
                min(self.avg_entry_price, self.current_price)
                if self.side == "SHORT"
                else max(self.avg_entry_price, self.current_price)
            )
        if self.partial_tp_price <= 0:
            if self.side == "SHORT" and self.take_profit_price < self.avg_entry_price:
                self.partial_tp_price = round(
                    self.avg_entry_price - 0.45 * (self.avg_entry_price - self.take_profit_price), 4
                )
            elif self.take_profit_price > self.avg_entry_price:
                self.partial_tp_price = round(
                    self.avg_entry_price + 0.45 * (self.take_profit_price - self.avg_entry_price), 4
                )
        if self.margin_used_usd <= 0:
            self.margin_used_usd = round((abs(self.quantity) * self.avg_entry_price) / max(1.0, self.leverage), 2)

    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.current_price

    @property
    def market_value(self) -> float:
        """Eigenkapital-Wert (Margin + Unrealisierter Gewinn/Verlust) dieser Position."""
        return round(self.margin_used_usd + self.unrealized_pnl, 2)

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "SHORT":
            return (self.avg_entry_price - self.current_price) * abs(self.quantity)
        return (self.current_price - self.avg_entry_price) * abs(self.quantity)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Rendite bezogen auf dein eingesetztes Eigenkapital (Margin) – inklusive Hebel-Effekt!"""
        if self.margin_used_usd <= 0:
            return 0.0
        return (self.unrealized_pnl / self.margin_used_usd) * 100.0
