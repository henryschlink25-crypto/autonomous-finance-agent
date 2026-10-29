import datetime
from typing import Dict, Optional, Tuple
import pytz
from config import AgentConfig
from models import MacroRegime, MarketSnapshot, PortfolioRiskMetrics, Position, RiskEvaluation, TradingSignal


def is_us_equity_market_open(now_utc: Optional[datetime.datetime] = None, symbol: str = "") -> Tuple[bool, str]:
    """
    Prüft exakt anhand der Börsenzeit, ob der jeweilige Heimatmarkt geöffnet ist:
    - Europäische Xetra-Aktien (.DE): Montag bis Freitag, 09:00 bis 17:30 Uhr Europe/Berlin
    - US-Aktien, US-ETFs & China/EU-ADRs (NYSE/Nasdaq): Montag bis Freitag, 09:30 bis 16:00 Uhr America/New_York
    - Krypto handelt 24/7.
    """
    if now_utc is None:
        now_utc = datetime.datetime.now(datetime.timezone.utc)

    if symbol.endswith(".DE"):
        de_tz = pytz.timezone("Europe/Berlin")
        de_now = now_utc.astimezone(de_tz)
        if de_now.weekday() >= 5:
            return False, f"Wochenende (Xetra: {de_now.strftime('%a %H:%M')} MEZ)"
        if datetime.time(9, 0) <= de_now.time() <= datetime.time(17, 30):
            return True, f"Xetra/Frankfurt OFFEN ({de_now.strftime('%H:%M')} MEZ)"
        return False, f"Xetra geschlossen ({de_now.strftime('%H:%M')} MEZ, offen 09:00-17:30)"

    ny_tz = pytz.timezone("America/New_York")
    ny_now = now_utc.astimezone(ny_tz)
    weekday = ny_now.weekday()  # 0 = Montag, 6 = Sonntag
    ny_time = ny_now.time()

    if weekday >= 5:
        return False, f"Wochenende (NY: {ny_now.strftime('%a %H:%M')} ET)"

    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)
    if market_open <= ny_time <= market_close:
        return True, f"NYSE/Nasdaq OFFEN (NY: {ny_now.strftime('%H:%M')} ET)"
    return False, f"NYSE/Nasdaq geschlossen (NY: {ny_now.strftime('%H:%M')} ET, offen 09:30-16:00 ET)"


def is_opening_bell_cooldown(
    now_utc: Optional[datetime.datetime] = None, symbol: str = "", cooldown_minutes: int = 15
) -> Tuple[bool, str]:
    """
    Prüft, ob sich die Börse in den ersten X Minuten nach Eröffnung befindet ('Opening Bell Chaos').
    Verhindert Neukäufe in den ersten 15 Minuten (09:30-09:45 ET bzw. 09:00-09:15 MEZ),
    während die Bid-Ask-Spreads weit sind und Fehlausbrüche ('Opening Fakeouts') stattfinden.
    """
    if cooldown_minutes <= 0:
        return False, ""
    if now_utc is None:
        now_utc = datetime.datetime.now(datetime.timezone.utc)

    if symbol.endswith(".DE"):
        de_tz = pytz.timezone("Europe/Berlin")
        de_now = now_utc.astimezone(de_tz)
        if de_now.weekday() < 5:
            open_dt = de_now.replace(hour=9, minute=0, second=0, microsecond=0)
            diff_min = (de_now - open_dt).total_seconds() / 60.0
            if 0 <= diff_min < cooldown_minutes:
                return True, f"Xetra Opening-Range ({int(diff_min)}/{cooldown_minutes} Min nach Eröffnung)"
        return False, ""

    ny_tz = pytz.timezone("America/New_York")
    ny_now = now_utc.astimezone(ny_tz)
    if ny_now.weekday() < 5:
        open_dt = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)
        diff_min = (ny_now - open_dt).total_seconds() / 60.0
        if 0 <= diff_min < cooldown_minutes:
            return True, f"US Opening-Range ({int(diff_min)}/{cooldown_minutes} Min nach 09:30 ET Eröffnung)"
    return False, ""


class DeterministicRiskManager:
    """
    Harte, deterministische Risiko-Engine (Guardrails).
    Prüft zusätzlich:
    1. Makro- & Intermarket-Ampel (VIX, SPY, 10Y-Zinsen, US-Dollar)
    2. Earnings-Blackout-Schutz (Verhindert Käufe unmittelbar vor Quartalszahlen)
    3. Sektor-/Cluster-Korrelationsschutz (max. 45% pro Cluster) & 10% Cash-Reserve
    4. Mathematische 60-Tage-Korrelations-Bremse (blockiert bei r >= 0.82, drosselt ab r >= 0.65)
    5. Kovarianz-basierter Portfolio Value-at-Risk (95%-Tages-VaR Deckel)
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def evaluate_signal(
        self,
        signal: TradingSignal,
        snapshot: MarketSnapshot,
        macro: Optional[MacroRegime],
        cash_usd: float,
        total_equity_usd: float,
        daily_pnl_pct: float,
        positions: Dict[str, Position],
        portfolio_risk: Optional[PortfolioRiskMetrics] = None,
    ) -> RiskEvaluation:
        existing_pos = positions.get(signal.symbol)

        # 0. Börsenöffnungszeiten-Schutz für US- & Xetra-Aktien/ETFs (Krypto handelt 24/7)
        if self.config.enforce_market_hours and snapshot.asset_class == "EQUITY" and signal.action in ("BUY", "SELL"):
            mkt_open, mkt_status = is_us_equity_market_open(symbol=signal.symbol)
            if not mkt_open:
                return RiskEvaluation(
                    approved=False,
                    reason=f"BÖRSENZEITEN-SCHUTZ: {mkt_status} – Watch-Only im Shadow-Tracker.",
                    adjusted_action="HOLD",
                )
            # 15-Minuten Opening-Bell-Schutz für NEUE Positionen (Schließen bestehender Positionen ist immer sofort erlaubt!)
            is_new_entry = (signal.action == "BUY" and (not existing_pos or existing_pos.side == "LONG")) or (
                signal.action == "SELL" and not existing_pos
            )
            if is_new_entry:
                in_cooldown, cd_reason = is_opening_bell_cooldown(
                    symbol=signal.symbol, cooldown_minutes=self.config.opening_bell_cooldown_minutes
                )
                if in_cooldown:
                    return RiskEvaluation(
                        approved=False,
                        reason=f"OPENING-BELL-SCHUTZ: {cd_reason} – warte auf stabile Opening Range (Schutz vor Eröffnungs-Spreads).",
                        adjusted_action="HOLD",
                    )

        # 1. Bei SELL-Signal prüfen: Entweder bestehende Long-Position schließen ODER neuen Short-Trade (Leerverkauf) eröffnen!
        if signal.action == "SELL":
            if existing_pos and existing_pos.side == "LONG" and existing_pos.quantity > 0:
                if signal.confidence < 0.60:
                    return RiskEvaluation(
                        approved=False,
                        reason=f"Verkaufskonfidenz ({signal.confidence:.2f}) unter Schwellenwert (0.60).",
                        adjusted_action="HOLD",
                    )
                return RiskEvaluation(
                    approved=True,
                    reason=f"Verkaufssignal bestätigt (Konfidenz {signal.confidence:.2f}). Schließe Long-Position.",
                    adjusted_action="SELL",
                    recommended_quantity=existing_pos.quantity,
                    notional_value_usd=round(existing_pos.quantity * signal.entry_price, 2),
                )
            elif existing_pos and existing_pos.side == "SHORT":
                return RiskEvaluation(
                    approved=False,
                    reason="Short-Position bereits im Portfolio aktiv.",
                    adjusted_action="HOLD",
                )

            # Aktives Short-Selling (Leerverkauf) bei hochkonfidenten A+ SELL-Signalen (>= 85%)
            if (
                self.config.enable_short_selling
                and snapshot.asset_class == "EQUITY"
                and snapshot.intrinsic_leverage <= 1.0
                and not snapshot.is_safe_haven
                and signal.confidence >= self.config.min_short_confidence
                and signal.expected_edge_score >= self.config.min_expected_edge_score
            ):
                if daily_pnl_pct <= -(self.config.max_daily_drawdown_pct * 100.0):
                    return RiskEvaluation(
                        approved=False,
                        reason=f"CIRCUIT BREAKER AKTIV: Tagesverlust ({daily_pnl_pct:.2f}%) überschreitet Limit.",
                        adjusted_action="HOLD",
                    )
                if snapshot.days_to_earnings is not None and 0 <= snapshot.days_to_earnings <= self.config.earnings_blackout_days:
                    return RiskEvaluation(
                        approved=False,
                        reason=f"EARNINGS-SCHUTZ: Quartalszahlen in {snapshot.days_to_earnings} Tag(en) – Kein Leerverkauf vor Zahlen.",
                        adjusted_action="HOLD",
                    )
                min_cash_res = total_equity_usd * self.config.min_cash_reserve_pct
                deployable_short_cash = max(0.0, (cash_usd - min_cash_res) * 0.99)
                atr_val = max(snapshot.atr_14, signal.entry_price * 0.015)
                short_risk_per_unit = 1.6 * atr_val
                short_max_risk_usd = total_equity_usd * (self.config.max_risk_per_trade_pct * 0.65)
                short_notional = min(
                    (short_max_risk_usd / short_risk_per_unit) * signal.entry_price,
                    total_equity_usd * 0.15,
                    deployable_short_cash,
                )
                if short_notional >= self.config.min_order_usd:
                    s_qty = round(short_notional / signal.entry_price, 4)
                    s_notional = round(s_qty * signal.entry_price, 2)
                    s_loss = round(s_qty * short_risk_per_unit, 2)
                    s_risk_pct = round((s_loss / total_equity_usd) * 100.0, 2)
                    return RiskEvaluation(
                        approved=True,
                        reason=f"SHORT-SELL FREIGEGEBEN [📉 Leerverkauf, Konfidenz {signal.confidence*100:.0f}%]: {s_qty:g} Stk. (${s_notional:,.0f}), Stop-Loss +1.6x ATR, Max-Risiko ${s_loss:.2f} ({s_risk_pct:.2f}%)",
                        adjusted_action="SELL",
                        leverage_used=1.0,
                        effective_total_leverage=1.0,
                        recommended_quantity=s_qty,
                        margin_required_usd=s_notional,
                        notional_value_usd=s_notional,
                        max_loss_usd=s_loss,
                        portfolio_risk_pct=s_risk_pct,
                    )

            return RiskEvaluation(
                approved=False,
                reason="Kein Bestand zum Verkaufen (und Short-Schwelle >= 85% nicht erreicht).",
                adjusted_action="HOLD",
            )

        # 2. Bei HOLD geduldig abwarten (Cash-Schutz)
        if signal.action == "HOLD":
            return RiskEvaluation(
                approved=False,
                reason="⏸️ ABWARTEN: Kein erstklassiges A+ Setup – Cash bleibt geschützt.",
                adjusted_action="HOLD",
            )

        # 2b. Falls eine SHORT-Position offen ist und das Signal auf BUY dreht -> Short-Position sofort eindecken (Cover)!
        if existing_pos and existing_pos.side == "SHORT":
            return RiskEvaluation(
                approved=True,
                reason=f"Trendwende auf BUY ({signal.confidence*100:.0f}%) -> Schließe Short-Position (Short Cover).",
                adjusted_action="BUY",
                recommended_quantity=abs(existing_pos.quantity),
                notional_value_usd=round(abs(existing_pos.quantity) * signal.entry_price, 2),
            )

        # 2c. Falls bereits eine regulär skalierte LONG-Position (>= 2% des Kontos) existiert -> kein doppeltes Nachkaufen
        if existing_pos and existing_pos.side == "LONG" and existing_pos.margin_used_usd >= total_equity_usd * 0.02:
            return RiskEvaluation(
                approved=False,
                reason=f"Position {signal.symbol} bereits in regulärer Zielgröße (${existing_pos.margin_used_usd:,.0f}) im Portfolio.",
                adjusted_action="HOLD",
            )

        # 3. Ab hier: Prüfung eines BUY-Signals
        # 3a. Daily Drawdown Circuit Breaker
        if daily_pnl_pct <= -(self.config.max_daily_drawdown_pct * 100.0):
            return RiskEvaluation(
                approved=False,
                reason=f"CIRCUIT BREAKER AKTIV: Tagesverlust ({daily_pnl_pct:.2f}%) überschreitet Limit.",
                adjusted_action="HOLD",
            )

        # 3b. Makro-Ampel Prüfung (VIX, SPY, Zinsen, Dollar)
        macro_mult = 1.0
        if macro is not None:
            if macro.regime_state == "RED_DEFENSIVE" and not snapshot.is_safe_haven:
                return RiskEvaluation(
                    approved=False,
                    reason=f"MAKRO-STOPP (Rote Ampel, VIX {macro.vix_level:.1f}): Nur Safe-Haven (GLD) erlaubt.",
                    adjusted_action="HOLD",
                )
            macro_mult = 1.0 if snapshot.is_safe_haven else macro.risk_multiplier

        # 3c. Earnings-Blackout-Schutz
        if (
            snapshot.days_to_earnings is not None
            and 0 <= snapshot.days_to_earnings <= self.config.earnings_blackout_days
        ):
            return RiskEvaluation(
                approved=False,
                reason=f"EARNINGS-SCHUTZ: Quartalszahlen in {snapshot.days_to_earnings} Tag(en) – Kein Neukauf vor Zahlen.",
                adjusted_action="HOLD",
            )

        # 3d. Strenge Mindest-Konfidenz ("Sniper-Patience": Im Zweifel abwarten!)
        if signal.confidence < self.config.min_signal_confidence:
            return RiskEvaluation(
                approved=False,
                reason=f"⏸️ ABWARTEN: Konfidenz ({signal.confidence*100:.0f}%) unter A+ Schwelle ({self.config.min_signal_confidence*100:.0f}%) – warte auf besseres Setup.",
                adjusted_action="HOLD",
            )

        # 3e. Mindest-Chance-Risiko-Verhältnis & Expected Edge Score
        if signal.risk_reward_ratio < self.config.min_risk_reward_ratio:
            return RiskEvaluation(
                approved=False,
                reason=f"⏸️ ABWARTEN: Chance/Risiko ({signal.risk_reward_ratio:.2f}) < Minimum ({self.config.min_risk_reward_ratio:.2f}).",
                adjusted_action="HOLD",
            )
        if signal.expected_edge_score < self.config.min_expected_edge_score:
            return RiskEvaluation(
                approved=False,
                reason=f"⏸️ ABWARTEN: Edge-Score ({signal.expected_edge_score:.2f}) < A+ Filter ({self.config.min_expected_edge_score:.2f}) – Kapital bleibt in Cash.",
                adjusted_action="HOLD",
            )

        # 3f. Prüfen ob Einzelposition bereits das maximale Positionsgewicht (30% Margin-Anteil) erreicht hat
        min_order = self.config.min_order_usd
        current_pos_value = existing_pos.market_value if existing_pos else 0.0
        max_pos_value = total_equity_usd * self.config.max_position_weight_pct
        remaining_pos_capacity = max_pos_value - current_pos_value
        if remaining_pos_capacity < min_order:
            return RiskEvaluation(
                approved=False,
                reason=f"Maximales Positionsgewicht ({self.config.max_position_weight_pct*100:.0f}%) für {signal.symbol} bereits erreicht.",
                adjusted_action="HOLD",
            )

        # 3g. Rollierende 60-Tage Paarweise Korrelations-Prüfung gegen bestehenden Bestand
        max_r = snapshot.max_corr_with_portfolio
        corr_partner = snapshot.most_correlated_holding
        if max_r >= self.config.max_pairwise_correlation:
            return RiskEvaluation(
                approved=False,
                reason=f"KORRELATIONS-SPERRE: Zu hoher Gleichlauf mit Bestand {corr_partner} (r={max_r:+.2f} >= {self.config.max_pairwise_correlation:.2f}).",
                adjusted_action="HOLD",
            )

        # 3h. Kovarianz Portfolio Value-at-Risk (95% Tages-VaR) Deckel prüfen
        if (
            portfolio_risk is not None
            and portfolio_risk.daily_var_95_pct >= (self.config.max_portfolio_var_95_pct * 100.0)
            and max_r > 0.25
        ):
            return RiskEvaluation(
                approved=False,
                reason=f"VaR-LIMIT ERREICHT: Portfolio 95%-Tages-VaR ({portfolio_risk.daily_var_95_pct:.2f}%) am Limit – nur unkorrelierte Hedges (r<=0.25) erlaubt.",
                adjusted_action="HOLD",
            )

        # 3i. Sektor-/Cluster-Korrelationsschutz (z.B. max. 60% in US_TECH oder CRYPTO)
        cluster_name = snapshot.cluster
        current_cluster_value = sum(
            p.market_value for p in positions.values() if p.cluster == cluster_name
        )
        max_cluster_value = total_equity_usd * self.config.max_cluster_weight_pct
        remaining_cluster_capacity = max_cluster_value - current_cluster_value
        if remaining_cluster_capacity < min_order:
            return RiskEvaluation(
                approved=False,
                reason=f"CLUSTER-LIMIT: Sektor '{cluster_name}' hat das Maximum ({self.config.max_cluster_weight_pct*100:.0f}%) erreicht.",
                adjusted_action="HOLD",
            )

        # 3j. Eiserne Cash-Reserve prüfen (mind. 8% Cash-Puffer behalten)
        min_cash_reserve_usd = total_equity_usd * self.config.min_cash_reserve_pct
        deployable_cash_usd = max(0.0, (cash_usd - min_cash_reserve_usd) * 0.99)
        if deployable_cash_usd < min_order:
            return RiskEvaluation(
                approved=False,
                reason=f"CASH-RESERVE-SCHUTZ: Mindest-Liquiditätsreserve ({self.config.min_cash_reserve_pct*100:.0f}%) erreicht.",
                adjusted_action="HOLD",
            )

        # 4. Smart-Leverage & Risiko-basierte Positionsgrößenberechnung inkl. Korrelations-Drosselung
        risk_per_unit = signal.entry_price - signal.stop_loss_price
        if risk_per_unit <= 0:
            return RiskEvaluation(
                approved=False,
                reason="Ungültiger Stop-Loss (muss bei BUY unter dem Einstiegspreis liegen).",
                adjusted_action="HOLD",
            )

        # Smart-Leverage Bestimmung:
        # - 3x ETFs (SOXL, TQQQ, SQQQ) haben bereits 3.0x Hebel eingebaut -> kein zusätzlicher Margin-Hebel (leverage_used = 1.0)
        # - Penny Stocks (< 5.00$) werden ohne Margin-Hebel gehandelt (Schutz vor Spreads & Volatilität)
        # - Standard-Aktien & Krypto (intrinsic_leverage == 1.0) erhalten bei GRÜNER Makro-Ampel bis zu 2.0x Smart-Margin-Hebel
        is_penny = snapshot.cluster == "PENNY_STOCKS" or snapshot.current_price < 5.0
        is_green_macro = macro is None or macro.regime_state == "GREEN_RISK_ON"
        if is_green_macro and snapshot.intrinsic_leverage <= 1.0 and not snapshot.is_safe_haven and not is_penny:
            leverage_used = self.config.smart_leverage_max
        else:
            leverage_used = 1.0
        effective_total_leverage = round(leverage_used * snapshot.intrinsic_leverage, 1)

        corr_penalty = 1.0
        if max_r >= 0.65:
            # Lineare Reduktion des Risikos auf bis zu 55% bei hoher Korrelation (0.65 <= r < 0.85)
            corr_penalty = round(max(0.55, 1.0 - ((max_r - 0.65) / 0.35)), 2)

        effective_risk_pct = self.config.max_risk_per_trade_pct * macro_mult * corr_penalty
        max_risk_dollars = total_equity_usd * effective_risk_pct
        risk_based_qty = max_risk_dollars / risk_per_unit
        risk_based_notional = risk_based_qty * signal.entry_price
        risk_based_margin = risk_based_notional / leverage_used

        allowed_margin = min(
            risk_based_margin,
            remaining_pos_capacity,
            remaining_cluster_capacity,
            deployable_cash_usd,
        )
        # Penny-Stock-Positionen auf max 500$ deckeln zum Schutz des 100k-Portfolios
        if is_penny:
            allowed_margin = min(allowed_margin, 500.0)

        if allowed_margin < min_order:
            return RiskEvaluation(
                approved=False,
                reason=f"Verfügbares Margin-Kapital ({allowed_margin:.2f}$) unter Mindestordergröße ({min_order:.0f}$).",
                adjusted_action="HOLD",
            )

        allowed_notional = allowed_margin * leverage_used
        qty = allowed_notional / signal.entry_price
        decimals = 6 if signal.asset_class == "CRYPTO" else 4
        qty = round(qty, decimals)
        notional = round(qty * signal.entry_price, 2)
        margin_required = round(notional / leverage_used, 2)
        actual_max_loss = round(qty * risk_per_unit, 2)
        portfolio_risk_pct = round((actual_max_loss / total_equity_usd) * 100.0, 2)

        corr_note = f", Corr-Drossel {corr_penalty:.2f}x" if corr_penalty < 1.0 else ""
        lev_note = f"⚡ {effective_total_leverage:.0f}x Hebel ({leverage_used:.0f}x Margin)" if leverage_used > 1.0 else (
            f"⚡ {effective_total_leverage:.0f}x ETF-Hebel" if snapshot.intrinsic_leverage > 1.0 else "1x"
        )
        return RiskEvaluation(
            approved=True,
            reason=f"Freigegeben [{lev_note}, Edge {signal.expected_edge_score:.2f}{corr_note}]: Margin {margin_required:.0f}$ (Kaufkraft {notional:.0f}$), Max-Risiko {actual_max_loss:.2f}$ ({portfolio_risk_pct:.2f}%)",
            adjusted_action="BUY",
            leverage_used=leverage_used,
            effective_total_leverage=effective_total_leverage,
            recommended_quantity=qty,
            margin_required_usd=margin_required,
            notional_value_usd=notional,
            max_loss_usd=actual_max_loss,
            portfolio_risk_pct=portfolio_risk_pct,
        )
