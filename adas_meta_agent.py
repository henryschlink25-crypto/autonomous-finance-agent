import datetime
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from config import AgentConfig
from models import MacroRegime, MarketSnapshot, Position, TradingSignal
from adas_reflection import (
    ReasoningStep, VoteTrace, ParamChange, StepRecorder,
    fmt_step, tuning_for, clean_float,
    BLAME_THRESHOLD, REVIEW_AFTER_EVALS, EVAL_MIN_MOVE_PCT, NEAR_MISS_MIN_MOVE_PCT, TRACE_MAX_AGE_DAYS
)


class SubAgentSpec(BaseModel):
    """
    Spezifikation eines spezialisierten Unteragenten im Sakana-AI / ADAS Schwarm.
    Enthält:
    - Gene & Parameter (Mutations-Fähigkeit)
    - Dynamische Prompt-Hypothese (LLM-generiert)
    - Trust-Weight (Stimmgewicht nach Trefferquote)
    - Fitness-Score (Gewinnbeitrag & vermiedene Verluste)
    - Schritt-für-Schritt Reflexions-Gedächtnis (blame_counts, near_miss_counts, param_history, reflection_log)
    """
    agent_id: str
    name: str
    role: str
    generation: int = 1
    is_spawned_by_adas: bool = False
    created_at: str = ""
    specialty_focus: str = ""
    trust_weight: float = 1.0          # Dynamisches Stimmgewicht (0.35 bis 1.85)
    signals_evaluated: int = 0
    correct_votes: int = 0
    false_votes: int = 0
    cumulative_edge_contributed: float = 0.0
    last_verdict_summary: str = ""
    rule_parameters: Dict[str, float] = Field(default_factory=dict)
    prompt_directive: str = ""         # Sakana-Prompt für qualitative Hypothesen-Prüfung
    fitness_score: float = 50.0
    param_defaults: Dict[str, float] = Field(default_factory=dict)
    blame_counts: Dict[str, int] = Field(default_factory=dict)
    near_miss_counts: Dict[str, int] = Field(default_factory=dict)
    param_history: List[ParamChange] = Field(default_factory=list)
    reflection_log: List[str] = Field(default_factory=list)

    @property
    def accuracy_pct(self) -> float:
        if self.signals_evaluated <= 0:
            return 50.0
        return round((self.correct_votes / self.signals_evaluated) * 100.0, 1)


class SubAgentVote(BaseModel):
    agent_id: str
    agent_name: str
    symbol: str
    vote_action: str       # "BUY", "SELL", "HOLD", "VETO_BUY"
    score: float           # -1.0 bis +1.0
    confidence: float      # 0.0 bis 1.0
    rationale: str


class ADASMetaState(BaseModel):
    updated_at: str = ""
    evolution_generation: int = 1
    total_evolution_cycles: int = 0
    active_subagents: Dict[str, SubAgentSpec] = Field(default_factory=dict)
    extinct_subagents: List[str] = Field(default_factory=list)
    evolution_log: List[str] = Field(default_factory=list)
    Active_entry_prices: Dict[str, float] = Field(default_factory=dict)
    entry_times: Dict[str, str] = Field(default_factory=dict)
    pending_traces: Dict[str, Dict[str, VoteTrace]] = Field(default_factory=dict)
    total_reflections: int = 0
    total_param_adjustments: int = 0
    total_reverts: int = 0


class ADASMetaArchitect:
    """
    🧬 Sakana AI Evolutionary Architecture (ADAS) mit autonomer Schritt-für-Schritt-Reflexion:
    1. Jeder Unteragent analysiert seine Prüfschritte (VoteTrace) einzeln.
    2. Nach Preisbewegungen evaluiert der Agent: Welche Schritte führten zu Gewinnen, welche zu Fehlern?
    3. Schuldige Schwellenwerte werden gezielt justiert und bei schlechterer Performance zurückgerollt.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        data_dir = os.path.dirname(self.config.portfolio_state_file)
        os.makedirs(data_dir, exist_ok=True)
        self.state_file = os.path.join(data_dir, "adas_meta_state.json")
        self.state = ADASMetaState()
        self.last_symbol_votes: Dict[str, List[SubAgentVote]] = {}
        self._load_or_initialize_swarm()

    def _load_or_initialize_swarm(self) -> None:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                legacy_tracker = raw.get("active_vote_Tracker")
                self.state = ADASMetaState(**raw)
                if legacy_tracker and not self.state.pending_traces:
                    self.state.Active_entry_prices = {}
                    self.state.entry_times = {}
            except Exception:
                self.state = ADASMetaState()

        if not self.state.active_subagents:
            now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]
            foundational = [
                SubAgentSpec(
                    agent_id="alpha_breakout_hunter",
                    name="🚀 Alpha-Breakout & RVOL Hunter",
                    role="Sucht explosive Ausbrüche mit hohem Relativ-Volumen (>1.6x) und starker Relativer Stärke vs. S&P 500.",
                    generation=1,
                    created_at=now_str,
                    specialty_focus="MOMENTUM_RVOL_RS",
                    trust_weight=1.15,
                    rule_parameters={"min_rvol": 1.6, "min_rs_spy": 4.0},
                    prompt_directive="Priorisiere Aktien mit frischem institutionellen Volumen und bestätigtem Trend.",
                ),
                SubAgentSpec(
                    agent_id="options_whale_profiler",
                    name="🐋 Options-Whale & IV-Skew Profiler",
                    role="Analysiert ungewöhnliche Options-Käufe (Vol/OI Ratio), Put/Call-Skew und Smart-Money Call-Sweeps.",
                    generation=1,
                    created_at=now_str,
                    specialty_focus="OPTIONS_WHALE_FLOW",
                    trust_weight=1.20,
                    rule_parameters={"whale_vol_oi_threshold": 1.8, "bullish_pc_max": 0.68},
                    prompt_directive="Finde institutionelle Smart-Money Call-Wetten mit massivem Hebelpotenzial.",
                ),
                SubAgentSpec(
                    agent_id="intraday_vwap_sniper",
                    name="🎯 Intraday 15m-VWAP & Pullback Sniper",
                    role="Prüft das Intraday-Timing auf 15-Minuten-Basis (Kurs über VWAP vs. gefährlicher Abverkauf vom Tageshoch).",
                    generation=1,
                    created_at=now_str,
                    specialty_focus="INTRADAY_MICROSTRUCTURE",
                    trust_weight=1.15,
                    rule_parameters={"fading_penalty": -0.65, "vwap_confirm_boost": 0.45},
                    prompt_directive="Verhindere das Kaufen von Intraday-Hochs; kaufe nur bei Kurs über VWAP.",
                ),
                SubAgentSpec(
                    agent_id="fundamental_catalyst_auditor",
                    name="📊 Fundamental & Earnings-Catalyst Auditor",
                    role="Bewertet Umsatzwachstum, Gewinnmargen, KGV-Bewertung und hochrelevante News-Katalysatoren.",
                    generation=1,
                    created_at=now_str,
                    specialty_focus="FUNDAMENTALS_AND_NEWS",
                    trust_weight=1.05,
                    rule_parameters={"min_rev_growth": 15.0, "max_healthy_pe": 42.0},
                    prompt_directive="Trenne nachhaltige fundamentale Qualität von reinen Hype-Blasen.",
                ),
                SubAgentSpec(
                    agent_id="cross_asset_macro_sentinel",
                    name="🌐 Cross-Asset & Macro-Regime Sentinel",
                    role="Überwacht Zinsen (10Y Yield), US-Dollar (DXY), VIX und Lead-Lag-Rotationen zwischen US, EU/Xetra und China.",
                    generation=1,
                    created_at=now_str,
                    specialty_focus="MACRO_INTERMARKET",
                    trust_weight=1.05,
                    rule_parameters={"max_corr_tolerance": 0.78},
                    prompt_directive="Erkenne Marktwendepunkte durch Zins- und Dollarbewegungen frühzeitig.",
                ),
                SubAgentSpec(
                    agent_id="tail_risk_adversary_guard",
                    name="🛡️ Adversarial Tail-Risk & Trap Guard",
                    role="Spielt des Teufels Advokat: Sucht nach Bullenfallen (RSI > 82 ohne News, Parabolic Exhaustion, Earnings-Risiko).",
                    generation=1,
                    created_at=now_str,
                    specialty_focus="RISK_VETO_GUARD",
                    trust_weight=1.25,
                    rule_parameters={"overbought_rsi_veto": 81.0, "max_volatility_pct": 115.0},
                    prompt_directive="Schütze das Kapital aggressiv vor Liquiditätsfallen und Bullenfallen.",
                ),
            ]
            for ag in foundational:
                self.state.active_subagents[ag.agent_id] = ag
            self.state.evolution_log.append(
                f"[{now_str[:10]}] 🧬 Sakana AI Initialisierung: 6 Kern-Unteragenten aktiv."
            )
            self._save()

    def _save(self) -> None:
        self.state.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state.model_dump(), f, indent=2, ensure_ascii=False)

    def run_evolution_and_spawning_cycle(
        self,
        snapshots: Dict[str, MarketSnapshot],
        macro: MacroRegime,
        positions: Dict[str, Position],
    ) -> List[str]:
        events: List[str] = []
        self.state.total_evolution_cycles += 1
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        now_str = now_dt.isoformat()[:19]

        symbols_to_clear: List[str] = []
        n_eval = n_correct = n_wrong = 0
        for sym, entry_p in list(self.state.Active_entry_prices.items()):
            t_str = self.state.entry_times.get(sym)
            if not t_str:
                self.state.entry_times[sym] = now_str
            else:
                try:
                    age = now_dt - datetime.datetime.fromisoformat(t_str).replace(tzinfo=datetime.timezone.utc)
                    if age.total_seconds() > TRACE_MAX_AGE_DAYS * 86400:
                        symbols_to_clear.append(sym)
                        continue
                except Exception:
                    pass

            snap = snapshots.get(sym)
            cur_p = clean_float(snap.current_price) if snap else None
            if cur_p is None or entry_p <= 0:
                continue
            move_pct = ((cur_p - entry_p) / entry_p) * 100.0
            if abs(move_pct) < EVAL_MIN_MOVE_PCT:
                continue

            direction = 1.0 if move_pct > 0 else -1.0
            for ag_id, trace in self.state.pending_traces.get(sym, {}).items():
                ag = self.state.active_subagents.get(ag_id)
                if not ag:
                    continue
                if abs(trace.score) >= 0.12:
                    ag.signals_evaluated += 1
                    n_eval += 1
                    if trace.score * direction > 0:
                        n_correct += 1
                        ag.correct_votes += 1
                        ag.cumulative_edge_contributed = round(ag.cumulative_edge_contributed + abs(move_pct) * 0.25, 2)
                        ag.trust_weight = round(float(np.clip(ag.trust_weight + 0.035, 0.35, 1.85)), 3)
                        ag.fitness_score = round(min(100.0, ag.fitness_score + 2.5), 1)
                    else:
                        n_wrong += 1
                        ag.false_votes += 1
                        ag.cumulative_edge_contributed = round(ag.cumulative_edge_contributed - abs(move_pct) * 0.25, 2)
                        ag.trust_weight = round(float(np.clip(ag.trust_weight - 0.04, 0.35, 1.85)), 3)
                        ag.fitness_score = round(max(0.0, ag.fitness_score - 3.5), 1)
                events.extend(self._reflect_on_trace(ag, trace, move_pct, now_str))
            symbols_to_clear.append(sym)

        for s in symbols_to_clear:
            self.state.Active_entry_prices.pop(s, None)
            self.state.entry_times.pop(s, None)
            self.state.pending_traces.pop(s, None)

        for ag in self.state.active_subagents.values():
            events.extend(self._review_param_changes(ag, now_str))

        if n_eval > 0:
            events.insert(
                0,
                f"[{now_str[11:19]}] 🔍 Schritt-für-Schritt-Reflexion: {n_eval} Voten analysiert "
                f"({n_correct} korrekt / {n_wrong} fehlgeschlagen) – Parameter angepasst.",
            )

        # Selektion
        for ag_id, ag in list(self.state.active_subagents.items()):
            if ag.is_spawned_by_adas and ag.signals_evaluated >= 10 and ag.accuracy_pct < 38.0 and ag.fitness_score < 25.0:
                msg = f"[{now_str[11:19]}] 💀 Selektion: '{ag.name}' aussortiert (Trefferquote {ag.accuracy_pct:.1f}% nach {ag.signals_evaluated} Voten)."
                self.state.extinct_subagents.append(f"{ag.name} (Gen #{ag.generation})")
                self.state.evolution_log.append(msg)
                events.append(msg)
                self.state.active_subagents.pop(ag_id, None)

        # Spawning
        for spec in self._detect_needed_specialist_agents(snapshots, macro):
            if spec.agent_id not in self.state.active_subagents:
                self.state.evolution_generation += 1
                spec.generation = self.state.evolution_generation
                spec.is_spawned_by_adas = True
                spec.created_at = now_str
                self.state.active_subagents[spec.agent_id] = spec
                msg = f"[{now_str[11:19]}] 🧬 Neu erschaffen (Gen #{spec.generation}): '{spec.name}' -> {spec.role}"
                self.state.evolution_log.append(msg)
                events.append(msg)
                self._send_evolution_alert(spec)

        self.state.evolution_log = self.state.evolution_log[-30:]
        self._save()
        return events

    def _send_evolution_alert(self, spec: SubAgentSpec) -> None:
        """Sendet einen sofortigen Push-Alert bei Geburt eines neuen Unteragenten."""
        try:
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
            safe_name = html.escape(spec.name)
            safe_role = html.escape(spec.role)
            text_body = (
                f"🤖 <b>🧬 Neuer KI-Spezialist autonom erschaffen!</b>\n\n"
                f"• <b>Agent:</b> {safe_name} (Gen #{spec.generation})\n"
                f"• <b>Spezialgebiet:</b> {safe_role}\n"
                f"• <b>Stimmgewicht:</b> {spec.trust_weight:.2f}x\n\n"
                f"Der Unteragent wurde live in den Handels-Schwarm integriert."
            )
            payload = json.dumps({"chat_id": cid, "text": text_body, "parse_mode": "HTML"}).encode("utf-8")
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, context=ctx, timeout=4.0)
        except Exception:
            pass

    def _adjust_param(self, ag: SubAgentSpec, key: str, op: str, tighten: bool, reason: str, now_str: str) -> Optional[str]:
        step, lo, hi = tuning_for(key, ag.param_defaults, ag.rule_parameters)
        old = float(ag.rule_parameters.get(key, ag.param_defaults.get(key, 0.0)))
        delta = step if ((op == ">=") == tighten) else -step
        new = round(min(hi, max(lo, old + delta)), 4)
        if abs(new - old) < 1e-9:
            return None
        ag.rule_parameters[key] = new
        ag.param_history.append(
            ParamChange(
                param_key=key, old_value=old, new_value=new, reason=reason, changed_at=now_str,
                evals_at_change=ag.signals_evaluated, correct_at_change=ag.correct_votes,
            )
        )
        ag.param_history = ag.param_history[-20:]
        ag.generation += 1
        self.state.total_param_adjustments += 1
        verb = "verschärft" if tighten else "gelockert"
        msg = f"[{now_str.replace('T', ' ')[:16]}] 🔧 {ag.name}: Schwelle '{key}' {verb} {old:.3f} → {new:.3f} ({reason})"
        ag.reflection_log.append(msg)
        self.state.evolution_log.append(msg)
        return msg

    def _reflect_on_trace(self, ag: SubAgentSpec, trace: VoteTrace, move_pct: float, now_str: str) -> List[str]:
        events: List[str] = []
        ts = now_str.replace("T", " ")[:16]
        steps_txt = " | ".join(fmt_step(s) for s in trace.steps) or "keine Prüfschritte"
        self.state.total_reflections += 1

        if abs(trace.score) < 0.12:
            if abs(move_pct) < NEAR_MISS_MIN_MOVE_PCT:
                return events
            move_dir = "bull" if move_pct > 0 else "bear"
            near: List[ReasoningStep] = []
            for s in trace.steps:
                if s.passed or not s.param_key or s.observed is None or s.direction != move_dir:
                    continue
                step_size, _, _ = tuning_for(s.param_key, ag.param_defaults, ag.rule_parameters)
                if abs(s.observed - s.threshold) <= 2 * step_size:
                    near.append(s)
            if not near:
                return events
            for s in near:
                ag.near_miss_counts[s.param_key] = ag.near_miss_counts.get(s.param_key, 0) + 1
            ag.reflection_log.append(
                f"[{ts}] ⚠️ {trace.symbol}: neutral, aber Kurs {move_pct:+.2f}%. Knapp verfehlt: "
                + ", ".join(f"{fmt_step(s)} ({ag.near_miss_counts[s.param_key]}/{BLAME_THRESHOLD})" for s in near)
            )
            for s in near:
                if ag.near_miss_counts.get(s.param_key, 0) >= BLAME_THRESHOLD:
                    msg = self._adjust_param(ag, s.param_key, s.op, tighten=False,
                                             reason=f"{BLAME_THRESHOLD}× große Bewegung knapp verpasst", now_str=now_str)
                    ag.near_miss_counts[s.param_key] = 0
                    if msg:
                        events.append(msg)
            ag.reflection_log = ag.reflection_log[-15:]
            return events

        correct = trace.score * (1.0 if move_pct > 0 else -1.0) > 0
        decisive = [trace.steps[i] for i in trace.decisive_idx if 0 <= i < len(trace.steps)]
        tunable = [s for s in decisive if s.param_key]
        categorical = [s for s in decisive if not s.param_key]

        if correct:
            for s in tunable:
                ag.blame_counts[s.param_key] = max(0, ag.blame_counts.get(s.param_key, 0) - 1)
            carried = ", ".join(fmt_step(s) for s in decisive) or "–"
            ag.reflection_log.append(
                f"[{ts}] ✅ {trace.symbol}: {trace.vote_action} ({trace.score:+.2f}) richtig, Kurs {move_pct:+.2f}%. Getragen von: {carried}"
            )
        else:
            for s in tunable:
                ag.blame_counts[s.param_key] = ag.blame_counts.get(s.param_key, 0) + 1
            cause_parts = [f"{s.param_key} ({ag.blame_counts[s.param_key]}/{BLAME_THRESHOLD})" for s in tunable]
            if categorical:
                cause_parts.append(
                    "kategoriale Signale [" + ", ".join(s.label for s in categorical) + "]"
                )
            ag.reflection_log.append(
                f"[{ts}] ❌ {trace.symbol}: {trace.vote_action} ({trace.score:+.2f}) falsch, Kurs {move_pct:+.2f}%. "
                f"Schritte: {steps_txt}. Verantwortlich: {'; '.join(cause_parts) or 'unklar'}"
            )
            for s in tunable:
                if ag.blame_counts.get(s.param_key, 0) >= BLAME_THRESHOLD:
                    msg = self._adjust_param(ag, s.param_key, s.op, tighten=True,
                                             reason=f"{BLAME_THRESHOLD}× als Fehlerursache identifiziert", now_str=now_str)
                    ag.blame_counts[s.param_key] = 0
                    if msg:
                        events.append(msg)

        ag.reflection_log = ag.reflection_log[-15:]
        return events

    def _review_param_changes(self, ag: SubAgentSpec, now_str: str) -> List[str]:
        events: List[str] = []
        for ch in ag.param_history:
            if ch.status != "PENDING":
                continue
            evals_since = ag.signals_evaluated - ch.evals_at_change
            if evals_since < REVIEW_AFTER_EVALS:
                continue
            acc_since = (ag.correct_votes - ch.correct_at_change) / evals_since * 100.0
            acc_before = (ch.correct_at_change / ch.evals_at_change * 100.0) if ch.evals_at_change > 0 else 50.0
            if acc_since + 10.0 < acc_before:
                ag.rule_parameters[ch.param_key] = ch.old_value
                ch.status = "REVERTED"
                self.state.total_reverts += 1
                msg = (
                    f"[{now_str.replace('T', ' ')[:16]}] ↩️ {ag.name}: Anpassung '{ch.param_key}' zurückgerollt "
                    f"({ch.new_value:.3f} → {ch.old_value:.3f}), Trefferquote danach {acc_since:.0f}% statt {acc_before:.0f}%."
                )
                ag.reflection_log.append(msg)
                self.state.evolution_log.append(msg)
                events.append(msg)
            else:
                ch.status = "CONFIRMED"
        ag.reflection_log = ag.reflection_log[-15:]
        return events

    def recent_reflections(self, n: int = 6) -> List[str]:
        items: List[str] = []
        for ag in self.state.active_subagents.values():
            for entry in ag.reflection_log:
                items.append(entry if ag.name in entry else entry.replace("] ", f"] {ag.name} · ", 1))
        items.sort()
        return items[-n:]

    def _detect_needed_specialist_agents(self, snapshots: Dict[str, MarketSnapshot], macro: MacroRegime) -> List[SubAgentSpec]:
        new_specs: List[SubAgentSpec] = []
        snaps = list(snapshots.values())

        if any(s.unusual_options_vol_oi >= 3.5 for s in snaps) and "gamma_squeeze_specialist" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="gamma_squeeze_specialist",
                name="⚡ Gamma-Squeeze & Extreme-Whale Specialist",
                role="Jagd auf asymmetrische Options-Gamma-Squeezes mit V/OI >= 3.5x und extremem Call-Kaufdruck.",
                specialty_focus="GAMMA_SQUEEZE",
                trust_weight=1.22,
                rule_parameters={"min_vol_oi_ratio": 3.5, "max_pc_ratio": 0.40},
            ))

        if any(s.change_24h_pct >= 9.0 and s.volume_ratio_vs_20d >= 1.8 for s in snaps) and "pead_earnings_drift_agent" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="pead_earnings_drift_agent",
                name="📈 Post-Earnings-Announcement-Drift (PEAD) Agent",
                role="Handelt mehrtägige institutionelle Trendfortsetzungen nach starken Quartalszahlen und Gap-Ups.",
                specialty_focus="PEAD_DRIFT",
                trust_weight=1.18,
                rule_parameters={"min_gap_pct": 8.5, "min_rvol": 1.8, "max_rsi_ceiling": 79.0},
            ))

        if "oversold_quality_dip_buyer" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="oversold_quality_dip_buyer",
                name="💎 Institutional Quality-Dip & Rebound Sniper",
                role="Kauft antizyklisch stark überverkaufte Qualitätsführer (RSI < 37, Umsatzwachstum > 15%), sobald sie den VWAP zurückerobern.",
                specialty_focus="QUALITY_MEAN_REVERSION",
                trust_weight=1.10,
                rule_parameters={"oversold_rsi": 37.0, "min_rev_growth": 15.0},
            ))

        if "short_breakdown_bear_agent" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="short_breakdown_bear_agent",
                name="🐻 Short-Breakdown & Bear-Put Hunter",
                role="Identifiziert hochkonfidente Leerverkaufs-Chancen (Short-Sells) bei gebrochenem Trend (< SMA20 & < VWAP).",
                specialty_focus="SHORT_SELL_SPECIALIST",
                trust_weight=1.12,
            ))

        if any(s.symbol.endswith(".DE") for s in snaps) and "xetra_opening_arbitrageur" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="xetra_opening_arbitrageur",
                name="🏰 Xetra & European Momentum Arbitrageur",
                role="Spezialisiert auf deutsche Industrie- und Tech-Marktführer (Rheinmetall, SAP, Siemens) im Xetra-Handel.",
                specialty_focus="XETRA_EUROPEAN_LEADERS",
                trust_weight=1.15,
                rule_parameters={"min_euro_volume_ratio": 1.35},
            ))

        # Autonome Spezialisten-Erstellung bei erkannten Werkzeug- & Markt-Lücken:
        # 1. Penny Stocks & Micro-Cap High-Volatility Sniper
        if (any(s.current_price < 5.0 or s.cluster == "PENNY_STOCKS" for s in snaps) or True) and "penny_stock_rocket_specialist" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="penny_stock_rocket_specialist",
                name="🎯 Penny Stock & Micro-Cap Rocket Sniper",
                role="Spezialist für hochvolatile Penny Stocks ($0.50-$5.00) mit extremem Relativ-Volumen (>1.9x) und Momentum-Ausbrüchen im Alpaca Paper Trading.",
                specialty_focus="PENNY_STOCK_MOMENTUM",
                trust_weight=1.18,
                rule_parameters={"min_penny_rvol": 1.9, "min_penny_chg": 3.8, "max_penny_price": 5.0, "max_penny_dump": -5.5},
            ))

        # 2. TTM Volatility Squeeze Ausbruchs-Sniper
        if "volatility_squeeze_specialist" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="volatility_squeeze_specialist",
                name="🔥 TTM Volatility Squeeze & Compression Sniper",
                role="Erkennt Bollinger-Band Kompression innerhalb Keltner-Kanälen (Carter TTM Squeeze) und handelt explosive Alpha-Zündungen.",
                specialty_focus="TTM_SQUEEZE_EXPLOSION",
                trust_weight=1.20,
                rule_parameters={"min_squeeze_mom": 0.05},
            ))

        # 3. SEC Form-4 Insider & Whale Accumulation Follower
        if "sec_insider_whale_follower" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="sec_insider_whale_follower",
                name="👔 SEC Form-4 Insider & Smart-Money Follower",
                role="Scannt offizielle Form-4 SEC-Einreichungen und folgt massiven Netto-Insider-Käufen von Vorständen und CEOs.",
                specialty_focus="INSIDER_ACCUMULATION",
                trust_weight=1.16,
            ))

        # 4. SuperTrend ATR-Ratchet Trend Specialist
        if "supertrend_regime_specialist" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="supertrend_regime_specialist",
                name="🌊 SuperTrend ATR-Ratchet Trend Specialist",
                role="Handelt saubere Trendfortsetzungen und Trend-Flips basierend auf dem quantitativen SuperTrend ATR-Oszillator.",
                specialty_focus="SUPERTREND_MOMENTUM",
                trust_weight=1.14,
            ))

        return new_specs


    def evaluate_symbol_with_swarm(
        self,
        snap: MarketSnapshot,
        baseline_signal: TradingSignal,
        macro: MacroRegime,
    ) -> Tuple[TradingSignal, List[SubAgentVote], str]:
        votes: List[SubAgentVote] = []
        traces: List[VoteTrace] = []

        for ag in self.state.active_subagents.values():
            vote, trace = self._compute_single_subagent_vote(ag, snap, macro)
            ag.last_verdict_summary = f"{snap.symbol}: {vote.vote_action} ({vote.score:+.2f})"
            votes.append(vote)
            traces.append(trace)

        self.last_symbol_votes[snap.symbol] = votes

        if snap.symbol not in self.state.Active_entry_prices:
            self.state.Active_entry_prices[snap.symbol] = snap.current_price
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()[:19]
            self.state.entry_times[snap.symbol] = now_iso
            self.state.pending_traces[snap.symbol] = {t.agent_id: t for t in traces}

        total_w = 0.0
        weighted_sum = 0.0
        bull_supporters: List[str] = []
        bear_supporters: List[str] = []
        veto_reasons: List[str] = []

        for v in votes:
            ag = self.state.active_subagents[v.agent_id]
            w = ag.trust_weight * max(0.35, v.confidence)
            total_w += w
            weighted_sum += v.score * w
            if v.vote_action == "VETO_BUY":
                veto_reasons.append(f"{ag.name}: {v.rationale}")
            elif v.score >= 0.28:
                bull_supporters.append(ag.name.split()[0] + " " + ag.name.split()[1])
            elif v.score <= -0.28:
                bear_supporters.append(ag.name.split()[0] + " " + ag.name.split()[1])

        swarm_consensus = weighted_sum / max(total_w, 1e-6)

        new_action = baseline_signal.action
        new_conf = baseline_signal.confidence
        swarm_tag = ""

        if new_action == "BUY":
            if veto_reasons and swarm_consensus < 0.35:
                new_action = "HOLD"
                new_conf = round(max(0.45, new_conf - 0.25), 2)
                swarm_tag = f" 🛡️ [Sakana-VETO: {veto_reasons[0]}]"
            elif swarm_consensus >= 0.30 and len(bull_supporters) >= 3:
                boost = min(0.08, swarm_consensus * 0.12)
                new_conf = round(min(0.97, new_conf + boost), 2)
                swarm_tag = f" 🐝 [Sakana-Konsens +{swarm_consensus:.2f}: {', '.join(bull_supporters[:3])}]"
            elif swarm_consensus < 0.05:
                new_action = "HOLD"
                new_conf = round(min(0.66, new_conf), 2)
                swarm_tag = f" ⏸️ [Sakana-Schwarm uneinig ({swarm_consensus:+.2f}) – Cash-Schutz]"

        elif new_action == "SELL":
            if swarm_consensus <= -0.25 and len(bear_supporters) >= 2:
                new_conf = round(min(0.95, new_conf + 0.06), 2)
                swarm_tag = f" 🐻 [Sakana-Short-Konsens {swarm_consensus:+.2f}: {', '.join(bear_supporters[:2])}]"

        elif new_action == "HOLD":
            if swarm_consensus >= 0.50 and len(bull_supporters) >= 4 and not veto_reasons and snap.rsi_14 < 78.0:
                new_action = "BUY"
                new_conf = round(max(0.80, min(0.93, 0.55 + swarm_consensus * 0.45)), 2)
                swarm_tag = f" 🚀 [Sakana-Alpha-Entdeckung ({len(bull_supporters)} Agenten): {', '.join(bull_supporters[:3])}]"

        updated_signal = TradingSignal(
            symbol=baseline_signal.symbol,
            asset_class=baseline_signal.asset_class,
            cluster=baseline_signal.cluster,
            intrinsic_leverage=baseline_signal.intrinsic_leverage,
            action=new_action,
            confidence=new_conf,
            sentiment_score=baseline_signal.sentiment_score,
            entry_price=baseline_signal.entry_price,
            stop_loss_price=baseline_signal.stop_loss_price,
            partial_take_profit_price=baseline_signal.partial_take_profit_price,
            take_profit_price=baseline_signal.take_profit_price,
            risk_reward_ratio=baseline_signal.risk_reward_ratio,
            catalyst_summary=baseline_signal.catalyst_summary,
            thesis=baseline_signal.thesis + swarm_tag,
            best_option_contract=baseline_signal.best_option_contract,
            analysis_source=f"SAKANA_SWARM_GEN{self.state.evolution_generation}",
        )
        return updated_signal, votes, swarm_tag

    def _compute_single_subagent_vote(
        self,
        ag: SubAgentSpec,
        snap: MarketSnapshot,
        macro: MacroRegime,
    ) -> Tuple[SubAgentVote, VoteTrace]:
        rec = StepRecorder(ag.rule_parameters, ag.param_defaults)
        focus = ag.specialty_focus
        score = 0.0
        conf = 0.5
        rationale = "Neutral"
        decisive_idx: List[int] = []

        if focus == "MOMENTUM_RVOL_RS":
            s_rvol = rec.num("RVOL", snap.volume_ratio_vs_20d, ">=", "min_rvol", 1.6, "bull")
            s_chg = rec.num("24h%", snap.change_24h_pct, ">=", "min_chg_bull", 2.5, "bull")
            s_rs = rec.num("RS vs SPY", snap.relative_strength_vs_spy, ">=", "min_rs_spy", 4.0, "bull")
            s_sqz = rec.flag("TTM Squeeze Fired", snap.ttm_squeeze_status == "🔥 SQUEEZE_FIRED", "bull")
            s_weak_chg = rec.num("24h% Schwäche", snap.change_24h_pct, "<=", "max_chg_bear", -3.0, "bear")
            s_weak_rs = rec.num("RS Schwäche", snap.relative_strength_vs_spy, "<=", "max_rs_bear", -4.0, "bear")
            if s_rvol.passed and s_chg.passed and s_rs.passed:
                score = min(0.95, 0.45 + (snap.volume_ratio_vs_20d - 1.0) * 0.15 + snap.relative_strength_vs_spy * 0.015)
                if s_sqz.passed:
                    score = min(0.98, score + 0.10)
                conf = 0.88 if s_sqz.passed else 0.85
                sqz_txt = " [🔥TTM Squeeze Fired]" if s_sqz.passed else ""
                rationale = f"RVOL {snap.volume_ratio_vs_20d:.1f}x & RS +{snap.relative_strength_vs_spy:.1f}% vs. SPY{sqz_txt}"
                decisive_idx = rec.idx(s_rvol, s_chg, s_rs, s_sqz)
            elif s_weak_chg.passed and s_weak_rs.passed:
                score = -0.65
                conf = 0.75
                rationale = f"Relative Schwäche ({snap.relative_strength_vs_spy:+.1f}% vs. SPY)"
                decisive_idx = rec.idx(s_weak_chg, s_weak_rs)

        elif focus == "OPTIONS_WHALE_FLOW":
            s_whale = rec.num("Vol/OI Whale", snap.unusual_options_vol_oi, ">=", "whale_vol_oi_threshold", 1.8, "bull")
            s_pc_bull = rec.num("Put/Call Bull", snap.options_put_call_ratio, "<=", "bullish_pc_max", 0.68, "bull")
            s_call_flow = rec.flag("Bullish Call Flow", snap.options_flow_sentiment == "BULLISH_CALL_FLOW", "bull")
            s_put_flow = rec.flag("Bearish Put Flow", snap.options_flow_sentiment == "BEARISH_PUT_FLOW", "bear")
            s_gamma = rec.num("Option Gamma", snap.option_gamma, ">=", "min_gamma", 0.015, "bull") if snap.option_gamma else None
            if s_call_flow.passed:
                score = 0.90 if s_whale.passed else 0.55
                if s_gamma and s_gamma.passed:
                    score = min(0.97, score + 0.06)
                conf = 0.92 if s_whale.passed else 0.80
                gamma_txt = f" (Γ:{snap.option_gamma:.4f})" if snap.option_gamma else ""
                rationale = f"Bullischer Call-Flow ({snap.best_option_contract}){gamma_txt}"
                decisive_idx = rec.idx(s_call_flow, s_whale)
                if s_gamma and s_gamma.passed:
                    decisive_idx.extend(rec.idx(s_gamma))
            elif s_put_flow.passed:
                score = -0.65
                conf = 0.80
                rationale = f"Bärische Put-Käufe (P/C {snap.options_put_call_ratio})"
                decisive_idx = rec.idx(s_put_flow)
            elif s_whale.passed and "CALL" in snap.best_option_contract:
                score = 0.70
                conf = 0.82
                rationale = f"Unusual Call Whale V/OI {snap.unusual_options_vol_oi:.1f}x"
                decisive_idx = rec.idx(s_whale)

        elif focus == "INTRADAY_MICROSTRUCTURE":
            s_above_vwap = rec.flag("Kurs > VWAP", snap.intraday_15m_trend == "BULLISH_ABOVE_VWAP", "bull")
            s_fading = rec.flag("Abverkauf vom Hoch", snap.intraday_15m_trend == "FADING_FROM_HIGH", "bear")
            s_below_vwap = rec.flag("Kurs < VWAP", snap.intraday_15m_trend == "BEARISH_BELOW_VWAP", "bear")
            if s_above_vwap.passed:
                score = 0.65
                conf = 0.82
                rationale = f"Stabil über 15m-VWAP (${snap.intraday_vwap})"
                decisive_idx = rec.idx(s_above_vwap)
            elif s_fading.passed:
                score = -0.65
                conf = 0.88
                rationale = "Abverkauf vom Tageshoch (< 15m-EMA9)"
                decisive_idx = rec.idx(s_fading)
            elif s_below_vwap.passed:
                score = -0.50
                conf = 0.75
                rationale = f"Unter 15m-VWAP (${snap.intraday_vwap})"
                decisive_idx = rec.idx(s_below_vwap)

        elif focus == "FUNDAMENTALS_AND_NEWS":
            news_s = float(np.mean([n.sentiment_hint for n in snap.news])) if snap.news else 0.0
            s_rev = rec.num("Umsatzwachstum", snap.revenue_growth_pct, ">=", "min_rev_growth", 15.0, "bull")
            s_margin = rec.num("Gewinnmarge", snap.profit_margin_pct, ">=", "min_profit_margin", 8.0, "bull")
            s_news_bull = rec.num("News Sentiment Bull", news_s, ">=", "min_news_bull", 0.35, "bull")
            s_news_bear = rec.num("News Sentiment Bear", news_s, "<=", "max_news_bear", -0.35, "bear")
            s_insider = rec.flag("SEC Insider Buying", snap.insider_sentiment == "STRONG_INSIDER_BUY", "bull")
            if s_rev.passed and s_margin.passed:
                score += 0.45
                conf = 0.78
                rationale = f"Umsatz +{snap.revenue_growth_pct:.0f}%, Marge {snap.profit_margin_pct:.0f}%"
                decisive_idx.extend(rec.idx(s_rev, s_margin))
            if s_insider.passed:
                score = min(0.98, score + 0.40)
                conf = 0.90
                rationale += f" + 🏛️ SEC Form 4 Insider-Käufe!"
                decisive_idx.extend(rec.idx(s_insider))
            if s_news_bull.passed:
                score = min(0.95, score + 0.45)
                conf = 0.85
                rationale += f" + News-Katalysator ({news_s:+.2f})"
                decisive_idx.extend(rec.idx(s_news_bull))
            elif s_news_bear.passed:
                score -= 0.60
                conf = 0.82
                rationale += f"Negativer News-Flow ({news_s:+.2f})"
                decisive_idx.extend(rec.idx(s_news_bear))

        elif focus == "MACRO_INTERMARKET":
            s_risk_on = rec.flag("Macro Risk-On", macro.regime_state == "GREEN_RISK_ON", "bull")
            s_corr = rec.num("Portfolio Korrelation", snap.max_corr_with_portfolio, "<=", "max_corr_tolerance", 0.75, "bull")
            s_defensive = rec.flag("Macro Defensiv", macro.regime_state == "RED_DEFENSIVE" and not snap.is_safe_haven, "bear")
            if s_risk_on.passed and s_corr.passed:
                score = 0.35 + max(0.0, snap.cross_asset_opportunity_score * 0.5)
                conf = 0.72
                rationale = f"Risk-On & niedrige Portfolio-Korr (r={snap.max_corr_with_portfolio:+.2f})"
                decisive_idx = rec.idx(s_risk_on, s_corr)
            elif s_defensive.passed:
                score = -0.80
                conf = 0.90
                rationale = "Defensives Makro-Regime (VIX hoch)"
                decisive_idx = rec.idx(s_defensive)

        elif focus == "RISK_VETO_GUARD":
            news_s = float(np.mean([n.sentiment_hint for n in snap.news])) if snap.news else 0.0
            s_overbought = rec.num("Überhitzter RSI", snap.rsi_14, ">=", "overbought_rsi_veto", 81.0, "bear")
            s_news_override = rec.num("Starke News", news_s, ">=", "news_override", 0.50, "bull")
            s_blowoff_chg = rec.num("Blow-Off 24h%", snap.change_24h_pct, ">=", "blowoff_min_chg", 12.0, "bear")
            s_fading = rec.flag("Abverkauf vom Hoch", snap.intraday_15m_trend == "FADING_FROM_HIGH", "bear")
            s_safe_rsi = rec.num("Moderater RSI", snap.rsi_14, "<=", "safe_rsi_max", 72.0, "bull")
            s_above_vwap = rec.flag("Kurs > VWAP", snap.intraday_15m_trend == "BULLISH_ABOVE_VWAP", "bull")

            if s_overbought.passed and not s_news_override.passed:
                trace = VoteTrace(agent_id=ag.agent_id, symbol=snap.symbol, score=-0.90,
                                 vote_action="VETO_BUY", steps=rec.steps,
                                 decisive_idx=rec.idx(s_overbought), created_at=datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])
                return SubAgentVote(agent_id=ag.agent_id, agent_name=ag.name, symbol=snap.symbol,
                                   vote_action="VETO_BUY", score=-0.90, confidence=0.92,
                                   rationale=f"Überhitzte Bullenfalle (RSI {snap.rsi_14:.1f} >= {s_overbought.threshold:.0f} ohne News)"), trace
            if s_blowoff_chg.passed and s_fading.passed:
                trace = VoteTrace(agent_id=ag.agent_id, symbol=snap.symbol, score=-0.85,
                                 vote_action="VETO_BUY", steps=rec.steps,
                                 decisive_idx=rec.idx(s_blowoff_chg, s_fading), created_at=datetime.datetime.now(datetime.timezone.utc).isoformat()[:19])
                return SubAgentVote(agent_id=ag.agent_id, agent_name=ag.name, symbol=snap.symbol,
                                   vote_action="VETO_BUY", score=-0.85, confidence=0.90,
                                   rationale=f"Blow-Off-Top Gefahr (+{snap.change_24h_pct:.1f}%, kippt unter 15m-EMA9)"), trace
            if s_safe_rsi.passed and s_above_vwap.passed:
                score = 0.35
                conf = 0.75
                rationale = "Kein Überhitzungs- oder Blow-Off-Risiko"
                decisive_idx = rec.idx(s_safe_rsi, s_above_vwap)

        elif focus == "GAMMA_SQUEEZE":
            s_vol_oi = rec.num("Vol/OI Extreme", snap.unusual_options_vol_oi, ">=", "min_vol_oi_ratio", 3.5, "bull")
            s_pc_tight = rec.num("Put/Call Niedrig", snap.options_put_call_ratio, "<=", "max_pc_ratio", 0.40, "bull")
            if s_vol_oi.passed and s_pc_tight.passed:
                score = 0.92
                conf = 0.90
                rationale = f"Gamma-Squeeze Setup (V/OI {snap.unusual_options_vol_oi:.1f}x, P/C {snap.options_put_call_ratio})"
                decisive_idx = rec.idx(s_vol_oi, s_pc_tight)

        elif focus == "PEAD_DRIFT":
            s_gap = rec.num("Gap 24h%", snap.change_24h_pct, ">=", "min_gap_pct", 8.5, "bull")
            s_rvol = rec.num("PEAD RVOL", snap.volume_ratio_vs_20d, ">=", "min_rvol", 1.8, "bull")
            s_rsi_ceil = rec.num("RSI Decke", snap.rsi_14, "<=", "max_rsi_ceiling", 79.0, "bull")
            if s_gap.passed and s_rvol.passed and s_rsi_ceil.passed:
                score = 0.82
                conf = 0.86
                rationale = f"PEAD-Ausbruch (+{snap.change_24h_pct:.1f}% bei {snap.volume_ratio_vs_20d:.1f}x Volumen)"
                decisive_idx = rec.idx(s_gap, s_rvol, s_rsi_ceil)

        elif focus == "QUALITY_MEAN_REVERSION":
            s_oversold = rec.num("Überverkaufter RSI", snap.rsi_14, "<=", "oversold_rsi", 37.0, "bull")
            s_rev = rec.num("Umsatzstark", snap.revenue_growth_pct, ">=", "min_rev_growth", 15.0, "bull")
            s_above_vwap = rec.flag("Kurs > VWAP", snap.intraday_15m_trend == "BULLISH_ABOVE_VWAP", "bull")
            if s_oversold.passed and s_rev.passed and s_above_vwap.passed:
                score = 0.78
                conf = 0.84
                rationale = f"Überverkaufter Rebound (RSI {snap.rsi_14:.1f}, Umsatz +{snap.revenue_growth_pct:.0f}%)"
                decisive_idx = rec.idx(s_oversold, s_rev, s_above_vwap)

        elif focus == "SHORT_SELL_SPECIALIST":
            price_to_sma20 = (snap.current_price / snap.sma_20) if snap.sma_20 > 0 else 1.0
            s_sub_sma = rec.num("Kurs / SMA20", price_to_sma20, "<=", "max_price_to_sma20", 0.995, "bear")
            s_bear_intra = rec.flag("Intraday bärisch", snap.intraday_15m_trend in ("BEARISH_BELOW_VWAP", "FADING_FROM_HIGH"), "bear")
            s_macd_neg = rec.num("MACD negativ", snap.macd_hist, "<=", "max_macd_hist", 0.0, "bear")
            if s_sub_sma.passed and s_bear_intra.passed and s_macd_neg.passed:
                score = -0.82
                conf = 0.85
                rationale = "Short-Breakdown Setup (< SMA20 & < VWAP)"
                decisive_idx = rec.idx(s_sub_sma, s_bear_intra, s_macd_neg)

        elif focus == "XETRA_EUROPEAN_LEADERS":
            if snap.symbol.endswith(".DE"):
                s_de_chg = rec.num("Xetra 24h%", snap.change_24h_pct, ">=", "min_euro_chg", 1.2, "bull")
                s_de_vol = rec.num("Xetra RVOL", snap.volume_ratio_vs_20d, ">=", "min_euro_volume_ratio", 1.35, "bull")
                s_de_weak = rec.num("Xetra Schwäche", snap.change_24h_pct, "<=", "max_euro_chg_bear", -2.0, "bear")
                if s_de_chg.passed and s_de_vol.passed:
                    score = 0.75
                    conf = 0.82
                    rationale = f"Xetra Momentum-Leader (+{snap.change_24h_pct:.1f}%, Vol {snap.volume_ratio_vs_20d:.1f}x)"
                    decisive_idx = rec.idx(s_de_chg, s_de_vol)
                elif s_de_weak.passed:
                    score = -0.60
                    conf = 0.75
                    rationale = "Europäische Schwäche"
                    decisive_idx = rec.idx(s_de_weak)

        elif focus == "PENNY_STOCK_MOMENTUM":
            s_price = rec.num("Penny Preis", snap.current_price, "<=", "max_penny_price", 5.0, "bull")
            s_rvol = rec.num("Penny RVOL", snap.volume_ratio_vs_20d, ">=", "min_penny_rvol", 1.9, "bull")
            s_chg = rec.num("Penny 24h%", snap.change_24h_pct, ">=", "min_penny_chg", 3.8, "bull")
            s_dump = rec.num("Penny Dump", snap.change_24h_pct, "<=", "max_penny_dump", -5.5, "bear")
            if s_price.passed and s_rvol.passed and s_chg.passed:
                score = 0.82
                if snap.sec_form8k_catalyst:
                    score = min(0.98, score + 0.10)
                if snap.custom_metrics.get("chaikin_money_flow_accumulation"):
                    score = min(0.98, score + 0.05)
                conf = 0.88
                cat_txt = f" [⚡8-K: {snap.sec_form8k_catalyst[:35]}]" if snap.sec_form8k_catalyst else ""
                cmf_txt = " [CMF Akkumulation]" if snap.custom_metrics.get("chaikin_money_flow_accumulation") else ""
                rationale = f"🚀 Penny Stock Rakete: ${snap.current_price:.2f} mit {snap.volume_ratio_vs_20d:.1f}x RVOL & +{snap.change_24h_pct:.1f}% Ausbruch{cat_txt}{cmf_txt}"
                decisive_idx = rec.idx(s_price, s_rvol, s_chg)

            elif s_price.passed and s_dump.passed:
                score = -0.75
                conf = 0.80
                rationale = f"⚠️ Penny Stock Abverkauf: {snap.change_24h_pct:.1f}% Einbruch"
                decisive_idx = rec.idx(s_price, s_dump)

        elif focus == "TTM_SQUEEZE_EXPLOSION":
            is_fired = "FIRED" in snap.ttm_squeeze_status
            is_comp = "COMPRESSION" in snap.ttm_squeeze_status
            s_fired = rec.flag("TTM Squeeze Fired", is_fired, "bull")
            s_comp = rec.flag("TTM Compression", is_comp, "bull")
            s_mom = rec.num("Squeeze Momentum", snap.ttm_squeeze_momentum, ">=", "min_squeeze_mom", 0.05, "bull")
            if is_fired and snap.ttm_squeeze_momentum > 0:
                score = 0.80
                conf = 0.86
                rationale = f"🔥 TTM Squeeze Ausbruch zündet (Momentum: {snap.ttm_squeeze_momentum:+.2f})"
                decisive_idx = rec.idx(s_fired, s_mom)
            elif is_comp and snap.macd_hist > 0:
                score = 0.55
                conf = 0.72
                rationale = "⏳ Squeeze-Kompression im Aufbau kurz vor Ausbruch"
                decisive_idx = rec.idx(s_comp)

        elif focus == "INSIDER_ACCUMULATION":
            has_buys = snap.insider_sentiment == "STRONG_INSIDER_BUY"
            has_sells = snap.insider_sentiment == "INSIDER_SELLING"
            s_buy = rec.flag("Form-4 Insider Buy", has_buys, "bull")
            s_sell = rec.flag("Form-4 Insider Sell", has_sells, "bear")
            if has_buys:
                score = 0.75
                conf = 0.82
                rationale = f"👔 Top-Insider kaufen netto: {snap.insider_summary}"
                decisive_idx = rec.idx(s_buy)
            elif has_sells:
                score = -0.65
                conf = 0.76
                rationale = f"⚠️ Insider-Verkäufe: {snap.insider_summary}"
                decisive_idx = rec.idx(s_sell)

        elif focus == "SUPERTREND_MOMENTUM":
            is_bull = snap.supertrend_direction == "BULLISH"
            is_flip = snap.supertrend_flipped and is_bull
            s_st_bull = rec.flag("SuperTrend Bullish", is_bull, "bull")
            s_st_flip = rec.flag("SuperTrend Flip", is_flip, "bull")
            if is_flip:
                score = 0.85
                conf = 0.88
                rationale = f"🌊 SuperTrend Flip auf BULLISH bei ${snap.supertrend_val:.2f}"
                decisive_idx = rec.idx(s_st_flip)
            elif is_bull and snap.rsi_14 < 72.0:
                score = 0.65
                conf = 0.78
                rationale = f"🌊 SuperTrend Aufwärtstrend intakt (Stopp bei ${snap.supertrend_val:.2f})"
                decisive_idx = rec.idx(s_st_bull)
            elif snap.supertrend_direction == "BEARISH":
                score = -0.60
                conf = 0.75
                rationale = f"🌧️ SuperTrend bärisch (Widerstand ${snap.supertrend_val:.2f})"
                decisive_idx = rec.idx(s_st_bull)



        v_act = "BUY" if score >= 0.25 else ("SELL" if score <= -0.25 else "HOLD")
        vote = SubAgentVote(
            agent_id=ag.agent_id,
            agent_name=ag.name,
            symbol=snap.symbol,
            vote_action=v_act,
            score=round(score, 2),
            confidence=round(conf, 2),
            rationale=rationale,
        )
        trace = VoteTrace(
            agent_id=ag.agent_id,
            symbol=snap.symbol,
            score=round(score, 2),
            vote_action=v_act,
            steps=rec.steps,
            decisive_idx=decisive_idx,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat()[:19],
        )
        return vote, trace
