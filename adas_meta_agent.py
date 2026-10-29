import datetime
import json
import os
from typing import Dict, List, Optional, Tuple, Any
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
    evolution_epoch: int = 1
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

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig()
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

        # Stelle sicher, dass neu definierte Spezialisten immer sofort im Schwarm aktiv sind
        default_macro = MacroRegime(
            regime_state="GREEN_RISK_ON",
            vix_level=16.0,
            risk_multiplier=1.0,
            allowed_clusters=["*"],
            spy_price=500.0,
            spy_sma_50=490.0,
            spy_sma_200=470.0,
            summary="Standard",
        )
        for spec in self._detect_needed_specialist_agents({}, default_macro):
            if spec.agent_id not in self.state.active_subagents:
                self.state.active_subagents[spec.agent_id] = spec
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

            # Kollektiver Schwarm-Wissens-Austausch: Destilliere neue Erkenntnisse & Synergien
            try:
                from swarm_knowledge_bus import get_swarm_knowledge_bus
                k_bus = get_swarm_knowledge_bus()
                k_events = k_bus.record_collective_outcome(
                    symbol=sym,
                    move_pct=move_pct,
                    agent_votes=self.state.pending_traces.get(sym, {}),
                    is_penny=(snap.cluster == "PENNY_STOCKS" or snap.current_price < 5.0) if snap else False,
                    had_catalyst=bool(snap.sec_form8k_catalyst or getattr(snap, "form4_cluster_buy", False)) if snap else False,
                    macro_credit_stress=getattr(macro, "credit_stress_active", False),
                )
                events.extend(k_events)
            except Exception:
                pass

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

        # 3. Autonome Werkzeug-Synthese (ADAS AI Scientist / Sakana AI Paradigm):
        # Programmiert, validiert per AST und deployed quantitative Werkzeuge eigenständig in die Toolbox.
        try:
            from adas_code_synthesizer import ADASCodeSynthesizer
            syn = ADASCodeSynthesizer()
            new_tool = syn.synthesize_next_optimal_tool()
            if new_tool:
                t_msg = f"[{now_str[11:19]}] ⚙️ Autonome Werkzeug-Synthese: '{new_tool['name']}' ({new_tool['tool_id']}) verifiziert & in QuantitativeToolbox integriert."
                self.state.evolution_log.append(t_msg)
                events.append(t_msg)
        except Exception:
            pass

        # 4. Autonome genetische Strategie-Zucht (AlphaZero / Walk-Forward Crisis Backtesting):
        # Läuft periodisch alle 12 Zyklen oder initial wenn < 2 gezüchtete Elite-Strategien aktiv sind.
        try:
            active_bred = [ag for ag in self.state.active_subagents.values() if ag.specialty_focus in ("PENNY_CATALYST_ROCKET", "VOLATILITY_SQUEEZE_EXPLOSION", "WHALE_ORDERFLOW_ACCUMULATOR", "SUPERTREND_REGIME_RIDER")]
            if len(active_bred) < 2 or (self.state.total_evolution_cycles % 12 == 0):
                from genetic_strategy_breeder import GeneticStrategyBreeder
                breeder = GeneticStrategyBreeder(self.config)
                population = breeder.breed_generation(pop_size=4, generation_num=max(1, self.state.evolution_generation))
                for cand in population:
                    if cand.get("is_elite"):
                        ok, fpath = breeder.deploy_elite_strategy(cand)
                        if ok:
                            b_msg = f"[{now_str[11:19]}] 🧬 Elite-Strategie gezüchtet & scharfgeschaltet: '{cand['class_name']}' (Sharpe {cand['performance']['sharpe_ratio']:.2f}, Win-Rate {cand['performance']['win_rate_pct']:.1f}%)"
                            self.state.evolution_log.append(b_msg)
                            events.append(b_msg)
        except Exception:
            pass

        self.state.evolution_log = self.state.evolution_log[-30:]
        self._save()
        return events

    def _send_evolution_alert(self, spec: SubAgentSpec) -> None:
        """Sendet einen Push-Alert bei Geburt eines neuen Unteragenten (gedrosselt via NotificationManager)."""
        try:
            from notification_manager import get_notifier
            get_notifier().send_agent_spawn_alert(
                name=spec.name,
                role=spec.role,
                gen=spec.generation,
                trust=spec.trust_weight,
            )
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

        # 5. Options Smart-Money & Max-Pain Hunter
        if "options_smart_money_hunter" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="options_smart_money_hunter",
                name="⚡ Options Smart-Money & Max-Pain Hunter",
                role="Analysiert Put/Call-Ratio, Max-Pain-Magnetismus und Unusual Options Whale Flow.",
                specialty_focus="OPTIONS_SMART_MONEY",
                trust_weight=1.18,
            ))

        # 6. Fundamental Solvency & Health Guardian
        if "fundamental_health_guardian" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="fundamental_health_guardian",
                name="🛡️ Fundamental Solvency & Health Guardian",
                role="Prüft Cashflow, Solvenz und Umsatzwachstum und schützt das Kapital vor toxischen Pleitekandidaten.",
                specialty_focus="FUNDAMENTAL_HEALTH",
                trust_weight=1.20,
            ))

        # 7. Statistical Arbitrage & Pairs-Trading Specialist
        if "pairs_stat_arb_specialist" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="pairs_stat_arb_specialist",
                name="🔀 Statistical Arbitrage & Cointegration Specialist",
                role="Scannt hochkorrelierte Paare auf Z-Score Extremabweichungen (|Z| >= 2.0) und handelt statistische Mean-Reversion.",
                specialty_focus="STATISTICAL_ARBITRAGE",
                trust_weight=1.22,
            ))

        # 8. Hurst Fractal & Trend-Persistence Cycler
        if "fractal_regime_cycler" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="fractal_regime_cycler",
                name="🌀 Hurst Fractal & Regime Cycler",
                role="Analysiert den Hurst-Exponenten (H). Forciert Trend-Folge bei H > 0.58, Mean-Reversion bei H < 0.44 und drosselt im Random-Walk-Chop.",
                specialty_focus="FRACTAL_REGIME",
                trust_weight=1.20,
            ))

        # 9. High-Yield Credit & Intermarket Sentinel
        if "intermarket_credit_sentinel" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="intermarket_credit_sentinel",
                name="🚨 High-Yield Credit & Intermarket Sentinel",
                role="Überwacht Junk-Bond Credit-Spreads (HYG/LQD) und Dr. Copper. Schützt vor systemischen Abverkäufen, bevor sie den Aktienmarkt treffen.",
                specialty_focus="INTERMARKET_CREDIT",
                trust_weight=1.25,
            ))

        # 10. Order-Book Depth & Liquidity Sniper
        if "microstructure_liquidity_sniper" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="microstructure_liquidity_sniper",
                name="🎯 Order-Book Depth & Liquidity Sniper",
                role="Prüft Corwin-Schultz Bid-Ask-Spreads und Amihud-Illiquidität. Verhindert Slippage-Fallen und optimiert Einstiegs-Routing.",
                specialty_focus="MARKET_MICROSTRUCTURE",
                trust_weight=1.15,
            ))

        # 11. Sector Rotation & Relative-Strength Leader Hunter
        if "sector_rotation_hunter" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="sector_rotation_hunter",
                name="🚀 Sector Momentum & Relative-Strength Hunter",
                role="Scannt die 11 SPDR-Sektoren und allokiert in führende Alpha-Sektoren mit relativer Stärke vs. SPY.",
                specialty_focus="SECTOR_ROTATION",
                trust_weight=1.22,
            ))

        # 12. Wall Street Analyst Consensus & Price Target Hunter
        if "wall_street_consensus_analyst" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="wall_street_consensus_analyst",
                name="📊 Wall Street Consensus & Analyst Target Hunter",
                role="Analysiert Konsens-Ratings (Strong Buy / Buy) und Kursziel-Aufwärtspotenziale renommierter Investmentbanken.",
                specialty_focus="WALL_STREET_CONSENSUS",
                trust_weight=1.24,
            ))

        # 13. Geopolitical Crisis & Defense Strategist
        if "geopolitical_crisis_strategist" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="geopolitical_crisis_strategist",
                name="🌐 Geopolitical Regime & Defense Strategist",
                role="Überwacht globale Konflikt- und Energie-Risiken, steuert Defense-/Rohstoff-Hedges und schützt vor geopolitischen Schocks.",
                specialty_focus="GEOPOLITICAL_STRATEGY",
                trust_weight=1.25,
            ))

        # 14. Hugging Face FinBERT NLP & Sentiment Sentinel (Sub-Agent #27)
        if "huggingface_sentiment_sentinel" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="huggingface_sentiment_sentinel",
                name="🤗 Hugging Face FinBERT Sentiment Sentinel",
                role="Überwacht Finanznachrichten & SEC-Headlines über das Hugging Face ProsusAI/finbert Modell. Erkennt negative Sentiment-Schocks und schützt vor Bad-News-Kollaps.",
                specialty_focus="HUGGINGFACE_SENTIMENT",
                trust_weight=1.22,
                rule_parameters={"min_positive_score": 0.35, "max_negative_veto_score": -0.40},
            ))

        # 15. Institutional VWAP & Volume-Profile Liquidity Sweeper (Sub-Agent #28)
        if "institutional_vwap_liquidity_sweeper" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="institutional_vwap_liquidity_sweeper",
                name="🎯 Institutional VWAP & Liquidity Sweeper",
                role="Erkennt institutionelle Liquidity-Sweeps an den VWAP-Standardabweichungsbändern (±2σ) und Value-Area Grenzen (POC/VAL). Handelt antizyklische Rebounds, wenn Market-Maker Retail-Stop-Losses abfischen.",
                specialty_focus="VWAP_LIQUIDITY_SWEEP",
                trust_weight=1.24,
                rule_parameters={"min_sweep_edge": 0.50, "max_exhaustion_score": -0.60},
            ))

        # 16. Macro Economic Event Horizon Sentinel (Sub-Agent #29)
        if "macro_event_horizon_sentinel" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="macro_event_horizon_sentinel",
                name="🏛️ Macro Event Horizon Sentinel",
                role="Überwacht US-Zinsentscheide (Fed FOMC), Inflationsdaten (CPI) und Arbeitsmarktberichte (NFP). Blockiert Neukäufe 60 Min. vor bis 30 Min. nach Veröffentlichungen.",
                specialty_focus="MACRO_EVENT_HORIZON",
                trust_weight=1.28,
            ))

        # 17. Probabilistic Forecast & Quantile Asymmetry Sentinel (Sub-Agent #30)
        if "probabilistic_forecast_sentinel" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="probabilistic_forecast_sentinel",
                name="📈 Probabilistic Forecast & Quantile Sentinel",
                role="Berechnet probabilistische Kurstrichter (P10/P50/P90 Quantile) über Monte-Carlo/Bootstrapping. Genehmigt Trades nur bei asymmetrischem Upside-Vorteil (>= 1.6x).",
                specialty_focus="PROBABILISTIC_FORECAST",
                trust_weight=1.25,
            ))

        # 18. Multi-Timeframe Alignment & Confluence Specialist (Sub-Agent #31)
        if "multi_timeframe_alignment_specialist" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="multi_timeframe_alignment_specialist",
                name="📐 Multi-Timeframe Trend & Resonance Specialist",
                role="Prüft fraktale Zeitebenen-Resonanz (Daily Macro + 4H Swing + 1H Intraday). Erkennt A+ Chancen bei synchroner Trend-Konfluenz.",
                specialty_focus="TIMEFRAME_CONFLUENCE",
                trust_weight=1.26,
            ))

        # 19. Adverse Volatility & Liquidity Evaporation Sentinel (Sub-Agent #32)
        if "adverse_volatility_shield_sentinel" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="adverse_volatility_shield_sentinel",
                name="⚡ Adverse Volatility & Liquidity Evaporation Sentinel",
                role="Erkennt schlagartige Sturz-Geschwindigkeiten, Liquiditäts-Vakua und Volatilitäts-Spikes in Echtzeit und verhängt Notfall-Vetos.",
                specialty_focus="ADVERSE_VOLATILITY_SHIELD",
                trust_weight=1.28,
            ))

        # 20. Tactical Timing & Entry-Optimization Strategist (Sub-Agent #33)
        if "tactical_timing_optimizer" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="tactical_timing_optimizer",
                name="⏱️ Tactical Timing & Entry-Optimization Strategist",
                role="Wägt bei interessanten Setups & Options-Flows ab: Sofort-Einstieg (Momentum/Ausbruch) vs. geduldiges Abwarten auf Pullback/Retest (VWAP, EMA, Support) zur Renditemaximierung.",
                specialty_focus="TACTICAL_TIMING",
                trust_weight=1.27,
            ))

        # 21. Post-Earnings Announcement Drift (PEAD) Hunter (Sub-Agent #34)
        if "pead_drift_hunter" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="pead_drift_hunter",
                name="⚡ Post-Earnings Announcement Drift (PEAD) Hunter",
                role="Erkennt institutionelle Folgekäufe nach Quartalszahlen-Überraschungen (Earnings Beat + Gap-and-Go + hohes Volumen) und reitet den 3- bis 10-Tage Drift.",
                specialty_focus="PEAD_MOMENTUM",
                trust_weight=1.28,
            ))

        # 22. Biopharma & FDA Clinical Catalyst Sentinel (Sub-Agent #35)
        if "biopharma_catalyst_sentinel" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="biopharma_catalyst_sentinel",
                name="🧬 Biopharma & FDA Clinical Catalyst Sentinel",
                role="Überwacht FDA-Zulassungen, Phase-3 Studienergebnisse und PDUFA-Entscheidungen im US-Biotechsektor mit asymmetrischem Chance-Risiko-Profil.",
                specialty_focus="BIOPHARMA_CATALYST",
                trust_weight=1.24,
            ))

        # 23. Crypto-Macro Lead-Lag Divergence Specialist (Sub-Agent #36)
        if "crypto_macro_lead_lag_specialist" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="crypto_macro_lead_lag_specialist",
                name="⚡ Crypto-Macro Lead-Lag Divergence Specialist",
                role="Nutzt 24/7 Bitcoin/Ethereum Liquiditäts-Impulse und Wochenend-Divergenzen als Frühindikator für montägliche Eröffnungslücken von Tech- und Krypto-Aktien.",
                specialty_focus="CRYPTO_LEAD_LAG",
                trust_weight=1.22,
            ))

        # 24. Short-Squeeze & High Cost-to-Borrow Hunter (Sub-Agent #37)
        if "short_squeeze_ctb_hunter" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="short_squeeze_ctb_hunter",
                name="🪤 Short-Squeeze & Extreme CTB Hunter",
                role="Identifiziert Aktien mit überhöhtem Short Interest (>15%), steigenden Leihezinsen (Cost-to-Borrow) und beginnender Short-Covering-Kettenreaktion.",
                specialty_focus="SHORT_SQUEEZE_CTB",
                trust_weight=1.25,
            ))

        # 25. TradingView Consensus & 26-Indicator Matrix Sentinel (Sub-Agent #38)
        if "tradingview_consensus_sentinel" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="tradingview_consensus_sentinel",
                name="📊 TradingView Consensus & 26-Matrix Sentinel",
                role="Prüft die 26 offiziellen TradingView Oszillatoren & Moving Averages. Verhindert Käufe gegen technischen Gegenwind (STRONG_SELL Veto) und verleiht Conviction-Boost bei STRONG_BUY.",
                specialty_focus="TRADINGVIEW_CONSENSUS",
                trust_weight=1.26,
            ))

        # 26. Dark Pool & Off-Exchange Block Trade Sentinel (Sub-Agent #39)
        if "dark_pool_block_sentinel" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="dark_pool_block_sentinel",
                name="🌑 Dark Pool & Off-Exchange Block Sentinel",
                role="Erkennt verdeckte institutionelle Akkumulation in Dark Pools (ATS-Quote >45%) und schützt vor verdeckter Distribution / Short-Druck.",
                specialty_focus="DARK_POOL_SENTINEL",
                trust_weight=1.28,
            ))

        # 27. SEC Edgar Filing Scout & Catalyst Extractor (Sub-Agent #40)
        if "sec_edgar_scout" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="sec_edgar_scout",
                name="📑 SEC Edgar Regulatory Filing Scout",
                role="Überwacht SEC Form 4 (Cluster- und Whale-Käufe) sowie Form 8-K (Material Events, Partnerschaften, Dilution Warnings) und postet Evidenz-Pakete auf das Blackboard.",
                specialty_focus="SEC_EDGAR_SCOUT",
                trust_weight=1.30,
            ))

        # 28. Microsoft Qlib Alpha158 Quantitative Factor Miner (Sub-Agent #41)
        if "qlib_alpha158_miner" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="qlib_alpha158_miner",
                name="📐 Microsoft Qlib Alpha158 Factor Miner",
                role="Berechnet 158 tiefgehende quantitative Faktoren (K-Line Geometrie, Multi-Horizont Momentum, Volatilitäts-Skew, Preis-Volumen-Korrelation). Verleiht Conviction-Boost bei multifaktorieller Konfluenz.",
                specialty_focus="QLIB_ALPHA158",
                trust_weight=1.32,
            ))

        # 29. Smart Recovery & Thesis Invalidation Auditor (Sub-Agent #42)
        if "recovery_thesis_auditor" not in self.state.active_subagents:
            new_specs.append(SubAgentSpec(
                agent_id="recovery_thesis_auditor",
                name="🛡️ Smart Recovery & Thesis Invalidation Auditor",
                role="Überwacht Positionen im Drawdown. Prüft Rebound-Potenzial (RSI-Erschöpfung, VWAP-Support, Qlib Alpha158, Dark-Pool-Dip-Buying) vs. These-Bruch und schützt vor vorzeitigem Stop-Hunting.",
                specialty_focus="SMART_RECOVERY",
                trust_weight=1.30,
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
                # Synergie-Multiplikator aus dem kollektiven Wissens-Bus
                try:
                    from swarm_knowledge_bus import get_swarm_knowledge_bus
                    k_bus = get_swarm_knowledge_bus()
                    bull_ids = [v.agent_id for v in votes if v.score >= 0.28]
                    syn_mult = k_bus.get_synergy_boost_for_agents(bull_ids)
                    boost *= syn_mult
                except Exception:
                    syn_mult = 1.0
                new_conf = round(min(0.97, new_conf + boost), 2)
                syn_str = f" · Synergie {syn_mult:.2f}x" if syn_mult > 1.0 else ""
                swarm_tag = f" 🐝 [Sakana-Konsens +{swarm_consensus:.2f}: {', '.join(bull_supporters[:3])}{syn_str}]"
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

        # 🏛️ CIO Conviction Synthesis (virattt / TradingAgents Komitee-Hierarchie)
        cio_tier = "TIER_2_SOLID_OPPORTUNITY"
        cio_score = 0.50
        cio_summary = ""
        try:
            from cio_conviction_synthesizer import get_cio_synthesizer
            cio = get_cio_synthesizer()
            cio_res = cio.synthesize_conviction(
                snap.symbol, votes, self.state.active_subagents, baseline_signal, macro, snap
            )
            cio_tier = cio_res.conviction_tier
            cio_score = cio_res.overall_conviction_score
            cio_summary = cio_res.executive_summary
            if cio_res.synthesized_action == "HOLD" and new_action == "BUY":
                new_action = "HOLD"
                swarm_tag += f" 🏛️ [{cio_tier}: CIO schützt Kapital]"
            elif cio_tier == "TIER_1_ALPHA_HIGH_CONVICTION":
                new_conf = round(min(0.98, new_conf + 0.05), 2)
                swarm_tag += " 🏛️ [CIO: TIER 1 ALPHA]"
        except Exception as e:
            cio_summary = f"CIO-Synthesizer Fallback ({e})"

        # 📋 Blackboard & A2A Stigmergic Envelope Coordination
        try:
            from swarm_blackboard import get_swarm_blackboard, A2AMessageEnvelope
            bb = get_swarm_blackboard()
            
            # 1. Prüfe bestehende Blackboard-Konversationen für das Symbol
            topic_ctx = bb.get_topic_context(f"ALPHA::{snap.symbol}")
            if topic_ctx:
                if topic_ctx.status == "VETOED_CLOSED" and new_action == "BUY":
                    new_action = "HOLD"
                    swarm_tag += " 📋 [Blackboard: EMERGENCY_VETO aktiv]"
                elif topic_ctx.status == "CONSENSUS_REACHED" and topic_ctx.leading_action == "BUY" and new_action == "BUY":
                    new_conf = round(min(0.99, new_conf + 0.03), 2)
                    swarm_tag += " 📋 [Blackboard: A2A-Konsens]"

            # 2. Poste das aggregierte Meta-Schwarm Votum als strukturierte A2A Envelope
            bb_intent = "PROPOSE_TRADE" if new_action in ("BUY", "SELL") else "CHALLENGE_CRITIQUE"
            if veto_reasons:
                bb_intent = "EMERGENCY_VETO"
            envelope = A2AMessageEnvelope(
                topic=f"ALPHA::{snap.symbol}",
                from_agent="adas_meta_architect",
                to_agent="ALL",
                intent=bb_intent,
                symbol=snap.symbol,
                confidence=new_conf,
                payload={
                    "action": new_action,
                    "swarm_consensus": round(swarm_consensus, 2),
                    "bull_supporters": bull_supporters[:3],
                    "veto_reasons": veto_reasons[:2],
                },
                summary_de=f"ADAS Schwarm-Votum: {new_action} ({new_conf*100:.0f}% Konfidenz)",
            )
            bb.post_envelope(envelope)
        except Exception as e:
            logger.debug(f"Blackboard-Protokoll Exception: {e}")

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
            cio_conviction_tier=cio_tier,
            cio_conviction_score=cio_score,
            cio_committee_consensus=cio_summary,
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
            has_strong_buys = snap.insider_sentiment == "STRONG_INSIDER_BUY" or getattr(snap, "form4_cluster_buy", False)
            has_mod_buys = snap.insider_sentiment == "MODERATE_INSIDER_BUY" or getattr(snap, "form4_net_usd_90d", 0.0) > 50000.0
            has_sells = snap.insider_sentiment == "INSIDER_SELLING"
            s_cluster = rec.flag("Form-4 Cluster Buy", getattr(snap, "form4_cluster_buy", False), "bull")
            s_buy = rec.flag("Form-4 Whale Buy", has_strong_buys, "bull")
            s_sell = rec.flag("Form-4 Insider Sell", has_sells, "bear")
            if has_strong_buys:
                score = 0.88 if getattr(snap, "form4_cluster_buy", False) else 0.80
                conf = 0.90
                cluster_tag = " [👥 Cluster-Buying]" if getattr(snap, "form4_cluster_buy", False) else ""
                rationale = f"👔 Top-Insider & Whales kaufen netto: {snap.insider_summary}{cluster_tag}"
                decisive_idx = rec.idx(s_cluster if getattr(snap, "form4_cluster_buy", False) else s_buy)
            elif has_mod_buys:
                score = 0.50
                conf = 0.72
                rationale = f"Insider-Käufe vorhanden: {snap.insider_summary}"
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

        elif focus == "OPTIONS_SMART_MONEY":
            flow_s = getattr(snap, "options_flow_score", 0.0)
            pcr = getattr(snap, "options_put_call_ratio", 1.0)
            max_p = getattr(snap, "options_max_pain", 0.0)
            s_opt_bull = rec.num("Options Flow Bull", flow_s, ">=", "min_flow_score", 0.35, "bull")
            s_opt_bear = rec.num("Options Flow Bear", flow_s, "<=", "max_flow_bear", -0.35, "bear")
            if s_opt_bull.passed:
                score = 0.82
                conf = 0.85
                pain_txt = f" (MaxPain ${max_p:.2f})" if max_p and max_p > 0 else ""
                rationale = f"⚡ Bullischer Smart-Money Call-Flow (Score {flow_s:+.2f}){pain_txt}"
                decisive_idx = rec.idx(s_opt_bull)
            elif s_opt_bear.passed:
                score = -0.70
                conf = 0.80
                rationale = f"🌧️ Starker Put-Flow / Absicherung (P/C {pcr or 1.2:.2f})"
                decisive_idx = rec.idx(s_opt_bear)

        elif focus == "FUNDAMENTAL_HEALTH":
            f_score = getattr(snap, "fundamental_score", 0.0)
            f_grade = getattr(snap, "fundamental_health_grade", "SOLID_VALUE")
            is_dist = getattr(snap, "is_fundamentally_distressed", False)
            s_elite = rec.flag("Elite Growth", f_grade == "ELITE_GROWTH", "bull")
            s_dist = rec.flag("Toxic Distressed", is_dist or f_grade == "TOXIC_DISTRESSED", "bear")
            if s_dist.passed:
                score = -0.95
                conf = 0.95
                rationale = "🛑 VETO: Fundamentale Insolvenz- oder Verwässerungsgefahr (Negativer Cashflow & Schulden)"
                decisive_idx = rec.idx(s_dist)
            elif s_elite.passed:
                score = 0.85
                conf = 0.88
                rationale = f"💎 Elite-Fundamentaldaten & hohes Umsatzwachstum (Score {f_score:+.2f})"
                decisive_idx = rec.idx(s_elite)
            elif f_score >= 0.15:
                score = 0.45
                conf = 0.75
                rationale = f"Solide Bilanz & Cashflow (Score {f_score:+.2f})"
                decisive_idx = rec.idx(s_elite)

        elif focus == "STATISTICAL_ARBITRAGE":
            z = getattr(snap, "stat_arb_z_score", None)
            pair = getattr(snap, "stat_arb_pair", "")
            thesis_txt = getattr(snap, "stat_arb_thesis", "")
            if z is not None:
                s_stat_arb = rec.num("StatArb Z-Divergenz", abs(z), ">=", "min_z_score", 1.8, "bull")
                if s_stat_arb.passed:
                    score = min(0.95, 0.50 + abs(z) * 0.18)
                    conf = 0.88
                    rationale = f"🔀 StatArb Mean-Reversion Opportunity in Paar {pair} (Z: {z:+.2f})"
                    decisive_idx = rec.idx(s_stat_arb)
                else:
                    score = 0.10
                    conf = 0.60
                    rationale = f"Paar {pair} im neutralen Z-Bereich ({z:+.2f})"
            else:
                score = 0.0
                conf = 0.50
                rationale = "Keine aktive Pairs-Trading Kointegration für dieses Symbol"

        elif focus == "FRACTAL_REGIME":
            h = getattr(snap, "hurst_exponent", 0.50)
            f_mode = getattr(snap, "fractal_regime_mode", "RANDOM_WALK_CHOP")
            s_trend = rec.num("Hurst Persistenter Trend", h, ">=", "min_hurst_trend", 0.58, "bull")
            s_rev = rec.num("Hurst Mean-Reversion", h, "<=", "max_hurst_mean_rev", 0.44, "bull")
            s_chop = rec.flag("Random Walk Chop", f_mode == "RANDOM_WALK_CHOP", "bear")
            if s_trend.passed:
                score = min(0.92, 0.55 + (h - 0.50) * 1.5)
                conf = 0.86
                rationale = f"🌀 Persistenter Trend-Modus (Hurst {h:.2f} >= 0.58) – Ausbrüche statistisch begünstigt"
                decisive_idx = rec.idx(s_trend)
            elif s_rev.passed and snap.rsi_14 <= 42.0:
                score = 0.78
                conf = 0.82
                rationale = f"🌀 Anti-persistente Mean-Reversion (Hurst {h:.2f}) bei überverkauftem RSI ({snap.rsi_14:.1f})"
                decisive_idx = rec.idx(s_rev)
            elif s_chop.passed:
                score = -0.35
                conf = 0.75
                rationale = f"Rauschiger Random-Walk Markt (Hurst {h:.2f}) – Gefahr von Fehlausbrüchen"
                decisive_idx = rec.idx(s_chop)

        elif focus == "INTERMARKET_CREDIT":
            cred_stress = getattr(macro, "credit_stress_active", False)
            bias = getattr(macro, "intermarket_bias", "NEUTRAL_STABLE")
            s_c_stress = rec.flag("Credit Crunch Stress", cred_stress, "bear")
            s_c_bull = rec.flag("Credit Tailwind", bias == "BULLISH_CREDIT_TAILWIND", "bull")
            if s_c_stress.passed:
                score = -0.92
                conf = 0.94
                rationale = "🚨 VETO: Kredit-Spreads kollabieren (HYG/LQD) – Systemischer Abverkauf droht!"
                decisive_idx = rec.idx(s_c_stress)
            elif s_c_bull.passed:
                score = 0.72
                conf = 0.82
                rationale = f"🟢 Kreditmärkte entspannt (HYG/LQD {getattr(macro, 'credit_spread_hyg_lqd', 0.75):.3f}) ➜ Bullischer Rückenwind"
                decisive_idx = rec.idx(s_c_bull)
            else:
                score = 0.15
                conf = 0.65
                rationale = f"Kredit- & Rohstoffmärkte stabil ({bias})"

        elif focus == "MARKET_MICROSTRUCTURE":
            spread = getattr(snap, "bid_ask_spread_pct", 0.10)
            amihud = getattr(snap, "amihud_illiquidity", 0.0)
            routing = getattr(snap, "microstructure_routing", "MARKET_EXECUTION_SAFE")
            s_trap = rec.flag("Slippage Trap", routing == "SLIPPAGE_TRAP_AVOID" or spread >= 0.70, "bear")
            s_tight = rec.num("Enge Spreads", spread, "<=", "max_tight_spread", 0.15, "bull")
            if s_trap.passed:
                score = -0.88
                conf = 0.90
                rationale = f"🛑 Dünnes Orderbuch (Spread {spread:.2f}%, Amihud {amihud:.3f}) – Hohes Slippage-Risiko"
                decisive_idx = rec.idx(s_trap)
            elif s_tight.passed:
                score = 0.60
                conf = 0.80
                rationale = f"🎯 Enge Spreads ({spread:.2f}%) & tiefe institutionelle Liquidität"
                decisive_idx = rec.idx(s_tight)

        elif focus == "SECTOR_ROTATION":
            try:
                from sector_momentum_expander import SectorMomentumExpander
                sec_exp = SectorMomentumExpander()
                sec_res = sec_exp.scan_sector_rotation()
                is_hot_stock = snap.symbol in sec_res.recommended_hot_stocks
                s_sector_lead = rec.flag("Sector Leader Stock", is_hot_stock, "bull")
                if s_sector_lead.passed:
                    score = 0.84
                    conf = 0.86
                    rationale = f"🚀 Sektor-Alpha Leader: Gehört zu den Top-Aktien des führenden Sektors ({sec_res.top_alpha_sector})"
                    decisive_idx = rec.idx(s_sector_lead)
                else:
                    score = 0.20
                    conf = 0.65
                    rationale = f"Sektor im neutralen Mittelfeld (Leader: {sec_res.top_alpha_sector})"
            except Exception:
                score = 0.0
                conf = 0.50
                rationale = "Sektor-Rotations-Daten temporär nicht verfügbar"

        elif focus == "WALL_STREET_CONSENSUS":
            cons = getattr(snap, "analyst_consensus", "NEUTRAL")
            upside = getattr(snap, "analyst_target_upside_pct", None)
            cnt = getattr(snap, "analyst_count", 0)
            s_strong_buy = rec.flag("Analyst Strong Buy", cons in ("STRONG_BUY", "BUY"), "bull")
            s_upside = rec.num("Analyst Upside %", upside or 0.0, ">=", "min_analyst_upside", 15.0, "bull")
            s_bear_rating = rec.flag("Analyst Underperform", cons in ("UNDERPERFORM", "SELL"), "bear")
            if s_bear_rating.passed:
                score = -0.75
                conf = 0.85
                rationale = f"⚠️ Wall Street Analysten warnen: Konsens ist {cons}"
                decisive_idx = rec.idx(s_bear_rating)
            elif s_strong_buy.passed and s_upside.passed:
                score = min(0.92, 0.60 + min(30.0, upside or 0.0) * 0.01)
                conf = min(0.95, 0.75 + min(30, cnt) * 0.005)
                rationale = f"📊 Starker Wall Street Konsens ({cons}, {cnt} Analysten): +{upside:.1f}% Kurspotenzial"
                decisive_idx = rec.idx(s_strong_buy, s_upside)
            elif s_strong_buy.passed:
                score = 0.45
                conf = 0.70
                rationale = f"Wall Street Konsens {cons} ({cnt} Analysten)"
                decisive_idx = rec.idx(s_strong_buy)
            else:
                score = 0.0
                conf = 0.50
                rationale = f"Analysten-Meinung unauffällig ({cons})"

        elif focus == "GEOPOLITICAL_STRATEGY":
            geo_score = getattr(macro, "geopolitical_risk_score", 3.5)
            geo_mode = getattr(macro, "geopolitical_regime_mode", "NEUTRAL_STABLE")
            geo_mandate = getattr(macro, "geopolitical_strategy_mandate", "BALANCED_MULTI_ASSET")
            is_def_or_energy = snap.cluster in ("DEFENSE", "ENERGY") or snap.symbol in ("LMT", "XLE", "USO", "GLD", "RHM.DE")
            s_escalation = rec.flag("Geopolitische Eskalation", geo_score >= 6.0 or geo_mode == "ESCALATION_CRISIS_REGIME", "bear")
            s_def_hedge = rec.flag("Defense Hedge Asset", is_def_or_energy and geo_score >= 5.0, "bull")
            if s_def_hedge.passed:
                score = 0.86
                conf = 0.90
                rationale = f"🛡️ Geopolitischer Krisen-Profiteur ({geo_mode}): Safe-Haven / Rüstung / Rohstoff-Absicherung aktiv"
                decisive_idx = rec.idx(s_def_hedge)
            elif s_escalation.passed and not snap.is_safe_haven:
                score = -0.80
                conf = 0.88
                rationale = f"🌐 VETO: Hohes weltpolitisches Risiko (GCI {geo_score:.1f}/10, {geo_mode}) – Mandat: {geo_mandate[:50]}"
                decisive_idx = rec.idx(s_escalation)
            elif geo_mode == "PEACE_DIVIDEND_EXPANSION":
                score = 0.65
                conf = 0.78
                rationale = f"🕊️ Friedensdividende / Geopolitische Entspannung (GCI {geo_score:.1f}/10) begünstigt globales Wachstum"
                decisive_idx = rec.idx(rec.flag("Peace Dividend", True, "bull"))
            else:
                score = 0.10
                conf = 0.60
                rationale = f"Geopolitische Lage kontrolliert (GCI {geo_score:.1f}/10, {geo_mode})"

        elif focus == "HUGGINGFACE_SENTIMENT":
            try:
                from huggingface_sentiment_sentinel import get_hf_sentiment_sentinel
                hf_sent = get_hf_sentiment_sentinel()
                if hasattr(snap, "hf_sentiment_score") and snap.hf_sentiment_score != 0.0:
                    hf_score = snap.hf_sentiment_score
                    neg_p = snap.hf_negative_prob
                    pos_p = snap.hf_positive_prob
                    label = snap.hf_sentiment_label
                else:
                    hf_snap = hf_sent.analyze_symbol(snap.symbol)
                    hf_score = hf_snap.compound_score
                    neg_p = hf_snap.negative_prob
                    pos_p = hf_snap.positive_prob
                    label = hf_snap.sentiment_label

                s_veto = rec.flag("FinBERT Negative Shock", neg_p >= 0.55 or hf_score <= -0.35, "bear")
                s_bull = rec.num("FinBERT Bullish", hf_score, ">=", "min_positive_score", 0.35, "bull")

                if s_veto.passed:
                    score = -0.88
                    conf = 0.90
                    rationale = f"🚨 VETO: Hugging Face FinBERT erkennt negativen Sentiment-Schock ({label}, Neg {neg_p*100:.0f}%)"
                    decisive_idx = rec.idx(s_veto)
                elif s_bull.passed:
                    score = min(0.90, 0.55 + pos_p * 0.35)
                    conf = min(0.94, 0.70 + pos_p * 0.25)
                    rationale = f"🤗 Bullisches FinBERT-Sentiment ({label}, Score {hf_score:+.2f}, Pos {pos_p*100:.0f}%)"
                    decisive_idx = rec.idx(s_bull)
                else:
                    score = 0.10 if hf_score >= 0 else -0.10
                    conf = 0.60
                    rationale = f"Hugging Face FinBERT Sentiment unauffällig ({label})"
            except Exception as e:
                score = 0.0
                conf = 0.50
                rationale = f"FinBERT-Sentiment temporär nicht verfügbar ({e})"

        elif focus == "VWAP_LIQUIDITY_SWEEP":
            try:
                sweep_st = getattr(snap, "liquidity_sweep_status", "NONE")
                z_vwap = getattr(snap, "vwap_z_score", 0.0)
                poc_px = getattr(snap, "volume_profile_poc", 0.0)

                s_bull_sweep = rec.flag("Bullish Liquidity Sweep", sweep_st == "BULLISH_SWEEP_RECLAIM", "bull")
                s_deep_dip = rec.flag("VWAP -2 Sigma Dip", z_vwap <= -1.8, "bull")
                s_climax_top = rec.flag("Bearish Climax Sweep", sweep_st == "BEARISH_CLIMAX_SWEEP" or z_vwap >= 2.4, "bear")

                if s_bull_sweep.passed:
                    score = 0.88
                    conf = 0.90
                    target_txt = f" Rebound-Ziel: POC ${poc_px:.2f}" if poc_px and poc_px > 0 else ""
                    rationale = f"🎯 Bullischer Liquidity-Sweep bestätigt: Stop-Losses unter -2σ absorbiert.{target_txt}"
                    decisive_idx = rec.idx(s_bull_sweep)
                elif s_climax_top.passed:
                    score = -0.78
                    conf = 0.85
                    rationale = f"⚠️ Bärische Buying-Exhaustion am +2σ VWAP-Band ({z_vwap:+.1f}σ) – Korrektur droht"
                    decisive_idx = rec.idx(s_climax_top)
                elif s_deep_dip.passed:
                    score = 0.55
                    conf = 0.72
                    rationale = f"Unterkühlt am -2σ VWAP-Band ({z_vwap:+.1f}σ) – Potenzieller Dip-Kauf"
                    decisive_idx = rec.idx(s_deep_dip)
                else:
                    score = 0.10 if z_vwap < 0 else -0.10
                    conf = 0.60
                    rationale = f"Kurs in Value Area unauffällig (VWAP Z: {z_vwap:+.1f}σ)"
            except Exception as e:
                score = 0.0
                conf = 0.50
                rationale = f"VWAP-Sweeper temporär nicht verfügbar ({e})"

        elif focus == "MACRO_EVENT_HORIZON":
            try:
                from macro_event_horizon_sentinel import get_macro_event_sentinel
                ev_sent = get_macro_event_sentinel()
                snap_ev = ev_sent.analyze_event_horizon()

                is_active = getattr(macro, "macro_event_horizon_active", False) or snap_ev.event_horizon_active
                is_caution = getattr(macro, "macro_event_protective_action", "") == "HALF_SIZE_ONLY" or snap_ev.horizon_state == "APPROACHING_CAUTION"
                ev_name = getattr(macro, "macro_event_horizon_name", "") or snap_ev.next_event_name
                mins = getattr(macro, "macro_minutes_to_event", snap_ev.minutes_to_next_event)

                s_event_active = rec.flag("Event Horizon Active", is_active, "bear")
                s_caution = rec.flag("Approaching Event", is_caution, "bear")

                if s_event_active.passed:
                    score = -0.95
                    conf = 0.98
                    rationale = f"🏛️ VETO: Macro Event Horizon aktiv ({ev_name}) – Whipsaw-Schutz aktiv!"
                    decisive_idx = rec.idx(s_event_active)
                elif s_caution.passed:
                    score = 0.20
                    conf = 0.70
                    rationale = f"⏳ Makro-Event naht in {mins} Min ({ev_name}) – Vorsichtige Positionierung"
                    decisive_idx = rec.idx(s_caution)
                else:
                    score = 0.50
                    conf = 0.75
                    rationale = f"🟢 Horizont frei: Nächstes Notenbank-Event ({ev_name}) in {mins//60}h"
            except Exception as e:
                score = 0.0
                conf = 0.50
                rationale = f"Makro-Event Radar temporär nicht verfügbar ({e})"

        elif focus == "PROBABILISTIC_FORECAST":
            try:
                asym = getattr(snap, "forecast_asymmetry_ratio", 1.0)
                p10 = getattr(snap, "forecast_p10_price", 0.0)
                p50 = getattr(snap, "forecast_p50_price", 0.0)
                p90 = getattr(snap, "forecast_p90_price", 0.0)
                fav = getattr(snap, "forecast_favorable", False)

                s_fav = rec.flag("Favorable Asymmetry", fav or asym >= 1.6, "bull")
                s_skew = rec.flag("Unfavorable Skew", asym <= 0.75 and asym > 0, "bear")

                if s_fav.passed:
                    score = min(0.92, 0.60 + min(0.30, (asym - 1.0) * 0.20))
                    conf = 0.88
                    rationale = f"📈 Asymmetrischer Quantil-Trichter: {asym:.2f}x Upside/Downside (P90 ${p90:.2f} vs P10 ${p10:.2f})"
                    decisive_idx = rec.idx(s_fav)
                elif s_skew.passed:
                    score = -0.80
                    conf = 0.85
                    rationale = f"⚠️ VETO: Ungünstiger Downside-Skew ({asym:.2f}x) – P10 Risiko (${p10:.2f}) übersteigt Chance"
                    decisive_idx = rec.idx(s_skew)
                else:
                    score = 0.10 if asym >= 1.0 else -0.10
                    conf = 0.60
                    rationale = f"Symmetrischer Prognose-Fächer (Asymmetrie {asym:.2f}x)"
            except Exception as e:
                score = 0.0
                conf = 0.50
                rationale = f"Probabilistischer Trichter temporär nicht verfügbar ({e})"

        elif focus == "TIMEFRAME_CONFLUENCE":
            try:
                tf_status = getattr(snap, "timeframe_alignment_status", "NEUTRAL")
                tf_score = getattr(snap, "timeframe_alignment_score", 0.0)
                htf = getattr(snap, "higher_tf_trend", "NEUTRAL")

                is_stacked_bull = snap.current_price >= snap.sma_20 >= snap.sma_50
                is_stacked_bear = snap.current_price <= snap.sma_20 <= snap.sma_50

                s_triple_bull = rec.flag("Triple Bull Alignment", tf_status == "TRIPLE_BULLISH_ALIGNMENT" or (tf_score >= 0.70 and is_stacked_bull), "bull")
                s_triple_bear = rec.flag("Triple Bear Alignment", tf_status == "TRIPLE_BEARISH_ALIGNMENT" or (tf_score <= -0.70 and is_stacked_bear), "bear")

                if s_triple_bull.passed:
                    score = 0.88
                    conf = 0.90
                    rationale = f"📐 Perfekte Multi-Timeframe Resonanz: Daily + 4H + 1H synchron bullisch (Score {tf_score:+.2f})"
                    decisive_idx = rec.idx(s_triple_bull)
                elif s_triple_bear.passed:
                    score = -0.85
                    conf = 0.88
                    rationale = f"🔻 Bärische Zeitebenen-Konfluenz: Alle Timeframes abwärtsgerichtet (Score {tf_score:+.2f})"
                    decisive_idx = rec.idx(s_triple_bear)
                elif tf_status == "DUAL_BULLISH_ALIGNMENT" or is_stacked_bull:
                    score = 0.50
                    conf = 0.75
                    rationale = f"Solide Zeitebenen-Unterstützung (Trend: {htf})"
                else:
                    score = -0.20 if is_stacked_bear else 0.0
                    conf = 0.60
                    rationale = f"Zeitebenen uneinheitlich (Status: {tf_status})"
            except Exception as e:
                score = 0.0
                conf = 0.50
                rationale = f"Multi-Timeframe Engine temporär nicht verfügbar ({e})"

        elif focus == "ADVERSE_VOLATILITY_SHIELD":
            try:
                alert = getattr(snap, "proactive_risk_alert", "NORMAL")
                vel = getattr(snap, "adverse_price_velocity_pct", 0.0)
                spike = getattr(snap, "volatility_spike_ratio", 1.0)
                void = getattr(snap, "liquidity_void_detected", False)

                s_void = rec.flag("Liquidity Void Hazard", void or alert == "LIQUIDITY_VOID_EVACUATE", "bear")
                s_shock = rec.flag("Volatility Shock", spike >= 2.2 or alert == "SURGE_VOLATILITY_HAZARD" or vel <= -2.5, "bear")
                s_stable = rec.flag("Microstructure Stability", spike < 1.4 and vel > -1.0 and not void, "bull")

                if s_void.passed:
                    score = -0.98
                    conf = 0.95
                    rationale = f"🚨 NOTFALL-VETO: Liquiditäts-Vakuum im Orderbuch erkannt (Spread/Tiefe kollabiert)"
                    decisive_idx = rec.idx(s_void)
                elif s_shock.passed:
                    score = -0.85
                    conf = 0.90
                    rationale = f"⚡ VETO: Volatilitäts-Schock (Spike {spike:.1f}x, Sturz-Tempo {vel:+.1f}%)"
                    decisive_idx = rec.idx(s_shock)
                elif s_stable.passed:
                    score = 0.45
                    conf = 0.80
                    rationale = f"🛡️ Mikrostruktur ruhig & liquide (Spike {spike:.1f}x, Tempo {vel:+.1f}%)"
                    decisive_idx = rec.idx(s_stable)
                else:
                    score = 0.0
                    conf = 0.65
                    rationale = f"Volatilität im Normalbereich ({spike:.1f}x)"
            except Exception as e:
                score = 0.0
                conf = 0.50
                rationale = f"Risk Shield temporär nicht verfügbar ({e})"

        elif focus == "TACTICAL_TIMING":
            try:
                cur_p = snap.current_price
                vwap = snap.intraday_vwap or cur_p
                vwap_gap_pct = ((cur_p - vwap) / vwap * 100.0) if vwap > 0 else 0.0
                rsi = snap.rsi_14
                rvol = snap.volume_ratio
                opt_flow = getattr(snap, "unusual_options_activity_detected", False)
                vah = getattr(snap, "volume_profile_vah", None)

                # 1. Fall: Stark überdehnter Kurs (Pullback abwarten!)
                is_overextended = (
                    vwap_gap_pct >= 1.8
                    or rsi >= 70.0
                    or (vah is not None and vah > 0 and cur_p > vah * 1.018)
                )
                # 2. Fall: Stark überverkauft bei Abwärtstrend (Relief-Bounce abwarten für Verkauf!)
                is_oversold_dump = rsi <= 28.0

                # 3. Fall: Frisches Ausbruchs-Momentum / Whale-Optionen direkt am Support (Sofort kaufen!)
                is_fresh_momentum = (
                    (rvol >= 1.8 or opt_flow)
                    and vwap_gap_pct < 1.4
                    and rsi < 68.0
                )

                s_wait_pullback = rec.flag("Wait For Pullback / Dip", is_overextended, "bear")
                s_immediate_entry = rec.flag("Immediate Breakout Execution", is_fresh_momentum, "bull")
                s_wait_bounce = rec.flag("Wait For Relief Bounce", is_oversold_dump, "bear")

                if s_wait_pullback.passed:
                    # Setup ist attraktiv, aber Einstieg wird verzögert
                    score = 0.35
                    conf = 0.88
                    rationale = (
                        f"⏳ ABWARTEN AUF PULLBACK: Setup attraktiv, aber intraday überhitzt "
                        f"(+{vwap_gap_pct:.1f}% über VWAP, RSI {rsi:.1f}). Besserer Einstieg nahe VWAP ${vwap:,.2f} abwarten!"
                    )
                    decisive_idx = rec.idx(s_wait_pullback)
                elif s_immediate_entry.passed:
                    score = 0.85
                    conf = 0.90
                    rationale = (
                        f"⚡ SOFORT-EINSTIEG: Frischer Momentum-Ausbruch (RVol {rvol:.1f}x) "
                        f"direkt an Basis/VWAP – sofort ausführen vor Slippage!"
                    )
                    decisive_idx = rec.idx(s_immediate_entry)
                elif s_wait_bounce.passed:
                    score = -0.30
                    conf = 0.82
                    rationale = (
                        f"⏳ ABWARTEN AUF REBOUND: Überverkauft (RSI {rsi:.1f}). Kein Panik-Verkauf am Tiefstkurs; "
                        f"besseren Ausstieg bei kurzem Relief-Bounce abwarten!"
                    )
                    decisive_idx = rec.idx(s_wait_bounce)
                else:
                    score = 0.20 if cur_p >= vwap else -0.10
                    conf = 0.65
                    rationale = f"⏱️ Einstiegs-Timing neutral/solide (VWAP-Abstand: {vwap_gap_pct:+.1f}%, RSI {rsi:.1f})"
            except Exception as e:
                score = 0.0
                conf = 0.50
                rationale = f"Tactical Timing Engine temporär nicht verfügbar ({e})"

        elif focus == "PEAD_MOMENTUM":
            try:
                days_earn = getattr(snap, "days_to_earnings", None)
                earn_rel = getattr(snap, "earnings_date_relative", "")
                is_post_earnings = (
                    (days_earn is not None and -5 <= days_earn <= 0)
                    or earn_rel in ("YESTERDAY", "TODAY_MORNING", "RECENT")
                    or (snap.change_24h_pct >= 4.5 and snap.volume_ratio >= 1.6)
                )
                chg_24h = snap.change_24h_pct
                rvol = snap.volume_ratio
                cur_p = snap.current_price
                vwap = snap.intraday_vwap or cur_p

                if is_post_earnings and chg_24h >= 3.5 and rvol >= 1.5 and cur_p >= vwap * 0.992:
                    score = min(1.0, 0.70 + (chg_24h / 25.0) * 0.30)
                    conf = min(0.96, 0.85 + (rvol / 5.0) * 0.11)
                    rationale = (
                        f"⚡ PEAD-DRIFT AKTIV: Explosiver Earnings-Beat (+{chg_24h:+.1f}%, Vol {rvol:.1f}x). "
                        f"Institutionelle Anschlusskäufe treiben mehrtägigen Aufwärtstrend (Post-Earnings Announcement Drift)!"
                    )
                elif is_post_earnings and chg_24h <= -4.5 and rvol >= 1.5:
                    score = max(-1.0, -0.75 - (abs(chg_24h) / 25.0) * 0.25)
                    conf = 0.88
                    rationale = (
                        f"⚡ PEAD-ABVERKAUF: Schwache Zahlen & Guidance (-{abs(chg_24h):.1f}%, Vol {rvol:.1f}x). "
                        f"Institutioneller Ausstieg setzt sich fort."
                    )
                else:
                    score = 0.05
                    conf = 0.50
                    rationale = "Kein akuter Post-Earnings-Drift (Neutrale Zwischenphase)."
            except Exception as e:
                score = 0.0
                conf = 0.50
                rationale = f"PEAD Hunter temporär neutral ({e})"

        elif focus == "BIOPHARMA_CATALYST":
            rvol_val = getattr(snap, "volume_ratio_vs_20d", getattr(snap, "volume_ratio", 1.0))
            has_bio_cat = snap.sec_form8k_catalyst or snap.cluster == "BIOTECH"
            vol_boost = rvol_val >= 1.5
            chg = snap.change_24h_pct
            s_cat = rec.flag("Biopharma Catalyst Present", has_bio_cat, "bull")
            s_vol = rec.flag("Biopharma Volume Surge", vol_boost, "bull")
            if s_cat.passed and chg >= 2.0 and s_vol.passed:
                score = min(0.95, 0.70 + (chg / 30.0) * 0.25)
                conf = 0.90
                cat_desc = f" [{snap.sec_form8k_catalyst[:40]}]" if snap.sec_form8k_catalyst else ""
                rationale = f"🧬 Biopharma A+ Catalyst: FDA/Trial Move +{chg:.1f}% bei {rvol_val:.1f}x Vol{cat_desc}"
                decisive_idx = rec.idx(s_cat, s_vol)
            else:
                score = 0.05
                conf = 0.50
                rationale = "Biopharma Catalyst neutral"

        elif focus == "CRYPTO_LEAD_LAG":
            rvol_val = getattr(snap, "volume_ratio_vs_20d", getattr(snap, "volume_ratio", 1.0))
            is_crypto_sym = snap.symbol in ("BTC-USD", "ETH-USD", "SOL-USD") or snap.cluster == "CRYPTO"
            is_crypto_stock = snap.symbol in ("COIN", "MSTR", "MARA", "RIOT", "CLSK")
            vol_hi = rvol_val >= 1.4
            chg = snap.change_24h_pct
            if (is_crypto_sym or is_crypto_stock) and chg >= 2.5 and vol_hi:
                score = 0.86
                conf = 0.89
                rationale = f"⚡ Krypto-Liquiditäts Lead-Lag Momentum: +{chg:.1f}% Ausbruch bei {rvol_val:.1f}x Vol"
            elif (is_crypto_sym or is_crypto_stock) and chg <= -3.5:
                score = -0.75
                conf = 0.82
                rationale = f"⚠️ Krypto-Schwäche Vorläufer: {chg:.1f}% Rücksetzer"
            else:
                score = 0.0
                conf = 0.50
                rationale = "Krypto Lead-Lag neutral"

        elif focus == "SHORT_SQUEEZE_CTB":
            rvol_val = getattr(snap, "volume_ratio_vs_20d", getattr(snap, "volume_ratio", 1.0))
            high_vol = rvol_val >= 1.8
            chg = snap.change_24h_pct
            whale_opt = getattr(snap, "unusual_options_activity_detected", False) or snap.unusual_options_vol_oi >= 1.8
            if high_vol and chg >= 3.5 and (whale_opt or getattr(snap, "intraday_vwap_rs", 0.0) > 3.0 or snap.relative_strength_vs_spy > 3.0):
                score = 0.88
                conf = 0.92
                rationale = f"🪤 Short-Squeeze Kettenreaktion: RVOL {rvol_val:.1f}x & +{chg:.1f}% Move mit starkem Smart Money Druck"
            else:
                score = 0.05
                conf = 0.50
                rationale = "Short-Squeeze Kriterien neutral"

        elif focus == "TRADINGVIEW_CONSENSUS":
            try:
                tv_rec = getattr(snap, "tradingview_recommendation", None)
                tv_buy = getattr(snap, "tradingview_buy_count", None)
                tv_sell = getattr(snap, "tradingview_sell_count", None)
                tv_sum = getattr(snap, "tradingview_summary", "")

                if not tv_rec:
                    from tradingview_sentinel import get_tradingview_sentinel
                    tv_analysis = get_tradingview_sentinel().analyze_symbol(snap.symbol)
                    tv_rec = tv_analysis.recommendation
                    tv_buy = tv_analysis.buy_count
                    tv_sell = tv_analysis.sell_count
                    tv_sum = tv_analysis.summary_text

                s_strong_buy = rec.flag("TV Strong Buy", tv_rec == "STRONG_BUY" or (tv_buy is not None and tv_buy >= 14), "bull")
                s_buy = rec.flag("TV Buy", tv_rec == "BUY", "bull")
                s_veto_sell = rec.flag("TV Strong Sell Veto", tv_rec in ("STRONG_SELL", "SELL") and (tv_sell is not None and tv_sell >= 10), "bear")

                if s_strong_buy.passed:
                    b_cnt = tv_buy if tv_buy is not None else 16
                    score = min(0.96, 0.75 + (b_cnt / 26.0) * 0.22)
                    conf = 0.92
                    rationale = f"📊 TradingView Matrix stark bullisch: {tv_rec} ({tv_buy} Buy / {tv_sell} Sell) – Hohe Konfluenz!"
                    decisive_idx = rec.idx(s_strong_buy)
                elif s_buy.passed:
                    score = 0.70
                    conf = 0.85
                    rationale = f"📊 TradingView Bestätigung: {tv_rec} ({tv_buy} Buy / {tv_sell} Sell)"
                    decisive_idx = rec.idx(s_buy)
                elif s_veto_sell.passed:
                    score = -0.88
                    conf = 0.90
                    rationale = f"🛑 VETO: TradingView Oszillatoren & Moving Averages melden Gegenwind ({tv_rec}: {tv_sell} Sell-Indikatoren)"
                    decisive_idx = rec.idx(s_veto_sell)
                else:
                    score = 0.05
                    conf = 0.55
                    rationale = f"TradingView Matrix neutral ({tv_sum or tv_rec})"
            except Exception as e:
                score = 0.05
                conf = 0.50
                rationale = f"TradingView Sentinel neutral ({e})"

        elif focus == "DARK_POOL_SENTINEL":
            try:
                dp_ratio = getattr(snap, "dark_pool_ratio", None)
                dp_sent = getattr(snap, "dark_pool_sentiment", None)
                dp_alert = getattr(snap, "dark_pool_block_alert", False)
                if dp_ratio is None or dp_sent is None:
                    from dark_pool_sentinel import get_dark_pool_sentinel
                    dp_res = get_dark_pool_sentinel().analyze_symbol(
                        snap.symbol,
                        volume_ratio_vs_20d=getattr(snap, "volume_ratio_vs_20d", 1.0),
                        current_price=snap.current_price,
                        vwap=getattr(snap, "vwap", None),
                    )
                    dp_ratio = dp_res.dark_pool_ratio
                    dp_sent = dp_res.sentiment
                    dp_alert = dp_res.block_trade_detected

                s_accum = rec.flag("Dark Pool Akkumulation", dp_sent == "ACCUMULATION" or (dp_ratio is not None and dp_ratio >= 0.48), "bull")
                s_dist = rec.flag("Dark Pool Distribution", dp_sent == "DISTRIBUTION", "bear")
                s_block = rec.flag("Block Trade Alert", dp_alert, "bull")

                if s_dist.passed:
                    score = -0.82
                    conf = 0.88
                    rationale = f"🛑 VETO: Verdeckte Dark-Pool Distribution / Short-Druck ({dp_ratio*100:.1f}% ATS-Ratio)"
                    decisive_idx = rec.idx(s_dist)
                elif s_accum.passed:
                    score = min(0.95, 0.76 + (dp_ratio - 0.45) * 0.8)
                    conf = 0.90
                    rationale = f"🌑 Institutionelle Dark-Pool Akkumulation ({dp_ratio*100:.1f}% ATS-Quote, Whales akkumulieren)"
                    decisive_idx = rec.idx(s_accum)
                else:
                    score = 0.05
                    conf = 0.55
                    rationale = f"Dark Pool Flow unauffällig ({dp_ratio*100:.1f}% Ratio)"
            except Exception as e:
                score = 0.05
                conf = 0.50
                rationale = f"Dark Pool Sentinel neutral ({e})"

        elif focus == "SEC_EDGAR_SCOUT":
            try:
                sec_cat = getattr(snap, "sec_catalyst_type", None)
                sec_sent = getattr(snap, "sec_catalyst_sentiment", None)
                sec_8k = getattr(snap, "sec_form8k_catalyst", "")
                if sec_cat is None:
                    from sec_edgar_scout import get_sec_edgar_scout
                    scout_res = get_sec_edgar_scout().scan_symbol_filings(
                        snap.symbol, sec_form8k_text=sec_8k, post_to_blackboard=False
                    )
                    sec_cat = scout_res.catalyst_type
                    sec_sent = scout_res.sentiment

                s_cluster = rec.flag("SEC Cluster/Whale Buy", sec_cat in ("INSIDER_CLUSTER_BUY", "WHALE_BUY"), "bull")
                s_pos_8k = rec.flag("SEC Positive 8-K", sec_cat == "FORM_8K_CATALYST", "bull")
                s_dilution = rec.flag("SEC Verwässerung VETO", sec_cat == "DILUTION_WARNING" or sec_sent == "BEARISH", "bear")

                if s_dilution.passed:
                    score = -0.88
                    conf = 0.92
                    rationale = f"🛑 VETO: SEC EDGAR Verwässerungs- oder Sell-Warnung ({sec_cat})"
                    decisive_idx = rec.idx(s_dilution)
                elif s_cluster.passed:
                    score = 0.92
                    conf = 0.93
                    rationale = f"📑 SEC EDGAR Top-Katalysator: {sec_cat} (Whales & Vorstände kaufen mit eigenem Privatgeld)"
                    decisive_idx = rec.idx(s_cluster)
                elif s_pos_8k.passed:
                    score = 0.80
                    conf = 0.85
                    rationale = "📑 SEC Form 8-K Material Event Katalysator bestätigt"
                    decisive_idx = rec.idx(s_pos_8k)
                else:
                    score = 0.05
                    conf = 0.55
                    rationale = "SEC EDGAR Meldungen unauffällig / neutral"
            except Exception as e:
                score = 0.05
                conf = 0.50
                rationale = f"SEC Edgar Scout neutral ({e})"

        elif focus == "QLIB_ALPHA158":
            try:
                qlib_sc = getattr(snap, "qlib_alpha158_score", None)
                qlib_rc = getattr(snap, "qlib_alpha158_recommendation", None)
                if qlib_sc is None:
                    from qlib_alpha158_engine import get_qlib_alpha158_engine
                    res = get_qlib_alpha158_engine().analyze_symbol(snap.symbol, current_price=snap.current_price)
                    qlib_sc = res.composite_alpha_score
                    qlib_rc = res.signal_recommendation

                s_strong_buy = rec.flag("Qlib Strong Buy", qlib_rc == "STRONG_BUY" or (qlib_sc is not None and qlib_sc >= 0.40), "bull")
                s_buy = rec.flag("Qlib Buy", qlib_rc == "BUY" or (qlib_sc is not None and qlib_sc >= 0.20), "bull")
                s_veto = rec.flag("Qlib Bear Veto", qlib_rc in ("STRONG_SELL", "SELL") or (qlib_sc is not None and qlib_sc <= -0.30), "bear")

                if s_veto.passed:
                    score = -0.85
                    conf = 0.90
                    rationale = f"🛑 VETO: Microsoft Qlib Alpha158 signalisiert quantitativen Gegenwind ({qlib_sc:+.2f})"
                    decisive_idx = rec.idx(s_veto)
                elif s_strong_buy.passed:
                    score = min(0.96, 0.78 + (qlib_sc if qlib_sc is not None else 0.45) * 0.35)
                    conf = 0.92
                    rationale = f"📐 Microsoft Qlib Alpha158 stark bullisch (+{qlib_sc:.2f}, quantitative Konfluenz)"
                    decisive_idx = rec.idx(s_strong_buy)
                elif s_buy.passed:
                    score = 0.70
                    conf = 0.82
                    rationale = f"📐 Microsoft Qlib Alpha158 bestätigt (+{qlib_sc:.2f})"
                    decisive_idx = rec.idx(s_buy)
                else:
                    score = 0.05
                    conf = 0.55
                    rationale = f"Qlib Alpha158 neutral ({qlib_sc:+.2f})"
            except Exception as e:
                score = 0.05
                conf = 0.50
                rationale = f"Qlib Alpha158 neutral ({e})"

        elif focus == "SMART_RECOVERY":
            try:
                rsi_val = getattr(snap, "rsi_14", 50.0)
                is_extreme_oversold = rsi_val <= 28.0
                is_oversold = rsi_val <= 35.0
                qlib_sc = getattr(snap, "qlib_alpha158_score", 0.0)
                dp_sent = getattr(snap, "dark_pool_sentiment", "")
                vwap_lvl = getattr(snap, "vwap_val", getattr(snap, "intraday_vwap", 0.0)) or 0.0
                has_vwap_support = vwap_lvl > 0 and abs(snap.current_price - vwap_lvl) / vwap_lvl <= 0.015
                has_sma_support = (snap.sma_50 > 0 and abs(snap.current_price - snap.sma_50) / snap.sma_50 <= 0.015)

                s_rebound = rec.flag("Extreme Oversold Rebound", is_extreme_oversold, "bull")
                s_support = rec.flag("Support Rebound Zone", is_oversold and (has_vwap_support or has_sma_support or qlib_sc >= 0.10 or dp_sent in ("BULLISH_FLOW", "ACCUMULATION")), "bull")
                s_thesis_break = rec.flag("Thesis Broken Breakdown", rsi_val >= 75.0 or (snap.change_24h_pct < -4.5 and not is_oversold), "bear")

                if s_rebound.passed:
                    score = 0.88
                    conf = 0.90
                    rationale = f"🛡️ Rebound-Verteidigung: Extrem überverkauft (RSI {rsi_val:.1f}) – Hohes Erholungspotenzial!"
                    decisive_idx = rec.idx(s_rebound)
                elif s_support.passed:
                    score = 0.74
                    conf = 0.85
                    rationale = f"🛡️ Erholungszone: Support stützt Kurs bei RSI {rsi_val:.1f} & positivem Zufluss."
                    decisive_idx = rec.idx(s_support)
                elif s_thesis_break.passed:
                    score = -0.75
                    conf = 0.80
                    rationale = "⚡ These-Bruch: Überhitzung oder freier Fall ohne Support."
                    decisive_idx = rec.idx(s_thesis_break)
                else:
                    score = 0.10
                    conf = 0.55
                    rationale = "Normale Marktzone – Erholungspotenzial neutral."
            except Exception as e:
                score = 0.0
                conf = 0.40
                rationale = f"Smart Recovery Auditor neutral ({e})"

        elif focus == "PENNY_CATALYST_ROCKET":
            rvol_val = getattr(snap, "volume_ratio_vs_20d", getattr(snap, "volume_ratio", 1.0))
            min_rvol = ag.rule_parameters.get("min_rvol", 1.95)
            min_chg = ag.rule_parameters.get("min_change_pct", 3.8)
            max_p = ag.rule_parameters.get("penny_max_price", 5.20)
            s_p = rec.num("Penny Max Preis", snap.current_price, "<=", "penny_max_price", max_p, "bull")
            s_rv = rec.num("Penny RVOL", rvol_val, ">=", "min_rvol", min_rvol, "bull")
            s_ch = rec.num("Penny Ausbruch", snap.change_24h_pct, ">=", "min_change_pct", min_chg, "bull")
            if s_p.passed and s_rv.passed and s_ch.passed:
                score = 0.86
                conf = 0.91
                rationale = f"🧬 Gezüchtete Elite-Strategie (Penny Catalyst Rocket): RVOL {rvol_val:.1f}x & +{snap.change_24h_pct:.1f}% Ausbruch"
                decisive_idx = rec.idx(s_p, s_rv, s_ch)
            else:
                score = 0.05
                conf = 0.50
                rationale = "Penny Catalyst Rocket neutral"

        elif focus == "VOLATILITY_SQUEEZE_EXPLOSION":
            rvol_val = getattr(snap, "volume_ratio_vs_20d", getattr(snap, "volume_ratio", 1.0))
            min_rvol = ag.rule_parameters.get("min_rvol", 1.50)
            min_chg = ag.rule_parameters.get("min_change_pct", 1.5)
            s_rv = rec.num("Squeeze RVOL", rvol_val, ">=", "min_rvol", min_rvol, "bull")
            s_ch = rec.num("Squeeze Ausbruch", snap.change_24h_pct, ">=", "min_change_pct", min_chg, "bull")
            is_sqz = "FIRED" in snap.ttm_squeeze_status or snap.macd_hist > 0
            if is_sqz and s_rv.passed and s_ch.passed:
                score = 0.85
                conf = 0.89
                rationale = f"🧬 Gezüchtete Elite-Strategie (Vol Squeeze Explosion): Ausbruch mit RVOL {rvol_val:.1f}x und MACD {snap.macd_hist:+.2f}"
                decisive_idx = rec.idx(s_rv, s_ch)
            else:
                score = 0.05
                conf = 0.50
                rationale = "Vol Squeeze Explosion neutral"

        elif focus == "WHALE_ORDERFLOW_ACCUMULATOR":
            rvol_val = getattr(snap, "volume_ratio_vs_20d", getattr(snap, "volume_ratio", 1.0))
            min_rvol = ag.rule_parameters.get("min_rvol", 1.65)
            has_whale = getattr(snap, "unusual_options_activity_detected", False) or snap.unusual_options_vol_oi >= 1.8 or getattr(snap, "options_call_put_ratio", 1.0) >= 1.5
            s_rv = rec.num("Whale RVOL", rvol_val, ">=", "min_rvol", min_rvol, "bull")
            s_wh = rec.flag("Whale Accumulation", has_whale, "bull")
            if s_rv.passed and s_wh.passed:
                score = 0.87
                conf = 0.92
                rationale = f"🧬 Gezüchtete Elite-Strategie (Whale Flow Accumulator): Smart Money Akkumulation & RVOL {rvol_val:.1f}x"
                decisive_idx = rec.idx(s_rv, s_wh)
            else:
                score = 0.05
                conf = 0.50
                rationale = "Whale Accumulator neutral"

        elif focus == "SUPERTREND_REGIME_RIDER":
            rvol_val = getattr(snap, "volume_ratio_vs_20d", getattr(snap, "volume_ratio", 1.0))
            min_rvol = ag.rule_parameters.get("min_rvol", 1.35)
            is_st_green = getattr(snap, "supertrend_direction", "BEARISH") == "BULLISH" or getattr(snap, "higher_tf_trend", "NEUTRAL") == "BULLISH_TREND"
            is_above_ma = snap.current_price >= snap.sma_50
            s_st = rec.flag("SuperTrend Bullish", is_st_green and is_above_ma, "bull")
            s_rv = rec.num("Trend RVOL", rvol_val, ">=", "min_rvol", min_rvol, "bull")
            if s_st.passed and s_rv.passed:
                score = 0.85
                conf = 0.89
                rationale = f"🧬 Gezüchtete Elite-Strategie (SuperTrend Regime Rider): Etablierter Megatrend & RVOL {rvol_val:.1f}x"
                decisive_idx = rec.idx(s_st, s_rv)
            else:
                score = 0.05
                conf = 0.50
                rationale = "SuperTrend Regime Rider neutral"

        # Autonome Werkzeug-Erweiterung: Berücksichtige vom ADAS AI Scientist synthetisierte Indikatoren
        if snap.custom_metrics:
            synth_boost = 0.0
            tool_reasons = []
            if snap.custom_metrics.get("vw_macd_signal") == "BULLISH_SURGE":
                synth_boost += 0.04
                tool_reasons.append("VW-MACD Surge")
            if snap.custom_metrics.get("rvi_status") == "VOLATILITY_EXPANSION_UP":
                synth_boost += 0.04
                tool_reasons.append("RVI Exp")
            if snap.custom_metrics.get("hull_ma_hma_trend") == "BULLISH":
                synth_boost += 0.04
                tool_reasons.append("HMA Bullish")
            if snap.custom_metrics.get("choppiness_index_regime") == "STRONG_TREND":
                synth_boost += 0.03
                tool_reasons.append("CHOP Trend")
            if snap.custom_metrics.get("elder_ray_status") == "STRONG_BULL_DOMINANCE":
                synth_boost += 0.03
                tool_reasons.append("Elder Ray Bull")
            if snap.custom_metrics.get("twiggs_mf_status") == "BULLISH_FLOW":
                synth_boost += 0.03
                tool_reasons.append("Twiggs Flow")

            if synth_boost > 0 and score > 0.20:
                score = min(1.0, score + synth_boost)
                conf = min(0.98, conf + synth_boost * 0.5)
                rationale += f" [⚙️ Synthetisierte Tools: {'/'.join(tool_reasons[:3])}]"

        # Kollektiver Schwarm-Wissens-Austausch (Cross-Agent Consultation via SwarmKnowledgeBus)
        try:
            from swarm_knowledge_bus import get_swarm_knowledge_bus
            k_bus = get_swarm_knowledge_bus()
            has_cat = bool(snap.sec_form8k_catalyst or getattr(snap, "form4_cluster_buy", False) or getattr(snap, "analyst_consensus", "") in ("STRONG_BUY", "BUY"))
            is_p = snap.cluster == "PENNY_STOCKS" or snap.current_price < 5.0
            credit_st = getattr(macro, "credit_stress_active", False)
            h_val = getattr(snap, "hurst_exponent", 0.50)

            k_adj, k_notes = k_bus.consult_shared_knowledge(
                symbol=snap.symbol,
                cluster=snap.cluster,
                specialty_focus=focus,
                current_score=score,
                has_catalyst=has_cat,
                is_penny=is_p,
                macro_credit_stress=credit_st,
                hurst_exponent=h_val,
            )
            if k_adj != 0.0:
                score = float(np.clip(score + k_adj, -1.0, 1.0))
                conf = float(np.clip(conf + abs(k_adj) * 0.35, 0.40, 0.98))
                if k_notes:
                    rationale += f" · {' · '.join(k_notes[:1])}"
        except Exception:
            pass

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

    def execute_nightly_bayesian_swarm_evolution(self) -> Dict[str, Any]:
        """
        Autonome nächtliche Schwarm-Evolution & Bayesianische Stimmgewicht-Rekalibrierung:
        - Führt nach Börsenschluss eine statistische Rekalibrierung aller 34 Unteragenten durch.
        - Agenten mit hoher Trefferquote und positivem Edge erhalten automatisches Gewicht-Upgrade.
        - Irreführende Unteragenten werden automatisch im Stimmgewicht gedrosselt.
        """
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()[:10]
        evolved_agents = []
        upgraded_count = 0
        downgraded_count = 0

        for ag_id, ag in self.state.active_subagents.items():
            tot = ag.signals_evaluated
            if tot >= 3:
                win_rate = ag.correct_votes / tot
                bayes_factor = 0.50 + win_rate
                new_weight = float(np.clip(ag.trust_weight * 0.85 + bayes_factor * 0.15, 0.40, 1.80))
                if new_weight > ag.trust_weight + 0.01:
                    upgraded_count += 1
                elif new_weight < ag.trust_weight - 0.01:
                    downgraded_count += 1
                ag.trust_weight = round(new_weight, 3)
                evolved_agents.append((ag.name, ag.trust_weight, win_rate))

        self.state.evolution_epoch += 1
        self._save()

        return {
            "date": now_str,
            "epoch": self.state.evolution_epoch,
            "active_subagents_count": len(self.state.active_subagents),
            "upgraded_count": upgraded_count,
            "downgraded_count": downgraded_count,
            "top_performers": sorted(evolved_agents, key=lambda x: x[1], reverse=True)[:5],
        }


# Alias für bequemen Import
ADASMetaAgent = ADASMetaArchitect
