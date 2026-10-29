"""
Reflexions-Bausteine für die ADAS-Unteragenten:
Jeder Unteragent protokolliert seine Prüfschritte (VoteTrace). Nach einer deutlichen Kursbewegung wird das
Protokoll nachanalysiert: Welche Schritte haben das Urteil getragen, waren sie richtig oder Ursache eines Fehlers?
Wiederholt schuldige Schwellen werden in kleinen, begrenzten Schritten angepasst und später überprüft.
"""
import math
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

# Individuelle Schrittweiten & Grenzen (step, min, max) für Schwellen mit abweichender Skala
TUNING_OVERRIDES: Dict[str, Tuple[float, float, float]] = {
    "max_price_to_sma20": (0.005, 0.95, 1.02),
    "max_macd_hist": (0.02, -0.5, 0.2),
    "news_override": (0.05, 0.2, 0.9),
    "min_news_bull": (0.05, 0.15, 0.7),
    "max_news_bear": (0.05, -0.7, -0.15),
    "max_corr_tolerance": (0.02, 0.5, 0.9),
    "overbought_rsi_veto": (1.0, 72.0, 90.0),
    "safe_rsi_max": (1.0, 60.0, 80.0),
    "oversold_rsi": (1.0, 25.0, 45.0),
    "max_rsi_ceiling": (1.0, 70.0, 88.0),
    "max_pc_ratio": (0.03, 0.2, 0.7),
}

BLAME_THRESHOLD = 3           # So oft muss ein Schritt Fehlerursache sein, bevor seine Schwelle angepasst wird
REVIEW_AFTER_EVALS = 8        # Nach so vielen weiteren Auswertungen wird eine Anpassung überprüft
EVAL_MIN_MOVE_PCT = 1.2       # Ab dieser Kursbewegung wird ein Votum als richtig/falsch bewertet
NEAR_MISS_MIN_MOVE_PCT = 2.5  # Ab dieser Bewegung werden neutrale Voten auf knapp verfehlte Schritte geprüft
TRACE_MAX_AGE_DAYS = 3        # Voten ohne nennenswerte Bewegung verfallen danach unbewertet


def clean_float(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


class ReasoningStep(BaseModel):
    label: str
    observed: Optional[float] = None
    op: str = "=="            # ">=", "<=" (justierbar) oder "==" (kategorial)
    param_key: str = ""       # leer = kategorialer Schritt
    threshold: float = 1.0
    passed: bool = False
    direction: str = "bull"   # "bull" oder "bear"


class VoteTrace(BaseModel):
    agent_id: str
    symbol: str
    score: float
    vote_action: str
    steps: List[ReasoningStep] = Field(default_factory=list)
    decisive_idx: List[int] = Field(default_factory=list)
    created_at: str = ""


class ParamChange(BaseModel):
    param_key: str
    old_value: float
    new_value: float
    reason: str
    changed_at: str
    evals_at_change: int
    correct_at_change: int
    status: str = "PENDING"   # "PENDING", "CONFIRMED", "REVERTED"


class StepRecorder:
    """Protokolliert jeden Prüfschritt mit beobachtetem Wert, Schwelle und Ergebnis."""

    def __init__(self, rule_parameters: Dict[str, float], param_defaults: Dict[str, float]):
        self.params = rule_parameters
        self.defaults = param_defaults
        self.steps: List[ReasoningStep] = []

    def num(self, label: str, observed, op: str, key: str, default: float, direction: str) -> ReasoningStep:
        if key not in self.params:
            self.params[key] = float(default)
        if key not in self.defaults:
            self.defaults[key] = float(self.params[key])
        thr = float(self.params[key])
        obs = clean_float(observed)
        passed = obs is not None and (obs >= thr if op == ">=" else obs <= thr)
        st = ReasoningStep(label=label, observed=obs, op=op, param_key=key, threshold=thr, passed=passed, direction=direction)
        self.steps.append(st)
        return st

    def flag(self, label: str, condition: bool, direction: str) -> ReasoningStep:
        st = ReasoningStep(label=label, observed=1.0 if condition else 0.0, op="==", param_key="",
                           threshold=1.0, passed=bool(condition), direction=direction)
        self.steps.append(st)
        return st

    def idx(self, *steps: ReasoningStep) -> List[int]:
        """Indizes der übergebenen Schritte, die bestanden wurden (= haben das Urteil getragen)."""
        out = []
        for s in steps:
            if s.passed:
                for i, existing in enumerate(self.steps):
                    if existing is s:
                        out.append(i)
                        break
        return out


def fmt_step(s: ReasoningStep) -> str:
    mark = "✔" if s.passed else "✘"
    if not s.param_key:
        return f"{mark} {s.label}"
    obs = "n/a" if s.observed is None else f"{s.observed:.2f}"
    op = "≥" if s.op == ">=" else "≤"
    return f"{mark} {s.label} {obs} {op} {s.threshold:.2f}"


def tuning_for(key: str, param_defaults: Dict[str, float], rule_parameters: Dict[str, float]) -> Tuple[float, float, float]:
    if key in TUNING_OVERRIDES:
        return TUNING_OVERRIDES[key]
    d = param_defaults.get(key, rule_parameters.get(key, 1.0))
    spread = abs(d) * 0.5 if d != 0 else 1.0
    step = max(abs(d) * 0.05, 0.01)
    return step, d - spread, d + spread
