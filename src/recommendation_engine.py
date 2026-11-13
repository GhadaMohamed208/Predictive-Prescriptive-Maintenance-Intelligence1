"""Transparent, configurable rules registry for maintenance decision support.

Every recommendation is produced by an explicit Rule with an id, trigger, evidence,
source, limitation and verification step. Rules never assert a physical root cause.

Evidence-strength categories (defined, not statistical confidences):
  Strong      - the rule's trigger is corroborated by at least one independent indicator
                (e.g. low predicted RUL AND persistent anomaly flags or large sensor shifts).
  Moderate    - a single model-based or statistical indicator triggered the rule.
  Informational - a data-quality / context notice; no engineering finding is implied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import config as C

STATUSES = ["Pending", "In Progress", "Completed", "Dismissed"]

# What kind of statement a recommendation makes. Only a physical inspection can produce a confirmed fault,
# so no rule here ever emits that class.
FC_ANOMALY = "Detected anomaly"
FC_SUSPECT = "Suspected degradation (model-based, not confirmed)"
FC_ROUTINE = "Routine monitoring (no abnormal finding)"
FC_DQ = "Data-quality notice"
FC_REC = "Recommended action"
FINDING_CLASS_HELP = {
    FC_ANOMALY: "Measured behaviour that differs from the asset's own earlier behaviour. It is an observation, not a diagnosis.",
    FC_SUSPECT: "A model-based estimate that the asset may be degrading. It is a suspicion that needs verification, not a proven fault.",
    FC_ROUTINE: "Nothing abnormal was found in the available data.",
    FC_DQ: "Something about the input data limits how far the results can be trusted.",
    FC_REC: "A suggested next step for a qualified maintenance engineer.",
}
CONFIRMATION_TEXT = ("No fault is confirmed by this system. A fault is confirmed only by a physical inspection or test "
                     "carried out by an authorised engineer.")
PRIORITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}

SRC_APP = "Application-configured rule (decision-support convention; not an industry standard)"
SRC_DATASET = ("Dataset documentation: Saxena et al., 'Damage Propagation Modeling for Aircraft Engine "
               "Run-to-Failure Simulation', PHM08 (simulated data; no component mapping for sensors)")


@dataclass
class Context:
    """Everything a rule may look at. All values come from actual computations."""
    engine_id: int
    last_cycle: int
    pred_rul: float | None   # None when no remaining-life model applies (sensor-only data)
    state: str | None
    thresholds: dict
    cap: float
    n_cycles: int
    at_cap: bool = False
    anomaly: dict | None = None            # from anomaly_detection.engine_summary
    shifts: list[dict] = field(default_factory=list)   # top shifted sensors (label, shift_sigma, trend_rho)
    shap_top: list[dict] = field(default_factory=list)  # top grouped contributions (label, shap)
    ood: dict | None = None
    dq_warnings: list[str] = field(default_factory=list)
    sigma_trigger: float = 3.0
    anomaly_fraction_trigger: float = 0.5


@dataclass
class Rule:
    rule_id: str
    title: str
    trigger_text: str
    action: str
    priority: str
    source: str
    limitation: str
    verification: str
    check: Callable[[Context], dict | None]
    finding_class: str = FC_REC
    condition: str = ""


def _sh(ctx: Context) -> list[dict]:
    return [s for s in ctx.shifts if s.get("abs_sigma", 0) >= ctx.sigma_trigger][:4]


def _shift_text(items: list[dict]) -> str:
    return "; ".join(f"{s['label']}: {'up' if s['shift_sigma'] > 0 else 'down'} by {abs(s['shift_sigma']):.1f}x normal noise"
                     for s in items)


def _r_crit(ctx):
    if ctx.pred_rul is not None and ctx.pred_rul < ctx.thresholds["critical"]:
        corr = (ctx.anomaly and ctx.anomaly["recent_flag_fraction"] >= ctx.anomaly_fraction_trigger) or bool(_sh(ctx))
        return {"evidence": f"The model estimates {ctx.pred_rul:.0f} cycles of remaining life, below the critical level of "
                            f"{ctx.thresholds['critical']:.0f} cycles.", "strength": "Strong" if corr else "Moderate"}


def _r_warn(ctx):
    if ctx.pred_rul is not None and ctx.thresholds["critical"] <= ctx.pred_rul <= ctx.thresholds["warning"]:
        corr = (ctx.anomaly and ctx.anomaly["recent_flag_fraction"] >= ctx.anomaly_fraction_trigger) or bool(_sh(ctx))
        return {"evidence": f"The model estimates {ctx.pred_rul:.0f} cycles of remaining life, inside the warning range of "
                            f"{ctx.thresholds['critical']:.0f} to {ctx.thresholds['warning']:.0f} cycles.",
                "strength": "Strong" if corr else "Moderate"}


def _r_healthy(ctx):
    if ctx.pred_rul is not None and ctx.pred_rul > ctx.thresholds["warning"]:
        cap_txt = " (the estimate is at its upper limit, so real remaining life may be longer)" if ctx.at_cap else ""
        return {"evidence": f"The model estimates {ctx.pred_rul:.0f} cycles of remaining life, above the warning level of "
                            f"{ctx.thresholds['warning']:.0f} cycles{cap_txt}.", "strength": "Moderate"}


def _r_anom(ctx):
    a = ctx.anomaly
    if a and a["recent_flag_fraction"] >= ctx.anomaly_fraction_trigger:
        return {"evidence": f"{a['recent_flag_fraction']*100:.0f}% of the last {a['recent_n']} records show unusual sensor behaviour compared with "
                            f"the early-life pattern (latest score {a['last_score']:.2f}"
                            + (f"; first flagged at cycle {a['first_flag_cycle']}" if a['first_flag_cycle'] else "") + ").",
                "strength": "Moderate"}


def _r_shift(ctx):
    items = _sh(ctx)
    if items:
        return {"evidence": "These sensors moved away from the asset's own early readings by more than "
                            f"{ctx.sigma_trigger:.0f} times their normal early noise: " + _shift_text(items) + ".",
                "strength": "Moderate"}


def _r_short(ctx):
    if ctx.n_cycles < C.MIN_HISTORY_CYCLES:
        return {"evidence": f"Only {ctx.n_cycles} records of history (fewer than {C.MIN_HISTORY_CYCLES}); results are less reliable.", "strength": "Informational"}


def _r_ood(ctx):
    o = ctx.ood
    if o and o.get("flag"):
        return {"evidence": "Some input values lie outside the range the model was built for, so the estimate may be unreliable.", "strength": "Informational"}


def _r_dq(ctx):
    if ctx.dq_warnings:
        return {"evidence": "Data-quality warnings: " + " | ".join(ctx.dq_warnings[:3]), "strength": "Informational"}


RULES: list[Rule] = [
    Rule("R-RUL-CRIT", "Escalate for engineering review and plan an inspection",
         "Estimated remaining life is below the configured critical level.",
         "Escalate to the responsible maintenance engineer; review the engine's recent operating and sensor "
         "history and plan an engineering inspection according to the applicable maintenance procedure.",
         "Critical", SRC_APP,
         "The critical level is a configurable convention. The model estimate has a typical error and is limited to an "
         "upper value; it does not identify which component is affected.",
         "Compare against the maintenance record and perform the inspection defined by the applicable procedure.",
         _r_crit),
    Rule("R-RUL-WARN", "Review recent sensor history and schedule an engineering check",
         "Estimated remaining life is inside the configured warning range.",
         "Review recent sensor and operating history and consider scheduling an engineering inspection within "
         "the planning horizon defined by your maintenance procedure.",
         "High", SRC_APP,
         "Threshold-based convention; the estimate is uncertain and not a failure date.",
         "Re-run the analysis after the next operating cycles and confirm the trend persists.",
         _r_warn),
    Rule("R-RUL-OK", "Continue routine monitoring",
         "Estimated remaining life is above the configured warning level.",
         "Continue routine monitoring under the standard maintenance plan.",
         "Low", SRC_APP,
         "A healthy estimate does not exclude a fault that the model cannot see.",
         "Re-evaluate as new cycles arrive.", _r_healthy),
    Rule("R-ANOM-PERSIST", "Review unusual behaviour flagged by the anomaly detector",
         "Unusual-behaviour flags in at least half of the recent records.",
         "Review the engine's operating conditions and sensor integrity for the flagged period; potential "
         "contributing pattern identified — further engineering inspection is required to determine the "
         "physical cause.",
         "Medium", SRC_APP + "; unsupervised anomaly detector (see README)",
         "An anomaly is not automatically a failure: it can reflect degradation, an unusual operating "
         "condition or a measurement issue. The detector has no fault labels.",
         "Check sensor calibration/records and compare with peer engines over the same cycles.",
         _r_anom),
    Rule("R-SENSOR-SHIFT", "Investigate sustained sensor drift",
         "One or more sensors moved away from the asset's own early readings by more than the configured trigger.",
         "Potential contributing pattern identified in the listed sensors. Review the trends with an engineer "
         "who knows the sensor configuration; further inspection is required to determine the physical cause.",
         "Medium", SRC_DATASET,
         "Sensors are not mapped to physical components in the supplied data; a shift does not identify a faulty part.",
         "Validate the sensors against engineering documentation and maintenance records.",
         _r_shift),
    Rule("R-DQ-SHORT", "Treat the estimate with caution: short history",
         "Fewer than the minimum number of cycles are available.",
         "Collect more cycles before relying on the estimate.", "Info", SRC_APP,
         "Short histories give less stable trend calculations.", "Re-run after more cycles are recorded.", _r_short),
    Rule("R-DQ-OOD", "Verify inputs: values outside the training range",
         "A noticeable share of input values lies outside the range the model was built for.",
         "Verify sensor values and units before using the prediction.", "Info", SRC_APP,
         "Out-of-range inputs make the model estimate unreliable.",
         "Check sensor scaling, units and calibration.", _r_ood),
    Rule("R-DQ-GENERAL", "Resolve data-quality warnings", "Validation produced warnings.",
         "Review the data-quality report.", "Info", SRC_APP,
         "Warnings do not invalidate the data but can reduce reliability.", "Fix the flagged items at source.", _r_dq),
]

_RULE_META = {
    "R-RUL-CRIT": (FC_SUSPECT, "Estimated remaining life is critically low"),
    "R-RUL-WARN": (FC_SUSPECT, "Estimated remaining life is in the warning range"),
    "R-RUL-OK": (FC_ROUTINE, "No abnormal condition detected"),
    "R-ANOM-PERSIST": (FC_ANOMALY, "Unusual sensor behaviour compared with the asset's early-life pattern"),
    "R-SENSOR-SHIFT": (FC_ANOMALY, "Sustained sensor drift away from the asset's own early readings"),
    "R-DQ-SHORT": (FC_DQ, "Too little history for a reliable assessment"),
    "R-DQ-OOD": (FC_DQ, "Input values outside the range the model was built for"),
    "R-DQ-GENERAL": (FC_DQ, "The input data has quality warnings"),
}
for _r in RULES:
    _r.finding_class, _r.condition = _RULE_META[_r.rule_id]


def evaluate(ctx: Context, rules: list[Rule] | None = None) -> list[dict]:
    """Run every rule; return the triggered recommendations sorted by priority."""
    out = []
    for r in rules or RULES:
        hit = r.check(ctx)
        if not hit:
            continue
        sensors = [s["label"] for s in _sh(ctx)] if r.rule_id in ("R-SENSOR-SHIFT", "R-RUL-CRIT", "R-RUL-WARN") else []
        out.append({
            "id": f"{r.rule_id}-E{ctx.engine_id}", "rule_id": r.rule_id, "engine_id": ctx.engine_id,
            "action": r.action, "title": r.title, "trigger": r.trigger_text, "evidence": hit["evidence"],
            "sensors": sensors, "priority": r.priority, "evidence_strength": hit["strength"],
            "source": r.source, "limitation": r.limitation, "verification": r.verification,
            "finding_type": ("Model-based finding + rule" if r.rule_id.startswith("R-RUL") else
                             "Rule-triggered (statistical/data-quality)"),
            "status": "Pending",
            "finding_class": r.finding_class, "condition": r.condition, "confirmation": CONFIRMATION_TEXT,
            "sensor_evidence": [{"sensor": x.get("sensor"), "label": x["label"], "baseline": x.get("baseline_mean"),
                                 "recent": x.get("recent_mean"), "shift_sigma": x["shift_sigma"],
                                 "direction": "increased" if x["shift_sigma"] > 0 else "decreased"}
                                for x in _sh(ctx)] if sensors else [],
        })
    out.sort(key=lambda d: PRIORITY_ORDER[d["priority"]])
    return out


EVIDENCE_DEFINITIONS = {
    "Strong": "Trigger corroborated by at least one independent indicator.",
    "Moderate": "A single model-based or statistical indicator triggered the rule.",
    "Informational": "Data-quality or contextual notice; no engineering finding implied.",
}
