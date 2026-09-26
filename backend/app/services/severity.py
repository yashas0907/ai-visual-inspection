"""Severity engine — deterministic, transparent business rules.

The severity of an inspection is a BUSINESS decision, distinct from the
model's prediction. Inputs (all measurable or model-supported):

1. defect class risk weight     — domain knowledge (which defect types are
                                  structurally worse for a hot-rolled strip)
2. confidence tier              — from the confidence policy (high/medium/low)
3. defect evidence coverage     — fraction of the surface where Grad-CAM
                                  evidence exceeds the activation threshold
                                  (proxy for defect area)

Scoring: each input contributes points; the total maps to
MINOR / MAJOR / CRITICAL. The full rubric and rationale live in
docs/severity.md. No LLM is involved; the function is pure and unit-tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    minor = "minor"
    major = "major"
    critical = "critical"


# Domain-informed risk weights. Patches/rolled-in scale are typically
# re-workable surface texture issues; inclusions and pitted surfaces imply
# material contamination; crazing and scratches can propagate under load.
DEFECT_RISK_WEIGHTS: dict[str, int] = {
    "patches": 1,
    "rolled-in_scale": 2,
    "scratches": 2,
    "crazing": 3,
    "pitted_surface": 3,
    "inclusion": 3,
}

CONFIDENCE_TIER_POINTS: dict[str, int] = {
    "high": 0,   # model is consistent -> evidence itself carries the weight
    "medium": 1,
    "low": 2,   # uncertain evidence pushes severity up (conservative policy)
}

# Coverage bands: fraction of the inspected surface with active evidence.
COVERAGE_BANDS: tuple[tuple[float, float, int], ...] = (
    # (min_ratio_inclusive, max_ratio_inclusive, points)
    (0.00, 0.10, 0),
    (0.10, 0.30, 1),  # upper band wins at exactly 0.10
    (0.30, 1.01, 2),  # upper band wins at exactly 0.30
)


def coverage_points(coverage: float) -> int:
    if coverage < 0:
        raise ValueError("evidence_coverage must be >= 0")
    coverage = min(coverage, 1.0)
    # bands are checked high-to-low so exact boundaries (0.10, 0.30)
    # take the more severe (documented) band.
    for lo, hi, pts in reversed(COVERAGE_BANDS):
        if lo <= coverage <= hi:
            return pts
    return COVERAGE_BANDS[-1][2]

SEVERITY_THRESHOLDS: tuple[int, int] = (3, 5)  # <=3 minor, 4-5 major, >=6 critical


@dataclass(frozen=True)
class SeverityInput:
    defect_label: str
    confidence_tier: str  # high | medium | low
    evidence_coverage: float  # area fraction in [0, 1]


@dataclass(frozen=True)
class SeverityAssessment:
    severity: Severity
    score: int
    components: dict[str, int]
    recommended_action: str


RECOMMENDED_ACTIONS: dict[Severity, str] = {
    Severity.minor: "Continue line; log for trend monitoring.",
    Severity.major: "Quarantine part; schedule rework or regrade.",
    Severity.critical: "Stop line; reject part; notify quality engineer.",
}


def risk_points(label: str) -> int:
    return DEFECT_RISK_WEIGHTS.get(label, 2)


def assess_severity(inp: SeverityInput) -> SeverityAssessment:
    """Pure function: label + tier + coverage -> severity + explanation."""
    label = str(inp.defect_label).lower()
    if inp.confidence_tier not in CONFIDENCE_TIER_POINTS:
        raise ValueError(f"unknown confidence tier: {inp.confidence_tier}")

    components = {
        "defect_risk": risk_points(label),
        "confidence_tier": CONFIDENCE_TIER_POINTS[inp.confidence_tier],
        "evidence_coverage": coverage_points(inp.evidence_coverage),
    }
    score = sum(components.values())
    if score <= SEVERITY_THRESHOLDS[0]:
        sev = Severity.minor
    elif score <= SEVERITY_THRESHOLDS[1]:
        sev = Severity.major
    else:
        sev = Severity.critical
    return SeverityAssessment(
        severity=sev,
        score=score,
        components=components,
        recommended_action=RECOMMENDED_ACTIONS[sev],
    )
