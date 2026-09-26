# Severity engine

Severity is a **business decision**, deliberately separated from the model
prediction. It is computed by a pure, unit-tested function
(`backend/app/services/severity.py`) from three inputs — no LLM, no hidden
logic.

## Inputs

1. **Defect risk weight** (domain knowledge, configurable dict)
2. **Confidence tier** (from the documented confidence policy)
3. **Evidence coverage** (fraction of the surface with Grad-CAM evidence
   above the activation threshold — a proxy for defect area)

## Rubric

### Defect risk points

| Defect | Points | Rationale |
|---|---|---|
| patches | 1 | re-workable texture blemish |
| rolled-in_scale | 2 | surface contamination, often regradeable |
| scratches | 2 | mechanical damage, depth-dependent |
| crazing | 3 | crack networks can propagate under load |
| pitted_surface | 3 | material-integrity corrosion |
| inclusion | 3 | embedded foreign material — structural risk |
| (unknown) | 2 | conservative default |

### Confidence tier points

| Tier | Points | Rationale |
|---|---|---|
| high | 0 | evidence carries the weight |
| medium | 1 | some uncertainty |
| low | 2 | uncertain evidence pushes severity up (conservative policy) |

Note the direction: *lower model confidence raises severity*. Rationale: an
uncertain prediction still warrants inspector attention; the cost of
under-reacting in steel QC exceeds the cost of a manual check.

### Evidence coverage points

| Coverage of surface | Points |
|---|---|
| ≤ 10% | 0 |
| 10–30% | 1 |
| > 30% | 2 |

(Exact boundaries 0.10 / 0.30 take the more severe band.)

### Score → severity

| Total | Severity | Recommended action |
|---|---|---|
| ≤ 3 | minor | Continue line; log for trend monitoring. |
| 4–5 | major | Quarantine part; schedule rework or regrade. |
| ≥ 6 | critical | Stop line; reject part; notify quality engineer. |

## Example

```python
assess_severity(SeverityInput(
    defect_label="inclusion",   # risk 3
    confidence_tier="medium",  # +1
    evidence_coverage=0.42,     # +2 (band >30%)
))
# → score 6 → critical → "Stop line; reject part; notify quality engineer."
```

## Confidence policy (separate concern)

Confidence tiers are an operational policy on top of softmax outputs
(**not** calibrated probabilities):

- ≥ 0.80 → **high**
- 0.60–0.79 → **medium**
- < 0.60 → **low** → prediction flagged `needs_review` in the DB and surfaced
  prominently in the UI.

Thresholds are environment-configurable (`VI_CONFIDENCE_HIGH`,
`VI_CONFIDENCE_MEDIUM`) without code changes.

## Separation of concerns

- The **model** predicts *what* the defect is (with softmax confidence).
- The **severity engine** decides *what it means operationally*.
Swapping either does not silently change the other; both are versioned
(severity constants live in code review, model weights in artifacts).
