# Scoring contract

The evaluator produces four scores on a 0–100 scale:

- L1: HTML, CSS, and JavaScript syntax and structural executability.
- L2: mathematical, scientific, logical, and pedagogical consistency.
- L3: rendered-page visual quality and layout consistency.
- L4: runtime behavior and effective interaction coverage.

## Reproducibility profile

An inspected archived 10k-RL per-sample report records the following weights.
This release uses them as its default profile; all historical runs have not yet
been audited individually:

```json
{"layer1": 0.20, "layer2": 0.25, "layer3": 0.25, "layer4": 0.30}
```

The weighted score is computed per sample. Runtime gating is then applied:

- L4 < 20: total score is capped at 20.
- 20 <= L4 < 40: total score is capped at 40.
- 40 <= L4 < 60: a passing total is capped at 59.

Dataset-level totals are averages of the gated per-sample totals. They therefore
cannot generally be reconstructed by applying the weights to the four published
dataset-level mean scores.

## Paper/code discrepancy

The manuscript formula currently shows weights 0.10/0.20/0.30/0.40. Those are
not the weights embedded in the archived per-sample reports. Before making a
formal reproducibility release, either update the manuscript description or
release both configurations with an explicit explanation. L1–L4 individual
scores are unaffected by this total-score weighting discrepancy.
