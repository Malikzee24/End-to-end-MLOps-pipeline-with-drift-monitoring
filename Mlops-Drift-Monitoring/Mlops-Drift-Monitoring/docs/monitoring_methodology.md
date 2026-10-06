# Monitoring methodology

## What can go wrong in production?

| Failure mode | What changes | Needs labels? | Detected by |
|---|---|---|---|
| **Covariate drift** | `P(x)` - inputs shift (new pricing, new customer mix) | No | per-feature PSI / KS / chi-square |
| **Prior / prediction drift** | `P(y)` or the distribution of model scores | No | prediction PSI (reported on every run) |
| **Concept drift** | `P(y|x)` - the relationship changes | **Yes** | live ROC-AUC vs. baseline |
| **Data-quality incidents** | schema, nulls, unseen categories | No | pydantic validation at the API + unseen-category handling in drift tests |

A monitoring system that only looks at inputs misses concept drift; one that only looks at performance reacts
weeks late, because labels (did the customer churn?) arrive with a delay. This project does both.

## Per-feature tests

**Population Stability Index (PSI)** - numeric features are binned on the *reference* deciles; categorical
features use category proportions:

    PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)

Conventional reading: < 0.1 stable, 0.1-0.2 moderate, >= 0.2 significant. Proportions are clipped at 1e-4 so
empty or unseen categories give a large but finite value.

**Two-sample KS test** (numeric) and **chi-square test of homogeneity** (categorical) give p-values.

**Effect sizes** - KS statistic and Jensen-Shannon distance - measure *how much*, not just *whether*.

### Decision rule

    drifted  =  PSI >= 0.2   OR   (p < 0.05  AND  effect_size >= 0.1)

Why not p-values alone? With n in the thousands, the KS test flags practically any difference. Requiring a
material effect size keeps false alarms down; `tests/test_drift.py::test_no_false_alarm_on_same_distribution` pins this.

**Dataset drift** is declared when at least `dataset_drift_share` (30 %) of features are flagged. Correlated features
(`monthly_charges` and `total_charges`) drift together, so the threshold is a count of *evidence*, not of
independent signals - tune it to your feature set.

## Performance monitoring

When delayed labels exist, ROC-AUC (plus average precision, F1 and Brier score) is computed on predictions that

1. are inside the window,
2. have a label, and
3. were made by the **current champion** (otherwise a freshly promoted model is blamed for its predecessor's mistakes).

Degradation = `baseline_auc - live_auc > max_auc_drop` (0.05). A minimum of 200 labeled rows and both classes is
required before anything is evaluated.

## Retraining policy

Triggered by `CRITICAL` status, then guarded by:

1. **Cooldown** - no new attempt within `cooldown_hours` of the last one (promoted or not).
2. **Data volume** - at least `min_labeled_samples` labeled rows in the lookback window.
3. **Hold-out gate** - 30 % of recent labeled data is withheld; challenger and champion are both scored on it and the
   challenger must win by `min_auc_improvement`.
4. **Absolute floor** - never promote a model below `min_absolute_auc`.

On promotion the new model's reference data is the *recent* training data (the "new normal"), so the same traffic
stops being reported as drift - visible in the drop to 0 % after week 11 in the results plot.
