import numpy as np
import pandas as pd

from mlops_pipeline.config import PerformanceConfig
from mlops_pipeline.monitoring import evaluate_live_performance


def _frame(n, signal, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + signal * (y - 0.5) + rng.normal(0, 0.2, n), 0.01, 0.99)
    return pd.DataFrame({"label": y, "probability": p})


def test_good_model_not_degraded():
    rep = evaluate_live_performance(_frame(1000, 0.8), baseline_auc=0.85, cfg=PerformanceConfig())
    assert rep.available and not rep.degraded and rep.roc_auc > 0.85


def test_degradation_is_flagged():
    rep = evaluate_live_performance(_frame(1000, 0.0), baseline_auc=0.85, cfg=PerformanceConfig())
    assert rep.available and rep.degraded and rep.auc_drop > 0.2


def test_too_few_labels_is_not_evaluated():
    rep = evaluate_live_performance(_frame(50, 0.8), baseline_auc=0.85, cfg=PerformanceConfig(min_labeled_samples=200))
    assert not rep.available and not rep.degraded


def test_single_class_is_not_evaluated():
    df = pd.DataFrame({"label": [1] * 300, "probability": [0.7] * 300})
    assert not evaluate_live_performance(df, 0.8, PerformanceConfig()).available
