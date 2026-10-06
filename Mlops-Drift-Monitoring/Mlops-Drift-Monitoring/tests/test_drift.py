import numpy as np
import pandas as pd

from mlops_pipeline.config import DriftConfig
from mlops_pipeline.data import DriftSpec, generate_churn_data
from mlops_pipeline.monitoring import detect_drift, psi


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.05


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi(ref, rng.normal(0.3, 1, 5000)) < psi(ref, rng.normal(1.0, 1, 5000))
    assert psi(ref, rng.normal(1.0, 1, 5000)) > 0.2


def test_no_false_alarm_on_same_distribution():
    ref = generate_churn_data(5000, seed=1)
    cur = generate_churn_data(1500, seed=2)
    report = detect_drift(ref, cur)
    assert report.drifted_features == []
    assert not report.dataset_drift


def test_covariate_drift_is_detected():
    ref = generate_churn_data(5000, seed=1)
    cur = generate_churn_data(1500, seed=2, drift=DriftSpec(charges_shift=18, support_calls_shift=0.9,
                                                            month_to_month_boost=0.22))
    report = detect_drift(ref, cur)
    assert {"monthly_charges", "num_support_calls", "contract_type"} <= set(report.drifted_features)
    assert report.dataset_drift


def test_concept_drift_alone_is_invisible_to_feature_tests():
    """Concept drift changes P(y|x), not P(x): feature tests must not (and cannot) see it."""
    ref = generate_churn_data(5000, seed=1)
    cur = generate_churn_data(1500, seed=2, drift=DriftSpec(concept_shift=1.0))
    assert not detect_drift(ref, cur).dataset_drift


def test_unseen_category_in_current_data_is_handled():
    ref = pd.DataFrame({"c": ["a", "b"] * 500})
    cur = pd.DataFrame({"c": ["a", "b", "z"] * 300})
    report = detect_drift(ref, cur, DriftConfig(), numeric=[], categorical=["c"])
    assert report.features[0].drifted  # a brand-new category is a real shift
    assert np.isfinite(report.features[0].psi)


def test_constant_reference_column_does_not_crash():
    ref = pd.DataFrame({"x": [1.0] * 500})
    cur = pd.DataFrame({"x": np.random.default_rng(0).normal(size=500)})
    report = detect_drift(ref, cur, numeric=["x"], categorical=[])
    assert np.isfinite(report.features[0].psi)
