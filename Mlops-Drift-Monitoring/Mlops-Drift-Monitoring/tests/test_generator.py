import pandas as pd

from mlops_pipeline.data import DriftSpec, generate_churn_data
from mlops_pipeline.schema import CATEGORY_VALUES, FEATURES, TARGET


def test_schema_and_reproducibility():
    a = generate_churn_data(500, seed=3)
    b = generate_churn_data(500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert set(FEATURES + [TARGET]) == set(a.columns)
    for col, allowed in CATEGORY_VALUES.items():
        assert set(a[col]) <= set(allowed)
    assert 0.15 < a[TARGET].mean() < 0.5


def test_covariate_drift_moves_inputs():
    base = generate_churn_data(4000, seed=1)
    drifted = generate_churn_data(4000, seed=1, drift=DriftSpec(charges_shift=20, month_to_month_boost=0.2))
    assert drifted["monthly_charges"].mean() > base["monthly_charges"].mean() + 10
    assert (drifted["contract_type"] == "month-to-month").mean() > (base["contract_type"] == "month-to-month").mean() + 0.1


def test_no_drift_spec_is_not_drifted():
    assert not DriftSpec().is_drifted
    assert DriftSpec(concept_shift=0.5).is_drifted
