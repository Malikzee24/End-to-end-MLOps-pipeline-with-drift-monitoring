import pytest

from mlops_pipeline.config import CandidateConfig, Settings, load_settings
from mlops_pipeline.data import generate_churn_data
from mlops_pipeline.training import promote_if_better, train


@pytest.fixture()
def settings(tmp_path) -> Settings:
    """Fast settings isolated in a temp dir (own MLflow DB, prediction DB and reports)."""
    s = load_settings(state_dir=tmp_path / "state")
    s.data.n_train = 3000
    s.training.cv_folds = 3
    s.training.candidates = [
        CandidateConfig(name="logistic_regression", params={"max_iter": 500}),
        CandidateConfig(name="hist_gradient_boosting", params={"max_iter": 30}),
    ]
    s.monitoring.min_samples = 200
    s.monitoring.performance.min_labeled_samples = 100
    s.monitoring.retrain.min_labeled_samples = 400
    return s


@pytest.fixture()
def trained(settings):
    """Settings with one trained + promoted champion (v1)."""
    df = generate_churn_data(settings.data.n_train, seed=1)
    res = train(settings, df)
    decision = promote_if_better(settings, res.model_version, challenger_test_auc=res.metrics["test_roc_auc"])
    assert decision.promoted
    return settings
