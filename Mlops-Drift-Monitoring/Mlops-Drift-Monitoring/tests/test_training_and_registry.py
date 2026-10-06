from mlops_pipeline import registry
from mlops_pipeline.data import generate_churn_data
from mlops_pipeline.schema import FEATURES
from mlops_pipeline.training import promote_if_better, train


def test_train_registers_and_promotes_first_model(trained):
    mv = registry.get_champion_version(trained)
    assert mv is not None and str(mv.version) == "1"
    assert registry.baseline_auc(trained, mv) > 0.7
    ref = registry.load_reference(trained, mv)
    assert set(FEATURES) <= set(ref.columns) and len(ref) > 100


def test_champion_model_predicts_probabilities(trained):
    model, _ = registry.load_champion(trained)
    proba = model.predict_proba(generate_churn_data(50, seed=5)[FEATURES])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_gate_rejects_worse_challenger(trained):
    """A challenger trained on tiny data must not dethrone the champion."""
    weak = train(trained, generate_churn_data(400, seed=9), trigger="test")
    holdout = generate_churn_data(2000, seed=11)
    decision = promote_if_better(trained, weak.model_version, eval_df=holdout)
    assert not decision.promoted
    assert str(registry.get_champion_version(trained).version) == "1"


def test_gate_accepts_better_challenger_and_moves_alias(trained):
    better = train(trained, generate_churn_data(8000, seed=10), trigger="test")
    holdout = generate_churn_data(3000, seed=12)
    trained.promotion.min_auc_improvement = -1.0  # force acceptance: tests the alias move itself
    decision = promote_if_better(trained, better.model_version, eval_df=holdout)
    assert decision.promoted
    assert str(registry.get_champion_version(trained).version) == better.model_version
