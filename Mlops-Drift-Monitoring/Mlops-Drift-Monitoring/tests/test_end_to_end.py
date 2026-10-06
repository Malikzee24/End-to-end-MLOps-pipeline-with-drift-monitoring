"""Integration test: traffic -> drift detected -> retrain -> new champion -> monitoring calms down."""

from datetime import datetime, timedelta

from mlops_pipeline import registry
from mlops_pipeline.data import DriftSpec, generate_churn_data
from mlops_pipeline.monitoring import run_monitoring
from mlops_pipeline.schema import TARGET
from mlops_pipeline.serving.predictor import ModelManager
from mlops_pipeline.storage import PredictionStore

T0 = datetime(2025, 3, 3)


def _send(manager, store, df, start, days=7, labeled=True):
    step = days * 86400 / len(df)
    stamps = [start + timedelta(seconds=int(i * step) + 1) for i in range(len(df))]
    scored = manager.predict_and_log(df, store, ts=stamps)
    if labeled:
        store.add_labels(list(zip(scored["prediction_id"], df[TARGET])), labeled_at=start + timedelta(days=days))


def test_insufficient_data_is_handled(trained):
    store = PredictionStore(trained.prediction_db)
    res = run_monitoring(trained, store, as_of=T0, write_report=False)
    assert res.status == "INSUFFICIENT_DATA" and res.action == "NONE"


def test_stable_traffic_stays_ok(trained):
    store, manager = PredictionStore(trained.prediction_db), ModelManager(trained, 0)
    manager.load()
    _send(manager, store, generate_churn_data(1200, seed=21), T0)
    res = run_monitoring(trained, store, as_of=T0 + timedelta(days=7))
    assert res.status == "OK" and res.retrain is None
    assert (trained.report_dir / "latest.html").exists()


def test_drift_triggers_retrain_and_recovers(trained):
    trained.monitoring.window_hours = 7 * 24
    trained.monitoring.retrain.lookback_hours = 14 * 24
    trained.promotion.min_auc_improvement = -1.0  # retrain on drifted data should be accepted here
    store, manager = PredictionStore(trained.prediction_db), ModelManager(trained, 0)
    manager.load()

    drift = DriftSpec(charges_shift=18, support_calls_shift=0.9, month_to_month_boost=0.22)
    _send(manager, store, generate_churn_data(1500, seed=31, drift=drift), T0)
    res = run_monitoring(trained, store, as_of=T0 + timedelta(days=7))

    assert res.status == "CRITICAL" and res.action == "RETRAIN"
    assert res.drift.dataset_drift
    assert res.retrain is not None and res.retrain.promoted
    assert str(registry.get_champion_version(trained).version) == res.retrain.new_version != "1"

    # the API-side manager hot-swaps to the new champion ...
    manager.maybe_refresh()
    assert manager.version == res.retrain.new_version

    # ... and, against the *new* reference, the same traffic regime is no longer "drift"
    _send(manager, store, generate_churn_data(1500, seed=32, drift=drift), T0 + timedelta(days=7))
    calm = run_monitoring(trained, store, as_of=T0 + timedelta(days=14), auto_retrain=False)
    assert not calm.drift.dataset_drift

    history = store.monitoring_history()
    assert list(history["status"]) == ["CRITICAL", calm.status]
    assert not store.retrain_history().empty


def test_cooldown_blocks_repeated_retraining(trained):
    from mlops_pipeline.pipeline import retrain

    trained.monitoring.window_hours = 7 * 24
    trained.monitoring.retrain.lookback_hours = 14 * 24
    store, manager = PredictionStore(trained.prediction_db), ModelManager(trained, 0)
    manager.load()
    _send(manager, store, generate_churn_data(1500, seed=41), T0)
    first = retrain(trained, store, as_of=T0 + timedelta(days=7), trigger="test")
    assert first.attempted
    second = retrain(trained, store, as_of=T0 + timedelta(days=8), trigger="test")
    assert not second.attempted and second.reason == "cooldown active"


def test_retrain_skipped_without_enough_labels(trained):
    from mlops_pipeline.pipeline import retrain

    store, manager = PredictionStore(trained.prediction_db), ModelManager(trained, 0)
    manager.load()
    _send(manager, store, generate_churn_data(300, seed=51), T0, labeled=False)
    out = retrain(trained, store, as_of=T0 + timedelta(days=7))
    assert not out.attempted and "insufficient labeled data" in out.reason
