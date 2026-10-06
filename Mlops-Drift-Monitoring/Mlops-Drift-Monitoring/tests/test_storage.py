from datetime import datetime, timedelta

from mlops_pipeline.storage import PredictionStore


def _rec(i, ts):
    return {"prediction_id": f"p{i}", "ts": ts, "model_version": "1", "probability": 0.3 + i / 100,
            "prediction": 0, "features": {"age": 30 + i, "contract_type": "one-year"}}


def test_roundtrip_window_and_labels(tmp_path):
    store = PredictionStore(tmp_path / "p.db")
    t0 = datetime(2025, 1, 1)
    store.log_predictions([_rec(i, t0 + timedelta(days=i)) for i in range(5)])
    store.add_labels([("p1", 1), ("p3", 0)])

    everything = store.get_predictions()
    assert len(everything) == 5 and everything["label"].notna().sum() == 2

    window = store.get_predictions(since=t0 + timedelta(days=1), until=t0 + timedelta(days=3))
    assert list(window["prediction_id"]) == ["p2", "p3"]  # (since, until]
    assert window.loc[window.prediction_id == "p3", "label"].item() == 0
    assert window["age"].tolist() == [32, 33]


def test_empty_store_returns_empty_frame(tmp_path):
    assert PredictionStore(tmp_path / "p.db").get_predictions().empty


def test_monitoring_and_retrain_history(tmp_path):
    store = PredictionStore(tmp_path / "p.db")
    store.save_monitoring_run({"ts": datetime(2025, 1, 1), "model_version": "1", "status": "OK", "action": "NONE",
                               "details": {"features": []}})
    assert store.latest_monitoring_run()["status"] == "OK"
    store.save_retrain_event({"ts": datetime(2025, 1, 2), "trigger": "drift", "promoted": False})
    assert store.last_retrain(promoted_only=True) is None
    assert store.last_retrain(promoted_only=False)["trigger"] == "drift"
