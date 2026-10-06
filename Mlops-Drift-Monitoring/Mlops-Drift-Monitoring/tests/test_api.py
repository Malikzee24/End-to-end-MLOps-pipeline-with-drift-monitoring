import pytest
from fastapi.testclient import TestClient

from mlops_pipeline.serving.api import create_app

CUSTOMER = {
    "tenure_months": 6, "age": 34, "monthly_charges": 89.5, "total_charges": 537.0, "num_support_calls": 4,
    "contract_type": "month-to-month", "internet_service": "fiber", "payment_method": "electronic_check",
    "has_partner": "no",
}


@pytest.fixture()
def client(trained):
    with TestClient(create_app(trained)) as c:
        yield c


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "1"}
    assert client.get("/model").json()["version"] == "1"


def test_predict_logs_prediction_and_accepts_feedback(client):
    r = client.post("/predict", json=CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1 and body["model_version"] == "1"

    store = client.app.state.store
    assert store.count_predictions() == 1
    fb = client.post("/feedback", json={"items": [{"prediction_id": body["prediction_id"], "actual_churn": True}]})
    assert fb.json() == {"accepted": 1}
    assert store.get_predictions()["label"].iloc[0] == 1


def test_batch_endpoint(client):
    r = client.post("/predict/batch", json={"customers": [CUSTOMER, {**CUSTOMER, "contract_type": "two-year"}]})
    assert r.status_code == 200 and len(r.json()) == 2
    # a long, cheap, two-year contract should be less risky than a month-to-month one
    assert r.json()[0]["churn_probability"] > r.json()[1]["churn_probability"]


def test_validation_rejects_bad_input(client):
    assert client.post("/predict", json={**CUSTOMER, "contract_type": "weekly"}).status_code == 422
    assert client.post("/predict", json={**CUSTOMER, "age": -3}).status_code == 422


def test_prometheus_metrics_exposed(client):
    client.post("/predict", json=CUSTOMER)
    text = client.get("/metrics").text
    assert "mlops_predictions_total" in text and 'mlops_model_info{model_version="1"} 1.0' in text


def test_503_without_model(settings):
    with TestClient(create_app(settings)) as c:  # nothing trained
        assert c.get("/health").json()["status"] == "no_model"
        assert c.post("/predict", json=CUSTOMER).status_code == 503
