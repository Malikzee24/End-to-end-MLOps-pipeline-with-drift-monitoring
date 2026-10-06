"""FastAPI inference service with request logging, delayed-label feedback and Prometheus metrics."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from mlops_pipeline import __version__
from mlops_pipeline.config import Settings, load_settings
from mlops_pipeline.schema import FEATURES
from mlops_pipeline.serving.predictor import ModelManager
from mlops_pipeline.storage import PredictionStore, from_iso, utcnow

log = logging.getLogger("mlops.api")


# ------------------------------------------------------------------------------ request schemas
class Customer(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, examples=[12])
    age: int = Field(..., ge=16, le=110, examples=[41])
    monthly_charges: float = Field(..., ge=0, examples=[79.9])
    total_charges: float = Field(..., ge=0, examples=[958.8])
    num_support_calls: int = Field(..., ge=0, le=100, examples=[3])
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]
    payment_method: Literal["electronic_check", "credit_card", "bank_transfer", "mailed_check"]
    has_partner: Literal["yes", "no"]


class Prediction(BaseModel):
    prediction_id: str
    churn_probability: float
    churn_prediction: bool
    model_version: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=5000)


class Feedback(BaseModel):
    prediction_id: str
    actual_churn: bool


class FeedbackBatch(BaseModel):
    items: list[Feedback] = Field(..., min_length=1, max_length=10000)


# ------------------------------------------------------------------------------------ metrics
class ApiMetrics:
    """Per-app registry (avoids duplicate-metric errors when the app is created repeatedly in tests)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("mlops_requests_total", "HTTP requests", ["endpoint", "status"], registry=r)
        self.latency = Histogram("mlops_request_latency_seconds", "Request latency", ["endpoint"], registry=r,
                                 buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
        self.predictions = Counter("mlops_predictions_total", "Predictions served", ["model_version", "predicted_churn"],
                                   registry=r)
        self.proba = Histogram("mlops_churn_probability", "Predicted churn probability", registry=r,
                               buckets=tuple(x / 10 for x in range(1, 10)) + (1.0,))
        self.model_info = Gauge("mlops_model_info", "Serving model version (value is always 1)", ["model_version"],
                                registry=r)
        # --- monitoring gauges, refreshed from the latest monitoring run at scrape time
        self.drift_share = Gauge("mlops_drift_share_of_features", "Share of drifted features", registry=r)
        self.dataset_drift = Gauge("mlops_dataset_drift", "1 if dataset drift detected", registry=r)
        self.feature_psi = Gauge("mlops_feature_psi", "PSI per feature", ["feature"], registry=r)
        self.feature_drifted = Gauge("mlops_feature_drifted", "1 if the feature drifted", ["feature"], registry=r)
        self.live_auc = Gauge("mlops_live_roc_auc", "Live ROC-AUC on delayed labels", registry=r)
        self.baseline_auc = Gauge("mlops_baseline_roc_auc", "Baseline ROC-AUC of the serving model", registry=r)
        self.pred_psi = Gauge("mlops_prediction_psi", "PSI of predicted probabilities vs reference", registry=r)
        self.status = Gauge("mlops_monitoring_status", "0=ok 1=warning 2=critical -1=insufficient data", registry=r)
        self.last_run = Gauge("mlops_monitoring_last_run_timestamp", "Unix time of last monitoring run", registry=r)

    def refresh_monitoring(self, store: PredictionStore) -> None:
        run = store.latest_monitoring_run()
        if not run:
            return
        self.status.set({"OK": 0, "WARNING": 1, "CRITICAL": 2}.get(run["status"], -1))
        self.last_run.set(from_iso(run["ts"]).timestamp())
        for gauge, key in ((self.drift_share, "share_drifted"), (self.live_auc, "live_auc"),
                           (self.baseline_auc, "baseline_auc"), (self.pred_psi, "prediction_psi")):
            if run.get(key) is not None:
                gauge.set(run[key])
        if run.get("dataset_drift") is not None:
            self.dataset_drift.set(run["dataset_drift"])
        for f in (run.get("details") or {}).get("features", []):
            self.feature_psi.labels(f["feature"]).set(f["psi"])
            self.feature_drifted.labels(f["feature"]).set(int(f["drifted"]))


# ------------------------------------------------------------------------------------ app
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = PredictionStore(settings.prediction_db)
    manager = ModelManager(settings)
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            if manager.load():
                log.info("Model v%s loaded", manager.version)
            else:
                log.warning("No champion model yet - /predict returns 503 until one is promoted.")
        except Exception:  # registry unavailable at boot must not crash the pod
            log.exception("Could not load model at startup")
        yield

    app = FastAPI(
        title="Churn Prediction API",
        version=__version__,
        description="Serves the registry *champion* model, logs every prediction for drift monitoring "
        "and accepts delayed ground-truth feedback.",
        lifespan=lifespan,
    )
    app.state.manager, app.state.store, app.state.metrics, app.state.settings = manager, store, metrics, settings

    def _require_model() -> None:
        try:
            manager.maybe_refresh()
        except Exception:
            log.exception("Model refresh failed; continuing with the loaded model")
        if not manager.ready:
            raise HTTPException(503, "No champion model available. Train and promote one first.")

    def _score(customers: list[Customer], endpoint: str) -> list[Prediction]:
        _require_model()
        df = pd.DataFrame([c.model_dump() for c in customers])
        t0 = time.perf_counter()
        # every prediction is persisted (single batched INSERT) - this is the monitoring data source
        scored = manager.predict_and_log(df, store, ts=utcnow())
        out = []
        for row in scored.itertuples():
            churn = row.probability >= 0.5
            metrics.predictions.labels(row.model_version, str(churn).lower()).inc()
            metrics.proba.observe(row.probability)
            out.append(Prediction(prediction_id=row.prediction_id, churn_probability=round(row.probability, 6),
                                  churn_prediction=bool(churn), model_version=row.model_version))
        metrics.latency.labels(endpoint).observe(time.perf_counter() - t0)
        metrics.requests.labels(endpoint, "200").inc()
        return out

    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok" if manager.ready else "no_model", "model_version": manager.version}

    @app.get("/model", tags=["ops"])
    def model_info():
        _require_model()
        run = store.latest_monitoring_run()
        return {
            "name": settings.training.registered_model_name,
            "version": manager.version,
            "features": FEATURES,
            "latest_monitoring": None if run is None else {k: run[k] for k in ("ts", "status", "action")},
        }

    @app.post("/predict", response_model=Prediction, tags=["inference"])
    def predict(customer: Customer):
        return _score([customer], "predict")[0]

    @app.post("/predict/batch", response_model=list[Prediction], tags=["inference"])
    def predict_batch(req: BatchRequest):
        return _score(req.customers, "predict_batch")

    @app.post("/feedback", tags=["inference"])
    def feedback(items: FeedbackBatch):
        """Submit ground truth for earlier predictions (e.g. 'customer churned 60 days later')."""
        n = store.add_labels([(i.prediction_id, int(i.actual_churn)) for i in items.items])
        metrics.requests.labels("feedback", "200").inc()
        return {"accepted": n}

    @app.post("/admin/reload", tags=["ops"])
    def reload_model():
        if not manager.load():
            raise HTTPException(503, "No champion model available")
        return {"model_version": manager.version}

    @app.get("/metrics", tags=["ops"])
    def prometheus_metrics():
        if manager.version:
            metrics.model_info.clear()
            metrics.model_info.labels(manager.version).set(1)
        metrics.refresh_monitoring(store)
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/monitoring/latest", tags=["ops"])
    def latest_monitoring():
        run = store.latest_monitoring_run()
        if run is None:
            raise HTTPException(404, "No monitoring run yet")
        return json.loads(json.dumps(run, default=str))

    return app


def app_factory() -> FastAPI:  # uvicorn --factory mlops_pipeline.serving.api:app_factory
    return create_app()
