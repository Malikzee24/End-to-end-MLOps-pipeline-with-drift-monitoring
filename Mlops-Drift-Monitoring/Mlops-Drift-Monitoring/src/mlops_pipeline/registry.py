"""Thin wrapper around MLflow Tracking + Model Registry.

Promotion uses registry **aliases** (``champion``), the modern replacement for the deprecated
"Staging/Production" stages. Serving always resolves ``models:/<name>@champion`` so a promotion
or rollback is a single alias move - no redeploy required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from mlops_pipeline.config import Settings

log = logging.getLogger(__name__)

REFERENCE_ARTIFACT_DIR = "reference"
REFERENCE_FILE = "reference.parquet"


def configure_mlflow(settings: Settings) -> MlflowClient:
    """Point MLflow at the configured backend and make sure the experiment exists."""
    mlflow.set_tracking_uri(settings.tracking_uri)
    client = MlflowClient()
    name = settings.training.experiment_name
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name, artifact_location=settings.artifact_root.as_uri())
    mlflow.set_experiment(name)
    return client


def get_champion_version(settings: Settings) -> ModelVersion | None:
    client = configure_mlflow(settings)
    try:
        return client.get_model_version_by_alias(
            settings.training.registered_model_name, settings.promotion.champion_alias
        )
    except mlflow.exceptions.MlflowException:
        return None


def load_champion(settings: Settings):
    """Return ``(sklearn_pipeline, ModelVersion)`` for the current champion, or ``(None, None)``."""
    mv = get_champion_version(settings)
    if mv is None:
        return None, None
    model = mlflow.sklearn.load_model(f"models:/{mv.name}/{mv.version}")
    return model, mv


def load_model_version(settings: Settings, version: str | int):
    configure_mlflow(settings)
    return mlflow.sklearn.load_model(f"models:/{settings.training.registered_model_name}/{version}")


def promote(settings: Settings, version: str | int) -> None:
    client = configure_mlflow(settings)
    client.set_registered_model_alias(
        settings.training.registered_model_name, settings.promotion.champion_alias, str(version)
    )
    log.info("Promoted %s v%s to alias '%s'", settings.training.registered_model_name, version,
             settings.promotion.champion_alias)


def load_reference(settings: Settings, mv: ModelVersion) -> pd.DataFrame:
    """Reference (training-time) dataset logged next to the model run - always in sync with the model."""
    client = configure_mlflow(settings)
    local = client.download_artifacts(mv.run_id, f"{REFERENCE_ARTIFACT_DIR}/{REFERENCE_FILE}")
    return pd.read_parquet(Path(local))


def baseline_auc(settings: Settings, mv: ModelVersion) -> float | None:
    """Expected ROC-AUC of a model version: explicit tag (set on retrain) or its logged test AUC."""
    client = configure_mlflow(settings)
    mv = client.get_model_version(mv.name, mv.version)  # refresh: tags may have been added after promotion
    tagged = mv.tags.get("baseline_auc") if mv.tags else None
    if tagged is not None:
        return float(tagged)
    run = client.get_run(mv.run_id)
    value = run.data.metrics.get("test_roc_auc")
    return float(value) if value is not None else None


def set_version_tag(settings: Settings, version: str | int, key: str, value: str) -> None:
    client = configure_mlflow(settings)
    client.set_model_version_tag(settings.training.registered_model_name, str(version), key, str(value))
