"""Training: model selection by cross-validation, full MLflow tracking and registry registration."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_pipeline import registry
from mlops_pipeline.config import CandidateConfig, Settings
from mlops_pipeline.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

log = logging.getLogger(__name__)

ESTIMATORS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


@dataclass
class TrainResult:
    run_id: str
    model_version: str
    best_candidate: str
    metrics: dict[str, float]
    cv_scores: dict[str, float]
    data_hash: str
    extra: dict = field(default_factory=dict)


def build_pipeline(candidate: CandidateConfig, seed: int = 42) -> Pipeline:
    """Preprocessing + estimator in ONE sklearn Pipeline, so serving can never drift from training."""
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
             NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    params = dict(candidate.params)
    cls = ESTIMATORS[candidate.name]
    if "random_state" in cls().get_params():
        params.setdefault("random_state", seed)
    return Pipeline([("preprocess", pre), ("model", cls(**params))])


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "avg_precision": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def data_fingerprint(df: pd.DataFrame) -> str:
    """Stable content hash of a dataframe (lineage: which data produced which model)."""
    import hashlib

    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def train(
    settings: Settings,
    df: pd.DataFrame,
    *,
    trigger: str = "initial",
    reference_override: pd.DataFrame | None = None,
    seed: int | None = None,
    tags: dict[str, str] | None = None,
) -> TrainResult:
    """Train every candidate, pick the best by CV ROC-AUC, evaluate on a hold-out and register it.

    ``reference_override`` lets retraining declare which rows represent the "new normal" for
    drift detection (instead of the whole mixed training set).
    """
    seed = settings.data.seed if seed is None else seed
    registry.configure_mlflow(settings)

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=settings.data.test_size, stratify=y, random_state=seed
    )
    cv = StratifiedKFold(n_splits=settings.training.cv_folds, shuffle=True, random_state=seed)
    data_hash = data_fingerprint(df)

    with mlflow.start_run(run_name=f"train-{trigger}") as parent:
        mlflow.set_tags({"trigger": trigger, "data_hash": data_hash, **(tags or {})})
        mlflow.log_params(
            {
                "n_rows": len(df),
                "churn_rate": round(float(y.mean()), 4),
                "seed": seed,
                "cv_folds": settings.training.cv_folds,
                "candidates": ",".join(c.name for c in settings.training.candidates),
            }
        )

        # ---- model selection (one nested run per candidate) --------------------------------
        cv_scores: dict[str, float] = {}
        for cand in settings.training.candidates:
            with mlflow.start_run(run_name=cand.name, nested=True):
                pipe = build_pipeline(cand, seed)
                scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
                cv_scores[cand.name] = float(scores.mean())
                mlflow.log_params({f"param_{k}": v for k, v in cand.params.items()})
                mlflow.log_metrics({"cv_roc_auc_mean": float(scores.mean()), "cv_roc_auc_std": float(scores.std())})
                log.info("CV  %-24s AUC=%.4f ± %.4f", cand.name, scores.mean(), scores.std())

        best_name = max(cv_scores, key=cv_scores.get)
        best_cfg = next(c for c in settings.training.candidates if c.name == best_name)
        model = build_pipeline(best_cfg, seed).fit(X_tr, y_tr)

        proba_te = model.predict_proba(X_te)[:, 1]
        test_metrics = classification_metrics(y_te, proba_te)
        mlflow.set_tag("best_candidate", best_name)
        mlflow.log_param("best_candidate", best_name)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("cv_roc_auc_best", cv_scores[best_name])

        # ---- reference data for drift monitoring: logged WITH the model ---------------------
        ref_src = reference_override if reference_override is not None else X_tr.assign(**{TARGET: y_tr})
        ref_cols = [c for c in FEATURES + [TARGET] if c in ref_src.columns]
        reference = ref_src[ref_cols]
        if len(reference) > settings.data.reference_max_rows:
            reference = reference.sample(settings.data.reference_max_rows, random_state=seed)
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / registry.REFERENCE_FILE
            reference.reset_index(drop=True).to_parquet(ref_path)
            mlflow.log_artifact(str(ref_path), registry.REFERENCE_ARTIFACT_DIR)

        # ---- model + signature -> registry --------------------------------------------------
        signature = infer_signature(X_tr.head(100), model.predict_proba(X_tr.head(100))[:, 1])
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_tr.head(5),
            registered_model_name=settings.training.registered_model_name,
        )

        client = mlflow.MlflowClient()
        versions = client.search_model_versions(
            f"name='{settings.training.registered_model_name}' and run_id='{parent.info.run_id}'"
        )
        version = str(max(int(v.version) for v in versions))
        client.set_model_version_tag(settings.training.registered_model_name, version, "trigger", trigger)
        client.set_model_version_tag(settings.training.registered_model_name, version, "data_hash", data_hash)

        log.info("Registered %s v%s (%s) test AUC=%.4f", settings.training.registered_model_name, version,
                 best_name, test_metrics["roc_auc"])
        return TrainResult(
            run_id=parent.info.run_id,
            model_version=version,
            best_candidate=best_name,
            metrics={f"test_{k}": v for k, v in test_metrics.items()},
            cv_scores=cv_scores,
            data_hash=data_hash,
        )
