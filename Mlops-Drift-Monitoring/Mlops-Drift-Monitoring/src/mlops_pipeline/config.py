"""Typed configuration: YAML file (configs/config.yaml) + environment overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class DataConfig(BaseModel):
    n_train: int = 10_000
    seed: int = 42
    test_size: float = 0.2
    reference_max_rows: int = 5_000


class CandidateConfig(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    experiment_name: str = "churn-prediction"
    registered_model_name: str = "churn-classifier"
    cv_folds: int = 5
    candidates: list[CandidateConfig] = Field(
        default_factory=lambda: [CandidateConfig(name="logistic_regression")]
    )


class PromotionConfig(BaseModel):
    champion_alias: str = "champion"
    min_absolute_auc: float = 0.65
    min_auc_improvement: float = 0.0


class DriftConfig(BaseModel):
    alpha: float = 0.05  # significance level for KS / chi-square tests
    psi_threshold: float = 0.2  # PSI >= 0.2 is conventionally "significant shift"
    min_effect_size: float = 0.1  # KS statistic / Jensen-Shannon distance guard
    dataset_drift_share: float = 0.3  # share of drifted features => dataset drift
    n_bins: int = 10


class PerformanceConfig(BaseModel):
    min_labeled_samples: int = 200
    max_auc_drop: float = 0.05


class RetrainConfig(BaseModel):
    enabled: bool = True
    cooldown_hours: float = 336.0
    lookback_hours: float = 672.0
    min_labeled_samples: int = 1_200
    history_rows: int = 1_000
    holdout_fraction: float = 0.3


class MonitoringConfig(BaseModel):
    window_hours: float = 336.0
    min_samples: int = 300
    drift: DriftConfig = Field(default_factory=DriftConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    retrain: RetrainConfig = Field(default_factory=RetrainConfig)


class ServingConfig(BaseModel):
    model_refresh_seconds: float = 30.0


class Settings(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    state_dir: Path = Path("state")

    # ---- derived locations (everything mutable lives under state_dir) -------------------
    @property
    def tracking_uri(self) -> str:
        # NB: deliberately NOT MLFLOW_TRACKING_URI - mlflow.set_tracking_uri() writes that variable
        # itself, which would leak one settings object's backend into the next.
        return os.getenv("MLOPS_TRACKING_URI") or f"sqlite:///{(self.state_dir / 'mlflow.db').resolve()}"

    @property
    def artifact_root(self) -> Path:
        return (self.state_dir / "mlartifacts").resolve()

    @property
    def prediction_db(self) -> Path:
        return self.state_dir / "predictions.db"

    @property
    def report_dir(self) -> Path:
        return self.state_dir / "reports"


def load_settings(config_path: str | Path | None = None, state_dir: str | Path | None = None) -> Settings:
    """Load settings from YAML (``MLOPS_CONFIG`` env var or default) and ensure state dirs exist."""
    path = Path(config_path or os.getenv("MLOPS_CONFIG") or DEFAULT_CONFIG_PATH)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    settings = Settings(**(raw or {}))
    settings.state_dir = Path(state_dir or os.getenv("MLOPS_STATE_DIR") or settings.state_dir)
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_root.mkdir(parents=True, exist_ok=True)
    return settings
