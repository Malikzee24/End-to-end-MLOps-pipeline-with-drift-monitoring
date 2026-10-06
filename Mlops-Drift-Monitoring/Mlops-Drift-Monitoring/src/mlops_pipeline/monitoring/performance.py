"""Live model performance once (delayed) ground-truth labels arrive."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from mlops_pipeline.config import PerformanceConfig


@dataclass
class PerformanceReport:
    available: bool
    n_labeled: int
    reason: str = ""
    roc_auc: float | None = None
    avg_precision: float | None = None
    f1: float | None = None
    brier: float | None = None
    baseline_auc: float | None = None
    auc_drop: float | None = None
    degraded: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_live_performance(
    labeled: pd.DataFrame, baseline_auc: float | None, cfg: PerformanceConfig | None = None
) -> PerformanceReport:
    """``labeled`` needs ``label`` and ``probability`` columns (rows with known outcomes only)."""
    cfg = cfg or PerformanceConfig()
    n = len(labeled)
    if n < cfg.min_labeled_samples:
        return PerformanceReport(False, n, f"only {n} labeled rows (< {cfg.min_labeled_samples})",
                                 baseline_auc=baseline_auc)
    y = labeled["label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return PerformanceReport(False, n, "labels contain a single class", baseline_auc=baseline_auc)

    proba = labeled["probability"].to_numpy(dtype=float)
    auc = float(roc_auc_score(y, proba))
    drop = None if baseline_auc is None else float(baseline_auc - auc)
    return PerformanceReport(
        available=True,
        n_labeled=n,
        roc_auc=auc,
        avg_precision=float(average_precision_score(y, proba)),
        f1=float(f1_score(y, (proba >= 0.5).astype(int))),
        brier=float(brier_score_loss(y, proba)),
        baseline_auc=baseline_auc,
        auc_drop=drop,
        degraded=bool(drop is not None and drop > cfg.max_auc_drop),
    )
