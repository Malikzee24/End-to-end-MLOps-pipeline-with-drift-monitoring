"""Automated retraining: recent labeled data -> challenger -> gate vs. champion -> promote."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_pipeline import registry
from mlops_pipeline.config import Settings
from mlops_pipeline.schema import FEATURES, TARGET
from mlops_pipeline.storage import PredictionStore, from_iso, utcnow
from mlops_pipeline.training import promote_if_better, train

log = logging.getLogger(__name__)


@dataclass
class RetrainOutcome:
    attempted: bool
    promoted: bool
    reason: str
    old_version: str | None = None
    new_version: str | None = None
    champion_auc: float | None = None
    challenger_auc: float | None = None


def retrain(
    settings: Settings,
    store: PredictionStore,
    *,
    as_of: datetime | None = None,
    trigger: str = "drift",
    force: bool = False,
) -> RetrainOutcome:
    """Retrain on recent labeled production data and promote only if the challenger wins.

    Safeguards: cooldown (no retrain thrash), minimum labeled volume, and a champion-vs-challenger
    comparison on a hold-out that the challenger never saw.
    """
    cfg = settings.monitoring.retrain
    as_of = as_of or utcnow()
    champion = registry.get_champion_version(settings)
    old_version = str(champion.version) if champion else None

    if not force:
        last = store.last_retrain(promoted_only=False)  # failed attempts also start the cooldown
        if last is not None:
            if as_of - from_iso(last["ts"]) < timedelta(hours=cfg.cooldown_hours):
                return RetrainOutcome(False, False, "cooldown active", old_version)

    recent = store.get_predictions(since=as_of - timedelta(hours=cfg.lookback_hours), until=as_of)
    labeled = recent[recent["label"].notna()].copy() if not recent.empty else pd.DataFrame()
    if len(labeled) < cfg.min_labeled_samples:
        reason = f"insufficient labeled data ({len(labeled)} < {cfg.min_labeled_samples})"
        log.warning("Retraining skipped: %s", reason)
        return RetrainOutcome(False, False, reason, old_version)

    labeled[TARGET] = labeled["label"].astype(int)
    labeled = labeled[FEATURES + [TARGET]]
    fit_part, holdout = train_test_split(
        labeled, test_size=cfg.holdout_fraction, stratify=labeled[TARGET], random_state=settings.data.seed
    )

    # Mix in a little older data (the champion's reference) for stability on rare segments.
    history = pd.DataFrame()
    if champion is not None and cfg.history_rows > 0:
        ref = registry.load_reference(settings, champion)
        if TARGET in ref.columns:
            history = ref.sample(min(cfg.history_rows, len(ref)), random_state=settings.data.seed)
    train_df = pd.concat([fit_part, history[FEATURES + [TARGET]]] if not history.empty else [fit_part],
                         ignore_index=True)

    result = train(
        settings,
        train_df,
        trigger=trigger,
        reference_override=fit_part,  # drift reference = the "new normal", not the old mix
        tags={"retrain_as_of": as_of.isoformat(), "n_recent_labeled": str(len(labeled))},
    )
    decision = promote_if_better(settings, result.model_version, eval_df=holdout)
    log.info("Retrain decision: %s", decision.reason)

    store.save_retrain_event(
        {
            "ts": as_of,
            "trigger": trigger,
            "old_version": old_version,
            "new_version": result.model_version,
            "promoted": decision.promoted,
            "champion_auc": decision.champion_auc,
            "challenger_auc": decision.challenger_auc,
            "details": {"reason": decision.reason, "n_train": len(train_df), "n_holdout": len(holdout),
                        "best_candidate": result.best_candidate},
        }
    )
    return RetrainOutcome(
        True, decision.promoted, decision.reason, old_version, result.model_version,
        decision.champion_auc, decision.challenger_auc,
    )
