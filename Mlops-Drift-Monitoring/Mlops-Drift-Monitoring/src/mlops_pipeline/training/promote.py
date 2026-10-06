"""Champion / challenger promotion gate."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

from mlops_pipeline import registry
from mlops_pipeline.config import Settings
from mlops_pipeline.schema import FEATURES, TARGET

log = logging.getLogger(__name__)


@dataclass
class PromotionDecision:
    promoted: bool
    reason: str
    challenger_version: str
    champion_version: str | None
    challenger_auc: float | None = None
    champion_auc: float | None = None


def evaluate_model(model, df: pd.DataFrame) -> float:
    return float(roc_auc_score(df[TARGET], model.predict_proba(df[FEATURES])[:, 1]))


def promote_if_better(
    settings: Settings,
    challenger_version: str,
    *,
    eval_df: pd.DataFrame | None = None,
    challenger_test_auc: float | None = None,
) -> PromotionDecision:
    """Promote the challenger if it clears the absolute quality bar and beats the champion.

    * No champion yet       -> promote if ``challenger_test_auc`` >= ``min_absolute_auc``.
    * Champion exists       -> both models are scored on the *same* ``eval_df`` (recent labeled
      production data), and the challenger must win by ``min_auc_improvement``.
    """
    cfg = settings.promotion
    champion = registry.get_champion_version(settings)
    champion_version = str(champion.version) if champion else None  # MLflow may return int or str

    if champion is None:
        auc = challenger_test_auc
        if auc is None and eval_df is not None:
            auc = evaluate_model(registry.load_model_version(settings, challenger_version), eval_df)
        if auc is None or auc < cfg.min_absolute_auc:
            return PromotionDecision(False, f"AUC {auc} below absolute bar {cfg.min_absolute_auc}",
                                     challenger_version, None, challenger_auc=auc)
        registry.promote(settings, challenger_version)
        registry.set_version_tag(settings, challenger_version, "baseline_auc", f"{auc:.6f}")
        return PromotionDecision(True, "first model - promoted to champion", challenger_version, None,
                                 challenger_auc=auc)

    if eval_df is None:
        raise ValueError("eval_df is required when a champion already exists")

    champ_auc = evaluate_model(registry.load_model_version(settings, champion_version), eval_df)
    chall_auc = evaluate_model(registry.load_model_version(settings, challenger_version), eval_df)
    registry.set_version_tag(settings, challenger_version, "eval_auc_vs_champion", f"{chall_auc:.6f}")

    if chall_auc < cfg.min_absolute_auc:
        return PromotionDecision(False, f"challenger AUC {chall_auc:.3f} below absolute bar", challenger_version,
                                 champion_version, chall_auc, champ_auc)
    if chall_auc > champ_auc + cfg.min_auc_improvement:  # strict: a tie never dethrones the champion
        registry.promote(settings, challenger_version)
        registry.set_version_tag(settings, challenger_version, "baseline_auc", f"{chall_auc:.6f}")
        return PromotionDecision(
            True, f"challenger beats champion ({chall_auc:.3f} vs {champ_auc:.3f})",
            challenger_version, champion_version, chall_auc, champ_auc,
        )
    return PromotionDecision(
        False, f"challenger does not beat champion ({chall_auc:.3f} vs {champ_auc:.3f})",
        challenger_version, champion_version, chall_auc, champ_auc,
    )
