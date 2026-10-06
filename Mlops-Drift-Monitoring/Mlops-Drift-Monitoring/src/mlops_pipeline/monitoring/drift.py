"""Statistical drift detection (implemented from scratch on scipy / numpy).

For every feature we compare the *reference* distribution (what the model was trained on)
with the *current* production window:

====================  ==========================  ===============================
                      numeric                     categorical
====================  ==========================  ===============================
significance test     two-sample KS test          chi-square test of homogeneity
effect size           KS statistic                Jensen-Shannon distance
population stability  PSI (reference quantiles)   PSI (category proportions)
====================  ==========================  ===============================

A feature is flagged as **drifted** when

    PSI >= psi_threshold                                   (large shift), or
    p-value < alpha  AND  effect_size >= min_effect_size   (significant AND material)

The effect-size guard matters: with thousands of rows, p-values flag *trivial* differences.
Requiring a material effect keeps the false-alarm rate low. Dataset drift is declared when the
share of drifted features reaches ``dataset_drift_share``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from mlops_pipeline.config import DriftConfig
from mlops_pipeline.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

EPS = 1e-4


def _psi_from_proportions(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    p_ref = np.clip(p_ref, EPS, None)
    p_cur = np.clip(p_cur, EPS, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi(reference: pd.Series | np.ndarray, current: pd.Series | np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index for a numeric variable, using reference-quantile bins.

    Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, >= 0.2 significant shift.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:  # (near-)constant reference
        edges = np.array([-np.inf, np.median(ref), np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, bins=edges)[0] / max(len(ref), 1)
    p_cur = np.histogram(cur, bins=edges)[0] / max(len(cur), 1)
    return _psi_from_proportions(p_ref, p_cur)


@dataclass
class FeatureDrift:
    feature: str
    kind: str  # "numeric" | "categorical"
    drifted: bool
    psi: float
    p_value: float
    effect_size: float  # KS statistic (numeric) or Jensen-Shannon distance (categorical)
    test: str
    ref_summary: dict = field(default_factory=dict)
    cur_summary: dict = field(default_factory=dict)


@dataclass
class DriftReport:
    features: list[FeatureDrift]
    share_drifted: float
    dataset_drift: bool
    n_reference: int
    n_current: int
    prediction_psi: float | None = None

    @property
    def drifted_features(self) -> list[str]:
        return [f.feature for f in self.features if f.drifted]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["drifted_features"] = self.drifted_features
        return d


def _numeric_drift(name: str, ref: pd.Series, cur: pd.Series, cfg: DriftConfig) -> FeatureDrift:
    ks = stats.ks_2samp(ref.dropna(), cur.dropna())
    p = psi(ref, cur, cfg.n_bins)
    drifted = p >= cfg.psi_threshold or (ks.pvalue < cfg.alpha and ks.statistic >= cfg.min_effect_size)
    return FeatureDrift(
        feature=name, kind="numeric", drifted=bool(drifted), psi=p, p_value=float(ks.pvalue),
        effect_size=float(ks.statistic), test="ks_2samp",
        ref_summary={"mean": float(ref.mean()), "std": float(ref.std())},
        cur_summary={"mean": float(cur.mean()), "std": float(cur.std())},
    )


def _categorical_drift(name: str, ref: pd.Series, cur: pd.Series, cfg: DriftConfig) -> FeatureDrift:
    cats = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()))
    ref_counts = ref.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    cur_counts = cur.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    p_ref, p_cur = ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum()
    p = _psi_from_proportions(p_ref, p_cur)
    jsd = float(jensenshannon(p_ref, p_cur, base=2)) if len(cats) > 1 else 0.0
    p_value = float(stats.chi2_contingency(np.vstack([ref_counts, cur_counts]))[1]) if len(cats) > 1 else 1.0
    drifted = p >= cfg.psi_threshold or (p_value < cfg.alpha and jsd >= cfg.min_effect_size)
    return FeatureDrift(
        feature=name, kind="categorical", drifted=bool(drifted), psi=p, p_value=p_value, effect_size=jsd,
        test="chi2_contingency",
        ref_summary=dict(zip(cats, map(float, p_ref))),
        cur_summary=dict(zip(cats, map(float, p_cur))),
    )


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: DriftConfig | None = None,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> DriftReport:
    cfg = cfg or DriftConfig()
    numeric = [c for c in (numeric or NUMERIC_FEATURES) if c in reference and c in current]
    categorical = [c for c in (categorical or CATEGORICAL_FEATURES) if c in reference and c in current]

    results = [_numeric_drift(c, reference[c], current[c], cfg) for c in numeric]
    results += [_categorical_drift(c, reference[c], current[c], cfg) for c in categorical]
    share = float(np.mean([r.drifted for r in results])) if results else 0.0
    return DriftReport(
        features=results,
        share_drifted=share,
        dataset_drift=share >= cfg.dataset_drift_share,
        n_reference=len(reference),
        n_current=len(current),
    )
