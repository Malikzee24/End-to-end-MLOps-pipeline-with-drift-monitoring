"""Time-compressed production simulation: months of traffic, drift and retraining in ~1 minute.

Timeline (one step = one week, ~800 customers/week, labels arrive one week late):

    weeks  0-3   stable traffic             -> monitoring stays OK
    weeks  4-8   gradual covariate drift    -> feature drift detected WITHOUT labels
    weeks  9-14  concept drift on top       -> live AUC drops once labels arrive
    afterwards   plateau at the new regime  -> models are retrained, monitoring settles again
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from mlops_pipeline import registry
from mlops_pipeline.config import Settings
from mlops_pipeline.data import DriftSpec, generate_churn_data
from mlops_pipeline.monitoring import run_monitoring
from mlops_pipeline.schema import TARGET
from mlops_pipeline.serving.predictor import ModelManager
from mlops_pipeline.storage import PredictionStore
from mlops_pipeline.training import promote_if_better, train

log = logging.getLogger(__name__)
START = datetime(2025, 1, 6)  # a Monday
MARKER = ".mlops_state"


def drift_scenario(week: int) -> DriftSpec:
    """Drift intensity for a given simulated week."""
    cov = min(max(week - 3, 0) / 5.0, 1.0)  # ramps 0 -> 1 over weeks 4..8
    concept = min(max(week - 8, 0) / 6.0, 1.0)  # ramps 0 -> 1 over weeks 9..14
    return DriftSpec(
        charges_shift=18.0 * cov,
        support_calls_shift=0.9 * cov,
        month_to_month_boost=0.22 * cov,
        age_shift=0.0,
        concept_shift=concept,
    )


KNOWN_STATE_ENTRIES = {"reports", "mlartifacts", "mlflow.db", "predictions.db", "predictions.db-wal",
                       "predictions.db-shm", MARKER}


def _prepare_state(settings: Settings, fresh: bool) -> None:
    """Wipe the demo state directory - but only if it only contains files this tool creates."""
    d = settings.state_dir
    if fresh and d.exists():
        if {p.name for p in d.iterdir()} <= KNOWN_STATE_ENTRIES:
            shutil.rmtree(d)
        else:
            raise RuntimeError(f"Refusing to delete {d}: it contains files not created by this tool.")
    d.mkdir(parents=True, exist_ok=True)
    (d / MARKER).write_text("created by mlops-drift-monitoring demo\n")
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_root.mkdir(parents=True, exist_ok=True)


def run_demo(
    settings: Settings,
    *,
    weeks: int = 22,
    batch_size: int = 800,
    label_delay_weeks: int = 1,
    seed: int = 7,
    fresh: bool = True,
    plot_path: str | Path | None = None,
) -> pd.DataFrame:
    _prepare_state(settings, fresh)
    registry.configure_mlflow(settings)
    store = PredictionStore(settings.prediction_db)
    manager = ModelManager(settings, refresh_seconds=0)

    # ---- week 0: train the initial champion on historical data -------------------------------
    history = generate_churn_data(settings.data.n_train, seed=settings.data.seed)
    result = train(settings, history, trigger="initial")
    promote_if_better(settings, result.model_version, challenger_test_auc=result.metrics["test_roc_auc"])
    manager.load()

    pending: dict[int, pd.DataFrame] = {}  # week -> (prediction_id, true label)
    timeline: list[dict] = []

    for week in range(weeks):
        week_start = START + timedelta(weeks=week)
        week_end = week_start + timedelta(weeks=1)
        spec = drift_scenario(week)
        batch = generate_churn_data(batch_size, seed=seed * 1000 + week, drift=spec)

        step = 7 * 24 * 3600 / batch_size  # spread the week's traffic evenly over the week
        stamps = [week_start + timedelta(seconds=int(i * step) + 1) for i in range(batch_size)]
        manager.maybe_refresh()
        scored = manager.predict_and_log(batch, store, ts=stamps)
        pending[week] = pd.DataFrame({"prediction_id": scored["prediction_id"], "label": batch[TARGET].to_numpy()})

        # ground truth for older weeks arrives with a delay
        due = week - label_delay_weeks
        if due in pending:
            lab = pending.pop(due)
            store.add_labels(list(zip(lab["prediction_id"], lab["label"])), labeled_at=week_end)

        res = run_monitoring(settings, store, as_of=week_end)
        manager.maybe_refresh()  # pick up a promotion made by auto-retraining

        timeline.append(
            {
                "week": week,
                "as_of": week_end,
                "status": res.status,
                "action": res.action,
                "model_version": int(res.model_version),
                "serving_version_next": int(manager.version),
                "share_drifted": res.drift.share_drifted if res.drift else None,
                "drifted_features": ",".join(res.drift.drifted_features) if res.drift else "",
                "live_auc": res.performance.roc_auc if res.performance and res.performance.available else None,
                "baseline_auc": res.performance.baseline_auc if res.performance else None,
                "retrained": bool(res.retrain and res.retrain.attempted),
                "promoted": bool(res.retrain and res.retrain.promoted),
                "true_drift_cov": round(spec.charges_shift / 18.0, 2),
                "true_drift_concept": round(spec.concept_shift, 2),
            }
        )
        t = timeline[-1]
        log.info("week %02d | %-17s | drifted=%s | live_auc=%s | model v%s%s", week, t["status"],
                 "-" if t["share_drifted"] is None else f"{t['share_drifted']:.0%}",
                 "-" if t["live_auc"] is None else f"{t['live_auc']:.3f}", t["serving_version_next"],
                 "  <-- RETRAINED & PROMOTED" if t["promoted"] else ("  <-- retrain attempted" if t["retrained"] else ""))

    df = pd.DataFrame(timeline)
    df.to_csv(settings.report_dir / "demo_timeline.csv", index=False)
    if plot_path:
        plot_timeline(df, plot_path, settings.monitoring.drift.dataset_drift_share,
                      settings.monitoring.performance.max_auc_drop)
    return df


def plot_timeline(df: pd.DataFrame, path: str | Path, drift_threshold: float = 0.3, max_drop: float = 0.05) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.1, 0.7]})
    wk = df["week"]
    promos = df.loc[df["promoted"], "week"].tolist()

    ax = axes[0]
    ax.plot(wk, df["true_drift_cov"], "--", color="#8b949e", label="injected covariate drift (0-1)")
    ax.plot(wk, df["share_drifted"], "o-", color="#cf222e", label="detected: share of drifted features")
    ax.axhline(drift_threshold, color="#cf222e", ls=":", lw=1)
    ax.text(wk.max(), drift_threshold + 0.02, "dataset-drift threshold", ha="right", fontsize=8, color="#cf222e")
    ax.set_ylabel("data drift")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Continuous drift monitoring with automated retraining (simulated {len(df)} weeks)",
                 fontsize=12, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(wk, df["baseline_auc"], "--", color="#8b949e", label="serving model's baseline AUC")
    ax.plot(wk, df["live_auc"], "o-", color="#0969da", label="live AUC (delayed labels)")
    ax.plot(wk, df["baseline_auc"].ffill() - max_drop, ":", color="#0969da", lw=1,
            label=f"alert line (baseline - {max_drop})")
    ax.set_ylabel("ROC-AUC")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    ax = axes[2]
    ax.step(wk, df["serving_version_next"], where="post", color="#1a7f37", lw=2)
    ax.set_ylabel("serving model\nversion")
    ax.set_xlabel("simulated week")
    ax.set_yticks(sorted(df["serving_version_next"].unique()))

    for a in axes:
        for p in promos:
            a.axvline(p, color="#1a7f37", lw=1, alpha=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    if promos:
        axes[0].text(promos[0], 1.0, " retrain + promote", color="#1a7f37", fontsize=8, va="top")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
