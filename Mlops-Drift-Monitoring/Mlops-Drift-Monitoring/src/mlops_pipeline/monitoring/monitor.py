"""Monitoring orchestrator: drift + performance -> status -> action (optionally auto-retrain)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from mlops_pipeline import registry
from mlops_pipeline.config import Settings
from mlops_pipeline.monitoring.drift import DriftReport, detect_drift, psi
from mlops_pipeline.monitoring.performance import PerformanceReport, evaluate_live_performance
from mlops_pipeline.monitoring.report import render_html_report
from mlops_pipeline.pipeline.retrain import RetrainOutcome, retrain
from mlops_pipeline.schema import FEATURES
from mlops_pipeline.storage import PredictionStore, to_iso, utcnow

log = logging.getLogger(__name__)

OK, WARNING, CRITICAL, INSUFFICIENT = "OK", "WARNING", "CRITICAL", "INSUFFICIENT_DATA"


@dataclass
class MonitoringResult:
    as_of: datetime
    model_version: str
    status: str
    action: str
    drift: DriftReport | None = None
    performance: PerformanceReport | None = None
    report_path: str | None = None
    retrain: RetrainOutcome | None = None
    notes: list[str] = field(default_factory=list)


def run_monitoring(
    settings: Settings,
    store: PredictionStore,
    *,
    as_of: datetime | None = None,
    auto_retrain: bool | None = None,
    write_report: bool = True,
) -> MonitoringResult:
    """One monitoring cycle. Safe to call from cron / a loop / CI.

    Decision logic
    --------------
    * fewer than ``min_samples`` rows in the window    -> INSUFFICIENT_DATA (no action)
    * dataset drift  OR  live AUC dropped too far      -> CRITICAL, action RETRAIN
    * some features drifted, but below the threshold   -> WARNING, action WATCH
    * otherwise                                        -> OK
    """
    cfg = settings.monitoring
    as_of = as_of or utcnow()
    auto_retrain = cfg.retrain.enabled if auto_retrain is None else auto_retrain

    model, champion = registry.load_champion(settings)
    if champion is None:
        raise RuntimeError("No champion model in the registry - run `mlops train` first.")

    window = store.get_predictions(since=as_of - timedelta(hours=cfg.window_hours), until=as_of)
    notes: list[str] = []
    drift: DriftReport | None = None
    perf: PerformanceReport | None = None
    reference: pd.DataFrame | None = None

    if window.empty or len(window) < cfg.min_samples:
        n = 0 if window.empty else len(window)
        status, action = INSUFFICIENT, "NONE"
        notes.append(f"{n} predictions in window (< min_samples={cfg.min_samples}); tests skipped.")
    else:
        reference = registry.load_reference(settings, champion)

        # ---- 1) data drift (needs no labels) ------------------------------------------------
        drift = detect_drift(reference[FEATURES], window[FEATURES], cfg.drift)
        ref_proba = model.predict_proba(reference[FEATURES])[:, 1]
        drift.prediction_psi = psi(ref_proba, window["probability"], cfg.drift.n_bins)

        # ---- 2) live performance (needs delayed labels) -------------------------------------
        # Only score predictions made by the *current* champion, otherwise a freshly promoted
        # model would be blamed for its predecessor's mistakes.
        own = window[(window["model_version"] == str(champion.version)) & window["label"].notna()]
        perf = evaluate_live_performance(own, registry.baseline_auc(settings, champion), cfg.performance)

        # ---- 3) decision ---------------------------------------------------------------------
        if drift.dataset_drift:
            notes.append(
                f"Dataset drift: {len(drift.drifted_features)}/{len(drift.features)} features drifted "
                f"({', '.join(drift.drifted_features)})."
            )
        if perf.degraded:
            notes.append(f"Performance degradation: AUC {perf.roc_auc:.3f} vs baseline {perf.baseline_auc:.3f}.")
        if perf is not None and not perf.available:
            notes.append(f"Performance not evaluated: {perf.reason}.")

        if drift.dataset_drift or perf.degraded:
            status, action = CRITICAL, "RETRAIN"
        elif drift.drifted_features:
            status, action = WARNING, "WATCH"
            notes.append(f"Drift in: {', '.join(drift.drifted_features)} (below dataset-drift threshold).")
        else:
            status, action = OK, "NONE"

    result = MonitoringResult(as_of, str(champion.version), status, action, drift, perf, notes=notes)

    if write_report:
        path = settings.report_dir / f"drift_report_{to_iso(as_of).replace(':', '-')}.html"
        render_html_report(
            path,
            title="Model & Data Monitoring Report",
            status=status,
            action=action,
            model_version=str(champion.version),
            as_of=to_iso(as_of),
            drift=drift,
            performance=perf,
            reference=reference,
            current=window[FEATURES] if not window.empty else None,
            notes=notes,
        )
        (settings.report_dir / "latest.html").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        result.report_path = str(path)

    # ---- 4) act ------------------------------------------------------------------------------
    if action == "RETRAIN" and auto_retrain:
        trigger = "drift" if drift is not None and drift.dataset_drift else "performance"
        result.retrain = retrain(settings, store, as_of=as_of, trigger=trigger)
        notes.append(f"Retrain: {result.retrain.reason}")

    store.save_monitoring_run(
        {
            "ts": as_of,
            "model_version": champion.version,
            "status": status,
            "action": action,
            "n_current": len(window),
            "n_labeled": perf.n_labeled if perf else 0,
            "share_drifted": drift.share_drifted if drift else None,
            "dataset_drift": drift.dataset_drift if drift else None,
            "live_auc": perf.roc_auc if perf and perf.available else None,
            "baseline_auc": perf.baseline_auc if perf else None,
            "auc_drop": perf.auc_drop if perf and perf.available else None,
            "prediction_psi": drift.prediction_psi if drift else None,
            "report_path": result.report_path,
            "details": {
                "features": [
                    {"feature": f.feature, "psi": f.psi, "effect_size": f.effect_size, "p_value": f.p_value,
                     "drifted": f.drifted}
                    for f in (drift.features if drift else [])
                ],
                "notes": notes,
                "retrain": None if result.retrain is None else vars(result.retrain),
            },
        }
    )
    log.info("[%s] model v%s status=%s action=%s", to_iso(as_of), champion.version, status, action)
    return result
