"""Self-contained HTML drift report (plots are embedded as base64 PNGs - one portable file)."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mlops_pipeline.monitoring.drift import DriftReport, FeatureDrift  # noqa: E402
from mlops_pipeline.monitoring.performance import PerformanceReport  # noqa: E402

STATUS_COLORS = {"OK": "#1a7f37", "WARNING": "#b08800", "CRITICAL": "#cf222e", "INSUFFICIENT_DATA": "#6e7781"}


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _feature_plot(fd: FeatureDrift, ref: pd.Series, cur: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    color_ref, color_cur = "#8b949e", "#cf222e" if fd.drifted else "#0969da"
    if fd.kind == "numeric":
        lo, hi = np.nanpercentile(pd.concat([ref, cur]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 30)
        ax.hist(ref.dropna(), bins=bins, density=True, alpha=0.55, color=color_ref, label="reference")
        ax.hist(cur.dropna(), bins=bins, density=True, alpha=0.55, color=color_cur, label="current")
    else:
        cats = sorted(set(ref.dropna()) | set(cur.dropna()))
        x = np.arange(len(cats))
        pr = ref.value_counts(normalize=True).reindex(cats, fill_value=0)
        pc = cur.value_counts(normalize=True).reindex(cats, fill_value=0)
        ax.bar(x - 0.2, pr, 0.4, color=color_ref, label="reference")
        ax.bar(x + 0.2, pc, 0.4, color=color_cur, label="current")
        ax.set_xticks(x, [c[:10] for c in cats], rotation=25, ha="right", fontsize=7)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _fig_to_b64(fig)


def render_html_report(
    path: Path,
    *,
    title: str,
    status: str,
    action: str,
    model_version: str,
    as_of: str,
    drift: DriftReport | None,
    performance: PerformanceReport | None,
    reference: pd.DataFrame | None,
    current: pd.DataFrame | None,
    notes: list[str],
) -> Path:
    color = STATUS_COLORS.get(status, "#6e7781")
    rows, cards = [], []
    if drift is not None and reference is not None and current is not None:
        for fd in sorted(drift.features, key=lambda f: -f.psi):
            flag = "<b style='color:#cf222e'>DRIFT</b>" if fd.drifted else "<span style='color:#1a7f37'>ok</span>"
            rows.append(
                f"<tr><td>{html.escape(fd.feature)}</td><td>{fd.kind}</td><td>{flag}</td>"
                f"<td>{fd.psi:.3f}</td><td>{fd.effect_size:.3f}</td><td>{fd.p_value:.2g}</td></tr>"
            )
            img = _feature_plot(fd, reference[fd.feature], current[fd.feature])
            cards.append(
                f"<div class='card'><div class='ct'>{html.escape(fd.feature)} {flag}</div>"
                f"<img src='data:image/png;base64,{img}'/></div>"
            )

    perf_html = "<p>Not available.</p>"
    if performance is not None:
        if performance.available:
            drop = f"{performance.auc_drop:+.3f}" if performance.auc_drop is not None else "n/a"
            base = f"{performance.baseline_auc:.3f}" if performance.baseline_auc is not None else "n/a"
            perf_html = (
                f"<table><tr><th>Live ROC-AUC</th><th>Baseline</th><th>Drop</th><th>Avg precision</th>"
                f"<th>F1</th><th>Brier</th><th>Labeled rows</th></tr><tr><td>{performance.roc_auc:.3f}</td>"
                f"<td>{base}</td><td>{drop}</td><td>{performance.avg_precision:.3f}</td>"
                f"<td>{performance.f1:.3f}</td><td>{performance.brier:.3f}</td><td>{performance.n_labeled}</td></tr></table>"
            )
        else:
            perf_html = f"<p>Not evaluated: {html.escape(performance.reason)}</p>"

    summary = ""
    if drift is not None:
        summary = (
            f"<p><b>{len(drift.drifted_features)}/{len(drift.features)}</b> features drifted "
            f"(share {drift.share_drifted:.0%}) &middot; dataset drift: <b>{'YES' if drift.dataset_drift else 'no'}</b>"
            f" &middot; reference n={drift.n_reference:,} &middot; current n={drift.n_current:,}"
            + (f" &middot; prediction PSI {drift.prediction_psi:.3f}" if drift.prediction_psi is not None else "")
            + "</p>"
        )

    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:2rem auto;max-width:1100px;color:#1f2328;padding:0 1rem}}
h1{{margin-bottom:.2rem}} .badge{{display:inline-block;padding:.2rem .7rem;border-radius:1rem;color:#fff;font-weight:600;background:{color}}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}} th,td{{border:1px solid #d0d7de;padding:.35rem .6rem;text-align:left;font-size:.9rem}}
th{{background:#f6f8fa}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:.8rem}}
.card{{border:1px solid #d0d7de;border-radius:6px;padding:.5rem}} .ct{{font-weight:600;font-size:.9rem;margin-bottom:.2rem}} img{{max-width:100%}}
.muted{{color:#656d76;font-size:.9rem}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class='muted'>As of {html.escape(as_of)} &middot; model <b>v{html.escape(str(model_version))}</b></p>
<p><span class='badge'>{html.escape(status)}</span> &nbsp; recommended action: <b>{html.escape(action)}</b></p>
{''.join(f"<p class='muted'>&bull; {html.escape(n)}</p>" for n in notes)}
<h2>Data drift</h2>{summary}
<table><tr><th>Feature</th><th>Type</th><th>Status</th><th>PSI</th><th>Effect size (KS / JSD)</th><th>p-value</th></tr>{''.join(rows)}</table>
<h2>Live model performance</h2>{perf_html}
<h2>Distributions (reference vs. current)</h2><div class='grid'>{''.join(cards)}</div>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
