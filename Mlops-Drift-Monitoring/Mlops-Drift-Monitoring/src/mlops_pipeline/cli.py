"""Command line interface:  ``mlops <command>``  (or ``python -m mlops_pipeline.cli <command>``)."""

from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from mlops_pipeline import registry
from mlops_pipeline.config import load_settings
from mlops_pipeline.data import generate_churn_data
from mlops_pipeline.monitoring import run_monitoring
from mlops_pipeline.pipeline import retrain
from mlops_pipeline.schema import TARGET
from mlops_pipeline.storage import PredictionStore
from mlops_pipeline.training import promote_if_better, train

log = logging.getLogger("mlops.cli")


def cmd_train(args, settings) -> int:
    df = pd.read_csv(args.data) if args.data else generate_churn_data(settings.data.n_train, settings.data.seed)
    if TARGET not in df.columns:
        log.error("Training data must contain the target column '%s'", TARGET)
        return 2
    result = train(settings, df, trigger="manual" if args.data else "initial")
    decision = promote_if_better(settings, result.model_version, challenger_test_auc=result.metrics["test_roc_auc"]) \
        if registry.get_champion_version(settings) is None else None
    if decision is None:  # a champion exists -> use the gate against the champion on the new test data
        from sklearn.model_selection import train_test_split

        _, holdout = train_test_split(df, test_size=settings.data.test_size, stratify=df[TARGET],
                                      random_state=settings.data.seed)
        decision = promote_if_better(settings, result.model_version, eval_df=holdout)
    print(f"model v{result.model_version} ({result.best_candidate}) test AUC={result.metrics['test_roc_auc']:.4f}"
          f" -> {'PROMOTED' if decision.promoted else 'not promoted'}: {decision.reason}")
    return 0


def cmd_monitor(args, settings) -> int:
    store = PredictionStore(settings.prediction_db)
    while True:
        try:
            res = run_monitoring(settings, store, auto_retrain=not args.no_retrain)
            print(f"[{res.as_of}] model v{res.model_version} status={res.status} action={res.action}"
                  + (f" report={res.report_path}" if res.report_path else ""))
            for n in res.notes:
                print("   -", n)
        except RuntimeError as e:  # e.g. no champion yet
            log.warning("%s", e)
        if not args.loop:
            return 0
        time.sleep(args.interval)


def cmd_retrain(args, settings) -> int:
    store = PredictionStore(settings.prediction_db)
    out = retrain(settings, store, trigger="manual", force=args.force)
    print(out)
    return 0 if out.attempted or out.reason else 1


def cmd_serve(args, settings) -> int:
    import uvicorn

    uvicorn.run("mlops_pipeline.serving.api:app_factory", factory=True, host=args.host, port=args.port,
                log_level="info")
    return 0


def cmd_demo(args, settings) -> int:
    from mlops_pipeline.simulation import run_demo

    df = run_demo(settings, weeks=args.weeks, fresh=not args.keep, plot_path=args.plot)
    cols = ["week", "status", "share_drifted", "live_auc", "serving_version_next", "retrained", "promoted"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nTimeline CSV : {settings.report_dir / 'demo_timeline.csv'}")
    print(f"Latest report: {settings.report_dir / 'latest.html'}")
    if args.plot:
        print(f"Plot         : {args.plot}")
    return 0


def cmd_status(args, settings) -> int:
    store = PredictionStore(settings.prediction_db)
    mv = registry.get_champion_version(settings)
    print("champion:", f"v{mv.version}" if mv else "none")
    print("predictions logged:", store.count_predictions())
    hist = store.monitoring_history()
    if not hist.empty:
        print(hist[["ts", "model_version", "status", "action", "share_drifted", "live_auc"]].tail(10).to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlops", description=__doc__)
    p.add_argument("--config", help="path to a YAML config (default: configs/config.yaml)")
    p.add_argument("--state-dir", help="where MLflow DB, predictions and reports live (default: ./state)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train, register and (if good enough) promote a model")
    t.add_argument("--data", help="CSV with the feature columns + 'churn' (default: synthetic data)")
    t.set_defaults(fn=cmd_train)

    m = sub.add_parser("monitor", help="run drift + performance monitoring (optionally auto-retrain)")
    m.add_argument("--loop", action="store_true", help="run forever")
    m.add_argument("--interval", type=float, default=300, help="seconds between runs with --loop")
    m.add_argument("--no-retrain", action="store_true", help="report only; never retrain automatically")
    m.set_defaults(fn=cmd_monitor)

    r = sub.add_parser("retrain", help="manually trigger retraining on recent labeled data")
    r.add_argument("--force", action="store_true", help="ignore the cooldown")
    r.set_defaults(fn=cmd_retrain)

    s = sub.add_parser("serve", help="start the inference API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    d = sub.add_parser("demo", help="simulate months of traffic + drift + auto-retraining in one go")
    d.add_argument("--weeks", type=int, default=22)
    d.add_argument("--plot", default=None, help="save the timeline chart to this path")
    d.add_argument("--keep", action="store_true", help="do not wipe the state directory first")
    d.set_defaults(fn=cmd_demo)

    st = sub.add_parser("status", help="show champion, traffic and recent monitoring runs")
    st.set_defaults(fn=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("alembic", "mlflow", "urllib3", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # demo uses a dedicated state dir so it never clobbers real data
    state_dir = args.state_dir or ("state/demo" if args.command == "demo" else None)
    settings = load_settings(args.config, state_dir)
    return args.fn(args, settings)


if __name__ == "__main__":
    sys.exit(main())
