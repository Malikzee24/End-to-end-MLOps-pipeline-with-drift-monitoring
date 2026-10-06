#!/usr/bin/env python
"""Send realistic production traffic (with injected drift) to a running inference API.

Phases (all durations in seconds, measured from start):
  0 .. --drift-after                 stable traffic
  --drift-after .. +--ramp           drift ramps up linearly (covariate, then concept)
  afterwards                          plateau at full drift

Ground-truth labels are posted to /feedback after --label-delay seconds, mimicking delayed outcomes.

    python scripts/traffic_generator.py --api-url http://localhost:8000 --drift-after 120 --ramp 240
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import httpx

from mlops_pipeline.data import DriftSpec, generate_churn_data
from mlops_pipeline.schema import FEATURES, TARGET


def spec_at(elapsed: float, drift_after: float, ramp: float) -> DriftSpec:
    x = min(max((elapsed - drift_after) / ramp, 0.0), 1.0) if ramp > 0 else float(elapsed >= drift_after)
    concept = min(max((x - 0.5) / 0.5, 0.0), 1.0)  # concept drift kicks in during the second half of the ramp
    return DriftSpec(charges_shift=18 * x, support_calls_shift=0.9 * x, month_to_month_boost=0.22 * x,
                     concept_shift=concept)


def wait_for_api(client: httpx.Client, url: str, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if client.get(f"{url}/health").json().get("status") == "ok":
                return
        except Exception:
            pass
        print("waiting for API / champion model ...")
        time.sleep(3)
    raise SystemExit("API did not become ready")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between batches")
    ap.add_argument("--drift-after", type=float, default=180.0)
    ap.add_argument("--ramp", type=float, default=300.0)
    ap.add_argument("--label-delay", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = run forever)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    pending: deque[tuple[float, list[dict]]] = deque()
    start, i = time.time(), 0
    with httpx.Client(timeout=30) as client:
        wait_for_api(client, args.api_url)
        while args.duration == 0 or time.time() - start < args.duration:
            elapsed = time.time() - start
            spec = spec_at(elapsed, args.drift_after, args.ramp)
            df = generate_churn_data(args.batch_size, seed=args.seed + i, drift=spec)
            r = client.post(f"{args.api_url}/predict/batch", json={"customers": df[FEATURES].to_dict("records")})
            r.raise_for_status()
            preds = r.json()
            pending.append((time.time() + args.label_delay,
                            [{"prediction_id": p["prediction_id"], "actual_churn": bool(y)}
                             for p, y in zip(preds, df[TARGET])]))
            while pending and pending[0][0] <= time.time():
                client.post(f"{args.api_url}/feedback", json={"items": pending.popleft()[1]}).raise_for_status()
            print(f"[{elapsed:6.0f}s] batch {i:04d} model v{preds[0]['model_version']} "
                  f"drift(cov={spec.charges_shift / 18:.2f}, concept={spec.concept_shift:.2f})")
            i += 1
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
