"""Synthetic telecom-churn data with *controllable* drift.

Why synthetic? A real dataset only gives you one static snapshot. To demonstrate monitoring we
need a data source whose production behaviour we can change on purpose:

* **Covariate drift**  - the input distributions move (higher bills, more support calls,
  more month-to-month contracts). Detectable *without labels*.
* **Concept drift**    - the relationship between inputs and churn changes (customers become
  more price-sensitive, tenure stops protecting against churn). Only visible once labels arrive,
  as a drop in live model performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlops_pipeline.schema import TARGET


@dataclass(frozen=True)
class DriftSpec:
    charges_shift: float = 0.0  # additive shift of monthly charges (USD)
    support_calls_shift: float = 0.0  # additive shift of mean support calls / month
    month_to_month_boost: float = 0.0  # extra probability mass on month-to-month contracts
    age_shift: float = 0.0  # shift of customer age (years)
    concept_shift: float = 0.0  # 0..1 - strength of the label-function change

    @property
    def is_drifted(self) -> bool:
        return any(v != 0 for v in vars(self).values())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_churn_data(n: int, seed: int = 0, drift: DriftSpec | None = None) -> pd.DataFrame:
    """Generate ``n`` customers. ``drift=None`` yields the reference ("training-time") world."""
    drift = drift or DriftSpec()
    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.gamma(shape=2.0, scale=14.0, size=n), 1, 72).astype(int)
    age = np.clip(rng.normal(45 + drift.age_shift, 14, size=n), 18, 85).round().astype(int)

    m2m = min(0.92, 0.55 + drift.month_to_month_boost)
    rest = 1.0 - m2m
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"], size=n, p=[m2m, rest * 0.25 / 0.45, rest * 0.20 / 0.45]
    )
    internet = rng.choice(["dsl", "fiber", "none"], size=n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(
        ["electronic_check", "credit_card", "bank_transfer", "mailed_check"], size=n, p=[0.35, 0.25, 0.25, 0.15]
    )
    partner = rng.choice(["yes", "no"], size=n, p=[0.5, 0.5])
    support_calls = rng.poisson(max(0.05, 1.2 + drift.support_calls_shift), size=n)

    base_price = np.select([internet == "dsl", internet == "fiber"], [45.0, 80.0], default=20.0)
    monthly = np.clip(base_price + rng.normal(0, 8, size=n) + drift.charges_shift, 18, None).round(2)
    total = (monthly * tenure * rng.uniform(0.95, 1.05, size=n)).round(2)

    # ---- label-generating function ("concept") --------------------------------------------
    # Under concept drift (c > 0) the world changes in ways the deployed model cannot know about:
    #   * contract type and support calls become weaker predictors,
    #   * tenure protects less, price matters more,
    #   * an *unobserved* factor (e.g. a competitor promotion) starts driving churn.
    c = drift.concept_shift
    hidden_factor = rng.normal(0, 1, size=n)  # never exposed as a feature
    logit = (
        -1.15
        + (1.0 - 0.6 * c) * np.select([contract == "month-to-month", contract == "two-year"], [0.9, -0.9], default=0.0)
        + (1.0 - 0.5 * c) * 0.45 * support_calls
        + 0.025 * (monthly - 60.0) * (1.0 + 1.0 * c)
        - 0.03 * tenure * (1.0 - 0.7 * c)
        + 0.5 * (internet == "fiber")
        - 0.3 * (partner == "yes")
        + 0.2 * (payment == "electronic_check")
        - 0.01 * (age - 45)
        + 1.6 * c * hidden_factor
        - 0.5 * c
    )
    churn = rng.binomial(1, _sigmoid(logit))

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "age": age,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_calls": support_calls,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "has_partner": partner,
            TARGET: churn,
        }
    )
