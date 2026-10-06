"""Single source of truth for the feature schema used across training, serving and monitoring."""

NUMERIC_FEATURES: list[str] = [
    "tenure_months",
    "age",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
]

CATEGORICAL_FEATURES: list[str] = [
    "contract_type",
    "internet_service",
    "payment_method",
    "has_partner",
]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET: str = "churn"

CATEGORY_VALUES: dict[str, list[str]] = {
    "contract_type": ["month-to-month", "one-year", "two-year"],
    "internet_service": ["dsl", "fiber", "none"],
    "payment_method": ["electronic_check", "credit_card", "bank_transfer", "mailed_check"],
    "has_partner": ["yes", "no"],
}
