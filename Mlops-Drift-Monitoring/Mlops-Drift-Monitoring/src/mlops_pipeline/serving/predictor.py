"""Model manager: loads the registry champion and hot-swaps it when the alias moves.

Used by both the REST API and the offline simulation, so the exact same prediction/logging
code path is exercised in tests, demos and production.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from mlops_pipeline import registry
from mlops_pipeline.config import Settings
from mlops_pipeline.schema import FEATURES
from mlops_pipeline.storage import PredictionStore, utcnow

log = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, settings: Settings, refresh_seconds: float | None = None):
        self.settings = settings
        self.refresh_seconds = settings.serving.model_refresh_seconds if refresh_seconds is None else refresh_seconds
        self._model = None
        self._version: str | None = None
        self._last_check = 0.0
        self._lock = threading.RLock()

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        """(Re)load the champion. Returns True if a model is available."""
        model, mv = registry.load_champion(self.settings)
        with self._lock:
            self._last_check = time.monotonic()
            if mv is None:
                return self._model is not None
            if str(mv.version) != self._version:
                log.info("Serving model switched: v%s -> v%s", self._version, mv.version)
            self._model, self._version = model, str(mv.version)
        return True

    def maybe_refresh(self) -> None:
        """Cheap alias check at most every ``refresh_seconds``; reload only if the version changed."""
        if time.monotonic() - self._last_check < self.refresh_seconds:
            return
        self._last_check = time.monotonic()
        mv = registry.get_champion_version(self.settings)
        if mv is not None and str(mv.version) != self._version:
            self.load()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        with self._lock:
            if self._model is None:
                raise RuntimeError("model not loaded")
            return self._model.predict_proba(df[FEATURES])[:, 1]

    def predict_and_log(
        self, df: pd.DataFrame, store: PredictionStore, ts: datetime | list[datetime] | None = None,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Score ``df``, persist every prediction (for monitoring) and return ids + probabilities."""
        with self._lock:
            proba = self.predict_proba(df)
            version = self._version
        n = len(df)
        stamps = ts if isinstance(ts, list) else [ts or utcnow()] * n
        ids = [uuid.uuid4().hex for _ in range(n)]
        records = [
            {
                "prediction_id": ids[i],
                "ts": stamps[i],
                "model_version": version,
                "probability": float(proba[i]),
                "prediction": int(proba[i] >= threshold),
                "features": {k: (v.item() if hasattr(v, "item") else v) for k, v in df.iloc[i][FEATURES].items()},
            }
            for i in range(n)
        ]
        store.log_predictions(records)
        return pd.DataFrame({"prediction_id": ids, "probability": proba, "model_version": version})
