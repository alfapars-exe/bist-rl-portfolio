"""Serializable contracts shared by training, evaluation, UI and persistence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Literal

import numpy as np


DataSource = Literal["real", "mixed", "synthetic", "unknown"]


@dataclass(frozen=True)
class DataProvenance:
    source: DataSource = "unknown"
    provider: str = ""
    reason: str = ""
    missing_tickers: tuple[str, ...] = ()
    requested_start: str = ""
    requested_end: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "DataProvenance":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            data = dict(value)
            data["missing_tickers"] = tuple(data.get("missing_tickers", ()))
            return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})
        return cls()


@dataclass(frozen=True)
class RunSpec:
    algo: str
    step_days: int = 1
    adaptive: bool = True
    reward_cfg: dict[str, Any] = field(default_factory=dict)
    agent_hp: dict[str, Any] = field(default_factory=dict)
    data_start: str = ""
    data_split: str = ""
    data_end: str = ""
    seed: int = 42
    feature_names: tuple[str, ...] = ()
    provenance: DataProvenance = field(default_factory=DataProvenance)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["feature_names"] = list(self.feature_names)
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunSpec":
        data = dict(value)
        data["feature_names"] = tuple(data.get("feature_names", ()))
        data["provenance"] = DataProvenance.from_value(data.get("provenance"))
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class BacktestResult:
    nav: np.ndarray
    rets: np.ndarray
    dates: list[Any]
    weights_before: np.ndarray
    target_weights: np.ndarray
    weights_after: np.ndarray
    turnover: np.ndarray
    reward_terms_history: list[dict]
    period_lengths: np.ndarray
    discounts: np.ndarray
    provenance: DataProvenance = field(default_factory=DataProvenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nav": self.nav,
            "rets": self.rets,
            "dates": self.dates,
            "weights": self.weights_after,
            "weights_before": self.weights_before,
            "target_weights": self.target_weights,
            "weights_after": self.weights_after,
            "turnover": self.turnover,
            "reward_terms_history": self.reward_terms_history,
            "period_lengths": self.period_lengths,
            "discounts": self.discounts,
            "provenance": self.provenance.to_dict(),
        }
