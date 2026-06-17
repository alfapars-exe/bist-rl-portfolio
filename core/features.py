"""Ozellik secim politikasi — SOLID P2 (DRY).

UI (app.py) ve CLI (train.py) ayni forecast-filtreleme kuralini ayri ayri
kopyaliyordu. Tek dogruluk kaynagi artik burasi.

v2 ablation bulgusu: PPO forecast feature'indan zarar gordu -> forecast yalniz
ForecastConfig.forecast_agents'taki ajanlara verilir.
"""
from __future__ import annotations

from config import ForecastConfig


def select_features(feats: dict, algo: str) -> dict:
    """forecast feature'ini yalniz ForecastConfig.forecast_agents'taki ajanlara ver.

    Diger feature'lar aynen gecer; forecast yoksa dict degismeden doner.
    (Davranis train._feats_for ile birebir ayni — o artik buna delege eder.)
    """
    if "forecast" in feats and algo not in ForecastConfig.forecast_agents:
        return {k: v for k, v in feats.items() if k != "forecast"}
    return feats
