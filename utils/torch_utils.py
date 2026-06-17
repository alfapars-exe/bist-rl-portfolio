"""Notr torch yardimcilari - SOLID P1 (DIP).

set_seed / get_device onceden agents/common.py'deydi; forecast/forecaster.py
oradan import ediyordu (forecast -> agents ters bagimliligi). Govdeler buraya
BIREBIR tasindi; agents/common.py geriye-uyumluluk icin re-export eder.

DAVRANIS KORUNUR: fonksiyon govdesi ve cagri imzalari ayni - RNG tuketim sirasi
degismez. Golden-master regresyonu (tests/golden) bu esdegerligi dogrular.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Global RNG'leri (random, numpy, torch) seed'ler.

    Onceki ajan-ici `_set_seed` ile birebir ayni. Ajan ctor'unda, aglar
    olusturulmadan ONCE cagrilmali (agirlik init'i bu seed'e baglidir).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device: str | None = None) -> torch.device:
    """Verilen device ya da otomatik (cuda varsa cuda, yoksa cpu)."""
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))