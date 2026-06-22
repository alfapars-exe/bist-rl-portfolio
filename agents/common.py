"""Ajanlar arasi paylasilan yardimcilar — Faz 1 (M3: kod tekrari giderme).

set_seed / get_device / mlp / ReplayBuffer uc ajanda da kopyalanmis kodu tek
yere toplar.

DAVRANIS KORUNUR: RNG tuketen islemlerin sirasi ve cagri imzalari onceki inline
kodla birebir aynidir —
  * mlp() katmanlari `sizes` sirasiyla olusturur -> nn.Linear agirlik init'i
    (global torch RNG) ayni sirada tuketilir -> ayni baslangic agirliklari.
  * ReplayBuffer.sample() ayni `np.random.randint(0, n, batch)` cagrisini yapar.
Golden-master regresyonu (tests/golden) bu esdegerligi dogrular.
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch.nn as nn

# SOLID P1 (DIP): set_seed/get_device notr utils/torch_utils.py'ye tasindi.
# Eski import yollari (agents.common.set_seed vb.) calismaya devam eder.
from utils.torch_utils import get_device, set_seed  # noqa: F401


def mlp(sizes: Sequence[int], activation: type[nn.Module],
        out_activation: type[nn.Module] | None = None) -> nn.Sequential:
    """sizes=[in, h1, ..., out] icin Linear+aktivasyon yigini kurar.

    Gizli katmanlar `activation`, son katman `out_activation` alir
    (None ise aktivasyon yok). Katmanlar `sizes` sirasiyla olusturulur, boylece
    nn.Linear agirlik init sirasi onceki inline nn.Sequential ile ayni kalir.

    Ornekler (onceki kodla ayni yapilar):
      QNetwork  : mlp([s, 256, 128, n], nn.ReLU)                  # son katman aktivasyonsuz
      SAC trunk : mlp([s, 256, 128], nn.ReLU, out_activation=nn.ReLU)   # govde, sonu aktivasyonlu
      PPO trunk : mlp([s, 256, 128], nn.Tanh, out_activation=nn.Tanh)
      ValueNet  : mlp([s, 256, 128, 1], nn.Tanh)
    """
    layers: list[nn.Module] = []
    n = len(sizes)
    for i in range(n - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        act = out_activation if i == n - 2 else activation
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class ReplayBuffer:
    """Uniform ornekleme deque buffer (DQN + SAC ortak).

    Tuple semasi: (s: float32[ ], a, r: float, s2: float32[ ], d: float).
    `a` cagiran tarafindan tiplenir (DQN: int, SAC: float32 vektor); push()
    s/s2/r/d'yi onceki `remember` ile ayni sekilde normalize eder.
    """

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, s, a, r, s2, d, discount=1.0) -> None:
        self.buffer.append((
            np.asarray(s, dtype=np.float32),
            a,
            float(r),
            np.asarray(s2, dtype=np.float32),
            float(d),
            float(discount),
        ))

    def sample(self, batch_size: int):
        """(s, a, r, s2, d) numpy yiginlari dondurur.

        np.array([...]) hem skaler (DQN) hem esit-boyutlu vektor (SAC) aksiyon
        listelerinde dogru yiginlamayi verir; onceki inline np.array/np.stack
        karisimiyla ayni degerleri uretir. Tensor dtype'i cagiran ayarlar.
        """
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        s  = np.stack([b[0] for b in batch])
        a  = np.array([b[1] for b in batch])
        r  = np.array([b[2] for b in batch])
        s2 = np.stack([b[3] for b in batch])
        d  = np.array([b[4] for b in batch])
        discount = np.array([b[5] for b in batch])
        return s, a, r, s2, d, discount
