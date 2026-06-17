"""Ortak ajan arayuzu — Faz 1 (H2: polimorfizm boslugu).

Eval/test dongusunun `if algo == 'DQN' ... elif 'PPO' ...` dallanmasini ortadan
kaldirmak icin minimal bir sozlesme: act_eval(state) -> aksiyon. Env iki tipi de
kabul eder (DiscretePortfolioEnv: idx, PortfolioEnv: vektor), dolayisiyla eval
dongusu ajan tipinden bagimsiz hale gelir.

Egitim-zamani sozlesmesi (off-policy remember/train_step vs on-policy
rollout->train) ajanlar arasinda KOKLU bicimde farklidir; bu yuzden BILEREK
buraya dahil edilmedi. O birlestirme, gercek tuketici ortaya ciktiginda
(Faz 3: core/trainer.py) onun ihtiyaclarina gore tasarlanacak. Simdilik yalnizca
tum tuketicilerin ortak kullandigi eval yuzeyini sabitliyoruz.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SupportsQValues(Protocol):
    """Q-deger introspeksiyonu sunan ajanlarin yapisal arayuzu — SOLID P5 (ISP).

    UI'nin `hasattr(agent, "q_values")` yoklamasini resmilestirir: runtime_checkable
    Protocol ile `isinstance(agent, SupportsQValues)` ayni anlami tasir ama niyet
    artik tipte gorunur. Su an yalniz DQNAgent saglar; baska bir ajan q_values
    eklerse UI paneli otomatik calisir (yeni if'e gerek yok).
    """

    def q_values(self, s: np.ndarray) -> np.ndarray: ...


class BaseAgent(ABC):
    """Tum RL ajanlarinin (DQN/PPO/SAC) uyguladigi asgari arayuz."""

    @abstractmethod
    def act_eval(self, state: np.ndarray):
        """Degerlendirme (greedy/deterministik) aksiyonunu dondur.

        Donus tipi ajana gore degisir, env her ikisini de kabul eder:
          DQN -> int           (ayrik portfoy sablonu indeksi)
          PPO -> np.ndarray    (29-boyutlu surekli logit; politikadan ornek)
          SAC -> np.ndarray    (29-boyutlu, tanh-deterministik)
        """
        raise NotImplementedError
