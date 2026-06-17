"""Deep Q-Network ajanı — PyTorch implementasyonu (ayrık aksiyonlu).

Prompt spec'i:
  - MLP: state_dim (393, v2) → FC(256, ReLU) → FC(128, ReLU) → 6 (Q-values)
  - Replay buffer: 50_000, uniform örnekleme, batch = 64
  - Target network: her 500 adımda hard update (θ⁻ ← θ)
  - ε-greedy: 1.0 → 0.05, 10_000 adımda lineer decay
  - Kayıp: Huber (δ=1.0)
  - Optimizer: Adam, lr=1e-3
  - γ = 0.99
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent
from .common import ReplayBuffer, get_device, mlp, set_seed


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int,
                 hidden: Tuple[int, int] = (256, 128)):
        super().__init__()
        self.net = mlp([state_dim, hidden[0], hidden[1], n_actions], nn.ReLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(self, state_dim: int, n_actions: int,
                 hidden: Tuple[int, int] = (256, 128),
                 lr: float = 1e-3, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: int = 10_000,
                 buffer_size: int = 50_000, batch_size: int = 64,
                 target_update: int = 500, huber_delta: float = 1.0,
                 seed: int = 42, device: str | None = None):
        set_seed(seed)
        self.device = get_device(device)
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = int(eps_decay)
        self.target_update = int(target_update)

        self.q        = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.q_target = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self._sync_target()

        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.HuberLoss(delta=huber_delta)

        self.buffer = ReplayBuffer(buffer_size)
        self.step_count = 0

    def _sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def eps(self) -> float:
        frac = min(1.0, self.step_count / max(self.eps_decay, 1))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def q_values(self, s: np.ndarray) -> np.ndarray:
        """UI/analiz için: 6 aksiyonun Q-değerlerini döner (numpy)."""
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.q(s_t).cpu().numpy()[0]

    def act(self, s: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and np.random.rand() < self.eps():
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_values(s)))

    def act_eval(self, s: np.ndarray) -> int:
        """Eval: greedy secim (epsilon yok) — ayrik sablon indeksi."""
        return self.act(s, greedy=True)

    def remember(self, s, a, r, s2, d):
        self.buffer.push(s, int(a), r, s2, d)

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(a,  dtype=torch.int64,   device=self.device)
        r  = torch.as_tensor(r,  dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(d,  dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_next = self.q_target(s2).max(dim=1).values
            td_target = r + (1.0 - d) * self.gamma * q_next

        q_pred_all = self.q(s)
        q_pred = q_pred_all.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_pred, td_target)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.opt.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self._sync_target()
        return float(loss.item())

    def save(self, path: str):
        torch.save({"q": self.q.state_dict(), "step_count": self.step_count}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"])
        self._sync_target()
        self.step_count = int(ckpt.get("step_count", 0))
