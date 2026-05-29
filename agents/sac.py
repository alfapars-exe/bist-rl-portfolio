"""Soft Actor-Critic — PyTorch, sürekli aksiyon, tanh-sıkıştırılmış Gaussian.

Twin critics Q1, Q2 + soft target güncellemesi (τ). Sabit entropi katsayısı α
(otomatik ayarlama kapalı — basit tutmak için). Policy loss: min-of-two critic
+ entropi bonusu. Replay buffer uniform örnekleme.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAgent
from .common import ReplayBuffer, get_device, mlp, set_seed


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Tuple[int, int] = (256, 128),
                 log_std_clip: Tuple[float, float] = (-5.0, 2.0)):
        super().__init__()
        self.trunk = mlp([state_dim, hidden[0], hidden[1]], nn.ReLU, out_activation=nn.ReLU)
        self.mu_head = nn.Linear(hidden[1], action_dim)
        self.log_std_head = nn.Linear(hidden[1], action_dim)
        self.log_std_clip = log_std_clip

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(*self.log_std_clip)
        return mu, log_std

    def sample(self, s: torch.Tensor, deterministic: bool = False):
        mu, log_std = self.forward(s)
        std = log_std.exp()
        if deterministic:
            u = mu
            logp = torch.zeros(s.shape[0], device=s.device)
        else:
            normal = torch.distributions.Normal(mu, std)
            u = normal.rsample()
            logp = normal.log_prob(u).sum(dim=-1)
        a = torch.tanh(u)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
        return a, logp


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Tuple[int, int] = (256, 128)):
        super().__init__()
        self.net = mlp([state_dim + action_dim, hidden[0], hidden[1], 1], nn.ReLU)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class SACAgent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Tuple[int, int] = (256, 128),
                 lr_pi: float = 3e-4, lr_q: float = 5e-4,
                 gamma: float = 0.99, tau: float = 0.01, alpha: float = 0.05,
                 buffer_size: int = 50_000, batch_size: int = 128,
                 seed: int = 42, device: str | None = None):
        set_seed(seed)
        self.device = get_device(device)
        self.state_dim = state_dim; self.action_dim = action_dim
        self.gamma = float(gamma); self.tau = float(tau); self.alpha = float(alpha)
        self.batch_size = int(batch_size)

        self.pi   = GaussianPolicy(state_dim, action_dim, hidden).to(self.device)
        self.q1   = QNet(state_dim, action_dim, hidden).to(self.device)
        self.q2   = QNet(state_dim, action_dim, hidden).to(self.device)
        self.q1_t = QNet(state_dim, action_dim, hidden).to(self.device)
        self.q2_t = QNet(state_dim, action_dim, hidden).to(self.device)
        self._sync()

        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr_pi)
        self.opt_q1 = torch.optim.Adam(self.q1.parameters(), lr=lr_q)
        self.opt_q2 = torch.optim.Adam(self.q2.parameters(), lr=lr_q)

        self.buffer = ReplayBuffer(buffer_size)

    def _sync(self):
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

    def _soft_update(self):
        with torch.no_grad():
            for src, dst in [(self.q1, self.q1_t), (self.q2, self.q2_t)]:
                for p_s, p_d in zip(src.parameters(), dst.parameters()):
                    p_d.data.mul_(1 - self.tau).add_(self.tau * p_s.data)

    def act(self, s: np.ndarray, deterministic: bool = False) -> np.ndarray:
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.pi.sample(s_t, deterministic=deterministic)
        return a.cpu().numpy()[0].astype(np.float32)

    def act_eval(self, s: np.ndarray) -> np.ndarray:
        """Eval: tanh-deterministik aksiyon."""
        return self.act(s, deterministic=True)

    def remember(self, s, a, r, s2, d):
        self.buffer.push(s, np.asarray(a, dtype=np.float32), r, s2, d)

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(a,  dtype=torch.float32, device=self.device)
        r  = torch.as_tensor(r,  dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(d,  dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a2, logp2 = self.pi.sample(s2)
            q_min = torch.min(self.q1_t(s2, a2), self.q2_t(s2, a2))
            target = r + (1.0 - d) * self.gamma * (q_min - self.alpha * logp2)

        for q, opt in [(self.q1, self.opt_q1), (self.q2, self.opt_q2)]:
            q_pred = q(s, a)
            q_loss = F.mse_loss(q_pred, target)
            opt.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
            opt.step()

        a_new, logp_new = self.pi.sample(s)
        q_new = torch.min(self.q1(s, a_new), self.q2(s, a_new))
        pi_loss = (self.alpha * logp_new - q_new).mean()
        self.opt_pi.zero_grad()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 5.0)
        self.opt_pi.step()

        self._soft_update()
        return float(pi_loss.item())
