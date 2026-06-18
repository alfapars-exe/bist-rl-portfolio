"""Twin Delayed DDPG (Fujimoto et al., 2018) — sürekli, deterministik politika.

Derste işlendi (RL_12) ve hoca sürekli-eylem problemleri için **TD3'ü açıkça
tavsiye etti**. DDPG üzerine üç hile: (1) twin critics + min-Q hedefi (overestimation
azaltma), (2) gecikmeli politika güncellemesi, (3) hedef-politika yumuşatma.
Env, ham aksiyon vektörünü softmax ile N+1 simpleksine projeler → aktörün tanh
çıktı aralığı uygundur. agents.common (mlp, ReplayBuffer, seeding) yeniden kullanılır.

Arayüz SAC ile birebir (act/act_eval/remember/train_step) → core.trainer'ın
off-policy generator'ı (train_td3) aynı adım-bazlı döngüyü kullanır.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAgent
from .common import ReplayBuffer, get_device, mlp, set_seed


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int, int] = (256, 128)):
        super().__init__()
        self.net = mlp([state_dim, hidden[0], hidden[1], action_dim], nn.ReLU)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(s))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int, int] = (256, 128)):
        super().__init__()
        self.net = mlp([state_dim + action_dim, hidden[0], hidden[1], 1], nn.ReLU)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class TD3Agent(BaseAgent):
    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int, int] = (256, 128),
                 lr_pi: float = 3e-4, lr_q: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, policy_noise: float = 0.2, noise_clip: float = 0.5,
                 policy_delay: int = 2, expl_noise: float = 0.1,
                 buffer_size: int = 50_000, batch_size: int = 128,
                 seed: int = 42, device: str | None = None):
        set_seed(seed)
        self.device = get_device(device)
        self.state_dim = state_dim; self.action_dim = action_dim
        self.gamma = float(gamma); self.tau = float(tau)
        self.policy_noise = float(policy_noise); self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay); self.expl_noise = float(expl_noise)
        self.batch_size = int(batch_size)

        self.actor = Actor(state_dim, action_dim, hidden).to(self.device)
        self.actor_t = Actor(state_dim, action_dim, hidden).to(self.device)
        self.q1 = Critic(state_dim, action_dim, hidden).to(self.device)
        self.q2 = Critic(state_dim, action_dim, hidden).to(self.device)
        self.q1_t = Critic(state_dim, action_dim, hidden).to(self.device)
        self.q2_t = Critic(state_dim, action_dim, hidden).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr_pi, weight_decay=0.0)
        self.opt_q1 = torch.optim.Adam(self.q1.parameters(), lr=lr_q, weight_decay=0.0)
        self.opt_q2 = torch.optim.Adam(self.q2.parameters(), lr=lr_q, weight_decay=0.0)
        self.buffer = ReplayBuffer(buffer_size)
        self._it = 0

    def _soft(self, src: nn.Module, dst: nn.Module):
        with torch.no_grad():
            for ps, pd in zip(src.parameters(), dst.parameters()):
                pd.data.mul_(1 - self.tau).add_(self.tau * ps.data)

    def act(self, s: np.ndarray, explore: bool = True) -> np.ndarray:
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s_t).cpu().numpy()[0]
        if explore:
            a = a + np.random.normal(0, self.expl_noise, size=a.shape)
        return a.astype(np.float32)

    def act_eval(self, s: np.ndarray) -> np.ndarray:
        """Eval: deterministik aktör (keşif gürültüsü yok)."""
        return self.act(s, explore=False)

    def remember(self, s, a, r, s2, d):
        self.buffer.push(s, np.asarray(a, dtype=np.float32), r, s2, d)

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        self._it += 1
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_t(s2) + noise).clamp(-1, 1)               # hedef-politika yumuşatma
            q_min = torch.min(self.q1_t(s2, a2), self.q2_t(s2, a2))    # twin min-Q hedefi
            target = r + (1 - d) * self.gamma * q_min

        for q, opt in [(self.q1, self.opt_q1), (self.q2, self.opt_q2)]:
            loss = F.mse_loss(q(s, a), target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0); opt.step()

        pi_loss_val = None
        if self._it % self.policy_delay == 0:                          # gecikmeli politika güncelle
            pi_loss = -self.q1(s, self.actor(s)).mean()
            self.opt_pi.zero_grad(); pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0); self.opt_pi.step()
            self._soft(self.actor, self.actor_t)
            self._soft(self.q1, self.q1_t); self._soft(self.q2, self.q2_t)
            pi_loss_val = float(pi_loss.item())
        return pi_loss_val
