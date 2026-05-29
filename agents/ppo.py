"""Proximal Policy Optimization (Schulman 2017) — PyTorch, sürekli aksiyon.

Policy ağının çıkardığı Gaussian logits (mu, log_std) ortam tarafından
softmax üzerinden portföy ağırlıklarına dönüştürülür. Değer ağı V(s) öğrenir.
Avantaj GAE(λ) ile hesaplanır, kırpılmış surrogate + value MSE + entropi
bonusu ile eğitilir.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import get_device, mlp, set_seed


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Tuple[int, int] = (256, 128),
                 log_std_init: float = -0.5):
        super().__init__()
        self.trunk = mlp([state_dim, hidden[0], hidden[1]], nn.Tanh, out_activation=nn.Tanh)
        self.mu_head = nn.Linear(hidden[1], action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        return mu, std


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden: Tuple[int, int] = (256, 128)):
        super().__init__()
        self.net = mlp([state_dim, hidden[0], hidden[1], 1], nn.Tanh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 hidden: Tuple[int, int] = (256, 128),
                 lr_p: float = 3e-4, lr_v: float = 1e-3,
                 gamma: float = 0.99, lam: float = 0.95, clip: float = 0.2,
                 ent_coef: float = 0.005, n_epochs: int = 8,
                 batch_size: int = 128, seed: int = 42,
                 log_std_init: float = -0.5, device: str | None = None):
        set_seed(seed)
        self.device = get_device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = float(gamma); self.lam = float(lam); self.clip = float(clip)
        self.ent_coef = float(ent_coef); self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)

        self.policy = PolicyNet(state_dim, action_dim, hidden, log_std_init).to(self.device)
        self.value  = ValueNet(state_dim, hidden).to(self.device)

        self.opt_p = torch.optim.Adam(self.policy.parameters(), lr=lr_p)
        self.opt_v = torch.optim.Adam(self.value.parameters(),  lr=lr_v)

        self.reset_rollout()

    def reset_rollout(self):
        self.S, self.A, self.R, self.D, self.Vs, self.LP = [], [], [], [], [], []

    @torch.no_grad()
    def _policy_dist(self, s_t: torch.Tensor):
        mu, std = self.policy(s_t)
        return torch.distributions.Normal(mu, std)

    def act(self, s: np.ndarray):
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.policy(s_t)
            dist = torch.distributions.Normal(mu, std)
            a = dist.sample()
            logp = dist.log_prob(a).sum(dim=-1)
            v = self.value(s_t)
        return (a.cpu().numpy()[0].astype(np.float32),
                float(logp.item()),
                float(v.item()))

    def remember(self, s, a, r, done, v, logp):
        self.S.append(np.asarray(s, dtype=np.float32))
        self.A.append(np.asarray(a, dtype=np.float32))
        self.R.append(float(r)); self.D.append(float(done))
        self.Vs.append(float(v)); self.LP.append(float(logp))

    def compute_gae(self, last_v: float):
        n = len(self.R)
        adv = np.zeros(n, dtype=np.float32)
        g = 0.0
        for i in reversed(range(n)):
            next_v = last_v if i == n - 1 else self.Vs[i + 1]
            mask = 1.0 - self.D[i]
            delta = self.R[i] + self.gamma * next_v * mask - self.Vs[i]
            g = delta + self.gamma * self.lam * mask * g
            adv[i] = g
        ret = adv + np.array(self.Vs, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def train(self, last_v: float) -> dict:
        adv_np, ret_np = self.compute_gae(last_v)
        S  = torch.as_tensor(np.stack(self.S), dtype=torch.float32, device=self.device)
        A  = torch.as_tensor(np.stack(self.A), dtype=torch.float32, device=self.device)
        LP_old = torch.as_tensor(np.asarray(self.LP, dtype=np.float32), device=self.device)
        ADV = torch.as_tensor(adv_np, device=self.device)
        RET = torch.as_tensor(ret_np, device=self.device)
        N = S.shape[0]

        p_losses, v_losses, ents, kls = [], [], [], []
        for _ in range(self.n_epochs):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, self.batch_size):
                b = idx[start: start + self.batch_size]
                if b.numel() < 8:
                    continue
                s_b, a_b = S[b], A[b]
                lp_old_b = LP_old[b]; adv_b = ADV[b]; ret_b = RET[b]

                mu, std = self.policy(s_b)
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(a_b).sum(dim=-1)
                ent  = dist.entropy().sum(dim=-1).mean()

                ratio = (logp - lp_old_b).clamp(-20.0, 20.0).exp()
                unclipped = ratio * adv_b
                clipped   = ratio.clamp(1 - self.clip, 1 + self.clip) * adv_b
                p_loss = -torch.min(unclipped, clipped).mean()
                total_p = p_loss - self.ent_coef * ent

                self.opt_p.zero_grad()
                total_p.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                self.opt_p.step()

                v_pred = self.value(s_b)
                v_loss = F.mse_loss(v_pred, ret_b)
                self.opt_v.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 5.0)
                self.opt_v.step()

                p_losses.append(float(p_loss.item()))
                v_losses.append(float(v_loss.item()))
                ents.append(float(ent.item()))
                kls.append(float((lp_old_b - logp).mean().item()))

        self.reset_rollout()
        return dict(
            p_loss=float(np.mean(p_losses)) if p_losses else 0.0,
            v_loss=float(np.mean(v_losses)) if v_losses else 0.0,
            ent=float(np.mean(ents)) if ents else 0.0,
            kl=float(np.mean(kls)) if kls else 0.0,
        )
