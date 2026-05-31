"""Portföy yönetimi MDP ortamı — vade bazlı preset'ler + adaptif ödül şekillendirici.

MDP tuple (S, A, P, r, γ):

  S : ℝ^393 — 28 hisse × 13 özellik (12 teknik + 1 forecast, z-skorlu) + 29 boyutlu önceki ağırlık
  A : 6 ayrık şablon (DiscretePortfolioEnv) VEYA ℝ^29 sürekli softmax (PortfolioEnv)
  P : Piyasa tarafından belirlenen stokastik süreç; s_{t+1} sonraki günün
      öznitelikleri + işlem sonrası ağırlıklardan oluşur
  r : r_t = log(1 + w_t · r_{t+1}) − η_t · ‖Δw‖₁ − λ_t · max(0, DD_t − τ_t)
      (η, λ, τ adaptif — AdaptiveRewardShaper tarafından anlık ölçeklenir)
  γ : Vadeye göre 0.95 (kısa) / 0.99 (orta) / 0.995 (uzun)

Sonlandırma koşulları:
  (i)  Veri sonu          (t >= T-1)
  (ii) İflas koruması      (NAV < bankruptcy_nav, varsayılan 0.01)
  (iii) 252 adım tavanı    (1 iş yılı; trunc=True)
"""
from __future__ import annotations

from typing import Dict, Literal, Tuple
import numpy as np
import pandas as pd

from config import HORIZON_PRESETS, RewardConfig


# ---------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------
ACTION_NAMES = [
    "Nakit",
    "Eşit Ağırlık",
    "Top-3 Momentum",
    "Top-5 Momentum",
    "Ters Volatilite",
    "Min Volatilite",
]

# HORIZON_PRESETS config.py'ye tasindi (Faz 2, H3) — dosya basinda import edildi.

MAX_EPISODE_STEPS = 252
# NAV bu eşiğin altına düşerse episod iflas olarak sonlandırılır.
# 0.01 = başlangıç sermayesinin %1'ine inmek (pratikte "para bitti").
BANKRUPTCY_NAV    = 0.01
# İflas gerçekleştiğinde ajan'a uygulanan ek ödül cezası (log-ölçeğinde çok büyük).
BANKRUPTCY_PENALTY = 10.0


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = x / max(temp, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------
# Adaptif Ödül Şekillendirici
# ---------------------------------------------------------------------
class AdaptiveRewardShaper:
    """Rolling realized vol + rolling turnover EWMA'larına göre (η, λ, τ) ölçekler.

    - turnover_ewma büyürse → η_t büyür (fazla işlem yapan ajana artan ceza).
    - vol_ewma büyürse       → λ_t büyür & τ_t daralır (volatil rejimde sert DD cezası).

    Toggle OFF ise `update_and_shape` base değerleri döndürür (ölçekleme yok).
    """

    def __init__(self, eta_base: float, lambda_base: float, tau_base: float,
                 vol_target: float = 0.02, turnover_target: float = 0.05,
                 ema_alpha: float = 0.05, enabled: bool = True):
        self.eta_base = float(eta_base)
        self.lambda_base = float(lambda_base)
        self.tau_base = float(tau_base)
        self.vol_target = float(vol_target)
        self.turnover_target = float(turnover_target)
        self.alpha = float(ema_alpha)
        self.enabled = bool(enabled)
        self.vol_ewma = float(vol_target)
        self.turnover_ewma = float(turnover_target)

    def reset(self):
        self.vol_ewma = self.vol_target
        self.turnover_ewma = self.turnover_target

    def update_and_shape(self, port_r: float, delta_w_l1: float) -> Tuple[float, float, float, float, float]:
        a = self.alpha
        self.vol_ewma      = (1 - a) * self.vol_ewma      + a * abs(float(port_r))
        self.turnover_ewma = (1 - a) * self.turnover_ewma + a * float(delta_w_l1)

        if not self.enabled:
            return (self.eta_base, self.lambda_base, self.tau_base,
                    self.vol_ewma, self.turnover_ewma)

        vol_ratio      = self.vol_ewma / max(self.vol_target, 1e-9)
        turnover_ratio = self.turnover_ewma / max(self.turnover_target, 1e-9)

        eta_t    = self.eta_base    * max(1.0, turnover_ratio)
        lambda_t = self.lambda_base * (1.0 + max(0.0, vol_ratio - 1.0))
        tau_t    = self.tau_base    * max(0.7, 1.0 / max(vol_ratio, 1e-9))
        return eta_t, lambda_t, tau_t, self.vol_ewma, self.turnover_ewma


# ---------------------------------------------------------------------
# Diferansiyel Sharpe (Moody & Saffell) — çevrim-içi risk-ayarlı ödül terimi
# ---------------------------------------------------------------------
class DifferentialSharpe:
    """Her adımda Sharpe oranındaki marjinal değişimi (DSR) döndürür.

    EWMA tahminleri A (ortalama getiri), B (ortalama kare getiri) ile:
      ΔA = R − A,  ΔB = R² − B
      DSR = (B·ΔA − ½·A·ΔB) / (B − A²)^{3/2}
    Sonra A,B η ile güncellenir. Sabit-pencere Sharpe'ın türevlenebilir, adım-bazlı
    (online) hâli — RL-in-finance'te risk-ayarlı ödül için standart.
    """

    def __init__(self, eta: float = 0.01, clip: float = 5.0):
        self.eta = float(eta)
        self.clip = float(clip)
        self.reset()

    def reset(self):
        self.A = 0.0
        self.B = 0.0
        self.initialized = False

    def update(self, r: float) -> float:
        r = float(r)
        if not self.initialized:
            self.A, self.B, self.initialized = r, r * r, True
            return 0.0
        dA = r - self.A
        dB = r * r - self.B
        denom = self.B - self.A * self.A
        dsr = 0.0 if denom <= 1e-12 else (self.B * dA - 0.5 * self.A * dB) / (denom ** 1.5)
        dsr = float(np.clip(dsr, -self.clip, self.clip))
        self.A += self.eta * dA
        self.B += self.eta * dB
        return dsr


# ---------------------------------------------------------------------
# Sürekli sürüm (PPO / SAC için)
# ---------------------------------------------------------------------
class PortfolioEnv:
    """Sürekli aksiyon portföy ortamı — softmax ile N+1 simplex ağırlıkları."""

    def __init__(self, prices: pd.DataFrame, features: Dict[str, pd.DataFrame],
                 window: int = 20, cash_asset: bool = True,
                 horizon: Literal["short", "medium", "long"] = "medium",
                 adaptive: bool = True,
                 eta_base: float | None = None,
                 lambda_base: float | None = None,
                 tau_base: float | None = None,
                 vol_target: float = 0.02, turnover_target: float = 0.05,
                 ema_alpha: float = 0.05,
                 max_steps: int = MAX_EPISODE_STEPS,
                 bankruptcy_nav: float | None = None,
                 bankruptcy_penalty: float | None = None,
                 random_start: bool = False,
                 seed: int | None = None,
                 w_dsr: float = RewardConfig.w_dsr,
                 dsr_eta: float = RewardConfig.dsr_eta):
        self.prices = prices.values.astype(np.float32)
        self.dates  = prices.index
        self.feat_names = list(features.keys())
        self.feat_tensor = np.stack(
            [features[k].values.astype(np.float32) for k in self.feat_names],
            axis=-1,
        )  # (T, N, F)

        self.horizon = horizon
        preset = HORIZON_PRESETS[horizon]
        self.rebalance_freq = preset["rebalance"]
        self.mom_window     = preset["mom_window"]
        self.minvol_window  = preset["minvol_window"]
        self.gamma          = preset["gamma"]
        eta_base    = preset["eta"] if eta_base    is None else float(eta_base)
        lambda_base = preset["lam"] if lambda_base is None else float(lambda_base)
        tau_base    = preset["tau"] if tau_base    is None else float(tau_base)

        self.shaper = AdaptiveRewardShaper(
            eta_base=eta_base, lambda_base=lambda_base, tau_base=tau_base,
            vol_target=vol_target, turnover_target=turnover_target,
            ema_alpha=ema_alpha, enabled=adaptive,
        )
        # v2: çevrim-içi risk-ayarlı ödül (Diferansiyel Sharpe)
        self.w_dsr = float(w_dsr)
        self.dsharpe = DifferentialSharpe(eta=dsr_eta)
        # Ayarlanabilir iflas parametreleri — None ise modül-düzeyi default'lar
        self.bankruptcy_nav = float(bankruptcy_nav) if bankruptcy_nav is not None else BANKRUPTCY_NAV
        self.bankruptcy_penalty = (float(bankruptcy_penalty)
                                   if bankruptcy_penalty is not None else BANKRUPTCY_PENALTY)

        self.N_assets = prices.shape[1]
        self.cash_asset = cash_asset
        self.N = self.N_assets + (1 if cash_asset else 0)
        self.window = max(window, self.minvol_window)
        self.F = self.feat_tensor.shape[2]
        self.state_dim = self.F * self.N_assets + self.N
        self.action_dim = self.N
        self.T = prices.shape[0]
        self.max_steps = max_steps
        # v2: env-yerel RNG — global np.random'a bagimli degil (tekrar-uretilebilirlik
        # kurulum sirasindan bagimsiz) + tohumlu rastgele-baslangic destegi.
        self.random_start = bool(random_start)
        self.rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self):
        lo = max(self.window, 21)
        if self.random_start:
            # episode'un max_steps adim + bir sonraki gun erisimi icin yer birak
            hi = self.T - self.max_steps - 1
            self.t = int(self.rng.integers(lo, hi)) if hi > lo else lo
        else:
            self.t = lo
        self.step_count = 0
        self.w = np.zeros(self.N, dtype=np.float32)
        self.w[-1] = 1.0  # nakitle başla
        self.nav = 1.0
        self.peak = 1.0
        self.nav_history = [1.0]
        self.weight_history = [self.w.copy()]
        self.ret_history = []
        self.reward_terms_history = []
        self.shaper.reset()
        self.dsharpe.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)   # env-yerel; global RNG'ye dokunmaz
        self._reset_state()
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        snap = self.feat_tensor[self.t].reshape(-1)
        return np.concatenate([snap, self.w]).astype(np.float32)

    def _risky_returns(self) -> np.ndarray:
        p0 = self.prices[self.t]
        p1 = self.prices[self.t + 1]
        r = (p1 - p0) / np.maximum(p0, 1e-9)
        if self.cash_asset:
            r = np.concatenate([r, [0.0]])
        return r

    def _should_rebalance(self) -> bool:
        return (self.step_count % self.rebalance_freq) == 0

    def _apply_action(self, action) -> np.ndarray:
        if self._should_rebalance():
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            if a.shape[0] != self.N:
                raise ValueError(f"action must have length {self.N}, got {a.shape}")
            return softmax(a, temp=1.0)
        return self.w.copy()

    def step(self, action):
        w_new = self._apply_action(action)
        delta_w_l1 = float(np.abs(w_new - self.w).sum())
        r_vec = self._risky_returns()
        gross_port_r = float((w_new * r_vec).sum())

        eta_t, lambda_t, tau_t, vol_ewma, to_ewma = \
            self.shaper.update_and_shape(gross_port_r, delta_w_l1)

        tx_cost = eta_t * delta_w_l1
        port_r_net = gross_port_r - tx_cost
        self.nav *= (1.0 + port_r_net)
        # Matematiksel olarak nav <= 0 olmamalı (çarpım süreci), ama extreme tx/return
        # kombinasyonunda ölçülebilir şekilde sıfıra yakın ya da negatif olabilir.
        # İflas eşiğinin altına düşerse clamp et ve episodu sonlandırıp ceza uygula.
        self.nav = max(self.nav, 0.0)
        bankrupt = bool(self.nav < self.bankruptcy_nav)
        bankruptcy_penalty = self.bankruptcy_penalty if bankrupt else 0.0

        self.peak = max(self.peak, self.nav)
        dd = (self.peak - self.nav) / max(self.peak, 1e-9)
        log_r = float(np.log(max(1.0 + gross_port_r, 1e-6)))
        dd_penalty = lambda_t * max(0.0, dd - tau_t)
        dsr = self.dsharpe.update(port_r_net)
        dsr_term = self.w_dsr * dsr
        total = log_r - tx_cost - dd_penalty - bankruptcy_penalty + dsr_term

        self.w = w_new
        self.t += 1
        self.step_count += 1
        self.nav_history.append(self.nav)
        self.weight_history.append(self.w.copy())
        self.ret_history.append(port_r_net)

        reward_terms = dict(
            log_return=log_r, tx_cost=tx_cost,
            drawdown_penalty=dd_penalty, total=total,
            dsr=dsr, dsr_term=dsr_term,
            eta_t=eta_t, lambda_t=lambda_t, tau_t=tau_t,
            vol_ewma=vol_ewma, turnover_ewma=to_ewma,
            dd=dd, gross_port_r=gross_port_r, delta_w_l1=delta_w_l1,
            bankruptcy_penalty=bankruptcy_penalty, bankrupt=bankrupt,
        )
        self.reward_terms_history.append(reward_terms)

        # İflas veya veri sonu → done; 252 adım tavanı → trunc
        done = (self.t >= (self.T - 1)) or bankrupt
        trunc = (self.step_count >= self.max_steps)
        info = dict(nav=self.nav, dd=dd, port_r=port_r_net, reward_terms=reward_terms,
                    prices_t=self.prices[self.t].copy())
        return self._obs(), float(total), bool(done), bool(trunc), info


# ---------------------------------------------------------------------
# Ayrık sürüm (DQN için) — 6 aksiyon şablonu
# ---------------------------------------------------------------------
class DiscretePortfolioEnv(PortfolioEnv):
    """DQN sarmalayıcı — 6 ayrık portföy şablonu (momentum/vol pencereleri vadeye göre)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_discrete = 6
        self.action_dim = self.n_discrete

    def _discrete_to_logits(self, a_idx: int) -> np.ndarray:
        mw = self.mom_window
        mvw = self.minvol_window
        lb_mom  = self.feat_tensor[self.t - mw:  self.t, :, self.feat_names.index("logret")]
        lb_vol  = self.feat_tensor[self.t - mvw: self.t, :, self.feat_names.index("logret")]
        mean_r = lb_mom.mean(axis=0)
        vol    = lb_vol.std(axis=0) + 1e-6
        N_risky = self.N - 1 if self.cash_asset else self.N

        logits = np.full(self.N, -1e6, dtype=np.float32)
        if a_idx == 0:          # Nakit
            logits[-1] = 10.0 if self.cash_asset else 0.0
        elif a_idx == 1:        # Eşit Ağırlık
            logits[:N_risky] = 0.0
        elif a_idx == 2:        # Top-3 Momentum
            top = np.argsort(mean_r)[-3:]
            logits[top] = 2.0
        elif a_idx == 3:        # Top-5 Momentum
            top = np.argsort(mean_r)[-5:]
            logits[top] = 2.0
        elif a_idx == 4:        # Ters Volatilite
            iv = 1.0 / vol
            logits[:N_risky] = np.log(iv / iv.sum() + 1e-9)
        elif a_idx == 5:        # Minimum Volatilite (tek varlık, uzun pencere)
            i = int(np.argmin(vol))
            logits[i] = 10.0
        return logits

    def step(self, action_idx):
        # Rebalans günü değilse üst sınıfın _apply_action'ı aksiyonu zaten
        # yok sayar (w_{t-1} korunur); yine de boyutu doğru aksiyon geçmeliyiz.
        a_idx = int(action_idx)
        if self._should_rebalance():
            logits = self._discrete_to_logits(a_idx)
        else:
            logits = np.zeros(self.N, dtype=np.float32)
        return super().step(logits)
