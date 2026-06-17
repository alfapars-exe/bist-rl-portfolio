"""Odul hesabi — SOLID P7 (SRP): env'in adim mekaniginden ayri odul motoru.

AdaptiveRewardShaper ve DifferentialSharpe env/portfolio_env.py'den BIREBIR
tasindi (golden-master 1e-6 korunur; portfolio_env re-export eder).

RewardEngine, PortfolioEnv.step() icindeki odul aritmetigini tek sinifta toplar.
compute() icindeki islem SIRASI step()'tekiyle birebir aynidir:
  shaper.update_and_shape -> tx_cost -> nav*=(1+port_r_net) -> max(nav,0)
  -> bankrupt -> peak -> dd -> log_r -> dd_penalty -> dsharpe.update(port_r_net)
  -> total
`terms` dict anahtarlari aynen korunur (test_env + UI panelleri bunlara bagli).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


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
# Odul motoru — step() aritmetiginin tek sahibi
# ---------------------------------------------------------------------
@dataclass
class RewardOutcome:
    """compute() ciktisi: env'in guncellemesi gereken durum + odul terimleri."""
    total: float
    nav: float
    peak: float
    port_r_net: float
    terms: dict


class RewardEngine:
    """Bir adimin odulunu ve NAV/peak gecisini hesaplar (SRP: env mekanikten ayri).

    Stateful: shaper + dsharpe EWMA'lari adimlar arasi tasinir; reset() episod
    basinda cagrilir (env._reset_state bunu yapar).
    """

    def __init__(self, shaper: AdaptiveRewardShaper, dsharpe: DifferentialSharpe,
                 w_dsr: float, bankruptcy_nav: float, bankruptcy_penalty: float):
        self.shaper = shaper
        self.dsharpe = dsharpe
        self.w_dsr = float(w_dsr)
        self.bankruptcy_nav = float(bankruptcy_nav)
        self.bankruptcy_penalty = float(bankruptcy_penalty)

    def reset(self):
        self.shaper.reset()
        self.dsharpe.reset()

    def compute(self, *, gross_port_r: float, delta_w_l1: float,
                nav: float, peak: float) -> RewardOutcome:
        """Aritmetik sirasi onceki PortfolioEnv.step() ile BIREBIR aynidir."""
        eta_t, lambda_t, tau_t, vol_ewma, to_ewma = \
            self.shaper.update_and_shape(gross_port_r, delta_w_l1)

        tx_cost = eta_t * delta_w_l1
        port_r_net = gross_port_r - tx_cost
        nav = nav * (1.0 + port_r_net)
        # Matematiksel olarak nav <= 0 olmamalı (çarpım süreci), ama extreme tx/return
        # kombinasyonunda ölçülebilir şekilde sıfıra yakın ya da negatif olabilir.
        # İflas eşiğinin altına düşerse clamp et; env episodu sonlandırır.
        nav = max(nav, 0.0)
        bankrupt = bool(nav < self.bankruptcy_nav)
        bankruptcy_penalty = self.bankruptcy_penalty if bankrupt else 0.0

        peak = max(peak, nav)
        dd = (peak - nav) / max(peak, 1e-9)
        log_r = float(np.log(max(1.0 + gross_port_r, 1e-6)))
        dd_penalty = lambda_t * max(0.0, dd - tau_t)
        dsr = self.dsharpe.update(port_r_net)
        dsr_term = self.w_dsr * dsr
        total = log_r - tx_cost - dd_penalty - bankruptcy_penalty + dsr_term

        terms = dict(
            log_return=log_r, tx_cost=tx_cost,
            drawdown_penalty=dd_penalty, total=total,
            dsr=dsr, dsr_term=dsr_term,
            eta_t=eta_t, lambda_t=lambda_t, tau_t=tau_t,
            vol_ewma=vol_ewma, turnover_ewma=to_ewma,
            dd=dd, gross_port_r=gross_port_r, delta_w_l1=delta_w_l1,
            bankruptcy_penalty=bankruptcy_penalty, bankrupt=bankrupt,
        )
        return RewardOutcome(total=total, nav=nav, peak=peak,
                             port_r_net=port_r_net, terms=terms)
