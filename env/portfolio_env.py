"""Portföy yönetimi MDP ortamı — vade bazlı preset'ler + adaptif ödül şekillendirici.

MDP tuple (S, A, P, r, γ):

  S : ℝ^397 (DQN/SAC/TD3) / ℝ^369 (PPO) — 28 hisse × F özellik (z-skorlu)
      + 4 makro (MacroConfig.enabled) + 29 boyutlu önceki ağırlık (nakit dâhil).
      F=13 (12 teknik + 1 forecast) forecast ajanlarında; PPO forecast'ı dışlar -> F=12.
      (393/365 = makro-öncesi V5 tabanı; +4 makro = 397/369)
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

from typing import Dict, Literal
import numpy as np
import pandas as pd

from config import EnvConfig, HORIZON_PRESETS, RewardConfig
# P7 (SRP): odul siniflari env/reward.py'ye tasindi; buradan re-export edilir
# (test_env ve dis kullanicilar `from env.portfolio_env import DifferentialSharpe`
# yapmaya devam edebilir).
from env.reward import (  # noqa: F401
    AdaptiveRewardShaper, DifferentialSharpe, RewardEngine, RewardOutcome,
)


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

# Tek dogruluk kaynagi config.EnvConfig (kesif bulgusu: degerler burada kopyalanmisti
# ve EnvConfig degisikligi sessizce yok sayiliyordu). Degerler birebir ayni.
MAX_EPISODE_STEPS = EnvConfig.max_episode_steps      # 252 (1 is yili)
# NAV bu eşiğin altına düşerse episod iflas olarak sonlandırılır.
# 0.01 = başlangıç sermayesinin %1'ine inmek (pratikte "para bitti").
BANKRUPTCY_NAV    = EnvConfig.bankruptcy_nav
# İflas gerçekleştiğinde ajan'a uygulanan ek ödül cezası (log-ölçeğinde çok büyük).
BANKRUPTCY_PENALTY = EnvConfig.bankruptcy_penalty


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = x / max(temp, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------
# Sürekli sürüm (PPO / SAC için)
# ---------------------------------------------------------------------
class PortfolioEnv:
    """Sürekli aksiyon portföy ortamı — softmax ile N+1 simplex ağırlıkları."""

    def __init__(self, prices: pd.DataFrame, features: Dict[str, pd.DataFrame],
                 window: int = EnvConfig.window, cash_asset: bool = True,
                 horizon: Literal["short", "medium", "long"] = "medium",
                 adaptive: bool = True,
                 eta_base: float | None = None,
                 lambda_base: float | None = None,
                 tau_base: float | None = None,
                 vol_target: float = EnvConfig.vol_target,
                 turnover_target: float = EnvConfig.turnover_target,
                 ema_alpha: float = EnvConfig.ema_alpha,
                 max_steps: int = MAX_EPISODE_STEPS,
                 bankruptcy_nav: float | None = None,
                 bankruptcy_penalty: float | None = None,
                 random_start: bool = False,
                 seed: int | None = None,
                 price_noise_std: float = EnvConfig.price_noise_std,
                 price_noise_train_only: bool = EnvConfig.price_noise_train_only,
                 w_dsr: float = RewardConfig.w_dsr,
                 dsr_eta: float = RewardConfig.dsr_eta,
                 macro=None, regime=None):
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

        # P7 (SRP): odul hesabi RewardEngine'de — shaper + DSR + iflas parametreleri
        # tek motorda toplanir; step() yalniz piyasa/portfoy mekanigini yurutur.
        shaper = AdaptiveRewardShaper(
            eta_base=eta_base, lambda_base=lambda_base, tau_base=tau_base,
            vol_target=vol_target, turnover_target=turnover_target,
            ema_alpha=ema_alpha, enabled=adaptive,
        )
        # v7: CVaR kuyruk cezasi vade-bagli olceklenir (kisa->yuksek tail-bilinci).
        cvar_factor = {"short": 1.6, "medium": 1.0, "long": 0.6}.get(horizon, 1.0)
        self.reward = RewardEngine(
            shaper=shaper,
            dsharpe=DifferentialSharpe(eta=dsr_eta),   # v2: cevrim-ici risk-ayar (DSR)
            w_dsr=float(w_dsr),
            # Ayarlanabilir iflas parametreleri — None ise modul-duzeyi default'lar
            bankruptcy_nav=float(bankruptcy_nav) if bankruptcy_nav is not None else BANKRUPTCY_NAV,
            bankruptcy_penalty=(float(bankruptcy_penalty)
                                if bankruptcy_penalty is not None else BANKRUPTCY_PENALTY),
            w_cvar=RewardConfig.w_cvar * cvar_factor,   # v7: rejim-amplified kuyruk cezasi
            cvar_alpha=RewardConfig.cvar_alpha,
            regime_beta=RewardConfig.regime_beta,
            cvar_amp=RewardConfig.cvar_amp,
        )

        self.N_assets = prices.shape[1]
        self.cash_asset = cash_asset
        self.N = self.N_assets + (1 if cash_asset else 0)
        self.window = max(window, self.minvol_window)
        self.F = self.feat_tensor.shape[2]
        # v6: makro rejim blogu (z-skorlu (T,M), state'e eklenir) + HAM regime
        # (T,) ∈[-1,1] — V7 odul kriz-amplifikasyonu icin step()'te kullanilir.
        self.macro = None if macro is None else np.asarray(macro, dtype=np.float32)
        self.regime = None if regime is None else np.asarray(regime, dtype=np.float32)
        self.M = 0 if self.macro is None else int(self.macro.shape[1])
        self.state_dim = self.F * self.N_assets + self.M + self.N
        self.action_dim = self.N
        # n_days: toplam zaman adimi (satir). 't' ile yalniz buyuk/kucuk harfle
        # ayrilan 'T' adi karisikliga yol aciyordu (SonarCloud python:S1845) -> n_days.
        self.n_days = prices.shape[0]
        self.max_steps = max_steps
        # v2: env-yerel RNG — global np.random'a bagimli degil (tekrar-uretilebilirlik
        # kurulum sirasindan bagimsiz) + tohumlu rastgele-baslangic destegi.
        self.random_start = bool(random_start)
        self.rng = np.random.default_rng(seed)
        # v8: fiyat gurultusu/slippage (hocanin sarti). train_only -> yalniz random_start
        # (egitim) acik; eval (random_start=False) -> kapali, golden eval determinizmi korunur.
        self.price_noise_std = float(price_noise_std)
        self._noise_active = (self.price_noise_std > 0.0 and
                              (self.random_start if price_noise_train_only else True))
        self._reset_state()

    def _reset_state(self):
        lo = max(self.window, 21)
        if self.random_start:
            # episode'un max_steps adim + bir sonraki gun erisimi icin yer birak.
            # DIKKAT: rng.integers yalniz hi > lo iken cagrilir (RNG tuketimi /
            # golden-duyarli); degenerate pencerede deterministik lo'ya duser.
            hi = self.n_days - self.max_steps - 1
            if hi > lo:
                self.t = int(self.rng.integers(lo, hi))
            else:
                self.t = lo
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
        self.reward.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)   # env-yerel; global RNG'ye dokunmaz
        self._reset_state()
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        snap = self.feat_tensor[self.t].reshape(-1)
        parts = [snap]
        if self.macro is not None:                 # v6: makro rejim blogu
            parts.append(self.macro[self.t])
        parts.append(self.w)
        return np.concatenate(parts).astype(np.float32)

    def _risky_returns(self) -> np.ndarray:
        p0 = self.prices[self.t]
        p1 = self.prices[self.t + 1]
        r = (p1 - p0) / np.maximum(p0, 1e-9)
        if self._noise_active:
            # v8: slippage/fiyat gurultusu (hocanin sarti, anti-ezber) — gerceklesen
            # riskli getiriye kucuk Gauss gurultusu. Env-yerel rng -> global RNG'ye
            # dokunmaz; yalniz egitimde (random_start), eval'de kapali (deterministik).
            r = r + self.rng.normal(0.0, self.price_noise_std, size=r.shape).astype(np.float32)
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
        # Piyasa/portfoy mekanigi burada; odul aritmetigi RewardEngine'de (P7, SRP).
        w_new = self._apply_action(action)
        delta_w_l1 = float(np.abs(w_new - self.w).sum())
        r_vec = self._risky_returns()
        gross_port_r = float((w_new * r_vec).sum())

        regime_t = float(self.regime[self.t]) if self.regime is not None else 0.0
        outcome = self.reward.compute(gross_port_r=gross_port_r, delta_w_l1=delta_w_l1,
                                      nav=self.nav, peak=self.peak, regime=regime_t)
        self.nav, self.peak = outcome.nav, outcome.peak
        reward_terms = outcome.terms

        self.w = w_new
        self.t += 1
        self.step_count += 1
        self.nav_history.append(self.nav)
        self.weight_history.append(self.w.copy())
        self.ret_history.append(outcome.port_r_net)
        self.reward_terms_history.append(reward_terms)

        # İflas veya veri sonu → done; 252 adım tavanı → trunc
        done = (self.t >= (self.n_days - 1)) or reward_terms["bankrupt"]
        trunc = (self.step_count >= self.max_steps)
        info = dict(nav=self.nav, dd=reward_terms["dd"], port_r=outcome.port_r_net,
                    reward_terms=reward_terms, prices_t=self.prices[self.t].copy())
        return self._obs(), float(outcome.total), bool(done), bool(trunc), info


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
        # P7 + Copilot PR #2: aralik-disi indeks onceden sessizce near-uniform
        # portfoye donusuyordu (tum logitler -1e6). Acik hata ver — DQN her zaman
        # gecerli indeks urettigi icin golden etkilenmez.
        if not (0 <= a_idx < self.n_discrete):
            raise ValueError(
                f"action_idx={a_idx} aralik disi; [0, {self.n_discrete}) bekleniyor")
        if self._should_rebalance():
            logits = self._discrete_to_logits(a_idx)
        else:
            logits = np.zeros(self.N, dtype=np.float32)
        return super().step(logits)
