"""Merkezi yapilandirma — hiperparametreler tek kaynakta (Faz 2, H3).

Onceki durumda 'Prompt Spec' README'de, HORIZON_PRESETS env'de ve ajan
default'lari hem ctor imzalarinda hem app.py:_make_agent'ta hem train.py cagri
yerlerinde tekrarlaniyordu. Bu modul, hattin (pipeline) kullandigi degerler icin
TEK dogruluk kaynagidir; env (HORIZON_PRESETS), train.py ve app.py buradan okur.

Bu dataclass degerleri, hattin GERCEKTEN kullandigi (pipeline-effective)
degerlerdir ve mevcut davranisi BIREBIR korur (golden-master <=1e-6). Ornegin
PPO ctor default'u n_epochs=8 olsa da hat 6 kullanir; burada 6 tutulur.

config.py bir 'leaf' moduldur: yalnizca stdlib import eder (env/agents/app/train
buradan guvenle import edebilsin diye — dongusel import yok).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Tum deneylerde sabit tohum (tekrar-uretilebilirlik).
SEED = 42


# ---------------------------------------------------------------------
# Vade preset'leri — env/portfolio_env.py'den tasindi (Faz 2).
# rebalans frekansi, momentum/minvol pencereleri, (eta, lam, tau, gamma) base.
# ---------------------------------------------------------------------
HORIZON_PRESETS: Dict[str, dict] = {
    "short":  dict(rebalance=1,  mom_window=5,  minvol_window=20,  eta=0.0015, lam=0.25, tau=0.03, gamma=0.95),
    "medium": dict(rebalance=5,  mom_window=20, minvol_window=60,  eta=0.0010, lam=0.50, tau=0.05, gamma=0.99),
    "long":   dict(rebalance=20, mom_window=60, minvol_window=120, eta=0.0005, lam=1.00, tau=0.08, gamma=0.995),
}


# ---------------------------------------------------------------------
# v2: zenginlestirilmis teknik feature seti (Close-turevli, ileri-bakissiz).
# Not: OHLCV-bagimli gostergeler (ATR/ADX/CCI/Stochastic) ertelendi — yfinance bu
# ortamda BIST OHLCV dondurmuyor; yalniz gercek Close cache'i mevcut. Hepsi
# olcek-bagimsiz (oran/yuzde) -> enflasyonlu fiyat seviyesinden bagimsiz.
# add_features bu sirayla doner; state vektoru layout'u bu listeye baglidir.
# ---------------------------------------------------------------------
FEATURES = [
    "logret", "ma5", "ma20", "vol20", "rsi",            # v1 (mevcut)
    "macd_hist", "bb_pctb", "bb_bw", "roc10",            # v2 yeni
    "mom60", "vol60", "ema_dist",
]


# ---------------------------------------------------------------------
# Ajan hiperparametreleri (hat-etkin default'lar = README 'Prompt Spec').
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DQNConfig:
    hidden: Tuple[int, int] = (256, 128)
    lr: float = 1e-3
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 10_000
    buffer_size: int = 50_000
    batch_size: int = 64
    target_update: int = 500
    huber_delta: float = 1.0


@dataclass(frozen=True)
class PPOConfig:
    hidden: Tuple[int, int] = (256, 128)
    lr_p: float = 3e-4
    lr_v: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.005
    n_epochs: int = 6           # hat degeri (ctor default 8 degil — bkz. modul docstring)
    batch_size: int = 128
    log_std_init: float = -0.5
    rollout_len: int = 400      # train.py / app.py rollout uzunlugu


@dataclass(frozen=True)
class SACConfig:
    hidden: Tuple[int, int] = (256, 128)
    lr_pi: float = 3e-4
    lr_q: float = 5e-4
    gamma: float = 0.99
    tau: float = 0.01
    alpha: float = 0.05
    buffer_size: int = 50_000
    batch_size: int = 128


# ---------------------------------------------------------------------
# Ortam (env) default'lari — odul sekillendirici hedefleri + iflas/episod.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class EnvConfig:
    window: int = 20
    vol_target: float = 0.02
    turnover_target: float = 0.05
    ema_alpha: float = 0.05
    max_episode_steps: int = 252
    bankruptcy_nav: float = 0.01
    bankruptcy_penalty: float = 10.0
    random_start: bool = True        # v2: egitimde rastgele pencere (eval'de False gecilir)
    seed: int = 42


# ---------------------------------------------------------------------
# v2: egitim dongusu uzunluklari. random_start cesitliligi sagladigi icin adim
# sayilari CPU-dostu tutuldu (cesitlilik sayidan cok rastgele-pencereden gelir).
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    dqn_episodes: int = 12
    ppo_updates: int = 24
    ppo_rollout_len: int = 400
    sac_episodes: int = 8
    sac_episode_len: int = 600


# ---------------------------------------------------------------------
# v2: odul terim agirliklari. Mevcut terimler (log-getiri; tx-cost eta ile;
# drawdown lambda ile) korunur. Ek olarak online risk-ayarli Diferansiyel
# Sharpe (Moody & Saffell) terimi: total += w_dsr * DSR_t.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class RewardConfig:
    w_dsr: float = 0.05      # Diferansiyel Sharpe agirligi (0 -> kapali)
    dsr_eta: float = 0.01    # DSR EWMA orani


# ---------------------------------------------------------------------
# v2: CNN-LSTM forecaster (predict-then-optimize). enabled=True ise state'e
# bir 'forecast' feature'i eklenir -> F = 12 + 1 = 13, durum R^393.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastConfig:
    enabled: bool = True
    window: int = 20
    conv_ch: int = 16
    hidden: int = 32
    epochs: int = 4
    lr: float = 1e-3
    batch: int = 256
