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
    "short":  dict(rebalance=1,  mom_window=5,  minvol_window=20,  eta=0.0015, lam=0.25, tau=0.03, gamma=0.95,
                   min_days=1,  max_days=30,  train_max_steps=30),
    "medium": dict(rebalance=5,  mom_window=20, minvol_window=60,  eta=0.0010, lam=0.50, tau=0.05, gamma=0.99,
                   min_days=30, max_days=90,  train_max_steps=90),
    "long":   dict(rebalance=20, mom_window=60, minvol_window=120, eta=0.0005, lam=1.00, tau=0.08, gamma=0.995,
                   min_days=90, max_days=360, train_max_steps=360),
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


@dataclass(frozen=True)
class TD3Config:
    # Twin Delayed DDPG (RL_12) — hocanin surekli-eylem icin TAVSIYE ettigi algoritma.
    hidden: Tuple[int, int] = (256, 128)
    lr_pi: float = 3e-4
    lr_q: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2     # hedef-politika yumusatma gurultusu
    noise_clip: float = 0.5
    policy_delay: int = 2         # gecikmeli politika guncellemesi
    expl_noise: float = 0.1       # eylem kesif gurultusu
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
    # v8: fiyat gurultusu / slippage — hocanin ACIK sarti (anti-ezber). Gerceklesen
    # getiriye kucuk Gauss gurultusu: "al dediginde tam o fiyattan alamazsin, yukaridan
    # alirsin". YALNIZ egitimde (random_start=True) aktif; eval'de KAPALI -> golden eval
    # determinizmi korunur. 0.001 ~ gunluk getiriye ±%0.1 mikro-slippage.
    price_noise_std: float = 0.001
    price_noise_train_only: bool = True
    # v10: PARAMETRIK nakit (risksiz) faiz. Nakit varlik artik 0 degil, gunluk risksiz
    # getiri kazanir. cash_annual_rate gercekci TR 2022-24 mevduat/repo seviyesi (~%40);
    # UI/CLI'dan ayarlanabilir. Gunluk oran bilesik tutarlilikla turetilir:
    #   cash_daily_rate = (1 + cash_annual_rate)^(1/252) - 1
    # SABIT oran -> RNG cagrisi YOK -> golden RNG sirasi korunur (yalniz deger degisir).
    cash_annual_rate: float = 0.40
    trading_days: int = 252          # yillik->gunluk bilesik donusum tabani

    @property
    def cash_daily_rate(self) -> float:
        """Yillik nakit faizinin bilesik gunluk karsiligi: (1+R)^(1/252) - 1."""
        return cash_daily_rate(self.cash_annual_rate, self.trading_days)


def cash_daily_rate(cash_annual_rate: float, trading_days: int = 252) -> float:
    """Yillik nakit (risksiz) faizini bilesik gunluk orana cevirir.

    (1 + r_daily)^trading_days = 1 + cash_annual_rate  =>  r_daily = (1+R)^(1/D) - 1.
    Bagimsiz modul-duzeyi yardimci: env ctor (cash_daily_rate is None) buradan turetir;
    UI/CLI yillik orani gecerse env gunluk orani hesaplar. RNG kullanmaz (deterministik).
    """
    D = max(int(trading_days), 1)
    return float((1.0 + float(cash_annual_rate)) ** (1.0 / D) - 1.0)


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
    td3_episodes: int = 8         # TD3 off-policy (SAC ile ayni rejim)
    td3_episode_len: int = 600


# ---------------------------------------------------------------------
# v2: odul terim agirliklari. Mevcut terimler (log-getiri; tx-cost eta ile;
# drawdown lambda ile) korunur. Ek olarak online risk-ayarli Diferansiyel
# Sharpe (Moody & Saffell) terimi: total += w_dsr * DSR_t.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class RewardConfig:
    w_dsr: float = 0.05      # Diferansiyel Sharpe agirligi (0 -> kapali)
    dsr_eta: float = 0.01    # DSR EWMA orani
    # v7: rejim-amplified kuyruk-riski (CVaR) cezasi. w_cvar vade-bagli olceklenir
    # (env: kisa->yuksek, uzun->dusuk). kappa = w_cvar*(1+regime_beta*max(0,regime))^cvar_amp.
    w_cvar: float = 0.06        # CVaR base agirligi (0 -> kapali)
    cvar_alpha: float = 0.05    # kuyruk seviyesi (%5)
    regime_beta: float = 1.0    # kriz amplifikasyon gucu
    cvar_amp: float = 1.0       # rejim amplifikasyon usteli
    # v9: OPT-IN ek terimler (default 0/kapali -> golden bit-ayni). reward_overrides
    # ile UI'dan ayarlanabilir; env ctor + build_env bunlari okur.
    w_gain: float = 0.0         # kazanc-carpani odul agirligi (nav>gain_floor uzeri)
    gain_floor: float = 1.0     # kazanc esigi (nav bunun uzerinde odullenir)
    w_gain_speed: float = 0.0   # erken-kazanc hiz faktoru (0 -> hizdan bagimsiz)
    w_ruin_timing: float = 0.0  # erken-iflas ceza olcegi (0 -> flat bankruptcy_penalty)


# ---------------------------------------------------------------------
# Veri penceresi sabitleri — data.py START/END/SPLIT ile BIREBIR (tek kaynak).
# data.py bu DataConfig'i sonraki dalgada (veri-muhendisi) okuyacak; simdilik
# yalniz config'te yansitilir (frozen -> kazara mutasyon engellenir).
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    start: str = "2015-01-01"      # data.py START
    end: str = "2024-12-31"        # data.py END
    train_end: str = "2022-01-01"  # data.py SPLIT (train/test ayrim tarihi)


# ---------------------------------------------------------------------
# v2: CNN-LSTM forecaster (predict-then-optimize). enabled=True ise state'e
# bir 'forecast' feature'i eklenir -> F = 12 + 1 = 13, durum R^397 (DQN/SAC/TD3) /
# R^369 (PPO; forecast haric F=12). Not: 393/365 = makro-oncesi V5 tabani; +4 makro
# (MacroConfig.enabled, asagida) = 397/369.
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
    # Hangi ajanlar forecast feature'ini kullansin? V3<->V4 ablation'a gore forecast
    # DQN/SAC'a yaradi (+10pp DQN), PPO'ya zarar verdi (-7.5pp) -> PPO haric tutulur.
    # TD3 surekli-kontrolde SAC gibi davranir -> forecast ona da verilir.
    forecast_agents: tuple = ("DQN", "SAC", "TD3")


# ---------------------------------------------------------------------
# v6: Makro rejim algisi (YALIN OMURGA) — faiz/dolar/altin + bilesik rejim.
# Tek bir 'regime' skoru hem state'e (algi) hem RewardEngine'e (V7 kriz-amplified
# kuyruk cezasi) girer. enabled=False -> V5 davranisi (makrosuz). Yalin tutuldu
# (4 oznitelik) — sinirli BIST verisinde overfitting'e karsi.
# series: indirilen ham makro tickerlari (yfinance). BIST takvimine hizalanir.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class MacroConfig:
    enabled: bool = True
    series: tuple = ("^VIX", "^GSPC", "^TNX", "^IRX", "USDTRY=X", "GC=F")
    # State'e eklenen 4 yalin oznitelik: rejim omurgasi + faiz/dolar/altin.
    features: tuple = ("regime", "slope", "usd_try_mom", "gold_tl_mom")
    mom_window: int = 20      # momentum/degisim penceresi (gun)
