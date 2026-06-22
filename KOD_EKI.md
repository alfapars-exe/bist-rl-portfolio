# Kod Eki — BIST 28 Portföy Yönetimi (Pekiştirmeli Öğrenme)

> PDF §10: tüm kaynak kod final raporunun sonuna eklenir. Bu dosya `python scripts/build_code_appendix.py` ile tekrar üretilir (testler `tests/` altında ayrıca yer alır).

**Toplam: 38 kaynak dosya, ~7538 satır.**

---

## `config.py`

```python
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
# DEPRECATED-COMPAT (v12): yeni model `step_days` (N-gun) + acik parametreler kullanir;
# bu tablo yalniz (a) model dosya-adi slot'u, (b) env preset-fallback (None gecilen
# param'lar icin), (c) DEFAULTS'un turetim kaynagi olarak korunur. Degerler DEGISMEZ.
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
# v12: TEK ADIM/ORTALAMA MODELI — horizon preset + Gun/Ay/Yil granulerlik yerine
# tek `step_days` (N) + BAGIMSIZ odul parametreleri. DEFAULTS = eski "medium"
# preset'inin BIREBIR karsiligi (golden re-baseline kanonik degerleri).
#   N=1 -> gunluk (resample no-op). N>=2 -> N seanslik blok-sonu kapanis + N seanslik adim.
#   Episode uzunlugu = secilen train tarih araliginin TAMAMI (slider yok).
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class StepDefaults:
    step_days: int     = 1        # N: adim/ortalama penceresi (gun)
    eta: float         = 0.0010   # = HORIZON_PRESETS["medium"]["eta"]
    lam: float         = 0.50     # = ...["medium"]["lam"]
    tau: float         = 0.05     # = ...["medium"]["tau"]
    gamma: float       = 0.99     # = ...["medium"]["gamma"]
    mom_window: int    = 20       # = ...["medium"]["mom_window"]
    minvol_window: int = 60       # = ...["medium"]["minvol_window"]


DEFAULTS = StepDefaults()

STEP_DAYS_MIN = 1
STEP_DAYS_MAX = 252


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


# ---------------------------------------------------------------------
# Adim granülerliği — UI/CLI'dan secilebilir; feature'lar HER ZAMAN gunluk
# hesaplanir (add_features degismez), sonra resample_to_granularity() ile
# istenen frekansi indirgenir. "daily" -> no-op (golden-guvenli).
# ---------------------------------------------------------------------
GRANULARITY_OPTIONS: tuple = ("daily", "monthly", "yearly")

# Her granülerlik icin minimum nokta uyari esigi (resample sonrasi).
# n_points < esik ise downstream kod uyari verebilir.
GRANULARITY_MIN_POINTS: dict = {
    "daily":   252,   # 1 tam yil is gunu
    "monthly":  20,   # ~20 ay (~1.7 yil)
    "yearly":    5,   # 5 yil
}


# ---------------------------------------------------------------------
# v12: degenerate-config NaN korumasi. Env warm-up lo = max(window,21);
# resample sonrasi train aralige yetersizse env dejenere olup sessiz NaN uretir
# (kullanicinin "cok-kisa tarih araligi" NaN'i). Bu helper egitimi NaN yerine
# DOSTCA bir hatayla durdurmak icin kullanilir (UI st.error / CLI SystemExit).
# Leaf: yalniz stdlib + EnvConfig (ayni modul) -> testler ucuz import eder.
# ---------------------------------------------------------------------
MIN_TRAIN_POINTS_WARMUP = 21       # env lo_raw = max(window, 21)
MIN_TRAIN_STEPS_HEADROOM = 5       # warm-up sonrasi birkac adim (dejenere olmasin)


def validate_train_range(n_train_points: int, *, window: int = EnvConfig.window,
                         step_days: int = 1) -> tuple:
    """(ok, mesaj) — (resample edilmis) train araligi gecerli env kurabilir mi?

    n_train_points: step_days resample SONRASI len(px_tr) (env'in gordugu satir sayisi).
    ok=False ise mesaj kullaniciya gosterilir ve EGITIM BASLATILMAZ (sessiz NaN yerine).
    """
    lo = max(int(window), MIN_TRAIN_POINTS_WARMUP)
    need = lo + MIN_TRAIN_STEPS_HEADROOM
    if int(n_train_points) < need:
        return False, (
            f"Egitim araligi cok kisa: adim (step_days={step_days}) resample sonrasi "
            f"{n_train_points} nokta kaldi; en az {need} gerekli (isinma penceresi {lo} "
            f"+ {MIN_TRAIN_STEPS_HEADROOM} adim). Tarih araligini genislet veya step_days'i "
            f"dusur. Egitim baslatilmadi (sessiz NaN yerine)."
        )
    return True, ""

```

---

## `data.py`

```python
"""BIST 30 veri indirme ve ön işleme (28 hisse, KOZAA + KOZAL hariç).

- Günlük ayarlı kapanış fiyatlarını yfinance ile indirir.
- İlk çalıştırmada `data/prices.parquet` olarak cache'ler.
- Ağ yoksa BIST istatistiklerine kalibre edilmiş sentetik GBM fallback devreye girer.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import DataConfig, MacroConfig
from core.contracts import DataProvenance

# BIST 30 tickers — KOZAA.IS ve KOZAL.IS prompt gereği hariç tutuldu (28 hisse).
BIST28 = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
    "EREGL.IS", "FROTO.IS", "GARAN.IS", "HEKTS.IS", "ISCTR.IS",
    "KCHOL.IS", "KRDMD.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
    "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS",
    "TOASO.IS", "TUPRS.IS", "VAKBN.IS", "YKBNK.IS", "ENKAI.IS",
    "DOHOL.IS", "TTKOM.IS", "MGROS.IS",
]
# Geriye uyumluluk
BIST30 = BIST28

# Geriye uyum: modül sabitleri DataConfig'e işaret eder (tek kaynak).
# Bu satırları import eden mevcut kod (main.py, app.py, testler) kırılmaz.
_dc = DataConfig()
START = _dc.start      # "2015-01-01"
END   = _dc.end        # "2024-12-31"
SPLIT = _dc.train_end  # "2022-01-01"

BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
RESULTS_DIR  = BASE_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = DATA_DIR / "prices.parquet"
MACRO_PARQUET = DATA_DIR / "macro_raw.parquet"


def _cache_covers(index, start, end) -> bool:
    idx = pd.DatetimeIndex(index)
    requested = pd.bdate_range(start=start, end=end)
    if not len(idx) or not len(requested) or idx.min() > requested[0]:
        return False
    # Exchange holidays are not represented by pandas' generic business-day
    # calendar. Permit at most two nominal business days at the right boundary,
    # while still rejecting materially truncated caches.
    missing_tail = len(pd.bdate_range(idx.max(), requested[-1], inclusive="right"))
    return missing_tail <= 2


def _with_provenance(df: pd.DataFrame, *, source: str, provider: str,
                     start, end, reason: str = "", missing=()) -> pd.DataFrame:
    df.attrs["synthetic"] = source == "synthetic"
    df.attrs["provenance"] = DataProvenance(
        source=source, provider=provider, reason=reason,
        missing_tickers=tuple(missing), requested_start=str(start), requested_end=str(end),
    ).to_dict()
    return df


def _sanitize_prices(px: pd.DataFrame) -> pd.DataFrame:
    """Fiyat matrisini NaN'dan arindir: ileri/geri doldur + tamamen-bos sutunu dus.

    Gercek BIST verisinde halt/eksik gunler NaN birakabilir; tek bir NaN getiri
    env'de NAV'i zehirleyebilir (v12 env ctor NaN'i reddeder de). Cache OKUMA yolu
    onceden temizlenmiyordu -> zehirli/eksik cache'e karsi savunma. Temiz veride no-op.
    """
    px = px.ffill().bfill()
    all_nan = px.columns[px.isna().all()]
    if len(all_nan):
        px = px.drop(columns=list(all_nan))
    return px


def download_bist(tickers=BIST28, start=None, end=None,
                  use_cache: bool = True) -> pd.DataFrame:
    """28 hisselik (T, N) ayarlı kapanış fiyat matrisi döner.

    Parametreler
    ------------
    start : str | None
        Başlangıç tarihi (dahil). None → DataConfig.start ("2015-01-01").
    end : str | None
        Bitiş tarihi (dahil). None → DataConfig.end ("2024-12-31").

    Önce parquet cache'i dener; varsa verilen [start, end] aralığına dilimler.
    Cache uyumsuzsa veya yoksa yfinance'tan indirir. yfinance başarısızsa BIST-
    benzeri sentetik GBM üretir (deney yine çalışır).
    """
    _dc_local = DataConfig()
    if start is None:
        start = _dc_local.start
    if end is None:
        end = _dc_local.end

    if use_cache and PARQUET_PATH.exists():
        try:
            px = pd.read_parquet(PARQUET_PATH)
            px.index = pd.to_datetime(px.index)
            expected = set(tickers)
            if (expected.issubset(set(px.columns)) and len(px) > 500
                    and _cache_covers(px.index, start, end)):
                # Cache tüm aralığı tutabilir; istenen [start, end]'e dilimle.
                px_slice = _sanitize_prices(px[list(tickers)])
                px_slice = px_slice.loc[
                    (px_slice.index >= pd.Timestamp(start)) &
                    (px_slice.index <= pd.Timestamp(end))
                ]
                if len(px_slice) > 100:
                    return _with_provenance(px_slice, source="real", provider="parquet-cache",
                                            start=start, end=end)
            print("[INFO] cache uyumsuz veya dilim boş, yeniden indiriliyor ...")
        except Exception as exc:
            print(f"[WARN] parquet okunamadı ({exc!r}); yeniden indiriliyor")

    synthetic = False
    try:
        import yfinance as yf
        inclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download(
            tickers, start=start, end=inclusive_end,
            auto_adjust=True, progress=False, threads=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            px = data["Close"].copy()
        else:
            px = data[["Close"]].copy()
        # yfinance bazen tz-aware / time-bilesenli index doner; sentetik-doldurma reindex'i
        # (naive bdate_range) eslesmezse TUM sentetik sutunlar NaN -> son dropna frame'i
        # BOSALTIR (HF Space MIXED-bos hatasi). Index'i tz-naive GUNE normalize et.
        _idx = pd.to_datetime(px.index)
        px.index = (_idx.tz_localize(None) if _idx.tz is not None else _idx).normalize()
        px = px.dropna(axis=1, thresh=int(0.9 * len(px)))
        px = px.ffill().dropna()
        if px.shape[1] < 10 or len(px) < 100:
            raise RuntimeError("Too few tickers returned")
        px = px[[c for c in tickers if c in px.columns]]
        # Evren degismezligi: yfinance kismi dondurduyse (bazi ticker'lar eksik /
        # >%10 NaN -> dropna ile dustu) eksikleri sentetik ile doldurup TAM BIST28'i
        # garanti et. Aksi halde N degisir -> UI ASSET_NAMES (sabit 29) ile uyumsuzluk
        # + kayitli ajan state-dim (397) ile uyumsuzluk olur.
        miss = [c for c in tickers if c not in px.columns]
        if miss:
            warnings.warn(
                f"{len(miss)} ticker yfinance'tan gelmedi; sentetik ile dolduruldu: "
                f"{miss}", RuntimeWarning, stacklevel=2)
            synth = _synthetic_bist(miss, start, end).reindex(px.index).ffill()
            for c in miss:
                px[c] = synth[c].to_numpy()
        px = px[list(tickers)].ffill().dropna()
        # EMNIYET AGI: islenmis frame bos/cok-kisa veya NaN'li ise (index eslesmezligi vb.)
        # bozuk MIXED frame'i DONDURME -> except'e dusur (CSV/sentetik fallback gecerli veri verir).
        if len(px) < 100 or bool(px.isna().any().any()):
            raise RuntimeError(f"islenmis yfinance verisi gecersiz ({len(px)} satir)")
        if miss:
            _with_provenance(px, source="mixed", provider="yfinance+synthetic",
                             start=start, end=end, reason="missing yfinance tickers", missing=miss)
        else:
            _with_provenance(px, source="real", provider="yfinance", start=start, end=end)
    except Exception as exc:
        # SENTETIGE DUSMEDEN ONCE: kanonik results/bist30_prices.csv (tam 2015-2024 GERCEK
        # BIST verisi). Bu dosya .gitignore'da DEGIL -> HF Space'e yuklenir (data/*.parquet
        # cache'i gitignore-aware upload nedeniyle Space'e GIDEMEZ). Ag/cache yoksa bunu kullan.
        csv_fb = RESULTS_DIR / "bist30_prices.csv"
        if use_cache and csv_fb.exists():
            try:
                pxc = pd.read_csv(csv_fb, index_col=0, parse_dates=True)
                if (set(tickers).issubset(set(pxc.columns))
                        and _cache_covers(pxc.index, start, end)):
                    pxc = pxc[list(tickers)]
                    pxc = pxc.loc[(pxc.index >= pd.Timestamp(start)) &
                                  (pxc.index <= pd.Timestamp(end))].ffill().dropna()
                    if len(pxc) > 100:
                        print("[INFO] yfinance yok; results/bist30_prices.csv (GERCEK BIST) kullanildi")
                        return _with_provenance(pxc, source="real", provider="csv-cache",
                                                start=start, end=end)
            except Exception as exc_csv:
                print(f"[WARN] bist30_prices.csv okunamadi ({exc_csv!r})")
        # Sessiz yutma yok: stderr'e gorunur uyari (CI loglari + kullanici).
        warnings.warn(
            f"yfinance basarisiz ({exc!r}); SENTETIK GBM verisi uretiliyor — "
            "bu GERCEK BIST fiyati DEGIL, sonuclar yalnizca demo amaclidir!",
            RuntimeWarning, stacklevel=2,
        )
        print(f"[WARN] yfinance başarısız ({exc!r}); sentetik BIST verisi üretiliyor")
        px = _synthetic_bist(tickers, start, end)
        synthetic = True
        _with_provenance(px, source="synthetic", provider="synthetic-gbm",
                         start=start, end=end, reason=repr(exc))

    if synthetic:
        # KRITIK: sentetik veri CACHE'E YAZILMAZ. Onceki surum yaziyordu;
        # bir kez ag hatasi -> sonraki TUM calistirmalar cache'ten sessizce
        # sahte veri okuyordu (cache zehirlenmesi).
        print("[WARN] sentetik veri cache'e yazılmadı; ağ gelince gerçek veri indirilecek")
        return px
    if px.attrs.get("provenance", {}).get("source") == "mixed":
        print("[WARN] karma gercek/sentetik veri cache'e yazilmadi")
        return px

    try:
        px.to_parquet(PARQUET_PATH)
    except Exception as exc:
        print(f"[WARN] parquet yazılamadı ({exc!r}); CSV fallback")
        px.to_csv(DATA_DIR / "prices.csv")
    return px


def _synthetic_bist(tickers, start, end) -> pd.DataFrame:
    """BIST istatistiklerine kalibre edilmiş çok değişkenli GBM seti."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start=start, end=end)
    T = len(idx)
    N = len(tickers)
    mu    = rng.normal(0.18, 0.08, size=N) / 252.0
    sigma = np.clip(rng.normal(0.40, 0.10, size=N), 0.2, 0.8) / np.sqrt(252.0)
    base = rng.normal(0, 1, size=(N, 4))
    C = np.corrcoef(base @ rng.normal(0, 1, size=(4, N)) + 0.5 * rng.normal(0, 1, (N, N)))
    C = 0.6 * C + 0.4 * np.eye(N)
    L = np.linalg.cholesky(C)
    Z = rng.normal(0, 1, size=(T, N)) @ L.T
    logret = mu + sigma * Z
    crash_idx = (idx >= "2020-03-05") & (idx <= "2020-03-30")
    logret[crash_idx] += rng.normal(-0.015, 0.02, size=(crash_idx.sum(), N))
    rally = (idx >= "2021-06-01") & (idx <= "2022-12-31")
    logret[rally] += 0.0012
    price = 100.0 * np.exp(np.cumsum(logret, axis=0))
    return pd.DataFrame(price, index=idx, columns=tickers)


# ---------------------------------------------------------------------
# v6: Makro rejim verisi (faiz/dolar/altin) — egzojen, BIST takvimine hizali.
# ---------------------------------------------------------------------
def download_macro(series=tuple(MacroConfig.series), start=None, end=None,
                   use_cache: bool = True) -> pd.DataFrame:
    """Ham makro seri matrisi (T, M): VIX, S&P, faiz (TNX/IRX), USDTRY, altin (GC=F).

    Parametreler
    ------------
    start : str | None
        Başlangıç tarihi (dahil). None → DataConfig.start ("2015-01-01").
    end : str | None
        Bitiş tarihi (dahil). None → DataConfig.end ("2024-12-31").

    parquet cache -> yfinance -> sentetik fallback. Cache varsa [start, end]'e
    dilimler. Sentetik veri CACHE'E YAZILMAZ (download_bist ile ayni zehirlenme
    korumasi)."""
    _dc_local = DataConfig()
    if start is None:
        start = _dc_local.start
    if end is None:
        end = _dc_local.end

    series = list(series)
    if use_cache and MACRO_PARQUET.exists():
        try:
            mc = pd.read_parquet(MACRO_PARQUET)
            mc.index = pd.to_datetime(mc.index)
            have = [s for s in series if s in mc.columns]
            if len(have) >= 3 and len(mc) > 500 and _cache_covers(mc.index, start, end):
                mc_slice = mc[have]
                mc_slice = mc_slice.loc[
                    (mc_slice.index >= pd.Timestamp(start)) &
                    (mc_slice.index <= pd.Timestamp(end))
                ]
                if len(mc_slice) > 100:
                    return _with_provenance(mc_slice, source="real", provider="parquet-cache",
                                            start=start, end=end)
        except Exception as exc:
            print(f"[WARN] makro cache okunamadı ({exc!r}); yeniden indiriliyor")

    synthetic = False
    try:
        import yfinance as yf
        inclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download(series, start=start, end=inclusive_end, auto_adjust=True,
                           progress=False, threads=True)
        mc = (data["Close"].copy() if isinstance(data.columns, pd.MultiIndex)
              else data[["Close"]].copy())
        mc = mc.ffill().dropna(axis=1, how="all")
        mc = mc[[s for s in series if s in mc.columns]]
        if mc.shape[1] < 3 or len(mc) < 100:
            raise RuntimeError("Too few macro series returned")
        missing = [s for s in series if s not in mc.columns]
        _with_provenance(mc, source="mixed" if missing else "real", provider="yfinance",
                         start=start, end=end,
                         reason="missing macro series" if missing else "", missing=missing)
    except Exception as exc:
        # SENTETIGE DUSMEDEN ONCE: results/macro_raw.csv (gercek makro; gitignore'da DEGIL
        # -> HF Space'te bulunur). Ag/cache yoksa bunu kullan.
        csv_fb = RESULTS_DIR / "macro_raw.csv"
        if use_cache and csv_fb.exists():
            try:
                mcc = pd.read_csv(csv_fb, index_col=0, parse_dates=True)
                have = [s for s in series if s in mcc.columns]
                if len(have) >= 3:
                    mcc = mcc[have].loc[(mcc.index >= pd.Timestamp(start)) &
                                        (mcc.index <= pd.Timestamp(end))].ffill()
                    if len(mcc) > 100 and _cache_covers(mcc.index, start, end):
                        print("[INFO] yfinance yok; results/macro_raw.csv (GERCEK makro) kullanildi")
                        return _with_provenance(mcc, source="real", provider="csv-cache",
                                                start=start, end=end)
            except Exception as exc_csv:
                print(f"[WARN] macro_raw.csv okunamadi ({exc_csv!r})")
        warnings.warn(
            f"yfinance makro basarisiz ({exc!r}); SENTETIK makro uretiliyor — "
            "gercek piyasa verisi DEGIL!", RuntimeWarning, stacklevel=2)
        print(f"[WARN] yfinance makro başarısız ({exc!r}); sentetik makro üretiliyor")
        mc = _synthetic_macro(series, start, end)
        synthetic = True
        _with_provenance(mc, source="synthetic", provider="synthetic-macro",
                         start=start, end=end, reason=repr(exc))

    if synthetic:
        print("[WARN] sentetik makro cache'e yazılmadı")
        return mc
    if mc.attrs.get("provenance", {}).get("source") == "mixed":
        print("[WARN] eksik serili karma makro veri cache'e yazilmadi")
        return mc
    try:
        mc.to_parquet(MACRO_PARQUET)
    except Exception as exc:
        print(f"[WARN] makro parquet yazılamadı ({exc!r})")
    return mc


def _synthetic_macro(series, start, end) -> pd.DataFrame:
    """Determinist sentetik makro (offline + testler). TL sürekli zayıflar (gerçekçi)."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(start=start, end=end)
    T = len(idx)
    vix    = np.clip(18 + 9 * np.abs(np.cumsum(rng.normal(0, 0.25, T)) % 3.0), 9.0, 80.0)
    gspc   = 1500.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, T)))
    tnx    = np.clip(3.0 + np.cumsum(rng.normal(0, 0.02, T)), 0.5, 6.0)
    irx    = np.clip(tnx - 0.5 + rng.normal(0, 0.1, T), 0.05, None)
    usdtry = 2.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, T)))   # TL erir
    gold   = 1300.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, T)))  # altın USD/oz
    full = {"^VIX": vix, "^GSPC": gspc, "^TNX": tnx, "^IRX": irx,
            "USDTRY=X": usdtry, "GC=F": gold}
    df = pd.DataFrame(full, index=idx)
    return df[[s for s in series if s in df.columns]]


def resample_to_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Fiyat/feature/makro DataFrame'ini istenen adım granülerliğine indirgeer.

    Parametreler
    ------------
    df : pd.DataFrame
        DatetimeIndex'li DataFrame (fiyat, feature veya makro).
    granularity : str
        "daily"   -> df'i AYNEN döndür (NO-OP; golden-güvenli, RNG sırası korunur).
        "monthly" -> aylık ortalama (pandas ME kuralı).
        "yearly"  -> yıllık ortalama (pandas YE kuralı).

    Dönüş
    ------
    pd.DataFrame
        DatetimeIndex korunur. "daily" dışında .ffill().bfill() uygulanır
        (boş dönem NaN'larına karşı güvenlik).

    Not: feature'lar her zaman günlük hesaplanır (add_features DEĞİŞMEZ).
    Bu fonksiyon yalnızca downstream adımda (env/UI) granülerliği indirger.
    """
    if granularity == "daily":
        # NO-OP: aynı nesneyi döndür — golden testleri etkilemez, RNG sırası korunur.
        return df

    rule_map = {
        "monthly": "ME",   # month-end
        "yearly":  "YE",   # year-end
    }
    rule = rule_map.get(granularity)
    if rule is None:
        raise ValueError(
            f"Geçersiz granularity={granularity!r}; "
            f"geçerli seçenekler: {('daily', 'monthly', 'yearly')}"
        )

    resampled = df.resample(rule).mean()
    # Boş ay/yıl periyotlarında oluşabilecek NaN'lara karşı güvenlik.
    resampled = resampled.ffill()
    return resampled


def resample_to_step_days(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """N-günlük blok-ortalama resample (v12 tek-adım modeli — granülerliğin yerine).

    n=1  -> df'i AYNEN döndür (NO-OP; golden-güvenli, RNG sırası korunur).
    n>=2 -> df.resample(f"{n}D").mean().ffill().bfill() (boş blok NaN koruması).

    ÇAĞRI SIRASI (leak-safe): add_features (GÜNLÜK, tam seri) -> train_test_split
    -> resample_to_step_days (HER split AYRI). Resample atomik nokta üretir -> train/test
    sınırı blok-hizalı, karışma yok. Feature'lar HER ZAMAN günlük hesaplanır (add_features DEĞİŞMEZ).
    DatetimeIndex korunur (env `self.dates = prices.index` için).
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"step_days en az 1 olmali; {n} geldi")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample_to_step_days DatetimeIndex gerektirir")
    if n == 1:
        df.attrs["session_counts"] = tuple([1] * len(df))
        df.attrs["step_days"] = 1
        return df
    if len(df) == 0:
        out = df.copy()
        out.attrs["session_counts"] = ()
        out.attrs["step_days"] = n
        return out

    # Consecutive exchange sessions, not calendar-day buckets. Each selected row
    # is a tradable period-end observation. Keep the trailing partial period.
    positions = list(range(n - 1, len(df), n))
    counts = [n] * len(positions)
    if not positions or positions[-1] != len(df) - 1:
        previous = positions[-1] + 1 if positions else 0
        positions.append(len(df) - 1)
        counts.append(len(df) - previous)
    out = df.iloc[positions].copy()
    out.attrs.update(df.attrs)
    out.attrs["session_counts"] = tuple(int(x) for x in counts)
    out.attrs["step_days"] = n
    return out


def align_macro(macro_raw: pd.DataFrame, index) -> pd.DataFrame:
    """Makroyu BIST işlem takvimine (index) reindex + ffill/bfill (causal)."""
    out = macro_raw.reindex(index).ffill()
    out.attrs.update(macro_raw.attrs)
    return out


def train_test_split(df: pd.DataFrame, split=None):
    """DataFrame'i train (< split) ve test (>= split) olarak ikiye böler.

    Parametreler
    ------------
    split : str | None
        Bölünme tarihi. None → DataConfig.train_end ("2022-01-01").
        Train kesinlikle test'ten önce gelir; sızıntı yoktur.
    """
    if split is None:
        split = DataConfig().train_end
    train = df[df.index < split].copy()
    test = df[df.index >= split].copy()
    train.attrs.update(df.attrs)
    test.attrs.update(df.attrs)
    return train, test


if __name__ == "__main__":
    px = download_bist()
    print(f"Prices shape: {px.shape}, {px.index[0].date()} → {px.index[-1].date()}")
    print(f"Tickers ({len(px.columns)}): {list(px.columns)}")
    px.to_csv(RESULTS_DIR / "bist30_prices.csv")
    # Feature CSV'leri artık utils.features tarafından üretilir; CLI için orada da kopyasını yaz
    try:
        from utils.features import add_features
        feats = add_features(px)
        for name, f in feats.items():
            f.to_csv(RESULTS_DIR / f"feat_{name}.csv")
        print("Saved prices & features.")
    except ImportError:
        print("Saved prices only (utils.features not yet available).")

```

---

## `utils/features.py`

```python
"""Özellik mühendisliği ve eğitim-seti z-score ölçekleyicisi.

Durum vektörü için her hisseden 12 teknik öznitelik üretilir (hepsi Close-türevli,
ölçek-bağımsız ve yalnız-geçmişe-bakar):
  - logret    : günlük log getiri
  - ma5/ma20  : 5g / 20g yüzde değişim
  - vol20/vol60 : 20g / 60g logret standart sapması
  - rsi       : 14 günlük RSI (0-1 ölçekli)
  - macd_hist : MACD histogramı (EMA12-EMA26-sinyal9), close ile normalize
  - bb_pctb   : merkezli Bollinger %b (20g): (close-SMA)/(2*std) = 2*%b-1, ~[-1,1]
  - bb_bw     : Bollinger bant genişliği (20g)
  - roc10     : 10g değişim
  - mom60     : 60g momentum
  - ema_dist  : EMA50'ye göreli uzaklık

`TrainScaler` istatistikleri YALNIZCA eğitim kümesinde fit eder;
test dönemine aynı istatistikler uygulanır → veri sızıntısı yok.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def add_features(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Close (T,N) matrisinden zenginleştirilmiş teknik feature dict'i döner (ham).

    Anahtar sırası `config.FEATURES` ile aynıdır. Tüm göstergeler yalnız-geçmişe
    bakar (rolling/ewm/pct_change/diff — causal) → ileri-bakış (lookahead) yok.
    """
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices bos olmayan DatetimeIndex'li DataFrame olmali")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("prices pozitif ve sonlu olmali")
    close = prices
    logret = np.log(close).diff().fillna(0.0)
    ma5    = close.pct_change(5).fillna(0.0)
    ma20   = close.pct_change(20).fillna(0.0)
    vol20  = logret.rolling(20).std().fillna(0.0)
    delta  = close.diff()
    up     = delta.clip(lower=0).rolling(14).mean()
    dn     = (-delta.clip(upper=0)).rolling(14).mean()
    rsi    = (100 - 100 / (1 + up / (dn + 1e-9))).fillna(50) / 100.0

    # --- v2 yeni göstergeler (Close-türevli, ölçek-bağımsız) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = ((macd - signal) / close).fillna(0.0)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_pctb = ((close - sma20) / (2 * std20 + 1e-9)).fillna(0.0)
    bb_bw   = ((4 * std20) / (sma20 + 1e-9)).fillna(0.0)
    roc10   = close.pct_change(10).fillna(0.0)
    mom60   = close.pct_change(60).fillna(0.0)
    vol60   = logret.rolling(60).std().fillna(0.0)
    ema50   = close.ewm(span=50, adjust=False).mean()
    ema_dist = ((close - ema50) / (ema50 + 1e-9)).fillna(0.0)

    return {"logret": logret, "ma5": ma5, "ma20": ma20, "vol20": vol20, "rsi": rsi,
            "macd_hist": macd_hist, "bb_pctb": bb_pctb, "bb_bw": bb_bw, "roc10": roc10,
            "mom60": mom60, "vol60": vol60, "ema_dist": ema_dist}


class TrainScaler:
    """Özellik-bazında (her feature tipi × her hisse) z-score ölçekleyici.

    fit(): eğitim DataFrame'lerinden mean/std hesaplar
    transform(): aynı istatistiklerle test DataFrame'lerini ölçekler
    fit_transform(): tek adımda yapar
    """

    def __init__(self, clip: float = 8.0):
        self.means: Dict[str, pd.Series] = {}
        self.stds:  Dict[str, pd.Series] = {}
        self.clip = clip
        self.fitted = False

    def fit(self, feats: Dict[str, pd.DataFrame]) -> "TrainScaler":
        if not feats or any(df.empty for df in feats.values()):
            raise ValueError("TrainScaler.fit bos feature matrisi kabul etmez")
        for k, df in feats.items():
            self.means[k] = df.mean(axis=0)
            self.stds[k]  = df.std(axis=0).replace(0, 1.0)
        self.fitted = True
        return self

    def transform(self, feats: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if not self.fitted:
            raise RuntimeError("TrainScaler.fit() önce çağrılmalı")
        out: Dict[str, pd.DataFrame] = {}
        for k, df in feats.items():
            mu = self.means[k]
            sd = self.stds[k]
            z = (df - mu) / sd
            z = z.clip(lower=-self.clip, upper=self.clip)
            out[k] = z.fillna(0.0)
        return out

    def fit_transform(self, feats_train: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        self.fit(feats_train)
        return self.transform(feats_train)

```

---

## `utils/macro.py`

```python
"""Makro rejim özellik mühendisliği — v6 (causal, leak-safe, YALIN omurga).

Ham egzojen makro panelini (VIX, S&P, faiz TNX/IRX, USDTRY, altın GC=F) küçük,
standart bir **rejim-koşullandırma** vektörüne çevirir → RL state'ine eklenir.
Tüm dönüşümler yalnız-geçmişe bakar (causal). Standardizasyon (`MacroScaler`)
YALNIZ train penceresinde fit edilir, test'e aynı uygulanır → sızıntı yok
(`utils.features.TrainScaler` ile birebir desen).

4 yalın öznitelik (kullanıcının "faiz, dolar, altın" üçlüsü + bileşik rejim):
  - regime      : tanh(vix_rel + 4·(−spx_dd) − 0.10) ∈ (−1,1)  [OMURGA — V7 amplify eder]
  - slope       : 10Y − 13W faiz farkı (getiri eğrisi eğimi)    [FAİZ]
  - usd_try_mom : USD/TRY 20g momentum                           [DOLAR]
  - gold_tl_mom : (altın×USDTRY = gram-altın/TL) 20g momentum    [ALTIN]

Gerekçe: makro-kör RL tahsisçisi tek fiyat-yoluna aşırı uyar, rejimler arası
başarısız olur. Makro rejimine koşullanmak ajanın *ortamı* öğrenmesini sağlar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MacroConfig


def add_macro_features(macro_raw: pd.DataFrame,
                       mom_window: int = MacroConfig.mom_window) -> pd.DataFrame:
    """Ham makro panel (T, seri) -> yalın makro öznitelikleri (T, 4).

    Kolon sırası `config.MacroConfig.features` ile birebir. Eksik kaynak seri
    nazikçe 0'a düşer (degrade gracefully)."""
    def col(name: str) -> pd.Series:
        if name in macro_raw.columns:
            return macro_raw[name].astype(float)
        return pd.Series(0.0, index=macro_raw.index)

    vix = col("^VIX")
    spx = col("^GSPC")
    tnx = col("^TNX")
    irx = col("^IRX")
    usdtry = col("USDTRY=X")
    gold = col("GC=F")

    feats: dict[str, pd.Series] = {}

    # OMURGA: bileşik causal rejim skoru ∈ (−1,1) — sakin < 0 < kriz.
    vix_rel = (vix / vix.rolling(252, min_periods=20).median() - 1.0).fillna(0.0)
    spx_dd  = (spx / spx.rolling(252, min_periods=20).max() - 1.0).fillna(0.0)  # <=0
    feats["regime"] = np.tanh(1.0 * vix_rel + 4.0 * (-spx_dd) - 0.10)

    # FAİZ: getiri eğrisi eğimi (10Y − 13W); resesyon/risk proxy.
    feats["slope"] = (tnx - irx).fillna(0.0)

    # DOLAR: USD/TRY momentum (TL zayıflaması).
    feats["usd_try_mom"] = usdtry.pct_change(mom_window).fillna(0.0)

    # ALTIN: gram-altın/TL ≈ altın(USD/oz)×USDTRY momentum (oran → sabit düşer).
    gold_tl = gold * usdtry
    feats["gold_tl_mom"] = gold_tl.pct_change(mom_window).fillna(0.0)

    df = pd.DataFrame(feats, index=macro_raw.index)
    cols = [c for c in MacroConfig.features if c in df.columns]
    return df[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)


class MacroScaler:
    """Makro öznitelik matrisi için train-only z-score (tek DataFrame).

    `utils.features.TrainScaler` ile aynı sözleşme; tek matris üzerinde çalışır.
    'regime' zaten ∈(−1,1) olsa da tutarlılık için o da z-skorlanır (state için);
    ödül amplifikasyonu HAM regime'i ayrıca kullanır (bkz. train.prepare_data)."""

    def __init__(self, clip: float = 8.0):
        self.mean: pd.Series | None = None
        self.std: pd.Series | None = None
        self.clip = clip
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "MacroScaler":
        if df.empty:
            raise ValueError("MacroScaler.fit bos veri kabul etmez")
        self.mean = df.mean(axis=0)
        self.std = df.std(axis=0).replace(0, 1.0)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("MacroScaler.fit() önce çağrılmalı")
        z = (df - self.mean) / self.std
        return z.clip(lower=-self.clip, upper=self.clip).fillna(0.0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

```

---

## `utils/metrics.py`

```python
"""Backtest metrikleri — CAGR, Sharpe, Sortino, MaxDD, Calmar, Turnover + başarı metriği.

Ayrica PDF §9.7 egitim teshisleri (moving_average + training_diagnostics): episode
getirisi hareketli ortalamasi, basari orani, ortalama adim, ceza/iflas sayisi. Bunlar
GOZLEMSEL'dir — egitim/odul/eval sayisal yolunu degistirmez (golden-master korunur).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _years_from_dates(dates) -> float | None:
    if dates is None or len(dates) < 2:
        return None
    idx = pd.DatetimeIndex(dates)
    days = float((idx[-1] - idx[0]).days)
    return days / 365.2425 if days > 0 else None


def _annual_periods(dates, n_obs: int) -> float:
    years = _years_from_dates(dates)
    if years and n_obs > 1:
        return float((n_obs - 1) / years)
    return float(TRADING_DAYS)


def cagr(nav: np.ndarray, dates=None) -> float:
    years = _years_from_dates(dates)
    if years is None:
        years = len(nav) / TRADING_DAYS
    return float(nav[-1] ** (1 / max(years, 1e-6)) - 1)


def sharpe(rets: np.ndarray, rf: float = 0.0, annual_periods: float = TRADING_DAYS) -> float:
    sd_raw = float(np.std(rets))
    if sd_raw < 1e-10:          # sabit-getiri (risksiz/nakit) -> Sharpe tanimsiz
        return float("nan")
    mu = np.mean(rets) - rf / annual_periods
    return float(np.sqrt(annual_periods) * mu / (sd_raw + 1e-9))


def sortino(rets: np.ndarray, rf: float = 0.0, annual_periods: float = TRADING_DAYS) -> float:
    """Sortino orani — payda kanonik downside deviation (Sortino/Price):
    sqrt(mean(min(r - hedef, 0)^2)), TUM gozlemler uzerinden (empyrical uyumu).

    Onceki surum np.std(negatif altkume) kullaniyordu — iki hata: (1) sapma
    negatif altkumenin KENDI ortalamasina goreydi (hedefe degil), (2) yalniz
    negatif gun sayisina bolunuyordu. Uniform kayiplarda payda ~0 olup orani
    sisiriyor, hic negatif getiri yokken NaN donuyordu (kesif bulgusu, HIGH).
    """
    target = rf / annual_periods
    mu = np.mean(rets) - target
    downside_dev = float(np.sqrt(np.mean(np.minimum(rets - target, 0.0) ** 2)))
    if downside_dev < 1e-12:
        # Hic asagi-yonlu sapma yok: pozitif ortalamada sonsuz, aksi halde 0.
        return float("nan") if mu > 0 else 0.0   # pozitif sabit-getiri -> tanimsiz
    return float(np.sqrt(annual_periods) * mu / downside_dev)


def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return float(dd.min())


def calmar(nav: np.ndarray, dates=None) -> float:
    mdd = abs(max_drawdown(nav))
    if mdd < 1e-10:             # dususuz (risksiz/nakit) -> Calmar tanimsiz
        return float("nan")
    return cagr(nav, dates=dates) / (mdd + 1e-9)


def turnover(weights: np.ndarray) -> float:
    """Günlük ‖Δw‖₁ ortalaması."""
    d = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    return float(d.mean()) if len(d) else 0.0


def summary(nav: np.ndarray, rets: np.ndarray | None = None, weights=None, dates=None,
            turnover_values=None) -> dict:
    nav = np.asarray(nav, dtype=float)
    if nav.ndim != 1 or nav.size == 0 or not np.isfinite(nav).all() or np.any(nav < 0):
        raise ValueError("nav sonlu, negatif olmayan ve bos olmayan 1D dizi olmali")
    if rets is None:
        rets = np.diff(nav) / np.maximum(nav[:-1], 1e-12)
    rets = np.asarray(rets, dtype=float)
    if rets.ndim != 1 or not np.isfinite(rets).all():
        raise ValueError("rets sonlu 1D dizi olmali")
    # Baseline contracts use the first price date as a bookkeeping row.
    metric_rets = rets[1:] if len(rets) > 1 and abs(float(rets[0])) < 1e-15 else rets
    ann = _annual_periods(dates, len(nav))
    out = dict(
        CAGR=cagr(nav, dates=dates),
        Sharpe=sharpe(metric_rets, annual_periods=ann),
        Sortino=sortino(metric_rets, annual_periods=ann),
        MaxDD=max_drawdown(nav),
        Calmar=calmar(nav, dates=dates),
        Volatility=float(np.std(metric_rets) * np.sqrt(ann)),
        FinalNAV=float(nav[-1]),
    )
    if turnover_values is not None:
        values = np.asarray(turnover_values, dtype=float)
        out["Turnover"] = float(values.mean()) if values.size else 0.0
    elif weights is not None:
        out["Turnover"] = turnover(np.asarray(weights))
    else:
        out["Turnover"] = 0.0
    return out


def moving_average(x, window: int = 5) -> np.ndarray:
    """Basit hareketli ortalama (PDF §9.7 'moving average return' — ogrenme egilimi).
    Pencere diziden buyukse pencere dizi boyuna kisilir; bos dizi bos doner."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    w = max(1, min(int(window), x.size))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def training_diagnostics(curve, reward_terms_history=None, ma_window: int = 5) -> dict:
    """PDF §9.7 toplu egitim metrikleri.

    curve: egitim telemetri kayitlari (dict listesi ya da DataFrame; 'reward',
    'success', 'steps' alanlari beklenir). reward_terms_history: test backtest'inin
    adim-adim odul terimleri (iflas/tx-cost/dusus-ceza sayimi icin).
    Tum ciktilar gozlemseldir; egitimi etkilemez.
    """
    df = curve if isinstance(curve, pd.DataFrame) else pd.DataFrame(curve)
    out: dict = {"episodes": int(len(df))}
    if "reward" in df.columns and len(df):
        r = df["reward"].to_numpy(dtype=float)
        out["mean_return"] = float(np.mean(r))
        ma = moving_average(r, ma_window)
        out["ma_return_last"] = float(ma[-1]) if ma.size else 0.0
    if "success" in df.columns and len(df):
        out["success_rate"] = float(np.mean(df["success"].to_numpy(dtype=float)))
    if "steps" in df.columns and len(df):
        out["avg_steps"] = float(np.mean(df["steps"].to_numpy(dtype=float)))
    if reward_terms_history:
        out["bankrupt_count"] = int(sum(1 for rt in reward_terms_history if rt.get("bankrupt")))
        out["drawdown_penalty_steps"] = int(
            sum(1 for rt in reward_terms_history if float(rt.get("drawdown_penalty", 0.0)) > 0))
        tx = [float(rt.get("tx_cost", 0.0)) for rt in reward_terms_history]
        out["mean_tx_cost"] = float(np.mean(tx)) if tx else 0.0
    return out


def success_vs_benchmark(nav_agent: np.ndarray, nav_bench: np.ndarray) -> int:
    """Prompt başarı tanımı: episod sonunda ajan NAV'ı benchmark NAV'ını geçtiyse 1, aksi halde 0."""
    if len(nav_agent) == 0 or len(nav_bench) == 0:
        return 0
    m = min(len(nav_agent), len(nav_bench))
    return int(nav_agent[m - 1] >= nav_bench[m - 1])


def real_nav(nav: np.ndarray, usdtry: np.ndarray) -> np.ndarray:
    """Reel (USD-bazli) NAV — nominal TL NAV'ini USD/TRY ile deflate eder (lira illuzyonu, §9.9).

    Hoca: "baslangic paranı o yilin degerine gore normalize et." Nominal NAV TL
    cinsindendir; test doneminde (2022-2024) TL hizla deger kaybettiginden nominal
    kazanc satin-alma gucunu abartir ("lira illuzyonu"). Reel NAV baslangic USD/TRY'ye
    normalize eder:  real_t = (nav_t / nav_0) / (usdtry_t / usdtry_0).
    Baslangicta 1.0'dan baslar -> nominal NAV ile dogrudan kiyaslanabilir; TL deger
    kaybettikce reel NAV nominalin altinda kalir.

    GOZLEMSEL (yalniz raporlama) — egitim/eval/odul sayisal yolunu DEGISTIRMEZ; golden
    metrikleri etkilenmez. Ajanlar-arasi GORELI siralama para biriminden bagimsizdir
    (hepsi ayni TL evreni) -> reel donusum sirayi degistirmez, mutlak yorumu duzeltir.
    """
    nav = np.asarray(nav, dtype=float)
    fx = np.asarray(usdtry, dtype=float)
    if nav.size == 0 or fx.size == 0:
        return nav.copy()
    m = min(len(nav), len(fx))
    nav, fx = nav[:m], fx[:m]
    nav0 = nav[0] if abs(nav[0]) > 1e-12 else 1e-12
    fx0 = fx[0] if abs(fx[0]) > 1e-12 else 1e-12
    return (nav / nav0) / (fx / fx0)

```

---

## `utils/baselines.py`

```python
"""Klasik baseline stratejiler — karşılaştırma için.

İŞLEM MALİYETİ SİMETRİSİ (C2):
RL ortamı her rebalansta `eta·||Δw||₁` işlem maliyeti düşer (env/reward.py →
tx_cost = eta_t * delta_w_l1). Baseline'lar maliyetsiz olursa ASİMETRİK bir
karşılaştırma çıkar (baseline yapay olarak avantajlı). Bu yüzden tüm
YENİDEN-DENGELEYEN baseline'lar (equal_weight, mean_variance, risk_parity,
inverse_volatility, min_variance, momentum) RL ile SİMETRİK işlem maliyeti
düşer: her rebalans gününde günlük net getiri = port_r − eta·||w_yeni−w_eski||₁.

eta KAYNAĞI: RL'in kanonik ortamıyla AYNI base oran —
`config.HORIZON_PRESETS["medium"]["eta"]` (kanonik vade = medium, eta_base=0.0010).
RL adaptif ölçekleme (turnover_ewma/turnover_target) uygular; baseline'lar
deterministik kalsın diye SABIT base eta kullanır (RNG yok, golden-uyumlu).
Fonksiyon imzaları geriye-uyumlu: eta opsiyonel, default = medium preset eta.

buy_and_hold_index rebalans yapmaz → maliyet 0 (değişmez).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HORIZON_PRESETS, EnvConfig, cash_daily_rate

# RL kanonik ortamıyla aynı base işlem-maliyeti oranı (vade = medium).
DEFAULT_ETA: float = float(HORIZON_PRESETS["medium"]["eta"])


def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = float(w.sum())
    return w / total if total > 1e-12 else np.full(len(w), 1.0 / len(w))


def _turnover_from_holdings(target: np.ndarray, held: np.ndarray) -> float:
    """L1 turnover including the implicit cash slot."""
    target = _normalize(target)
    held = np.asarray(held, dtype=float)
    return float(np.abs(target - held).sum() + abs((1.0 - target.sum()) - (1.0 - held.sum())))


def _simulate(prices: pd.DataFrame, target_fn, *, eta: float,
              rebalance_fn) -> dict:
    """Self-financing long-only simulator shared by all risky baselines."""
    r = prices.pct_change().fillna(0.0).to_numpy(dtype=float)
    T, N = r.shape
    nav = np.empty(T, dtype=float)
    rets = np.empty(T, dtype=float)
    weights = np.empty((T, N), dtype=float)
    target_hist = np.empty((T, N), dtype=float)
    turnover = np.zeros(T, dtype=float)
    held = np.zeros(N, dtype=float)  # all cash before the first allocation
    wealth = 1.0
    pending_initial_cost = 0.0
    for t in range(T):
        if rebalance_fn(t):
            target = _normalize(target_fn(t, r, held.copy()))
            turn = _turnover_from_holdings(target, held)
        else:
            target = held.copy()
            turn = 0.0
        if t == 0:
            # Report the common pre-trade starting wealth. Initial allocation
            # cost is realized together with the first investable period.
            pending_initial_cost = float(eta) * turn
            gross = net = 0.0
        else:
            gross = float(target @ r[t])
            net = gross - float(eta) * turn - pending_initial_cost
            pending_initial_cost = 0.0
            wealth = max(0.0, wealth * (1.0 + net))
        grown = target * (1.0 + (r[t] if t > 0 else 0.0))
        held = _normalize(grown) if float(grown.sum()) > 1e-12 else target
        nav[t], rets[t], weights[t], target_hist[t], turnover[t] = \
            wealth, net, held, target, turn
    return {"nav": nav, "rets": rets, "weights": weights,
            "target_weights": target_hist, "turnover": turnover,
            "dates": list(prices.index)}


def equal_weight(prices: pd.DataFrame, eta: float = DEFAULT_ETA) -> dict:
    """Günlük rebalansla eşit ağırlık — RL ile simetrik işlem maliyeti düşülür.

    Her gün eşit-ağırlığa geri dönülür (drift düzeltme). Önceki günkü fiyat
    hareketiyle kayan ağırlıklar (w_drift) tekrar 1/N'e çekilir; bu rebalansın
    ||Δw||₁'i kadar `eta` maliyeti net getiriden düşülür. Drift küçük olduğundan
    maliyet de küçüktür ama 0 değildir (RL ile simetri)."""
    n = prices.shape[1]
    target = np.full(n, 1.0 / n)
    return _simulate(prices, lambda t, r, w: target, eta=eta,
                     rebalance_fn=lambda t: True)


def buy_and_hold_index(prices: pd.DataFrame, eta: float = DEFAULT_ETA) -> dict:
    """Eşit ağırlık alıp tut; yalnız ilk alım işlem maliyeti doğurur."""
    n = prices.shape[1]
    target = np.full(n, 1.0 / n)
    return _simulate(prices, lambda t, r, w: target, eta=eta,
                     rebalance_fn=lambda t: t == 0)


def mean_variance(prices: pd.DataFrame, lookback: int = 120,
                  rebalance: int = 20, risk_aversion: float = 5.0,
                  eta: float = DEFAULT_ETA) -> dict:
    """Kısıtlı long-only Markowitz; yuvarlanan pencere + periyodik rebalans.

    RL ile simetrik: her rebalans gününde ||w_yeni−w_eski||₁ kadar `eta`
    işlem maliyeti net getiriden düşülür."""
    from scipy.optimize import minimize

    n = prices.shape[1]
    equal = np.full(n, 1.0 / n)

    def target_fn(t, r, held):
        if t < lookback:
            return equal
        hist = r[t - lookback:t]
        mu = hist.mean(axis=0)
        cov = np.cov(hist.T) + 1e-5 * np.eye(n)

        def obj(w_, mu=mu, cov=cov, ra=risk_aversion):
            return -(w_ @ mu) + 0.5 * ra * w_ @ cov @ w_

        res = minimize(obj, equal, bounds=[(0, 0.2)] * n,
                       constraints=({"type": "eq", "fun": lambda w_: w_.sum() - 1}))
        return res.x if res.success and np.isfinite(res.x).all() else equal

    return _simulate(
        prices, target_fn, eta=eta,
        rebalance_fn=lambda t: t == 0 or (t >= lookback and (t - lookback) % rebalance == 0),
    )


# =====================================================================
# Ek akademik baseline'lar — hepsi DETERMINISTIK (RNG YOK), long-only,
# tam-yatirim (sum(w)=1). PARS sibling strategies.py'den odunc alindi ve
# kod/ API'sine ({"nav", "rets", "weights"} numpy) uyarlandi. Ortak
# simulasyon cekirdegi _rolling_backtest: yuvarlanan pencere + periyodik
# rebalans, ilk lookback gunu esit-agirlik isinma (warm-up).
# =====================================================================
def _rolling_backtest(prices: pd.DataFrame, weight_fn, lookback: int,
                      rebalance: int, eta: float = DEFAULT_ETA) -> dict:
    """Determinist yuvarlanan-pencere backtest cekirdegi (RL-simetrik tx-cost).

    weight_fn(hist_returns:(L,N), w_prev:(N,)) -> w_new:(N,) dondurur.
    Donen agirliklar long-only normalize edilir (clip>=0, toplam=1).
    Her rebalans gununde RL ile SIMETRIK islem maliyeti dusulur:
        net_r = port_r - eta * ||w_yeni - w_eski||_1
    eta = config.HORIZON_PRESETS['medium']['eta'] (RL kanonik base oran).
    Hicbir np.random cagrisi yok -> golden RNG sirasi etkilenmez (determinist)."""
    n = prices.shape[1]
    equal = np.full(n, 1.0 / n)

    def target_fn(t, r, held):
        if t < lookback:
            return equal
        candidate = np.asarray(weight_fn(r[t - lookback:t], held), dtype=float)
        return candidate if candidate.shape == (n,) and np.isfinite(candidate).all() else equal

    return _simulate(
        prices, target_fn, eta=eta,
        rebalance_fn=lambda t: t == 0 or (t >= lookback and (t - lookback) % rebalance == 0),
    )


def inverse_volatility(prices: pd.DataFrame, lookback: int = 60,
                       rebalance: int = 20, eta: float = DEFAULT_ETA) -> dict:
    """Ters-volatilite (1/sigma) agirligi — yuvarlanan vol uzerinden.

    Mantik: w_i ∝ 1/sigma_i (sigma_i = pencere getiri std'i), normalize edilir.
    Riske gore dengeleme'nin (risk parity) kovaryanssiz, naif halidir: yalniz
    kosegen (varyans) bilgisi kullanilir, korelasyon yok sayilir. Dusuk-vol
    hisseye daha cok agirlik -> portfoy vol'unu duzler. 'Volatility tilt'
    olarak bilinen iyi-belgelenmis defensif bir anomalidir (Asness ve dig.)."""
    def f(hist, w):
        iv = 1.0 / (hist.std(axis=0) + 1e-9)
        return iv / iv.sum()
    return _rolling_backtest(prices, f, lookback, rebalance, eta)


def risk_parity(prices: pd.DataFrame, lookback: int = 120,
                rebalance: int = 20, n_iter: int = 100,
                eta: float = DEFAULT_ETA) -> dict:
    """Esit risk katkisi (ERC / risk parity) — iteratif sabit-nokta.

    Hedef: her varligin TOPLAM portfoy riskine katkisi esit olsun:
        RC_i = w_i (Sigma w)_i ,  hedef RC_i = sigma_p^2 / N  (tum i icin esit).
    Cozum, marjinal risk katkisi MRC = Sigma w ile sabit-nokta iterasyonu:
        w_i <- 1 / MRC_i ,  ardindan normalize (Spinu/Maillard tarzi). Yakinsayinca
    w_i * MRC_i sabittir -> esit risk katkisi. Inverse-vol'un aksine korelasyonu
    (tam Sigma) hesaba katar; konsantrasyonu cesitlendirip kuyruk riskini azaltir.
    Tamamen determinist (sabit baslangic w=1/N, np.random yok)."""
    def f(hist, w):
        cov = np.cov(hist.T) + 1e-6 * np.eye(hist.shape[1])
        n = cov.shape[0]
        wv = np.ones(n) / n
        for _ in range(n_iter):
            mrc = cov @ wv
            wv = 1.0 / np.maximum(mrc, 1e-8)
            wv /= wv.sum()
        return wv
    return _rolling_backtest(prices, f, lookback, rebalance, eta)


def min_variance(prices: pd.DataFrame, lookback: int = 120,
                 rebalance: int = 20, max_weight: float = 0.4,
                 eta: float = DEFAULT_ETA) -> dict:
    """Saf minimum-varyans portfoyu — getiri tahmini KULLANMAZ.

    Cozulen problem:  min_w  w' Sigma w   s.t.  sum(w)=1, 0<=w_i<=max_weight.
    mean_variance'tan farki: beklenen getiri (mu) terimi YOK -> yalniz risk
    minimize edilir. Ortalama getiri tahmini en gurultulu girdidir (mu'nun
    ornekleme hatasi devasadir); onu atip yalniz kovaryansa guvenmek genelde
    daha kararli, dusuk-vol portfoy verir (DeMiguel ve dig. bulgusu: min-var
    cogu zaman tahmini-getiri optimizasyonunu out-of-sample yener). SLSQP
    determinist baslar (w0=1/N), RNG yok."""
    from scipy.optimize import minimize as _min
    def f(hist, w):
        n = hist.shape[1]
        cov = np.cov(hist.T) + 1e-5 * np.eye(n)
        res = _min(lambda x: x @ cov @ x, np.ones(n) / n,
                   bounds=[(0, max_weight)] * n,
                   constraints=({"type": "eq", "fun": lambda x: x.sum() - 1}))
        return res.x if res.success and np.isfinite(res.x).all() else np.ones(n) / n
    return _rolling_backtest(prices, f, lookback, rebalance, eta)


def momentum(prices: pd.DataFrame, lookback: int = 60, rebalance: int = 20,
             top_k: int = 5, eta: float = DEFAULT_ETA) -> dict:
    """Kesitsel momentum — pencere getirisi en yuksek top_k hisseye esit agirlik.

    Mantik: lookback penceresinde kumulatif getiri  cum_i = prod(1+r) - 1
    hesaplanir; en yuksek top_k hisse secilip esit (1/top_k) agirlik verilir.
    Klasik 'kazananlari al' (Jegadeesh & Titman 1993) etkisi: gecmis galipler
    kisa-orta vadede ortalama ustu getirmeye egilimlidir. lookback vade
    penceresine baglanir (kisa=5, orta=20, uzun=60 gun); secim determinist
    (np.argsort kararli) -> RNG yok."""
    k = max(1, int(top_k))
    def f(hist, w):
        n = hist.shape[1]
        cum = (1.0 + hist).prod(axis=0) - 1.0
        idx = np.argsort(cum)[-k:]          # en yuksek k (kararli siralama)
        out = np.zeros(n)
        out[idx] = 1.0 / len(idx)
        return out
    return _rolling_backtest(prices, f, lookback, rebalance, eta)


def cash_riskfree(prices: pd.DataFrame,
                  daily_rf: float | None = None) -> dict:
    """Nakit / risk-free benchmark — tum sermaye nakitte, sabit gunluk getiri.

    En muhafazakar referans: piyasaya HIC maruz kalmadan elde edilen getiri.
    daily_rf default'u RL ortamiyla AYNI gercek risksiz faizden gelir:
        cash_daily_rate(EnvConfig.cash_annual_rate)  (~%40 yillik -> ~0.001336/gun)
    -> NAV = (1+rf)^t ~ 1.40x/yil (252 gun). Boylece nakit baseline'i de RL'in
    nakit varligiyla SIMETRIK risksiz getiri kazanir (eski NAV=1.0 / %0 degil).
    daily_rf acikca verilirse o kullanilir (geriye-uyumlu override). Risk
    varliklarina agirlik 0 -> weights tamamen sifirdir (Turnover=0).
    'RL piyasaya girmeyi hak ediyor mu?' sorusunun tabanini olusturur: bir ajan
    bu (artik egimi pozitif) cizgiyi risk-ayarli gecemiyorsa piyasa riskini
    almak bosunadir. Tamamen determinist (kapali-form, np.random yok)."""
    if daily_rf is None:
        daily_rf = cash_daily_rate(EnvConfig.cash_annual_rate, EnvConfig.trading_days)
    T, N = prices.shape
    rets = np.full(T, float(daily_rf))
    if T:
        rets[0] = 0.0
    nav = np.cumprod(1.0 + rets)
    weights = np.zeros((T, N))
    return dict(nav=nav, rets=rets, weights=weights,
                target_weights=weights.copy(), turnover=np.zeros(T),
                dates=list(prices.index))

```

---

## `utils/deflated_sharpe.py`

```python
"""Deflated / Probabilistic Sharpe Ratio + CSCV Backtest-Overfitting Olasılığı (PBO).

Kaynaklar: Bailey & López de Prado (2014) "The Deflated Sharpe Ratio" (SSRN 2460551);
Bailey, Borwein, López de Prado & Zhu (2017) "The Probability of Backtest Overfitting"
(SSRN 2326253). Sharpe girdileri GÖZLEM-BAŞINA (ör. günlük).

PARS referans ağacından (`Reinforcement Learning Final/generalization/deflated_sharpe.py`)
kanonik BIST ağacına port edildi. scipy.stats.norm zaten projede bir bağımlılık
(env/reward.py) → eski `np.math` fallback'ları kaldırıldı (numpy 2.0-güvenli).

GÖZLEMSEL / golden-güvenli: yalnız getiri dizileri üzerinde saf istatistik —
eğitim/değerlendirme/ödül sayısal yolunu DEĞİŞTİRMEZ (golden-master 1e-6 korunur).
"""
from __future__ import annotations

from itertools import combinations
from math import e

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(sr: float, n_obs: int, skew: float = 0.0,
                               kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """PSR: tahmin hatası + çarpıklık/basıklık altında P(gerçek SR > benchmark).

    sr ve sr_benchmark GÖZLEM-BAŞINA Sharpe oranlarıdır.
    """
    if n_obs < 2:
        return 0.5
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """n_trials bağımsız deneme altında H0'da beklenen MAKSİMUM (gözlem-başı) Sharpe.

    E[max SR] ≈ sqrt(Var_SR) · [ (1−γ)·Z⁻¹(1−1/N) + γ·Z⁻¹(1−1/(N·e)) ].
    """
    n_trials = max(int(n_trials), 1)
    if n_trials == 1 or sr_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(sr_variance) * ((1 - g) * z1 + g * z2))


def deflated_sharpe_ratio(sr: float, n_obs: int, n_trials: int, sr_variance: float,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """DSR = beklenen-maksimum-Sharpe benchmark'ına karşı değerlendirilen PSR.

    DSR > 0.95 → gözlenen Sharpe'ın çoklu-deneme (multiple-testing) tesadüfü olması
    olası değil. Tüm Sharpe büyüklükleri gözlem-başınadır.
    """
    sr_star = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_benchmark=sr_star)


def cscv_pbo(perf_matrix: np.ndarray, n_splits: int = 10) -> dict:
    """Kombinatoryal-Simetrik Çapraz-Doğrulama ile Backtest-Overfitting Olasılığı.

    perf_matrix: (T_gözlem, N_config) her aday config için periyot-başı getiri.
    Zaman ``n_splits`` bloğa bölünür; her dengeli train/test bölüşümünde IS-en-iyi
    config seçilir ve OOS sırası → logit kaydedilir. PBO = IS-en-iyi config'in
    OOS-medyan-altı olduğu bölüşüm oranı. {pbo, n_combos, mean_logit} döner.
    """
    R = np.asarray(perf_matrix, dtype=float)
    T, N = R.shape
    if N < 2 or T < n_splits * 2:
        return {"pbo": float("nan"), "n_combos": 0, "mean_logit": float("nan")}
    S = n_splits if n_splits % 2 == 0 else n_splits - 1
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for train_idx in combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in train_idx])
        te = np.concatenate([blocks[i] for i in range(S) if i not in train_idx])
        is_sr = R[tr].mean(0) / (R[tr].std(0) + 1e-12)
        oos_sr = R[te].mean(0) / (R[te].std(0) + 1e-12)
        n_star = int(np.argmax(is_sr))
        order = np.argsort(oos_sr)                      # 1=en kötü .. N=en iyi OOS
        rank = int(np.nonzero(order == n_star)[0][0]) + 1
        w = min(max(rank / (N + 1), 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    logits = np.asarray(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combos": int(len(logits)),
            "mean_logit": float(logits.mean())}

```

---

## `utils/stress_mc.py`

```python
"""Monte-Carlo stres — şişman-kuyruklu parametrik + durağan blok bootstrap.

İleri çok-varlık getiri patikalarını iki simülatörle üretip portföy ağırlıklarıyla
terminal-getiri dağılımına ve kuyruk riskine (VaR/CVaR) taşır:
  * Student-t parametrik — çok-değişkenli t (tarihsel ortalama/kovaryans), ağır kuyruk;
  * durağan blok bootstrap — ardışık tarihsel blokları yeniden örnekler; dağılım
    varsayımı olmadan otokorelasyon/volatilite kümelenmesini KORUR (Politis & Romano 1994).

PARS referans ağacından (`Reinforcement Learning Final/stress/montecarlo.py`) kanonik
BIST ağacına port edildi. Env-yerel `np.random.default_rng(seed)` → global RNG'ye dokunmaz.
GÖZLEMSEL / golden-güvenli: mevcut backtest çıktıları üzerinde post-hoc stres analizi.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _assets_weights(weights, n_assets):
    """Ağırlık vektörü n+1 (nakit dahil) ise nakit elemanını düşürür → (n,)."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] == n_assets + 1:
        w = w[:n_assets]
    return w


def summarize_mc(terminal_returns: np.ndarray, var_alpha: float = 0.05,
                 es_alpha: float = 0.025) -> Dict[str, float]:
    """Terminal getiri dağılımının özet istatistikleri + kuyruk riski (VaR/CVaR)."""
    tr = np.asarray(terminal_returns, dtype=float)
    q_var = np.quantile(tr, var_alpha)
    q_es = np.quantile(tr, es_alpha)
    tail = tr[tr <= q_es]
    return {
        "mean": float(tr.mean()), "median": float(np.median(tr)),
        "p05": float(np.quantile(tr, 0.05)), "p95": float(np.quantile(tr, 0.95)),
        "VaR": float(max(0.0, -q_var)),
        "CVaR": float(max(0.0, -(tail.mean() if tail.size else q_es))),
        "worst": float(tr.min()), "best": float(tr.max()),
        "prob_loss": float((tr < 0).mean()),
        "prob_loss_20pct": float((tr < -0.20).mean()),
    }


def mc_student_t(asset_returns: np.ndarray, weights: np.ndarray, horizon: int = 252,
                 paths: int = 5000, dof: float = 5.0, seed: int = 42) -> dict:
    """Çok-değişkenli Student-t MC (tarihsel ortalama/kovaryans, ağır kuyruk)."""
    R = np.asarray(asset_returns, dtype=float)
    n = R.shape[1]
    w = _assets_weights(weights, n)
    rng = np.random.default_rng(seed)
    mu = R.mean(0)
    cov = np.cov(R.T) + 1e-8 * np.eye(n)
    L = np.linalg.cholesky(cov)
    scale = np.sqrt((dof - 2) / dof) if dof > 2 else 1.0
    terminal = np.empty(paths)
    for p in range(paths):
        z = rng.standard_normal((horizon, n))
        g = rng.chisquare(dof, size=(horizon, 1)) / dof
        t = z / np.sqrt(g)                               # standart çok-değişkenli-t yenilik
        daily = mu + (t * scale) @ L.T
        port = daily @ w
        terminal[p] = np.prod(1 + port) - 1
    return {"terminal": terminal, "summary": summarize_mc(terminal)}


def mc_block_bootstrap(asset_returns: np.ndarray, weights: np.ndarray, horizon: int = 252,
                       paths: int = 5000, block: int = 20, seed: int = 42) -> dict:
    """Tarihsel getirilerin durağan blok bootstrap'ı → terminal dağılım."""
    R = np.asarray(asset_returns, dtype=float)
    T, n = R.shape
    w = _assets_weights(weights, n)
    rng = np.random.default_rng(seed)
    pblock = 1.0 / block
    terminal = np.empty(paths)
    for p in range(paths):
        idx = np.empty(horizon, dtype=int)
        i = rng.integers(0, T)
        for h in range(horizon):
            if h > 0 and rng.random() < pblock:
                i = rng.integers(0, T)                   # yeni blok başlat
            idx[h] = i
            i = (i + 1) % T
        port = R[idx] @ w
        terminal[p] = np.prod(1 + port) - 1
    return {"terminal": terminal, "summary": summarize_mc(terminal)}

```

---

## `utils/portfolio_tl.py`

```python
"""TL cinsinden portföy telemetrisi — env'in NAV ve ağırlık çıktılarını
gerçek TL bakiyesine, lot sayısına, alım-satım logu'na ve holding-days'e çevirir.

Env'i değiştirmeden, tamamen UI katmanında türetim yapar.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

HOLDING_EPS = 0.005  # w[i] >= bu eşik ise "tutuluyor" say


def compute_tl_step(nav: float, w_now: np.ndarray, w_prev: np.ndarray,
                    prices_t: np.ndarray, initial_capital: float,
                    prev_portfolio_tl: float, holding_days_prev: np.ndarray,
                    tx_cost_rate: float = 0.0,
                    target_weights: np.ndarray | None = None,
                    holding_period_days: int = 1,
                    eps: float = HOLDING_EPS) -> Dict[str, np.ndarray | float]:
    """Tek adımın TL türevlerini hesapla.

    Args:
        nav: env.nav (scalar)
        w_now: adım sonu ağırlık vektörü (N+1,)
        w_prev: adım öncesi ağırlık (N+1,)
        prices_t: o adımdaki risk'li varlık fiyat vektörü (N,)
        initial_capital: kullanıcının girdiği TL bakiyesi
        prev_portfolio_tl: önceki adımın portföy TL'si (step P&L için)
        holding_days_prev: (N,) — önceki adım sonu holding gün sayıları
        tx_cost_rate: env'in bu adım için uyguladığı tx_cost oranı (reward_terms["tx_cost"])
        eps: holding sayımı için ağırlık eşiği
    """
    w_now = np.asarray(w_now, dtype=np.float64).reshape(-1)
    w_prev = np.asarray(w_prev, dtype=np.float64).reshape(-1)
    prices_t = np.asarray(prices_t, dtype=np.float64).reshape(-1)
    safe_px = np.maximum(prices_t, 1e-9)

    N_risky = prices_t.shape[0]
    portfolio_tl = float(initial_capital) * float(nav)
    cash_tl      = portfolio_tl * float(w_now[-1])
    asset_tl     = portfolio_tl * w_now[:N_risky]
    shares       = asset_tl / safe_px
    trade_w = (w_now if target_weights is None
               else np.asarray(target_weights, dtype=np.float64).reshape(-1))
    trade_tl     = portfolio_tl * (trade_w[:N_risky] - w_prev[:N_risky])
    trade_shares = trade_tl / safe_px
    commission_tl = portfolio_tl * float(tx_cost_rate)

    holding_days_prev = np.asarray(holding_days_prev, dtype=np.int64).reshape(-1)
    holding_days = np.where(w_now[:N_risky] >= eps,
                            holding_days_prev + max(1, int(holding_period_days)), 0)

    return dict(
        portfolio_tl=portfolio_tl,
        cash_tl=cash_tl,
        asset_tl=asset_tl,
        shares=shares,
        trade_tl=trade_tl,
        trade_shares=trade_shares,
        commission_tl=commission_tl,
        holding_days=holding_days,
        step_pnl_tl=portfolio_tl - float(prev_portfolio_tl),
        cum_pnl_tl=portfolio_tl - float(initial_capital),
        w_now=w_now.copy(),
    )


def compute_tl_series(nav_hist: List[float],
                      weight_hist: List[np.ndarray],
                      prices_matrix: np.ndarray,
                      initial_capital: float,
                      tx_cost_rates: List[float] | None = None,
                      t_start: int = 0) -> List[Dict]:
    """Bir episod/trace için adım-adım TL dict listesi döner.

    Args:
        nav_hist: [nav_0=1.0, nav_1, nav_2, ...] — env.nav_history (len = steps+1)
        weight_hist: [w_0, w_1, ...] — env.weight_history (len = steps+1)
        prices_matrix: shape (T, N_risky) — env.prices tamamı
        initial_capital: başlangıç TL bakiyesi
        tx_cost_rates: her adımın tx_cost oranı listesi (len = steps); None ise 0
        t_start: weight_hist[i]'ye karşılık gelen prices_matrix indeks başlangıcı
                 (env'de env.reset sonrası t = max(window,21), step_count=0).
                 weight_hist[0] reset sonrası w, weight_hist[k] env.t = t_start+k
                 sonrasındaki ağırlık.
    """
    steps = len(nav_hist) - 1
    N_risky = prices_matrix.shape[1]
    if tx_cost_rates is None:
        tx_cost_rates = [0.0] * steps
    rows: List[Dict] = []
    holding_days_prev = np.zeros(N_risky, dtype=np.int64)
    prev_portfolio_tl = float(initial_capital) * float(nav_hist[0])
    for k in range(1, steps + 1):
        t_idx = min(t_start + k, prices_matrix.shape[0] - 1)
        snap = compute_tl_step(
            nav=nav_hist[k],
            w_now=weight_hist[k],
            w_prev=weight_hist[k - 1],
            prices_t=prices_matrix[t_idx],
            initial_capital=initial_capital,
            prev_portfolio_tl=prev_portfolio_tl,
            holding_days_prev=holding_days_prev,
            tx_cost_rate=tx_cost_rates[k - 1],
        )
        rows.append(snap)
        holding_days_prev = snap["holding_days"]
        prev_portfolio_tl = snap["portfolio_tl"]
    return rows


def build_portfolio_table(tickers: List[str], snap: Dict,
                          include_cash: bool = True,
                          weight_threshold: float = HOLDING_EPS,
                          w_prev: np.ndarray | None = None) -> pd.DataFrame:
    """Adım anının portföy panosu: hisse, ağırlık, lot, TL değeri, holding days,
    bu adım Δ TL ve Durum (YENİ/TUTULUYOR/ÇIKIŞ/NAKİT).

    snap: compute_tl_step / compute_tl_series elemanı + 'w' (mevcut ağırlık vektörü).
    w_prev: önceki adımın ağırlık vektörü (Durum sütunu için gerekli).
    """
    w_raw = snap.get("w_now")
    if w_raw is None:
        # Fallback: asset_tl ve cash_tl'den ağırlık tahmin et
        tot = snap["portfolio_tl"] or 1.0
        w_risky = snap["asset_tl"] / tot
        w_cash = snap["cash_tl"] / tot
        w = np.concatenate([w_risky, [w_cash]])
    else:
        w = np.asarray(w_raw).reshape(-1)
    N_risky = len(snap["asset_tl"])
    w_prev_arr = (np.asarray(w_prev).reshape(-1) if w_prev is not None
                  else np.zeros_like(w))
    rows = []
    # Tutulan veya bu adımda çıkılan hisseler — hepsini göster
    for i in range(N_risky):
        weight = float(w[i])
        prev_w = float(w_prev_arr[i]) if i < len(w_prev_arr) else 0.0
        is_in  = weight >= weight_threshold
        was_in = prev_w >= weight_threshold
        if not (is_in or was_in):
            continue
        if is_in and not was_in:
            status = "YENİ"
        elif is_in and was_in:
            status = "TUTULUYOR"
        else:  # was_in and not is_in
            status = "ÇIKIŞ"
        rows.append({
            "Hisse": tickers[i],
            "Durum": status,
            "Ağırlık": round(weight, 4),
            "Lot": round(float(snap["shares"][i]), 3),
            "TL Değeri": round(float(snap["asset_tl"][i]), 2),
            "Tutuluyor (gün)": int(snap["holding_days"][i]),
            "Bu adım Δ TL": round(float(snap["trade_tl"][i]), 2),
        })
    # Durum önceliği: ÇIKIŞ → YENİ → TUTULUYOR, sonra ağırlığa göre
    status_order = {"ÇIKIŞ": 0, "YENİ": 1, "TUTULUYOR": 2}
    rows.sort(key=lambda r: (status_order.get(r["Durum"], 9), -r["Ağırlık"]))
    if include_cash:
        rows.append({
            "Hisse": "NAKİT",
            "Durum": "—",
            "Ağırlık": round(float(w[-1]), 4),
            "Lot": None,
            "TL Değeri": round(float(snap["cash_tl"]), 2),
            "Tutuluyor (gün)": None,
            "Bu adım Δ TL": None,
        })
    return pd.DataFrame(rows)


def build_cumulative_trade_log(tickers: List[str], tl_snaps: List[Dict],
                               dates: List[str],
                               threshold_tl: float = 1.0,
                               up_to_step: int | None = None) -> pd.DataFrame:
    """Episod/test başından `up_to_step` dahil (None ise hepsi) tüm al-sat kayıtları."""
    end_k = len(tl_snaps) if up_to_step is None else min(up_to_step + 1, len(tl_snaps))
    rows = []
    for k in range(end_k):
        snap = tl_snaps[k]
        trade_tl = np.asarray(snap["trade_tl"])
        trade_sh = np.asarray(snap["trade_shares"])
        for i in range(len(trade_tl)):
            tt = float(trade_tl[i])
            if abs(tt) < threshold_tl:
                continue
            rows.append({
                "Adım": k,
                "Tarih": dates[k] if k < len(dates) else "",
                "Hisse": tickers[i],
                "İşlem": "ALIŞ" if tt > 0 else "SATIŞ",
                "Lot": round(float(trade_sh[i]), 3),
                "TL": round(tt, 2),
            })
    return pd.DataFrame(rows)


def build_trade_log(tickers: List[str], snap: Dict,
                    threshold_tl: float = 1.0) -> pd.DataFrame:
    """Bu adımda yapılan alış/satışlar — |trade_tl| > eşik olanlar."""
    N_risky = len(snap["trade_tl"])
    rows = []
    for i in range(N_risky):
        tt = float(snap["trade_tl"][i])
        if abs(tt) < threshold_tl:
            continue
        lots = float(snap["trade_shares"][i])
        rows.append({
            "Hisse": tickers[i],
            "İşlem": "ALIŞ" if tt > 0 else "SATIŞ",
            "Lot": round(lots, 3),
            "TL": round(tt, 2),
        })
    rows.sort(key=lambda r: -abs(r["TL"]))
    return pd.DataFrame(rows)


def step_rows_for_training(curve_snaps: List[Dict], dates: np.ndarray,
                           action_names: List[str] | None = None,
                           action_indices: List[int] | None = None,
                           reward_terms_list: List[Dict] | None = None,
                           initial_capital: float = 0.0) -> pd.DataFrame:
    """Eğitim episodunun tüm adımları için tablo — 252 satır.

    Args:
        curve_snaps: compute_tl_series çıktısı
        dates: env.dates[t_start+1 : t_start+1+steps]
        action_names: adım başına aksiyon adı listesi (DQN) veya None
        action_indices: adım başına aksiyon idx (DQN) veya None
        reward_terms_list: adım başına reward_terms (delta_w_l1 için)
        initial_capital: yüzde hesabı için
    """
    rows = []
    for k, snap in enumerate(curve_snaps):
        action_name = ""
        if action_names and k < len(action_names):
            action_name = action_names[k]
        elif action_indices and k < len(action_indices):
            action_name = f"a={action_indices[k]}"
        delta_w = 0.0
        if reward_terms_list and k < len(reward_terms_list):
            delta_w = float(reward_terms_list[k].get("delta_w_l1", 0.0))
        held = int(np.sum(np.asarray(snap["holding_days"]) > 0))
        cum_pct = 100.0 * (snap["cum_pnl_tl"] / initial_capital) if initial_capital else 0.0
        dstr = str(pd.Timestamp(dates[k]).date()) if k < len(dates) else ""
        rows.append({
            "Gün #": k + 1,
            "Tarih": dstr,
            "Aksiyon": action_name,
            "Nakit TL": round(snap["cash_tl"], 2),
            "Portföy TL": round(snap["portfolio_tl"], 2),
            "Adım P&L": round(snap["step_pnl_tl"], 2),
            "Kümülatif P&L": round(snap["cum_pnl_tl"], 2),
            "Kümülatif %": round(cum_pct, 3),
            "Komisyon TL": round(snap["commission_tl"], 2),
            "Δ turnover": round(delta_w, 4),
            "Tutulan hisse": held,
        })
    return pd.DataFrame(rows)

```

---

## `utils/torch_utils.py`

```python
"""Notr torch yardimcilari - SOLID P1 (DIP).

set_seed / get_device onceden agents/common.py'deydi; forecast/forecaster.py
oradan import ediyordu (forecast -> agents ters bagimliligi). Govdeler buraya
BIREBIR tasindi; agents/common.py geriye-uyumluluk icin re-export eder.

DAVRANIS KORUNUR: fonksiyon govdesi ve cagri imzalari ayni - RNG tuketim sirasi
degismez. Golden-master regresyonu (tests/golden) bu esdegerligi dogrular.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Global RNG'leri (random, numpy, torch) seed'ler.

    Onceki ajan-ici `_set_seed` ile birebir ayni. Ajan ctor'unda, aglar
    olusturulmadan ONCE cagrilmali (agirlik init'i bu seed'e baglidir).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device: str | None = None) -> torch.device:
    """Verilen device ya da otomatik (cuda varsa cuda, yoksa cpu)."""
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
```

---

## `forecast/forecaster.py`

```python
"""CNN-LSTM bir-adim getiri tahmincisi — predict-then-optimize (Faz V4).

Train-only fit edilir; her (varlik, t) icin t+1 log-getirisini t'de biten
pencereden tahmin eder. Bu tahmin state'e 'forecast' feature'i olarak eklenir
(ajan t'de t+1 hakkinda bir ON-GORUYU gorur — gerçek gelecek degil, modelin
ciktisi). Literatur: hibrit LSTM->PPO (arXiv:2511.17963), predict-then-optimize
(arXiv:2501.17992).

SIZINTISIZLIK (kritik): tahmin penceresi yalniz <=t; model YALNIZ train'de fit
edilir, test'i hic gormez. Determinizm: set_seed(seed) ile fit tekrar-uretilebilir
(golden gecerli kalir). test_forecaster bunlari kilitler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.torch_utils import get_device, set_seed


class ReturnForecaster(nn.Module):
    """Conv1d -> LSTM -> Linear; (B, window, 1) -> (B,) bir-adim getiri tahmini."""

    def __init__(self, window: int = 20, conv_ch: int = 16, hidden: int = 32):
        super().__init__()
        self.conv = nn.Conv1d(1, conv_ch, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(conv_ch, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x.transpose(1, 2)))   # (B, conv_ch, window)
        out, _ = self.lstm(h.transpose(1, 2))          # (B, window, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)    # (B,)


def _make_windows(logret: np.ndarray, window: int):
    """logret (T,N) -> X (M,window,1), y (M,); ornek (t,j) icin y = logret[t+1,j]."""
    T, N = logret.shape
    xs, ys = [], []
    for j in range(N):
        col = logret[:, j]
        for t in range(window - 1, T - 1):
            xs.append(col[t - window + 1: t + 1])
            ys.append(col[t + 1])
    x = np.asarray(xs, dtype=np.float32)[:, :, None]
    y = np.asarray(ys, dtype=np.float32)
    return x, y


def _logret(prices: pd.DataFrame) -> np.ndarray:
    return np.log(prices).diff().fillna(0.0).values.astype(np.float32)


def build_forecast_feature(prices_full: pd.DataFrame, prices_train: pd.DataFrame,
                           window: int = 20, conv_ch: int = 16, hidden: int = 32,
                           epochs: int = 4, lr: float = 1e-3, batch: int = 256,
                           seed: int = 42) -> pd.DataFrame:
    """YALNIZ train'de fit; tüm seri için bir-adım getiri tahmini (T,N) döner.

    prices_train ile fit edilir (sizinti yok); prices_full uzerinde causal tahmin
    uretilir (her t icin pencere <=t). Donen DataFrame feats['forecast'] olur.
    """
    if window < 2:
        raise ValueError("forecast window en az 2 olmali")
    if len(prices_train) <= window:
        raise ValueError(
            f"Forecaster egitimi icin en az window+1={window + 1} satir gerekli; "
            f"{len(prices_train)} geldi")
    if list(prices_full.columns) != list(prices_train.columns):
        raise ValueError("prices_full ve prices_train kolonlari ayni olmali")
    if not prices_train.index.isin(prices_full.index).all():
        raise ValueError("prices_train prices_full'un alt kumesi olmali")
    for name, frame in (("prices_full", prices_full), ("prices_train", prices_train)):
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"{name} pozitif ve sonlu olmali")

    set_seed(seed)                          # fit'i tekrar-uretilebilir kil (golden)
    device = get_device()
    lr_tr = _logret(prices_train)
    lr_full = _logret(prices_full)

    x, y = _make_windows(lr_tr, window)
    model = ReturnForecaster(window, conv_ch, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.MSELoss()
    x_all = torch.as_tensor(x, device=device)
    y_all = torch.as_tensor(y, device=device)
    n = x_all.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            b = perm[s: s + batch]
            opt.zero_grad()
            loss_fn(model(x_all[b]), y_all[b]).backward()
            opt.step()

    # Causal tahmin: t'deki feature = t+1 ongorusu (pencere t'de biter).
    model.eval()
    T, N = lr_full.shape
    out = np.zeros((T, N), dtype=np.float32)
    if T > window - 1:
        idx = list(range(window - 1, T))
        with torch.no_grad():
            for j in range(N):
                col = lr_full[:, j]
                W = np.stack([col[t - window + 1: t + 1] for t in idx]).astype(np.float32)[:, :, None]
                pred = model(torch.as_tensor(W, device=device)).cpu().numpy()
                out[idx, j] = pred
    return pd.DataFrame(out, index=prices_full.index, columns=prices_full.columns)

```

---

## `env/portfolio_env.py`

```python
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

from math import ceil
from typing import Dict, Literal
import numpy as np
import pandas as pd

from config import DEFAULTS, EnvConfig, HORIZON_PRESETS, RewardConfig
from core.contracts import DataProvenance
from config import cash_daily_rate as _cash_daily_rate_from_annual
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
                 force_price_noise: bool = False,
                 episode_clean: bool = False,
                 rebalance_freq: int | None = None,
                 gamma: float | None = None,            # v12: None -> preset (golden); override
                 mom_window: int | None = None,         # v12: None -> preset; override
                 minvol_window: int | None = None,      # v12: None -> preset; override
                 step_days: int = DEFAULTS.step_days,
                 start_index: int | None = None,
                 cash_daily_rate: float | None = None,
                 w_dsr: float = RewardConfig.w_dsr,
                 dsr_eta: float = RewardConfig.dsr_eta,
                 w_cvar: float | None = None,
                 cvar_alpha: float = RewardConfig.cvar_alpha,
                 regime_beta: float = RewardConfig.regime_beta,
                 cvar_amp: float = RewardConfig.cvar_amp,
                 w_gain: float = 0.0,
                 gain_floor: float = 1.0,
                 w_gain_speed: float = 0.0,
                 w_ruin_timing: float = 0.0,
                 macro=None, regime=None):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices DatetimeIndex gerektirir")
        if prices.empty or prices.shape[1] == 0:
            raise ValueError("prices bos olamaz")
        if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
            raise ValueError("prices pozitif ve sonlu degerlerden olusmali")
        if any(not prices.index.equals(v.index) or list(prices.columns) != list(v.columns)
               for v in features.values()):
            raise ValueError("tum feature matrisleri prices ile ayni index/kolonlara sahip olmali")
        self.prices = prices.values.astype(np.float32)
        self.dates  = prices.index
        self.step_days = max(1, int(step_days))
        raw_counts = prices.attrs.get("session_counts")
        self.session_counts = (np.ones(len(prices), dtype=np.int32) if raw_counts is None
                               else np.asarray(raw_counts, dtype=np.int32))
        if self.session_counts.shape != (len(prices),) or np.any(self.session_counts < 1):
            raise ValueError("prices.attrs['session_counts'] gecersiz")
        self.provenance = DataProvenance.from_value(prices.attrs.get("provenance"))
        self.start_index = None if start_index is None else int(start_index)
        self.feat_names = list(features.keys())
        self.feat_tensor = np.stack(
            [features[k].values.astype(np.float32) for k in self.feat_names],
            axis=-1,
        )  # (T, N, F)

        self.horizon = horizon
        preset = HORIZON_PRESETS[horizon]
        # rebalance_freq: None -> preset (CLI/golden bit-ayni); UI override -> max(1, int).
        self.rebalance_freq = preset["rebalance"] if rebalance_freq is None else max(1, int(rebalance_freq))
        # v12: mom/minvol/gamma artik OVERRIDE-edilebilir (None -> preset, golden bit-ayni).
        mom_days = preset["mom_window"] if mom_window is None else max(1, int(mom_window))
        minvol_days = preset["minvol_window"] if minvol_window is None else max(1, int(minvol_window))
        self.mom_window = max(1, int(ceil(mom_days / self.step_days)))
        self.minvol_window = max(1, int(ceil(minvol_days / self.step_days)))
        self.mom_lookback_days = mom_days
        self.minvol_lookback_days = minvol_days
        self.gamma          = preset["gamma"]         if gamma         is None else float(gamma)
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
        # w_cvar base: None ise config default (golden-guvenli); aksi halde override.
        # Her iki halde vade-bagli cvar_factor ile carpilir (mevcut davranis korunur).
        w_cvar_base = RewardConfig.w_cvar if w_cvar is None else float(w_cvar)
        self.reward = RewardEngine(
            shaper=shaper,
            dsharpe=DifferentialSharpe(eta=dsr_eta),   # v2: cevrim-ici risk-ayar (DSR)
            w_dsr=float(w_dsr),
            # Ayarlanabilir iflas parametreleri — None ise modul-duzeyi default'lar
            bankruptcy_nav=float(bankruptcy_nav) if bankruptcy_nav is not None else BANKRUPTCY_NAV,
            bankruptcy_penalty=(float(bankruptcy_penalty)
                                if bankruptcy_penalty is not None else BANKRUPTCY_PENALTY),
            w_cvar=w_cvar_base * cvar_factor,           # v7: rejim-amplified kuyruk cezasi
            cvar_alpha=float(cvar_alpha),
            regime_beta=float(regime_beta),
            cvar_amp=float(cvar_amp),
            # v9: OPT-IN kazanc-carpani + iflas-timing (default 0/kapali -> golden bit-ayni)
            w_gain=float(w_gain), gain_floor=float(gain_floor),
            w_gain_speed=float(w_gain_speed), w_ruin_timing=float(w_ruin_timing),
        )

        self.N_assets = prices.shape[1]
        self.cash_asset = cash_asset
        self.N = self.N_assets + (1 if cash_asset else 0)
        self.n_days = prices.shape[0]

        # --- Kısa-veri adaptasyonu (granülerlik: monthly/yearly) ---
        # DAILY V11-EXACT: V11'de self.window = max(window, minvol_window) = max(20,60)=60
        # (DQN min-vol/momentum SABLONLARI minvol_window gecmisi gerektirir; lo=60 olmali,
        # aksi halde sablon dilimi BOS -> NaN). Bunu KORU; yalniz COARSE veride (n_days kucuk)
        # pencereleri veriye sigacak sekilde asagi cap'le.
        # KRITIK (adversarial bulgu): DiscreteEnv sablonlari feat_tensor[t-mom/minvol:t]
        # kullanir; window cap'lenip mom/minvol cap'lenmezse coarse'da NEGATIF/BOS slice ->
        # NaN agirlik/NAV. Bu yuzden mom_window/minvol_window'u DA cap'le (window'dan ONCE).
        _cap = max(1, self.n_days // 3)
        self.minvol_window = min(self.minvol_window, _cap)   # daily: min(60,851)=60 degismez
        self.mom_window    = min(self.mom_window, _cap)      # daily: min(20,851)=20 degismez
        # Günlük (n_days~2554): window=min(max(20,60),851)=60 -> lo=60 V11-AYNI.
        # Yıllık (~10): minvol=mom=3,window=3; Monthly (~120): minvol=40,mom=20,window=40.
        self.window = min(max(window, self.minvol_window), _cap)
        self.F = self.feat_tensor.shape[2]
        # v6: makro rejim blogu (z-skorlu (T,M), state'e eklenir) + HAM regime
        # (T,) ∈[-1,1] — V7 odul kriz-amplifikasyonu icin step()'te kullanilir.
        self.macro = None if macro is None else np.asarray(macro, dtype=np.float32)
        self.regime = None if regime is None else np.asarray(regime, dtype=np.float32)
        self.M = 0 if self.macro is None else int(self.macro.shape[1])
        self.state_dim = self.F * self.N_assets + self.M + self.N
        self.action_dim = self.N
        # n_days yukarida atandi (window cap hesabi icin ctor'da erken gerekiyordu).
        # max_steps V11-EXACT (CAP YOK): coarse veride episode dogal olarak done-at-data-end
        # ile biter (step t>=n_days-1 -> done; son gecerli _risky_returns t=n_days-2). Cap
        # eklemek daily eval'i 1 adim kisaltir -> golden kayardi; bu yuzden cap YOK.
        self.max_steps = max_steps
        # v2: env-yerel RNG — global np.random'a bagimli degil (tekrar-uretilebilirlik
        # kurulum sirasindan bagimsiz) + tohumlu rastgele-baslangic destegi.
        self.random_start = bool(random_start)
        self.rng = np.random.default_rng(seed)
        # v8: fiyat gurultusu/slippage (hocanin sarti). train_only -> yalniz random_start
        # (egitim) acik; eval (random_start=False) -> kapali, golden eval determinizmi korunur.
        self.price_noise_std = float(price_noise_std)
        # force_price_noise: eval'de de gurultu acar (gurultu-artirimli coklu-episode
        # degerlendirmesi icin). Varsayilan False -> mevcut train-only davranis + golden korunur.
        self._noise_active = (self.price_noise_std > 0.0 and
                              (force_price_noise or
                               (self.random_start if price_noise_train_only else True)))
        # OPT-IN episode-clean (kullanici istegi: "1. iterasyon orijinal, 2-12 farkli noise").
        # _episode_idx: ctor'da -1; her reset()'te +1 -> 1. episode idx=0 (TEMIZ/orijinal),
        # idx>=1 gurultulu. DEFAULT KAPALI -> CLI/golden V11 davranisi (her episode gurultulu)
        # BIT-AYNI; yalniz UI episode_clean=True gecer. Kapaliyken sayac kullanilmaz -> golden-no-op.
        self._episode_clean = bool(episode_clean)
        self._episode_idx = -1
        # v10: nakit (risksiz) gunluk faiz. None -> config EnvConfig.cash_annual_rate'ten
        # bilesik turetilir; UI/CLI gunluk orani dogrudan gecebilir. SABIT skaler -> RNG
        # cagrisi YOK, _risky_returns'te 0.0 yerine bu oran nakit varliga atanir.
        self.cash_daily_rate = float(
            _cash_daily_rate_from_annual(EnvConfig.cash_annual_rate, EnvConfig.trading_days)
            if cash_daily_rate is None else cash_daily_rate
        )
        self._reset_state()

    def _reset_state(self):
        # lo = self.window (V11'de minvol_window'u kapsar, >=21). Coarse veride n_days-2'ye
        # cap'lenir (yearly ~10: lo<=8 -> gecerli; step done-at-data-end ile t+1 sinir guvenli).
        # Gunluk: self.window=60 -> min(60, 2552)=60 -> V11-AYNI.
        lo_raw = max(self.window, 21)
        lo = min(lo_raw, max(1, self.n_days - 2))
        if self.start_index is not None:
            lo = max(lo, self.start_index)
        # v12: dejenere (asiri-kisa) veri korumasi — sessiz NaN/bos-slice yerine acik hata.
        if self.n_days < 3 or lo >= self.n_days - 1:
            raise ValueError(
                f"Yetersiz veri: n_days={self.n_days}, warm-up lo={lo}. step_days cok buyuk "
                f"veya tarih araligi cok dar (resample sonrasi {self.n_days} nokta)."
            )
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
        self.episode_start_date = self.dates[self.t]
        self.step_count = 0
        self.w = np.zeros(self.N, dtype=np.float32)
        self.w[-1] = 1.0  # nakitle başla
        self.nav = 1.0
        self.peak = 1.0
        self.nav_history = [1.0]
        self.weight_history = [self.w.copy()]
        self.ret_history = []
        self.reward_terms_history = []
        self.weights_before_history = []
        self.target_weight_history = []
        self.turnover_history = []
        self.period_length_history = []
        self.discount_history = []
        self.date_history = []
        self.reward.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)   # env-yerel; global RNG'ye dokunmaz
        self._episode_idx += 1                        # 1. episode -> idx=0 (temiz); >=1 -> noise
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
        # ROBUSTLUK (NaN-guvenligi): gercek BIST verisinde eksik/halt gunleri NaN birakabilir.
        # Tek bir NaN getiri, w*r agirlikli toplamini (0*nan=nan dahil) zehirleyip NAV'i TUM
        # episod boyunca nan yapardi (HF Space'te gozlemlenen hata). Eksik veriyi 'hareket yok'
        # (0 getiri) say -> NAV daima sonlu. Temiz veride no-op (sonlu degerler degismez) ->
        # golden-master korunur.
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        # episode_clean ACIK ise 1. episode (idx=0) gurultusuz orijinal, idx>=1 noise'lu.
        # KAPALI ise kosul daima True -> V11 davranisi (her episode gurultulu, golden bit-ayni).
        if self._noise_active and (not self._episode_clean or self._episode_idx >= 1):
            # v8: slippage/fiyat gurultusu (hocanin sarti, anti-ezber) — gerceklesen
            # riskli getiriye kucuk Gauss gurultusu. Env-yerel rng -> global RNG'ye
            # dokunmaz; yalniz egitimde (random_start), eval'de kapali (deterministik).
            r = r + self.rng.normal(0.0, self.price_noise_std, size=r.shape).astype(np.float32)
        if self.cash_asset:
            # v10: nakit varlik artik 0 degil; gunluk risksiz faiz kazanir (SABIT skaler,
            # RNG kullanmaz -> golden RNG sirasi korunur, yalniz deger degisir).
            sessions = int(self.session_counts[self.t + 1])
            cash_period_rate = (1.0 + self.cash_daily_rate) ** sessions - 1.0
            r = np.concatenate([r, [cash_period_rate]])
        return r

    def _should_rebalance(self) -> bool:
        return True

    def _apply_action(self, action) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.N:
            raise ValueError(f"action must have length {self.N}, got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError("action NaN/Inf iceremez")
        return softmax(a, temp=1.0)

    def step(self, action):
        # Piyasa/portfoy mekanigi burada; odul aritmetigi RewardEngine'de (P7, SRP).
        weights_before = self.w.copy()
        w_target = self._apply_action(action)
        delta_w_l1 = float(np.abs(w_target - weights_before).sum())
        r_vec = self._risky_returns()
        gross_port_r = float((w_target * r_vec).sum())

        regime_t = float(self.regime[self.t]) if self.regime is not None else 0.0
        outcome = self.reward.compute(gross_port_r=gross_port_r, delta_w_l1=delta_w_l1,
                                      nav=self.nav, peak=self.peak, regime=regime_t,
                                      step_count=self.step_count, max_steps=self.max_steps)
        self.nav, self.peak = outcome.nav, outcome.peak
        reward_terms = outcome.terms

        # Holdings drift after the market move. Fees are modeled as a wealth drag;
        # relative holdings are normalized by gross portfolio wealth.
        gross_wealth = 1.0 + gross_port_r
        if gross_wealth <= 1e-12 or not np.isfinite(gross_wealth):
            w_after = np.zeros(self.N, dtype=np.float32)
            w_after[-1] = 1.0
        else:
            w_after = np.asarray(w_target * (1.0 + r_vec) / gross_wealth, dtype=np.float32)
            w_after = np.clip(w_after, 0.0, None)
            total_w = float(w_after.sum())
            if total_w <= 1e-12 or not np.isfinite(total_w):
                raise FloatingPointError("donem-sonu portfoy agirliklari gecersiz")
            w_after /= total_w
        self.w = w_after
        self.t += 1
        self.step_count += 1
        self.nav_history.append(self.nav)
        self.weight_history.append(self.w.copy())
        self.ret_history.append(outcome.port_r_net)
        self.reward_terms_history.append(reward_terms)
        period_len = int(self.session_counts[self.t])
        discount = float(self.gamma ** period_len)
        self.weights_before_history.append(weights_before)
        self.target_weight_history.append(w_target.copy())
        self.turnover_history.append(delta_w_l1)
        self.period_length_history.append(period_len)
        self.discount_history.append(discount)
        self.date_history.append(self.dates[self.t])

        # İflas veya veri sonu → done; 252 adım tavanı → trunc
        done = (self.t >= (self.n_days - 1)) or reward_terms["bankrupt"]
        trunc = (self.step_count >= self.max_steps)
        info = dict(nav=self.nav, dd=reward_terms["dd"], port_r=outcome.port_r_net,
                    reward_terms=reward_terms, prices_t=self.prices[self.t].copy(),
                    weights_before=weights_before, target_weights=w_target.copy(),
                    weights_after=self.w.copy(), turnover=delta_w_l1,
                    period_length=period_len, discount=discount,
                    date=self.dates[self.t])
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
        # Slice baslangici 0'a clamp (kemer-ve-askı): coarse veride t-mw/t-mvw negatife dusup
        # BOS slice -> NaN olmasini engeller. Daily: t>=window>=mvw -> t-mvw>=0 -> max(0,.)
        # ETKISIZ -> golden bit-ayni. (mom/minvol ctor'da da cap'lendi; bu ikinci savunma hatti.)
        lo_mom = max(0, self.t - mw)
        lo_vol = max(0, self.t - mvw)
        # Portfolio templates operate on raw returns. Scaled feature values would
        # destroy cross-asset momentum and volatility magnitudes.
        lb_mom = np.diff(np.log(self.prices[lo_mom:self.t + 1]), axis=0)
        lb_vol = np.diff(np.log(self.prices[lo_vol:self.t + 1]), axis=0)
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
        logits = self._discrete_to_logits(a_idx)
        return super().step(logits)

```

---

## `env/reward.py`

```python
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
from scipy.stats import norm


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
        self.vol_var_ewma = float(vol_target) ** 2
        self.vol_ewma = float(vol_target)
        self.turnover_ewma = float(turnover_target)

    def reset(self):
        self.vol_var_ewma = self.vol_target ** 2
        self.vol_ewma = self.vol_target
        self.turnover_ewma = self.turnover_target

    def update_and_shape(self, port_r: float, delta_w_l1: float) -> Tuple[float, float, float, float, float]:
        a = self.alpha
        self.vol_var_ewma  = ((1 - a) * self.vol_var_ewma
                              + a * float(port_r) * float(port_r))
        self.vol_ewma      = float(np.sqrt(max(self.vol_var_ewma, 0.0)))
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
                 w_dsr: float, bankruptcy_nav: float, bankruptcy_penalty: float,
                 w_cvar: float = 0.0, cvar_alpha: float = 0.05,
                 regime_beta: float = 1.0, cvar_amp: float = 1.0,
                 w_gain: float = 0.0, gain_floor: float = 1.0,
                 w_gain_speed: float = 0.0, w_ruin_timing: float = 0.0):
        self.shaper = shaper
        self.dsharpe = dsharpe
        self.w_dsr = float(w_dsr)
        self.bankruptcy_nav = float(bankruptcy_nav)
        self.bankruptcy_penalty = float(bankruptcy_penalty)
        # v7: rejim-amplified kuyruk-riski (CVaR) cezasi.
        self.w_cvar = float(w_cvar)
        self.regime_beta = float(regime_beta)
        self.cvar_amp = float(cvar_amp)
        # v9: OPT-IN kazanc-carpani odulu + iflas-timing cezasi (default KAPALI ->
        # golden bit-ayni). Yeni RNG cagrisi YOK; yalniz nav/step_frac/bankrupt kullanir.
        self.w_gain = float(w_gain)                # kazanc-carpani agirligi (0 -> kapali)
        self.gain_floor = float(gain_floor)        # esik nav; uzeri odullenir
        self.w_gain_speed = float(w_gain_speed)    # erken-kazanc hiz faktoru (0 -> kapali)
        self.w_ruin_timing = float(w_ruin_timing)  # erken-iflas cezasi olcegi (0 -> flat)
        # Parametrik (Gauss) ileri CVaR/ES carpani: ES_alpha = sigma * phi(z_alpha)/alpha.
        # vol_ewma'yi (mevcut makine) sigma proxy'si olarak yeniden kullanir (ileri-bakisli).
        z = float(norm.ppf(cvar_alpha))
        self._cvar_mult = float(norm.pdf(z) / max(cvar_alpha, 1e-9))

    def reset(self):
        self.shaper.reset()
        self.dsharpe.reset()

    def compute(self, *, gross_port_r: float, delta_w_l1: float,
                nav: float, peak: float, regime: float = 0.0,
                step_count: int = 0, max_steps: int = 1) -> RewardOutcome:
        """Aritmetik sirasi onceki PortfolioEnv.step() ile ayni; +CVaR terimi (V7).

        v9: step_count/max_steps yalniz OPT-IN kazanc-hizi ve iflas-timing
        terimlerinde kullanilir; w_gain=w_gain_speed=w_ruin_timing=0 (default) iken
        sonuca sifir katki -> golden bit-ayni, yeni RNG cagrisi yok.
        """
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

        peak = max(peak, nav)
        dd = (peak - nav) / max(peak, 1e-9)
        # C7: log-getiri NET getiri uzerinden (port_r_net = gross - tx_cost). Boylece
        # islem maliyeti odule TEK SEFER girer (hem NAV'da hem ayri -tx_cost teriminde
        # cift sayilmasi giderildi). Log-zenginlik maksimizasyonuyla (Kelly/Moody-Saffell)
        # tutarli: ajan gerceklesen net log-fayda gorur.
        log_r = float(np.log(max(1.0 + port_r_net, 1e-6)))
        dd_penalty = lambda_t * max(0.0, dd - tau_t)
        dsr = self.dsharpe.update(port_r_net)
        dsr_term = self.w_dsr * dsr

        # v7: ileri-parametrik CVaR kuyruk cezasi; krizde (regime>0) kappa amplify olur.
        cvar = vol_ewma * self._cvar_mult                       # ES_alpha ~ sigma·mult
        rm = 1.0 + self.regime_beta * max(0.0, float(regime))   # kriz amplifikasyonu
        cvar_penalty = self.w_cvar * (rm ** self.cvar_amp) * cvar

        # v9: episod ilerleme orani (0->1). max_steps>=1 garanti (env gecirir).
        step_frac = float(step_count) / max(float(max_steps), 1e-9)

        # v9 (OPT-IN, default KAPALI): kazanc-carpani odulu. nav esigin (gain_floor)
        # uzerindeyse dogrusal odul; opsiyonel hiz faktoru erken kazanci kayirir
        # (1 + w_gain_speed*(1-step_frac)). w_gain=0 -> 0 (golden bit-ayni).
        gain_bonus = (self.w_gain * max(0.0, nav - self.gain_floor)
                      * (1.0 + self.w_gain_speed * (1.0 - step_frac)))

        # v9 (OPT-IN, default KAPALI): iflas-timing carpani. Erken iflas daha sert
        # cezalandirilir: mult = 1 + w_ruin_timing*(1-step_frac). w_ruin_timing=0 ->
        # mult=1.0 -> ham flat bankruptcy_penalty KORUNUR (golden bit-ayni).
        ruin_timing_mult = 1.0 + self.w_ruin_timing * (1.0 - step_frac)
        ruin_pen = (self.bankruptcy_penalty * ruin_timing_mult) if bankrupt else 0.0

        # C7: -tx_cost AYRI terimi KALDIRILDI; islem maliyeti artik log_r icinde
        # (log(1+port_r_net), port_r_net = gross - tx_cost). terms["tx_cost"] raporlamada
        # KALIR ama total'e ayrica EKLENMEZ (cift-sayim giderildi).
        total = (log_r - dd_penalty - ruin_pen
                 + dsr_term - cvar_penalty + gain_bonus)

        # NOT: terms["bankruptcy_penalty"] = ruin_pen (efektif ceza). w_ruin_timing=0
        # iken ruin_pen == flat bankruptcy_penalty oldugundan mevcut anahtar anlami korunur.
        # tx_cost terms'te raporlanmaya devam eder (panel/teshis) ama total'e girmez.
        terms = dict(
            log_return=log_r, tx_cost=tx_cost,
            drawdown_penalty=dd_penalty, total=total,
            dsr=dsr, dsr_term=dsr_term,
            cvar_penalty=cvar_penalty, regime=float(regime),   # v7
            eta_t=eta_t, lambda_t=lambda_t, tau_t=tau_t,
            vol_ewma=vol_ewma, turnover_ewma=to_ewma,
            dd=dd, gross_port_r=gross_port_r, delta_w_l1=delta_w_l1,
            bankruptcy_penalty=ruin_pen, bankrupt=bankrupt,
            gain_bonus=gain_bonus, ruin_timing_mult=ruin_timing_mult,   # v9 (additive)
        )
        return RewardOutcome(total=total, nav=nav, peak=peak,
                             port_r_net=port_r_net, terms=terms)

```

---

## `agents/base.py`

```python
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

```

---

## `agents/common.py`

```python
"""Ajanlar arasi paylasilan yardimcilar — Faz 1 (M3: kod tekrari giderme).

set_seed / get_device / mlp / ReplayBuffer uc ajanda da kopyalanmis kodu tek
yere toplar.

DAVRANIS KORUNUR: RNG tuketen islemlerin sirasi ve cagri imzalari onceki inline
kodla birebir aynidir —
  * mlp() katmanlari `sizes` sirasiyla olusturur -> nn.Linear agirlik init'i
    (global torch RNG) ayni sirada tuketilir -> ayni baslangic agirliklari.
  * ReplayBuffer.sample() ayni `np.random.randint(0, n, batch)` cagrisini yapar.
Golden-master regresyonu (tests/golden) bu esdegerligi dogrular.
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch.nn as nn

# SOLID P1 (DIP): set_seed/get_device notr utils/torch_utils.py'ye tasindi.
# Eski import yollari (agents.common.set_seed vb.) calismaya devam eder.
from utils.torch_utils import get_device, set_seed  # noqa: F401


def mlp(sizes: Sequence[int], activation: type[nn.Module],
        out_activation: type[nn.Module] | None = None) -> nn.Sequential:
    """sizes=[in, h1, ..., out] icin Linear+aktivasyon yigini kurar.

    Gizli katmanlar `activation`, son katman `out_activation` alir
    (None ise aktivasyon yok). Katmanlar `sizes` sirasiyla olusturulur, boylece
    nn.Linear agirlik init sirasi onceki inline nn.Sequential ile ayni kalir.

    Ornekler (onceki kodla ayni yapilar):
      QNetwork  : mlp([s, 256, 128, n], nn.ReLU)                  # son katman aktivasyonsuz
      SAC trunk : mlp([s, 256, 128], nn.ReLU, out_activation=nn.ReLU)   # govde, sonu aktivasyonlu
      PPO trunk : mlp([s, 256, 128], nn.Tanh, out_activation=nn.Tanh)
      ValueNet  : mlp([s, 256, 128, 1], nn.Tanh)
    """
    layers: list[nn.Module] = []
    n = len(sizes)
    for i in range(n - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        act = out_activation if i == n - 2 else activation
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class ReplayBuffer:
    """Uniform ornekleme deque buffer (DQN + SAC ortak).

    Tuple semasi: (s: float32[ ], a, r: float, s2: float32[ ], d: float).
    `a` cagiran tarafindan tiplenir (DQN: int, SAC: float32 vektor); push()
    s/s2/r/d'yi onceki `remember` ile ayni sekilde normalize eder.
    """

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, s, a, r, s2, d, discount=1.0) -> None:
        self.buffer.append((
            np.asarray(s, dtype=np.float32),
            a,
            float(r),
            np.asarray(s2, dtype=np.float32),
            float(d),
            float(discount),
        ))

    def sample(self, batch_size: int):
        """(s, a, r, s2, d) numpy yiginlari dondurur.

        np.array([...]) hem skaler (DQN) hem esit-boyutlu vektor (SAC) aksiyon
        listelerinde dogru yiginlamayi verir; onceki inline np.array/np.stack
        karisimiyla ayni degerleri uretir. Tensor dtype'i cagiran ayarlar.
        """
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        s  = np.stack([b[0] for b in batch])
        a  = np.array([b[1] for b in batch])
        r  = np.array([b[2] for b in batch])
        s2 = np.stack([b[3] for b in batch])
        d  = np.array([b[4] for b in batch])
        discount = np.array([b[5] for b in batch])
        return s, a, r, s2, d, discount

```

---

## `agents/dqn.py`

```python
"""Deep Q-Network ajanı — PyTorch implementasyonu (ayrık aksiyonlu).

Prompt spec'i:
  - MLP: state_dim (397: 13×28 + 4 makro + 29 ağırlık) → FC(256, ReLU) → FC(128, ReLU) → 6 (Q-values)
    (393 = makro-öncesi V5 tabanı; +4 makro [MacroConfig.enabled] = 397)
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
        self.env_step_count = 0

    def _sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def eps(self) -> float:
        frac = min(1.0, self.env_step_count / max(self.eps_decay, 1))
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

    def observe_step(self) -> None:
        self.env_step_count += 1

    def remember(self, s, a, r, s2, d, discount=None):
        self.buffer.push(s, int(a), r, s2, d,
                         self.gamma if discount is None else discount)

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d, discount = self.buffer.sample(self.batch_size)
        s  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(a,  dtype=torch.int64,   device=self.device)
        r  = torch.as_tensor(r,  dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(d,  dtype=torch.float32, device=self.device)
        discount = torch.as_tensor(discount, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_next = self.q_target(s2).max(dim=1).values
            td_target = r + (1.0 - d) * discount * q_next

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
        torch.save({"q": self.q.state_dict(), "step_count": self.step_count,
                    "env_step_count": self.env_step_count}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(ckpt["q"])
        self._sync_target()
        self.step_count = int(ckpt.get("step_count", 0))
        self.env_step_count = int(ckpt.get("env_step_count", self.step_count))

```

---

## `agents/ppo.py`

```python
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

from .base import BaseAgent
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


class PPOAgent(BaseAgent):
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
        self.S, self.A, self.R = [], [], []
        self.Terminal, self.Boundary = [], []
        self.Vs, self.NextVs, self.LP = [], [], []
        self.Discounts, self.GaeDiscounts = [], []

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

    def act_eval(self, s: np.ndarray) -> np.ndarray:
        """Eval: deterministik ortalama (mu) kullan — SAC/TD3/DQN ile tutarli.

        Stokastik sample() yerine politika aginin mu ciktisini dogrudan dondurur;
        boylece PPO backtest tekraranabilir ve tek-deterministik olur.
        """
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.policy(s_t)
        return mu.cpu().numpy()[0].astype(np.float32)

    def remember(self, s, a, r, done, v, logp, *, next_v=0.0,
                 boundary=None, discount=None, period_length=1):
        self.S.append(np.asarray(s, dtype=np.float32))
        self.A.append(np.asarray(a, dtype=np.float32))
        self.R.append(float(r))
        self.Terminal.append(float(done))
        self.Boundary.append(float(done if boundary is None else boundary))
        self.Vs.append(float(v)); self.NextVs.append(float(next_v)); self.LP.append(float(logp))
        sessions = max(1, int(period_length))
        self.Discounts.append(float(self.gamma ** sessions if discount is None else discount))
        self.GaeDiscounts.append(float((self.gamma * self.lam) ** sessions))

    def compute_gae(self, last_v: float):
        n = len(self.R)
        adv = np.zeros(n, dtype=np.float32)
        g = 0.0
        for i in reversed(range(n)):
            terminal_mask = 1.0 - self.Terminal[i]
            carry_mask = 1.0 - self.Boundary[i]
            delta = (self.R[i] + self.Discounts[i] * self.NextVs[i] * terminal_mask
                     - self.Vs[i])
            g = delta + self.GaeDiscounts[i] * carry_mask * g
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

```

---

## `agents/sac.py`

```python
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

    def remember(self, s, a, r, s2, d, discount=None):
        self.buffer.push(s, np.asarray(a, dtype=np.float32), r, s2, d,
                         self.gamma if discount is None else discount)

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, d, discount = self.buffer.sample(self.batch_size)
        s  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(a,  dtype=torch.float32, device=self.device)
        r  = torch.as_tensor(r,  dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(d,  dtype=torch.float32, device=self.device)
        discount = torch.as_tensor(discount, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a2, logp2 = self.pi.sample(s2)
            q_min = torch.min(self.q1_t(s2, a2), self.q2_t(s2, a2))
            target = r + (1.0 - d) * discount * (q_min - self.alpha * logp2)

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

```

---

## `agents/td3.py`

```python
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

    def remember(self, s, a, r, s2, d, discount=None):
        self.buffer.push(s, np.asarray(a, dtype=np.float32), r, s2, d,
                         self.gamma if discount is None else discount)

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        self._it += 1
        s, a, r, s2, d, discount = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)
        discount = torch.as_tensor(discount, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_t(s2) + noise).clamp(-1, 1)               # hedef-politika yumuşatma
            q_min = torch.min(self.q1_t(s2, a2), self.q2_t(s2, a2))    # twin min-Q hedefi
            target = r + (1 - d) * discount * q_min

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

```

---

## `core/contracts.py`

```python
"""Serializable contracts shared by training, evaluation, UI and persistence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Literal

import numpy as np


DataSource = Literal["real", "mixed", "synthetic", "unknown"]


@dataclass(frozen=True)
class DataProvenance:
    source: DataSource = "unknown"
    provider: str = ""
    reason: str = ""
    missing_tickers: tuple[str, ...] = ()
    requested_start: str = ""
    requested_end: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> "DataProvenance":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            data = dict(value)
            data["missing_tickers"] = tuple(data.get("missing_tickers", ()))
            return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})
        return cls()


@dataclass(frozen=True)
class RunSpec:
    algo: str
    step_days: int = 1
    adaptive: bool = True
    reward_cfg: dict[str, Any] = field(default_factory=dict)
    agent_hp: dict[str, Any] = field(default_factory=dict)
    data_start: str = ""
    data_split: str = ""
    data_end: str = ""
    seed: int = 42
    feature_names: tuple[str, ...] = ()
    provenance: DataProvenance = field(default_factory=DataProvenance)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["feature_names"] = list(self.feature_names)
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunSpec":
        data = dict(value)
        data["feature_names"] = tuple(data.get("feature_names", ()))
        data["provenance"] = DataProvenance.from_value(data.get("provenance"))
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class BacktestResult:
    nav: np.ndarray
    rets: np.ndarray
    dates: list[Any]
    weights_before: np.ndarray
    target_weights: np.ndarray
    weights_after: np.ndarray
    turnover: np.ndarray
    reward_terms_history: list[dict]
    period_lengths: np.ndarray
    discounts: np.ndarray
    provenance: DataProvenance = field(default_factory=DataProvenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nav": self.nav,
            "rets": self.rets,
            "dates": self.dates,
            "weights": self.weights_after,
            "weights_before": self.weights_before,
            "target_weights": self.target_weights,
            "weights_after": self.weights_after,
            "turnover": self.turnover,
            "reward_terms_history": self.reward_terms_history,
            "period_lengths": self.period_lengths,
            "discounts": self.discounts,
            "provenance": self.provenance.to_dict(),
        }

```

---

## `core/features.py`

```python
"""Ozellik secim politikasi — SOLID P2 (DRY).

UI (app.py) ve CLI (train.py) ayni forecast-filtreleme kuralini ayri ayri
kopyaliyordu. Tek dogruluk kaynagi artik burasi.

v2 ablation bulgusu: PPO forecast feature'indan zarar gordu -> forecast yalniz
ForecastConfig.forecast_agents'taki ajanlara verilir.
"""
from __future__ import annotations

from config import ForecastConfig


def select_features(feats: dict, algo: str) -> dict:
    """forecast feature'ini yalniz ForecastConfig.forecast_agents'taki ajanlara ver.

    Diger feature'lar aynen gecer; forecast yoksa dict degismeden doner.
    (Davranis train._feats_for ile birebir ayni — o artik buna delege eder.)
    """
    if "forecast" in feats and algo not in ForecastConfig.forecast_agents:
        return {k: v for k, v in feats.items() if k != "forecast"}
    return feats

```

---

## `core/factory.py`

```python
"""Ajan + ortam fabrikasi — SOLID P3 (OCP + DRY).

UI (app.py) ve CLI (train.py) ajan/ortam kurulumunu ayri ayri kopyaliyordu;
ayrica `if algo ==` zinciri yeni algoritma eklemeyi mevcut kodu degistirmeye
zorluyordu (OCP ihlali). Artik:

  - AGENT_BUILDERS: "DQN"|"PPO"|"SAC" -> builder registry'si. Yeni algoritma =
    yeni kayit; mevcut kod degismez (OCP).
  - build_agent: hp dict'i config default'lariyla birlestirip ajani kurar.
  - build_env  : discrete<->continuous secimi + select_features + odul
    override'lari TEK noktada.

DAVRANIS KORUNUR: parametre birlestirme sirasi onceki app._make_agent ve
train.make_env ile birebir ayni; ajan ctor'lari ayni sirayla cagrilir (RNG
tuketimi degismez). Golden-master <=1e-6 bunu dogrular.

DIKKAT: UI ve CLI farkli max_steps kullanir (orn. UI SAC=1200, CLI SAC=600).
Bunlar bilincli olarak BIRLESTIRILMEZ; max_steps'i cagiran taraf gecirir.
"""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from agents import DQNAgent, PPOAgent, SACAgent, TD3Agent
from config import DEFAULTS, SEED, DQNConfig, EnvConfig, PPOConfig, RewardConfig, SACConfig, TD3Config
from core.features import select_features
from env.portfolio_env import DiscretePortfolioEnv, PortfolioEnv


def _build_dqn(state_dim: int, action_dim: int, hp: dict, seed: int) -> DQNAgent:
    return DQNAgent(
        state_dim, action_dim,
        hidden=tuple(hp.get("hidden", DQNConfig.hidden)),
        lr=hp.get("lr", DQNConfig.lr),
        gamma=hp.get("gamma", DQNConfig.gamma),
        eps_start=hp.get("eps_start", DQNConfig.eps_start),
        eps_end=hp.get("eps_end", DQNConfig.eps_end),
        eps_decay=hp.get("eps_decay", DQNConfig.eps_decay),
        buffer_size=hp.get("buffer_size", DQNConfig.buffer_size),
        batch_size=hp.get("batch_size", DQNConfig.batch_size),
        target_update=hp.get("target_update", DQNConfig.target_update),
        huber_delta=hp.get("huber_delta", DQNConfig.huber_delta),
        seed=seed,
    )


def _build_ppo(state_dim: int, action_dim: int, hp: dict, seed: int) -> PPOAgent:
    return PPOAgent(
        state_dim, action_dim,
        hidden=tuple(hp.get("hidden", PPOConfig.hidden)),
        gamma=hp.get("gamma", PPOConfig.gamma),
        lam=hp.get("lam", PPOConfig.lam),
        lr_p=hp.get("lr_p", PPOConfig.lr_p), lr_v=hp.get("lr_v", PPOConfig.lr_v),
        clip=hp.get("clip", PPOConfig.clip), ent_coef=hp.get("ent_coef", PPOConfig.ent_coef),
        batch_size=hp.get("batch_size", PPOConfig.batch_size),
        n_epochs=hp.get("n_epochs", PPOConfig.n_epochs),
        log_std_init=hp.get("log_std_init", PPOConfig.log_std_init),
        seed=seed,
    )


def _build_sac(state_dim: int, action_dim: int, hp: dict, seed: int) -> SACAgent:
    return SACAgent(
        state_dim, action_dim,
        hidden=tuple(hp.get("hidden", SACConfig.hidden)),
        gamma=hp.get("gamma", SACConfig.gamma),
        lr_pi=hp.get("lr_pi", SACConfig.lr_pi), lr_q=hp.get("lr_q", SACConfig.lr_q),
        alpha=hp.get("alpha", SACConfig.alpha), tau=hp.get("tau", SACConfig.tau),
        buffer_size=hp.get("buffer_size", SACConfig.buffer_size),
        batch_size=hp.get("batch_size", SACConfig.batch_size), seed=seed,
    )


def _build_td3(state_dim: int, action_dim: int, hp: dict, seed: int) -> TD3Agent:
    return TD3Agent(
        state_dim, action_dim,
        hidden=tuple(hp.get("hidden", TD3Config.hidden)),
        lr_pi=hp.get("lr_pi", TD3Config.lr_pi), lr_q=hp.get("lr_q", TD3Config.lr_q),
        gamma=hp.get("gamma", TD3Config.gamma), tau=hp.get("tau", TD3Config.tau),
        policy_noise=hp.get("policy_noise", TD3Config.policy_noise),
        noise_clip=hp.get("noise_clip", TD3Config.noise_clip),
        policy_delay=hp.get("policy_delay", TD3Config.policy_delay),
        expl_noise=hp.get("expl_noise", TD3Config.expl_noise),
        buffer_size=hp.get("buffer_size", TD3Config.buffer_size),
        batch_size=hp.get("batch_size", TD3Config.batch_size), seed=seed,
    )


# OCP: yeni algoritma eklemek = bu registry'ye kayit eklemek.
AGENT_BUILDERS: Dict[str, Callable] = {
    "DQN": _build_dqn,
    "PPO": _build_ppo,
    "SAC": _build_sac,
    "TD3": _build_td3,
}


def build_agent(algo: str, state_dim: int, action_dim: int,
                hp: dict | None = None, *, seed: int = SEED):
    """Registry uzerinden ajan kurar; hp eksik anahtarlarda config default'a duser."""
    builder = AGENT_BUILDERS.get(algo)
    if builder is None:
        raise ValueError(f"Bilinmeyen algoritma: {algo!r} "
                         f"(kayitli: {sorted(AGENT_BUILDERS)})")
    return builder(state_dim, action_dim, hp or {}, seed)


def build_env(algo: str, prices: pd.DataFrame, feats: dict, *,
              horizon: str = "medium", adaptive: bool = True,
              max_steps: int, random_start: bool = False, seed: int = SEED,
              reward_overrides: dict | None = None,
              price_noise_std: float | None = None,
              force_price_noise: bool = False,
              cash_daily_rate: float | None = None,
              episode_clean: bool = False,
              rebalance_freq: int | None = None,
              gamma: float | None = None,
              mom_window: int | None = None,
              minvol_window: int | None = None,
              step_days: int = DEFAULTS.step_days,
              start_index: int | None = None,
              macro=None, regime=None) -> PortfolioEnv:
    """Tek ortam kurulum noktasi: discrete<->continuous secimi + feature secimi.

    reward_overrides (UI'nin reward_cfg'i): None/eksik anahtarlar env'in preset
    default'larina duser — onceki app._make_env mapping'i ile birebir ayni.

    cash_daily_rate: None -> env ctor kendi config'inden turetir (EnvConfig.cash_daily_rate).
    UI/CLI parametrik gunluk nakit faiz oranini dogrudan gecebilir.
    """
    cfg = reward_overrides or {}
    cls = DiscretePortfolioEnv if algo == "DQN" else PortfolioEnv
    # cash_daily_rate: None gecilirse env ctor config default'a duser (EnvConfig.cash_daily_rate).
    # Parametrik gecilirse (UI/CLI) env ctor None olmayan degeri kullanir.
    extra_cash = {} if cash_daily_rate is None else {"cash_daily_rate": float(cash_daily_rate)}
    return cls(
        prices, select_features(feats, algo),
        horizon=horizon, adaptive=adaptive, max_steps=max_steps,
        random_start=random_start, seed=seed,
        eta_base=cfg.get("eta_base"),
        lambda_base=cfg.get("lambda_base"),
        tau_base=cfg.get("tau_base"),
        vol_target=float(cfg.get("vol_target", EnvConfig.vol_target)),
        turnover_target=float(cfg.get("turnover_target", EnvConfig.turnover_target)),
        ema_alpha=float(cfg.get("ema_alpha", EnvConfig.ema_alpha)),
        bankruptcy_nav=cfg.get("bankruptcy_nav"),
        bankruptcy_penalty=cfg.get("bankruptcy_penalty"),
        price_noise_std=(EnvConfig.price_noise_std if price_noise_std is None else float(price_noise_std)),
        force_price_noise=bool(force_price_noise),   # gurultu-artirimli coklu-episode (eval'de gurultu); default kapali
        episode_clean=bool(episode_clean),   # OPT-IN: UI training True (1. iter orijinal); CLI/golden False
        rebalance_freq=rebalance_freq,        # OPT-IN: None -> preset (CLI/golden); UI override eder
        gamma=gamma, mom_window=mom_window, minvol_window=minvol_window,  # v12: acik override (None->preset)
        step_days=step_days,
        start_index=start_index,
        # Mevcut 6 odul param'i parametrik akisa acilir — eksik/None anahtar config
        # default'una duser (golden-guvenli; eta_base/bankruptcy_penalty deseni ile ayni).
        w_dsr=float(cfg.get("w_dsr", RewardConfig.w_dsr)),
        dsr_eta=float(cfg.get("dsr_eta", RewardConfig.dsr_eta)),
        w_cvar=cfg.get("w_cvar"),   # None -> env'de config default*cvar_factor (golden-guvenli)
        cvar_alpha=float(cfg.get("cvar_alpha", RewardConfig.cvar_alpha)),
        regime_beta=float(cfg.get("regime_beta", RewardConfig.regime_beta)),
        cvar_amp=float(cfg.get("cvar_amp", RewardConfig.cvar_amp)),
        # v9: OPT-IN kazanc-carpani + iflas-timing (default 0/kapali -> golden bit-ayni)
        w_gain=float(cfg.get("w_gain", 0.0)),
        gain_floor=float(cfg.get("gain_floor", 1.0)),
        w_gain_speed=float(cfg.get("w_gain_speed", 0.0)),
        w_ruin_timing=float(cfg.get("w_ruin_timing", 0.0)),
        macro=macro, regime=regime,   # v6: makro rejim blogu + ham regime (V7)
        **extra_cash,
    )

```

---

## `core/trainer.py`

```python
"""Generator-tabanli egitim cekirdegi (Faz 3, H1).

train.py (CLI, toplu) ve app.py (UI, canli/artimli) ARTIK ayni egitim dongusunu
paylasir. Her algoritmanin dongusu kendi generator'inda kalir:
  - DQN / SAC : off-policy, adim-bazli (act -> step -> remember -> train_step)
  - PPO       : on-policy, rollout-bazli (rollout_len adim topla -> train)
Her iterasyonda ORTAK bir telemetri dict'i yield edilir. CLI generator'i n_iters
ile sinirli tuketir; UI n_iters=None ile sonsuz akisi 'Durdur' ile keser.

Dispatch (train) ajan tipine gore dogru generator'i secen TEK noktadir.

DAVRANIS KORUNUR: her dongunun RNG tuketen cagri sirasi (agent.act,
agent.train_step / agent.train, np.random warmup, env.step) onceki train.py
donguleriyle birebir aynidir. Eklenen alanlar (loss/actions/success/agent/env)
gozlemdir, RNG tuketmez. Golden-master <=1e-6 bunu dogrular.
"""
from __future__ import annotations

import itertools
from typing import Iterator, Optional

import numpy as np

from agents import DQNAgent, PPOAgent, SACAgent, TD3Agent
from utils.metrics import success_vs_benchmark


def _counter(n: Optional[int]):
    """n is None ise sonsuz (UI), degilse range(n) (CLI)."""
    return itertools.count() if n is None else range(n)


# --------------------------------------------------------------------- DQN
def train_dqn(agent, env, n_episodes: Optional[int] = None,
              ew_nav: Optional[np.ndarray] = None) -> Iterator[dict]:
    for ep in _counter(n_episodes):
        s, _ = env.reset()
        done = trunc = False
        ep_reward = 0.0
        losses, actions = [], []
        while not (done or trunc):
            a = agent.act(s)
            actions.append(int(a))
            s2, r, done, trunc, info = env.step(a)
            agent.remember(s, a, r, s2, float(done), discount=info["discount"])
            agent.observe_step()
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            s = s2
            ep_reward += r
        nav_agent = np.array(env.nav_history[1:])
        success = (success_vs_benchmark(nav_agent, ew_nav[:len(nav_agent)])
                   if ew_nav is not None else 0)
        yield {
            "algo": "DQN", "iter": ep, "episode": ep,
            "reward": float(ep_reward),
            "nav": float(env.nav), "train_nav": float(env.nav),
            "gain": float(env.nav) - 1.0,
            "eps": agent.eps(),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "success": int(success),
            "actions": actions,
            "agent": agent, "env": env,
        }


# --------------------------------------------------------------------- PPO
def train_ppo(agent, env, n_updates: Optional[int] = None,
              rollout_len: int = 400,
              ew_nav: Optional[np.ndarray] = None) -> Iterator[dict]:
    import torch
    s, _ = env.reset()
    for upd in _counter(n_updates):
        ep_navs = []
        rollout_reward = 0.0
        for _ in range(rollout_len):
            a, lp, v = agent.act(s)
            s2, r, done, trunc, info = env.step(a)
            with torch.no_grad():
                next_v = float(agent.value(
                    torch.as_tensor(s2, dtype=torch.float32,
                                    device=agent.device).unsqueeze(0)
                ).item())
            agent.remember(s, a, r, done, v, lp, next_v=next_v,
                           boundary=(done or trunc), discount=info["discount"],
                           period_length=info["period_length"])
            rollout_reward += r
            s = s2
            if done or trunc:
                ep_navs.append(float(env.nav))
                s, _ = env.reset()
        with torch.no_grad():
            last_v = float(agent.value(
                torch.as_tensor(s, dtype=torch.float32,
                                device=agent.device).unsqueeze(0)
            ).item())
        info = agent.train(last_v)
        mean_nav = float(np.mean(ep_navs)) if ep_navs else float(env.nav)
        success = (int(mean_nav >= float(ew_nav[min(len(ew_nav) - 1, env.step_count)]))
                   if ew_nav is not None else 0)
        yield {
            "algo": "PPO", "iter": upd, "update": upd,
            "p_loss": float(info["p_loss"]), "v_loss": float(info["v_loss"]),
            "ent": float(info["ent"]), "kl": float(info["kl"]),
            "mean_nav": mean_nav, "nav": mean_nav,
            "reward": float(rollout_reward),  # L3: rollout boyu toplam cevre odulu (eski: -p_loss)
            "gain": mean_nav - 1.0,
            "loss": float(info["v_loss"]),
            "success": int(success),
            "agent": agent, "env": env,
        }


# --------------------------------------------------- SAC / TD3 (off-policy, ortak)
def _offpolicy_step(agent, env, s, step: int, warmup: int, train_every: int):
    """Tek off-policy adim: aksiyon sec (warmup'ta rastgele) -> env.step -> remember
    -> kosullu train_step. Donguden cikarildi (SonarCloud S3776 bilissel karmasiklik).
    RNG tuketim sirasi onceki SAC/TD3 donguleriyle birebir aynidir (golden-duyarli)."""
    if len(agent.buffer) < warmup:
        a = np.random.randn(env.action_dim).astype(np.float32) * 0.5
    else:
        a = agent.act(s)
    s2, r, done, trunc, info = env.step(a)
    agent.remember(s, a, r, s2, float(done), discount=info["discount"])
    loss = None
    if len(agent.buffer) > warmup and step % train_every == 0:
        loss = agent.train_step()
    return s2, r, done, trunc, loss


def _train_offpolicy(agent, env, algo: str, n_episodes: Optional[int],
                     warmup: int, train_every: int,
                     ew_nav: Optional[np.ndarray]) -> Iterator[dict]:
    """SAC ve TD3 icin ORTAK adim-bazli off-policy generator (DRY). Tek fark 'algo'
    etiketi; SAC stokastik, TD3 deterministik politikayi ajan icinde uygular."""
    for ep in _counter(n_episodes):
        s, _ = env.reset()
        done = trunc = False
        step = 0
        ep_reward = 0.0
        losses = []
        while not (done or trunc):
            s, r, done, trunc, loss = _offpolicy_step(agent, env, s, step, warmup, train_every)
            if loss is not None:
                losses.append(loss)
            step += 1
            ep_reward += r
        nav_agent = np.array(env.nav_history[1:])
        success = (success_vs_benchmark(nav_agent, ew_nav[:len(nav_agent)])
                   if ew_nav is not None else 0)
        yield {
            "algo": algo, "iter": ep, "episode": ep,
            "reward": float(ep_reward),  # L3: tum ajanlarda = iterasyon boyu toplam cevre odulu
            "nav": float(env.nav), "train_nav": float(env.nav),
            "gain": float(env.nav) - 1.0,
            "steps": step,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "success": int(success),
            "agent": agent, "env": env,
        }


def train_sac(agent, env, n_episodes: Optional[int] = None,
              warmup: int = 500, train_every: int = 4,
              ew_nav: Optional[np.ndarray] = None) -> Iterator[dict]:
    yield from _train_offpolicy(agent, env, "SAC", n_episodes, warmup, train_every, ew_nav)


def train_td3(agent, env, n_episodes: Optional[int] = None,
              warmup: int = 500, train_every: int = 4,
              ew_nav: Optional[np.ndarray] = None) -> Iterator[dict]:
    """TD3 off-policy — SAC ile ayni adim-bazli semayi (`_train_offpolicy`) paylasir;
    politika ajan icinde deterministiktir (hedef-politika yumusatma + gecikmeli guncelleme)."""
    yield from _train_offpolicy(agent, env, "TD3", n_episodes, warmup, train_every, ew_nav)


# --------------------------------------------------------------------- dispatch
def _launch_dqn(agent, env, n_iters, rollout_len, ew_nav):
    return train_dqn(agent, env, n_episodes=n_iters, ew_nav=ew_nav)


def _launch_ppo(agent, env, n_iters, rollout_len, ew_nav):
    return train_ppo(agent, env, n_updates=n_iters, rollout_len=rollout_len, ew_nav=ew_nav)


def _launch_sac(agent, env, n_iters, rollout_len, ew_nav):
    return train_sac(agent, env, n_episodes=n_iters, ew_nav=ew_nav)


def _launch_td3(agent, env, n_iters, rollout_len, ew_nav):
    return train_td3(agent, env, n_episodes=n_iters, ew_nav=ew_nav)


# SOLID P4 (OCP): yeni ajan tipi eklemek = bu registry'ye kayit eklemek;
# train() govdesi degismez. Kayit yoksa TypeError (onceki davranisla ayni).
_TRAINERS: dict = {
    DQNAgent: _launch_dqn,
    PPOAgent: _launch_ppo,
    SACAgent: _launch_sac,
    TD3Agent: _launch_td3,
}


def train(agent, env, *, n_iters: Optional[int] = None,
          rollout_len: int = 400,
          ew_nav: Optional[np.ndarray] = None) -> Iterator[dict]:
    """Ajan tipine gore dogru egitim generator'ini secen TEK dispatch noktasi.

    (Registry/Strategy pattern: dallanma tablo uzerinden bir kez yapilir,
    tuketicilerde degil.)
    """
    launcher = _TRAINERS.get(type(agent))
    if launcher is None:
        raise TypeError(f"Bilinmeyen ajan tipi: {type(agent).__name__}")
    return launcher(agent, env, n_iters, rollout_len, ew_nav)

```

---

## `core/rollout.py`

```python
"""Ajan-agnostik degerlendirme/rollout dongusu (Faz 3, H2).

Onceki train.py:evaluate ve app.py:evaluate_with_trace'in `if algo == 'DQN' ...`
dallanmasini ortadan kaldirir: BaseAgent.act_eval(state) kullanilir. Env zaten
hem ayrik (DiscretePortfolioEnv: idx) hem surekli (PortfolioEnv: vektor)
aksiyonu kabul eder, dolayisiyla dongu ajan tipinden bagimsizdir.
"""
from __future__ import annotations

import numpy as np

from core.contracts import BacktestResult


def evaluate(agent, env) -> dict:
    """Egitilmis ajani env uzerinde bir kez kosturur; backtest cikti dict'i dondurur.

    Onceki train.py:evaluate ile birebir ayni RNG tuketimi: act_eval
    (DQN greedy / PPO ornek / SAC deterministik) -> env.step.
    """
    s, _ = env.reset()
    done = trunc = False
    while not (done or trunc):
        a = agent.act_eval(s)
        s, r, done, trunc, _ = env.step(a)
    nav = np.array(env.nav_history)
    rets = np.concatenate([[0.0], np.array(env.ret_history)])
    initial_w = np.asarray(env.weight_history[0])
    result = BacktestResult(
        nav=nav,
        rets=rets,
        dates=[env.episode_start_date] + list(env.date_history),
        weights_before=np.vstack([initial_w, np.asarray(env.weights_before_history)]),
        target_weights=np.vstack([initial_w, np.asarray(env.target_weight_history)]),
        weights_after=np.asarray(env.weight_history),
        turnover=np.concatenate([[0.0], np.asarray(env.turnover_history, dtype=float)]),
        reward_terms_history=env.reward_terms_history,
        period_lengths=np.concatenate([[0], np.asarray(env.period_length_history, dtype=np.int32)]),
        discounts=np.concatenate([[1.0], np.asarray(env.discount_history, dtype=float)]),
        provenance=env.provenance,
    )
    return result.to_dict()

```

---

## `core/walkforward.py`

```python
"""Walk-forward dogrulama (Faz V5) — backtest overfitting'e karsi.

Alanin 1 numarali riski backtest overfitting / zayif genelleme (Liu 2022
arXiv:2209.05559; Velay 2023 arXiv:2306.10950). Bu modul train donemini
genisleyen pencerelere boler; her fold'da train-sub uzerinde egitip val-sub
uzerinde degerlendirir. Fold'lar arasi metrik stabilitesi = genelleme gostergesi.

SIZINTISIZLIK: her fold KENDI train-sub'inda TrainScaler ile olceklenir (val-sub
ayni istatistiklerle DONUSTURULUR, orada fit EDILMEZ). Caller forecast feature'i
DAHIL ETMEMELI (full-train forecaster fold val'ini gormus olur); teknik feature
ile cagrilmali (main.py boyle yapar).
"""
from __future__ import annotations

from collections import deque
from math import ceil

import numpy as np

from config import DEFAULTS
from core.rollout import evaluate
from core.trainer import train as train_loop
from env.portfolio_env import DiscretePortfolioEnv, PortfolioEnv
from utils.features import TrainScaler
from utils.metrics import summary
from utils.macro import MacroScaler


def _slice(feats_raw, idx):
    return {k: v.loc[idx] for k, v in feats_raw.items()}


def walk_forward(prices, feats_raw, agent_factory, *, discrete: bool = False,
                 n_folds: int = 3, val_frac: float = 0.2, purge: int = 5,
                 n_iters: int = 10, rollout_len: int = 400, seed: int = 42,
                 step_days: int = DEFAULTS.step_days, adaptive: bool = True,
                 macro=None, regime=None) -> dict:
    """Genisleyen-pencere walk-forward.

    agent_factory(state_dim, action_dim, seed) -> ajan. Doner:
    {"folds": [metrik dict...], "mean": {...}, "std": {...}}.

    macro: (T, F_macro) numpy dizisi (tam veri uzunlugu, fold icinde dilimlenir).
           None -> makrosuz ortam (V5 davranisi).
    regime: (T,) numpy dizisi (tam veri uzunlugu, fold icinde dilimlenir).
            None -> rejim amplifikasyonu kapali.
    NOT: forecast feature WF'de DAHIL EDILMEZ (sızıntı-güvenli mevcut karar KORUNUR).
    """
    T = len(prices)
    val_len = max(1, int(T * val_frac / n_folds))
    env_cls = DiscretePortfolioEnv if discrete else PortfolioEnv
    # Egitim env'i max_steps + random_start ile kurulur; env reset'i gecerli ve CESITLI bir
    # rastgele-baslangic araligi icin lo < tr_end - max_steps - 1 ister, aksi halde sessizce
    # sabit-baslangica duser (bkz. PortfolioEnv._reset_state) ve fold tek-pencereye dejenere
    # olur. Bu yuzden fold-atlama esigini sabit 80 yerine env'in episode-pencere gereksinimine
    # baglariz (lo, env ile ayni: max(window=20, minvol_window, 21)).
    step_days = int(step_days)
    if step_days < 1:
        raise ValueError("step_days en az 1 olmali")
    purge_steps = ceil(purge / step_days)
    lo = max(ceil(20 / step_days), ceil(DEFAULTS.minvol_window / step_days), 21)
    min_train = lo + 2
    fold_metrics = []
    for i in range(n_folds):
        val_end = T - (n_folds - 1 - i) * val_len
        val_start = val_end - val_len
        tr_end = val_start - purge_steps
        if tr_end < min_train:                 # random_start icin yeterli/cesitli train yok -> atla
            continue
        tr_idx = prices.index[:tr_end]
        va_idx = prices.index[val_start:val_end]

        sc = TrainScaler().fit(_slice(feats_raw, tr_idx))     # fold-yerel (sizintisiz)
        f_tr = sc.transform(_slice(feats_raw, tr_idx))
        context = min(val_start, lo + 1)
        ctx_start = val_start - context
        va_context_idx = prices.index[ctx_start:val_end]
        f_va = sc.transform(_slice(feats_raw, va_context_idx))

        # T6 (C6): macro/regime fold dilimleri (None gecilirse None kalir -> makrosuz).
        if macro is not None and hasattr(macro, "iloc"):
            raw_tr = macro.iloc[:tr_end]
            raw_va = macro.iloc[ctx_start:val_end]
            macro_scaler = MacroScaler().fit(raw_tr)
            macro_tr = macro_scaler.transform(raw_tr).to_numpy(np.float32)
            macro_va = macro_scaler.transform(raw_va).to_numpy(np.float32)
        else:
            # Legacy callers may provide already-scaled arrays. New callers pass
            # raw DataFrames so each fold owns its scaler fit.
            macro_tr = macro[:tr_end] if macro is not None else None
            macro_va = macro[ctx_start:val_end] if macro is not None else None
        regime_tr = regime[:tr_end] if regime is not None else None
        regime_va = regime[ctx_start:val_end] if regime is not None else None

        tr_env = env_cls(
            prices.loc[tr_idx], f_tr, adaptive=adaptive,
            max_steps=len(tr_idx), random_start=False, seed=seed,
            macro=macro_tr, regime=regime_tr, rebalance_freq=1,
            step_days=step_days, gamma=DEFAULTS.gamma,
            mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window,
        )
        action_dim = tr_env.n_discrete if discrete else tr_env.action_dim
        agent = agent_factory(tr_env.state_dim, action_dim, seed)
        # generator'i sonuna kadar tuket (egitim yan-etkili; ciktiya gerek yok)
        deque(train_loop(agent, tr_env, n_iters=n_iters, rollout_len=rollout_len), maxlen=0)

        va_env = env_cls(
            prices.loc[va_context_idx], f_va, adaptive=adaptive,
            max_steps=len(va_idx) + 10, random_start=False, seed=seed,
            macro=macro_va, regime=regime_va, rebalance_freq=1,
            step_days=step_days, gamma=DEFAULTS.gamma,
            mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window,
            start_index=context,
        )
        bt = evaluate(agent, va_env)
        if len(bt["nav"]) > 0:
            fold_metrics.append(summary(bt["nav"], bt["rets"], bt["weights"], dates=bt["dates"],
                                        turnover_values=bt.get("turnover")))

    keys = list(fold_metrics[0].keys()) if fold_metrics else []
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return {"folds": fold_metrics, "mean": mean, "std": std}

```

---

## `core/persistence.py`

```python
"""Egitilmis ajan kalici hale getirme — diske kaydet/yukle (PDF §11 sunum sarti).

Sorun: egitilmis ajan yalniz Streamlit session_state'te tutuluyordu; sayfa
yenilenince kayboluyordu. Bu modul ajanin AG AGIRLIKLARINI diske yazar/okur ki
sunumda yeniden egitmeden test edilebilsin.

Tasarim: jenerik. Ajanin nn.Module attribute'lari (DQN: q/q_target, PPO:
policy/value, SAC: pi/q1/q2/...) `vars(agent)` ile toplanir; ayri ajan-basi kod
gerekmez. Yukleme core.factory.build_agent ile ayni mimaride iskelet kurup
state_dict'leri ad'a gore geri yukler (cikarim icin ag agirliklari yeterli —
optimizer/replay buffer kaydedilmez).

N11: Dosya adi semasindan algo_{horizon}_{adaptive}.pt — ayni algoritmay
farkli vade/adaptive ile kaydedince birbirinin uzerine yazmaz.

Isimli kayit (UI-only): named_model_path(name) -> models/{guvenli_isim}.pt
Meta'da name + saved_at (ISO, saniye hassasiyeti) tutulur; geriye-uyumlu.

DAVRANIS: golden-irrelevant. Yalniz kalicilik; egitim/odul/eval sayisal yoluna
dokunmaz.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from core.contracts import RunSpec
from core.factory import build_agent

FORMAT = 2

# Tum kaydedilmis modellerin bulundugu dizin (proje koku / models/).
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def model_path(algo: str, horizon: str | int = "medium", adaptive: bool = True) -> Path:
    """N11: algo_{horizon}_{adaptive}.pt — vade+adaptive farklilastirir.

    Ornek: model_path("DQN", "short", False) -> models/DQN_short_False.pt
    Eski tek-dosya yolunun yerine gecer; farkli vade/adaptive birbirini ezmez.
    CLI/golden bu fonksiyonu kullanir — degistirilmez.
    """
    suffix = f"{adaptive}".lower()
    slot = f"step{horizon}" if isinstance(horizon, int) else str(horizon)
    return MODELS_DIR / f"{algo}_{slot}_{suffix}.pt"


def named_model_path(name: str, saved_at: str = "") -> Path:
    """Kullanici-verilen isimle kayit yolu: models/{guvenli_isim}.pt

    Sanitize: harf/rakam/_/- disini _ ile degistir; bos ise 'model_{saved_at}'.
    saved_at yalnizca fallback icin kullanilir (isim bossa).
    """
    safe = re.sub(r"[^\w\-]", "_", name.strip()) if name.strip() else ""
    if not safe:
        ts = re.sub(r"[^\w\-]", "_", saved_at) if saved_at else "model"
        safe = f"model_{ts}"
    return MODELS_DIR / f"{safe}.pt"


def _action_dim(agent) -> int:
    """DQN n_actions, PPO/SAC action_dim — ajandan turet."""
    ad = getattr(agent, "action_dim", None)
    if ad is None:
        ad = getattr(agent, "n_actions")
    return int(ad)


def _modules(agent) -> dict:
    """Ajanin tum nn.Module attribute'larinin {ad: state_dict}'i."""
    return {name: m.state_dict()
            for name, m in vars(agent).items() if isinstance(m, nn.Module)}


def _hidden_from_modules(algo: str, modules: dict) -> tuple[int, int] | None:
    module_name = {"DQN": "q", "PPO": "policy", "SAC": "pi", "TD3": "actor"}.get(algo)
    state = modules.get(module_name, {})
    weights = [v for k, v in state.items() if k.endswith("weight") and getattr(v, "ndim", 0) == 2]
    if len(weights) < 2:
        return None
    return int(weights[0].shape[0]), int(weights[1].shape[0])


def _agent_config(agent, algo: str, modules: dict) -> dict:
    hidden = _hidden_from_modules(algo, modules)
    cfg = {"hidden": list(hidden)} if hidden else {}
    for key in ("gamma", "batch_size", "eps_decay", "target_update", "lam", "clip",
                "ent_coef", "n_epochs", "alpha", "tau", "policy_noise", "noise_clip",
                "policy_delay", "expl_noise"):
        if hasattr(agent, key):
            value = getattr(agent, key)
            if isinstance(value, (str, int, float, bool)):
                cfg[key] = value
    return cfg


def save_agent(agent, algo: str, path, *, horizon: str = "medium",
               adaptive: bool = True, name: str = "",
               saved_at: str | None = None, run_spec: RunSpec | dict | None = None) -> str:
    """Ajanin ag agirliklarini + meta'yi `path`'e yazar; yolu doner.

    name: kullanici-verilen model adi (bos olabilir — meta'da saklanir).
    saved_at: ISO datetime str (sn hassasiyeti); None ise simdi hesaplanir.
    """
    if saved_at is None:
        saved_at = datetime.now().isoformat(timespec="seconds")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    modules = _modules(agent)
    spec_dict = (run_spec.to_dict() if isinstance(run_spec, RunSpec)
                 else dict(run_spec or {}))
    torch.save({
        "format": FORMAT,
        "algo": algo,
        "horizon": horizon,
        "adaptive": bool(adaptive),
        "state_dim": int(agent.state_dim),
        "action_dim": _action_dim(agent),
        "name": name,
        "saved_at": saved_at,
        "modules": modules,
        "agent_config": _agent_config(agent, algo, modules),
        "run_spec": spec_dict,
    }, path)
    return str(path)


def load_agent(path):
    """Kaydedilmis ajani yukler. (agent, meta) doner; meta: algo/horizon/adaptive/name/saved_at.

    build_agent ile ayni mimaride iskelet kurulur (config default hidden=(256,128)
    egitimdekiyle ayni), sonra state_dict'ler ad'a gore yuklenir.
    Eski modellerde name/saved_at yoksa bos string doner (geriye-uyumlu, KeyError yok).
    """
    # weights_only=True (guvenli unpickler): checkpoint yalniz metadata (str/int/bool)
    # + tensor state_dict'leri icerir; rastgele kod calistirma riski yok (SonarCloud S5042).
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    agent_cfg = dict(ckpt.get("agent_config", {}))
    if "hidden" not in agent_cfg:
        hidden = _hidden_from_modules(ckpt["algo"], ckpt["modules"])
        if hidden:
            agent_cfg["hidden"] = hidden
    agent = build_agent(ckpt["algo"], int(ckpt["state_dim"]), int(ckpt["action_dim"]), agent_cfg)
    for name, sd in ckpt["modules"].items():
        module = getattr(agent, name, None)
        if isinstance(module, nn.Module):
            module.load_state_dict(sd)
    meta = {
        "algo": ckpt["algo"],
        "horizon": ckpt.get("horizon", "medium"),
        "adaptive": bool(ckpt.get("adaptive", True)),
        "name": ckpt.get("name", ""),
        "saved_at": ckpt.get("saved_at", ""),
        "format": int(ckpt.get("format", 1)),
        "legacy": int(ckpt.get("format", 1)) < FORMAT,
        "agent_config": agent_cfg,
        "run_spec": ckpt.get("run_spec", {}),
    }
    return agent, meta


def read_meta(path) -> dict:
    """Ajani KURMADAN yalniz meta'yi okur (liste goruntuleme icin hizli yol).

    torch.load ile checkpoint yuklenir; 'modules' (buyuk tensor'lar) goz ardi edilir.
    Eski/meta'siz dosyada guvenli default doner — KeyError/exception yok.
    """
    try:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
        return {
            "algo": ckpt.get("algo", ""),
            "horizon": ckpt.get("horizon", "medium"),
            "adaptive": bool(ckpt.get("adaptive", True)),
            "name": ckpt.get("name", ""),
            "saved_at": ckpt.get("saved_at", ""),
            "format": int(ckpt.get("format", 1)),
            "legacy": int(ckpt.get("format", 1)) < FORMAT,
            "run_spec": ckpt.get("run_spec", {}),
        }
    except Exception:
        return {"algo": "", "horizon": "medium", "adaptive": True, "name": "", "saved_at": "",
                "format": 0, "legacy": True, "run_spec": {}}

```

---

## `train.py`

```python
"""Main training + backtest driver — step_days + adaptive reward.

Trains DQN (discrete, 6 templates), PPO (continuous), SAC (continuous) ve TD3
(continuous, hocanin tavsiyesi) on BIST 28 — 2015-2021 train, 2022-2024 test
setinde backtest eder. Tüm ajanlar PyTorch'tadır ve
özellikler `utils.features.TrainScaler` ile train-only z-score standardize edilir.
"""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from math import ceil
import numpy as np
import pandas as pd

from data import (download_bist, train_test_split, download_macro, align_macro,
                  resample_to_step_days)
from utils.features import add_features, TrainScaler
from utils.macro import add_macro_features, MacroScaler
from utils.metrics import summary, training_diagnostics
from utils.baselines import equal_weight, mean_variance, buy_and_hold_index
from config import DEFAULTS, SEED, TrainConfig, EnvConfig, ForecastConfig, MacroConfig, TD3Config  # noqa: F401
from core.contracts import DataProvenance, RunSpec
from core.factory import build_agent, build_env
from core.features import select_features
from core.persistence import save_agent, model_path
from core.rollout import evaluate as rollout_evaluate
from core.trainer import train as train_loop

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"
RES.mkdir(exist_ok=True)
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

# -------------------- veri (P6: modul-global state yerine acik DataBundle) --------------------
@dataclass
class DataBundle:
    """Egitim/degerlendirme veri paketi — prepare_data() uretir, tuketiciler
    acikca alir (DIP/testability: global durum yok, sahte bundle enjekte edilebilir)."""
    px: pd.DataFrame
    px_tr: pd.DataFrame
    px_te: pd.DataFrame
    feats_tr: dict
    feats_te: dict
    scaler: TrainScaler
    # v6: makro rejim blogu (z-skorlu state) + ham regime (V7 odul amplify). None -> V5.
    macro_tr: "np.ndarray | None" = None
    macro_te: "np.ndarray | None" = None
    regime_tr: "np.ndarray | None" = None
    regime_te: "np.ndarray | None" = None
    step_days: int = DEFAULTS.step_days
    provenance: DataProvenance = DataProvenance()


def prepare_data(step_days: int = DEFAULTS.step_days) -> DataBundle:
    """BIST verisini yukler, train/test ayirir, train-only z-score uygular;
    DataBundle dondurur (onceki surum modul globallerini dolduruyordu)."""
    px_daily = download_bist()
    feats_all_raw = add_features(px_daily)
    px_tr_daily, px_te_daily = train_test_split(px_daily)
    if ForecastConfig.enabled:                     # v2: forecast feature (train-only fit)
        from forecast.forecaster import build_forecast_feature
        feats_all_raw["forecast"] = build_forecast_feature(
            px_daily, px_tr_daily, window=ForecastConfig.window, conv_ch=ForecastConfig.conv_ch,
            hidden=ForecastConfig.hidden, epochs=ForecastConfig.epochs,
            lr=ForecastConfig.lr, batch=ForecastConfig.batch, seed=SEED)
    px_tr = resample_to_step_days(px_tr_daily, step_days)
    px_te = resample_to_step_days(px_te_daily, step_days)
    px = pd.concat([px_tr, px_te])
    px.attrs.update(px_daily.attrs)
    feats_tr_raw = {k: resample_to_step_days(v.loc[px_tr_daily.index], step_days)
                    for k, v in feats_all_raw.items()}
    feats_te_raw = {k: resample_to_step_days(v.loc[px_te_daily.index], step_days)
                    for k, v in feats_all_raw.items()}
    scaler = TrainScaler().fit(feats_tr_raw)
    feats_tr = scaler.transform(feats_tr_raw)
    feats_te = scaler.transform(feats_te_raw)
    print(f"Train: {px_tr.shape}, Test: {px_te.shape}, tickers: {px.shape[1]}")

    # v6: makro rejim (faiz/dolar/altin) — train-only z-score (leak-safe), ham regime ayri.
    macro_tr = macro_te = regime_tr = regime_te = None
    if MacroConfig.enabled:
        mraw = align_macro(download_macro(), px_daily.index)
        mfeat = add_macro_features(mraw)                       # (T,4) ham
        regime_full = mfeat["regime"]                          # ham ∈[-1,1] -> V7 amplify
        mtr = resample_to_step_days(mfeat.loc[px_tr_daily.index], step_days)
        mte = resample_to_step_days(mfeat.loc[px_te_daily.index], step_days)
        msc = MacroScaler().fit(mtr)
        macro_tr = msc.transform(mtr).to_numpy(np.float32)
        macro_te = msc.transform(mte).to_numpy(np.float32)
        regime_tr = resample_to_step_days(
            regime_full.loc[px_tr_daily.index].to_frame(), step_days).iloc[:, 0].to_numpy(np.float32)
        regime_te = resample_to_step_days(
            regime_full.loc[px_te_daily.index].to_frame(), step_days).iloc[:, 0].to_numpy(np.float32)
        print(f"Makro: {mfeat.shape[1]} oznitelik (regime/slope/usd_try/gold_tl)")

    return DataBundle(px=px, px_tr=px_tr, px_te=px_te,
                      feats_tr=feats_tr, feats_te=feats_te, scaler=scaler,
                      macro_tr=macro_tr, macro_te=macro_te,
                      regime_tr=regime_tr, regime_te=regime_te,
                      step_days=int(step_days),
                      provenance=DataProvenance.from_value(px_daily.attrs.get("provenance")))


def _feats_for(feats: dict, algo: str) -> dict:
    """Shim — SOLID P2: tek dogruluk kaynagi core.features.select_features.
    (test_env bu adi cagirir; geriye-uyumluluk icin korunur.)"""
    return select_features(feats, algo)


def _ew_nav_train(bundle: DataBundle) -> np.ndarray:
    """Egitim seti icin esit-agirlikli benchmark NAV dizisi (C3: basari metriginde
    kullanilir; train fonksiyonlari bu diziyi ew_nav olarak train_loop'a gecer).
    Sifir boylukta guvenli (None yerine bos dizi degil)."""
    return equal_weight(bundle.px_tr)["nav"]


# -------------------- DQN training --------------------
def train_dqn(bundle: DataBundle, n_episodes: int = TrainConfig.dqn_episodes,
              adaptive: bool = True, step_days: int | None = None):
    step_days = bundle.step_days if step_days is None else int(step_days)
    train_max_steps = len(bundle.px_tr)
    env = build_env("DQN", bundle.px_tr, bundle.feats_tr, adaptive=adaptive,
                    max_steps=train_max_steps, random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr, rebalance_freq=1,
                    step_days=step_days, gamma=DEFAULTS.gamma,
                    mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window)
    agent = build_agent("DQN", env.state_dim, env.n_discrete, seed=SEED)
    # C3: egitim seti EW benchmark'i — trainer success_vs_benchmark'e gecer.
    ew_nav_tr = _ew_nav_train(bundle)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes, ew_nav=ew_nav_tr):
        # C3: rec["success"] kernel'den gelir (success_vs_benchmark); override yok.
        curve.append(dict(episode=rec["episode"], reward=rec["reward"],
                          train_nav=rec["train_nav"], eps=rec["eps"],
                          gain=rec["gain"], success=rec["success"],
                          steps=len(rec.get("actions") or [])))
        print(f"[DQN] ep {rec['episode']:02d}  ret={rec['reward']:+.3f}  "
              f"NAV={rec['train_nav']:.3f}  eps={rec['eps']:.3f}")
    return agent, curve


# -------------------- PPO training --------------------
def train_ppo(bundle: DataBundle, n_updates: int = TrainConfig.ppo_updates,
              rollout_len: int = TrainConfig.ppo_rollout_len,
              adaptive: bool = True, step_days: int | None = None):
    # T7: PPO on-policy; rollout_len zaten episode'u belirler; max_steps > rollout_len olsun.
    # Vade bazlı train_max_steps ile uyumlu (büyük değer verirsek sorun yok ama
    # train_max_steps * 5 ile rollout'ların kesintisiz akmasına izin verelim).
    step_days = bundle.step_days if step_days is None else int(step_days)
    train_max_steps = len(bundle.px_tr)
    env = build_env("PPO", bundle.px_tr, bundle.feats_tr, adaptive=adaptive,
                    max_steps=train_max_steps, random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr, rebalance_freq=1,
                    step_days=step_days, gamma=DEFAULTS.gamma,
                    mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window)
    agent = build_agent("PPO", env.state_dim, env.action_dim, seed=SEED)
    # C3: egitim seti EW benchmark'i.
    ew_nav_tr = _ew_nav_train(bundle)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_updates, rollout_len=rollout_len,
                          ew_nav=ew_nav_tr):
        # C3: rec["success"] kernel'den gelir; override yok.
        curve.append(dict(update=rec["update"], p_loss=rec["p_loss"], v_loss=rec["v_loss"],
                          ent=rec["ent"], kl=rec["kl"], mean_nav=rec["mean_nav"],
                          reward=rec["reward"], gain=rec["gain"],
                          success=rec["success"], steps=rollout_len))
        print(f"[PPO] upd {rec['update']:02d}  p_loss={rec['p_loss']:.3f} "
              f"v_loss={rec['v_loss']:.3f} ent={rec['ent']:.2f} kl={rec['kl']:.3f}")
    return agent, curve


# -------------------- SAC training --------------------
def train_sac(bundle: DataBundle, n_episodes: int = TrainConfig.sac_episodes,
              adaptive: bool = True, step_days: int | None = None):
    step_days = bundle.step_days if step_days is None else int(step_days)
    env = build_env("SAC", bundle.px_tr, bundle.feats_tr, adaptive=adaptive,
                    max_steps=len(bundle.px_tr),
                    random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr, rebalance_freq=1,
                    step_days=step_days, gamma=DEFAULTS.gamma,
                    mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window)
    agent = build_agent("SAC", env.state_dim, env.action_dim, seed=SEED)
    # C3: egitim seti EW benchmark'i.
    ew_nav_tr = _ew_nav_train(bundle)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes, ew_nav=ew_nav_tr):
        # C3: rec["success"] kernel'den gelir; override yok.
        curve.append(dict(episode=rec["episode"], train_nav=rec["train_nav"], steps=rec["steps"],
                          reward=rec["reward"], gain=rec["gain"],
                          success=rec["success"]))
        print(f"[SAC] ep {rec['episode']:02d}  NAV={rec['train_nav']:.3f}  buf={len(agent.buffer)}")
    return agent, curve


# -------------------- TD3 training --------------------
def train_td3(bundle: DataBundle, n_episodes: int = TrainConfig.td3_episodes,
              adaptive: bool = True, step_days: int | None = None):
    # TD3 surekli-kontrol (hocanin tavsiyesi) — SAC ile ayni off-policy rejim.
    # Her episode secilen train tarih araliginin tamamini kullanir.
    step_days = bundle.step_days if step_days is None else int(step_days)
    env = build_env("TD3", bundle.px_tr, bundle.feats_tr, adaptive=adaptive,
                    max_steps=len(bundle.px_tr),
                    random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr, rebalance_freq=1,
                    step_days=step_days, gamma=DEFAULTS.gamma,
                    mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window)
    agent = build_agent("TD3", env.state_dim, env.action_dim, seed=SEED)
    # C3: egitim seti EW benchmark'i.
    ew_nav_tr = _ew_nav_train(bundle)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes, ew_nav=ew_nav_tr):
        # C3: rec["success"] kernel'den gelir; override yok.
        curve.append({"episode": rec["episode"], "train_nav": rec["train_nav"], "steps": rec["steps"],
                      "reward": rec["reward"], "gain": rec["gain"],
                      "success": rec["success"]})
        print(f"[TD3] ep {rec['episode']:02d}  NAV={rec['train_nav']:.3f}  buf={len(agent.buffer)}")
    return agent, curve


# -------------------- Evaluation --------------------
def evaluate(bundle: DataBundle, agent, algo: str, adaptive: bool = True,
             step_days: int | None = None):
    # T7: eval env TAM test donemini kosturur — max_steps = len(test) (backtest
    # tum test penceresini kapsar, egitim episode uzunlugundan bagimsiz).
    step_days = bundle.step_days if step_days is None else int(step_days)
    context = min(len(bundle.px_tr), max(21, ceil(DEFAULTS.minvol_window / step_days)) + 1)
    px_context = bundle.px_tr.iloc[-context:]
    px_eval = pd.concat([px_context, bundle.px_te])
    px_eval.attrs.update(bundle.px_te.attrs)
    px_eval.attrs["session_counts"] = np.concatenate([
        np.asarray(bundle.px_tr.attrs.get(
            "session_counts", np.ones(len(bundle.px_tr), dtype=int)))[-context:],
        np.asarray(bundle.px_te.attrs.get("session_counts", np.ones(len(bundle.px_te), dtype=int))),
    ])
    feats_eval = {k: pd.concat([bundle.feats_tr[k].iloc[-context:], bundle.feats_te[k]])
                  for k in bundle.feats_te}
    macro_eval = (None if bundle.macro_te is None else
                  np.concatenate([bundle.macro_tr[-context:], bundle.macro_te], axis=0))
    regime_eval = (None if bundle.regime_te is None else
                   np.concatenate([bundle.regime_tr[-context:], bundle.regime_te], axis=0))
    eval_max_steps = len(bundle.px_te) + 10
    env = build_env(algo, px_eval, feats_eval, adaptive=adaptive,
                    max_steps=eval_max_steps, macro=macro_eval, regime=regime_eval,
                    rebalance_freq=1, step_days=step_days, gamma=DEFAULTS.gamma,
                    mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window,
                    start_index=context)
    return rollout_evaluate(agent, env)


# -------------------- Main --------------------
def run(step_days: int = DEFAULTS.step_days):
    """Tam egitim + backtest akisi: seed -> veri -> 3 ajan -> eval -> CSV.
    main.py bunu DOGRUDAN cagirir (runpy yerine). Modul import'u yan etkisizdir (M1)."""
    np.random.seed(SEED)
    bundle = prepare_data(step_days=step_days)
    t0 = time.time()
    print("=" * 60)
    dqn_agent, dqn_curve = train_dqn(bundle)
    print("DQN total time:", round(time.time() - t0, 1), "s")

    t1 = time.time()
    ppo_agent, ppo_curve = train_ppo(bundle)
    print("PPO total time:", round(time.time() - t1, 1), "s")

    t2 = time.time()
    sac_agent, sac_curve = train_sac(bundle)
    print("SAC total time:", round(time.time() - t2, 1), "s")

    t3 = time.time()
    td3_agent, td3_curve = train_td3(bundle)          # hocanin tavsiyesi — SAC'tan SONRA
    print("TD3 total time:", round(time.time() - t3, 1), "s")

    print("=" * 60)

    # ---- T2 (C1): once ham eval; sonra ORTAK min_len ile hizala; metrics.csv
    # hizali dizilerden uretilir -> RL + baseline ayni gun sayisinda kiyaslanir. ----
    raw_results = {}
    for name, agent in [("DQN", dqn_agent), ("PPO", ppo_agent),
                        ("SAC", sac_agent), ("TD3", td3_agent)]:
        bt = evaluate(bundle, agent, name)
        raw_results[name] = bt

    bh = buy_and_hold_index(bundle.px_te)
    ew = equal_weight(bundle.px_te)
    mv = mean_variance(bundle.px_te, lookback=120, rebalance=20)
    for name, d in [("BuyHold", bh), ("EqualWeight", ew), ("MeanVar", mv)]:
        raw_results[name] = d

    date_indexes = [pd.DatetimeIndex(v.get("dates", bundle.px_te.index[:len(v["nav"])]))
                    for v in raw_results.values()]
    common_dates = date_indexes[0]
    for idx in date_indexes[1:]:
        common_dates = common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    if len(common_dates) < 2:
        raise RuntimeError("Stratejiler arasinda yeterli ortak degerleme tarihi yok")

    results = {}
    for name, bt in raw_results.items():
        # Hizalama: son min_len elemanı al (ortak takvim kuyruğu) VE ortak pencere
        # başlangıcına YENİDEN-TABANLA (NAV[0]=1) -> FinalNAV/CAGR tüm stratejiler
        # için AYNI pencere büyümesini ölçer; baseline'lar RL'in görmediği ilk
        # ~window günü dahil etmez (C1 adil karşılaştırma tam olarak sağlanır).
        idx = pd.DatetimeIndex(bt.get("dates", bundle.px_te.index[:len(bt["nav"])]))
        nav_series = pd.Series(np.asarray(bt["nav"], dtype=float), index=idx)
        nav_aligned = nav_series.reindex(common_dates).to_numpy()
        rets_aligned = np.concatenate([[0.0], np.diff(nav_aligned) / nav_aligned[:-1]])
        w_aligned    = bt.get("weights")
        if w_aligned is not None:
            w_aligned = pd.DataFrame(np.asarray(w_aligned), index=idx).reindex(common_dates).to_numpy()
        turn_aligned = None
        if bt.get("turnover") is not None:
            turn_aligned = pd.Series(np.asarray(bt["turnover"], dtype=float), index=idx).reindex(common_dates).to_numpy()
        # Hizali diziler uzerinden metrik hesapla (C1 duzeltme).
        m = summary(nav_aligned, rets_aligned, w_aligned, dates=common_dates,
                    turnover_values=turn_aligned)
        results[name] = dict(backtest=bt, nav_aligned=nav_aligned, metrics=m)
        if name in ("DQN", "PPO", "SAC", "TD3"):
            print(f"[TEST] {name:<3}  CAGR={m['CAGR']:+.2%}  Sharpe={m['Sharpe']:+.2f}  "
                  f"MaxDD={m['MaxDD']:+.2%}  Final={m['FinalNAV']:.3f}")

    print("-" * 60)
    for name in ("BuyHold", "EqualWeight", "MeanVar"):
        m = results[name]["metrics"]
        print(f"[TEST] {name:<12}  CAGR={m['CAGR']:+.2%}  Sharpe={m['Sharpe']:+.2f}  "
              f"MaxDD={m['MaxDD']:+.2%}  Final={m['FinalNAV']:.3f}")

    # navs_aligned.csv: hizali NAV dizileri (T2: metrics.csv ile AYNI pencere).
    dfn = pd.DataFrame({k: v["nav_aligned"] for k, v in results.items()})
    dfn.index = common_dates
    dfn.to_csv(RES / "navs_aligned.csv")

    # metrics.csv: hizali dizilerden hesaplanan metrikler (C1 duzeltme).
    met_df = pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T
    met_df.to_csv(RES / "metrics.csv")
    print(met_df.round(4))

    pd.DataFrame(dqn_curve).to_csv(RES / "dqn_curve.csv", index=False)
    pd.DataFrame(ppo_curve).to_csv(RES / "ppo_curve.csv", index=False)
    pd.DataFrame(sac_curve).to_csv(RES / "sac_curve.csv", index=False)
    pd.DataFrame(td3_curve).to_csv(RES / "td3_curve.csv", index=False)

    # PDF §9.7 toplu egitim teshisleri (gozlemsel; golden metriklerini etkilemez).
    curves = {"DQN": dqn_curve, "PPO": ppo_curve, "SAC": sac_curve, "TD3": td3_curve}
    diag = {n: training_diagnostics(curves[n], results[n]["backtest"].get("reward_terms_history"))
            for n in ("DQN", "PPO", "SAC", "TD3")}
    pd.DataFrame(diag).T.to_csv(RES / "training_diagnostics.csv")
    print(pd.DataFrame(diag).T.round(4))

    # PDF §11: egitilmis modelleri diske kaydet (sunumda yeniden egitmeden test).
    config_by_algo = {"DQN": asdict(__import__("config").DQNConfig()),
                      "PPO": asdict(__import__("config").PPOConfig()),
                      "SAC": asdict(__import__("config").SACConfig()),
                      "TD3": asdict(__import__("config").TD3Config())}
    for name, agent in [("DQN", dqn_agent), ("PPO", ppo_agent),
                        ("SAC", sac_agent), ("TD3", td3_agent)]:
        spec = RunSpec(name, step_days=bundle.step_days, adaptive=True,
                       agent_hp=config_by_algo[name], feature_names=tuple(bundle.feats_tr),
                       provenance=bundle.provenance)
        save_agent(agent, name, model_path(name, bundle.step_days, True),
                   horizon=f"step{bundle.step_days}", adaptive=True, run_spec=spec)
    print("Modeller kaydedildi:", MODELS)

    for name in ["DQN", "PPO", "SAC", "TD3"]:
        W = results[name]["backtest"]["weights"]
        cols = list(bundle.px_te.columns) + ["CASH"]
        pd.DataFrame(W, columns=cols).to_csv(RES / f"weights_{name}.csv", index=False)
        bt = results[name]["backtest"]
        trace = pd.DataFrame({
            "date": bt["dates"][1:],
            "gross_return": [x["gross_port_r"] for x in bt["reward_terms_history"]],
            "net_return": bt["rets"][1:], "turnover": bt["turnover"][1:],
            "period_length": bt["period_lengths"][1:],
        })
        trace.to_csv(RES / f"backtest_{name}.csv", index=False)

    (RES / "run_manifest.json").write_text(json.dumps({
        "step_days": bundle.step_days, "seed": SEED,
        "provenance": bundle.provenance.to_dict(),
    }, ensure_ascii=True, indent=2), encoding="utf-8")

    print("=" * 60)
    print("DONE. Total wall time:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    run()

```

---

## `main.py`

```python
"""Tek komutla tüm deneyi çalıştırır.

Akış:
  1) BIST 28 fiyatlarını yfinance ile indir (veya varsa cache'den yükle)
  2) DQN + PPO + SAC + TD3 eğit, test setinde tüm stratejileri backtest et
  3) Titizlik (rigor) katmanı: Deflated Sharpe + PBO + Monte-Carlo stres + reel-NAV
  4) 13 figürü (f1..f13) figures/ klasörüne kaydet

Kullanım:
  python main.py                 # her şeyi çalıştır
  python main.py --skip-data     # veri zaten indiyse
  python main.py --skip-train    # eğitim CSV'leri varsa sadece figür üret
  python main.py --skip-plots    # sadece eğit, çizim yapma
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

# Paketin kendi klasörünü path'e al -> her yerden çalıştırılabilir
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RES = HERE / "results"
FIG = HERE / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def step_data(allow_synthetic: bool = False):
    print("=" * 70)
    print("[1/4] BIST 28 fiyatları hazırlanıyor ...")
    print("=" * 70)
    from data import download_bist
    from utils.features import add_features
    prices_path = RES / "bist30_prices.csv"
    px = download_bist()
    # T5 (C5): sentetik veri koruması — akademik sonuç sentetik veriyle üretilemez.
    source = px.attrs.get("provenance", {}).get("source", "unknown")
    if source != "real":
        print(f"  UYARI: {source.upper()} veri kullaniliyor; tum ciktilar etiketlenecek.")
    px.to_csv(prices_path)
    feats = add_features(px)
    for name, f in feats.items():
        f.to_csv(RES / f"feat_{name}.csv")
    print(f"  Kaydedildi: {prices_path}  (shape={px.shape})")


def step_train(allow_synthetic: bool = False, step_days: int = 1):
    print("=" * 70)
    print("[2/4] DQN + PPO + SAC + TD3 eğitimi ve backtest ...")
    print("=" * 70)
    # T5 (C5): egitim adiminda da sentetik veri kontrolu (veri indirilmeden
    # dogrudan --skip-data ile train atlandiysa cache'den gelir).
    from data import download_bist
    px = download_bist()
    source = px.attrs.get("provenance", {}).get("source", "unknown")
    if source != "real":
        print(f"  UYARI: {source.upper()} veri ile egitim; manifest provenance tasiyacak.")
    import train as train_mod
    train_mod.run(step_days=step_days)


def step_rigor():
    print("=" * 70)
    print("[3/4] Titizlik katmanı: Deflated Sharpe + PBO + Monte-Carlo stres + reel-NAV ...")
    print("=" * 70)
    from scripts import rigor_analysis
    rigor_analysis.run()


def step_plots():
    print("=" * 70)
    print("[4/4] 13 figür üretiliyor ...")
    print("=" * 70)
    import plots
    plots.run()
    print(f"  Figürler: {FIG}")


def step_walkforward(step_days: int = 1):
    print("=" * 70)
    print("[WF] Walk-forward dogrulama (PPO, train donemi, fold-yerel olcekleme) ...")
    print("=" * 70)
    from data import (download_bist, train_test_split, download_macro, align_macro,
                      resample_to_step_days)
    from utils.features import add_features
    from utils.macro import add_macro_features
    from core.walkforward import walk_forward
    from agents import PPOAgent
    from config import PPOConfig, SEED, MacroConfig
    import numpy as np
    px = download_bist()
    px_tr_daily, _ = train_test_split(px)
    feats_daily = add_features(px)
    px_tr = resample_to_step_days(px_tr_daily, step_days)
    feats_raw = {
        k: resample_to_step_days(v.loc[px_tr_daily.index], step_days)
        for k, v in feats_daily.items()
    }

    # T6 (C6): macro/regime walk-forward'a gecirilir (fold icinde dilimlenir).
    # Forecast feature WF'de DAHIL EDILMEZ (sizinti-guvenli mevcut karar KORUNUR).
    macro_tr = regime_tr = None
    if MacroConfig.enabled:
        mraw = align_macro(download_macro(), px.index)
        mfeat = add_macro_features(mraw)
        regime_full = mfeat["regime"]
        # Raw fold panel: core.walkforward fits MacroScaler independently in
        # every train fold, preventing future-fold statistics from leaking.
        macro_tr = resample_to_step_days(mfeat.loc[px_tr_daily.index], step_days)
        regime_tr = resample_to_step_days(
            regime_full.loc[px_tr_daily.index].to_frame(), step_days
        ).iloc[:, 0].to_numpy(np.float32)

    def ppo_factory(sd, ad, seed):
        return PPOAgent(sd, ad, hidden=PPOConfig.hidden, lr_p=PPOConfig.lr_p,
                        lr_v=PPOConfig.lr_v, batch_size=PPOConfig.batch_size,
                        n_epochs=PPOConfig.n_epochs, seed=seed)

    rep = walk_forward(
        px_tr, feats_raw, ppo_factory, n_folds=3, n_iters=12, seed=SEED,
        step_days=step_days, macro=macro_tr, regime=regime_tr,
    )
    print(f"  Fold sayisi: {len(rep['folds'])}")
    for key in ("CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar"):
        print(f"  {key:<8} mean={rep['mean'].get(key, 0):+.4f}  std={rep['std'].get(key, 0):.4f}")
    print("  -> fold'lar arasi dusuk std = stabil genelleme (overfitting kontrolu)")


def main():
    ap = argparse.ArgumentParser(description="BIST 30 RL Portföy Yönetimi — tam akış")
    ap.add_argument("--skip-data",  action="store_true", help="Veri indirme adımını atla")
    ap.add_argument("--skip-train", action="store_true", help="Eğitim + backtest adımını atla")
    ap.add_argument("--skip-plots", action="store_true", help="Çizim adımını atla")
    ap.add_argument("--skip-rigor", action="store_true", help="Titizlik (DSR/PBO/stres) adımını atla")
    ap.add_argument("--walkforward", action="store_true", help="Walk-forward doğrulama çalıştır (v2)")
    # T5 (C5): sentetik veri koruması — akademik sonuç sentetik veriyle üretilemez.
    ap.add_argument("--allow-synthetic", action="store_true",
                    help="Sentetik/eksik BIST verisiyle çalışmaya izin ver (yalnız test/debug için)")
    ap.add_argument("--step-days", type=int, default=1,
                    help="Karar/rebalans araligi: ardışık BIST seansi sayisi (1..252)")
    args = ap.parse_args()
    if not 1 <= args.step_days <= 252:
        ap.error("--step-days 1..252 araliginda olmali")

    t0 = time.time()
    if not args.skip_data:  step_data(allow_synthetic=args.allow_synthetic)
    if not args.skip_train: step_train(allow_synthetic=args.allow_synthetic,
                                      step_days=args.step_days)
    if not args.skip_rigor: step_rigor()       # plot'tan ÖNCE (F11-F13 rigor çıktısını okur)
    if not args.skip_plots: step_plots()
    if args.walkforward:    step_walkforward(step_days=args.step_days)

    print("=" * 70)
    print(f"BİTTİ.  Toplam süre: {time.time() - t0:.1f} s")
    print(f"  Sonuçlar : {RES}")
    print(f"  Figürler : {FIG}")
    print("=" * 70)


if __name__ == "__main__":
    main()

```

---

## `plots.py`

```python
"""Generate all figures for the paper and the presentation."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"
FIG  = BASE / "figures"; FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
    "savefig.dpi": 200,
})
PAL = {"DQN":"#d62728", "PPO":"#1f77b4", "SAC":"#2ca02c", "TD3":"#17becf",
       "BuyHold":"#7f7f7f", "EqualWeight":"#ff7f0e", "MeanVar":"#9467bd"}
# RL ajanlari (kalin/duz cizgi); baseline'lar ince/kesik. TD3 4. ajan olarak eklendi.
RL_AGENTS = ["DQN", "PPO", "SAC", "TD3"]


def run():
    # ---------- F1: Cumulative NAV ----------
    navs = pd.read_csv(RES/"navs_aligned.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in navs.columns:
        ax.plot(navs.index, navs[c], label=c, color=PAL.get(c, None),
                lw=1.8 if c in RL_AGENTS else 1.2,
                ls="-" if c in RL_AGENTS else "--")
    ax.set_ylabel("Portföy Değeri (NAV, başlangıç=1)")
    ax.set_xlabel("Tarih"); ax.set_title("BIST 30 — RL vs Klasik Stratejiler (Test dönemi)")
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout(); plt.savefig(FIG/"f1_cumulative_nav.png"); plt.close()
    print("f1 ok")

    # ---------- F2: Drawdown ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in navs.columns:
        peak = navs[c].cummax()
        dd = (navs[c] - peak) / peak * 100
        ax.fill_between(navs.index, dd, 0, alpha=0.25, color=PAL.get(c, None))
        ax.plot(navs.index, dd, label=c, color=PAL.get(c, None), lw=1.2)
    ax.set_ylabel("Drawdown (%)"); ax.set_title("Rolling Drawdown")
    ax.legend(ncol=3, fontsize=9); plt.tight_layout()
    plt.savefig(FIG/"f2_drawdown.png"); plt.close()
    print("f2 ok")

    # ---------- F3: Rolling 60d Sharpe ----------
    rets = navs.pct_change().dropna()
    roll_sh = rets.rolling(60).apply(lambda x: np.sqrt(252) * x.mean() / (x.std() + 1e-9))
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in roll_sh.columns:
        ax.plot(roll_sh.index, roll_sh[c], label=c, color=PAL.get(c, None),
                lw=1.5 if c in RL_AGENTS else 1.0,
                ls="-" if c in RL_AGENTS else "--")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("60-Gün Rolling Sharpe"); ax.set_title("Koşullu Risk-Getiri Dengesi")
    ax.legend(ncol=2, fontsize=9); plt.tight_layout()
    plt.savefig(FIG/"f3_rolling_sharpe.png"); plt.close()
    print("f3 ok")

    # ---------- F4: Metrics bar ----------
    met = pd.read_csv(RES/"metrics.csv", index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(axes, ["CAGR", "Sharpe", "MaxDD"],
                              ["Yıllık Getiri (CAGR)", "Sharpe Oranı", "Maksimum Drawdown"]):
        colors = [PAL.get(k, "#333") for k in met.index]
        ax.bar(met.index, met[col], color=colors, edgecolor="k", lw=0.5)
        ax.set_title(title); ax.tick_params(axis='x', rotation=25)
    plt.suptitle("Test Dönemi Performans Karşılaştırması", y=1.02)
    plt.tight_layout(); plt.savefig(FIG/"f4_metrics_bar.png", bbox_inches="tight"); plt.close()
    print("f4 ok")

    # ---------- F5: Risk-Return scatter (collision-free labels) ----------
    OFF = {
        "DQN":         (10,  14),
        "PPO":         (10, -18),
        "SAC":         (14,   8),
        "MeanVar":     (14,  16),
        "BuyHold":     (-60, 30),
        "EqualWeight": (-85, -10),
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    xs_all, ys_all = [], []
    for k in met.index:
        x = met.loc[k, "Volatility"] * 100
        y = met.loc[k, "CAGR"] * 100
        xs_all.append(x); ys_all.append(y)
        ax.scatter(x, y, s=170, color=PAL.get(k, "#333"), edgecolor="k",
                   lw=1.1, zorder=5, label=k)

    for k in met.index:
        x = met.loc[k, "Volatility"] * 100
        y = met.loc[k, "CAGR"] * 100
        dx, dy = OFF.get(k, (8, 8))
        ax.annotate(
            k, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=10, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                      ec=PAL.get(k, "#333"), lw=1.0, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=PAL.get(k, "#555"),
                            lw=0.9, alpha=0.75,
                            shrinkA=0, shrinkB=6),
            zorder=6,
        )

    ax.set_xlabel("Yıllık Volatilite (%)")
    ax.set_ylabel("Yıllık Getiri (CAGR, %)")
    ax.set_title("Risk-Getiri Düzlemi")

    xmax = max(xs_all) * 1.15
    ymax = max(ys_all) * 1.25
    xs_line = np.linspace(0.1, xmax, 100)
    for sh in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ax.plot(xs_line, sh * xs_line, ls=":", color="gray", lw=0.8, alpha=0.6)
        yend = sh * xmax
        if yend < ymax:
            ax.text(xmax, yend, f"  Sharpe={sh}", fontsize=8, color="gray", va="center")
        else:
            xend = (ymax - 2) / sh
            ax.text(xend, ymax - 2, f"Sharpe={sh}", fontsize=8, color="gray", ha="center", va="bottom")

    ax.set_xlim(left=0, right=xmax * 1.08)
    ax.set_ylim(bottom=min(0, min(ys_all)) - 5, top=ymax)
    plt.tight_layout(); plt.savefig(FIG/"f5_risk_return.png", bbox_inches="tight"); plt.close()
    print("f5 ok")

    # ---------- F6: PPO / SAC / DQN / TD3 weights heatmap ----------
    for algo in ["PPO", "SAC", "DQN", "TD3"]:
        W = pd.read_csv(RES/f"weights_{algo}.csv")
        # sample every N days, transpose for display
        step = max(1, len(W) // 200)
        Wp = W.iloc[::step].T
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(Wp.values, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=min(0.3, Wp.values.max() * 1.05))
        ax.set_yticks(range(Wp.shape[0])); ax.set_yticklabels(Wp.index, fontsize=7)
        ax.set_xticks([0, Wp.shape[1] - 1]); ax.set_xticklabels([f"t=0", f"t={len(W)}"])
        ax.set_title(f"{algo} — Günlük Portföy Ağırlıkları (Test dönemi)")
        plt.colorbar(im, ax=ax, label="Ağırlık")
        plt.tight_layout(); plt.savefig(FIG/f"f6_weights_{algo}.png"); plt.close()
    print("f6 ok")

    # ---------- F7: Training curves ----------
    dqn_c = pd.read_csv(RES/"dqn_curve.csv")
    ppo_c = pd.read_csv(RES/"ppo_curve.csv")
    sac_c = pd.read_csv(RES/"sac_curve.csv")
    td3_c = pd.read_csv(RES/"td3_curve.csv")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(dqn_c["episode"], dqn_c["reward"], "o-", color=PAL["DQN"])
    axes[0].set_title("DQN — Epizot Ödülü"); axes[0].set_xlabel("Epizot"); axes[0].set_ylabel("Kümülatif Ödül")
    ax2 = axes[0].twinx()
    if "train_nav" in dqn_c.columns:
        ax2.plot(dqn_c["episode"], dqn_c["train_nav"], "s--", color="k", alpha=0.6, label="Tren NAV")
        ax2.set_ylabel("Tren NAV")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ppo_c["update"], -ppo_c["p_loss"], "o-", color=PAL["PPO"], label="surrogate obj")
    axes[1].plot(ppo_c["update"], ppo_c["v_loss"], "s--", color="k", alpha=0.5, label="value loss")
    axes[1].set_title("PPO — Surrogate & Value Loss"); axes[1].set_xlabel("Güncelleme")
    axes[1].legend(fontsize=9)

    axes[2].plot(sac_c["episode"], sac_c["train_nav"], "o-", color=PAL["SAC"])
    axes[2].set_title("SAC — Epizot Sonu NAV"); axes[2].set_xlabel("Epizot"); axes[2].set_ylabel("NAV (tren)")

    axes[3].plot(td3_c["episode"], td3_c["train_nav"], "o-", color=PAL["TD3"])
    axes[3].set_title("TD3 — Epizot Sonu NAV"); axes[3].set_xlabel("Epizot"); axes[3].set_ylabel("NAV (tren)")
    plt.tight_layout(); plt.savefig(FIG/"f7_training_curves.png"); plt.close()
    print("f7 ok")

    # ---------- F8: MDP diagram ----------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    agent_box = patches.FancyBboxPatch((0.8, 1.8), 2.4, 1.3, boxstyle="round,pad=0.1",
                                        fc="#cde", ec="#246", lw=1.5)
    env_box   = patches.FancyBboxPatch((8.8, 1.8), 2.4, 1.3, boxstyle="round,pad=0.1",
                                        fc="#ffe5c8", ec="#a64", lw=1.5)
    ax.add_patch(agent_box); ax.add_patch(env_box)
    ax.text(2.0, 2.45, "AJAN\n(DQN/PPO/SAC/TD3)", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(10.0, 2.45, "ORTAM\n(BIST 30 Piyasa)", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.annotate("", xy=(8.8, 2.7), xytext=(3.2, 2.7),
                arrowprops=dict(arrowstyle="->", lw=2, color="#246"))
    ax.text(6.0, 2.95, "a_t = π(s_t)   (portföy ağırlıkları)", ha="center", fontsize=10, color="#246")
    ax.annotate("", xy=(3.2, 2.2), xytext=(8.8, 2.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#a64"))
    ax.text(6.0, 1.95, "s_{t+1},  r_t = log(1+w_t·r_{t+1}) − λ·DD − η·|Δw|",
            ha="center", fontsize=10, color="#a64")
    ax.text(6.0, 4.5, "Portföy Yönetimi MDP Formülasyonu",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6.0, 0.7, "S: 13 özellik × 28 hisse + mevcut ağırlıklar = 397 boyut (makro dâhil; 393 = makro-öncesi V5 tabanı)        "
            "A: softmax(29-boyutlu simpleks)        γ = 0.99        T ≈ 760 gün/bölüm",
            ha="center", fontsize=9, color="#555")
    plt.tight_layout(); plt.savefig(FIG/"f8_mdp.png"); plt.close()
    print("f8 ok")

    # ---------- F9: Algo architecture sketch ----------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (name, desc) in zip(axes, [
        ("DQN", "s -> MLP(256,128) -> Q(s,a)\nayrik eylem: 6 portfoy sablonu\nTD hedefi + hedef ag"),
        ("PPO", "s -> policy -> Normal(mu, sigma) -> softmax(w)\nGAE avantaji\nclipped surrogate loss"),
        ("SAC", "s -> policy -> tanh(Normal)\ncift-Q elestirmen\nentropi-duzenlenmis amac"),
        ("TD3", "s -> Actor -> tanh (deterministik)\ncift-Q min + gecikmeli politika\nhedef-politika yumusatma")
    ]):
        ax.axis("off")
        ax.text(0.5, 0.88, name, ha="center", fontsize=20, fontweight="bold",
                color=PAL[name])
        ax.text(0.5, 0.5, desc, ha="center", fontsize=11, va="center")
        rect = patches.FancyBboxPatch((0.05, 0.08), 0.9, 0.84, boxstyle="round,pad=0.02",
                                       fc="none", ec=PAL[name], lw=2)
        ax.add_patch(rect)
    plt.suptitle("Dört Ajanın Mimari Özeti", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG/"f9_arch.png", bbox_inches="tight"); plt.close()
    print("f9 ok")

    # ---------- F10: Moving-average episode return (PDF §9.7 ogrenme egilimi) ----------
    from utils.metrics import moving_average
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for algo, c in [("DQN", dqn_c), ("PPO", ppo_c), ("SAC", sac_c), ("TD3", td3_c)]:
        if "reward" not in c.columns:
            continue
        r = c["reward"].to_numpy(dtype=float)
        ax.plot(range(len(r)), r, color=PAL[algo], alpha=0.30, lw=1.0)
        ma = moving_average(r, 5)
        if ma.size:
            x_ma = range(len(r) - ma.size, len(r))
            ax.plot(list(x_ma), ma, color=PAL[algo], lw=2.2, label=f"{algo} (MA-5)")
    ax.set_title("Hareketli Ortalama Episode Getirisi (öğrenme eğilimi)")
    ax.set_xlabel("İterasyon (episode / güncelleme)"); ax.set_ylabel("Çevre Ödülü Σr")
    ax.legend(fontsize=9); plt.tight_layout()
    plt.savefig(FIG/"f10_moving_avg_return.png"); plt.close()
    print("f10 ok")

    # ===== Titizlik (rigor) figürleri — scripts/rigor_analysis.py çıktıları (golden-güvenli) =====
    # ---------- F11: Deflated & Probabilistic Sharpe (López de Prado) + PBO ----------
    rp = RES/"rigor_metrics.csv"
    if rp.exists():
        rig = pd.read_csv(rp, index_col=0)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = np.arange(len(rig.index)); wd = 0.38
        ax.bar(x - wd/2, rig["PSR"], wd, label="PSR", color="#1f77b4", edgecolor="k", lw=0.5)
        ax.bar(x + wd/2, rig["DSR"], wd, label="DSR (deflated)", color="#d62728", edgecolor="k", lw=0.5)
        ax.axhline(0.95, ls="--", color="gray", lw=1)
        ax.text(len(x) - 0.5, 0.965, "0.95 eşiği", fontsize=8, color="gray", ha="right")
        ax.set_xticks(x); ax.set_xticklabels(rig.index, rotation=25); ax.set_ylim(0, 1.05)
        ax.set_title("Probabilistic & Deflated Sharpe (çoklu-deneme düzeltmeli) — López de Prado")
        ax.set_ylabel("Olasılık"); ax.legend(fontsize=9, loc="lower right")
        sp = RES/"rigor_summary.csv"
        if sp.exists():
            s = pd.read_csv(sp, index_col=0, header=None).iloc[:, 0]
            try:
                ax.text(0.01, 0.93, f"PBO = {float(s.get('PBO')):.2f}", transform=ax.transAxes,
                        fontsize=11, fontweight="bold", color="#555")
            except (TypeError, ValueError):
                pass
        plt.tight_layout(); plt.savefig(FIG/"f11_deflated_sharpe.png"); plt.close()
        print("f11 ok")

    # ---------- F12: Monte-Carlo stres — terminal getiri dağılımı + VaR/CVaR ----------
    mp = RES/"rigor_mc_terminal.csv"
    if mp.exists():
        mc = pd.read_csv(mp)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for col, color, lbl in [("block_bootstrap", "#2ca02c", "Blok bootstrap"),
                                ("student_t", "#9467bd", "Student-t (ağır kuyruk)")]:
            if col in mc.columns:
                d = mc[col].dropna().to_numpy() * 100
                ax.hist(d, bins=60, alpha=0.5, color=color, label=lbl, density=True)
                ax.axvline(np.quantile(d, 0.05), ls="--", color=color, lw=1.2)   # %5 VaR
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title("Monte-Carlo Stres — 1-yıl ileri terminal getiri dağılımı (en iyi RL ajan)")
        ax.set_xlabel("Terminal getiri (%)  ·  kesik çizgi = %5 VaR")
        ax.set_ylabel("Yoğunluk"); ax.legend(fontsize=9)
        plt.tight_layout(); plt.savefig(FIG/"f12_mc_stress.png"); plt.close()
        print("f12 ok")

    # ---------- F13: Nominal (TL) vs Reel (USD-bazlı) NAV — lira illüzyonu (§9.9) ----------
    nr = RES/"navs_real.csv"
    if nr.exists():
        real = pd.read_csv(nr, index_col=0, parse_dates=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        for c in [a for a in RL_AGENTS if a in navs.columns and a in real.columns]:
            ax.plot(navs.index, navs[c], color=PAL.get(c), lw=1.9, label=f"{c} nominal (TL)")
            ax.plot(real.index, real[c], color=PAL.get(c), lw=1.4, ls="--", label=f"{c} reel (USD)")
        ax.axhline(1.0, color="k", lw=0.5)
        ax.set_title("Nominal (TL) vs Reel (USD-bazlı) NAV — Lira İllüzyonu (§9.9)")
        ax.set_ylabel("NAV (başlangıç=1)"); ax.set_xlabel("Tarih"); ax.legend(fontsize=8, ncol=2)
        plt.tight_layout(); plt.savefig(FIG/"f13_real_nav.png"); plt.close()
        print("f13 ok")

    print(f"\nAll figures saved in {FIG}")


if __name__ == "__main__":
    run()

```

---

## `app.py`

```python
"""BIST 28 Portföy RL — İnteraktif Streamlit Demo (ince giriş noktası, P5).

Tek komutla çalışır:
    streamlit run app.py

Sekmeler:
  1. Veri & MDP      : 28 ticker, MDP tuple, vade preset tablosu, adaptif ödül açıklaması
  2. Eğitim          : Seçili ajanı canlı güncellenen eğri-placeholder'larıyla eğit
  3. Test            : Adım-adım oynatma; durum, aksiyon, Q/ağırlık/ödül panelleri
  4. Karşılaştırma   : Tüm ajanlar + baseline'lar; metrik tablosu, NAV, heatmap

SOLID P5 (SRP): 1000+ satırlık monolit `ui/` paketine bölündü —
  ui/state.py (session defaults) · ui/services.py (veri/eğitim/test servisleri)
  ui/charts.py (saf grafikler) · ui/sidebar.py (kontroller) · ui/tabs/ (4 sekme)
Bu dosya yalnız giriş noktası + sekme dispatch + geriye-uyumluluk re-export'larıdır.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import streamlit as st

# ---------------------------------------------------------------------
# Geriye-uyumluluk re-export'ları — eski `app.X` erişimleri çalışmaya devam
# eder (testler app._make_agent'i çağırır; dış kullanıcılar sabitleri okur).
# ---------------------------------------------------------------------
from data import BIST28, SPLIT, START, END, download_bist, train_test_split  # noqa: F401
from env.portfolio_env import (  # noqa: F401
    ACTION_NAMES, HORIZON_PRESETS, DiscretePortfolioEnv, PortfolioEnv,
)
from agents import DQNAgent, PPOAgent, SACAgent  # noqa: F401
from config import SEED, FEATURES, DQNConfig, PPOConfig, SACConfig, ForecastConfig  # noqa: F401
from core.trainer import train as train_loop  # noqa: F401

from ui.state import ASSET_NAMES, _agent_key, _init_state, env_rebalance_hint  # noqa: F401
from ui.services import (  # noqa: F401
    _compute_test_tl_snaps, _load_data, _make_agent, _make_env,
    evaluate_with_trace, train_generator,
)
from ui.charts import (  # noqa: F401
    _horizon_preset_table, _q_bar, _reward_bar, _state_top_features, _weights_pie,
)
from ui.sidebar import _sidebar_reward_editor, sidebar_controls  # noqa: F401
from ui.tabs import tab_compare, tab_mdp, tab_test, tab_train  # noqa: F401


def main():
    st.set_page_config(
        page_title="BIST 28 RL Portföy Demosu",
        page_icon="📈",
        layout="wide",
    )
    _init_state()
    algo, step_days, adaptive, hp = sidebar_controls()

    st.title("📈 BIST 28 Pekiştirmeli Öğrenme Portföy Yönetimi")
    st.caption("UYİK 2026 · DQN/PPO/SAC/TD3 · N-seans karar adimi · Adaptif odul")
    provenance = st.session_state.get("data_provenance") or {}
    if provenance:
        source = str(provenance.get("source", "unknown")).upper()
        message = f"Veri kaynagi: {source} ({provenance.get('provider', 'bilinmiyor')})"
        if source == "REAL":
            st.success(message)
        else:
            st.warning(message + " - sonuclar bu kaynak etiketiyle kaydedilir.")

    t1, t2, t3, t4 = st.tabs([
        "📐 Veri & MDP",
        "🎓 Eğitim",
        "🎬 Test (Adım-Adım)",
        "📊 Karşılaştırma",
    ])
    with t1: tab_mdp()
    with t2: tab_train(algo, step_days, adaptive, hp)
    with t3: tab_test(algo, step_days, adaptive)
    with t4: tab_compare()


if __name__ == "__main__":
    main()

```

---

## `ui/state.py`

```python
"""Session-state varsayilanlari ve anahtar yardimcilari (app.py'den tasindi, P5)."""
from __future__ import annotations

import streamlit as st

from config import DEFAULTS, DataConfig, EnvConfig, RewardConfig
from core.contracts import DataProvenance, RunSpec
from data import BIST28
from env.portfolio_env import HORIZON_PRESETS

ASSET_NAMES = BIST28 + ["CASH"]

_dc = DataConfig()
_rc = RewardConfig()


def _init_state():
    """st.session_state varsayılanları — ilk yüklemede."""
    defaults = {
        "data_loaded": False,
        "prices": None,
        "px_tr": None,
        "px_te": None,
        "feats_tr": None,
        "feats_te": None,
        "scaler": None,
        "macro_tr": None, "macro_te": None,      # v6: makro rejim blogu (z-skorlu)
        "regime_tr": None, "regime_te": None,    # v6: ham regime (V7 amplify)
        "trained_agents": {},     # {(algo, horizon, adaptive): (agent, curve)}
        "test_traces": {},        # aynı anahtar: trajectory listesi
        "baselines": None,        # dict(name -> backtest dict)
        "step_idx": 0,
        "playing": False,
        "selected_algo": "DQN",
        "horizon": "medium",
        "step_days": DEFAULTS.step_days,
        "adaptive": True,
        "initial_capital": 100_000.0,
        "train_delay": 0.0,
        "reward_cfg": {},  # kullanıcı ayarları; boş ise env preset'leri kullanır
        "n_episodes": 12,                          # parametrik episode sayısı (UI)
        "price_noise_std": EnvConfig.price_noise_std,  # fiyat gürültüsü σ (UI kontrolü)
        "episode_clean": True,   # 1. iterasyon orijinal veri (anti-ezber); UI default açık
        "train_rebalance": None,  # rebalans frekansı override (None -> vade preset'i; golden-güvenli)
        # Tarih aralığı — DataConfig tek kaynak
        "data_start": _dc.start,
        "data_split": _dc.train_end,
        "data_end": _dc.end,
        # Genişletilmiş ödül parametreleri — RewardConfig tek kaynak
        # (6 açılan parametre; değer None/boş olunca env preset default'una düşer)
        "reward_w_dsr": _rc.w_dsr,
        "reward_w_cvar": _rc.w_cvar,
        "reward_dsr_eta": _rc.dsr_eta,
        "reward_cvar_alpha": _rc.cvar_alpha,
        "reward_regime_beta": _rc.regime_beta,
        "reward_cvar_amp": _rc.cvar_amp,
        # 4 opt-in deneysel terim (default 0 → davranış değişmez)
        "reward_w_gain": 0.0,
        "reward_gain_floor": 1.0,
        "reward_w_gain_speed": 0.0,
        "reward_w_ruin_timing": 0.0,
        # Adım granülerliği — "daily" no-op (golden-güvenli)
        "granularity": "daily",
        "granularity_n_points": None,   # resample sonrası satır sayısı (uyarı için)
        "active_run_spec": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def set_active_run_spec(algo: str, step_days: int, adaptive: bool, hp: dict) -> RunSpec:
    prices = st.session_state.get("prices")
    provenance = DataProvenance.from_value(
        getattr(prices, "attrs", {}).get("provenance") if prices is not None else None)
    feats = st.session_state.get("feats_tr") or {}
    spec = RunSpec(
        algo=algo, step_days=int(step_days), adaptive=bool(adaptive),
        reward_cfg=dict(st.session_state.get("reward_cfg", {}) or {}),
        agent_hp=dict(hp or {}), data_start=str(st.session_state.get("data_start", "")),
        data_split=str(st.session_state.get("data_split", "")),
        data_end=str(st.session_state.get("data_end", "")),
        feature_names=tuple(feats.keys()), provenance=provenance,
    )
    st.session_state.active_run_spec = spec.to_dict()
    return spec


def _agent_key(algo: str, step_days, adaptive: bool) -> tuple:
    if isinstance(step_days, str):
        return (algo, step_days, bool(adaptive))  # legacy checkpoint/UI key
    spec_data = st.session_state.get("active_run_spec")
    if spec_data:
        spec = RunSpec.from_dict(spec_data)
        if spec.algo == algo and spec.step_days == int(step_days) and spec.adaptive == bool(adaptive):
            return (algo, int(step_days), bool(adaptive), spec.fingerprint)
    return (algo, int(step_days), bool(adaptive), "unbound")


def env_rebalance_hint(step_days) -> int:
    return int(step_days) if not isinstance(step_days, str) else int(HORIZON_PRESETS[step_days]["rebalance"])

```

---

## `ui/services.py`

```python
"""UI veri/egitim/test servisleri (app.py'den tasindi, P5).

Streamlit session_state'i ile core katmani arasindaki kopru: veri yukleme,
env/ajan kurulumu (core.factory'ye delege), canli egitim generator'i ve test
trajectory yakalama.
"""
from __future__ import annotations

from pathlib import Path
from math import ceil

import numpy as np
import pandas as pd
import streamlit as st

from agents.base import SupportsQValues
from config import DEFAULTS, SEED, ForecastConfig, MacroConfig, validate_train_range
from core.factory import build_agent, build_env
from core.episodes import episode_metrics
from core.persistence import (load_agent, model_path, named_model_path,
                              read_meta, save_agent, MODELS_DIR as _MODELS_DIR)
from core.trainer import train as train_loop
from data import (align_macro, download_bist, download_macro,
                  resample_to_step_days, train_test_split)
from env.portfolio_env import ACTION_NAMES
from ui.state import _agent_key
from utils.baselines import equal_weight
from utils.features import TrainScaler, add_features
from utils.macro import MacroScaler, add_macro_features
from utils.portfolio_tl import compute_tl_step

# PDF §11: egitilmis modeller diske burada kaydedilir/yuklenir (sunum kaliciligi).
# N11: MODELS_DIR ve model_path artik core.persistence'da tanimlidi; _MODELS_DIR olarak yukarda import edildi.
MODELS_DIR = _MODELS_DIR  # noqa: N816  — dis erisim icin re-export (sidebar import eder)


def save_trained_agent(algo: str, step_days: int, adaptive: bool, name: str = ""):
    """Session'daki egitilmis ajani diske kaydeder; yolu doner (yoksa None).

    N11: isim verilmisse named_model_path(name).pt kullanilir (UI isimli kayit).
         isim bossa fallback: {algo}_{horizon} ismiyle named_model_path.
         Geriye-uyumluluk: model_path(algo,horizon,adaptive) CLI/golden yolu KORUNUR —
         bu fonksiyon yalnizca UI "Kaydet" butonundan cagirilir.
    name: kullanici-girilen model adi; bos olursa "{algo}_{horizon}" kullanilir.
    """
    entry = st.session_state.trained_agents.get(_agent_key(algo, step_days, adaptive))
    if not entry or entry[0] is None:
        return None
    effective_name = name.strip() if name.strip() else f"{algo}_step{step_days}"
    path = named_model_path(effective_name)
    return save_agent(entry[0], algo, path,
                      horizon=f"step{step_days}", adaptive=adaptive,
                      name=effective_name, run_spec=st.session_state.get("active_run_spec"))


def list_saved_models() -> list[dict]:
    """models/*.pt dosyalarini tarar; her biri icin read_meta ile meta okur.

    Donus: [{path, name, saved_at, algo, horizon, adaptive}, ...]
    saved_at'e gore yeniden-eskiye sirali. Bozuk/okunamayan dosyalar atlanmaz;
    meta bos string'lerle doldurulur (read_meta guvenli default doner).
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for pt in _MODELS_DIR.glob("*.pt"):
        meta = read_meta(pt)
        # Eger name meta'da bossa dosya adini goster (eski format geriye-uyumlu)
        display_name = meta["name"] if meta["name"] else pt.stem
        results.append({
            "path": str(pt),
            "name": display_name,
            "saved_at": meta["saved_at"],
            "algo": meta["algo"],
            "horizon": meta["horizon"],
            "adaptive": meta["adaptive"],
            "step_days": int(meta.get("run_spec", {}).get("step_days", 1)),
            "legacy": bool(meta.get("legacy", True)),
        })
    # saved_at'e gore yeniden→eskiye sirala (ISO string karsilastirmasi dogru calisir)
    results.sort(key=lambda x: x["saved_at"], reverse=True)
    return results


def load_saved_agent(algo: str, horizon: str = "medium", adaptive: bool = True):
    """Diskteki modeli yukler, session_state.trained_agents'a koyar; anahtari doner.

    N11: horizon + adaptive parametreleri dosya adini belirler.
    """
    path = model_path(algo, horizon, adaptive)
    if not path.exists():
        return None
    agent, meta = load_agent(path)
    key = _agent_key(meta["algo"], meta["horizon"], meta["adaptive"])
    st.session_state.trained_agents[key] = (agent, [])   # disk'ten geldi; egitim egrisi yok
    return key


def load_saved_agent_from_path(path):
    """N11: Tam yol verilince dogrudan yukler (selectbox ile secilen model icin)."""
    from pathlib import Path as _Path
    path = _Path(path)
    if not path.exists():
        return None
    agent, meta = load_agent(path)
    if meta.get("run_spec"):
        st.session_state.active_run_spec = meta["run_spec"]
        step_days = int(meta["run_spec"].get("step_days", 1))
        key = _agent_key(meta["algo"], step_days, meta["adaptive"])
    else:
        key = _agent_key(meta["algo"], meta["horizon"], meta["adaptive"])
    st.session_state.trained_agents[key] = (agent, [])
    return key, meta


def _load_data():
    """Veri indir + granülerliğe resample + z-score scaler'ı fit et.

    Tarih aralığı session_state.data_start / data_split / data_end'den okunur
    (sidebar tarih seçici). Scaler/forecaster/MacroScaler YALNIZ px_tr'de fit
    edilir — sızıntı yok.

    Granülerlik akışı (golden-güvenli Approach 1):
      1. prices_daily  = download_bist(...)           — her zaman GÜNLÜK
      2. feats_daily   = add_features(prices_daily)   — GÜNLÜK (add_features DEĞİŞMEZ)
      3. Forecast da prices_daily üzerinde hesaplanır (GÜNLÜK)
      4. prices / feats_all_raw = resample_to_granularity(...)  — g="daily" → no-op
      5. train_test_split RESAMPLE'LANMIŞ fiyat üzerinde yapılır
      6. Scaler RESAMPLE'LANMIŞ feats_tr'de fit edilir (sızıntı korunur)
      7. Makro: download_macro GÜNLÜK → align (GÜNLÜK prices_daily.index) →
         add_macro_features (GÜNLÜK) → resample → MacroScaler resample'lı px_tr'de fit
    """
    from config import DataConfig as _DC
    _dc_defaults = _DC()
    data_start = st.session_state.get("data_start", _dc_defaults.start)
    data_split = st.session_state.get("data_split", _dc_defaults.train_end)
    data_end   = st.session_state.get("data_end",   _dc_defaults.end)
    step_days = int(st.session_state.get("step_days", DEFAULTS.step_days))

    with st.spinner("Veri indiriliyor / cache okunuyor ..."):
        prices_daily = download_bist(start=data_start, end=data_end)
    # Sentetik-veri görünürlüğü: ağ + cache yoksa download_bist SENTETİK GBM'e düşer
    # (px.attrs["synthetic"]=True). Bu durumda sonuçlar GERÇEK DEĞİLDİR ve NaN/anlamsız
    # değerler çıkabilir → kullanıcıya AÇIK uyarı (sessiz NaN yerine).
    source = prices_daily.attrs.get("provenance", {}).get("source", "unknown")
    if source != "real":
        st.warning(
            f"Veri kaynagi {source.upper()}. Egitim ve test devam eder; modeller, "
            "CSV'ler ve ekran bu provenance etiketini korur."
        )

    # --- 1. Feature'lar her zaman GÜNLÜK hesaplanır (add_features DEĞİŞMEZ) ---
    feats_daily = add_features(prices_daily)

    # --- 2. Forecast feature GÜNLÜK prices_daily üzerinde fit edilir ---
    if ForecastConfig.enabled:                     # v2: forecast feature (train-only fit)
        from forecast.forecaster import build_forecast_feature
        # Forecast fit için geçici günlük train split kullanılır (sızıntısız)
        _px_tr_daily, _ = train_test_split(prices_daily, split=data_split)
        feats_daily["forecast"] = build_forecast_feature(
            prices_daily, _px_tr_daily,
            window=ForecastConfig.window, conv_ch=ForecastConfig.conv_ch,
            hidden=ForecastConfig.hidden, epochs=ForecastConfig.epochs,
            lr=ForecastConfig.lr, batch=ForecastConfig.batch, seed=SEED)

    # Split daily data first, then independently select N-session endpoints. A
    # period can never mix train and test observations.
    px_tr_daily, px_te_daily = train_test_split(prices_daily, split=data_split)
    px_tr = resample_to_step_days(px_tr_daily, step_days)
    px_te = resample_to_step_days(px_te_daily, step_days)
    prices = pd.concat([px_tr, px_te])
    prices.attrs.update(prices_daily.attrs)
    feats_tr_raw = {
        k: resample_to_step_days(v.loc[px_tr_daily.index], step_days) for k, v in feats_daily.items()
    }
    feats_te_raw = {
        k: resample_to_step_days(v.loc[px_te_daily.index], step_days) for k, v in feats_daily.items()
    }
    ok, message = validate_train_range(len(px_tr), step_days=step_days)
    if not ok:
        raise ValueError(message)

    # SIZINTI KORUMASI: scaler YALNIZ eğitim kısmında fit edilir, test'e transform uygulanır.
    scaler = TrainScaler().fit(feats_tr_raw)
    st.session_state.prices = prices
    st.session_state.px_tr = px_tr
    st.session_state.px_te = px_te
    st.session_state.feats_tr = scaler.transform(feats_tr_raw)
    st.session_state.feats_te = scaler.transform(feats_te_raw)
    st.session_state.scaler = scaler

    # Nokta sayısını kaydet (uyarı için sidebar/tab kullanabilir)
    st.session_state.granularity_n_points = len(prices)
    st.session_state.data_provenance = prices_daily.attrs.get("provenance", {})

    # --- 6. Makro: GÜNLÜK indir → GÜNLÜK align → add_macro_features (GÜNLÜK) → resample ---
    # MacroScaler RESAMPLE'LANMIŞ px_tr'de fit edilir (sızıntı korunur).
    macro_tr = macro_te = regime_tr = regime_te = None
    if MacroConfig.enabled:
        mfeat_daily = add_macro_features(
            align_macro(download_macro(start=data_start, end=data_end), prices_daily.index)
        )
        # Makro + regime'i granülerliğe resample et
        regime_daily = mfeat_daily["regime"].to_frame("regime")
        mtr_daily, mte_daily = train_test_split(mfeat_daily, split=data_split)
        rtr_daily, rte_daily = train_test_split(regime_daily, split=data_split)
        mtr = resample_to_step_days(mtr_daily, step_days)
        mte = resample_to_step_days(mte_daily, step_days)
        scaler_m = MacroScaler().fit(mtr)
        macro_tr = scaler_m.transform(mtr).to_numpy(np.float32)
        macro_te = scaler_m.transform(mte).to_numpy(np.float32)
        regime_tr = resample_to_step_days(rtr_daily, step_days)["regime"].to_numpy(np.float32)
        regime_te = resample_to_step_days(rte_daily, step_days)["regime"].to_numpy(np.float32)
    st.session_state.macro_tr = macro_tr
    st.session_state.macro_te = macro_te
    st.session_state.regime_tr = regime_tr
    st.session_state.regime_te = regime_te
    st.session_state.data_loaded = True


def _make_env(is_train: bool, algo: str, step_days: int, adaptive: bool, max_steps: int,
              cash_daily_rate: float | None = None, *,
              force_noise: bool = False, noise_eval: float = 0.0,
              seed_override: int | None = None):
    """UI ortam kurulumu — session_state'i okuyup core.factory.build_env'e delege eder (P3).

    N12: cash_daily_rate None verilirse session_state.cash_daily_rate okunur;
         o da yoksa env kendi config default'unu kullanir.
    N10: is_train=True ise max_steps egitim episode uzunlugu (sidebar slider'dan);
         is_train=False ise max_steps=10_000 (tam test donemi — degistirilmez).
    """
    px_df = st.session_state.px_tr if is_train else st.session_state.px_te
    feats = st.session_state.feats_tr if is_train else st.session_state.feats_te
    macro = st.session_state.get("macro_tr" if is_train else "macro_te")
    regime = st.session_state.get("regime_tr" if is_train else "regime_te")
    start_index = None
    if not is_train:
        context = min(len(st.session_state.px_tr), max(21, ceil(DEFAULTS.minvol_window / step_days)) + 1)
        px_context = st.session_state.px_tr.iloc[-context:]
        px_df = pd.concat([px_context, px_df])
        counts = np.concatenate([
            np.asarray(st.session_state.px_tr.attrs.get(
                "session_counts", np.ones(len(st.session_state.px_tr), dtype=int)))[-context:],
            np.asarray(st.session_state.px_te.attrs.get(
                "session_counts", np.ones(len(st.session_state.px_te), dtype=int))),
        ])
        px_df.attrs.update(st.session_state.px_te.attrs)
        px_df.attrs["session_counts"] = counts
        feats = {k: pd.concat([st.session_state.feats_tr[k].iloc[-context:], v])
                 for k, v in st.session_state.feats_te.items()}
        if macro is not None:
            macro = np.concatenate([st.session_state.macro_tr[-context:], macro], axis=0)
        if regime is not None:
            regime = np.concatenate([st.session_state.regime_tr[-context:], regime], axis=0)
        start_index = context
    # Eğitimde UI'dan okunan σ geçilir; eval'de None → env gürültüyü zaten
    # random_start=False ile kapatır, ama yine de None göndererek kasıtsız gürültüyü engelle.
    noise_std = (st.session_state.get("price_noise_std") if is_train else None)
    # Episode-clean (kullanici istegi: 1. iterasyon ORIJINAL veri, 2+ farkli noise'lu).
    # Yalniz egitimde + UI toggle (default True) acikken. Eval'de noise zaten kapali -> etkisiz.
    ep_clean = bool(is_train and st.session_state.get("episode_clean", True))
    # Rebalans frekansı override (sidebar): None -> preset (golden-güvenli). Train+eval'e
    # AYNI değer uygulanır (model hangi frekansla eğitildiyse onunla test edilsin).
    # N12: nakit faiz — önce parametre, sonra session_state, sonra env default (None).
    if cash_daily_rate is None:
        cash_daily_rate = st.session_state.get("cash_daily_rate", None)
    return build_env(
        algo, px_df, feats, adaptive=adaptive, max_steps=max_steps,
        random_start=is_train,                     # v2: egitimde rastgele pencere, eval'de sabit
        seed=(int(seed_override) if seed_override is not None else SEED),  # noise-episode: per-episode tohum
        reward_overrides=st.session_state.get("reward_cfg", {}) or {},
        price_noise_std=(float(noise_eval) if force_noise else noise_std),  # UI σ (train) / noise-episode (eval)
        force_price_noise=bool(force_noise),       # gurultu-artirimli coklu-episode: eval'de gurultu ac
        episode_clean=ep_clean,                    # 1. iterasyon orijinal (anti-ezber)
        macro=macro, regime=regime,                # v6: makro rejim blogu + ham regime
        cash_daily_rate=cash_daily_rate,           # N12: UI nakit faiz oranı
        rebalance_freq=1, step_days=int(step_days),
        gamma=float(st.session_state.get("gamma_daily", DEFAULTS.gamma)),
        mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window,
        start_index=start_index,
    )


def evaluate_noise_episodes_ui(agent, algo: str, step_days: int, adaptive: bool, *,
                               n_episodes: int = 5, noise_std: float = 0.01) -> list:
    """Egitilmis ajani test araligi boyunca SIRAYLA birden cok episode'da kosturur;
    HER episode icin tam adim-adim trace toplar (Test sekmesindeki detay tablosu icin).

    Episode 0 = orijinal (gurultusuz) referans — normal testle AYNI kurulum (_make_env
    eval yolu: context penceresi + start_index) -> episode 0 normal test sonucuyla
    ortusur. Episode 1..N = ayni test serisine getiri-seviyesinde Gauss gurultusu
    (force_price_noise; episode basina FARKLI tohum) eklenmis YENI patikalar (anti-ezber).
    Her episode TUM veri tarih araligini kapsar ve SIRAYLA kosar — biri tam BITMEDEN
    (N gun varsa N adim) digeri BASLAMAZ. Donen her episode dict'i: episode, noise_std,
    steps, nav, dates, trace (adim-adim), + final_nav/total_return/max_drawdown/sharpe.
    """
    results = []
    for i in range(int(n_episodes)):
        nstd = 0.0 if i == 0 else float(noise_std)
        env = _make_env(False, algo, step_days, adaptive, max_steps=10_000,
                        force_noise=(nstd > 0.0), noise_eval=nstd, seed_override=SEED + i)
        # Tam aralik, adim-adim trace (tekli-test ile AYNI dongey reuse eder) — episode
        # done'a (veri sonu) kadar kosar; sonraki episode ancak bu bittikten sonra baslar.
        trace = _run_trace_loop(env, agent, algo, light=True)
        nav = np.asarray(env.nav_history, dtype=float)
        rets = np.asarray(getattr(env, "ret_history", []), dtype=float)
        results.append(dict(
            episode=i, noise_std=nstd, steps=len(trace),
            nav=nav, dates=[t["date"] for t in trace], trace=trace,
            **episode_metrics(nav, rets),
        ))
    return results


def _make_agent(algo: str, state_dim: int, action_dim: int, hp: dict):
    """Shim — SOLID P3: tek dogruluk kaynagi core.factory.build_agent.
    (test_config_wiring app._make_agent'i cagirir; app.py bunu re-export eder.)"""
    return build_agent(algo, state_dim, action_dim, hp)


# =====================================================================
# Eğitim jeneratörü — canlı UI için episod başına yield
# =====================================================================
def train_generator(algo: str, step_days: int, adaptive: bool, hp: dict,
                    rollout_len: int = 400, resume_agent=None,
                    n_episodes: int | None = None):
    """Episod/update başına bir telemetri kaydı yield eder.

    n_episodes verilirse (UI'dan gelir) tam o kadar episode/update koşar ve
    generator kendiliğinden biter. None ise sonsuz akış — tüketici (tab_train)
    'Eğitimi Durdur' butonuyla keser.

    resume_agent verilirse (G4) yeni ajan kurulmaz; durdurulan ajan AYNI
    ağırlık/optimizer/replay buffer'la kaldığı yerden öğrenmeye devam eder.

    N10: session_state.train_max_steps varsa o değer kullanılır (sidebar slider);
         yoksa algo'ya özgü sabit fallback.
    """
    _algo_defaults = {"DQN": 252, "PPO": 10_000, "SAC": 1200, "TD3": 1200}
    max_steps = max(1, len(st.session_state.px_tr))
    env = _make_env(True, algo, step_days, adaptive, max_steps=max_steps)
    if resume_agent is not None:
        agent = resume_agent
    else:
        action_dim = env.n_discrete if algo == "DQN" else env.action_dim
        agent = _make_agent(algo, env.state_dim, action_dim, hp)
    # Başarı kıyası için tren EW NAV'ı (core.trainer success'i bununla hesaplar)
    ew_tr = equal_weight(st.session_state.px_tr)["nav"]
    # n_episodes=None → sonsuz akış; int → tam o kadar episode/update sonra generator biter.
    # core.trainer.train ajan tipine göre dispatch eder; telemetri dict'i CLI ile ortaktır.
    yield from train_loop(agent, env, n_iters=n_episodes, rollout_len=rollout_len, ew_nav=ew_tr)


# =====================================================================
# Test dönemi — adım adım trajectory yakalama
# =====================================================================
def evaluate_with_trace(agent, algo: str, step_days: int, adaptive: bool) -> list:
    env = _make_env(False, algo, step_days, adaptive, max_steps=10_000)
    return _run_trace_loop(env, agent, algo)


def _run_trace_loop(env, agent, algo: str, light: bool = False) -> list:
    """Bir env'i adim-adim kosturup TAM trace dondurur (ORTAK cekirdek: tekli test
    playback'i + gurultu-artirimli per-episode tablo ayni dongey reuse eder). Episode
    veri sonuna (done) kadar SIRAYLA kosar. light=True -> 'state'/'q_values' atlanir
    (cok-episode bellek; detay tablosu icin gereksiz). light=False -> tam trace."""
    s, _ = env.reset()
    trace = []
    done = trunc = False
    # P5 (ISP): hasattr yoklamasi yerine resmi Protocol — ayni semantik, acik niyet.
    has_q = (not light) and isinstance(agent, SupportsQValues)  # yalnizca DQN introspeksiyonu
    while not (done or trunc):
        decision_date = env.dates[env.t]
        weights_before = env.w.copy()
        state_snapshot = None if light else s.copy()

        q_vals = agent.q_values(s) if has_q else None
        a = agent.act_eval(s)                       # ajan-agnostik (BaseAgent.act_eval)
        if isinstance(a, (int, np.integer)):
            action_idx = int(a)
            action_name = (ACTION_NAMES[action_idx]
                           if 0 <= action_idx < len(ACTION_NAMES) else str(action_idx))
        else:
            action_idx = None
            action_name = f"{algo} (sürekli aksiyon)"
        s2, r, done, trunc, info = env.step(a)

        trace.append({
            "step": len(trace),
            "date": str(pd.Timestamp(info["date"]).date()),
            "decision_date": str(pd.Timestamp(decision_date).date()),
            "state": state_snapshot,
            "action_idx": action_idx,
            "action_name": action_name,
            "q_values": q_vals,
            "weights_before": weights_before,
            "target_weights": info["target_weights"].copy(),
            "weights_after": env.w.copy(),
            "reward_terms": info["reward_terms"],
            "period_length": info["period_length"],
            "nav": float(env.nav),
            "prices_t": info["prices_t"].copy(),
        })
        s = s2
    return trace


def _compute_test_tl_snaps(trace: list, initial_capital: float) -> list:
    """Test trace'i için adım-adım TL türevleri — session_state.initial_capital değişince
    sayfa her yeniden render'da bu yeniden hesaplanır (ucuz, ~1000 adım)."""
    N_risky = trace[0]["prices_t"].shape[0] if trace else 0
    holding_days_prev = np.zeros(N_risky, dtype=np.int64)
    prev_portfolio_tl = float(initial_capital)
    snaps = []
    for t in trace:
        snap = compute_tl_step(
            nav=t["nav"],
            w_now=t["weights_after"],
            w_prev=t["weights_before"],
            prices_t=t["prices_t"],
            initial_capital=initial_capital,
            prev_portfolio_tl=prev_portfolio_tl,
            holding_days_prev=holding_days_prev,
            tx_cost_rate=float(t["reward_terms"].get("tx_cost", 0.0)),
            target_weights=t.get("target_weights"),
            holding_period_days=int(t.get("period_length", 1)),
        )
        snap["w_now"] = t["weights_after"]
        snaps.append(snap)
        holding_days_prev = snap["holding_days"]
        prev_portfolio_tl = snap["portfolio_tl"]
    return snaps

```

---

## `ui/charts.py`

```python
"""Saf grafik/tablo ureticileri (app.py'den tasindi, P5) — UI-durumsuz.

Hicbiri st.session_state okumaz; girdi -> figur/DataFrame donusumu yapar.
Bu saflik onlari Streamlit'siz test edilebilir kilar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import FEATURES, MacroConfig
from data import BIST28
from env.portfolio_env import ACTION_NAMES, HORIZON_PRESETS
from ui.state import ASSET_NAMES


def _weights_pie(weights: np.ndarray, title: str = "Portföy Ağırlıkları"):
    # Güvenlik ağı: evren beklenen 28+nakit'ten farklıysa (ör. yfinance kısmi veri
    # döndürdü) etiketleri ağırlık uzunluğuna hizala — UI çökmemeli (ASSET_NAMES sabit 29).
    w = list(np.asarray(weights).ravel())
    names = list(ASSET_NAMES)
    if len(names) != len(w):
        names = names[:max(0, len(w) - 1)] + ["CASH"]
        names = (names + [f"A{i}" for i in range(len(names), len(w))])[:len(w)]
    df = pd.DataFrame({"asset": names, "weight": w})
    df = df[df["weight"] > 0.005]  # çok küçük dilimleri gizle
    fig = px.pie(df, names="asset", values="weight", title=title, hole=0.35)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=380, showlegend=False)
    return fig


def _q_bar(q_values: np.ndarray, chosen: int):
    colors = ["#d62728" if i == chosen else "#888" for i in range(len(q_values))]
    fig = go.Figure(
        go.Bar(x=ACTION_NAMES[:6], y=q_values, marker_color=colors,
               text=[f"{v:+.3f}" for v in q_values], textposition="outside")
    )
    fig.update_layout(title="Q(s, ·) — seçim kırmızı", height=320,
                      yaxis_title="Q-değeri", margin=dict(t=40, b=40))
    return fig


def _reward_bar(rt: dict):
    names = ["log_return", "tx_cost", "dd_penalty", "cvar", "dsr", "total"]
    vals = [rt["log_return"], -rt["tx_cost"], -rt["drawdown_penalty"],
            -rt.get("cvar_penalty", 0.0), rt.get("dsr_term", 0.0), rt["total"]]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors,
                           text=[f"{v:+.5f}" for v in vals], textposition="outside"))
    fig.update_layout(title="Ödül terimleri (bu adım)", height=280,
                      margin=dict(t=40, b=30))
    return fig


def _state_top_features(state_vec: np.ndarray, top_k: int = 8) -> pd.DataFrame:
    """Durum vektöründen top-k öznitelik çıkar (mutlak değer sıralı). Feature
    sayısı config.FEATURES'tan dinamik okunur (v2: 12 özellik)."""
    n_assets = len(BIST28)
    # v6: state = [F teknik × n_assets] + [M makro] + [n_assets+1 agirlik]. Makro blogunu
    # cikararak teknik oznitelik sayisi F'i state uzunlugundan turet.
    M = len(MacroConfig.features) if MacroConfig.enabled else 0
    F = (len(state_vec) - M - (n_assets + 1)) // n_assets   # 12 (PPO) ya da 13 (DQN/SAC)
    names = list(FEATURES) + ["forecast"]
    feat_names = (names + [f"f{i}" for i in range(F)])[:F]
    snap = state_vec[: F * n_assets].reshape(F, n_assets)
    rows = []
    for fi, fname in enumerate(feat_names):
        for ai, aname in enumerate(BIST28):
            rows.append((fname, aname, float(snap[fi, ai])))
    df = pd.DataFrame(rows, columns=["feature", "asset", "z-score"])
    df["|z|"] = df["z-score"].abs()
    df = df.sort_values("|z|", ascending=False).head(top_k).drop(columns="|z|")
    return df.reset_index(drop=True)


def _horizon_preset_table() -> pd.DataFrame:
    rows = []
    for h, p in HORIZON_PRESETS.items():
        rows.append({
            "Vade": {"short": "Kısa", "medium": "Orta", "long": "Uzun"}[h],
            "Rebalans (gün)": p["rebalance"],
            "Momentum penceresi": p["mom_window"],
            "Min-vol penceresi": p["minvol_window"],
            "η (tx cost)": p["eta"],
            "λ (DD penalty)": p["lam"],
            "τ (DD eşiği)": p["tau"],
            "γ (discount)": p["gamma"],
        })
    return pd.DataFrame(rows)

```

---

## `ui/sidebar.py`

```python
"""Kontrol paneli (sidebar) — app.py'den tasindi (P5)."""
from __future__ import annotations

import datetime

import streamlit as st

from config import (DEFAULTS, SEED, STEP_DAYS_MAX, DataConfig, EnvConfig, RewardConfig,
                    cash_daily_rate as _cash_daily_rate)
from env.portfolio_env import HORIZON_PRESETS
from core.persistence import MODELS_DIR, model_path
from ui.services import (_load_data, list_saved_models, load_saved_agent,
                         load_saved_agent_from_path, save_trained_agent)
from ui.state import _agent_key, set_active_run_spec

_dc = DataConfig()
_rc = RewardConfig()

# Tarih aralığı sınırları (UI kısıtı)
_DATE_MIN = datetime.date(2015, 1, 1)
_DATE_MAX = datetime.date(2024, 12, 31)

# SonarCloud S1192: 3+ kez tekrar eden UI literal'leri tek sabitte topla.
_EP_HINT = ("Eğitim **N episode** koşar (sidebar'daki 'Episode sayısı' değeri); "
            "'Eğitimi Durdur' ile erken kesilebilir.")
_LBL_POLICY_LR = "Policy LR"
_LBL_BATCH = "Batch"


def _sidebar_reward_editor(preset: dict):
    """⚖️ Ödül & Ceza katsayıları düzenleyicisi — session_state.reward_cfg'i günceller.

    None/boş değerler env'in preset default'larına düşer. Kullanıcı 'Preset'e dön'
    ile tüm override'ları sıfırlayabilir.
    """
    cfg = st.session_state.setdefault("reward_cfg", {})
    with st.sidebar.expander("⚖️ Ödül & Ceza Katsayıları", expanded=False):
        st.caption("Ödül = log-getiri − η·turnover − λ·max(0, DD−τ) − iflas cezası")

        if st.button("↺ Preset'e dön (tüm override'ları sıfırla)",
                     key="reset_reward_cfg", width='stretch'):
            st.session_state.reward_cfg = {}
            st.rerun()

        st.markdown("**Ödül terimleri** (vade preset'i default olarak)")
        cfg["eta_base"] = st.number_input(
            "η — İşlem (turnover) maliyeti katsayısı",
            value=float(cfg.get("eta_base", preset["eta"])),
            min_value=0.0, max_value=0.1, step=0.0001, format="%.4f",
            help="Ağırlık değişiminin L1 normu bu katsayı ile çarpılıp ödülden düşülür "
                 "(ve NAV'ı azaltır). Büyütünce ajan daha az işlem yapar.",
        )
        cfg["lambda_base"] = st.number_input(
            "λ — Drawdown (DD) ceza katsayısı",
            value=float(cfg.get("lambda_base", preset["lam"])),
            min_value=0.0, max_value=10.0, step=0.05, format="%.3f",
            help="max(0, DD − τ) bu katsayı ile çarpılıp ödülden düşülür.",
        )
        cfg["tau_base"] = st.number_input(
            "τ — DD eşiği (oran)",
            value=float(cfg.get("tau_base", preset["tau"])),
            min_value=0.0, max_value=0.5, step=0.005, format="%.3f",
            help="Tepe-den DD > τ olduğunda ceza başlar. 0.05 = %5'lik DD toleransı.",
        )

        st.markdown("**İflas (simülasyonu durdurma)**")
        cfg["bankruptcy_nav"] = st.number_input(
            "İflas NAV eşiği",
            value=float(cfg.get("bankruptcy_nav", EnvConfig.bankruptcy_nav)),
            min_value=0.0, max_value=0.9, step=0.01, format="%.2f",
            help="NAV bu eşiğin altına düşerse episod iflas olarak sonlandırılır. "
                 "Örn. 0.01 = başlangıç sermayesinin %1'ine inmek.",
        )
        cfg["bankruptcy_penalty"] = st.number_input(
            "İflas ek ceza değeri",
            value=float(cfg.get("bankruptcy_penalty", EnvConfig.bankruptcy_penalty)),
            min_value=0.0, max_value=1000.0, step=1.0, format="%.1f",
            help="İflas anında toplam ödüle eklenen negatif terim. Log-ölçeğinde büyük değer "
                 "(normal adım ödülü ~±0.01). Ajan iflasa gitmemeyi öğrenir.",
        )

        st.markdown("**Adaptif şekillendirici hedefleri**")
        cfg["vol_target"] = st.number_input(
            "vol_target (hedef realize vol)",
            value=float(cfg.get("vol_target", EnvConfig.vol_target)),
            min_value=0.0001, max_value=0.5, step=0.001, format="%.4f",
            help="Adaptif mod açıkken λ ve τ bu hedefe göre ölçeklenir.",
        )
        cfg["turnover_target"] = st.number_input(
            "turnover_target (hedef turnover)",
            value=float(cfg.get("turnover_target", EnvConfig.turnover_target)),
            min_value=0.001, max_value=1.0, step=0.005, format="%.3f",
            help="Adaptif mod açıkken η bu hedefe göre ölçeklenir.",
        )
        cfg["ema_alpha"] = st.slider(
            "EMA α (adaptif hafıza)",
            min_value=0.001, max_value=0.5, value=float(cfg.get("ema_alpha", EnvConfig.ema_alpha)),
            step=0.005, format="%.3f",
            help="Büyük α = daha hızlı uyum, küçük α = daha stabil.",
        )

        st.markdown("**DSR & CVaR risk terimleri**")
        cfg["w_dsr"] = st.number_input(
            "w_dsr — Diferansiyel Sharpe ağırlığı",
            value=float(cfg.get("w_dsr", _rc.w_dsr)),
            min_value=0.0, max_value=0.2, step=0.005, format="%.3f",
            help="DSR terimi ağırlığı: online risk-ayarlı Sharpe gradyanı. "
                 "0 = kapalı, 0.05 = hafif etkin.",
        )
        cfg["w_cvar"] = st.number_input(
            "w_cvar — CVaR kuyruk cezası ağırlığı",
            value=float(cfg.get("w_cvar", _rc.w_cvar)),
            min_value=0.0, max_value=0.2, step=0.005, format="%.3f",
            help="CVaR (Conditional Value at Risk) ceza ağırlığı. "
                 "0 = kapalı; kriz dönemlerinde regime_beta ile amplify edilir.",
        )
        cfg["dsr_eta"] = st.number_input(
            "dsr_eta — DSR EWMA oranı",
            value=float(cfg.get("dsr_eta", _rc.dsr_eta)),
            min_value=0.001, max_value=0.1, step=0.001, format="%.3f",
            help="Diferansiyel Sharpe hesabındaki EWMA pencere oranı. "
                 "Küçük = yavaş adaptasyon, büyük = hızlı.",
        )
        cfg["cvar_alpha"] = st.number_input(
            "cvar_alpha — CVaR kuyruk yüzdesi",
            value=float(cfg.get("cvar_alpha", _rc.cvar_alpha)),
            min_value=0.01, max_value=0.2, step=0.005, format="%.3f",
            help="CVaR için kuyruk yüzdesi (α). 0.05 = en kötü %5'lik getiri ortalaması.",
        )
        cfg["regime_beta"] = st.number_input(
            "regime_beta — Kriz amplifikasyon gücü",
            value=float(cfg.get("regime_beta", _rc.regime_beta)),
            min_value=0.0, max_value=5.0, step=0.1, format="%.2f",
            help="CVaR cezasını kriz rejiminde büyüten çarpan. "
                 "0 = rejim bağımsız, 5 = kriz anında 6× ceza.",
        )
        cfg["cvar_amp"] = st.number_input(
            "cvar_amp — Rejim amplifikasyon üsteli",
            value=float(cfg.get("cvar_amp", _rc.cvar_amp)),
            min_value=0.5, max_value=3.0, step=0.1, format="%.2f",
            help="κ = w_cvar·(1 + regime_beta·max(0,regime))^cvar_amp formülündeki üstel. "
                 "1.0 = doğrusal amplifikasyon.",
        )

        st.caption("⚠️ Bu ayarları değiştirdikten sonra ajanları **yeniden eğitmek** "
                   "anlamlı olur; eski ajan farklı ortamda öğrenilmiştir.")

    with st.sidebar.expander("🧪 Deneysel ödül terimleri (opt-in, varsayılan kapalı)",
                             expanded=False):
        st.caption(
            "Bu terimler varsayılan 0 ile tamamen kapalıdır — aktif etmek için "
            "sıfırdan farklı değer girin. Yeni ajan eğitmeden etkisi görülmez."
        )
        cfg["w_gain"] = st.number_input(
            "w_gain — Kazanç-çarpanı ödülü ağırlığı",
            value=float(cfg.get("w_gain", _rc.w_gain)),
            min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
            help="NAV gain_floor eşiğini aştığında verilen ödül ağırlığı. "
                 "2× → w_gain ödül, 3× → 2·w_gain ödül. 0 = kapalı.",
        )
        cfg["gain_floor"] = st.number_input(
            "gain_floor — Ödül eşiği (NAV)",
            value=float(cfg.get("gain_floor", _rc.gain_floor)),
            min_value=1.0, max_value=2.0, step=0.05, format="%.2f",
            help="w_gain ödülünün başlayacağı NAV çarpanı. "
                 "1.0 = başlangıçtan itibaren, 1.5 = %50 büyüme sonrası.",
        )
        cfg["w_gain_speed"] = st.number_input(
            "w_gain_speed — Hız bonusu ağırlığı",
            value=float(cfg.get("w_gain_speed", _rc.w_gain_speed)),
            min_value=0.0, max_value=2.0, step=0.05, format="%.2f",
            help="Erken büyümeye daha yüksek ödül veren hız faktörü. "
                 "0 = zamandan bağımsız, pozitif = erken kazanç daha değerli.",
        )
        cfg["w_ruin_timing"] = st.number_input(
            "w_ruin_timing — İflas-timing ceza ağırlığı",
            value=float(cfg.get("w_ruin_timing", _rc.w_ruin_timing)),
            min_value=0.0, max_value=3.0, step=0.1, format="%.2f",
            help="Erken iflas anına daha sert ceza uygular. "
                 "0 = düz (flat) iflas_penalty, pozitif = erken iflasa üstel ceza.",
        )


def sidebar_controls():
    st.sidebar.title("⚙️ Kontrol Paneli")

    st.sidebar.subheader("💰 Başlangıç Sermayesi")
    st.session_state.initial_capital = st.sidebar.number_input(
        "Başlangıç TL Parası",
        min_value=1_000.0, value=float(st.session_state.initial_capital),
        step=10_000.0, format="%.0f",
        help="Eğitim ve test boyunca bu bakiye üzerinden lot/TL hesaplanır. "
             "Değeri istediğin zaman değiştirebilirsin — eğitilmiş ajanı yeniden eğitmen gerekmez.",
    )

    st.sidebar.subheader("⏱ Eğitim Hızı")
    st.session_state.train_delay = st.sidebar.slider(
        "İter arası gecikme (sn)", 0.0, 3.0, float(st.session_state.train_delay), step=0.1,
        help="0 = tam hız. Büyütünce iter'ler arasında yapay bekleme olur; "
             "canlı adım tablolarını rahat okumak için kullan.",
    )

    st.session_state.n_episodes = st.sidebar.number_input(
        "Episode / iterasyon sayısı",
        min_value=1, max_value=1000,
        value=int(st.session_state.n_episodes),
        step=1,
        help="Eğitim tam bu kadar episode/iterasyon koşar; '1. iterasyon orijinal' açıksa "
             "1. iterasyon gürültüsüz ORİJİNAL, 2.–N. iterasyonlar FARKLI gürültülü "
             "realizasyondur. 'Eğitimi Durdur' erken kesebilir. (PPO için birim 'update'.)",
    )

    st.session_state.price_noise_std = st.sidebar.slider(
        "Fiyat gürültüsü σ (anti-ezber)",
        min_value=0.0, max_value=0.01,
        value=float(st.session_state.price_noise_std),
        step=0.0005, format="%.4f",
        help="σ = gürültünün STANDART SAPMASI (ölçek) — eklenen SABİT sayı DEĞİL. Her train "
             "ADIMINDA her hisseye N(0, σ)'dan ÇEKİLEN AYRI bir rastgele sayı eklenir "
             "(rng.normal, env-yerel); ardışık adımlar ve her episode farklı realizasyon → "
             "ezberi önler. Eğitim-YALNIZ; eval'de hep KAPALI. 0 = kapalı.",
    )

    st.session_state.episode_clean = st.sidebar.checkbox(
        "1. iterasyon orijinal veri (anti-ezber)",
        value=bool(st.session_state.get("episode_clean", True)),
        help="Açık (varsayılan): 1. iterasyon gürültüsüz ORİJİNAL fiyatlarla; 2.–N. "
             "iterasyonlar her biri N(0, σ)'dan FARKLI gürültü realizasyonuyla eğitilir "
             "(kullanıcı isteği — '1 iterasyon orijinal, kalanı noise'lu'). Kapalı: tüm "
             "iterasyonlar gürültülü (klasik). σ=0 ise etkisiz.",
    )

    st.sidebar.divider()
    _algos = ["DQN", "PPO", "SAC", "TD3"]
    _cur = st.session_state.selected_algo if st.session_state.selected_algo in _algos else "DQN"
    algo = st.sidebar.radio("Ajan", _algos, index=_algos.index(_cur))
    st.session_state.selected_algo = algo

    step_days = int(st.sidebar.number_input(
        "Karar adimi (BIST seansi)", min_value=1, max_value=STEP_DAYS_MAX,
        value=int(st.session_state.get("step_days", DEFAULTS.step_days)), step=1,
        help="Ajan her N BIST seansinin donem-sonu kapanisinda karar verir ve rebalans yapar.",
    ))
    if step_days != int(st.session_state.get("step_days", DEFAULTS.step_days)):
        st.session_state.step_days = step_days
        st.session_state.data_loaded = False
        st.session_state.trained_agents = {}
        st.session_state.test_traces = {}
        st.session_state.baselines = None
        st.sidebar.warning("Adim uzunlugu degisti; veriyi yeniden yukleyin.")
    else:
        st.session_state.step_days = step_days
    preset = HORIZON_PRESETS["medium"]  # legacy-equivalent reward defaults
    st.sidebar.caption(f"Her {step_days} seansta karar + rebalans; episode = tum train araligi")

    """LEGACY_UI_REMOVED
    horizon_label = st.sidebar.radio(
        "Vade (Yatırım Ufku)",
        ["Kısa", "Orta", "Uzun"],
        index=["short", "medium", "long"].index(st.session_state.horizon),
        help="Rebalans frekansı + (η, λ, τ, γ) preset'lerini değiştirir.",
    )
    st.session_state.horizon = {"Kısa": "short", "Orta": "medium", "Uzun": "long"}[horizon_label]
    preset = HORIZON_PRESETS[st.session_state.horizon]
    st.sidebar.caption(
        f"Rebalans: {preset['rebalance']}g · η={preset['eta']} · "
        f"λ={preset['lam']} · τ={preset['tau']} · γ={preset['gamma']}"
    )

    # ------------------------------------------------------------------
    # Adım granülerliği seçici
    # ------------------------------------------------------------------
    _GRAN_LABELS = {"daily": "Gün (Daily)", "monthly": "Ay (Monthly)", "yearly": "Yıl (Yearly)"}
    _GRAN_REVERSE = {v: k for k, v in _GRAN_LABELS.items()}
    _cur_gran = st.session_state.get("granularity", "daily")
    _cur_gran_label = _GRAN_LABELS.get(_cur_gran, "Gün (Daily)")
    _sel_gran_label = st.sidebar.selectbox(
        "Adım granülerliği",
        options=[_GRAN_LABELS[g] for g in GRANULARITY_OPTIONS],
        index=list(GRANULARITY_OPTIONS).index(_cur_gran),
        key="ui_granularity",
        help=(
            "Veri adım büyüklüğü. **Gün**: ham BIST günlük fiyatlar (varsayılan, "
            "golden-güvenli). **Ay**: aylık ortalama (~120 nokta/10 yıl). "
            "**Yıl**: yıllık ortalama (~10 nokta/10 yıl — kaba sonuç). "
            "Feature'lar her zaman günlük hesaplanır, sonra resample edilir."
        ),
    )
    _new_gran = _GRAN_REVERSE[_sel_gran_label]

    # Granülerlik değişince veriyi geçersiz kıl (kullanıcıya "Veriyi Yükle" uyarısı)
    if _new_gran != st.session_state.get("granularity", "daily"):
        st.session_state.granularity = _new_gran
        if st.session_state.get("data_loaded"):
            st.session_state.data_loaded = False
            for k in ["prices", "px_tr", "px_te", "feats_tr", "feats_te", "scaler",
                      "macro_tr", "macro_te", "regime_tr", "regime_te",
                      "trained_agents", "test_traces", "baselines"]:
                st.session_state[k] = (
                    {} if k in ("trained_agents", "test_traces") else None
                )
            st.sidebar.warning(
                "Granülerlik değişti — veriyi yeniden yükleyin ('Veriyi Yükle / İndir')."
            )
    else:
        st.session_state.granularity = _new_gran

    # Tahmini nokta sayısı + minimum nokta uyarısı
    _n_pts = st.session_state.get("granularity_n_points")
    _min_pts = GRANULARITY_MIN_POINTS.get(_new_gran, 0)
    if _n_pts is not None:
        _pts_caption = f"Mevcut veri: {_n_pts} {_sel_gran_label.lower()} noktası"
        if _n_pts < _min_pts:
            st.sidebar.warning(
                f"{_pts_caption} — önerilen minimum {_min_pts}. "
                "Yıllık granülerlikte sonuçlar kaba olabilir."
            )
        else:
            st.sidebar.caption(_pts_caption)
    else:
        _approx = {"daily": "~2520", "monthly": "~120", "yearly": "~10"}.get(_new_gran, "?")
        st.sidebar.caption(
            f"Tahmini nokta sayısı (2015–2024, 10 yıl): {_approx} — "
            f"{'yeterli' if _new_gran != 'yearly' else 'az nokta → kaba sonuç'}."
        )

    # ------------------------------------------------------------------
    # Episode uzunlugu VADEDEN BAGIMSIZ: her zaman secili veri tarih
    # araliginin TAMAMI (train env = len(px_tr); eval env = tam test donemi).
    # Vade (kisa/orta/uzun) yalniz pencere/rebalans/odul preset'ini etkiler,
    # episode uzunlugunu DEGIL. (Eski horizon-bazli slider kaldirildi.)
    # ------------------------------------------------------------------
    _max_d = preset["max_days"]                 # asagidaki rebalans araligi icin
    _px_tr = st.session_state.get("px_tr")
    _ep_txt = (f"≈{len(_px_tr)} adım (train aralığının tamamı)"
               if _px_tr is not None else "tüm seçili tarih aralığı")
    st.sidebar.caption(
        f"📏 **Episode uzunluğu: {_ep_txt}** — vade (kısa/orta/uzun) bundan BAĞIMSIZ. "
        "1 episode = seçili veri tarih aralığının TAMAMI; eval de tam test dönemini koşar."
    )
    st.session_state.train_max_steps = 10_000   # geriye-uyum (kullanilmiyor; egitim len(px_tr) kullanir)

    # Parametrik rebalans frekansı — vade preset'ini override eder (default = preset).
    # _make_env → build_env → env.rebalance_freq'e bağlanır; CLI/golden preset kullanır (golden-güvenli).
    _default_reb = int(preset["rebalance"])
    _reb_max = max(2, int(_max_d))
    _cur_reb = int(st.session_state.get("train_rebalance") or _default_reb)
    _cur_reb = max(1, min(_reb_max, _cur_reb))
    st.session_state.train_rebalance = st.sidebar.number_input(
        "Rebalans frekansı (gün)",
        min_value=1, max_value=_reb_max,
        value=_cur_reb, step=1,
        help=(
            f"Kaç günde bir ağırlıklar yeniden ayarlanır (al-sat). Preset ({st.session_state.horizon}): {_default_reb} gün. "
            "1 = her gün; arada günlerde önceki ağırlık tutulur. "
            "Eğitim & test envlerine uygulanır; CLI/golden preset'i kullanır."
        ),
    )
    """
    st.session_state.train_rebalance = 1
    st.session_state.train_max_steps = 10_000

    # ------------------------------------------------------------------
    # N12: Nakit yıllık faiz oranı
    # ------------------------------------------------------------------
    _cur_annual = float(st.session_state.get("cash_annual_rate", EnvConfig.cash_annual_rate))
    _new_annual = st.sidebar.number_input(
        "Nakit yıllık faiz (risksiz) %",
        min_value=0.0, max_value=1.0,
        value=_cur_annual,
        step=0.05, format="%.2f",
        help=(
            "Portföydeki nakit kısmının yıllık bileşik getirisi (risksiz faiz). "
            "TR 2022-24 mevduat/repo ~ %40 → 0.40. "
            "Günlük oran: (1+R)^(1/252)-1 formülüyle türetilir. "
            "Eğitim & test envlerine uygulanır — 'hisse mi nakit mi daha karlı' kıyasını canlı gösterir."
        ),
        key="ui_cash_annual_rate",
    )
    st.session_state.cash_annual_rate = _new_annual
    # Günlük oranı hesapla ve session'a yaz — _make_env buradan okur.
    _daily = _cash_daily_rate(_new_annual, EnvConfig.trading_days)
    st.session_state.cash_daily_rate = _daily
    st.sidebar.caption(
        f"Günlük nakit getirisi: {_daily*100:.5f}%  "
        f"(yıllık %{_new_annual*100:.1f} → (1+R)^(1/252)-1)"
    )

    _sidebar_reward_editor(preset)

    st.session_state.adaptive = st.sidebar.checkbox(
        "Adaptif ödül aktif", value=st.session_state.adaptive,
        help="Açıkken η_t, λ_t, τ_t rolling vol & turnover EWMA'larına göre ölçeklenir."
    )

    st.sidebar.divider()
    st.sidebar.subheader("📊 Veri")

    # ------------------------------------------------------------------
    # Tarih seçici — train/test aralığı
    # ------------------------------------------------------------------
    with st.sidebar.expander("📅 Tarih Aralığı", expanded=False):
        st.caption(
            "Eğitim başlangıcı → Train/Test ayırım → Test bitişi. "
            "Ayırım sonrası veriler test dönemi olarak kullanılır."
        )
        _start_val = datetime.date.fromisoformat(
            st.session_state.get("data_start", _dc.start)
        )
        _split_val = datetime.date.fromisoformat(
            st.session_state.get("data_split", _dc.train_end)
        )
        _end_val = datetime.date.fromisoformat(
            st.session_state.get("data_end", _dc.end)
        )

        sel_start = st.date_input(
            "Train başlangıcı",
            value=_start_val,
            min_value=_DATE_MIN,
            max_value=_DATE_MAX,
            key="ui_data_start",
            help="Eğitim verisinin başlangıç tarihi (dahil).",
        )
        sel_split = st.date_input(
            "Train/Test ayırım tarihi",
            value=_split_val,
            min_value=_DATE_MIN,
            max_value=_DATE_MAX,
            key="ui_data_split",
            help="Bu tarihten itibaren test verisi başlar (dahil). "
                 "Scaler/forecaster yalnız eğitim kısmında fit edilir (sızıntı yok).",
        )
        sel_end = st.date_input(
            "Test bitişi",
            value=_end_val,
            min_value=_DATE_MIN,
            max_value=_DATE_MAX,
            key="ui_data_end",
            help="Test verisinin bitiş tarihi (dahil).",
        )

        # Sızıntı / tutarlılık doğrulaması
        _date_valid = (sel_start < sel_split <= sel_end)
        if not _date_valid:
            st.sidebar.error(
                "Tarih hatası: Train başlangıcı < Ayırım tarihi ≤ Test bitişi "
                "koşulu sağlanmalı. Veriyi yükleyemezsiniz."
            )
        else:
            st.session_state.data_start = sel_start.isoformat()
            st.session_state.data_split = sel_split.isoformat()
            st.session_state.data_end   = sel_end.isoformat()
            st.caption(
                f"Eğitim: {sel_start} → {sel_split}  |  "
                f"Test: {sel_split} → {sel_end}"
            )

    if not st.session_state.data_loaded:
        _date_valid_outer = (
            datetime.date.fromisoformat(st.session_state.get("data_start", _dc.start))
            < datetime.date.fromisoformat(st.session_state.get("data_split", _dc.train_end))
            <= datetime.date.fromisoformat(st.session_state.get("data_end", _dc.end))
        )
        if st.sidebar.button(
            "Veriyi Yükle / İndir",
            width='stretch',
            disabled=not _date_valid_outer,
        ):
            _load_data()
            st.rerun()
        if not _date_valid_outer:
            st.sidebar.caption("Tarih aralığı geçersiz — düzeltin.")
    else:
        st.sidebar.success(f"Veri yüklü: {st.session_state.prices.shape[0]} gün × "
                           f"{st.session_state.prices.shape[1]} hisse")
        if st.sidebar.button("Veriyi yeniden yükle", width='stretch'):
            for k in ["data_loaded", "prices", "px_tr", "px_te",
                      "feats_tr", "feats_te", "scaler",
                      "trained_agents", "test_traces", "baselines"]:
                st.session_state[k] = False if k == "data_loaded" else (
                    {} if k in ["trained_agents", "test_traces"] else None)
            st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader(f"🎛 Hiperparametreler ({algo})")
    hp = {}
    # γ (discount): vade preset default'u; override edilirse hp üzerinden TÜM ajanlara uygulanır.
    hp["gamma"] = st.sidebar.number_input(
        "γ (discount / iskonto)", value=float(preset["gamma"]),
        min_value=0.90, max_value=0.999, step=0.005, format="%.3f",
        key=f"gamma_step_{step_days}",
        help="İskonto faktörü. Vade preset default verir (Kısa 0.95 / Orta 0.99 / Uzun 0.995); "
             "burada değiştirilebilir — DQN/PPO/SAC/TD3'ün hepsine uygulanır.",
    )
    if algo == "DQN":
        st.sidebar.caption(_EP_HINT)
        hp["lr"]        = st.sidebar.select_slider("Öğrenme oranı",
            options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3], value=1e-3)
        hp["eps_decay"] = st.sidebar.slider("ε decay adımı", 2_000, 30_000, 10_000, step=1_000)
        hp["batch_size"]= st.sidebar.select_slider(_LBL_BATCH, options=[32, 64, 128], value=64)
        hp["target_update"] = st.sidebar.slider("Target sync", 100, 2000, 500, step=100)
    elif algo == "PPO":
        st.sidebar.caption(_EP_HINT.replace("episode", "update"))
        hp["rollout_len"]= st.sidebar.slider("Rollout uzunluğu", 128, 1024, 400, step=64)
        hp["lr_p"]       = st.sidebar.select_slider(_LBL_POLICY_LR,
            options=[1e-4, 3e-4, 1e-3], value=3e-4)
        hp["lr_v"]       = st.sidebar.select_slider("Value LR",
            options=[3e-4, 1e-3, 3e-3], value=1e-3)
        hp["clip"]       = st.sidebar.slider("Clip ε", 0.05, 0.4, 0.2, step=0.05)
        hp["ent_coef"]   = st.sidebar.select_slider("Entropi katsayısı",
            options=[0.0, 0.001, 0.005, 0.01, 0.02], value=0.005)
        hp["batch_size"] = st.sidebar.select_slider("Mini-batch", options=[64, 128, 256], value=128)
        hp["n_epochs"]   = st.sidebar.slider("Epoch", 2, 10, 6, step=1)
    elif algo == "SAC":
        st.sidebar.caption(_EP_HINT)
        hp["lr_pi"]      = st.sidebar.select_slider(_LBL_POLICY_LR,
            options=[1e-4, 3e-4, 1e-3], value=3e-4)
        hp["lr_q"]       = st.sidebar.select_slider("Q LR",
            options=[3e-4, 5e-4, 1e-3], value=5e-4)
        hp["alpha"]      = st.sidebar.slider("Entropi α", 0.0, 0.5, 0.05, step=0.01)
        hp["tau"]        = st.sidebar.select_slider("Soft update τ",
            options=[0.005, 0.01, 0.05], value=0.01)
        hp["batch_size"] = st.sidebar.select_slider(_LBL_BATCH, options=[64, 128, 256], value=128)
    else:  # TD3 — sürekli/deterministik politika (hocanın tavsiyesi)
        st.sidebar.caption(_EP_HINT)
        hp["lr_pi"]      = st.sidebar.select_slider(_LBL_POLICY_LR,
            options=[1e-4, 3e-4, 1e-3], value=3e-4)
        hp["lr_q"]       = st.sidebar.select_slider("Q LR",
            options=[1e-4, 3e-4, 5e-4, 1e-3], value=3e-4)
        hp["policy_noise"] = st.sidebar.slider("Hedef-politika gürültüsü", 0.0, 0.5, 0.2, step=0.05,
            help="Hedef aksiyona eklenen clamped Gauss gürültüsü (TD3 smoothing).")
        hp["expl_noise"] = st.sidebar.slider("Keşif gürültüsü", 0.0, 0.5, 0.1, step=0.05,
            help="Eğitimde aksiyona eklenen keşif gürültüsü (eval'de kapalı).")
        hp["tau"]        = st.sidebar.select_slider("Soft update τ",
            options=[0.005, 0.01, 0.05], value=0.005)
        hp["batch_size"] = st.sidebar.select_slider(_LBL_BATCH, options=[64, 128, 256], value=128)

    # 💾 Model kalıcılığı (PDF §11 + N11): eğitilmiş modeli diske kaydet / diskten yükle.
    # İsimli kayıt: kullanıcı ad girer → named_model_path(name).pt olarak kaydedilir.
    # Geriye-uyumluluk: eski algo_{horizon}_{adaptive}.pt dosyaları listede görünmeye devam eder.
    st.sidebar.divider()
    st.sidebar.subheader("💾 Model (kaydet / yükle)")

    cur_key = _agent_key(algo, step_days, st.session_state.adaptive)
    has_trained = (cur_key in st.session_state.trained_agents
                   and st.session_state.trained_agents[cur_key][0] is not None)

    if has_trained:
        _default_name = f"{algo}_step{step_days}"
        _model_name = st.sidebar.text_input(
            "Model adı (kaydetmek için)",
            value=_default_name,
            key="ui_model_name_input",
            help="Kaydedilecek modelin adı. Harf/rakam/_ ve - kullanılabilir; "
                 "diğer karakterler _ ile değiştirilir. Boş bırakılırsa "
                 f"'{_default_name}' kullanılır.",
        )
        if st.sidebar.button("💾 Eğitilmiş modeli kaydet", width='stretch'):
            p = save_trained_agent(
                algo, step_days, st.session_state.adaptive,
                name=_model_name,
            )
            if p:
                from core.persistence import read_meta as _rm
                from pathlib import Path as _P
                _saved_meta = _rm(_P(p))
                st.sidebar.success(
                    f"Kaydedildi: **{_saved_meta['name']}** "
                    f"({_saved_meta['saved_at']})  \n`{_P(p).name}`"
                )
            else:
                st.sidebar.error("Kaydetme başarısız — önce modeli eğitin.")

    # Kayıtlı model listesi: list_saved_models() meta okur, saved_at'e göre sıralı.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _saved_models = list_saved_models()
    if _saved_models:
        st.sidebar.markdown("**Kayıtlı modeller**")

        def _model_label(m: dict) -> str:
            """"{name} · {saved_at} · {algo}/{horizon}/{adaptive}" formatı."""
            _hor_tr = f"{m.get('step_days', 1)} seans" if not m.get("legacy") else f"legacy:{m['horizon']}"
            _adp_tr = "adaptif" if m["adaptive"] else "sabit"
            _name = m["name"] or "(isimsiz)"
            _at = m["saved_at"] or "—"
            _algo_s = m["algo"] or "?"
            return f"{_name} · {_at} · {_algo_s}/{_hor_tr}/{_adp_tr}"

        _labels = [_model_label(m) for m in _saved_models]
        _selected_label = st.sidebar.selectbox(
            "Model seç",
            options=_labels,
            key="ui_model_selectbox",
            help="Kayıtlı modeller — isim · tarih · algo/vade/adaptif. "
                 "Seçip 'Yükle' butonuna bas.",
        )
        _sel_idx = _labels.index(_selected_label) if _selected_label in _labels else 0
        _sel_model = _saved_models[_sel_idx]
        st.sidebar.caption(
            f"Dosya: `{_sel_model['path'].split('/')[-1].split(chr(92))[-1]}`"
        )

        if st.sidebar.button("📂 Seçili modeli yükle", width='stretch'):
            result = load_saved_agent_from_path(_sel_model["path"])
            if result:
                _lkey, _lmeta = result
                # Yüklenen modelin algo/horizon/adaptive değerlerini session'a al
                # → mevcut resume (G4 "Devam Et") akışı bu değerleri kullanır.
                st.session_state.selected_algo = _lmeta["algo"]
                if _lmeta.get("run_spec"):
                    st.session_state.step_days = int(_lmeta["run_spec"].get("step_days", 1))
                st.session_state.adaptive = _lmeta["adaptive"]
                _hor_tr = (f"{_lmeta['run_spec'].get('step_days', 1)} seans"
                           if _lmeta.get("run_spec") else f"legacy:{_lmeta['horizon']}")
                st.sidebar.success(
                    f"**{_sel_model['name']}** yüklendi  \n"
                    f"{_lmeta['algo']} / {_hor_tr} / "
                    f"{'adaptif' if _lmeta['adaptive'] else 'sabit'}  \n"
                    "Test sekmesinde çalıştırılabilir; 'Devam Et' ile eğitime devam edilebilir."
                )
                st.rerun()
            else:
                st.sidebar.error("Model yüklenemedi.")
    else:
        st.sidebar.caption(
            "Kayıtlı model yok. Eğit ve 'Kaydet' butonunu kullan "
            "ya da CLI ile üret: python main.py"
        )

    st.session_state.gamma_daily = float(hp["gamma"])
    set_active_run_spec(algo, step_days, st.session_state.adaptive, hp)
    st.sidebar.caption(f"Seed: {SEED} (sabit)")
    return algo, step_days, st.session_state.adaptive, hp

```

---

## `ui/tabs/mdp.py`

```python
"""Sekme 1 — Veri & MDP (app.py'den tasindi, P5)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data import BIST28, SPLIT, START, END
from config import DEFAULTS


def tab_mdp():
    st.header("📐 Veri ve MDP Formülasyonu")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Evren: BIST 28 Hisse")
        st.write(f"**Tarih aralığı:** {START} → {END} · **Split:** {SPLIT}")
        st.write(f"**KOZAA.IS ve KOZAL.IS hariç tutuldu.**")
        chips = "  ·  ".join(BIST28)
        st.caption(chips)

        if st.session_state.data_loaded:
            px_df = st.session_state.prices
            norm = (px_df / px_df.iloc[0]).resample("W").last()
            fig_px = px.line(norm, title="Normalize kapanış (haftalık)")
            fig_px.update_layout(height=280, showlegend=False,
                                 margin=dict(t=40, b=20), xaxis_title="",
                                 yaxis_title="NAV (ilk gün = 1.0)")
            st.plotly_chart(fig_px, width='stretch', key="mdp_px")

    with col2:
        st.subheader("MDP Tuple (𝒮, 𝒜, 𝒫, r, γ)")
        mdp = pd.DataFrame([
            ("𝒮 Durum Uzayı", "ℝ³⁹³ — 28 hisse × 13 özellik (12 teknik + 1 forecast, z-skorlu) + 29 boyutlu ağırlık"),
            ("𝒜 Eylem Uzayı (DQN)", "6 şablon: Nakit, Eşit Ağırlık, Top-3/Top-5 Mom, Ters-Vol, Min-Vol"),
            ("𝒜 Eylem Uzayı (PPO/SAC)", "ℝ²⁹ → softmax → portföy simpleksi"),
            ("𝒫 Geçiş", "Piyasa tarafından belirlenen stokastik süreç"),
            ("r Ödül", "log(1+w·r) − η_t·‖Δw‖₁ − λ_t·max(0, DD−τ_t)"),
            ("γ İndirgeme", "gamma_daily ^ dönem_seans_sayısı"),
            ("Sonlandırma", "veri sonu VEYA NAV<0.01 (iflas)"),
        ], columns=["Bileşen", "Tanım"])
        st.dataframe(mdp, hide_index=True, width='stretch')

    st.divider()
    st.subheader("N-Seans Adım Sözleşmesi & Adaptif Ödül")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.dataframe(pd.DataFrame([{
            "Seçili N": int(st.session_state.get("step_days", DEFAULTS.step_days)),
            "Rebalans": "Her adım",
            "Momentum (işlem günü)": DEFAULTS.mom_window,
            "Min-vol (işlem günü)": DEFAULTS.minvol_window,
            "γ günlük": DEFAULTS.gamma,
        }]), hide_index=True, width='stretch')
    with c2:
        st.markdown("""
**Adaptif şekillendirici** (`AdaptiveRewardShaper`):
EWMA ile rolling realized vol (`vol_ewma`) ve rolling turnover (`turnover_ewma`) izlenir.

- **η_t** = η_base × max(1, turnover_ewma / turnover_target)
  — çok işlem yapan ajan kendi kendini cezalandırır
- **λ_t** = λ_base × (1 + max(0, vol_ratio − 1))
  — yüksek-vol rejiminde DD cezası büyür
- **τ_t** = τ_base × max(0.7, 1 / vol_ratio)
  — volatil rejimde eşik daralır → daha hassas DD cezası
""")
        st.info(f"Adaptif mod: **{'Açık' if st.session_state.adaptive else 'Kapalı'}** · "
                f"Seçili adım: **{st.session_state.get('step_days', 1)} BIST seansı**")

```

---

## `ui/tabs/train.py`

```python
"""Sekme 2 — Eğitim (canlı) (app.py'den tasindi, P5)."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import BIST28
from env.portfolio_env import ACTION_NAMES
from ui.services import train_generator
from ui.state import _agent_key
from utils.metrics import training_diagnostics
from utils.portfolio_tl import (
    build_portfolio_table, compute_tl_series, step_rows_for_training,
)

# Grafik/tablo serilestirme her N iterde bir (performans — kesif bulgusu:
# onceki surum her iterde 6 plotly + 2 dataframe serialize ediyordu, bu canli
# egitimde %10-40 ek yuk demekti). Metrikler/durum/stop butonu her iter guncel
# kalir; egitim bitiminde son durum HER ZAMAN render edilir.
RENDER_EVERY = 5


def tab_train(algo: str, step_days: int, adaptive: bool, hp: dict):
    st.header(f"🎓 Eğitim — {algo} · {step_days} seans/adım · "
              f"Adaptif: {'Açık' if adaptive else 'Kapalı'}")

    if not st.session_state.data_loaded:
        st.warning("Önce sidebar'dan 'Veriyi Yükle' butonuna basın.")
        return

    key = _agent_key(algo, step_days, adaptive)
    already = key in st.session_state.trained_agents
    if already:
        st.success(f"Bu konfigürasyon daha önce eğitildi. "
                   f"({len(st.session_state.trained_agents[key][1])} iterasyon)")

    # G4: 'Devam Et' yalnız in-memory eğitilmiş + eğrisi olan ajan için anlamlı.
    can_resume = already and bool(st.session_state.trained_agents[key][1])
    col_run, col_resume, col_clear = st.columns([1, 1, 1])
    with col_run:
        run = st.button(f"{'Yeniden Eğit' if already else 'Eğit'}", type="primary")
    resume = False
    with col_resume:
        if can_resume:
            resume = st.button("▶ Devam Et",
                               help="Durdurulan eğitime AYNI ajanla (ağırlık+optimizer+buffer) kaldığı yerden devam")
    with col_clear:
        if already and st.button("Bu konfigürasyonu unut"):
            st.session_state.trained_agents.pop(key, None)
            st.session_state.test_traces.pop(key, None)
            st.rerun()

    st.caption("💡 'Eğit' sıfırdan başlatır · '▶ Devam Et' durdurulan eğitimi aynı ajanla sürdürür · "
               "eğitim sırasında **⏹ Eğitimi Durdur** ile istediğin noktada kesebilirsin.")

    if not (run or resume):
        if already:
            _render_training_curves(st.session_state.trained_agents[key][1], algo)
            _render_episode_browser(key, algo, float(st.session_state.initial_capital))
        return

    # --- CANLI EĞİTİM (sınırsız) ---
    status = st.empty()
    status.info("Başlıyor — 'Eğitimi Durdur' ile istediğin noktada kes.")

    # Üst satır: canlı throughput metrikleri
    tp_cols = st.columns(4)
    ph_iter    = tp_cols[0].empty()
    ph_rate    = tp_cols[1].empty()
    ph_avg     = tp_cols[2].empty()
    ph_elapsed = tp_cols[3].empty()

    row1 = st.columns(2)
    row2 = st.columns(2)
    ph_reward  = row1[0].empty()
    ph_gain    = row1[1].empty()
    ph_success = row2[0].empty()
    ph_loss    = row2[1].empty()
    stop_slot  = st.empty()

    # Canlı TL paneli placeholder'ları
    st.markdown("### 💰 Son episod TL izlemesi")
    tl_metric_cols = st.columns(6)
    ph_tl_start = tl_metric_cols[0].empty()
    ph_tl_end   = tl_metric_cols[1].empty()
    ph_tl_net   = tl_metric_cols[2].empty()
    ph_tl_min   = tl_metric_cols[3].empty()
    ph_tl_max   = tl_metric_cols[4].empty()
    ph_tl_dd    = tl_metric_cols[5].empty()
    tl_chart_cols = st.columns(2)
    ph_tl_line  = tl_chart_cols[0].empty()
    ph_tl_bar   = tl_chart_cols[1].empty()
    ph_tl_table = st.empty()
    ph_tl_port  = st.empty()
    ph_bankrupt = st.empty()

    # G4: 'Devam Et' ise mevcut ajanı + curve'ü taşı; iter ofseti = önceki iter sayısı.
    resume_agent = st.session_state.trained_agents[key][0] if resume else None
    curve = list(st.session_state.trained_agents[key][1]) if resume else []
    iter_offset = len(curve)
    n_episodes = int(st.session_state.get("n_episodes", 12))
    gen = train_generator(algo, step_days, adaptive, hp,
                          rollout_len=int(hp.get("rollout_len", 400)),
                          resume_agent=resume_agent,
                          n_episodes=n_episodes)
    t0 = time.time()
    iter_times = []  # son N iter süresi (iter/sn için)
    trained_agent = None
    stopped_early = False
    initial_capital = float(st.session_state.initial_capital)

    # İlerleme çubuğu: N episode'a göre doldurulur
    progress_bar = st.progress(0.0, text=f"Episode 0 / {n_episodes}")

    last_rec = None
    # Per-episode telemetri (episode seçici için): her episode'un TL izini sakla (UI-only).
    ep_snaps: list = []
    ep_shared: dict = {"prices": None, "dates": None}
    for rec in gen:
        iter_start_elapsed = time.time() - t0
        trained_agent = rec["agent"]
        env = rec["env"]
        last_rec = rec
        d = {k: v for k, v in rec.items() if k not in ("agent", "env", "actions")}
        d["iter"] = iter_offset + rec["iter"]      # G4: devam'da iterasyon numarası süreklilik
        curve.append(d)
        # Her iter sonunda session'a yaz → kullanıcı durdurursa veya refresh etse bile son hali kalır
        st.session_state.trained_agents[key] = (trained_agent, list(curve))

        # Per-episode snapshot — selectbox ile sonradan incelemek için (UI-only, golden-etkisiz).
        if ep_shared["prices"] is None:
            ep_shared["prices"] = env.prices
            ep_shared["dates"] = env.dates
        ep_snaps.append({
            "iter": int(d["iter"]),
            "reward": float(rec["reward"]), "nav": float(rec["nav"]),
            "gain": float(rec.get("gain", rec["nav"] - 1.0)),
            "loss": float(rec.get("loss", 0.0)),
            "success": int(rec.get("success", 0)),
            "nav_history": list(env.nav_history),
            "weight_history": [np.asarray(w, dtype=np.float32) for w in env.weight_history],
            "reward_terms_history": [dict(rt) for rt in env.reward_terms_history],
            "t": int(env.t), "step_count": int(env.step_count),
            "actions": list(rec.get("actions") or []),
        })
        st.session_state.setdefault("episode_snaps", {})[key] = ep_snaps
        st.session_state.setdefault("episode_shared", {})[key] = ep_shared

        # Agir serilestirme (4 egri + TL paneli) yalniz her RENDER_EVERY iterde
        render_now = (len(curve) == 1) or (len(curve) % RENDER_EVERY == 0)
        if render_now:
            _render_live_curves(pd.DataFrame(curve),
                                ph_reward, ph_gain, ph_success, ph_loss, seq=d["iter"])

        # --- Throughput metrikleri ---
        iter_end_elapsed = time.time() - t0
        iter_times.append(iter_end_elapsed - iter_start_elapsed)
        recent = iter_times[-5:]
        rate = (len(recent) / sum(recent)) if sum(recent) > 0 else 0.0
        avg = sum(iter_times) / len(iter_times)
        cur_ep = iter_offset + rec["iter"] + 1
        ph_iter.metric("Episode", f"{cur_ep} / {n_episodes}")
        ph_rate.metric("Iter/sn", f"{rate:.2f}")
        ph_avg.metric("Ort. iter süresi", f"{avg:.2f}s")
        mm = int(iter_end_elapsed // 60); ss = int(iter_end_elapsed % 60)
        ph_elapsed.metric("Toplam elapsed", f"{mm:02d}:{ss:02d}")

        # İlerleme çubuğu: Episode i/N
        _prog = min(cur_ep / n_episodes, 1.0)
        progress_bar.progress(_prog, text=f"Episode {cur_ep} / {n_episodes}")

        # --- Canlı TL paneli (son episod için env.nav_history / weight_history kullan) ---
        if render_now:
            _render_train_tl_panel(
                env=env, algo=algo, rec=rec, initial_capital=initial_capital,
                ph_tl_start=ph_tl_start, ph_tl_end=ph_tl_end, ph_tl_net=ph_tl_net,
                ph_tl_min=ph_tl_min, ph_tl_max=ph_tl_max, ph_tl_dd=ph_tl_dd,
                ph_tl_line=ph_tl_line, ph_tl_bar=ph_tl_bar,
                ph_tl_table=ph_tl_table, ph_tl_port=ph_tl_port,
                ph_bankrupt=ph_bankrupt, seq=d["iter"],
            )

        status.info(f"Episode {cur_ep}/{n_episodes} · NAV={rec['nav']:.3f} · "
                    f"elapsed {iter_end_elapsed:.1f}s — "
                    f"'Eğitimi Durdur' ile erken kesilebilir")

        # Kullanıcı ayarladığı gecikmeyi iter arası uygula (slider canlı okunur).
        delay = float(st.session_state.get("train_delay", 0.0))
        if delay > 0:
            time.sleep(delay)

        # Döngü içi durdurma — her iter sonunda placeholder'a buton render eder.
        # Kullanıcı basarsa bir sonraki iter başlamaz, o ana kadar eğitilen ajan session'da kalmış olur.
        if stop_slot.button("⏹ Eğitimi Durdur", key=f"stop_loop_{iter_offset}_{rec['iter']}"):
            stopped_early = True
            break

    st.session_state.trained_agents[key] = (trained_agent, curve)
    elapsed = time.time() - t0
    stop_slot.empty()
    progress_bar.progress(1.0, text=f"Tamamlandı — {len(curve)} episode")
    # Son durumu HER ZAMAN render et (throttle yuzunden son iterler atlanmis olabilir).
    # seq="final": dongu-ici render'larin (seq=iter no) HICBIRIYLE cakismaz -> ayni
    # st.empty() slot'una son kez yazar, benzersiz key (StreamlitDuplicateElementKey yok).
    if curve:
        _render_live_curves(pd.DataFrame(curve), ph_reward, ph_gain, ph_success, ph_loss,
                            seq="final")
    if last_rec is not None:
        _render_train_tl_panel(
            env=last_rec["env"], algo=algo, rec=last_rec, initial_capital=initial_capital,
            ph_tl_start=ph_tl_start, ph_tl_end=ph_tl_end, ph_tl_net=ph_tl_net,
            ph_tl_min=ph_tl_min, ph_tl_max=ph_tl_max, ph_tl_dd=ph_tl_dd,
            ph_tl_line=ph_tl_line, ph_tl_bar=ph_tl_bar,
            ph_tl_table=ph_tl_table, ph_tl_port=ph_tl_port,
            ph_bankrupt=ph_bankrupt, seq="final",
        )
    if stopped_early:
        status.warning(f"{algo} eğitimi {len(curve)}. iter sonunda durduruldu "
                       f"({elapsed:.1f}s) — son ajan session'a kaydedildi.")
    else:
        status.success(f"{algo} eğitildi ({len(curve)} iter, {elapsed:.1f}s) "
                       f"ve session'a kaydedildi. Tab 3'te test edebilirsiniz.")

    # --- PDF §9.7 Pedagojik Eğitim Metrikleri ---
    if curve:
        rt_hist = list(last_rec["env"].reward_terms_history) if last_rec is not None else None
        diag = training_diagnostics(curve, reward_terms_history=rt_hist)
        st.markdown("#### §9.7 Eğitim Tanılama Metrikleri")
        d_cols = st.columns(5)
        d_cols[0].metric(
            "Hareketli Ort. Return",
            f"{diag.get('ma_return_last', 0.0):.4f}",
            help="Son 5 episod ödülünün hareketli ortalaması (öğrenme eğilimi)",
        )
        d_cols[1].metric(
            "Başarı Oranı",
            f"{diag.get('success_rate', 0.0):.1%}",
            help="EW benchmark'ı geçen episod oranı",
        )
        if "avg_steps" in diag:
            d_cols[2].metric(
                "Ort. Adım/Episod",
                f"{diag['avg_steps']:.1f}",
                help="Trainer 'steps' alanından hesaplandı",
            )
        else:
            d_cols[2].metric(
                "Ort. Adım/Episod",
                "—",
                help="Trainer kayıtlarında 'steps' alanı yok — trainer'a dokunulmadan atlandı",
            )
        d_cols[3].metric(
            "Drawdown Ceza Adımı",
            str(diag.get("drawdown_penalty_steps", "—")),
            help="reward_terms_history'den: drawdown_penalty > 0 olan adım sayısı",
        )
        d_cols[4].metric(
            "Ort. İşlem Maliyeti",
            f"{diag.get('mean_tx_cost', 0.0):.5f}",
            help="reward_terms_history'den: adım başı ortalama tx_cost",
        )

    # Episode seçici — eğitilen her episode'un detayını (TL izi/grafik/değerler) incele.
    _render_episode_browser(key, algo, initial_capital)


def _render_live_curves(df, ph_reward, ph_gain, ph_success, ph_loss, seq=0):
    """4 canli egitim egrisini placeholder'lara cizer (throttle edilmis cagri).

    seq: render sirasi (iter no). Streamlit 1.5x st.empty() slot'una DONGU icinde
    tekrar cizimde ELEMAN ID'sini her seferinde yeniden kaydeder; sabit key ->
    StreamlitDuplicateElementKey. Render-basina BENZERSIZ key (seq) -> her cizim
    benzersiz; empty() slot yine yalniz son grafigi gosterir (yerinde gunceller).
    """
    fig_r = px.line(df, x="iter", y="reward",
                    title="Kümülatif Ödül (iterasyon başına — çevre ödülü Σr)",
                    markers=True)
    fig_r.update_layout(height=260, margin=dict(t=40, b=20))
    ph_reward.plotly_chart(fig_r, width='stretch', key=f"train_live_reward_{seq}")

    fig_g = px.line(df, x="iter", y="gain",
                    title="Kazanç (nihai NAV − 1.0)",
                    markers=True)
    fig_g.update_layout(height=260, margin=dict(t=40, b=20))
    ph_gain.plotly_chart(fig_g, width='stretch', key=f"train_live_gain_{seq}")

    fig_s = px.bar(df, x="iter", y="success",
                   title="Başarı (EW benchmark'a göre 0/1)")
    fig_s.update_layout(height=260, margin=dict(t=40, b=20),
                        yaxis=dict(range=[0, 1.2], tickvals=[0, 1]))
    ph_success.plotly_chart(fig_s, width='stretch', key=f"train_live_success_{seq}")

    if "loss" in df.columns:
        fig_l = px.line(df, x="iter", y="loss",
                        title="Ortalama loss (düşüş beklenir)",
                        markers=True)
        fig_l.update_layout(height=260, margin=dict(t=40, b=20))
        ph_loss.plotly_chart(fig_l, width='stretch', key=f"train_live_loss_{seq}")


def _render_train_tl_panel(env, algo, rec, initial_capital,
                            ph_tl_start, ph_tl_end, ph_tl_net, ph_tl_min, ph_tl_max, ph_tl_dd,
                            ph_tl_line, ph_tl_bar, ph_tl_table, ph_tl_port,
                            ph_bankrupt=None, seq=0):
    """Son episod/iter için TL türevlerini hesapla ve placeholder'ları güncelle."""
    nav_hist = list(env.nav_history)
    weight_hist = list(env.weight_history)
    rt_hist = list(env.reward_terms_history)
    if len(nav_hist) < 2:
        return
    # env.t reset sonrası max(window, 21)'de başladı; her adımda +1.
    # Şu an env.t = t_start + step_count. weight_hist[k] env'in k'ıncı adım sonrası w'si
    # → prices[t_start + k] ile hizalanır.
    t_end = int(env.t)
    steps_done = int(env.step_count)
    t_start = t_end - steps_done
    tx_rates = [float(rt.get("tx_cost", 0.0)) for rt in rt_hist]
    snaps = compute_tl_series(
        nav_hist=nav_hist, weight_hist=weight_hist,
        prices_matrix=env.prices, initial_capital=initial_capital,
        tx_cost_rates=tx_rates, t_start=t_start,
    )
    if not snaps:
        return

    port_tl_arr = np.array([s["portfolio_tl"] for s in snaps])
    step_pnl_arr = np.array([s["step_pnl_tl"] for s in snaps])
    cum_pnl_arr = np.array([s["cum_pnl_tl"] for s in snaps])
    peak = np.maximum.accumulate(port_tl_arr)
    dd_pct = (peak - port_tl_arr) / np.maximum(peak, 1e-9)

    end_tl = float(port_tl_arr[-1])
    net = end_tl - initial_capital
    net_pct = 100.0 * net / initial_capital
    ph_tl_start.metric("Başlangıç TL", f"{initial_capital:,.0f} ₺")
    ph_tl_end.metric("Bitiş TL", f"{end_tl:,.0f} ₺")
    ph_tl_net.metric("Net Kâr", f"{net:+,.0f} ₺", delta=f"{net_pct:+.2f}%")
    ph_tl_min.metric("Min TL", f"{float(port_tl_arr.min()):,.0f} ₺")
    ph_tl_max.metric("Max TL", f"{float(port_tl_arr.max()):,.0f} ₺")
    ph_tl_dd.metric("Max DD %", f"{100*float(dd_pct.max()):.2f}%")

    # Portfolio TL zaman serisi + başlangıç referans
    idx = np.arange(1, len(snaps) + 1)
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(x=idx, y=port_tl_arr, mode="lines",
                                name="Portföy TL", line=dict(color="#1f77b4")))
    fig_tl.add_hline(y=initial_capital, line_dash="dot",
                     annotation_text="başlangıç", line_color="#888")
    fig_tl.update_layout(title="Portföy Değeri (TL)", height=280,
                         margin=dict(t=40, b=30), xaxis_title="Gün",
                         yaxis_title="TL")
    ph_tl_line.plotly_chart(fig_tl, width='stretch', key=f"train_tl_line_{seq}")

    # Adım P&L bar chart (yeşil/kırmızı)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in step_pnl_arr]
    fig_bar = go.Figure(go.Bar(x=idx, y=step_pnl_arr, marker_color=colors))
    fig_bar.update_layout(title="Adım P&L (TL)", height=280,
                          margin=dict(t=40, b=30), xaxis_title="Gün",
                          yaxis_title="TL")
    ph_tl_bar.plotly_chart(fig_bar, width='stretch', key=f"train_tl_bar_{seq}")

    # Tam adım tablosu
    dates_slice = env.dates[t_start + 1 : t_start + 1 + steps_done]
    actions = rec.get("actions") or []
    action_names = None
    action_indices = None
    if algo == "DQN" and actions:
        action_names = [ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else str(a)
                        for a in actions]
    elif algo == "PPO":
        action_names = ["PPO-Gaussian"] * len(snaps)
    elif algo == "SAC":
        action_names = ["SAC-tanh"] * len(snaps)
    df_rows = step_rows_for_training(
        curve_snaps=snaps, dates=np.asarray(dates_slice),
        action_names=action_names, action_indices=action_indices,
        reward_terms_list=rt_hist, initial_capital=initial_capital,
    )
    ph_tl_table.dataframe(df_rows, hide_index=True, width='stretch', height=500,
                          key=f"train_tl_table_{seq}")

    # Episod sonu portföy panosu (w_prev = sondan bir önceki adım)
    last = snaps[-1]
    w_prev = weight_hist[-2] if len(weight_hist) >= 2 else None
    df_port = build_portfolio_table(BIST28, last, include_cash=True, w_prev=w_prev)
    ph_tl_port.dataframe(df_port, hide_index=True, width='stretch',
                         key=f"train_tl_port_{seq}")

    # İflas olduysa eğitim panelinin altında uyarı göster
    last_rt = rt_hist[-1] if rt_hist else {}
    if ph_bankrupt is not None:
        if last_rt.get("bankrupt"):
            ph_bankrupt.error(
                f"💀 Bu episod **iflas** ile sonlandı — NAV={nav_hist[-1]:.4f} "
                f"iflas eşiğinin altına düştü. Ajan'a ek ödül cezası "
                f"−{last_rt.get('bankruptcy_penalty', 0):.1f} uygulandı.")
        else:
            ph_bankrupt.empty()


def _render_training_curves(curve: list, algo: str):
    if not curve:   # diskten yüklenen model — eğitim eğrisi yok
        st.info("Bu model diskten yüklendi (eğitim eğrisi yok). "
                "Test sekmesinde doğrudan çalıştırabilir veya '▶ Devam Et' ile eğitebilirsiniz.")
        return
    df = pd.DataFrame(curve)
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    c1.plotly_chart(px.line(df, x="iter", y="reward", markers=True,
                            title="Kümülatif Ödül"), width='stretch',
                    key="train_curve_reward")
    c2.plotly_chart(px.line(df, x="iter", y="gain", markers=True,
                            title="Kazanç (NAV − 1)"), width='stretch',
                    key="train_curve_gain")
    c3.plotly_chart(px.bar(df, x="iter", y="success", title="Başarı (0/1)"),
                    width='stretch', key="train_curve_success")
    if "loss" in df.columns:
        c4.plotly_chart(px.line(df, x="iter", y="loss", markers=True,
                                title="Loss"), width='stretch',
                        key="train_curve_loss")

    # --- PDF §9.7 Pedagojik Eğitim Metrikleri (statik görüntüleme) ---
    diag = training_diagnostics(curve)
    st.markdown("#### §9.7 Eğitim Tanılama Metrikleri")
    d_cols = st.columns(5)
    d_cols[0].metric(
        "Hareketli Ort. Return",
        f"{diag.get('ma_return_last', 0.0):.4f}",
        help="Son 5 episod ödülünün hareketli ortalaması (öğrenme eğilimi)",
    )
    d_cols[1].metric(
        "Başarı Oranı",
        f"{diag.get('success_rate', 0.0):.1%}",
        help="EW benchmark'ı geçen episod oranı",
    )
    if "avg_steps" in diag:
        d_cols[2].metric(
            "Ort. Adım/Episod",
            f"{diag['avg_steps']:.1f}",
            help="Trainer 'steps' alanından hesaplandı",
        )
    else:
        d_cols[2].metric(
            "Ort. Adım/Episod",
            "—",
            help="Trainer kayıtlarında 'steps' alanı yok — trainer'a dokunulmadan atlandı",
        )
    d_cols[3].metric(
        "Drawdown Ceza Adımı",
        str(diag.get("drawdown_penalty_steps", "—")),
        help="reward_terms_history'den: drawdown_penalty > 0 olan adım sayısı (yeniden eğitimde mevcut)",
    )
    d_cols[4].metric(
        "Ort. İşlem Maliyeti",
        f"{diag.get('mean_tx_cost', 0.0):.5f}",
        help="reward_terms_history'den: adım başı ortalama tx_cost (yeniden eğitimde mevcut)",
    )


def _render_episode_browser(key, algo, initial_capital):
    """Saklanan her episode'un TL grafiklerini + değerlerini selectbox ile gösterir.

    Eğitim sırasında `episode_snaps[key]`'e yazılan her episode'un telemetrisi
    (nav/weight/reward_terms history + t/step_count + actions) buradan replay edilir.
    `_render_train_tl_panel` env attribute'larını okuduğundan, snapshot bir
    SimpleNamespace (sahte env) olarak ona geçirilir. UI-only — golden etkisiz.
    """
    if key is None:
        return
    snaps = st.session_state.get("episode_snaps", {}).get(key)
    shared = st.session_state.get("episode_shared", {}).get(key)
    if not snaps or not shared or shared.get("prices") is None:
        return
    st.markdown("---")
    st.markdown("### 🔎 Episode incele (her iterasyonun detayı)")
    n = len(snaps)

    def _label(i):
        s = snaps[i]
        return f"Episode {s['iter'] + 1} / {n}  ·  ödül {s['reward']:.2f} · NAV {s['nav']:.3f}"

    sel = st.selectbox("Hangi episode?", list(range(n)), index=n - 1,
                       format_func=_label, key=f"ep_browse_sel_{key}")
    s = snaps[sel]
    c = st.columns(4)
    c[0].metric("Episode", f"{s['iter'] + 1} / {n}")
    c[1].metric("Kümülatif Ödül (Σr)", f"{s['reward']:.3f}")
    c[2].metric("Kazanç (NAV−1)", f"{s['gain']:+.3f}")
    c[3].metric("Başarı (EW)", "✅" if s["success"] else "—")

    # Canlı TL panelle aynı placeholder yapısı (6 metrik + 2 grafik + tablo + porto)
    mcols = st.columns(6)
    pm = [mcols[i].empty() for i in range(6)]
    ccols = st.columns(2)
    ph_line, ph_bar = ccols[0].empty(), ccols[1].empty()
    ph_table, ph_port, ph_bank = st.empty(), st.empty(), st.empty()

    fake_env = SimpleNamespace(
        nav_history=s["nav_history"], weight_history=s["weight_history"],
        reward_terms_history=s["reward_terms_history"],
        prices=shared["prices"], dates=shared["dates"],
        t=s["t"], step_count=s["step_count"],
    )
    fake_rec = {"actions": s["actions"]}
    _render_train_tl_panel(
        env=fake_env, algo=algo, rec=fake_rec, initial_capital=initial_capital,
        ph_tl_start=pm[0], ph_tl_end=pm[1], ph_tl_net=pm[2],
        ph_tl_min=pm[3], ph_tl_max=pm[4], ph_tl_dd=pm[5],
        ph_tl_line=ph_line, ph_tl_bar=ph_bar,
        ph_tl_table=ph_table, ph_tl_port=ph_port, ph_bankrupt=ph_bank,
        seq=f"browse_{key}_{sel}",
    )

```

---

## `ui/tabs/test.py`

```python
"""Sekme 3 — Test (adım adım oynatma) (app.py'den tasindi, P5)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import BIST28
from ui.charts import _q_bar, _reward_bar, _state_top_features, _weights_pie
from ui.services import (
    _compute_test_tl_snaps, evaluate_noise_episodes_ui, evaluate_with_trace,
)
from core.episodes import summarize_episodes
from ui.state import _agent_key, env_rebalance_hint
from utils.portfolio_tl import (
    build_cumulative_trade_log, build_portfolio_table, build_trade_log,
    step_rows_for_training,
)


def tab_test(algo: str, step_days: int, adaptive: bool):
    st.header(f"🎬 Test — {algo} · {step_days} seans/adım · "
              f"Adaptif: {'Açık' if adaptive else 'Kapalı'}")

    key = _agent_key(algo, step_days, adaptive)
    if key not in st.session_state.trained_agents:
        st.warning("Bu konfigürasyon henüz eğitilmedi. Tab 2'ye git ve 'Eğit' butonuna bas.")
        return

    if st.button("Test dönemini çalıştır (rollout + trajectory)", type="primary"):
        agent = st.session_state.trained_agents[key][0]
        with st.spinner("Ajan test döneminde adım adım çalıştırılıyor..."):
            trace = evaluate_with_trace(agent, algo, step_days, adaptive)
        st.session_state.test_traces[key] = trace
        st.session_state.step_idx = 0
        st.success(f"{len(trace)} adım yakalandı.")

    trace = st.session_state.test_traces.get(key)
    if not trace:
        st.info("Henüz trajectory yok. Yukarıdaki butona basın.")
        return

    # =================================================================
    # Gürültü-artırımlı çoklu episode (robustluk / anti-ezber)
    # =================================================================
    st.session_state.setdefault("noise_episodes", {})
    with st.expander("🎲 Gürültü-artırımlı çoklu episode (robustluk testi)", expanded=False):
        st.caption(
            "Eğitilmiş ajan **tüm test aralığı boyunca** sırayla birden çok episode'da "
            "koşturulur. Episode 0 = orijinal seri; sonraki episode'lar orijinal hisse "
            "getirilerine eklenen Gauss gürültüsüyle üretilen **yeni patikalardır** "
            "(anti-ezber). Düşük dağılım = ajan dayanıklı; yüksek dağılım = tek tarihsel "
            "yola aşırı uyum riski."
        )
        cc = st.columns(3)
        n_ep = cc[0].slider("Episode sayısı", 2, 30, 6, key="noise_n_ep")
        nstd = cc[1].slider("Gürültü σ (günlük log-getiri)", 0.0, 0.05, 0.01,
                            step=0.005, key="noise_std_ep")
        run_noise = cc[2].button("▶ Episode'ları sırayla çalıştır", key="run_noise_eps")
        if run_noise:
            ag = st.session_state.trained_agents[key][0]
            with st.spinner(f"{n_ep} episode test aralığı boyunca çalıştırılıyor..."):
                st.session_state.noise_episodes[key] = evaluate_noise_episodes_ui(
                    ag, algo, step_days, adaptive,
                    n_episodes=int(n_ep), noise_std=float(nstd))
        eps = st.session_state.noise_episodes.get(key)
        if eps:
            cap = float(st.session_state.initial_capital)
            fig_eps = go.Figure()
            for r in eps:
                nav = np.asarray(r["nav"], dtype=float)
                is_orig = (r["episode"] == 0)
                fig_eps.add_trace(go.Scatter(
                    x=list(range(len(nav))), y=nav * cap, mode="lines",
                    name=("Orijinal" if is_orig else f"Ep{r['episode']} (σ={r['noise_std']:.3f})"),
                    line=dict(width=3 if is_orig else 1,
                              color="#1f77b4" if is_orig else None),
                    opacity=1.0 if is_orig else 0.55))
            fig_eps.add_hline(y=cap, line_dash="dot", line_color="#888")
            fig_eps.update_layout(title="Portföy Değeri (TL) — episode başına",
                                  height=380, xaxis_title="Adım (gün)",
                                  yaxis_title="TL", margin=dict(t=40, b=30))
            st.plotly_chart(fig_eps, use_container_width=True, key="noise_eps_chart")

            df_eps = pd.DataFrame([{
                "Episode": ("Orijinal" if r["episode"] == 0 else r["episode"]),
                "Gürültü σ": round(r["noise_std"], 3),
                "Adım": r["steps"],
                "Final NAV": round(r["final_nav"], 4),
                "Getiri %": round(r["total_return"] * 100, 2),
                "Final TL": round(r["final_nav"] * cap, 0),
                "Max DD %": round(r["max_drawdown"] * 100, 1),
                "Sharpe": round(r["sharpe"], 2),
            } for r in eps])
            st.dataframe(df_eps, hide_index=True, use_container_width=True)

            summ = summarize_episodes(eps)
            mc = st.columns(4)
            mc[0].metric("Final NAV ort.", f"{summ['final_nav_mean']:.4f}")
            mc[1].metric("Final NAV std", f"{summ['final_nav_std']:.4f}")
            mc[2].metric("Ort. getiri", f"{summ['total_return_mean'] * 100:+.2f}%")
            mc[3].metric("Zarar olasılığı", f"{summ['prob_loss'] * 100:.0f}%")

            # ---- Per-episode adım-adım detay tablosu (ekrandaki tam tablo) ----
            st.markdown("**📋 Episode detay tablosu (adım-adım)** — episode'lar SIRAYLA "
                        "koşar; biri tam bitmeden (N gün → N adım) diğeri başlamaz.")
            _ep_lbl = [("Orijinal (gürültüsüz)" if r["episode"] == 0
                        else f"Episode {r['episode']} · σ={r['noise_std']:.3f}") for r in eps]
            sel = st.selectbox("Episode seç (detay tablosu)", range(len(eps)),
                               format_func=lambda i: _ep_lbl[i], key="noise_ep_detail_sel")
            ep_tr = eps[sel].get("trace") or []
            if ep_tr:
                snaps = _compute_test_tl_snaps(ep_tr, cap)
                df_detail = step_rows_for_training(
                    snaps, [t["date"] for t in ep_tr],   # düz list (str); np.array -> numpy.str_ pd.Timestamp hatasi
                    action_names=[t["action_name"] for t in ep_tr],
                    action_indices=[t["action_idx"] for t in ep_tr],
                    reward_terms_list=[t["reward_terms"] for t in ep_tr],
                    initial_capital=cap)
                st.caption(f"{_ep_lbl[sel]} — {len(df_detail)} adım (tüm veri tarih aralığı)")
                st.dataframe(df_detail, hide_index=True, use_container_width=True, height=420)

    max_step = len(trace) - 1
    # ---- Oynatma kontrolleri ----
    ctrl = st.columns([1, 1, 1, 1, 3])
    if ctrl[0].button("⏮", help="İlk"):
        st.session_state.step_idx = 0
    if ctrl[1].button("◀", help="Önceki"):
        st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
    play_label = "⏸" if st.session_state.playing else "▶"
    if ctrl[2].button(play_label, help="Oynat/Durdur"):
        st.session_state.playing = not st.session_state.playing
    if ctrl[3].button("▶▶", help="Sonraki"):
        st.session_state.step_idx = min(max_step, st.session_state.step_idx + 1)

    st.session_state.step_idx = ctrl[4].slider(
        "Adım", 0, max_step, st.session_state.step_idx,
        label_visibility="collapsed", key="step_slider"
    )
    idx = st.session_state.step_idx
    snap = trace[idx]

    # TL türevleri — başlangıç kapital değişince anında yeniden hesaplanır
    initial_capital = float(st.session_state.initial_capital)
    tl_snaps = _compute_test_tl_snaps(trace, initial_capital)
    tl_now = tl_snaps[idx]

    # Üst bilgi satırı — TL metrikleri dahil (6 kolon)
    info_cols = st.columns(6)
    info_cols[0].metric("Tarih", snap["date"])
    info_cols[1].metric("NAV", f"{snap['nav']:.4f}")
    info_cols[2].metric("Portföy TL", f"{tl_now['portfolio_tl']:,.0f} ₺")
    info_cols[3].metric("Nakit TL", f"{tl_now['cash_tl']:,.0f} ₺")
    cum_pct = 100 * tl_now["cum_pnl_tl"] / initial_capital if initial_capital else 0
    info_cols[4].metric("Kümülatif Kâr",
                        f"{tl_now['cum_pnl_tl']:+,.0f} ₺",
                        delta=f"{cum_pct:+.2f}%")
    info_cols[5].metric("Adım P&L", f"{tl_now['step_pnl_tl']:+,.0f} ₺")

    # Paneller
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("📊 Durum vektöründen top-8 öznitelik (z-score)")
        df_state = _state_top_features(snap["state"], top_k=8)
        st.dataframe(df_state, hide_index=True, width='stretch', height=310)

        if snap["q_values"] is not None:
            st.plotly_chart(_q_bar(snap["q_values"], snap["action_idx"]),
                            width='stretch', key="test_q_bar")

    with right:
        st.plotly_chart(_weights_pie(snap["weights_after"]),
                        width='stretch', key="test_weights_pie")
        st.plotly_chart(_reward_bar(snap["reward_terms"]),
                        width='stretch', key="test_reward_bar")

    # --- İflas uyarısı (eğer bu adım bankrupt ise) ---
    if snap["reward_terms"].get("bankrupt"):
        st.error(
            f"💀 **İFLAS** — NAV {snap['nav']:.4f} iflas eşiğinin altına düştü "
            f"(başlangıç sermayesi tükendi). Simülasyon bu adımda sonlandı. "
            f"Uygulanan ek ceza: −{snap['reward_terms'].get('bankruptcy_penalty', 0):.1f} "
            f"(ödüle eklendi)."
        )

    # --- TL paneli: portföy tablosu + işlem logu ---
    st.divider()
    w_prev = snap["weights_before"]
    tl_left, tl_right = st.columns([1.3, 1])
    with tl_left:
        st.subheader("💰 Portföy Tablosu")
        st.caption("Durum: **YENİ** = bu adımda alındı · **TUTULUYOR** = pozisyon devam ediyor · "
                   "**ÇIKIŞ** = bu adımda satıldı")
        df_port = build_portfolio_table(BIST28, tl_now, include_cash=True, w_prev=w_prev)
        st.dataframe(df_port, hide_index=True, width='stretch', height=360)
    with tl_right:
        st.subheader("🧾 Bu adımın işlem logu")
        df_trade = build_trade_log(BIST28, tl_now, threshold_tl=1.0)
        if df_trade.empty:
            rebal = env_rebalance_hint(step_days)
            st.info(f"Bu adımda işlem yok (rebalans her {rebal} günde bir).")
        else:
            st.dataframe(df_trade, hide_index=True, width='stretch', height=280)
        commission_tl = float(tl_now["commission_tl"])
        st.caption(f"Bu adımda komisyon: **{commission_tl:,.2f} ₺** "
                   f"(tx_cost oranı × portföy TL)")

    # --- Kümülatif işlem geçmişi (başından bu adıma kadar) ---
    st.subheader(f"📋 Kümülatif İşlem Geçmişi (adım 0 → {idx})")
    dates_list = [t["date"] for t in trace]
    df_hist = build_cumulative_trade_log(BIST28, tl_snaps, dates_list,
                                         threshold_tl=1.0, up_to_step=idx)
    if df_hist.empty:
        st.info("Henüz işlem kaydı yok.")
    else:
        total_buys = float(df_hist.loc[df_hist["İşlem"] == "ALIŞ", "TL"].sum())
        total_sells = float(df_hist.loc[df_hist["İşlem"] == "SATIŞ", "TL"].sum())
        hc = st.columns(3)
        hc[0].metric("Toplam işlem", f"{len(df_hist):,}")
        hc[1].metric("Toplam ALIŞ TL", f"{total_buys:,.0f} ₺")
        hc[2].metric("Toplam SATIŞ TL", f"{total_sells:,.0f} ₺")
        # En yeni işlemler üstte
        st.dataframe(df_hist.iloc[::-1].reset_index(drop=True),
                     hide_index=True, width='stretch', height=320)

    # --- Kümülatif/Step PnL grafikleri (başlangıçtan bu adıma) ---
    st.subheader("📈 TL Kazanç Zaman Serisi (başlangıçtan bu adıma)")
    cum_arr = np.array([s["cum_pnl_tl"] for s in tl_snaps[: idx + 1]])
    step_arr = np.array([s["step_pnl_tl"] for s in tl_snaps[: idx + 1]])
    steps_idx = np.arange(len(cum_arr))
    pnl_left, pnl_right = st.columns(2)
    with pnl_left:
        fig_cum = go.Figure(go.Scatter(x=steps_idx, y=cum_arr, mode="lines",
                                       name="Kümülatif P&L",
                                       line=dict(color="#1f77b4")))
        fig_cum.add_hline(y=0, line_dash="dot", line_color="#888")
        fig_cum.update_layout(title="Kümülatif Kâr/Zarar (TL)", height=300,
                              margin=dict(t=40, b=30), xaxis_title="Adım",
                              yaxis_title="TL")
        st.plotly_chart(fig_cum, width='stretch', key="test_cum_pnl")
    with pnl_right:
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in step_arr]
        fig_step = go.Figure(go.Bar(x=steps_idx, y=step_arr, marker_color=colors))
        fig_step.update_layout(title="Adım P&L (TL)", height=300,
                               margin=dict(t=40, b=30), xaxis_title="Adım",
                               yaxis_title="TL")
        st.plotly_chart(fig_step, width='stretch', key="test_step_pnl")

    # Adaptif katsayı zaman serisi (o ana kadar)
    st.subheader("📈 Adaptif katsayılar (baştan bu adıma kadar)")
    rt_hist = pd.DataFrame([t["reward_terms"] for t in trace[: idx + 1]])
    rt_hist["step"] = range(len(rt_hist))
    cc = st.columns(3)
    cc[0].plotly_chart(
        px.line(rt_hist, x="step", y="eta_t", title="η_t (tx cost katsayısı)"),
        width='stretch', key="test_eta_t")
    cc[1].plotly_chart(
        px.line(rt_hist, x="step", y="lambda_t", title="λ_t (DD penalty katsayısı)"),
        width='stretch', key="test_lambda_t")
    cc[2].plotly_chart(
        px.line(rt_hist, x="step", y="tau_t", title="τ_t (DD eşiği)"),
        width='stretch', key="test_tau_t")

    # Oynatma döngüsü — session_state.playing true iken otomatik ilerle
    if st.session_state.playing and idx < max_step:
        time.sleep(0.15)
        st.session_state.step_idx += 1
        st.rerun()
    elif st.session_state.playing and idx >= max_step:
        st.session_state.playing = False

```

---

## `ui/tabs/compare.py`

```python
"""Sekme 4 — Karşılaştırma (app.py'den tasindi, P5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from env.portfolio_env import ACTION_NAMES
from ui.state import ASSET_NAMES
from utils.baselines import buy_and_hold_index, equal_weight, mean_variance
from utils.metrics import summary


def tab_compare():
    st.header("📊 Karşılaştırma — Ajanlar vs. Baseline'lar")

    if not st.session_state.data_loaded:
        st.warning("Önce sidebar'dan veriyi yükleyin.")
        return

    if st.button("Baseline'ları çalıştır (EW + BuyHold + MeanVar)"):
        with st.spinner("Baseline'lar hesaplanıyor..."):
            px_te = st.session_state.px_te
            st.session_state.baselines = {
                "EqualWeight": equal_weight(px_te),
                "BuyHold":     buy_and_hold_index(px_te),
                "MeanVar":     mean_variance(px_te, lookback=120, rebalance=20),
            }
        st.success("Baseline'lar hazır.")

    trained = st.session_state.trained_agents
    traces = st.session_state.test_traces
    baselines = st.session_state.baselines or {}

    if not trained and not baselines:
        st.info("Henüz eğitilmiş ajan yok. Tab 2'de en az bir ajan eğitin, sonra Tab 3'te trajectory çıkarın.")
        return

    # Metrik tablosu
    rows = {}
    nav_map = {}

    for key, (agent, _curve) in trained.items():
        algo, step_days, adaptive = key[:3]
        if key not in traces:
            continue
        tr = traces[key]
        initial_w = np.zeros_like(tr[0]["weights_before"])
        initial_w[-1] = 1.0
        nav = np.concatenate([[1.0], [t["nav"] for t in tr]])
        rets = np.concatenate([[0.0], [
            t["reward_terms"]["gross_port_r"] - t["reward_terms"]["tx_cost"] for t in tr]])
        W = np.vstack([initial_w, [t["weights_after"] for t in tr]])
        dates = pd.to_datetime([tr[0]["decision_date"]] + [t["date"] for t in tr])
        turns = np.concatenate([[0.0], [t["reward_terms"]["delta_w_l1"] for t in tr]])
        m = summary(nav, rets, W, dates=dates, turnover_values=turns)
        name = f"{algo}-{step_days}g{'·A' if adaptive else ''}"
        rows[name] = m
        nav_map[name] = nav

    for bn, bd in baselines.items():
        m = summary(bd["nav"], bd["rets"], bd.get("weights"), dates=bd.get("dates"),
                    turnover_values=bd.get("turnover"))
        rows[bn] = m
        nav_map[bn] = bd["nav"]

    if not rows:
        st.info("Ajan metriklerini görmek için önce Tab 3'te test çalıştırın.")
        return

    met_df = pd.DataFrame(rows).T.round(4)
    st.dataframe(met_df, width='stretch')
    st.download_button("metrics.csv olarak indir",
                       data=met_df.to_csv().encode("utf-8"),
                       file_name="metrics.csv", mime="text/csv")

    # Hizalanmış NAV eğrileri
    nav_series = {}
    for name, values in nav_map.items():
        if name in rows and name.split("-")[0] in {"DQN", "PPO", "SAC", "TD3"}:
            matching = next((traces[k] for k in trained if k in traces and
                             f"{k[0]}-{k[1]}g{'·A' if k[2] else ''}" == name), None)
            idx = (pd.to_datetime([matching[0]["decision_date"]] + [t["date"] for t in matching])
                   if matching else st.session_state.px_te.index[-len(values):])
        else:
            idx = pd.DatetimeIndex(baselines[name].get("dates", st.session_state.px_te.index[:len(values)]))
        nav_series[name] = pd.Series(values, index=idx)
    nav_df = pd.concat(nav_series, axis=1, join="inner").sort_index()
    fig = go.Figure()
    for c in nav_df.columns:
        fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df[c], name=c, mode="lines"))
    fig.update_layout(title="Test dönemi NAV (başlangıç = 1.0)",
                      height=440, yaxis_title="NAV", xaxis_title="Tarih")
    st.plotly_chart(fig, width='stretch', key="compare_nav")

    # Ağırlık heatmap (seçilebilir)
    st.subheader("🔥 Ağırlık Isı Haritası")
    agent_keys_with_traces = [k for k in trained if k in traces]
    if agent_keys_with_traces:
        chosen = st.selectbox(
            "Ajan seç",
            agent_keys_with_traces,
            format_func=lambda k: f"{k[0]}-{k[1]}g{'·A' if k[2] else ''}"
        )
        tr = traces[chosen]
        W = np.array([t["weights_after"] for t in tr])
        # Örnekleme: çok uzunsa her N'incisini al
        step = max(1, len(W) // 200)
        Wp = W[::step]
        fig_hm = px.imshow(
            Wp.T,
            aspect="auto",
            origin="lower",
            color_continuous_scale="RdBu_r",
            zmin=0, zmax=min(0.3, float(Wp.max()) * 1.05) if Wp.max() > 0 else 0.3,
            labels=dict(x="Gün (örneklenmiş)", y="Varlık", color="Ağırlık"),
        )
        fig_hm.update_yaxes(ticktext=ASSET_NAMES,
                            tickvals=list(range(len(ASSET_NAMES))))
        fig_hm.update_layout(height=520, title=f"{chosen[0]} ağırlıkları")
        st.plotly_chart(fig_hm, width='stretch', key="compare_heatmap")

        # DQN → aksiyon dağılımı
        if chosen[0] == "DQN":
            action_counts = pd.Series(
                [t["action_idx"] for t in tr if t["action_idx"] is not None]
            ).value_counts().reindex(range(6), fill_value=0)
            action_counts.index = ACTION_NAMES[:6]
            fig_a = px.bar(action_counts, title="DQN — aksiyon dağılımı (gün sayısı)")
            fig_a.update_layout(height=320, showlegend=False,
                                yaxis_title="Gün sayısı", xaxis_title="Şablon")
            st.plotly_chart(fig_a, width='stretch', key="compare_dqn_actions")

        # Adaptif katsayılar — zaman serisi
        st.subheader("📐 Seçili ajanın adaptif katsayıları (test dönemi)")
        rt_df = pd.DataFrame([t["reward_terms"] for t in tr])
        rt_df["date"] = [t["date"] for t in tr]
        sub = st.columns(3)
        sub[0].plotly_chart(px.line(rt_df, x="date", y="eta_t", title="η_t"),
                            width='stretch', key="compare_eta_t")
        sub[1].plotly_chart(px.line(rt_df, x="date", y="lambda_t", title="λ_t"),
                            width='stretch', key="compare_lambda_t")
        sub[2].plotly_chart(px.line(rt_df, x="date", y="tau_t", title="τ_t"),
                            width='stretch', key="compare_tau_t")

```
