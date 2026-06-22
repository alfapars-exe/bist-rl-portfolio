r"""Ödül katsayısı duyarlılık analizi (OAT) — hakem eleştirisi #5 (reward engineering / test-tuning).

AMAÇ
----
"Ödül test setine uyarlandı mı (reward engineering)?" eleştirisini niceliksel olarak
yanıtlamak. Sonuçların ödül katsayılarına AŞIRI duyarlı OLMADIĞINI (sağlamlık) ya da
hangi katsayının kritik/kırılgan olduğunu gösteririz.

YÖNTEM — One-At-a-Time (OAT) lokal duyarlılık
---------------------------------------------
Her katsayıyı kendi default'u (config.DEFAULTS / RewardConfig) etrafında ~3
değerde varyasyonla tararız; DİĞER TÜM katsayıları default'ta SABİT tutarız. OAT bir
lokal türev yaklaşımıdır (∂metrik/∂katsayı): ucuzdur (Σ değerler ≈ 15 eğitim) ve "tek
parametreyi oynatınca sonuç ne kadar değişiyor?" sorusuna doğrudan cevap verir. Katsayılar
arası etkileşimi YAKALAMAZ (onun için Sobol/Morris gerekir) — burada amaç o değil,
"her bir katsayıya kırılganlık" testidir.

Taranan katsayılar (reward_overrides üzerinden enjekte edilir):
  eta_base   — işlem-maliyeti (turnover) cezası tabanı   (DEFAULTS.eta)
  lambda_base— drawdown ceza tabanı                      (DEFAULTS.lam)
  tau_base   — drawdown tolerans eşiği                   (DEFAULTS.tau)
  w_dsr      — Diferansiyel Sharpe ödül ağırlığı            (RewardConfig.w_dsr)
  w_cvar     — CVaR kuyruk-riski ceza ağırlığı              (RewardConfig.w_cvar)
  regime_beta— (opsiyonel) kriz amplifikasyon gücü          (RewardConfig.regime_beta)

REFERANS KURULUM
----------------
Tek temsili ajan: SAC, seed=42, varsayilan step_days. Tüm (katsayı, değer) noktaları AYNI
seed + AYNI veri bölünmesi + AYNI eğitim bütçesi ile koşar → metrik farkı yalnızca o
katsayıdan gelir (ceteris paribus). Her nokta için test setinde backtest → Sharpe / MaxDD
/ Calmar / FinalNAV.

GOLDEN-GÜVENLİ: kanonik config default'ları, golden-master, results/metrics.csv DEĞİŞMEZ.
Bu script yalnız results/sensitivity.csv üretir (gitignore'lu). reward_overrides akışı
(core.factory.build_env) zaten parametriktir; biz yalnız bu mevcut kapıyı kullanırız.

ÇALIŞTIRMA (Windows / PowerShell)
---------------------------------
  # TAM OAT sweep (~15 SAC eğitimi; arka planda koşturulması önerilir):
  .venv\Scripts\python.exe scripts\reward_sensitivity.py

  # SMOKE (yalnız 1 katsayı x 2 değer; format doğrulama, hızlı):
  .venv\Scripts\python.exe scripts\reward_sensitivity.py --smoke

  # Belirli katsayı(lar)ı tara:
  .venv\Scripts\python.exe scripts\reward_sensitivity.py --params w_dsr,w_cvar

  # regime_beta'yı da dahil et:
  .venv\Scripts\python.exe scripts\reward_sensitivity.py --include-regime

ÇIKTI: results/sensitivity.csv  (kolonlar: param, value, sharpe, maxdd, calmar, final_nav)

YORUM
-----
Her 'param' bloğunda metriklerin değer ızgarası boyunca YAYILIMINA (range / std) bak:
  - DÜŞÜK yayılım  -> sonuç o katsayıya DUYARSIZ = SAĞLAM (reward o katsayıdan dolayı
    test'e uydurulmamış; default 'şanslı bir nokta' değil).
  - YÜKSEK yayılım -> o katsayı KRİTİK/KIRILGAN; default seçimi gerekçelendirilmeli ve
    raporda açıkça belirtilmeli (cherry-picking riski o katsayıda yoğunlaşır).
Ayrıca default değerin komşularına göre bir 'tepe'de (sadece orada iyi) olup olmadığına
bak: default civarı düz/monoton ise sağlamlık lehine güçlü kanıttır.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULTS, SEED, RewardConfig, TrainConfig  # noqa: E402
from core.factory import build_agent, build_env                       # noqa: E402
from core.rollout import evaluate as rollout_evaluate                 # noqa: E402
from core.trainer import train as train_loop                          # noqa: E402
from utils.metrics import summary                                     # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"

# --- Referans kurulum: tek temsili ajan (sweep boyu sabit) -------------------
ALGO = "SAC"


def _defaults() -> dict:
    """Taranan 6 katsayının kanonik default'u (sweep'in 'merkez' noktası).

    eta/lambda/tau StepDefaults; w_dsr/w_cvar/regime_beta RewardConfig kaynaklidir.
    build_env override anahtarlari ile birebir ayni isimlendirme kullanilir.
    """
    return {
        "eta_base": DEFAULTS.eta,
        "lambda_base": DEFAULTS.lam,
        "tau_base": DEFAULTS.tau,
        "w_dsr": RewardConfig.w_dsr,
        "w_cvar": RewardConfig.w_cvar,
        "regime_beta": RewardConfig.regime_beta,
    }


# --- OAT değer ızgaraları (default ~ orta değer; alt/üst komşular) -----------
# Izgaralar varsayilan degerin alt/ust komsularini kapsar.
GRIDS = {
    "eta_base":    [0.0005, 0.0010, 0.0020],   # default(medium)=0.0010
    "lambda_base": [0.25,   0.50,   1.00],     # default(medium)=0.50
    "tau_base":    [0.03,   0.05,   0.08],     # default(medium)=0.05
    "w_dsr":       [0.00,   0.05,   0.10],     # default=0.05  (0 -> DSR kapalı)
    "w_cvar":      [0.00,   0.06,   0.12],     # default=0.06  (0 -> CVaR kapalı)
    "regime_beta": [0.0,    1.0,    2.0],      # default=1.0   (opsiyonel)
}

# regime_beta opsiyonel: --include-regime ile açılır.
DEFAULT_PARAMS = ["eta_base", "lambda_base", "tau_base", "w_dsr", "w_cvar"]


_NAN_METRICS = {"sharpe": float("nan"), "maxdd": float("nan"),
                "calmar": float("nan"), "final_nav": float("nan")}


def _run_point(bundle, overrides: dict, *, n_episodes: int,
               max_steps: int) -> dict:
    """Tek (katsayı, değer) noktası: SAC eğit (train) + test backtest (eval).

    overrides: build_env reward_overrides — DİĞER katsayılar default'ta kalır.
    AYNI seed + AYNI veri + AYNI bütçe -> metrik farkı yalnız bu override'dan.
    """
    # Test seti yoksa/çok kısaysa (bu shell yalnız kısmi BIST cache içerebilir;
    # kanonik env'de dolu) -> eğitim atlanır, NaN metrik döner. Kanonik tam-veri
    # akışında bu dal çalışmaz.
    if len(bundle.px_te) <= 60:
        print("    [uyari] test seti bos/cok kisa -> bu nokta atlandi (NaN). "
              "Kanonik env'de (tam veri) calisir.")
        return dict(_NAN_METRICS)

    # Eğitim ortamı (kanonik train.train_sac ile aynı kurulum: random_start,
    # max_steps, makro/regime). Tek fark: reward_overrides enjekte edilir.
    train_env = build_env(
        ALGO, bundle.px_tr, bundle.feats_tr, adaptive=True,
        max_steps=max_steps, random_start=True, seed=SEED,
        reward_overrides=overrides,
        macro=bundle.macro_tr, regime=bundle.regime_tr,
        rebalance_freq=1, step_days=bundle.step_days, gamma=DEFAULTS.gamma,
        mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window,
    )
    agent = build_agent(ALGO, train_env.state_dim, train_env.action_dim, seed=SEED)
    # Generator'i sonuna kadar tüket (eğitim yan-etkili).
    for _ in train_loop(agent, train_env, n_iters=n_episodes):
        pass

    # Test ortamı: AYNI override'lar (reward terimleri eval NAV'ı doğrudan
    # değiştirmez ama tutarlılık + olası reward-bağlı kapı durumları için aktarılır;
    # eval'de random_start/price_noise kapalı -> determinizm).
    eval_env = build_env(
        ALGO, bundle.px_te, bundle.feats_te, adaptive=True,
        max_steps=10_000, random_start=False, seed=SEED,
        reward_overrides=overrides,
        macro=bundle.macro_te, regime=bundle.regime_te,
        rebalance_freq=1, step_days=bundle.step_days, gamma=DEFAULTS.gamma,
        mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window,
    )
    bt = rollout_evaluate(agent, eval_env)
    m = summary(bt["nav"], bt["rets"], bt["weights"], dates=bt["dates"],
                turnover_values=bt.get("turnover"))
    return {
        "sharpe": float(m["Sharpe"]),
        "maxdd": float(m["MaxDD"]),
        "calmar": float(m["Calmar"]),
        "final_nav": float(m["FinalNAV"]),
    }


def run(params: list[str] | None = None, *, smoke: bool = False,
        include_regime: bool = False) -> pd.DataFrame:
    """OAT sweep'i koşturur, results/sensitivity.csv yazar, DataFrame döner."""
    import numpy as np
    from train import prepare_data

    np.random.seed(SEED)              # train.run() ile aynı global seed sırası
    bundle = prepare_data()

    base = _defaults()

    if smoke:
        # MİNİMAL: 1 katsayı x 2 değer (format doğrulama; tam sweep değil).
        params = ["w_dsr"]
        grids = {"w_dsr": [0.00, 0.10]}
        n_episodes = max(2, TrainConfig.sac_episodes // 4)
        max_steps = 200
    else:
        if params is None:
            params = list(DEFAULT_PARAMS)
            if include_regime:
                params.append("regime_beta")
        grids = {p: GRIDS[p] for p in params}
        n_episodes = TrainConfig.sac_episodes      # kanonik bütçe
        max_steps = TrainConfig.sac_episode_len

    rows = []
    total = sum(len(grids[p]) for p in params)
    done = 0
    print("=" * 64)
    print(f"ODUL DUYARLILIK (OAT) — {ALGO} seed={SEED} step_days={bundle.step_days} "
          f"{'[SMOKE]' if smoke else ''}")
    print(f"  Taranan katsayilar: {params}")
    print(f"  Egitim butcesi: n_episodes={n_episodes} max_steps={max_steps}")
    print(f"  Toplam nokta: {total}")
    print("=" * 64)

    for p in params:
        for v in grids[p]:
            done += 1
            overrides = dict(base)     # DİĞER katsayılar default
            overrides[p] = v           # yalnız bu katsayı varyasyonu
            print(f"[{done:02d}/{total}] {p} = {v} ...", flush=True)
            metr = _run_point(bundle, overrides, n_episodes=n_episodes,
                              max_steps=max_steps)
            row = {"param": p, "value": float(v), **metr}
            rows.append(row)
            print(f"    -> sharpe={metr['sharpe']:+.3f} maxdd={metr['maxdd']:+.3f} "
                  f"calmar={metr['calmar']:+.3f} final_nav={metr['final_nav']:.3f}")

    df = pd.DataFrame(rows, columns=["param", "value", "sharpe", "maxdd",
                                     "calmar", "final_nav"])
    RES.mkdir(exist_ok=True)
    out = RES / "sensitivity.csv"
    df.to_csv(out, index=False)
    print("=" * 64)
    print(f"Kaydedildi: {out}  (satir={len(df)})")

    # Duyarlılık özeti: her katsayı icin metrik yayilimi (range = max-min).
    # Yuksek range = o katsayi kritik/kirilgan; dusuk range = saglam.
    if len(df) > 1:
        print("-" * 64)
        print("Katsayi bazli yayilim (range = max - min; YUKSEK = duyarli/kirilgan):")
        for p in params:
            sub = df[df["param"] == p]
            if len(sub) < 2:
                continue
            print(f"  {p:<12} sharpe_range={sub['sharpe'].max() - sub['sharpe'].min():.3f}  "
                  f"finalnav_range={sub['final_nav'].max() - sub['final_nav'].min():.3f}")
    print("=" * 64)
    return df


def main():
    ap = argparse.ArgumentParser(description="Odul katsayisi duyarlilik analizi (OAT)")
    ap.add_argument("--smoke", action="store_true",
                    help="Minimal calistirma (1 katsayi x 2 deger; format dogrulama)")
    ap.add_argument("--params", type=str, default=None,
                    help="Virgulle ayrilmis katsayi listesi (orn: w_dsr,w_cvar)")
    ap.add_argument("--include-regime", action="store_true",
                    help="regime_beta katsayisini da tara (opsiyonel)")
    args = ap.parse_args()

    params = None
    if args.params:
        params = [p.strip() for p in args.params.split(",") if p.strip()]
        unknown = [p for p in params if p not in GRIDS]
        if unknown:
            raise SystemExit(f"Bilinmeyen katsayi(lar): {unknown}; "
                             f"gecerli: {sorted(GRIDS)}")

    run(params=params, smoke=args.smoke, include_regime=args.include_regime)


if __name__ == "__main__":
    main()
