"""Coklu-seed deney koreografisi — bilimsel saglamlik (hakem elestirisi #1).

Tek-seed sonuc (results/metrics.csv) bir RNG cekilisinin orneklemi; raporlanan
CAGR/Sharpe'in seed'e ne kadar duyarli oldugunu gostermez. Bu script her algoritmayi
N farkli seed ile bagimsiz egitir, test setinde degerlendirir ve metrikleri
ortalama±std olarak toplar (hakem: "tek seed yetersiz").

KANONIK AKISI BIREBIR TEKRARLAR (yeni pipeline ICAT ETMEZ):
  train.run()'in tek-seed yolu DOGRUDAN cagrilir — yeniden-implemente edilmez:
    np.random.seed(S) -> prepare_data() -> train_dqn -> train_ppo -> train_sac
    -> train_td3 -> evaluate(...) -> utils.metrics.summary.
  train.py'deki train_dqn/ppo/sac/td3 ve evaluate fonksiyonlari OLDUGU GIBI
  cagrilir; tek fark: kanonik SEED=42 sabiti yerine, dis dongunun seed'i (S)
  train modulunun SEED global'ine gecici olarak zerk edilir (_seed_scope).
  train_* ve prepare_data RNG'yi (build_env seed, build_agent seed, forecast
  set_seed) runtime'da train.SEED'ten okudugundan, S=42 satiri kanonik tek-seed
  cikti (results/metrics.csv / tests/golden/metrics_baseline.csv) ile BIT-AYNI olur.

NEDEN train_* DOGRUDAN CAGRILIR (onceki _train_one'in kusuru):
  Onceki surum build_env'i `random_start=False` ile cagiriyordu; oysa kanonik
  train_dqn/ppo/sac/td3 `random_start=EnvConfig.random_start` (=True) gecer.
  random_start=True (rastgele baslangic penceresi + fiyat-gurultusu/slippage)
  egitim dagilimini tamamen degistirir; RNG'ye en duyarli ajan olan DQN bu yuzden
  S=42'de bile kanonik 1.28 yerine 2.63 uretiyordu. train_* fonksiyonunu dogrudan
  cagirmak bu tur sessiz cagri-sirasi/parametre sapmalarini yapisal olarak onler.

RNG IZOLASYONU: her seed kendi np.random.seed(S)'iyle baslar (train.run gibi);
ajan ctor'lari da set_seed(S) cagirir (agirlik init S'e baglidir). Seed'ler arasi
durum sizmaz — her (algo, seed) bagimsiz bir cekilis. DQN'in seed'ler arasi
yuksek std'si GERCEK bir bulgudur (RNG-duyarliligi); ayni seed+ayni sira ise
bit-ozdestir (determinizm).

Cikti (results/ — gitignore'lu, yerel artefakt):
  - multiseed_runs.csv     : her (algo, seed) bir satir (ham metrikler)
  - multiseed_summary.csv  : algo basina mean / std (tum metrikler)

Kullanim (Windows / PowerShell):
  .venv\\Scripts\\python.exe scripts\\multiseed.py --seeds 42 43 44 45 46
  .venv\\Scripts\\python.exe scripts\\multiseed.py --seeds 42 --algos DQN   # smoke
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

# kod/ kokunu path'e ekle (scripts/ alt-dizininden cagrildiginda).
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd

# train.py'nin KANONIK yolunu DOGRUDAN yeniden kullan (taklit degil, ayni
# fonksiyonlar): prepare_data + train_dqn/ppo/sac/td3 + evaluate.
import train as canonical
from utils.metrics import summary

RES = BASE / "results"
RES.mkdir(exist_ok=True)

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
ALL_ALGOS = ["DQN", "PPO", "SAC", "TD3"]

# train.run()'daki sira ve metrik anahtarlari (utils.metrics.summary cikti seti).
METRIC_KEYS = ["CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar",
               "Volatility", "FinalNAV", "Turnover"]

# Kanonik train_* fonksiyonlari (train.run()'daki sira: DQN -> PPO -> SAC -> TD3).
# build_agent/build_env/train_loop cagri zinciri ICLERINDE; multiseed bunlari
# YENIDEN-IMPLEMENTE ETMEZ, oldugu gibi cagirir (cagri-sirasi sapmasi imkansiz).
_TRAINERS = {
    "DQN": canonical.train_dqn,
    "PPO": canonical.train_ppo,
    "SAC": canonical.train_sac,
    "TD3": canonical.train_td3,
}


@contextlib.contextmanager
def _seed_scope(seed: int):
    """train modulunun SEED global'ini gecici olarak `seed` yap.

    train.prepare_data ve train.train_dqn/ppo/sac/td3 RNG tohumunu (build_env
    seed=, build_agent seed=, forecast set_seed(SEED)) runtime'da `train.SEED`'ten
    okur. Bu kapsamda SEED=seed olunca, kanonik akis tipki SEED=seed ile
    main.py calismis gibi davranir. seed=42 -> kanonik tek-seed ile bit-ayni.
    """
    old = canonical.SEED
    canonical.SEED = seed
    try:
        yield
    finally:
        canonical.SEED = old


# ---------------------------------------------------------------- koreografi
def run(seeds: list[int], algos: list[str]) -> pd.DataFrame:
    """Her seed icin kanonik train.run() yolunu BIREBIR tekrarla.

    Dis dongu seed, ic dongu algo — train.run()'in tek-seed'de
    np.random.seed(SEED) -> prepare_data() -> DQN -> PPO -> SAC -> TD3 -> evaluate
    sirasiyla AYNIDIR. Tek fark: SEED sabiti yerine S (_seed_scope ile zerk).

    NOT: --algos bir alt-kume olsa bile, kanonik RNG akisini bozmamak icin
    EGITIM her zaman tam DQN->PPO->SAC->TD3 sirasinda yurutulur; yalniz istenen
    algolar degerlendirilip raporlanir. (Aksi halde atlanan bir ajanin RNG
    tuketimi sonraki ajanlarin cekilisini kaydirir -> kanonik esitlik bozulur.)
    """
    rows = []
    for s in seeds:
        with _seed_scope(s):
            # train.run() ile birebir: once global numpy RNG'yi tohumla.
            np.random.seed(canonical.SEED)
            t_seed = time.time()
            print("=" * 64)
            print(f"[SEED {s}] veri hazirlaniyor (random_start={'AC' if True else ''})...")
            bundle = canonical.prepare_data()

            # Tam kanonik sira (DQN -> PPO -> SAC -> TD3); ajanlari sakla.
            agents: dict = {}
            for algo in ALL_ALGOS:
                t0 = time.time()
                agent, _curve = _TRAINERS[algo](bundle)
                agents[algo] = agent
                print(f"[SEED {s}] {algo:<3} egitildi ({time.time() - t0:.1f}s)")

            # Degerlendirme + raporlama (yalniz istenen algolar icin satir uret).
            for algo in algos:
                bt = canonical.evaluate(bundle, agents[algo], algo)
                m = summary(bt["nav"], bt["rets"], bt["weights"])
                row = {"algo": algo, "seed": s, **{k: m[k] for k in METRIC_KEYS}}
                rows.append(row)
                print(f"[SEED {s}] {algo:<3}  CAGR={m['CAGR']:+.2%}  "
                      f"Sharpe={m['Sharpe']:+.2f}  MaxDD={m['MaxDD']:+.2%}  "
                      f"Final={m['FinalNAV']:.3f}")
            print(f"[SEED {s}] tamam ({time.time() - t_seed:.1f}s)")

    runs = pd.DataFrame(rows, columns=["algo", "seed", *METRIC_KEYS])
    return runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Algo basina mean ± std (tum metrikler). Sutun adlari '<metrik>_mean' /
    '<metrik>_std' — rapor tablosu ve ileride T-test/CI icin makine-okunur."""
    g = runs.groupby("algo")[METRIC_KEYS]
    mean = g.mean().add_suffix("_mean")
    std = g.std(ddof=1).add_suffix("_std")   # ornek std (N-1) — seed cekilisi orneklem
    out = pd.concat([mean, std], axis=1)
    # Sutunlari metrik-bazli grupla (CAGR_mean, CAGR_std, Sharpe_mean, ...).
    ordered = [f"{k}_{stat}" for k in METRIC_KEYS for stat in ("mean", "std")]
    out = out[ordered]
    out.insert(0, "n_seeds", g.size())
    return out


def main():
    ap = argparse.ArgumentParser(description="Coklu-seed RL deneyi (ort±std).")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help=f"Seed listesi (default: {DEFAULT_SEEDS})")
    ap.add_argument("--algos", type=str, nargs="+", default=ALL_ALGOS,
                    choices=ALL_ALGOS, help=f"Algoritmalar (default: {ALL_ALGOS})")
    args = ap.parse_args()

    print(f"Seeds: {args.seeds}  |  Algos: {args.algos}")
    t0 = time.time()
    runs = run(args.seeds, args.algos)
    summ = summarize(runs)

    runs_path = RES / "multiseed_runs.csv"
    summ_path = RES / "multiseed_summary.csv"
    runs.to_csv(runs_path, index=False)
    summ.to_csv(summ_path)

    print("=" * 64)
    print("HAM KOSULAR (her (algo, seed) bir satir):")
    print(runs.round(4).to_string(index=False))
    print("-" * 64)
    print("OZET (algo basina mean / std):")
    print(summ.round(4).to_string())
    print("-" * 64)
    print(f"Yazildi: {runs_path}")
    print(f"Yazildi: {summ_path}")
    print(f"Toplam sure: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
