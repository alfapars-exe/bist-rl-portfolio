"""Coklu-seed deney koreografisi — bilimsel saglamlik (hakem elestirisi #1).

Tek-seed sonuc (results/metrics.csv) bir RNG cekilisinin orneklemi; raporlanan
CAGR/Sharpe'in seed'e ne kadar duyarli oldugunu gostermez. Bu script her algoritmayi
N farkli seed ile bagimsiz egitir, test setinde degerlendirir ve metrikleri
ortalama±std olarak toplar (hakem: "tek seed yetersiz").

KANONIK AKISI TAKLIT EDER (yeni pipeline ICAT ETMEZ):
  train.run() ile birebir ayni cagri zinciri —
    np.random.seed(S) -> prepare_data() -> [DQN, PPO, SAC, TD3] sirayla
    her biri build_env/build_agent(seed=S) -> core.trainer.train -> core.rollout.evaluate
    -> utils.metrics.summary.
  Tek fark: SEED sabiti yerine dis dongunun seed'i (S) her asamaya zerk edilir.
  Bu sayede S=42 satiri, kanonik tek-seed cikti (results/metrics.csv /
  tests/golden/metrics_baseline.csv) ile AYNI olmalidir (determinizm cross-check).

RNG IZOLASYONU: her seed kendi set_seed(S)'iyle baslar; ajan ctor'lari da
build_agent(seed=S) ile set_seed(S) cagirir (agirlik init S'e baglidir). Seed'ler
arasi durum sizmaz — her (algo, seed) bagimsiz bir cekilis.

Cikti (results/ — gitignore'lu, yerel artefakt):
  - multiseed_runs.csv     : her (algo, seed) bir satir (ham metrikler)
  - multiseed_summary.csv  : algo basina mean / std (tum metrikler)

Kullanim (Windows / PowerShell):
  .venv\\Scripts\\python.exe scripts\\multiseed.py --seeds 42 43 44 45 46
  .venv\\Scripts\\python.exe scripts\\multiseed.py --seeds 42 43 --algos DQN   # smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# kod/ kokunu path'e ekle (scripts/ alt-dizininden cagrildiginda).
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd

from config import SEED, TrainConfig
from core.factory import build_agent, build_env
from core.rollout import evaluate as rollout_evaluate
from core.trainer import train as train_loop
from utils.metrics import summary
from utils.torch_utils import set_seed

# train.py'nin KANONIK veri hazirligini birebir yeniden kullan (taklit degil,
# dogrudan ayni fonksiyon) — prepare_data() train-only scaler + (ops.) makro/forecast.
from train import prepare_data, DataBundle

RES = BASE / "results"
RES.mkdir(exist_ok=True)

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
ALL_ALGOS = ["DQN", "PPO", "SAC", "TD3"]

# train.run()'daki sira ve metrik anahtarlari (utils.metrics.summary cikti seti).
METRIC_KEYS = ["CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar",
               "Volatility", "FinalNAV", "Turnover"]


# ---------------------------------------------------------------- tek (algo,seed)
def _train_one(algo: str, bundle: DataBundle, seed: int,
               horizon: str = "medium", adaptive: bool = True):
    """Tek bir algoritmayi tek bir seed ile egitir (train.py:train_* ile ayni
    build_env/build_agent/train_loop zinciri; SEED yerine seed=S zerk edilir).

    max_steps / n_iters / episode_len degerleri train.py'deki tek-seed
    varsayilanlarla BIREBIR aynidir (TrainConfig tek-kaynak)."""
    if algo == "DQN":
        env = build_env("DQN", bundle.px_tr, bundle.feats_tr, horizon=horizon,
                        adaptive=adaptive, max_steps=252, random_start=False, seed=seed,
                        macro=bundle.macro_tr, regime=bundle.regime_tr)
        agent = build_agent("DQN", env.state_dim, env.n_discrete, seed=seed)
        for _ in train_loop(agent, env, n_iters=TrainConfig.dqn_episodes):
            pass
        return agent

    if algo == "PPO":
        env = build_env("PPO", bundle.px_tr, bundle.feats_tr, horizon=horizon,
                        adaptive=adaptive, max_steps=10_000, random_start=False, seed=seed,
                        macro=bundle.macro_tr, regime=bundle.regime_tr)
        agent = build_agent("PPO", env.state_dim, env.action_dim, seed=seed)
        for _ in train_loop(agent, env, n_iters=TrainConfig.ppo_updates,
                            rollout_len=TrainConfig.ppo_rollout_len):
            pass
        return agent

    if algo == "SAC":
        env = build_env("SAC", bundle.px_tr, bundle.feats_tr, horizon=horizon,
                        adaptive=adaptive, max_steps=TrainConfig.sac_episode_len,
                        random_start=False, seed=seed,
                        macro=bundle.macro_tr, regime=bundle.regime_tr)
        agent = build_agent("SAC", env.state_dim, env.action_dim, seed=seed)
        for _ in train_loop(agent, env, n_iters=TrainConfig.sac_episodes):
            pass
        return agent

    if algo == "TD3":
        env = build_env("TD3", bundle.px_tr, bundle.feats_tr, horizon=horizon,
                        adaptive=adaptive, max_steps=TrainConfig.td3_episode_len,
                        random_start=False, seed=seed,
                        macro=bundle.macro_tr, regime=bundle.regime_tr)
        agent = build_agent("TD3", env.state_dim, env.action_dim, seed=seed)
        for _ in train_loop(agent, env, n_iters=TrainConfig.td3_episodes):
            pass
        return agent

    raise ValueError(f"Bilinmeyen algoritma: {algo!r}")


def _evaluate_one(algo: str, bundle: DataBundle, agent,
                  horizon: str = "medium", adaptive: bool = True) -> dict:
    """train.py:evaluate ile birebir: test ortaminda rollout + summary metrikleri."""
    env = build_env(algo, bundle.px_te, bundle.feats_te, horizon=horizon,
                    adaptive=adaptive, max_steps=10_000,
                    macro=bundle.macro_te, regime=bundle.regime_te)
    bt = rollout_evaluate(agent, env)
    return summary(bt["nav"], bt["rets"], bt["weights"])


# ---------------------------------------------------------------- koreografi
def run(seeds: list[int], algos: list[str]) -> pd.DataFrame:
    """Her seed icin (kanonik sira: veri -> DQN, PPO, SAC, TD3) egit + test et.

    DONGU SIRASI (KANONIK ESLESME): dis dongu seed, ic dongu algo — tipki
    train.run()'in tek seed'de DQN->PPO->SAC->TD3 sirasi gibi. Her seed icin
    np.random.seed(S) + set_seed(S) once cagrilir (train.run np.random.seed(SEED)
    yapardi; set_seed ek torch determinizmi icin — ajan ctor'u zaten set_seed(S)
    cagirdigindan S=42 RNG akisi degismez)."""
    rows = []
    for s in seeds:
        # Seed izolasyonu — her cekilis kendi global RNG durumuyla baslar.
        np.random.seed(s)
        set_seed(s)
        t_seed = time.time()
        print("=" * 64)
        print(f"[SEED {s}] veri hazirlaniyor...")
        bundle = prepare_data()
        for algo in algos:
            t0 = time.time()
            agent = _train_one(algo, bundle, seed=s)
            m = _evaluate_one(algo, bundle, agent)
            row = {"algo": algo, "seed": s, **{k: m[k] for k in METRIC_KEYS}}
            rows.append(row)
            print(f"[SEED {s}] {algo:<3}  CAGR={m['CAGR']:+.2%}  Sharpe={m['Sharpe']:+.2f}  "
                  f"MaxDD={m['MaxDD']:+.2%}  Final={m['FinalNAV']:.3f}  "
                  f"({time.time() - t0:.1f}s)")
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
