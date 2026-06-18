"""PDF §9.7 titizlik (rigor) katmanı — Deflated Sharpe + PBO + Monte-Carlo stres + reel-NAV.

PARS referans ağacından kanonik BIST ağacına port edilen `utils/deflated_sharpe.py`
ve `utils/stress_mc.py` modüllerini, KANONIK deterministik eval çıktılarına
(`results/navs_aligned.csv`, `results/weights_*.csv`) uygular.

GÖZLEMSEL / golden-güvenli: yalnız mevcut deterministik eval NAV/getirilerini okur;
golden-master `results/metrics.csv` DEĞİŞMEZ. Ayrı dosyalar üretir:
  results/rigor_metrics.csv   — strateji-başı DSR/PSR + nominal/reel FinalNAV
  results/rigor_summary.csv   — PBO + Monte-Carlo özet skalerleri
  results/rigor_mc_terminal.csv — MC terminal getiri dağılımları (F12 için)
  results/navs_real.csv       — reel (USD-bazlı) NAV zaman serileri (F13 için)

Çalıştır: `python scripts/rigor_analysis.py`  (veya `python main.py` → step_rigor).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurt
from scipy.stats import skew as _skew

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SEED                                   # noqa: E402
from utils.deflated_sharpe import (cscv_pbo,              # noqa: E402
                                   deflated_sharpe_ratio,
                                   probabilistic_sharpe_ratio)
from utils.metrics import real_nav                        # noqa: E402
from utils.stress_mc import mc_block_bootstrap, mc_student_t  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"

RL_AGENTS = ["DQN", "PPO", "SAC", "TD3"]
# Gelistirme boyu denenen config sayisi (V1->V8 durum/odul iterasyonu) — coklu-deneme
# deflasyonu icin alt-sinir. Strateji sayisi bundan buyukse o kullanilir. Dokumante.
N_DEV_TRIALS = 8


def _per_obs_sharpe(rets: np.ndarray) -> float:
    r = np.asarray(rets, dtype=float)
    sd = r.std()
    return float(r.mean() / sd) if sd > 1e-12 else 0.0


def compute_rigor(navs: pd.DataFrame, weights: dict, asset_rets, usdtry) -> dict:
    """navs: (T, N_strateji) NAV; weights: {ajan: (T,29) df}; asset_rets: (T,28) BIST
    test getirileri (MC için, opsiyonel); usdtry: (T,) USD/TRY (reel-NAV için, opsiyonel)."""
    strategies = list(navs.columns)
    rets = {s: navs[s].pct_change().dropna().to_numpy(float) for s in strategies}
    sr_perobs = {s: _per_obs_sharpe(rets[s]) for s in strategies}
    # Var_SR: deneme Sharpe'larının kesit varyansı (CSCV/DSR için multiple-testing ölçeği)
    sr_var = float(np.var(list(sr_perobs.values()), ddof=1)) if len(strategies) > 1 else 0.0
    n_trials = max(len(strategies), N_DEV_TRIALS)

    rows, real_cols = [], {}
    for s in strategies:
        r = rets[s]
        sk = float(_skew(r)) if len(r) > 2 else 0.0
        ku = float(_kurt(r, fisher=False)) if len(r) > 2 else 3.0   # Pearson (normal=3)
        psr = probabilistic_sharpe_ratio(sr_perobs[s], len(r), sk, ku, 0.0)
        dsr = deflated_sharpe_ratio(sr_perobs[s], len(r), n_trials, sr_var, sk, ku)
        row = dict(strategy=s, sharpe_perobs=sr_perobs[s],
                   sharpe_ann=sr_perobs[s] * np.sqrt(252.0),
                   PSR=psr, DSR=dsr, skew=sk, kurtosis=ku, n_obs=len(r),
                   FinalNAV_nominal=float(navs[s].to_numpy(float)[-1]))
        if usdtry is not None:
            rn = real_nav(navs[s].to_numpy(float), usdtry)
            row["FinalNAV_real_usd"] = float(rn[-1])
            real_cols[s] = rn
        rows.append(row)
    rigor = pd.DataFrame(rows).set_index("strategy")
    navs_real = (pd.DataFrame(real_cols, index=navs.index[:len(next(iter(real_cols.values())))])
                 if real_cols else None)

    # PBO: tüm strateji config'leri üzerinde CSCV
    min_t = min(len(rets[s]) for s in strategies)
    perf = np.column_stack([rets[s][-min_t:] for s in strategies])
    pbo = cscv_pbo(perf, n_splits=10)

    # Monte-Carlo stres: gözlem-başı Sharpe'a göre en iyi RL ajan
    mc = {}
    rl = [s for s in strategies if s in RL_AGENTS]
    if rl and asset_rets is not None and len(asset_rets) > 30:
        best = max(rl, key=lambda s: sr_perobs[s])
        if best in weights:
            w_avg = weights[best].to_numpy(float).mean(axis=0)
            hz = int(min(252, len(asset_rets)))
            bb = mc_block_bootstrap(asset_rets, w_avg, horizon=hz, paths=3000, block=20, seed=SEED)
            st = mc_student_t(asset_rets, w_avg, horizon=hz, paths=3000, dof=5.0, seed=SEED)
            mc = {"best_agent": best, "block_bootstrap": bb["summary"], "student_t": st["summary"],
                  "terminal": pd.DataFrame({"block_bootstrap": bb["terminal"],
                                            "student_t": st["terminal"]})}
    return {"rigor": rigor, "navs_real": navs_real, "pbo": pbo, "mc": mc,
            "n_trials": n_trials, "sr_var": sr_var}


def _load_market():
    """BIST test getirileri (MC) + USD/TRY (reel-NAV). Yüklenemezse (None, None)."""
    try:
        from config import MacroConfig
        from data import align_macro, download_bist, download_macro, train_test_split
        px = download_bist()
        _, px_te = train_test_split(px)
        asset_rets = px_te.pct_change().dropna().to_numpy(np.float64)
        usdtry = None
        if MacroConfig.enabled:
            m = align_macro(download_macro(), px.index)
            if "USDTRY=X" in m.columns:
                usdtry = m["USDTRY=X"].loc[px_te.index].to_numpy(np.float64)
        return asset_rets, usdtry
    except Exception as exc:                              # veri yoksa MC/reel-NAV atla
        print(f"[rigor] piyasa verisi yuklenemedi ({exc!r}); MC + reel-NAV atlandi")
        return None, None


def run():
    navs_path = RES / "navs_aligned.csv"
    if not navs_path.exists():
        print("[rigor] results/navs_aligned.csv yok — once `python main.py` calistir.")
        return None
    navs = pd.read_csv(navs_path, index_col=0, parse_dates=True)
    weights = {a: pd.read_csv(RES / f"weights_{a}.csv")
               for a in RL_AGENTS if (RES / f"weights_{a}.csv").exists()}
    asset_rets, usdtry = _load_market()

    out = compute_rigor(navs, weights, asset_rets, usdtry)

    out["rigor"].to_csv(RES / "rigor_metrics.csv")
    meta = {"PBO": out["pbo"]["pbo"], "PBO_n_combos": out["pbo"]["n_combos"],
            "n_trials": out["n_trials"], "sr_var": out["sr_var"]}
    if out["mc"]:
        meta["MC_best_agent"] = out["mc"]["best_agent"]
        for k, v in out["mc"]["block_bootstrap"].items():
            meta[f"MC_bb_{k}"] = v
        for k, v in out["mc"]["student_t"].items():
            meta[f"MC_st_{k}"] = v
        out["mc"]["terminal"].to_csv(RES / "rigor_mc_terminal.csv", index=False)
    pd.Series(meta).to_csv(RES / "rigor_summary.csv", header=False)
    if out["navs_real"] is not None:
        out["navs_real"].to_csv(RES / "navs_real.csv")

    print("=" * 60)
    print("RIGOR METRIKLERI (Deflated/Probabilistic Sharpe + reel-NAV):")
    cols = [c for c in ["sharpe_ann", "PSR", "DSR", "FinalNAV_nominal", "FinalNAV_real_usd"]
            if c in out["rigor"].columns]
    print(out["rigor"][cols].round(4).to_string())
    print(f"PBO (backtest-overfitting olasiligi) = {out['pbo']['pbo']:.3f} "
          f"(n_combos={out['pbo']['n_combos']}, n_trials={out['n_trials']})")
    if out["mc"]:
        s = out["mc"]["block_bootstrap"]
        print(f"Monte-Carlo stres (en iyi RL = {out['mc']['best_agent']}, blok bootstrap): "
              f"VaR={s['VaR']:.3f} CVaR={s['CVaR']:.3f} P(zarar)={s['prob_loss']:.2f} "
              f"P(>%20 dusus)={s['prob_loss_20pct']:.3f}")
    print("=" * 60)
    return out


if __name__ == "__main__":
    run()
