"""Ek akademik baseline'lar — test donemi metriklerini hesaplayip metrics.csv'ye EK satir yazar.

Akademik karsilastirma setini genisletir (D4 raporu icin). Eklenen stratejiler
(hepsi DETERMINISTIK, RNG YOK):
  RiskParity      — esit risk katkisi (ERC), iteratif sabit-nokta
  InverseVol      — 1/sigma agirlik (yuvarlanan vol)
  MinVariance     — saf min x'Sigma x (getiri tahmini yok, simplex kisitli)
  Momentum        — kesitsel top-k momentum (vade penceresine gore)
  CashRiskFree    — nakit/risk-free taban benchmark

GOLDEN-GUVENLI:
  * RL ajanlari YENIDEN EGITILMEZ (gereksiz + RNG riski) — yalniz test FIYATLARI
    yuklenir (train.py ile birebir ayni: download_bist cache + train_test_split).
  * Mevcut metrics.csv satirlari (DQN/PPO/SAC/TD3/BuyHold/EqualWeight/MeanVar)
    DEGISTIRILMEZ; yeni baseline'lar EK SATIR olarak append edilir.
  * Yeni baseline'lar np.random kullanmaz -> golden RNG cagri sirasi etkilenmez.

Calistir: `.venv\\Scripts\\python.exe scripts\\extra_baselines.py`
Idempotent: tekrar calistirilirsa ayni satirlar uzerine yazar (cift satir uretmez).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULTS                                  # noqa: E402
from data import download_bist, train_test_split            # noqa: E402
from utils.baselines import (cash_riskfree,                 # noqa: E402
                             inverse_volatility, min_variance,
                             momentum, risk_parity)
from utils.metrics import summary                           # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
METRICS_CSV = RES / "metrics.csv"
PRICES_CSV = RES / "bist30_prices.csv"   # train.py'nin step_data'da yazdigi kanonik fiyat dosyasi

# Vade preset'i — kanonik backtest 'medium' kullanir (train.py ile tutarli).
# Momentum/min-var pencereleri ile rebalans frekansi buradan okunur (tek kaynak).
REBALANCE = 1
MOM_WINDOW = DEFAULTS.mom_window
MINVOL_WINDOW = DEFAULTS.minvol_window

# Yeni baseline'larin metrics.csv satir adlari (mevcut satirlarla CAKISMAZ).
NEW_ROWS = ["RiskParity", "InverseVol", "MinVariance", "Momentum", "CashRiskFree"]


def compute_extra_baselines(px_te: pd.DataFrame) -> dict:
    """Test fiyatlarindan 5 ek baseline'in backtest sozlugunu uretir (determinist)."""
    return {
        # ERC: kovaryans gerektigi icin daha uzun yuvarlanan pencere (minvol_window).
        "RiskParity": risk_parity(px_te, lookback=MINVOL_WINDOW, rebalance=REBALANCE),
        # 1/sigma: tek-degisken vol -> kisa pencere yeterli (mom_window).
        "InverseVol": inverse_volatility(px_te, lookback=MINVOL_WINDOW, rebalance=REBALANCE),
        # Saf min-var: kovaryans tahmini -> minvol_window penceresi.
        "MinVariance": min_variance(px_te, lookback=MINVOL_WINDOW, rebalance=REBALANCE),
        # Kesitsel momentum: vade momentum penceresi (kisa/orta/uzun -> 5/20/60).
        "Momentum": momentum(px_te, lookback=MOM_WINDOW, rebalance=REBALANCE, top_k=5),
        # Nakit taban: GERCEK risksiz faiz (config cash_annual_rate ~%40) -> RL'in
        # nakit varligiyla SIMETRIK (env de faiz kazanir); NAV ~ (1+rf)^t.
        "CashRiskFree": cash_riskfree(px_te),
    }


def _load_test_prices() -> pd.DataFrame:
    """Test donemi fiyatlarini train.py ile BIREBIR ayni evrenden uretir.

    Oncelik download_bist() (train.py'nin kullandigi yol). Ancak bu ortamda
    parquet cache budanmis bir tarih dilimi tutabilir (env-lock) -> test split
    bos cikar. Bu durumda train.py'nin step_data'da DISKE yazdigi kanonik
    `results/bist30_prices.csv` (tam 2015-2024) fallback olarak okunur; bu dosya
    golden metrics.csv'yi ureten ayni fiyat matrisidir, dolayisiyla test
    fiyatlari kanonik ile birebir ayni olur."""
    px = download_bist()
    _, px_te = train_test_split(px)
    if len(px_te) > 0:
        return px_te
    if PRICES_CSV.exists():
        print(f"[extra_baselines] parquet cache test dilimi BOS; kanonik "
              f"{PRICES_CSV.name} fallback'i kullaniliyor (tam 2015-2024).")
        px_full = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
        _, px_te = train_test_split(px_full)
        if len(px_te) > 0:
            return px_te
    raise RuntimeError(
        "Test donemi fiyati uretilemedi (cache budanmis ve bist30_prices.csv yok). "
        "Once `python main.py --skip-train --skip-plots --skip-rigor` (veya tam main.py) "
        "calistirip results/bist30_prices.csv'yi olustur.")


def run() -> pd.DataFrame:
    if not METRICS_CSV.exists():
        raise FileNotFoundError(
            f"{METRICS_CSV} yok — once `python main.py` (veya train.run()) calistir.")

    # 1) Test fiyatlari — train.py ile BIREBIR ayni evren (kanonik fiyat matrisi).
    px_te = _load_test_prices()
    print(f"[extra_baselines] Test fiyatlari: {px_te.shape}  "
          f"{px_te.index[0].date()} -> {px_te.index[-1].date()}  "
          f"(step_days=1, her adimda rebalans)")

    # 2) Yeni baseline metriklerini hesapla — ANA STRATEJILERLE AYNI hizali
    #    pencerede (C1): navs_aligned.csv ortak min_len + yeniden-tabanlama (NAV[0]=1).
    navs_aligned_csv = RES / "navs_aligned.csv"
    min_len = (len(pd.read_csv(navs_aligned_csv, index_col=0))
               if navs_aligned_csv.exists() else len(px_te))
    bts = compute_extra_baselines(px_te)
    new_metrics = {}
    for name, d in bts.items():
        nav = np.asarray(d["nav"], dtype=float)[-min_len:]
        nav = nav / nav[0]                      # ortak pencere baslangicina tabanla
        rets = np.concatenate([[0.0], np.diff(nav) / nav[:-1]])
        w = d.get("weights")
        if w is not None:
            w = np.asarray(w)[-min_len:]
        dates = pd.DatetimeIndex(d.get("dates", px_te.index))[-min_len:]
        turn = np.asarray(d.get("turnover", []), dtype=float)[-min_len:]
        m = summary(nav, rets, w, dates=dates, turnover_values=turn)
        new_metrics[name] = m
        print(f"[TEST] {name:<12}  CAGR={m['CAGR']:+.2%}  Sharpe={m['Sharpe']:+.2f}  "
              f"Sortino={m['Sortino']:+.2f}  MaxDD={m['MaxDD']:+.2%}  "
              f"Calmar={m['Calmar']:+.2f}  Vol={m['Volatility']:.2%}  "
              f"Final={m['FinalNAV']:.3f}  Turn={m['Turnover']:.4f}")

    # 3) metrics.csv'ye EK satir yaz — mevcut satirlari KORU (kolon sirasini izle).
    existing = pd.read_csv(METRICS_CSV, index_col=0)
    new_df = pd.DataFrame(new_metrics).T.reindex(columns=existing.columns)
    # Eski-baseline satirlarini koru, yeni satirlari (varsa) guncelle -> idempotent.
    keep = existing.drop(index=[r for r in NEW_ROWS if r in existing.index], errors="ignore")
    out = pd.concat([keep, new_df], axis=0)
    out.to_csv(METRICS_CSV)
    print(f"[extra_baselines] metrics.csv guncellendi: +{len(new_df)} satir "
          f"(toplam {len(out)} strateji). Mevcut satirlar korundu.")
    return out


if __name__ == "__main__":
    df = run()
    print(df.round(4))
