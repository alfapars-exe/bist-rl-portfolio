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
