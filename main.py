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


def step_data():
    print("=" * 70)
    print("[1/4] BIST 28 fiyatları hazırlanıyor ...")
    print("=" * 70)
    from data import download_bist
    from utils.features import add_features
    prices_path = RES / "bist30_prices.csv"
    px = download_bist()
    px.to_csv(prices_path)
    feats = add_features(px)
    for name, f in feats.items():
        f.to_csv(RES / f"feat_{name}.csv")
    print(f"  Kaydedildi: {prices_path}  (shape={px.shape})")


def step_train():
    print("=" * 70)
    print("[2/4] DQN + PPO + SAC + TD3 eğitimi ve backtest ...")
    print("=" * 70)
    import train
    train.run()


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


def step_walkforward():
    print("=" * 70)
    print("[WF] Walk-forward dogrulama (PPO, train donemi, fold-yerel olcekleme) ...")
    print("=" * 70)
    from data import download_bist, train_test_split
    from utils.features import add_features
    from core.walkforward import walk_forward
    from agents import PPOAgent
    from config import PPOConfig, SEED
    px = download_bist()
    px_tr, _ = train_test_split(px)
    feats_raw = add_features(px_tr)        # teknik feat (forecast haric -> fold-yerel leak-safe)

    def ppo_factory(sd, ad, seed):
        return PPOAgent(sd, ad, hidden=PPOConfig.hidden, lr_p=PPOConfig.lr_p,
                        lr_v=PPOConfig.lr_v, batch_size=PPOConfig.batch_size,
                        n_epochs=PPOConfig.n_epochs, seed=seed)

    rep = walk_forward(px_tr, feats_raw, ppo_factory, n_folds=3, n_iters=12, seed=SEED)
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
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_data:  step_data()
    if not args.skip_train: step_train()
    if not args.skip_rigor: step_rigor()       # plot'tan ÖNCE (F11-F13 rigor çıktısını okur)
    if not args.skip_plots: step_plots()
    if args.walkforward:    step_walkforward()

    print("=" * 70)
    print(f"BİTTİ.  Toplam süre: {time.time() - t0:.1f} s")
    print(f"  Sonuçlar : {RES}")
    print(f"  Figürler : {FIG}")
    print("=" * 70)


if __name__ == "__main__":
    main()
