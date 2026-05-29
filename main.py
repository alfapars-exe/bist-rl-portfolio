"""Tek komutla tüm deneyi çalıştırır.

Akış:
  1) BIST 30 fiyatlarını yfinance ile indir (veya varsa CSV'den yükle)
  2) DQN + PPO + SAC eğit, test setinde tüm stratejileri backtest et
  3) 9 figürü (f1..f9) figures/ klasörüne kaydet

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
    print("[1/3] BIST 28 fiyatları hazırlanıyor ...")
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
    print("[2/3] DQN + PPO + SAC eğitimi ve backtest ...")
    print("=" * 70)
    import train
    train.run()


def step_plots():
    print("=" * 70)
    print("[3/3] 9 figür üretiliyor ...")
    print("=" * 70)
    import plots
    plots.run()
    print(f"  Figürler: {FIG}")


def main():
    ap = argparse.ArgumentParser(description="BIST 30 RL Portföy Yönetimi — tam akış")
    ap.add_argument("--skip-data",  action="store_true", help="Veri indirme adımını atla")
    ap.add_argument("--skip-train", action="store_true", help="Eğitim + backtest adımını atla")
    ap.add_argument("--skip-plots", action="store_true", help="Çizim adımını atla")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_data:  step_data()
    if not args.skip_train: step_train()
    if not args.skip_plots: step_plots()

    print("=" * 70)
    print(f"BİTTİ.  Toplam süre: {time.time() - t0:.1f} s")
    print(f"  Sonuçlar : {RES}")
    print(f"  Figürler : {FIG}")
    print("=" * 70)


if __name__ == "__main__":
    main()
