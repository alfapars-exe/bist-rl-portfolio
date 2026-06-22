"""Forecast feature degerlendirme — CNN-LSTM bir-adim getiri tahmincisinin
   gercek tahmin gucunu olcer (golden-guvenli, yan-etkisiz yeni script).

Soru: state'e eklenen 'forecast' feature'i naive baseline'lardan (persistence,
zero) gercekten daha iyi mi? Finansal getiri tahmini zordur — bu script
abartmadan, hizalama-sizinti-guvenli sekilde olcer.

Yaklasim train.py::prepare_data (satir 57-65) ile birebir ayni, sizinti-guvenli:
  1. px = download_bist(); px_tr, px_te = train_test_split(px)  (< 2022-01-01 / >=)
  2. build_forecast_feature(px, px_tr, **ForecastConfig, seed=SEED)  — YALNIZ train fit
  3. Test doneminde bir-adim-ileri tahmin; causal hizalama.

Hizalama (kritik, sizintisiz):
  - forecaster ciktisi pred_df[t] = t'de biten pencereden uretilen t+1 ongorusu.
  - ground-truth  gt[t]      = logret[t+1]
  - persistence   persist[t] = logret[t]            (naif: yarin = bugun)
  - zero          zero[t]    = 0                     (naif: yarin = 0)
  Maske: t test_index icinde, t >= window-1 (pencere dolu), t < T-1 (t+1 var).

Metrikler (her tahminci x [genel, 28-hisse ort.]): RMSE, MAE, yon dogrulugu (%),
bir-adim Pearson korelasyon.

CIKTI: results/forecast_eval.csv (3 satir) + stdout ozeti. results/ gitignore'lu.
core/env/agents/train/golden'a DOKUNULMAZ.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from data import download_bist, train_test_split          # noqa: E402
from config import SEED, ForecastConfig, DataConfig        # noqa: E402
from forecast.forecaster import build_forecast_feature, _logret  # noqa: E402

RES = BASE / "results"
RES.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# Metrik yardimcilari
# --------------------------------------------------------------------------- #
def _rmse(p: np.ndarray, g: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - g) ** 2)))


def _mae(p: np.ndarray, g: np.ndarray) -> float:
    return float(np.mean(np.abs(p - g)))


def _sign_acc(p: np.ndarray, g: np.ndarray) -> float:
    """Yon dogrulugu %: sign(pred) == sign(gt). 0'lar (hareketsiz gun) yon
    icermez -> hem pred hem gt 0-disi olan ornekler uzerinden hesaplanir."""
    mask = (np.sign(p) != 0) & (np.sign(g) != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(p[mask]) == np.sign(g[mask])) * 100.0)


def _pearson(p: np.ndarray, g: np.ndarray) -> float:
    """Bir-adim Pearson korelasyon (genel havuz). Sabit-tahmin (zero) -> std=0 -> nan."""
    if np.std(p) < 1e-12 or np.std(g) < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, g)[0, 1])


def _finite_mean(values: list[float]) -> float:
    values_arr = np.asarray(values, dtype=float)
    finite = values_arr[np.isfinite(values_arr)]
    return float(finite.mean()) if finite.size else float("nan")


def _metrics_pooled(p: np.ndarray, g: np.ndarray) -> dict:
    """Genel (havuzlanmis: tum (t,j) ornekleri tek vektorde) metrikler."""
    return dict(rmse=_rmse(p, g), mae=_mae(p, g),
                sign_acc=_sign_acc(p, g), corr=_pearson(p, g))


def _metrics_per_asset(P: np.ndarray, G: np.ndarray) -> dict:
    """28-hisse ortalamasi: her hisse icin metrik, sonra hisseler arasi nan-safe ort."""
    N = P.shape[1]
    rmse, mae, sgn, cor = [], [], [], []
    for j in range(N):
        pj, gj = P[:, j], G[:, j]
        rmse.append(_rmse(pj, gj))
        mae.append(_mae(pj, gj))
        sgn.append(_sign_acc(pj, gj))
        cor.append(_pearson(pj, gj))
    return dict(rmse=_finite_mean(rmse), mae=_finite_mean(mae),
                sign_acc=_finite_mean(sgn), corr=_finite_mean(cor))


# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 72)
    print("FORECAST FEATURE DEGERLENDIRME (CNN-LSTM bir-adim getiri tahmincisi)")
    print("=" * 72)

    # 1) Veri — train.py ile birebir ayni cagri
    px = download_bist()
    provenance = px.attrs.get("provenance", {})
    source = provenance.get("source", "unknown")
    px_tr, px_te = train_test_split(px)   # default split = DataConfig.train_end
    print(f"Veri kaynagi : {source.upper()} ({provenance.get('provider', 'unknown')})")
    if provenance.get("missing_tickers"):
        print(f"Eksik ticker  : {', '.join(provenance['missing_tickers'])}")
    if provenance.get("reason"):
        print(f"Fallback      : {provenance['reason']}")
    print(f"Train        : {px_tr.shape}  (< {DataConfig.train_end})")
    print(f"Test         : {px_te.shape}  (>= {DataConfig.train_end})")
    print(f"Hisse sayisi : {px.shape[1]}")
    print(f"Forecaster   : window={ForecastConfig.window} conv_ch={ForecastConfig.conv_ch} "
          f"hidden={ForecastConfig.hidden} epochs={ForecastConfig.epochs} "
          f"lr={ForecastConfig.lr} batch={ForecastConfig.batch} seed={SEED}")

    # 2) Forecaster — YALNIZ train'de fit (train.py satir 62-65 ile ayni imza)
    pred_df = build_forecast_feature(
        px, px_tr, window=ForecastConfig.window, conv_ch=ForecastConfig.conv_ch,
        hidden=ForecastConfig.hidden, epochs=ForecastConfig.epochs,
        lr=ForecastConfig.lr, batch=ForecastConfig.batch, seed=SEED)

    # 3) Hizalama (tum seri uzerinde hesapla, sonra test maskesi uygula)
    logret = _logret(px)                          # (T,N) ; logret[t] = log(p_t/p_{t-1})
    T, N = logret.shape
    window = ForecastConfig.window

    pred_full = pred_df.to_numpy(np.float32)       # pred[t] = t+1 ongorusu (causal)
    # ground-truth: gt[t] = logret[t+1]  (t+1'in gerceklesen getirisi)
    gt_full = np.zeros((T, N), dtype=np.float32)
    gt_full[:-1] = logret[1:]
    persist_full = logret.copy()                   # persist[t] = logret[t] (yarin=bugun)
    zero_full = np.zeros((T, N), dtype=np.float32)  # zero[t] = 0

    # Maske: t test doneminde, pencere dolu (t>=window-1), t+1 mevcut (t<T-1)
    pos = np.arange(T)
    test_pos = px.index.get_indexer(px_te.index)   # test satirlarinin global konumlari
    in_test = np.zeros(T, dtype=bool)
    in_test[test_pos] = True
    row_mask = in_test & (pos >= window - 1) & (pos < T - 1)
    rows = np.where(row_mask)[0]
    print(f"Degerlendirilen test satiri (gun): {len(rows)}  "
          f"-> ornek = {len(rows)} x {N} = {len(rows) * N}")

    # Maskeli matrisler (gun x hisse)
    Pred = pred_full[rows]
    GT = gt_full[rows]
    Per = persist_full[rows]
    Zer = zero_full[rows]

    # Havuzlanmis (genel) vektorler
    p_pool = Pred.ravel(); g_pool = GT.ravel()
    per_pool = Per.ravel(); zer_pool = Zer.ravel()

    estimators = {
        "forecast":    (_metrics_pooled(p_pool, g_pool),  _metrics_per_asset(Pred, GT)),
        "persistence": (_metrics_pooled(per_pool, g_pool), _metrics_per_asset(Per, GT)),
        "zero":        (_metrics_pooled(zer_pool, g_pool), _metrics_per_asset(Zer, GT)),
    }

    # 4) CSV
    rows_csv = []
    for name, (pooled, pa) in estimators.items():
        rows_csv.append(dict(
            estimator=name,
            rmse=pooled["rmse"], mae=pooled["mae"],
            sign_acc=pooled["sign_acc"], corr=pooled["corr"],
            rmse_per_asset=pa["rmse"], mae_per_asset=pa["mae"],
            sign_acc_per_asset=pa["sign_acc"], corr_per_asset=pa["corr"],
        ))
    df = pd.DataFrame(rows_csv)
    out_csv = RES / "forecast_eval.csv"
    df.to_csv(out_csv, index=False)

    # 5) Stdout ozeti
    print("\n" + "-" * 72)
    print("GENEL (havuzlanmis tum (t,j) ornekleri):")
    print(f"{'tahminci':<13}{'RMSE':>12}{'MAE':>12}{'yon%':>10}{'corr':>10}")
    for name, (pooled, _) in estimators.items():
        print(f"{name:<13}{pooled['rmse']:>12.6f}{pooled['mae']:>12.6f}"
              f"{pooled['sign_acc']:>10.2f}{pooled['corr']:>10.4f}")

    print("\n28-HISSE ORTALAMASI (hisse-bazi metrik, sonra ortalama):")
    print(f"{'tahminci':<13}{'RMSE':>12}{'MAE':>12}{'yon%':>10}{'corr':>10}")
    for name, (_, pa) in estimators.items():
        print(f"{name:<13}{pa['rmse']:>12.6f}{pa['mae']:>12.6f}"
              f"{pa['sign_acc']:>10.2f}{pa['corr']:>10.4f}")

    # 6) Dürüst hüküm
    f_pool = estimators["forecast"][0]
    p_pool_m = estimators["persistence"][0]
    z_pool = estimators["zero"][0]
    print("\n" + "=" * 72)
    print("HÜKÜM:")
    rmse_better = f_pool["rmse"] < p_pool_m["rmse"]
    mae_better = f_pool["mae"] < p_pool_m["mae"]
    # Yon dogrulugu ve korelasyon = gercek SINYAL gostergeleri (hata-buyuklugu degil).
    sign_better = f_pool["sign_acc"] > p_pool_m["sign_acc"]
    corr_better = (not np.isnan(f_pool["corr"])) and (
        np.isnan(p_pool_m["corr"]) or f_pool["corr"] > p_pool_m["corr"])
    drmse = (p_pool_m["rmse"] - f_pool["rmse"]) / max(p_pool_m["rmse"], 1e-12) * 100
    print(f"  forecast RMSE persistence'a gore: {drmse:+.2f}% "
          f"({'daha iyi' if rmse_better else 'daha kotu/esit'})")
    print(f"  forecast MAE  persistence'a gore: "
          f"{'daha iyi' if mae_better else 'daha kotu/esit'}")
    print(f"  forecast yon% persistence'a gore: {f_pool['sign_acc']:.2f} vs "
          f"{p_pool_m['sign_acc']:.2f} ({'daha iyi' if sign_better else 'daha kotu/esit'})")
    print(f"  forecast corr persistence'a gore: {f_pool['corr']:.4f} vs "
          f"{p_pool_m['corr']:.4f} ({'daha iyi' if corr_better else 'daha kotu/esit'})")

    # KRITIK NUANS: forecast'in dusuk RMSE/MAE'si genelde "neredeyse 0 tahmin"den
    # gelir (getiri ort.~0). Bunu zero baseline ile kiyasla: forecast ~ zero ise
    # hata-avantaji sinyalden DEGIL, sadece kuculten (shrinkage) gelir.
    rmse_vs_zero = (z_pool["rmse"] - f_pool["rmse"]) / max(z_pool["rmse"], 1e-12) * 100
    near_zero = abs(rmse_vs_zero) < 2.0   # forecast RMSE ~ zero RMSE (%2 icinde)
    print(f"  forecast RMSE zero(=0)'a gore : {rmse_vs_zero:+.2f}% "
          f"-> forecast {'~ ZERO (sinyal yok, sadece shrinkage)' if near_zero else 'zero-disi sinyal var'}")

    # Hata-buyuklugu metrikleri (RMSE/MAE) yaniltici cunku zero da persistence'i geciyor.
    # Gercek karar: SINYAL gostergeleri (yon% > %50 ve corr > persistence).
    signal_won = sum([sign_better, corr_better]) + (1 if f_pool["sign_acc"] > 50.0 else 0)
    if sign_better and corr_better and f_pool["sign_acc"] > 50.0:
        verdict = "persistence'i GERCEKTEN GECTI (sinyal var)"
    elif near_zero:
        verdict = ("persistence'i hata-buyuklugunde gecti AMA bu sadece shrinkage "
                   "(forecast~0); SINYAL yok -> yon%<persistence ve corr<persistence")
    else:
        verdict = "karisik; sinyal gostergeleri zayif"
    print(f"  -> SONUC: {verdict}")
    if source in {"mixed", "synthetic"}:
        print(f"  NOT: Veri kaynagi {source.upper()}; sonuc tamamen gercek veri olarak "
              "etiketlenemez.")
    print(f"\nYazildi: {out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
