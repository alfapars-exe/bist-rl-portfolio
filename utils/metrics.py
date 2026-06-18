"""Backtest metrikleri — CAGR, Sharpe, Sortino, MaxDD, Calmar, Turnover + başarı metriği.

Ayrica PDF §9.7 egitim teshisleri (moving_average + training_diagnostics): episode
getirisi hareketli ortalamasi, basari orani, ortalama adim, ceza/iflas sayisi. Bunlar
GOZLEMSEL'dir — egitim/odul/eval sayisal yolunu degistirmez (golden-master korunur).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(nav: np.ndarray) -> float:
    years = len(nav) / TRADING_DAYS
    return float(nav[-1] ** (1 / max(years, 1e-6)) - 1)


def sharpe(rets: np.ndarray, rf: float = 0.0) -> float:
    mu = np.mean(rets) - rf / TRADING_DAYS
    sd = np.std(rets) + 1e-9
    return float(np.sqrt(TRADING_DAYS) * mu / sd)


def sortino(rets: np.ndarray, rf: float = 0.0) -> float:
    """Sortino orani — payda kanonik downside deviation (Sortino/Price):
    sqrt(mean(min(r - hedef, 0)^2)), TUM gozlemler uzerinden (empyrical uyumu).

    Onceki surum np.std(negatif altkume) kullaniyordu — iki hata: (1) sapma
    negatif altkumenin KENDI ortalamasina goreydi (hedefe degil), (2) yalniz
    negatif gun sayisina bolunuyordu. Uniform kayiplarda payda ~0 olup orani
    sisiriyor, hic negatif getiri yokken NaN donuyordu (kesif bulgusu, HIGH).
    """
    target = rf / TRADING_DAYS
    mu = np.mean(rets) - target
    downside_dev = float(np.sqrt(np.mean(np.minimum(rets - target, 0.0) ** 2)))
    if downside_dev < 1e-12:
        # Hic asagi-yonlu sapma yok: pozitif ortalamada sonsuz, aksi halde 0.
        return float("inf") if mu > 0 else 0.0
    return float(np.sqrt(TRADING_DAYS) * mu / downside_dev)


def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return float(dd.min())


def calmar(nav: np.ndarray) -> float:
    mdd = abs(max_drawdown(nav))
    return cagr(nav) / (mdd + 1e-9)


def turnover(weights: np.ndarray) -> float:
    """Günlük ‖Δw‖₁ ortalaması."""
    d = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    return float(d.mean()) if len(d) else 0.0


def summary(nav: np.ndarray, rets: np.ndarray, weights=None) -> dict:
    out = dict(
        CAGR=cagr(nav),
        Sharpe=sharpe(rets),
        Sortino=sortino(rets),
        MaxDD=max_drawdown(nav),
        Calmar=calmar(nav),
        Volatility=float(np.std(rets) * np.sqrt(TRADING_DAYS)),
        FinalNAV=float(nav[-1]),
    )
    if weights is not None:
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
