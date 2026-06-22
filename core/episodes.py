"""Gurultu-artirimli (noise-augmented) coklu-episode degerlendirme.

Kullanici istegi (kisa vade): her adim gunluk, episode TUM test araligi boyunca
kosar; orijinal hisse serilerinden gurultu eklenmis YENI episode'lar uretip
SIRAYLA hepsi degerlendirilir. Amac, egitilmis ajanin tek bir tarihsel patikaya
EZBER olup olmadigini sinamak (hocanin "gurultu ile anti-ezber" sarti) — robustluk.

Iki katman saglar:
  * make_noisy_prices(): orijinal fiyat serisinden gurultu eklenmis YENI seri
    (log-getirilere bagimsiz Gauss gurultusu -> yeniden kumulatif fiyat).
  * evaluate_noise_episodes(): make_env(episode, noise_std, seed) callback'i ile
    her episode icin bir eval ortami kurar, ajani TAM ARALIK kosturur, episode
    basina NAV + ozet metrikleri toplar. Episode 0 = orijinal (gurultusuz) referans.

GOLDEN-GUVENLI: yeni modul; mevcut eval/golden yollarina dokunmaz. act_eval ile
ajan-agnostiktir (DQN/PPO/SAC/TD3).
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd


def make_noisy_prices(prices: pd.DataFrame, noise_std: float,
                      seed: int) -> pd.DataFrame:
    """Orijinal fiyat serisinden gurultu eklenmis YENI bir seri uretir.

    Gunluk log-getirilere bagimsiz UNIFORM gurultusu ([-noise_std, +noise_std])
    ekler ve fiyati yeniden kumulatifler. Baslangic fiyati (ilk satir) korunur;
    boylece ayni baslangic, farkli yol (gercek anti-ezber). noise_std=0 ->
    orijinalin birebir kopyasi (identity; KORUNUR).

    Gauss yerine Uniform tercih edilmesinin sebebi: outlier-gurultu uretmez,
    deterministik aralik garantisi saglar (|eps| <= noise_std) — ajan FARKLI
    fiyat yolu gorur ama cok buyuk anlamsiz sapma olusturmaz.

    Args:
        prices: (T, N) ayarli kapanis matrisi (orijinal hisseler).
        noise_std: gunluk log-getiriye eklenen uniform gurultu genisligi
                   (aralik: [-noise_std, +noise_std]). Ornek: 0.01.
        seed: tekrar-uretilebilir gurultu icin tohum (episode basina farkli ver).
    """
    px = prices.to_numpy(dtype=np.float64)
    px = np.where(np.isfinite(px) & (px > 0), px, np.nan)
    # NaN bosluklari ileri/geri doldur (gurultu uretimi sonlu kalsin)
    df = pd.DataFrame(px).ffill().bfill()
    px = df.to_numpy(dtype=np.float64)
    logret = np.diff(np.log(np.maximum(px, 1e-9)), axis=0)
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        # UNIFORM gurultu: [-noise_std, +noise_std] (Gauss yerine; anti-ezber)
        logret = logret + rng.uniform(-noise_std, noise_std, size=logret.shape)
    new = np.empty_like(px)
    new[0] = px[0]
    new[1:] = px[0] * np.exp(np.cumsum(logret, axis=0))
    return pd.DataFrame(new, index=prices.index, columns=prices.columns)


def episode_metrics(nav: np.ndarray, rets: np.ndarray,
                    periods_per_year: int = 252) -> Dict[str, float]:
    """Bir episode NAV/getiri serisinden ozet metrikler (sonlu-guvenli)."""
    nav = np.asarray(nav, dtype=float)
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    final_nav = float(nav[-1]) if nav.size else float("nan")
    total_return = final_nav - 1.0
    # Max drawdown
    if nav.size:
        peak = np.maximum.accumulate(nav)
        dd = (peak - nav) / np.maximum(peak, 1e-9)
        max_dd = float(np.max(dd))
    else:
        max_dd = float("nan")
    # Yillik Sharpe (risksiz=0 varsayimi)
    if rets.size > 1 and rets.std() > 1e-12:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0
    return dict(final_nav=final_nav, total_return=total_return,
                max_drawdown=max_dd, sharpe=sharpe)


def evaluate_noise_episodes(make_env: Callable[..., object], agent, *,
                            n_episodes: int = 5, noise_std: float = 0.01,
                            seed: int = 42,
                            include_original: bool = True) -> List[Dict]:
    """Egitilmis ajani sirayla birden cok TAM-ARALIK episode uzerinde kosturur.

    make_env(episode=i, noise_std=nstd, seed=seed+i) -> eval ortami dondurmeli
    (random_start=False; max_steps test araligini kapsayacak kadar buyuk). Boylece
    "gurultu" iki sekilde uygulanabilir: (a) UI -> ayni px_te + force_price_noise
    (getiri-seviyesi), (b) CLI -> make_noisy_prices ile perturbe fiyat + ozellik
    yeniden hesabi. evaluate_noise_episodes her iki stratejiye de agnostiktir.

    Returns: episode basina dict listesi {episode, noise_std, steps, nav, dates,
    final_nav, total_return, max_drawdown, sharpe}.
    """
    results: List[Dict] = []
    for i in range(int(n_episodes)):
        nstd = 0.0 if (i == 0 and include_original) else float(noise_std)
        env = make_env(episode=i, noise_std=nstd, seed=int(seed) + i)
        s, _ = env.reset()
        done = trunc = False
        while not (done or trunc):
            s, _r, done, trunc, _info = env.step(agent.act_eval(s))
        nav = np.asarray(env.nav_history, dtype=float)
        rets = np.asarray(getattr(env, "ret_history", []), dtype=float)
        offset = getattr(env, "window", 0)
        dates = list(env.dates[offset: offset + len(nav)])
        results.append(dict(
            episode=i, noise_std=nstd, steps=len(nav) - 1,
            nav=nav, dates=dates, **episode_metrics(nav, rets),
        ))
    return results


def summarize_episodes(results: List[Dict]) -> Dict[str, float]:
    """Episode'lar arasi ozet (ortalama/std/min/max final NAV + getiri)."""
    fn = np.array([r["final_nav"] for r in results], dtype=float)
    tr = np.array([r["total_return"] for r in results], dtype=float)
    dd = np.array([r["max_drawdown"] for r in results], dtype=float)
    return dict(
        n_episodes=len(results),
        final_nav_mean=float(np.nanmean(fn)), final_nav_std=float(np.nanstd(fn)),
        final_nav_min=float(np.nanmin(fn)), final_nav_max=float(np.nanmax(fn)),
        total_return_mean=float(np.nanmean(tr)),
        max_drawdown_mean=float(np.nanmean(dd)),
        prob_loss=float(np.mean(tr < 0.0)),
    )
