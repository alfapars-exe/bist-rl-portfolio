"""Deflated / Probabilistic Sharpe Ratio + CSCV Backtest-Overfitting Olasılığı (PBO).

Kaynaklar: Bailey & López de Prado (2014) "The Deflated Sharpe Ratio" (SSRN 2460551);
Bailey, Borwein, López de Prado & Zhu (2017) "The Probability of Backtest Overfitting"
(SSRN 2326253). Sharpe girdileri GÖZLEM-BAŞINA (ör. günlük).

PARS referans ağacından (`Reinforcement Learning Final/generalization/deflated_sharpe.py`)
kanonik BIST ağacına port edildi. scipy.stats.norm zaten projede bir bağımlılık
(env/reward.py) → eski `np.math` fallback'ları kaldırıldı (numpy 2.0-güvenli).

GÖZLEMSEL / golden-güvenli: yalnız getiri dizileri üzerinde saf istatistik —
eğitim/değerlendirme/ödül sayısal yolunu DEĞİŞTİRMEZ (golden-master 1e-6 korunur).
"""
from __future__ import annotations

from itertools import combinations
from math import e

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(sr: float, n_obs: int, skew: float = 0.0,
                               kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """PSR: tahmin hatası + çarpıklık/basıklık altında P(gerçek SR > benchmark).

    sr ve sr_benchmark GÖZLEM-BAŞINA Sharpe oranlarıdır.
    """
    if n_obs < 2:
        return 0.5
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """n_trials bağımsız deneme altında H0'da beklenen MAKSİMUM (gözlem-başı) Sharpe.

    E[max SR] ≈ sqrt(Var_SR) · [ (1−γ)·Z⁻¹(1−1/N) + γ·Z⁻¹(1−1/(N·e)) ].
    """
    n_trials = max(int(n_trials), 1)
    if n_trials == 1 or sr_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(sr_variance) * ((1 - g) * z1 + g * z2))


def deflated_sharpe_ratio(sr: float, n_obs: int, n_trials: int, sr_variance: float,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """DSR = beklenen-maksimum-Sharpe benchmark'ına karşı değerlendirilen PSR.

    DSR > 0.95 → gözlenen Sharpe'ın çoklu-deneme (multiple-testing) tesadüfü olması
    olası değil. Tüm Sharpe büyüklükleri gözlem-başınadır.
    """
    sr_star = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(sr, n_obs, skew, kurt, sr_benchmark=sr_star)


def cscv_pbo(perf_matrix: np.ndarray, n_splits: int = 10) -> dict:
    """Kombinatoryal-Simetrik Çapraz-Doğrulama ile Backtest-Overfitting Olasılığı.

    perf_matrix: (T_gözlem, N_config) her aday config için periyot-başı getiri.
    Zaman ``n_splits`` bloğa bölünür; her dengeli train/test bölüşümünde IS-en-iyi
    config seçilir ve OOS sırası → logit kaydedilir. PBO = IS-en-iyi config'in
    OOS-medyan-altı olduğu bölüşüm oranı. {pbo, n_combos, mean_logit} döner.
    """
    R = np.asarray(perf_matrix, dtype=float)
    T, N = R.shape
    if N < 2 or T < n_splits * 2:
        return {"pbo": float("nan"), "n_combos": 0, "mean_logit": float("nan")}
    S = n_splits if n_splits % 2 == 0 else n_splits - 1
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for train_idx in combinations(range(S), S // 2):
        tr = np.concatenate([blocks[i] for i in train_idx])
        te = np.concatenate([blocks[i] for i in range(S) if i not in train_idx])
        is_sr = R[tr].mean(0) / (R[tr].std(0) + 1e-12)
        oos_sr = R[te].mean(0) / (R[te].std(0) + 1e-12)
        n_star = int(np.argmax(is_sr))
        order = np.argsort(oos_sr)                      # 1=en kötü .. N=en iyi OOS
        rank = int(np.where(order == n_star)[0][0]) + 1
        w = min(max(rank / (N + 1), 1e-6), 1 - 1e-6)
        logits.append(np.log(w / (1 - w)))
    logits = np.asarray(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combos": int(len(logits)),
            "mean_logit": float(logits.mean())}
