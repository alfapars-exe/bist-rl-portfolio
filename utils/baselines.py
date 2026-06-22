"""Klasik baseline stratejiler — karşılaştırma için.

İŞLEM MALİYETİ SİMETRİSİ (C2):
RL ortamı her rebalansta `eta·||Δw||₁` işlem maliyeti düşer (env/reward.py →
tx_cost = eta_t * delta_w_l1). Baseline'lar maliyetsiz olursa ASİMETRİK bir
karşılaştırma çıkar (baseline yapay olarak avantajlı). Bu yüzden tüm
YENİDEN-DENGELEYEN baseline'lar (equal_weight, mean_variance, risk_parity,
inverse_volatility, min_variance, momentum) RL ile SİMETRİK işlem maliyeti
düşer: her rebalans gününde günlük net getiri = port_r − eta·||w_yeni−w_eski||₁.

eta KAYNAĞI: RL'in kanonik ortamıyla AYNI base oran —
`config.HORIZON_PRESETS["medium"]["eta"]` (kanonik vade = medium, eta_base=0.0010).
RL adaptif ölçekleme (turnover_ewma/turnover_target) uygular; baseline'lar
deterministik kalsın diye SABIT base eta kullanır (RNG yok, golden-uyumlu).
Fonksiyon imzaları geriye-uyumlu: eta opsiyonel, default = medium preset eta.

buy_and_hold_index rebalans yapmaz → maliyet 0 (değişmez).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HORIZON_PRESETS, EnvConfig, cash_daily_rate

# RL kanonik ortamıyla aynı base işlem-maliyeti oranı (vade = medium).
DEFAULT_ETA: float = float(HORIZON_PRESETS["medium"]["eta"])


def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = float(w.sum())
    return w / total if total > 1e-12 else np.full(len(w), 1.0 / len(w))


def _turnover_from_holdings(target: np.ndarray, held: np.ndarray) -> float:
    """L1 turnover including the implicit cash slot."""
    target = _normalize(target)
    held = np.asarray(held, dtype=float)
    return float(np.abs(target - held).sum() + abs((1.0 - target.sum()) - (1.0 - held.sum())))


def _simulate(prices: pd.DataFrame, target_fn, *, eta: float,
              rebalance_fn) -> dict:
    """Self-financing long-only simulator shared by all risky baselines."""
    r = prices.pct_change().fillna(0.0).to_numpy(dtype=float)
    T, N = r.shape
    nav = np.empty(T, dtype=float)
    rets = np.empty(T, dtype=float)
    weights = np.empty((T, N), dtype=float)
    target_hist = np.empty((T, N), dtype=float)
    turnover = np.zeros(T, dtype=float)
    held = np.zeros(N, dtype=float)  # all cash before the first allocation
    wealth = 1.0
    pending_initial_cost = 0.0
    for t in range(T):
        if rebalance_fn(t):
            target = _normalize(target_fn(t, r, held.copy()))
            turn = _turnover_from_holdings(target, held)
        else:
            target = held.copy()
            turn = 0.0
        if t == 0:
            # Report the common pre-trade starting wealth. Initial allocation
            # cost is realized together with the first investable period.
            pending_initial_cost = float(eta) * turn
            gross = net = 0.0
        else:
            gross = float(target @ r[t])
            net = gross - float(eta) * turn - pending_initial_cost
            pending_initial_cost = 0.0
            wealth = max(0.0, wealth * (1.0 + net))
        grown = target * (1.0 + (r[t] if t > 0 else 0.0))
        held = _normalize(grown) if float(grown.sum()) > 1e-12 else target
        nav[t], rets[t], weights[t], target_hist[t], turnover[t] = \
            wealth, net, held, target, turn
    return {"nav": nav, "rets": rets, "weights": weights,
            "target_weights": target_hist, "turnover": turnover,
            "dates": list(prices.index)}


def equal_weight(prices: pd.DataFrame, eta: float = DEFAULT_ETA) -> dict:
    """Günlük rebalansla eşit ağırlık — RL ile simetrik işlem maliyeti düşülür.

    Her gün eşit-ağırlığa geri dönülür (drift düzeltme). Önceki günkü fiyat
    hareketiyle kayan ağırlıklar (w_drift) tekrar 1/N'e çekilir; bu rebalansın
    ||Δw||₁'i kadar `eta` maliyeti net getiriden düşülür. Drift küçük olduğundan
    maliyet de küçüktür ama 0 değildir (RL ile simetri)."""
    n = prices.shape[1]
    target = np.full(n, 1.0 / n)
    return _simulate(prices, lambda t, r, w: target, eta=eta,
                     rebalance_fn=lambda t: True)


def buy_and_hold_index(prices: pd.DataFrame, eta: float = DEFAULT_ETA) -> dict:
    """Eşit ağırlık alıp tut; yalnız ilk alım işlem maliyeti doğurur."""
    n = prices.shape[1]
    target = np.full(n, 1.0 / n)
    return _simulate(prices, lambda t, r, w: target, eta=eta,
                     rebalance_fn=lambda t: t == 0)


def mean_variance(prices: pd.DataFrame, lookback: int = 120,
                  rebalance: int = 20, risk_aversion: float = 5.0,
                  eta: float = DEFAULT_ETA) -> dict:
    """Kısıtlı long-only Markowitz; yuvarlanan pencere + periyodik rebalans.

    RL ile simetrik: her rebalans gününde ||w_yeni−w_eski||₁ kadar `eta`
    işlem maliyeti net getiriden düşülür."""
    from scipy.optimize import minimize

    n = prices.shape[1]
    equal = np.full(n, 1.0 / n)

    def target_fn(t, r, held):
        if t < lookback:
            return equal
        hist = r[t - lookback:t]
        mu = hist.mean(axis=0)
        cov = np.cov(hist.T) + 1e-5 * np.eye(n)

        def obj(w_, mu=mu, cov=cov, ra=risk_aversion):
            return -(w_ @ mu) + 0.5 * ra * w_ @ cov @ w_

        res = minimize(obj, equal, bounds=[(0, 0.2)] * n,
                       constraints=({"type": "eq", "fun": lambda w_: w_.sum() - 1}))
        return res.x if res.success and np.isfinite(res.x).all() else equal

    return _simulate(
        prices, target_fn, eta=eta,
        rebalance_fn=lambda t: t == 0 or (t >= lookback and (t - lookback) % rebalance == 0),
    )


# =====================================================================
# Ek akademik baseline'lar — hepsi DETERMINISTIK (RNG YOK), long-only,
# tam-yatirim (sum(w)=1). PARS sibling strategies.py'den odunc alindi ve
# kod/ API'sine ({"nav", "rets", "weights"} numpy) uyarlandi. Ortak
# simulasyon cekirdegi _rolling_backtest: yuvarlanan pencere + periyodik
# rebalans, ilk lookback gunu esit-agirlik isinma (warm-up).
# =====================================================================
def _rolling_backtest(prices: pd.DataFrame, weight_fn, lookback: int,
                      rebalance: int, eta: float = DEFAULT_ETA) -> dict:
    """Determinist yuvarlanan-pencere backtest cekirdegi (RL-simetrik tx-cost).

    weight_fn(hist_returns:(L,N), w_prev:(N,)) -> w_new:(N,) dondurur.
    Donen agirliklar long-only normalize edilir (clip>=0, toplam=1).
    Her rebalans gununde RL ile SIMETRIK islem maliyeti dusulur:
        net_r = port_r - eta * ||w_yeni - w_eski||_1
    eta = config.HORIZON_PRESETS['medium']['eta'] (RL kanonik base oran).
    Hicbir np.random cagrisi yok -> golden RNG sirasi etkilenmez (determinist)."""
    n = prices.shape[1]
    equal = np.full(n, 1.0 / n)

    def target_fn(t, r, held):
        if t < lookback:
            return equal
        candidate = np.asarray(weight_fn(r[t - lookback:t], held), dtype=float)
        return candidate if candidate.shape == (n,) and np.isfinite(candidate).all() else equal

    return _simulate(
        prices, target_fn, eta=eta,
        rebalance_fn=lambda t: t == 0 or (t >= lookback and (t - lookback) % rebalance == 0),
    )


def inverse_volatility(prices: pd.DataFrame, lookback: int = 60,
                       rebalance: int = 20, eta: float = DEFAULT_ETA) -> dict:
    """Ters-volatilite (1/sigma) agirligi — yuvarlanan vol uzerinden.

    Mantik: w_i ∝ 1/sigma_i (sigma_i = pencere getiri std'i), normalize edilir.
    Riske gore dengeleme'nin (risk parity) kovaryanssiz, naif halidir: yalniz
    kosegen (varyans) bilgisi kullanilir, korelasyon yok sayilir. Dusuk-vol
    hisseye daha cok agirlik -> portfoy vol'unu duzler. 'Volatility tilt'
    olarak bilinen iyi-belgelenmis defensif bir anomalidir (Asness ve dig.)."""
    def f(hist, w):
        iv = 1.0 / (hist.std(axis=0) + 1e-9)
        return iv / iv.sum()
    return _rolling_backtest(prices, f, lookback, rebalance, eta)


def risk_parity(prices: pd.DataFrame, lookback: int = 120,
                rebalance: int = 20, n_iter: int = 100,
                eta: float = DEFAULT_ETA) -> dict:
    """Esit risk katkisi (ERC / risk parity) — iteratif sabit-nokta.

    Hedef: her varligin TOPLAM portfoy riskine katkisi esit olsun:
        RC_i = w_i (Sigma w)_i ,  hedef RC_i = sigma_p^2 / N  (tum i icin esit).
    Cozum, marjinal risk katkisi MRC = Sigma w ile sabit-nokta iterasyonu:
        w_i <- 1 / MRC_i ,  ardindan normalize (Spinu/Maillard tarzi). Yakinsayinca
    w_i * MRC_i sabittir -> esit risk katkisi. Inverse-vol'un aksine korelasyonu
    (tam Sigma) hesaba katar; konsantrasyonu cesitlendirip kuyruk riskini azaltir.
    Tamamen determinist (sabit baslangic w=1/N, np.random yok)."""
    def f(hist, w):
        cov = np.cov(hist.T) + 1e-6 * np.eye(hist.shape[1])
        n = cov.shape[0]
        wv = np.ones(n) / n
        for _ in range(n_iter):
            mrc = cov @ wv
            wv = 1.0 / np.maximum(mrc, 1e-8)
            wv /= wv.sum()
        return wv
    return _rolling_backtest(prices, f, lookback, rebalance, eta)


def min_variance(prices: pd.DataFrame, lookback: int = 120,
                 rebalance: int = 20, max_weight: float = 0.4,
                 eta: float = DEFAULT_ETA) -> dict:
    """Saf minimum-varyans portfoyu — getiri tahmini KULLANMAZ.

    Cozulen problem:  min_w  w' Sigma w   s.t.  sum(w)=1, 0<=w_i<=max_weight.
    mean_variance'tan farki: beklenen getiri (mu) terimi YOK -> yalniz risk
    minimize edilir. Ortalama getiri tahmini en gurultulu girdidir (mu'nun
    ornekleme hatasi devasadir); onu atip yalniz kovaryansa guvenmek genelde
    daha kararli, dusuk-vol portfoy verir (DeMiguel ve dig. bulgusu: min-var
    cogu zaman tahmini-getiri optimizasyonunu out-of-sample yener). SLSQP
    determinist baslar (w0=1/N), RNG yok."""
    from scipy.optimize import minimize as _min
    def f(hist, w):
        n = hist.shape[1]
        cov = np.cov(hist.T) + 1e-5 * np.eye(n)
        res = _min(lambda x: x @ cov @ x, np.ones(n) / n,
                   bounds=[(0, max_weight)] * n,
                   constraints=({"type": "eq", "fun": lambda x: x.sum() - 1}))
        return res.x if res.success and np.isfinite(res.x).all() else np.ones(n) / n
    return _rolling_backtest(prices, f, lookback, rebalance, eta)


def momentum(prices: pd.DataFrame, lookback: int = 60, rebalance: int = 20,
             top_k: int = 5, eta: float = DEFAULT_ETA) -> dict:
    """Kesitsel momentum — pencere getirisi en yuksek top_k hisseye esit agirlik.

    Mantik: lookback penceresinde kumulatif getiri  cum_i = prod(1+r) - 1
    hesaplanir; en yuksek top_k hisse secilip esit (1/top_k) agirlik verilir.
    Klasik 'kazananlari al' (Jegadeesh & Titman 1993) etkisi: gecmis galipler
    kisa-orta vadede ortalama ustu getirmeye egilimlidir. lookback vade
    penceresine baglanir (kisa=5, orta=20, uzun=60 gun); secim determinist
    (np.argsort kararli) -> RNG yok."""
    k = max(1, int(top_k))
    def f(hist, w):
        n = hist.shape[1]
        cum = (1.0 + hist).prod(axis=0) - 1.0
        idx = np.argsort(cum)[-k:]          # en yuksek k (kararli siralama)
        out = np.zeros(n)
        out[idx] = 1.0 / len(idx)
        return out
    return _rolling_backtest(prices, f, lookback, rebalance, eta)


def cash_riskfree(prices: pd.DataFrame,
                  daily_rf: float | None = None) -> dict:
    """Nakit / risk-free benchmark — tum sermaye nakitte, sabit gunluk getiri.

    En muhafazakar referans: piyasaya HIC maruz kalmadan elde edilen getiri.
    daily_rf default'u RL ortamiyla AYNI gercek risksiz faizden gelir:
        cash_daily_rate(EnvConfig.cash_annual_rate)  (~%40 yillik -> ~0.001336/gun)
    -> NAV = (1+rf)^t ~ 1.40x/yil (252 gun). Boylece nakit baseline'i de RL'in
    nakit varligiyla SIMETRIK risksiz getiri kazanir (eski NAV=1.0 / %0 degil).
    daily_rf acikca verilirse o kullanilir (geriye-uyumlu override). Risk
    varliklarina agirlik 0 -> weights tamamen sifirdir (Turnover=0).
    'RL piyasaya girmeyi hak ediyor mu?' sorusunun tabanini olusturur: bir ajan
    bu (artik egimi pozitif) cizgiyi risk-ayarli gecemiyorsa piyasa riskini
    almak bosunadir. Tamamen determinist (kapali-form, np.random yok)."""
    if daily_rf is None:
        daily_rf = cash_daily_rate(EnvConfig.cash_annual_rate, EnvConfig.trading_days)
    T, N = prices.shape
    rets = np.full(T, float(daily_rf))
    if T:
        rets[0] = 0.0
    nav = np.cumprod(1.0 + rets)
    weights = np.zeros((T, N))
    return dict(nav=nav, rets=rets, weights=weights,
                target_weights=weights.copy(), turnover=np.zeros(T),
                dates=list(prices.index))
