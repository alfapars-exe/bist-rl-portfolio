"""Klasik baseline stratejiler — karşılaştırma için."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(prices: pd.DataFrame) -> dict:
    """Günlük rebalansla eşit ağırlık."""
    r = prices.pct_change().fillna(0).values
    N = prices.shape[1]
    w = np.ones(N) / N
    rets = (r * w).sum(axis=1)
    nav = np.cumprod(1 + rets)
    return dict(nav=nav, rets=rets, weights=np.tile(w, (len(rets), 1)))


def buy_and_hold_index(prices: pd.DataFrame) -> dict:
    """Eşit ağırlık alıp tut (rebalans yok)."""
    p0 = prices.iloc[0].values
    shares = 1.0 / p0 / prices.shape[1]
    nav = (prices.values * shares).sum(axis=1)
    rets = np.diff(np.log(nav))
    return dict(
        nav=nav / nav[0],
        rets=np.concatenate([[0.0], rets]),
        weights=None,
    )


def mean_variance(prices: pd.DataFrame, lookback: int = 120,
                  rebalance: int = 20, risk_aversion: float = 5.0) -> dict:
    """Kısıtlı long-only Markowitz; yuvarlanan pencere + periyodik rebalans."""
    from scipy.optimize import minimize

    r = prices.pct_change().fillna(0).values
    T, N = r.shape
    navs = [1.0]; rets = []; w_hist = []
    w = np.ones(N) / N
    for t in range(T):
        if t >= lookback and (t - lookback) % rebalance == 0:
            hist = r[t - lookback: t]
            mu = hist.mean(axis=0)
            cov = np.cov(hist.T) + 1e-5 * np.eye(N)

            def obj(w_, mu=mu, cov=cov, ra=risk_aversion):
                return -(w_ @ mu) + 0.5 * ra * w_ @ cov @ w_

            res = minimize(
                obj, np.ones(N) / N,
                bounds=[(0, 0.2)] * N,
                constraints=({"type": "eq", "fun": lambda w_: w_.sum() - 1}),
            )
            w = res.x
        port_r = float((w * r[t]).sum())
        rets.append(port_r)
        navs.append(navs[-1] * (1 + port_r))
        w_hist.append(w.copy())
    return dict(
        nav=np.array(navs[1:]),
        rets=np.array(rets),
        weights=np.array(w_hist),
    )


# =====================================================================
# Ek akademik baseline'lar — hepsi DETERMINISTIK (RNG YOK), long-only,
# tam-yatirim (sum(w)=1). PARS sibling strategies.py'den odunc alindi ve
# kod/ API'sine ({"nav", "rets", "weights"} numpy) uyarlandi. Ortak
# simulasyon cekirdegi _rolling_backtest: yuvarlanan pencere + periyodik
# rebalans, ilk lookback gunu esit-agirlik isinma (warm-up).
# =====================================================================
def _rolling_backtest(prices: pd.DataFrame, weight_fn, lookback: int,
                      rebalance: int) -> dict:
    """Determinist yuvarlanan-pencere backtest cekirdegi.

    weight_fn(hist_returns:(L,N), w_prev:(N,)) -> w_new:(N,) dondurur.
    Donen agirliklar long-only normalize edilir (clip>=0, toplam=1).
    mean_variance ile ayni stilde tx-cost UYGULANMAZ — net edge karsilastirmasi
    Turnover kolonu uzerinden yapilir (golden tutarliligi). Hicbir np.random
    cagrisi yok -> golden RNG sirasi etkilenmez."""
    r = prices.pct_change().fillna(0.0).values
    T, N = r.shape
    navs = [1.0]; rets = []; w_hist = []
    w = np.ones(N) / N
    for t in range(T):
        if t >= lookback and (t - lookback) % rebalance == 0:
            hist = r[t - lookback: t]
            w_new = np.asarray(weight_fn(hist, w), dtype=float)
            w_new = np.clip(w_new, 0.0, None)
            s = w_new.sum()
            w = w_new / s if s > 1e-12 else np.ones(N) / N
        port_r = float((w * r[t]).sum())
        rets.append(port_r)
        navs.append(navs[-1] * (1 + port_r))
        w_hist.append(w.copy())
    return dict(nav=np.array(navs[1:]), rets=np.array(rets),
                weights=np.array(w_hist))


def inverse_volatility(prices: pd.DataFrame, lookback: int = 60,
                       rebalance: int = 20) -> dict:
    """Ters-volatilite (1/sigma) agirligi — yuvarlanan vol uzerinden.

    Mantik: w_i ∝ 1/sigma_i (sigma_i = pencere getiri std'i), normalize edilir.
    Riske gore dengeleme'nin (risk parity) kovaryanssiz, naif halidir: yalniz
    kosegen (varyans) bilgisi kullanilir, korelasyon yok sayilir. Dusuk-vol
    hisseye daha cok agirlik -> portfoy vol'unu duzler. 'Volatility tilt'
    olarak bilinen iyi-belgelenmis defensif bir anomalidir (Asness ve dig.)."""
    def f(hist, w):
        iv = 1.0 / (hist.std(axis=0) + 1e-9)
        return iv / iv.sum()
    return _rolling_backtest(prices, f, lookback, rebalance)


def risk_parity(prices: pd.DataFrame, lookback: int = 120,
                rebalance: int = 20, n_iter: int = 100) -> dict:
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
    return _rolling_backtest(prices, f, lookback, rebalance)


def min_variance(prices: pd.DataFrame, lookback: int = 120,
                 rebalance: int = 20, max_weight: float = 0.4) -> dict:
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
        return res.x
    return _rolling_backtest(prices, f, lookback, rebalance)


def momentum(prices: pd.DataFrame, lookback: int = 60, rebalance: int = 20,
             top_k: int = 5) -> dict:
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
    return _rolling_backtest(prices, f, lookback, rebalance)


def cash_riskfree(prices: pd.DataFrame, daily_rf: float = 0.0) -> dict:
    """Nakit / risk-free benchmark — tum sermaye nakitte, sabit gunluk getiri.

    En muhafazakar referans: piyasaya HIC maruz kalmadan elde edilen getiri.
    daily_rf=0 -> NAV duz 1.0 (sermaye korunumu, sifir risk). Pozitif daily_rf
    ile basit bir mevduat/repo getirisi modellenir (NAV=(1+rf)^t). Risk
    varliklarina agirlik 0 oldugundan weights tamamen sifirdir (Turnover=0).
    'RL piyasaya girmeyi hak ediyor mu?' sorusunun tabanini olusturur: bir ajan
    bu duz cizgiyi risk-ayarli olarak gecemiyorsa piyasa riskini almak bosunadir.
    Tamamen determinist (kapali-form, np.random yok)."""
    T, N = prices.shape
    rets = np.full(T, float(daily_rf))
    nav = np.cumprod(1.0 + rets)
    weights = np.zeros((T, N))
    return dict(nav=nav, rets=rets, weights=weights)
