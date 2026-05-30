"""CNN-LSTM forecaster — egitim + sizintisizlik (Faz V4).

Kritik: forecast t'de yalniz <=t getirilerine baglidir (lookahead yok) ve model
yalniz train'de fit edilir. test_forecast_is_causal_no_lookahead bunu kilitler.
"""
import numpy as np
import pandas as pd

from forecast.forecaster import build_forecast_feature


def _prices(seed=0, n=400, k=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, (n, k)), axis=0)),
                        index=idx, columns=[f"A{i}" for i in range(k)])


def test_forecast_feature_shape_and_finite():
    px = _prices()
    tr = px.iloc[:300]
    f = build_forecast_feature(px, tr, window=20, epochs=2, batch=128, seed=0)
    assert f.shape == px.shape
    assert np.isfinite(f.values).all()
    assert (f.iloc[:19] == 0).all().all()           # warmup (t < window-1) sifir


def test_forecast_is_causal_no_lookahead():
    """t'deki forecast yalniz <=t getirilerine bagli; gelecek degisse <=t degismez."""
    px = _prices(seed=1)
    tr = px.iloc[:300]                               # train AYNI -> model AYNI (seed=0)
    t = 350
    f_full = build_forecast_feature(px, tr, window=20, epochs=2, batch=128, seed=0)
    px2 = px.copy()
    px2.iloc[t + 1:] *= 1.4                           # yalniz gelecegi degistir
    f_trunc = build_forecast_feature(px2, tr, window=20, epochs=2, batch=128, seed=0)
    a = f_full.iloc[: t + 1].values
    b = f_trunc.iloc[: t + 1].values
    assert np.allclose(a, b, atol=1e-6), "forecast lookahead sizintisi!"


def test_forecast_deterministic_given_seed():
    px = _prices(seed=2)
    tr = px.iloc[:300]
    f1 = build_forecast_feature(px, tr, window=20, epochs=2, batch=128, seed=3)
    f2 = build_forecast_feature(px, tr, window=20, epochs=2, batch=128, seed=3)
    assert np.allclose(f1.values, f2.values)         # tohum -> tekrar-uretilebilir (golden)
