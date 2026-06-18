"""v6 makro rejim öznitelikleri — sıra, rejim aralığı, leak-safe ölçekleme.

Makro blok (faiz/dolar/altın + rejim omurgası) state'e eklenir; bu testler
öznitelik sözleşmesini (sıra, [-1,1] rejim, sızıntısızlık) kilitler.
"""
import numpy as np
import pandas as pd

from config import MacroConfig
from utils.macro import MacroScaler, add_macro_features


def _toy_macro(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({
        "^VIX": np.clip(18 + 5 * np.abs(np.cumsum(rng.normal(0, 0.3, n)) % 2.0), 9, 70),
        "^GSPC": 1500 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))),
        "^TNX": np.clip(3 + np.cumsum(rng.normal(0, 0.02, n)), 0.5, 6),
        "^IRX": np.clip(2 + np.cumsum(rng.normal(0, 0.02, n)), 0.05, 5),
        "USDTRY=X": 2 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, n))),
        "GC=F": 1300 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, n))),
    }, index=idx)


def test_macro_feature_order_and_regime_range():
    f = add_macro_features(_toy_macro())
    assert list(f.columns) == list(MacroConfig.features)   # sıra MacroConfig ile sabit
    assert f.shape[1] == 4
    assert f["regime"].between(-1.0, 1.0).all()            # rejim ∈ [-1,1]
    assert int(f.isna().sum().sum()) == 0                  # causal fill -> NaN yok


def test_macro_scaler_leak_safe_and_bounded():
    f = add_macro_features(_toy_macro())
    sc = MacroScaler().fit(f.iloc[:300])                   # YALNIZ train fit
    z = sc.transform(f)
    assert z.shape == f.shape
    assert abs(float(z.iloc[:300].mean().mean())) < 0.2    # train z-score ~0
    assert np.isfinite(z.to_numpy()).all()
    assert float(z.abs().max().max()) <= 8.0 + 1e-9        # ±8σ clip


def test_macro_missing_series_degrades_to_zero():
    m = _toy_macro().drop(columns=["GC=F"])                # altın serisi yok
    f = add_macro_features(m)
    assert (f["gold_tl_mom"] == 0).all()                   # eksik seri -> 0 (graceful)
    assert list(f.columns) == list(MacroConfig.features)
