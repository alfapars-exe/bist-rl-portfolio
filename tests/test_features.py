"""TrainScaler sizintisizlik (leak-safety) degismezleri.

Kritik bildirisel iddia: olcek istatistikleri YALNIZCA train'de fit edilir;
test donemine ayni istatistikler uygulanir, orada yeniden fit EDILMEZ.
"""
import numpy as np
import pandas as pd
import pytest

from utils.features import TrainScaler


def _toy_feats(seed=0, n=120, k=4):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"A{i}" for i in range(k)]
    def mk(scale, loc=0.0):
        return pd.DataFrame(rng.normal(loc, scale, size=(n, k)), index=idx, columns=cols)
    return {"logret": mk(0.01), "vol20": mk(0.02, 0.05)}


def test_scaler_fit_uses_only_train_stats():
    feats = _toy_feats()
    sc = TrainScaler().fit(feats)
    for k, df in feats.items():
        pd.testing.assert_series_equal(sc.means[k], df.mean(axis=0), check_names=False)
        expected_std = df.std(axis=0).replace(0, 1.0)
        pd.testing.assert_series_equal(sc.stds[k], expected_std, check_names=False)


def test_transform_does_not_refit_on_new_data():
    train = _toy_feats(seed=0)
    test = _toy_feats(seed=99)  # farkli dagilim
    sc = TrainScaler().fit(train)
    means_before = {k: v.copy() for k, v in sc.means.items()}
    stds_before = {k: v.copy() for k, v in sc.stds.items()}
    _ = sc.transform(test)
    # transform stored istatistikleri DEGISTIRMEMELI (yeniden fit yok)
    for k in sc.means:
        pd.testing.assert_series_equal(sc.means[k], means_before[k])
        pd.testing.assert_series_equal(sc.stds[k], stds_before[k])


def test_transform_applies_train_zscore():
    train = _toy_feats(seed=0)
    test = _toy_feats(seed=1)
    sc = TrainScaler(clip=8.0).fit(train)
    out = sc.transform(test)
    for k in test:
        mu, sd = sc.means[k], sc.stds[k]
        expected = ((test[k] - mu) / sd).clip(-8.0, 8.0).fillna(0.0)
        pd.testing.assert_frame_equal(out[k], expected)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        TrainScaler().transform(_toy_feats())


def test_zero_variance_column_no_div_by_zero():
    idx = pd.bdate_range("2020-01-01", periods=50)
    const = pd.DataFrame(np.full((50, 2), 3.0), index=idx, columns=["A", "B"])
    sc = TrainScaler().fit({"f": const})
    out = sc.transform({"f": const})
    assert np.isfinite(out["f"].values).all()
    # sabit kolon: std 1.0'a sabitlenir -> z-score (x-mean)/1 = 0
    assert np.allclose(out["f"].values, 0.0)
