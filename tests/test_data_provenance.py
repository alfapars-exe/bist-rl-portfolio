from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

import data as data_mod


def test_cache_coverage_requires_full_requested_range():
    idx = pd.bdate_range("2020-02-01", "2020-11-30")
    assert not data_mod._cache_covers(idx, "2020-01-01", "2020-12-31")
    assert data_mod._cache_covers(idx, "2020-03-01", "2020-10-31")


def test_align_macro_never_backfills_from_future():
    market = pd.bdate_range("2020-01-01", periods=5)
    macro = pd.DataFrame({"X": [3.0]}, index=[market[2]])
    aligned = data_mod.align_macro(macro, market)
    assert aligned.iloc[:2].isna().all().all()
    assert aligned.iloc[2:]["X"].eq(3.0).all()


def test_yfinance_end_is_requested_as_inclusive(monkeypatch, tmp_path):
    calls = {}
    idx = pd.bdate_range("2020-01-01", periods=130)
    tickers = data_mod.BIST28
    columns = pd.MultiIndex.from_product([["Close"], tickers])
    frame = pd.DataFrame(100.0, index=idx, columns=columns)

    def download(*args, **kwargs):
        calls.update(kwargs)
        return frame

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    monkeypatch.setattr(data_mod, "PARQUET_PATH", tmp_path / "prices.parquet")
    out = data_mod.download_bist(tickers=tickers, start="2020-01-01", end="2020-06-30",
                                 use_cache=False)
    assert calls["end"] == "2020-07-01"
    assert out.attrs["provenance"]["source"] == "real"


def test_partial_provider_data_is_marked_mixed(monkeypatch, tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=130)
    present = data_mod.BIST28[:12]
    columns = pd.MultiIndex.from_product([["Close"], present])
    frame = pd.DataFrame(
        100.0 * np.exp(np.cumsum(np.zeros((len(idx), len(present))), axis=0)),
        index=idx, columns=columns,
    )
    monkeypatch.setitem(sys.modules, "yfinance",
                        types.SimpleNamespace(download=lambda *a, **k: frame))
    monkeypatch.setattr(data_mod, "PARQUET_PATH", tmp_path / "prices.parquet")
    out = data_mod.download_bist(start="2020-01-01", end="2020-06-30", use_cache=False)
    assert out.attrs["provenance"]["source"] == "mixed"
    assert set(out.attrs["provenance"]["missing_tickers"]) == set(data_mod.BIST28[12:])
    assert not (tmp_path / "prices.parquet").exists()


def test_tz_aware_provider_index_does_not_empty_mixed_frame(monkeypatch, tmp_path):
    """Regresyon (HF Space MIXED-bos hatasi): yfinance tz-aware index dondururse
    sentetik-doldurma reindex'i naive bdate_range ile ESLESMEZ -> tum sentetik
    sutunlar NaN -> son dropna frame'i BOSALTIRDI (add_features ValueError). Index
    tz-naive GUNE normalize edildigi icin frame artik BOS DEGIL + tam BIST28."""
    idx = pd.bdate_range("2015-01-01", "2024-12-31", tz="UTC")        # TZ-AWARE
    present = data_mod.BIST28[:15]
    columns = pd.MultiIndex.from_product([["Close"], present])
    frame = pd.DataFrame(
        100.0 * np.exp(np.cumsum(
            np.random.default_rng(1).normal(0, 0.01, (len(idx), len(present))), axis=0)),
        index=idx, columns=columns,
    )
    monkeypatch.setitem(sys.modules, "yfinance",
                        types.SimpleNamespace(download=lambda *a, **k: frame))
    monkeypatch.setattr(data_mod, "PARQUET_PATH", tmp_path / "prices.parquet")
    out = data_mod.download_bist(start="2015-01-01", end="2024-12-31", use_cache=False)
    assert not out.empty, "tz-aware index frame'i bosaltti (HF Space hatasi geri geldi)"
    assert isinstance(out.index, pd.DatetimeIndex) and out.index.tz is None
    assert out.shape[1] == len(data_mod.BIST28)                       # tam BIST28
    assert not out.isna().any().any()
    assert out.attrs["provenance"]["source"] == "mixed"
    from utils.features import add_features                           # asil Space hatasi
    assert len(add_features(out)) >= 12
