"""Golden-master regresyon kapisi.

Faz 0'da `python main.py` ciktisi tests/golden/ altina dondurulur. Her refactor
sonrasi yeniden `python main.py` calistirilip bu test ile karsilastirilir:
seed=42 sabit oldugundan davranisi koruyan bir refactor ayni metrikleri uretmeli.

Faz 0'da iki bagimsiz run karsilastirildi: metrikler <=1e-6 toleransta eslesti
(egitim bu makinede seed=42 + sabit kurulum sirasi sayesinde deterministik).
Bu yuzden siki tolerans kullaniyoruz — refactor en kucuk davranis sapmasini yakalar.
results/metrics.csv yoksa (taze calistirma yapilmadiysa) test atlanir.

v10 NaN-guard: CashRiskFree Sharpe/Sortino/Calmar tanimsiz (sifir-vol) -> NaN;
assert_allclose equal_nan=True ile NaN hucreleri eslestirir (NaN==NaN kabul).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT   = Path(__file__).resolve().parent.parent
GOLDEN_METRICS  = Path(__file__).resolve().parent / "golden" / "metrics_baseline.csv"
GOLDEN_NAVS     = Path(__file__).resolve().parent / "golden" / "navs_aligned_baseline.csv"
CURRENT_METRICS = ROOT / "results" / "metrics.csv"
CURRENT_NAVS    = ROOT / "results" / "navs_aligned.csv"

# Faz 0: iki bagimsiz run <=1e-6'da eslesti (deterministik dogrulandi).
ATOL = 1e-6
RTOL = 1e-6


@pytest.mark.skipif(not GOLDEN_METRICS.exists(),  reason="golden baseline yok (Faz 0 capture gerekli)")
@pytest.mark.skipif(not CURRENT_METRICS.exists(), reason="results/metrics.csv yok — once `python main.py` calistir")
def test_metrics_match_golden():
    golden  = pd.read_csv(GOLDEN_METRICS,  index_col=0).sort_index()
    current = pd.read_csv(CURRENT_METRICS, index_col=0).sort_index()

    # Golden, baseline'lanmis stratejileri kapsar. Sonradan eklenen YENI ajanlar
    # current'ta FAZLADAN satir olarak gorunur — bu bir regresyon DEGIL.
    # (1) golden'in TUM satirlari current'ta bulunmali (eksik satir = regresyon),
    # (2) eslesen satirlar <=1e-6 oturmali.
    missing = sorted(set(golden.index) - set(current.index))
    assert not missing, f"Golden stratejileri current'ta eksik (regresyon?): {missing}"
    assert list(golden.columns) == list(current.columns), (
        f"Metrik kolonlari farkli: golden={list(golden.columns)} current={list(current.columns)}")

    cur = current.reindex(golden.index)   # yalniz baseline'lanmis satirlar

    # v10 NaN-guard: CashRiskFree Sharpe/Sortino/Calmar tanimsiz (sifir-vol).
    # equal_nan=True -> NaN==NaN eslesir (NaN siralanmis yerde NaN beklenir).
    np.testing.assert_allclose(
        cur.to_numpy(dtype=float, na_value=np.nan),
        golden.to_numpy(dtype=float, na_value=np.nan),
        atol=ATOL, rtol=RTOL, equal_nan=True,
        err_msg="Metrikler golden-master'dan tolerans disinda sapti — regresyon?",
    )


@pytest.mark.skipif(not GOLDEN_NAVS.exists(),  reason="navs_aligned golden yok")
@pytest.mark.skipif(not CURRENT_NAVS.exists(), reason="results/navs_aligned.csv yok — once `python main.py` calistir")
def test_navs_aligned_match_golden():
    """C1 re-base: NAV[0]=1 ve ayni uzunlukta; golden-master navs_aligned.csv eslesmeli."""
    golden  = pd.read_csv(GOLDEN_NAVS,  index_col=0).sort_index()
    current = pd.read_csv(CURRENT_NAVS, index_col=0).sort_index()

    # Yeni stratejiler current'ta fazladan sutun olabilir — golden sutunlari subset olmali.
    missing_cols = sorted(set(golden.columns) - set(current.columns))
    assert not missing_cols, f"NAV sutunlari current'ta eksik: {missing_cols}"

    cur = current[golden.columns]   # yalniz golden sutunlari

    # Satir sayisi esit olmali (ayni test penceresi).
    assert len(cur) == len(golden), (
        f"NAV uzunluk farki: golden={len(golden)} current={len(cur)}")

    np.testing.assert_allclose(
        cur.to_numpy(dtype=float, na_value=np.nan),
        golden.to_numpy(dtype=float, na_value=np.nan),
        atol=ATOL, rtol=RTOL, equal_nan=True,
        err_msg="navs_aligned golden-master'dan sapti — regresyon?",
    )
