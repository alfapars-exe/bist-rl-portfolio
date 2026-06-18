"""Golden-master regresyon kapisi.

Faz 0'da `python main.py` ciktisi tests/golden/ altina dondurulur. Her refactor
sonrasi yeniden `python main.py` calistirilip bu test ile karsilastirilir:
seed=42 sabit oldugundan davranisi koruyan bir refactor ayni metrikleri uretmeli.

Faz 0'da iki bagimsiz run karsilastirildi: metrikler <=1e-6 toleransta eslesti
(egitim bu makinede seed=42 + sabit kurulum sirasi sayesinde deterministik).
Bu yuzden siki tolerans kullaniyoruz — refactor en kucuk davranis sapmasini yakalar.
results/metrics.csv yoksa (taze calistirma yapilmadiysa) test atlanir.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "golden" / "metrics_baseline.csv"
CURRENT = ROOT / "results" / "metrics.csv"

# Faz 0: iki bagimsiz run <=1e-6'da eslesti (deterministik dogrulandi).
ATOL = 1e-6
RTOL = 1e-6


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden baseline yok (Faz 0 capture gerekli)")
@pytest.mark.skipif(not CURRENT.exists(), reason="results/metrics.csv yok — once `python main.py` calistir")
def test_metrics_match_golden():
    golden = pd.read_csv(GOLDEN, index_col=0).sort_index()
    current = pd.read_csv(CURRENT, index_col=0).sort_index()
    # Golden, baseline'lanmis stratejileri (DQN/PPO/SAC + 3 baseline) kapsar.
    # Sonradan eklenen YENI ajanlar (or. TD3) current'ta FAZLADAN satir olarak
    # gorunur — bu bir regresyon DEGIL, kapsam genislemesidir. Bu yuzden:
    #   (1) golden'in TUM satirlari current'ta bulunmali (eksik satir = regresyon),
    #   (2) eslesen satirlar <=1e-6 oturmali. Yeni ajanin kendi davranisi
    #       birim testleriyle (test_agents/test_trainer) korunur; golden satiri ise
    #       davranis-degistiren bir sonraki iterasyonda (or. V8 gurultu) eklenir.
    missing = sorted(set(golden.index) - set(current.index))
    assert not missing, f"Golden stratejileri current'ta eksik (regresyon?): {missing}"
    assert list(golden.columns) == list(current.columns), "Metrik kolonlari farkli"
    cur = current.reindex(golden.index)               # yalniz baseline'lanmis satirlar
    np.testing.assert_allclose(
        cur.values, golden.values, atol=ATOL, rtol=RTOL,
        err_msg="Metrikler golden-master'dan tolerans disinda sapti — regresyon?",
    )
