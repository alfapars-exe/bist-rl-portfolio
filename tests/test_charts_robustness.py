"""UI grafik dayanikliligi — evren beklenenden kucukse (yfinance kismi veri)
_weights_pie cokmemeli (ASSET_NAMES sabit 29; agirlik vektoru daha kisa olabilir).
Regresyon: 'All arrays must be of the same length' (HF Space, ui/charts.py:20)."""
import numpy as np

from ui.charts import _weights_pie
from ui.state import ASSET_NAMES


def test_weights_pie_full_universe():
    w = np.full(len(ASSET_NAMES), 1.0 / len(ASSET_NAMES), dtype=np.float32)
    assert _weights_pie(w) is not None


def test_weights_pie_short_universe_no_crash():
    # 26 riskli + nakit = 27 (yfinance 2 ticker dondurmedi senaryosu)
    w = np.full(27, 1.0 / 27, dtype=np.float32)
    fig = _weights_pie(w)                       # cokmemeli
    assert fig is not None


def test_weights_pie_tiny_universe_no_crash():
    fig = _weights_pie(np.array([0.5, 0.3, 0.2], dtype=np.float32))
    assert fig is not None
