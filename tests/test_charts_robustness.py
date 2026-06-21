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


def test_live_curves_keys_unique_across_loop_and_final():
    """Regresyon (Streamlit 1.5x StreamlitDuplicateElementKey): _render_live_curves
    egitim DONGUSU icinde (seq=iter) + DONGU-SONRASI 'final flush' (seq='final')
    cagrilir. Streamlit 1.5x st.empty() slot'una her cizimde key'i yeniden kaydeder
    -> TUM render'lardaki plotly key'leri benzersiz olmali. Bug (sabit key veya
    flush'ta default seq=0) bu testi kirardi: 'train_live_reward_0' iki kez."""
    import pandas as pd
    from ui.tabs import train as T

    class _RecPH:
        def __init__(self, sink):
            self.sink = sink
        def plotly_chart(self, fig, **kw):
            self.sink.append(kw["key"])      # key zorunlu olmali (yoksa KeyError -> regresyon)

    keys = []
    phs = [_RecPH(keys) for _ in range(4)]
    df = pd.DataFrame({"iter": range(3), "reward": [1.0, 2.0, 3.0],
                       "gain": [0.1, 0.2, 0.3], "success": [1, 0, 1],
                       "loss": [0.5, 0.4, 0.3]})
    for seq in [0, 1, 2, "final"]:           # dongu render'lari (iter) + final flush
        T._render_live_curves(df, *phs, seq=seq)
    assert len(keys) == len(set(keys)), f"Cakisan plotly key: {keys}"
    # final flush ile dongu-ilk render (seq=0) AYRI key'ler olmali
    assert "train_live_reward_final" in keys
    assert "train_live_reward_0" in keys
