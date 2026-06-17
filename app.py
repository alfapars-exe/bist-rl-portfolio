"""BIST 28 Portföy RL — İnteraktif Streamlit Demo (ince giriş noktası, P5).

Tek komutla çalışır:
    streamlit run app.py

Sekmeler:
  1. Veri & MDP      : 28 ticker, MDP tuple, vade preset tablosu, adaptif ödül açıklaması
  2. Eğitim          : Seçili ajanı canlı güncellenen eğri-placeholder'larıyla eğit
  3. Test            : Adım-adım oynatma; durum, aksiyon, Q/ağırlık/ödül panelleri
  4. Karşılaştırma   : Tüm ajanlar + baseline'lar; metrik tablosu, NAV, heatmap

SOLID P5 (SRP): 1000+ satırlık monolit `ui/` paketine bölündü —
  ui/state.py (session defaults) · ui/services.py (veri/eğitim/test servisleri)
  ui/charts.py (saf grafikler) · ui/sidebar.py (kontroller) · ui/tabs/ (4 sekme)
Bu dosya yalnız giriş noktası + sekme dispatch + geriye-uyumluluk re-export'larıdır.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import streamlit as st

# ---------------------------------------------------------------------
# Geriye-uyumluluk re-export'ları — eski `app.X` erişimleri çalışmaya devam
# eder (testler app._make_agent'i çağırır; dış kullanıcılar sabitleri okur).
# ---------------------------------------------------------------------
from data import BIST28, SPLIT, START, END, download_bist, train_test_split  # noqa: F401
from env.portfolio_env import (  # noqa: F401
    ACTION_NAMES, HORIZON_PRESETS, DiscretePortfolioEnv, PortfolioEnv,
)
from agents import DQNAgent, PPOAgent, SACAgent  # noqa: F401
from config import SEED, FEATURES, DQNConfig, PPOConfig, SACConfig, ForecastConfig  # noqa: F401
from core.trainer import train as train_loop  # noqa: F401

from ui.state import ASSET_NAMES, _agent_key, _init_state, env_rebalance_hint  # noqa: F401
from ui.services import (  # noqa: F401
    _compute_test_tl_snaps, _load_data, _make_agent, _make_env,
    evaluate_with_trace, train_generator,
)
from ui.charts import (  # noqa: F401
    _horizon_preset_table, _q_bar, _reward_bar, _state_top_features, _weights_pie,
)
from ui.sidebar import _sidebar_reward_editor, sidebar_controls  # noqa: F401
from ui.tabs import tab_compare, tab_mdp, tab_test, tab_train  # noqa: F401


def main():
    st.set_page_config(
        page_title="BIST 28 RL Portföy Demosu",
        page_icon="📈",
        layout="wide",
    )
    _init_state()
    algo, horizon, adaptive, hp = sidebar_controls()

    st.title("📈 BIST 28 Pekiştirmeli Öğrenme Portföy Yönetimi")
    st.caption("UYİK 2026 · DQN/PPO/SAC · Vade preset'leri · Adaptif ödül şekillendirici")

    t1, t2, t3, t4 = st.tabs([
        "📐 Veri & MDP",
        "🎓 Eğitim",
        "🎬 Test (Adım-Adım)",
        "📊 Karşılaştırma",
    ])
    with t1: tab_mdp()
    with t2: tab_train(algo, horizon, adaptive, hp)
    with t3: tab_test(algo, horizon, adaptive)
    with t4: tab_compare()


if __name__ == "__main__":
    main()
