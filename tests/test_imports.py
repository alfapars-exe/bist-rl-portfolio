"""Paket sagligi smoke testi — yan etkisiz moduller import edilebilmeli.

NOT: train.py / main.py / app.py / plots.py KASITLI olarak haric birakildi:
- train.py modul seviyesinde download_bist() + scaler.fit() calistirir (M1: import
  yan etkisi), import etmek tam veri yuklemeyi tetikler.
- app.py / plots.py streamlit / interaktif calisma zamani ya da hazir CSV bekler.
Bu listenin kisitli olmasi, M1 (modul-seviyesi yan etki) bulgusunun da kanitidir.
"""
import importlib

import pytest

SAFE_MODULES = [
    "data",
    "env", "env.portfolio_env",
    "agents", "agents.dqn", "agents.ppo", "agents.sac",
    "utils", "utils.features", "utils.metrics",
    "utils.baselines", "utils.portfolio_tl",
]


@pytest.mark.parametrize("mod", SAFE_MODULES)
def test_safe_module_imports(mod):
    importlib.import_module(mod)
