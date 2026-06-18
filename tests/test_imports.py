"""Paket sagligi smoke testi — moduller import edilebilmeli.

Faz 4 (M1) sonrasi train/plots/main ARTIK import-guvenli: modul seviyesinde veri
indirme/egitim YOK (train.prepare_data / *.run() ile tetiklenir). Bu testin
train/plots/main'i icermesi, M1'in (import yan etkisi) cozuldugunun kanitidir.
app.py streamlit-agir oldugundan burada degil; import'u test_config_wiring +
test_app_smoke ile ayrica dogrulanir.
"""
import importlib

import pytest

SAFE_MODULES = [
    "config",
    "data",
    "env", "env.portfolio_env", "env.reward",
    "agents", "agents.base", "agents.common",
    "agents.dqn", "agents.ppo", "agents.sac",
    "utils", "utils.features", "utils.metrics",
    "utils.baselines", "utils.portfolio_tl", "utils.torch_utils", "utils.macro",
    "forecast", "forecast.forecaster",
    "core", "core.trainer", "core.rollout", "core.features", "core.factory",
    "train", "plots", "main",
]


@pytest.mark.parametrize("mod", SAFE_MODULES)
def test_safe_module_imports(mod):
    importlib.import_module(mod)
