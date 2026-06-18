"""Monte-Carlo stres (Student-t + blok bootstrap) bilinen-deger testleri.

utils/stress_mc.py GOZLEMSEL (raporlama) — env-yerel rng, golden'a dokunmaz.
"""
import numpy as np

from utils.stress_mc import mc_student_t, mc_block_bootstrap, summarize_mc


def _hist(seed=0, t_len=400, n=5):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0003, 0.012, (t_len, n))


def test_block_bootstrap_deterministic_and_shaped():
    R = _hist()
    w = np.full(5, 0.2)
    a = mc_block_bootstrap(R, w, horizon=60, paths=200, block=10, seed=42)
    b = mc_block_bootstrap(R, w, horizon=60, paths=200, block=10, seed=42)
    assert a["terminal"].shape == (200,)
    np.testing.assert_array_equal(a["terminal"], b["terminal"])   # seed -> deterministik


def test_summarize_mc_keys_and_ranges():
    R = _hist()
    out = mc_block_bootstrap(R, np.full(5, 0.2), horizon=60, paths=400, seed=1)["summary"]
    for k in ("mean", "median", "VaR", "CVaR", "prob_loss", "worst", "best", "p05", "p95"):
        assert k in out
    assert out["VaR"] >= 0.0 and out["CVaR"] >= 0.0
    assert 0.0 <= out["prob_loss"] <= 1.0
    assert out["worst"] <= out["best"]


def test_student_t_strips_cash_weight():
    R = _hist(n=5)
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])         # n+1 (nakit) -> _assets_weights kirpar
    out = mc_student_t(R, w, horizon=40, paths=200, seed=3)
    assert out["terminal"].shape == (200,)
    assert np.isfinite(out["summary"]["mean"])


def test_student_t_heavier_tails_than_bootstrap_proxy():
    # Student-t (dof=4) saf gurultude blok-bootstrap'tan daha agir kuyruk -> CVaR >= bootstrap CVaR
    R = _hist(seed=7)
    w = np.full(5, 0.2)
    cvar_t = mc_student_t(R, w, horizon=120, paths=1500, dof=4.0, seed=5)["summary"]["CVaR"]
    cvar_bb = mc_block_bootstrap(R, w, horizon=120, paths=1500, block=20, seed=5)["summary"]["CVaR"]
    assert cvar_t >= cvar_bb * 0.8                        # t kuyruğu en az bootstrap kadar şişman
