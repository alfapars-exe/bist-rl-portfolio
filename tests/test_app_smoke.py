"""Streamlit UI duman testi (Faz 3) — app.py tarayicisiz render edilebilmeli.

streamlit.testing.v1.AppTest ile headless calistirir; ilk render (veri
yuklenmeden) istisnasiz tamamlanmali. train_generator/evaluate_with_trace
buton tiklamasi gerektirdiginden burada test edilmez (onlarin cekirdek mantigi
test_trainer/test_rollout'ta kapsanir); bu test modul + ilk render + sidebar +
sekme yapisinin saglam oldugunu garanti eder.
"""
import pytest

pytest.importorskip("streamlit.testing.v1")


@pytest.mark.slow
def test_app_initial_render_no_exception():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=90).run()
    assert not at.exception, f"UI ilk render istisnasi: {list(at.exception)}"
