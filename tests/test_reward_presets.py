"""Odul preset persistence + gain_bonus onizleme formul testleri.

Golden-guvenli: env/reward.py veya egitim cekirdegine dokunmaz.
Yalniz core.persistence'in yeni JSON fonksiyonlarini ve ui.tabs.reward'daki
pür-Python yardimcilarini test eder.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Yardimci: gecici REWARD_PRESETS_DIR
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_presets_dir(tmp_path, monkeypatch):
    """REWARD_PRESETS_DIR'i gecici dizine yonlendir."""
    import core.persistence as pers
    monkeypatch.setattr(pers, "REWARD_PRESETS_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# 1) Round-trip: kaydet -> yukle -> ayni dict
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_presets_dir):
    from core.persistence import save_reward_preset, load_reward_preset

    cfg = {
        "eta_base": 0.0015,
        "lambda_base": 0.75,
        "tau_base": 0.04,
        "w_gain": 0.0,
        "w_ruin_timing": 0.0,
    }
    saved_path = save_reward_preset(cfg, "test_preset", "test aciklamasi")
    assert Path(saved_path).exists(), "Dosya olusturulmadi"

    loaded = load_reward_preset("test_preset")
    assert loaded is not None, "load_reward_preset None dondu"
    assert loaded == cfg, f"Round-trip esitsizligi: {loaded} != {cfg}"


def test_load_nonexistent_returns_none(tmp_presets_dir):
    from core.persistence import load_reward_preset

    result = load_reward_preset("var_olmayan_preset")
    assert result is None


def test_save_sanitizes_name(tmp_presets_dir):
    """Isim sanitizasyonu: bosluk/ozel karakter -> _ ; dosya olusur."""
    from core.persistence import save_reward_preset, reward_preset_path

    save_reward_preset({"eta_base": 0.001}, "benim preset im!", "")
    expected = reward_preset_path("benim preset im!")
    assert expected.exists(), f"Beklenen dosya yok: {expected}"


def test_save_empty_name_uses_preset(tmp_presets_dir):
    """Bos isim -> 'preset' fallback dosyasi olusur."""
    from core.persistence import save_reward_preset, reward_preset_path

    save_reward_preset({"eta_base": 0.001}, "", "")
    expected = reward_preset_path("")
    assert expected.exists()
    assert expected.stem == "preset"


# ---------------------------------------------------------------------------
# 2) list_reward_presets: siralama + bozuk dosya atlama
# ---------------------------------------------------------------------------

def test_list_reward_presets_sorted_newest_first(tmp_presets_dir):
    """saved_at farkliligi: JSON'a dogrudan farkli tarihler yaz."""
    import json as _json

    # Dosyalari elle olustur — saved_at farkini garantile (saniye hassasiyeti)
    eski_path = tmp_presets_dir / "eski_preset.json"
    yeni_path = tmp_presets_dir / "yeni_preset.json"

    eski_path.write_text(_json.dumps({
        "name": "eski_preset", "description": "", "format": 1,
        "saved_at": "2024-01-01T10:00:00",
        "reward_cfg": {"eta_base": 0.001},
    }), encoding="utf-8")

    yeni_path.write_text(_json.dumps({
        "name": "yeni_preset", "description": "", "format": 1,
        "saved_at": "2024-06-01T12:00:00",
        "reward_cfg": {"eta_base": 0.002},
    }), encoding="utf-8")

    from core.persistence import list_reward_presets
    presets = list_reward_presets()
    assert len(presets) == 2
    # Yeni -> eski siralama (buyuk saved_at once gelir)
    assert presets[0]["name"] == "yeni_preset", (
        f"Beklenen: yeni_preset, Gelen: {presets[0]['name']}"
    )
    assert presets[1]["name"] == "eski_preset"


def test_list_reward_presets_skips_corrupt_json(tmp_presets_dir):
    from core.persistence import save_reward_preset, list_reward_presets

    save_reward_preset({"eta_base": 0.001}, "iyi_preset", "")
    # Bozuk JSON yaz
    corrupt = tmp_presets_dir / "bozuk.json"
    corrupt.write_text("{bu gecersiz json", encoding="utf-8")

    presets = list_reward_presets()
    names = [p["name"] for p in presets]
    assert "iyi_preset" in names, "Gecerli preset listede olmali"
    # Bozuk dosya listede gorünmemeli
    assert all(p["name"] != "bozuk" for p in presets), "Bozuk dosya atlanmali"


def test_list_empty_dir(tmp_presets_dir):
    from core.persistence import list_reward_presets

    presets = list_reward_presets()
    assert presets == []


def test_list_nonexistent_dir(monkeypatch, tmp_path):
    """REWARD_PRESETS_DIR yoksa bos liste doner, hata vermez."""
    import core.persistence as pers
    nonexistent = tmp_path / "hayir_boyle_bir_dizin_yok"
    monkeypatch.setattr(pers, "REWARD_PRESETS_DIR", nonexistent)

    from core.persistence import list_reward_presets
    assert list_reward_presets() == []


# ---------------------------------------------------------------------------
# 3) gain_bonus onizleme formulu
# ---------------------------------------------------------------------------

def test_gain_bonus_w_gain_zero():
    """w_gain=0 -> her nav ve step_frac icin 0 (golden opt-in kapali)."""
    from ui.tabs.reward import _gain_bonus

    assert _gain_bonus(2.0, 0.0, 0.0, 1.0, 0.0) == 0.0
    assert _gain_bonus(3.0, 0.5, 0.0, 1.0, 1.0) == 0.0
    assert _gain_bonus(0.5, 0.0, 0.0, 1.0, 0.0) == 0.0


def test_gain_bonus_2x_start():
    """w_gain=0.1, gain_floor=1.0, w_gain_speed=0 -> nav=2.0, step_frac=0 => 0.1."""
    from ui.tabs.reward import _gain_bonus

    result = _gain_bonus(2.0, 0.0, 0.1, 1.0, 0.0)
    assert abs(result - 0.1) < 1e-9, f"Beklenen 0.1, gelen {result}"


def test_gain_bonus_3x_start():
    """w_gain=0.1, gain_floor=1.0, w_gain_speed=0 -> nav=3.0, step_frac=0 => 0.2."""
    from ui.tabs.reward import _gain_bonus

    result = _gain_bonus(3.0, 0.0, 0.1, 1.0, 0.0)
    assert abs(result - 0.2) < 1e-9, f"Beklenen 0.2, gelen {result}"


def test_gain_bonus_below_floor():
    """nav < gain_floor -> bonus = 0."""
    from ui.tabs.reward import _gain_bonus

    result = _gain_bonus(0.8, 0.0, 0.1, 1.0, 0.0)
    assert result == 0.0


def test_gain_bonus_speed_factor():
    """w_gain_speed > 0 -> step_frac=0'da speed=1+w_gain_speed, step_frac=1'de speed=1."""
    from ui.tabs.reward import _gain_bonus

    # step_frac=0 (basi): (1 + 1.0*(1-0)) = 2.0
    result_start = _gain_bonus(2.0, 0.0, 0.1, 1.0, 1.0)
    assert abs(result_start - 0.2) < 1e-9, f"Beklenen 0.2, gelen {result_start}"

    # step_frac=1 (sonu): (1 + 1.0*(1-1)) = 1.0
    result_end = _gain_bonus(2.0, 1.0, 0.1, 1.0, 1.0)
    assert abs(result_end - 0.1) < 1e-9, f"Beklenen 0.1, gelen {result_end}"


# ---------------------------------------------------------------------------
# 4) ruin_penalty onizleme formulu
# ---------------------------------------------------------------------------

def test_ruin_penalty_flat():
    """w_ruin_timing=0 -> her step_frac icin bankruptcy_penalty'e esit."""
    from ui.tabs.reward import _ruin_penalty

    assert abs(_ruin_penalty(10.0, 0.0, 0.0) - 10.0) < 1e-9
    assert abs(_ruin_penalty(10.0, 0.0, 0.5) - 10.0) < 1e-9
    assert abs(_ruin_penalty(10.0, 0.0, 0.99) - 10.0) < 1e-9


def test_ruin_penalty_early_vs_late():
    """w_ruin_timing > 0 -> erken iflas cezasi gec iflasa gore buyuk."""
    from ui.tabs.reward import _ruin_penalty

    early = _ruin_penalty(10.0, 1.0, 0.05)
    late  = _ruin_penalty(10.0, 1.0, 0.99)
    assert early > late, f"Erken iflas cezasi ({early}) gec iflasa ({late}) gore buyuk olmali"


# ---------------------------------------------------------------------------
# 5) JSON meta alanlari
# ---------------------------------------------------------------------------

def test_save_includes_meta_fields(tmp_presets_dir):
    """Kaydedilen JSON; name, description, saved_at, format, reward_cfg icerir."""
    from core.persistence import save_reward_preset, reward_preset_path

    cfg = {"eta_base": 0.001}
    save_reward_preset(cfg, "meta_test", "bir aciklama")
    path = reward_preset_path("meta_test")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["name"] == "meta_test"
    assert data["description"] == "bir aciklama"
    assert "saved_at" in data and data["saved_at"]
    assert data["format"] == 1
    assert data["reward_cfg"] == cfg
