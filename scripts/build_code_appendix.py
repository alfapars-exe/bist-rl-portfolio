"""Tum kaynak kodu tek dosyada toplar -> KOD_EKI.md (PDF §10: kod raporun ekine).

Tekrar-uretilebilir:  python scripts/build_code_appendix.py
Haric: tests/, scripts/, 'Reinforcement Learning Final/', .venv*, __pycache__,
models/, results/, figures/, data/ (uretilen/yardimci dosyalar).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "KOD_EKI.md"

# Sabit, okunabilir sira: yapılandirma -> yardimcilar -> ortam/odul -> ajanlar
# -> cekirdek -> CLI -> arayuz.
ORDER = [
    "config.py", "data.py",
    "utils/features.py", "utils/metrics.py", "utils/baselines.py",
    "utils/portfolio_tl.py", "utils/torch_utils.py",
    "forecast/forecaster.py",
    "env/portfolio_env.py", "env/reward.py",
    "agents/base.py", "agents/common.py", "agents/dqn.py", "agents/ppo.py", "agents/sac.py",
    "core/features.py", "core/factory.py", "core/trainer.py", "core/rollout.py",
    "core/walkforward.py", "core/persistence.py",
    "train.py", "main.py", "plots.py",
    "app.py",
    "ui/state.py", "ui/services.py", "ui/charts.py", "ui/sidebar.py",
    "ui/tabs/mdp.py", "ui/tabs/train.py", "ui/tabs/test.py", "ui/tabs/compare.py",
]


def main():
    present = [rel for rel in ORDER if (ROOT / rel).exists()]
    total_lines = 0
    body = []
    for rel in present:
        code = (ROOT / rel).read_text(encoding="utf-8")
        total_lines += code.count("\n") + 1
        body.append(f"\n---\n\n## `{rel}`\n\n```python\n{code}\n```\n")

    header = (
        "# Kod Eki — BIST 28 Portföy Yönetimi (Pekiştirmeli Öğrenme)\n\n"
        "> PDF §10: tüm kaynak kod final raporunun sonuna eklenir. Bu dosya "
        "`python scripts/build_code_appendix.py` ile tekrar üretilir "
        "(testler `tests/` altında ayrıca yer alır).\n\n"
        f"**Toplam: {len(present)} kaynak dosya, ~{total_lines} satır.**\n"
    )
    OUT.write_text(header + "".join(body), encoding="utf-8")
    print(f"Yazildi: {OUT}  ({len(present)} dosya, ~{total_lines} satir)")


if __name__ == "__main__":
    main()
