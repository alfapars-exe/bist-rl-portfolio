"""Guard: ajan tanımı (`.claude/agents/*.md`) tutarlılığı — sapmayı CI'da yakalar.

Kapsam: name↔dosya adı, README roster↔frontmatter (isim seti + model), CLAUDE.md routing
kapsamı, en-az-ayrıcalık (salt-değerlendiricilerde Edit/Bash yok), bilinen örtüşme çiftlerinde
karşılıklı sınır referansı. Saf markdown-parse; RNG/golden/ağır-bağımlılık YOK → her zaman yeşil olmalı.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / ".claude" / "agents"
README = AGENTS_DIR / "README.md"
CLAUDE_MD = ROOT / "CLAUDE.md"

# Salt-değerlendiriciler: kanonik teslimi korumak için Edit/Bash ALMAZLAR (Write opsiyonel).
EVALUATORS = {"proje-rubrik-bekcisi", "akademisyen-degerlendirici", "fon-yoneticisi", "finans-uzmani"}

# Bilinen bitişik çiftler: en az BİR yön diğerini adıyla anmalı (sınır kabulü / örtüşme guard'ı).
OVERLAP_PAIRS = [
    ("rl-arastirma-muhendisi", "odul-ceza-tasarimcisi"),   # reward dikişi (C1)
    ("veri-muhendisi", "rl-arastirma-muhendisi"),          # forecaster sahipliği
    ("finans-uzmani", "finansal-regulasyon-uzmani"),       # finans lens ayrımı (C3)
]


def _agent_files():
    return sorted(p for p in AGENTS_DIR.glob("*.md") if p.name != "README.md")


def _frontmatter_block(text: str) -> str:
    assert text.startswith("---"), "frontmatter '---' ile başlamalı"
    end = text.index("\n---", 3)
    return text[4:end]  # ilk '---\n' sonrası, kapanış '---' öncesi


def _parse_fm(block: str) -> dict:
    fm = {}
    for line in block.split("\n"):
        if line and not line[0].isspace():
            m = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
            if m:
                fm[m.group(1)] = m.group(2).strip()
    return fm


def _load_agents() -> dict:
    out = {}
    for p in _agent_files():
        block = _frontmatter_block(p.read_text(encoding="utf-8"))
        out[p.stem] = {"fm": _parse_fm(block), "block": block}
    return out


def test_name_equals_filename():
    for stem, a in _load_agents().items():
        assert a["fm"].get("name") == stem, f"{stem}: frontmatter name dosya adına eşit değil"


def test_readme_roster_matches_frontmatter():
    agents = _load_agents()
    roster = {}
    for line in README.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^\|\s*\*\*([a-z0-9-]+)\*\*\s*\|.*\|\s*(opus|sonnet)\s*\|\s*$", line)
        if m:
            roster[m.group(1)] = m.group(2)
    assert set(roster) == set(agents), f"README roster ≠ dosya seti: {set(roster) ^ set(agents)}"
    for name, model in roster.items():
        assert agents[name]["fm"].get("model") == model, (
            f"{name}: model README({model}) ≠ frontmatter({agents[name]['fm'].get('model')})"
        )


def test_claude_md_routing_covers_all_agents():
    txt = CLAUDE_MD.read_text(encoding="utf-8")
    for stem in _load_agents():
        assert stem in txt, f"{stem} CLAUDE.md yönlendirmesinde geçmiyor"


def test_least_privilege_evaluators():
    agents = _load_agents()
    for ev in EVALUATORS:
        assert ev in agents, f"değerlendirici {ev} yok"
        tools = [t.strip() for t in agents[ev]["fm"].get("tools", "").split(",")]
        assert "Edit" not in tools, f"{ev}: salt-değerlendirici Edit almamalı (least-privilege)"
        assert "Bash" not in tools, f"{ev}: salt-değerlendirici Bash almamalı (least-privilege)"


def test_known_overlap_pairs_cross_reference():
    agents = _load_agents()
    for a, b in OVERLAP_PAIRS:
        assert a in agents and b in agents, f"örtüşme çifti eksik: {a},{b}"
        refs_each_other = (b in agents[a]["block"]) or (a in agents[b]["block"])
        assert refs_each_other, f"örtüşme çifti ({a},{b}) hiçbir yönde sınır referansı taşımıyor"
