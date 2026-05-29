"""Pytest yapilandirmasi — proje kokunu sys.path'e ekler.

Proje flat-layout (moduller kok dizinde). Henuz `pip install -e .` yapilmamis
olabilecegi icin, testlerin `from utils.features import ...` gibi importlari
calismasi adina kok dizini path'e ekliyoruz. Bu, main/train/app/plots'taki
mevcut `sys.path.insert(0, HERE)` desenini yansitir.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
