"""HF Space deploy — projeyi argeinfina/rlfinal Space'ine yukler (Docker SDK).

NEDEN BU SCRIPT: Streamlit artik HF'de bagimsiz SDK degil (gradio|docker|static);
bu yuzden Docker SDK + Dockerfile ile streamlit'i 7860 portunda kosuyoruz.

KULLANIM (kendi terminalinde, kod/ klasorunde):
  1) Token sagla (ikisinden biri):
       huggingface-cli login          # token'i yapistir   (kalici)
     VEYA  PowerShell:  $env:HF_TOKEN = "hf_..."           # tek seferlik
     VEYA  Git Bash:    export HF_TOKEN="hf_..."
  2) Calistir:
       .venv\\Scripts\\python.exe hf_deploy.py

NOT: Token'i BU DOSYAYA YAZMA. Yukleme bitince HF token'ini revoke edip yenile
(sohbete dusuk metin yapistirildigi icin aciga cikti).
"""
import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "argeinfina/rlfinal"
token = os.environ.get("HF_TOKEN")  # None -> 'huggingface-cli login' cache'i kullanilir

api = HfApi(token=token)
print("Auth:", api.whoami().get("name"))

create_repo(REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True, token=token)
print("Space hazir (docker):", REPO_ID)

# 1) Kod + veri + modeller + dokuman (junk haric)
api.upload_folder(
    folder_path=".", repo_id=REPO_ID, repo_type="space",
    commit_message="Upload BIST 28 RL project (code+data+models+docs)",
    ignore_patterns=[".git/*", "*/.git/*", ".venv/*", "*.pyc", "*__pycache__*",
                     ".claude/*", "Reinforcement Learning Final/*", "README.md",
                     "Dockerfile", ".github/*", "*.code-workspace", "*.lnk",
                     "hf_deploy.py"],
)
print("Kod/dokuman yuklendi.")

# KRITIK: .gitignore data/*.parquet + models/ -> upload_folder (huggingface_hub 1.20+
# .gitignore-aware) bunlari ATLAR. Demo'nun GERCEK BIST verisiyle calismasi (yoksa HF'de
# ag erisimi olmadigindan download_bist/download_macro SENTETIK GBM'e duser -> NaN) icin
# fiyat + makro cache'ini ve egitilmis modelleri ACIKCA yukle (gitignore bypass).
import glob as _glob  # noqa: E402
_must = ["data/prices.parquet", "data/macro_raw.parquet"] + sorted(_glob.glob("models/*.pt"))
for _f in _must:
    if os.path.exists(_f):
        api.upload_file(path_or_fileobj=_f, path_in_repo=_f, repo_id=REPO_ID,
                        repo_type="space", commit_message=f"Upload {_f} (gitignore bypass)")
        print(f"  + {_f}")
    else:
        print(f"  ! ATLANDI (yok): {_f}")
print("Gercek veri (prices+macro) + modeller yuklendi (gitignore bypass).")

# 2) Space README (Docker metadata)
README = """---
title: BIST 28 Portfoy RL
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# BIST 28 Portfoy Yonetimi - Pekistirmeli Ogrenme (RL)

DQN/PPO/SAC/TD3 ile BIST 28 hissesi uzerinde portfoy-yonetimi RL demosu (UYIK 2026).
Streamlit arayuzu (Docker SDK): canli egitim, adim-adim test, coklu-ajan karsilastirma;
parametrik odul/ceza, vade preset'leri, train/test tarih secimi, parametrik episode + gurultu.

Tam dokumantasyon: `DOKUMANTASYON.md` | Kod eki: `KOD_EKI.md`

> Sidebar'dan 'Veriyi Yukle' -> Egitim / Test / Karsilastirma. yfinance erisilemezse
> sentetik veri ile calisir. Egitilmis modeller (models/*.pt) yuklu.
"""
api.upload_file(path_or_fileobj=README.encode("utf-8"), path_in_repo="README.md",
                repo_id=REPO_ID, repo_type="space", commit_message="Space README (docker)")
print("README yuklendi.")

# 3) Dockerfile — streamlit'i 7860'te kosar (CPU torch)
DOCKERFILE = """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \\
    && pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
RUN chmod -R 777 /app
ENV HOME=/app MPLCONFIGDIR=/tmp/mpl STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 7860
CMD ["streamlit","run","app.py","--server.port=7860","--server.address=0.0.0.0","--server.headless=true"]
"""
api.upload_file(path_or_fileobj=DOCKERFILE.encode("utf-8"), path_in_repo="Dockerfile",
                repo_id=REPO_ID, repo_type="space", commit_message="Dockerfile (streamlit on 7860)")
print("Dockerfile yuklendi.")
print("BITTI -> https://huggingface.co/spaces/" + REPO_ID)
