# CLAUDE.md — BIST 28 Portföy Yönetimi RL

Bu dosya ana Claude oturumunun proje konvansiyonlarını ve **ajan yönlendirme** kurallarını
içerir. Detaylı ajan roster'ı: [`.claude/agents/README.md`](.claude/agents/README.md).
Canlı mimari bellek: [`docs/MIMARI_HAFIZA.md`](docs/MIMARI_HAFIZA.md).

## Proje özeti
- **Ne**: BIST 28 hissesi (KOZAA/KOZAL hariç, 2015–2024) üzerinde portföy-yönetimi RL.
  DQN/PPO/SAC/TD3 (saf PyTorch, CPU). UYİK 2026 bildirisi / `RL_FinalProje.pdf` teslimi.
- **Ana deliverable**: interaktif Streamlit demo (`streamlit run app.py`) + CLI pipeline (`main.py`).
- **Ödül**: `log(1+w·r) − η_t·‖Δw‖₁ − λ_t·max(0,DD−τ_t)` (+ ops. DSR/CVaR); vade preset'leri Kısa/Orta/Uzun.

## KIRILMAZ invariant'lar (her değişiklikte koru)
1. **Golden-master ≤1e-6** (`tests/golden/`) — eğitim/eval'de **RNG çağrı sırasını koru**.
2. **Sızıntısızlık** — feature'lar causal (≤t); `TrainScaler`/forecaster **yalnız train**'de fit.
3. **Tek-kaynak config** — hiperparametre/preset `config.py` (`HORIZON_PRESETS`).
4. **Tek komut UI** — `streamlit run app.py` tek başına çalışır.
5. **CLI ↔ UI ortak çekirdek** — `core/trainer.py` & `core/rollout.py`; mantık çatallanmaz.
6. **Determinizm** — `seed=42`, CPU.

## Çalıştırma (Windows / PowerShell)
- Python: `.venv\Scripts\python.exe` (sanal ortam). `$env:PYTHONUTF8=1` UTF-8 için.
- Testler: `.venv\Scripts\python.exe -m pytest -q` · kapsam: `--cov=. --cov-report=xml`
- CLI: `.venv\Scripts\python.exe main.py [--skip-data|--skip-train|--skip-plots|--walkforward]`
- Rigor: `.venv\Scripts\python.exe scripts\rigor_analysis.py`
- gh/git veya çok-satırlı komutlar için **Bash tool**'u tercih et (PowerShell 5.1 gömülü
  tırnakları bozabilir).

## Ajan Yönlendirme (routing)
Bir görevi şu uzmana yönlendir (`@ajan-adi` veya otomatik delegasyon):

| Görev türü | Ajan |
|-----------|------|
| DQN/PPO/SAC/TD3, ödül/state matematiği, eğitim kararlılığı | `rl-arastirma-muhendisi` |
| Ödül/ceza tasarımı, magnitüd/denge, yeni reward terim (2x/3x bonus, iflas-timing) | `odul-ceza-tasarimcisi` |
| `core/` eğitim çekirdeği, pipeline, config kablolama, persistence | `backend-muhendisi` |
| Streamlit `app.py`/`ui/`/`plots.py`, sekmeler, grafikler | `frontend-muhendisi` |
| `data.py`/feature/makro, sızıntısızlık | `veri-muhendisi` |
| pytest, golden-master, leak-safety, determinizm | `test-muhendisi` |
| Kapsam, kabul kriteri, kullanıcı akışı, önceliklendirme | `urun-yoneticisi` |
| Mandat/ister değerlendirme, baseline kıyas | `fon-yoneticisi` |
| BIST komisyon/BSMV/takas/lot, işlem-maliyeti gerçekliği | `finansal-regulasyon-uzmani` |
| Finansal "neden", makro rejim, terim doğruluğu | `finans-uzmani` |
| DSR/PBO/CSCV, Monte-Carlo, walk-forward, VaR/CVaR | `kantitatif-strateji-uzmani` |
| Rapor §9.1–§9.9 akademik kalite, literatür | `akademisyen-degerlendirici` |
| `RL_FinalProje.pdf` ister-uyum denetimi | `proje-rubrik-bekcisi` |
| Mimari bellek (`docs/MIMARI_HAFIZA.md`) | `mimari-hafiza-koruyucusu` |
| README/DOKUMANTASYON/KOD_EKI üretim & senkron | `dokumantasyon-yazari` |
| Büyük/çok-disiplinli işi parçalara böl & dağıt | `orkestrator-planlayici` |

### Refleksler
- Kaynak kod değişti → **`test-muhendisi`** (ilgili testler + golden).
- Büyük mimari değişiklik/karar → **`mimari-hafiza-koruyucusu`**.
- Teslim öncesi → **`proje-rubrik-bekcisi`** + **`akademisyen-degerlendirici`**.
- Kod değişikliği dokümana yansıdı → **`dokumantasyon-yazari`**.

### Kapasite notu
Alt-ajanlar başka alt-ajan çağıramaz ve durumsuzdur. `orkestrator-planlayici` bir **dağıtım
planı** üretir; **bu planı ana oturum (sen) uygular** — ajanları sırayla/uygunsa paralel çağırarak.

## Sınırlar
- Mevcut kaynak kod/doküman gereksiz yere değiştirilmez; `kod/` kanonik teslimdir.
- `sibling Reinforcement Learning Final/` (PARS) bir **referanstır**, teslim değildir;
  oradan rigor tekniği ödünç alınır ama `kod/`'a uyarlanır.
- Commit/push yalnızca kullanıcı onayıyla.
