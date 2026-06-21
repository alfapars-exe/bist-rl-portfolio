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

## Ajan Yönlendirme (routing) — karar prosedürü
Ana oturum bir görevi **tek bir birincil ajana** atar. Kuralları **yukarıdan aşağı** oku;
**ilk eşleşen kazanır** (sıra = tie-breaker). Birincil ajan teslim eder; "danışman" ajanlar
yalnız görüş verir, sahiplenmez. Elle çağrı: `@ajan-adi`.

### 1) Dosya-yolu sahipliği (yol → tek sahip; konu sezgisinden ÖNCE gelir)
| Yol deseni | Sahip |
|-----------|-------|
| `agents/*.py`, ödülün algoritma-içi kullanımı (GAE/loss/normalize/clip) | `rl-arastirma-muhendisi` |
| `env/reward.py` terim yapısı/büyüklüğü/yeni shaping, `RewardConfig` ödül-tarafı | `odul-ceza-tasarimcisi` |
| `core/*`, `main.py`, `train.py`, `config.py` kablolama, persistence, walkforward; `scripts/` çalıştırma | `backend-muhendisi` |
| `data.py`, `utils/features.py`, `utils/macro.py`, `forecast/forecaster.py` (tümü) | `veri-muhendisi` |
| `app.py`, `ui/**`, `plots.py` | `frontend-muhendisi` |
| `tests/**`, `tests/golden/**`, `.github/**`, `sonar-project.properties`, `coverage.xml` | `test-muhendisi` |
| `utils/deflated_sharpe.py`, `utils/stress_mc.py`, `scripts/{rigor_analysis,multiseed,*_sensitivity,extra_baselines,forecast_eval}.py` (analitik içerik) | `kantitatif-strateji-uzmani` |
| `utils/portfolio_tl.py`, η maliyet kalibrasyonu | `finansal-regulasyon-uzmani` |
| `hf_deploy.py`, Dockerfile/Space metadata, deploy reprodüksiyonu | `dagitim-tekrarlanabilirlik-uzmani` |
| `docs/MIMARI_HAFIZA.md` | `mimari-hafiza-koruyucusu` |
| `README.md`, `DOKUMANTASYON.md`, `KOD_EKI.md` | `dokumantasyon-yazari` |

### 2) Konu sinyali (yol net değilse)
| Görev türü | Ajan |
|-----------|------|
| DQN/PPO/SAC/TD3 matematiği, state tasarımı, eğitim kararlılığı | `rl-arastirma-muhendisi` |
| Ödül/ceza tasarımı, magnitüd/denge, yeni reward terim (2x/3x, iflas-timing; opt-in) | `odul-ceza-tasarimcisi` |
| Eğitim çekirdeği/pipeline/config kablolama/persistence; bağımlılık/`.venv`/`requirements` | `backend-muhendisi` |
| Streamlit sekme/grafik/widget | `frontend-muhendisi` |
| Feature/makro/forecaster veri tarafı, **sızıntısızlık** | `veri-muhendisi` |
| pytest/golden/leak/determinizm/coverage/CI | `test-muhendisi` |
| Kapsam, kabul kriteri, kullanıcı akışı, önceliklendirme | `urun-yoneticisi` |
| Mandat/ister, baseline kıyas, "yatırılabilir mi" | `fon-yoneticisi` |
| BIST komisyon/BSMV/takas/lot, işlem-maliyeti gerçekliği | `finansal-regulasyon-uzmani` |
| Finansal "neden", makro rejim, terim doğruluğu | `finans-uzmani` |
| DSR/PBO/CSCV, Monte-Carlo, walk-forward, VaR/CVaR | `kantitatif-strateji-uzmani` |
| Rapor §9.1–§9.9 akademik kalite, literatür | `akademisyen-degerlendirici` |
| `RL_FinalProje.pdf` ister-uyum denetimi | `proje-rubrik-bekcisi` |
| HF Space deploy / Docker / Space verisi / token hijyeni | `dagitim-tekrarlanabilirlik-uzmani` |
| Mimari bellek (`docs/MIMARI_HAFIZA.md`) | `mimari-hafiza-koruyucusu` |
| README/DOKUMANTASYON/KOD_EKI üretim & senkron | `dokumantasyon-yazari` |
| Büyük/çok-disiplinli işi parçalara böl & dağıt | `orkestrator-planlayici` |

### 3) Örtüşme bölgesi tie-breaker'ları (yalnız bunlar belirsiz)
- **C1 — Ödül işi (`rl-arastirma-muhendisi` vs `odul-ceza-tasarimcisi`):**
  "Ajan NEYİ maksimize etmeli?" (terim ekle/çıkar, büyüklük/denge) → `odul-ceza-tasarimcisi`.
  "Ajan bunu nasıl ÖĞRENİR?" (GAE/loss/normalize/clip, gradyan) → `rl-arastirma-muhendisi`.
  İkisi gerekiyorsa: ödül-ceza ÖNCE tasarlar (opt-in, default KAPALI), rl-araştırma SONRA doğrular.
- **C2 — Config/pipeline (`backend` vs `veri`):** veri/feature ÜRETİMİ veya causal doku →
  `veri-muhendisi`; orkestrasyon/kablolama/persistence/CLI/preset-bağlama → `backend-muhendisi`.
  Yeni feature state boyutunu değiştiriyorsa: `veri` birincil, `backend`+`rl` zorunlu danışman.
- **C3 — "Bu sonuç iyi mi?" (4 lens):** karışıksa bu sırayla SOR:
  ① "İstatistiksel GERÇEK mi / overfit mi?" → `kantitatif-strateji-uzmani` *(önce: gerçek değilse gerisi anlamsız)* ·
  ② "İşlem maliyeti GERÇEKÇİ mi?" → `finansal-regulasyon-uzmani` ·
  ③ "Finansal ANLAMLI mı?" (makro/nominal-reel/terim) → `finans-uzmani` ·
  ④ "Bir FON yatırır mı?" (mandat/risk/baseline değeri) → `fon-yoneticisi` *(üst sentez; varsayılan tek-atama)*.
- **C4 — "Rapor uygun mu?":** "Zorunlu ödev maddesi VAR mı/eksik mi?" → `proje-rubrik-bekcisi` (önce);
  "İçerik akademik GÜÇLÜ/DÜRÜST mü?" → `akademisyen-degerlendirici`. Teslim öncesi İKİSİ (önce rubrik).

## Öncelik Kafesi (çatışma çözümü)
İki ajan çeliştiğinde ana oturum **ORTALAMA ALMAZ**; şu sıralı kafesi uygular — üst kat alt katı EZER,
alt kat üst katı asla İHLAL EDEMEZ:

`P0 INVARIANT` (golden ≤1e-6 · sızıntısızlık · determinizm · tek-config · tek-UI · CLI↔UI ortak çekirdek)
`> P1 Teslim uyumu` (PDF zorunlu maddeleri) `> P2 RL doğruluğu` `> P3 İstatistiksel sağlamlık`
`> P4 Finansal gerçeklik` `> P5 UI/doküman/sunum cilası`.

- **Kapı bekçisi — SERT DUR:** yalnız `test-muhendisi`, yalnız P0. Golden/leak/determinizm kırılıyorsa
  iş **İLERLEMEZ**; çözüm yolları R2'de (sınıflandır → düzelt/onayla). Onaysız hiçbir değişiklik P0'ı geçemez.
- **Kapı bekçisi — DANIŞMAN VETO:** `proje-rubrik-bekcisi` (P1). "Bu teslimi düşürür" diyebilir; iş durmaz
  ama itiraz çözülmeden teslim edilmez. Geçersiz kılınan her veto → `docs/MIMARI_HAFIZA.md` Karar Günlüğü'ne satır.

## Refleks Yaşam Döngüsü (olay → ajan → kabul kriteri)
Refleksleri **ANA OTURUM** tetikler (otonom ajan döngüsü YOK). Kabul kriteri sağlanmadan sonraki adıma GEÇME.

| # | Olay | Ajan | Kabul kriteri |
|---|------|------|---------------|
| R1 | `agents/`,`core/`,`env/`,`forecast/` kodu değişti | `test-muhendisi` | golden ≤1e-6 yeşil + ilgili testler. Kırıldıysa → R2. |
| R2 | Golden KIRILDI | `test-muhendisi` (+kaynak ajan) | Sınıflandır: (a) gerçek regresyon→düzelt; (b) RNG sırası kazası→kaynak düzeltir; (c) onaylı değişiklik→KULLANICI onayı + re-baseline + Karar Günlüğü. Sınıflandırmadan İLERLEME. |
| R3 | Ödül terimi/büyüklüğü değişti | `odul-ceza`→`rl-arastirma`→`test` | Yeni terim opt-in & default KAPALI → kapalıyken golden yeşil; yeni RNG YOK. |
| R4 | Veri/feature/causal doku değişti | `veri`→`test` | leak testleri geçti; state boyutu değiştiyse `backend`+`rl` danışman çağrıldı. |
| R5 | Büyük mimari karar/değişiklik | `mimari-hafiza-koruyucusu` | `MIMARI_HAFIZA.md` Karar Günlüğü'ne satır; bellek↔kod çelişkisi yok. |
| R6 | Kod/davranış dokümana yansıdı | `dokumantasyon-yazari` | README/DOC/KOD_EKI senkron; her sayı kaynaktan teyitli. |
| R7 | `results/` metrikleri yenilendi | C3 lensi (varsayılan `fon`) | İddialar `results/`'a dayanıyor; rigor şüphesi→`kantitatif` çapraz doğrular. |
| R8 | TESLİM ÖNCESİ | `proje-rubrik`→`akademisyen`→`test` (tam pytest+golden) | Rubrik ✅; akademik dürüstlük geçti; `pytest -q`+golden yeşil; commit yalnız onayla. |
| R9 | HF Space'e deploy | `dagitim-tekrarlanabilirlik-uzmani`→`test` | Space GERÇEK veriyle çalışır (sentetik-NaN'e düşmez); token sızmaz. |

### Kapasite notu
Alt-ajanlar başka alt-ajan çağıramaz ve durumsuzdur (her çağrı temiz bağlam; paylaşılan durum yalnız
repo dosyalarında — `config.py`/`docs/MIMARI_HAFIZA.md`/`CLAUDE.md`). `orkestrator-planlayici` bir
**dağıtım planı** üretir; **bu planı ana oturum (sen) uygular** — ajanları sırayla/uygunsa paralel çağırarak.
Roster **17 ajan** (`.claude/agents/README.md`). En-az-ayrıcalık: salt-değerlendiricilere `Edit`/`Bash` verilmez.

## Sınırlar
- Mevcut kaynak kod/doküman gereksiz yere değiştirilmez; `kod/` kanonik teslimdir.
- `sibling Reinforcement Learning Final/` (PARS) bir **referanstır**, teslim değildir;
  oradan rigor tekniği ödünç alınır ama `kod/`'a uyarlanır.
- Commit/push yalnızca kullanıcı onayıyla.
