# MİMARİ HAFIZA — BIST 28 Portföy RL

> Bu dosya `mimari-hafiza-koruyucusu` ajanının **canlı kayıt defteridir**. Alt-ajanlar
> durumsuzdur; projenin mimari belleği burada yaşar. Her büyük değişiklikten sonra
> güncellenir. **Kaynak-doğruluk**: bu dosyadaki bir iddia kodla çelişiyorsa kod kazanır;
> çelişkiyi düzelt.

**Proje**: BIST 28 portföy-yönetimi RL · UYİK 2026 bildirisi / `RL_FinalProje.pdf` teslimi
**Kanonik kök**: `kod/` · **Son güncelleme**: 2026-06-22 (V12 N-seans modeli, self-financing muhasebe, checkpoint v2; **golden yeniden tabanlanacak**)

---

## 0. Blackboard Sözleşmesi (oryantasyon — ÖNCE BURAYI OKU)

> Bu dosya tek **paylaşılan-durum** kaynağıdır (blackboard). Alt-ajanlar durumsuzdur; kod-dokunan
> her ajan (`rl-arastirma-muhendisi`, `odul-ceza-tasarimcisi`, `backend-muhendisi`, `frontend-muhendisi`,
> `veri-muhendisi`, `test-muhendisi`) çalışmadan ÖNCE §2 (veri akışı/MDP) + §3 (invariant'lar) + §4
> (sözleşmeler) bölümlerini okur. Maddi bir değişiklik SONRASI ana oturum `mimari-hafiza-koruyucusu`'nu
> çağırır (yaz-sonra). Kod ⟂ doc çelişirse **KOD kazanır** (grep ile teyit, doc'u düzelt).

**Bölüm indeksi:** §1 Paket · §2 Veri Akışı · §3 İnvariant'lar · §4 Sözleşmeler · §5 Karar Günlüğü · §6 Riskler · §7 Drift Guard.

### MDP Sözleşmesi (özet — sayılar `config.py`'den türetilir, elle yazma)
- **STATE_DIM**: 397 (DQN/SAC/TD3) / 369 (PPO; forecast hariç) = 28 hisse × (12 feature [+1 forecast]) + 29 ağırlık + 4 makro.
- **ACTION**: 29-boyut softmax simpleks (PPO/SAC/TD3) / 6 şablon (DQN, ayrık).
- **REWARD**: `log(1+w·r_net) − η_t·‖Δw‖₁ − λ_t·max(0,DD−τ_t) − bankruptcy_penalty [+ w_dsr·DSR] [+ w_cvar·CVaR] [+ opt-in gain/ruin]`.
- **STEP**: `step_days=N`; ardışık N BIST seansının son kapanışı. Her adım karar ve rebalanstır; dönem-sonu ağırlıklar fiyat hareketiyle sürüklenir.
- **TIME SCALE**: nakit ve discount gerçek seans sayısıyla bileşir; lookback günleri `ceil(gün/N)` adıma çevrilir; metrikler tarihten yıllıklandırılır.

<!-- derived:config — guard: tests/test_config_single_source.py — ELLE DÜZENLEME (config.py'den türetilir) -->
seed = 42
n_features = 12
state_dim_dqn = 397
state_dim_ppo = 369
cash_annual_rate = 0.40
bankruptcy_penalty = 10.0
w_dsr = 0.05
w_cvar = 0.06
<!-- /derived:config -->

### Yaz-sonra tetikleri (ana oturum → `mimari-hafiza-koruyucusu`)
STATE_DIM/action/feature · reward terim/preset · contract anahtarı · golden re-baseline · yeni ajan/dosya ·
yeni invariant/risk değişti → bu dosyayı güncelle (+ §5 Karar Günlüğü satırı: tarih + karar + gerekçe + "golden: değişti/değişmedi").

## 7. Drift Guard Defteri (guard testleri)
- `tests/test_agent_roster.py` — ajan tanımı tutarlılığı (name↔dosya adı, README↔frontmatter model, CLAUDE.md routing kapsamı, en-az-ayrıcalık, örtüşme cross-ref).
- `tests/test_config_single_source.py` — yukarıdaki `derived:config` blok ⟂ `config.py`; STATE_DIM doc ⟂ config-hesabı.
- `tests/test_golden_regression.py` — davranış kilidi (≤1e-6, RNG sırası).

---

## 1. Paket Yapısı (üst seviye)

```
kod/
├── app.py              # Streamlit UI girişi (core.trainer/rollout tüketir)
├── main.py             # CLI orkestratörü: train.run() + plots.run()
├── train.py            # CLI eğitim + backtest sürücüsü
├── plots.py            # Yayın figürleri (matplotlib)
├── data.py             # BIST 28 yfinance indirme + parquet cache; resample_to_granularity()
├── config.py           # Merkezi hiperparametreler + HORIZON_PRESETS (TEK KAYNAK)
├── agents/             # RL algoritmaları (PyTorch): base, common, dqn, ppo, sac, td3
├── env/                # portfolio_env.py (MDP) + reward.py (RewardEngine, AdaptiveRewardShaper, DifferentialSharpe)
├── core/               # trainer.py, rollout.py, walkforward.py, factory.py, persistence.py, features.py
├── forecast/           # forecaster.py (CNN-LSTM bir-adım getiri tahmini, predict-then-optimize)
├── utils/              # features, metrics, baselines, deflated_sharpe, macro, stress_mc, portfolio_tl, torch_utils
├── ui/                 # Streamlit SRP katmanı: state, services, charts, sidebar, tabs/{mdp,train,test,compare}
├── tests/              # pytest (≈101 test) + golden/ (donmuş baseline'lar, ≤1e-6 gate)
├── scripts/            # rigor_analysis.py, build_code_appendix.py
├── results/            # CSV çıktıları (metrics, navs_aligned, weights_*, rigor_*)
└── figures/            # PNG figürler
```

## 2. Veri Akışı (uçtan uca)

```
data.py (yfinance → parquet)  →  utils/features.py (add_features + TrainScaler, train-only fit)
   →  forecast/forecaster.py ('forecast' feature, train-only)
   →  data.resample_to_step_days(df, step_days)  ← N ardışık seansın dönem sonu
   →  env/portfolio_env.py (MDP state; coarse'da window/mom/minvol cap'lenir)
   →  agents/* (DQN/PPO/SAC/TD3)  →  core/trainer.py (generator eğitim)
   →  core/rollout.py (deterministik eval)  →  utils/metrics.py
   →  plots.py (CLI figürler) / ui/* (interaktif)  →  results/ + figures/
```

**NOT:** UI ve CLI aynı `step_days` sözleşmesini ve aynı `core.trainer/rollout` çekirdeğini kullanır.

## 3. KIRILMAZ İnvariant'lar (her değişiklikte koru)

1. **Golden-master ≤1e-6** — `tests/golden/metrics_baseline.csv`, `navs_aligned_baseline.csv`.
   Eğitim/eval'de **RNG çağrı sırası** değişmemeli; aksi hâlde golden test kırılır.
   Kanonik (V11): SAC 2.141/5.595, DQN 0.484/1.348. 227 test yeşil (commit 12d62c3).
2. **Sızıntısızlık (leak-safety)** — tüm feature'lar causal (yalnız ≤t). `TrainScaler` ve
   `forecaster` **YALNIZ** train kümesinde fit edilir, test'e uygulanır ama orada fit edilmez.
   `tests/test_features.py`, `test_price_noise.py` bunu kilitler.
   Granülerlik sızıntısızlığı: `resample_to_granularity` feature hesabından SONRA çağrılır;
   resample atomik nokta üretir → train/test sınırı periyot-hizalı, karışma imkânsız.
3. **Tek-kaynak config** — tüm hiperparametre/preset `config.py` (özellikle `HORIZON_PRESETS`,
   `GRANULARITY_OPTIONS`, `MIN_POINTS`). Sihirli sayıları kodun içine gömme; config'e bağla.
4. **Tek komut UI** — `streamlit run app.py` tek başına çalışır.
5. **CLI ↔ UI ortak çekirdek** — `core/trainer.py` & `core/rollout.py` her ikisini de besler;
   mantık çatallanmamalı.
6. **Determinizm** — `seed=42`; `torch.manual_seed` + `np.random.seed`; CPU.
7. **add_features dokunulmazlığı** — granülerlik ne olursa olsun `add_features` HİÇ değişmez;
   feature'lar her zaman 2554-günlük tam seride hesaplanır → daily V11 bit-aynı, NaN imkânsız.
8. **Coarse env pencere üçlüsü cap** — `DiscretePortfolioEnv.__init__`'te `window`,
   `mom_window`, `minvol_window` ÜÇÜ AYNI ANDA `min(preset, max(1, n_days//3))` ile cap'lenir.
   YALNIZ `window` cap'lenip diğerleri cap'lenmezse `feat_tensor[t-minvol_window:t]` negatif
   index → boş slice → NaN ağırlık/NAV (DQN + coarse granülerlik). Regresyon:
   `tests/test_granularity.py::test_discrete_env_coarse_all_actions_nan_free`.
9. **episode_clean opt-in** — `EnvConfig.episode_clean=False` (default). Değiştirilirse
   golden ETKİLENMEZ (CLI/main.py bu flag'i kullanmaz); yalnız UI `_make_env(train)` açar.

## 4. Önemli Sözleşmeler (contracts)

- **`core.trainer.train_*()`** generator'dır; her adımda dict yield eder:
  `{"algo","iter","reward","nav","loss","success","actions","agent","env"}`. UI ve CLI bu
  sözleşmeye bağımlıdır — anahtar adlarını değiştirme.
- **`agent.act_eval(state)`** deterministik eylem döndürür (eval/rollout için).
- **`core.contracts.RunSpec`** çalışma kimliğidir; algo/adım/ödül/HP/tarih/feature/provenance içerir ve checkpoint/UI anahtarını belirler.
- **`core.contracts.BacktestResult`** net getiri, tarih, işlem öncesi/hedef/dönem-sonu ağırlık, turnover ve provenance taşır.
- **`config.HORIZON_PRESETS`** yalnız legacy checkpoint ve eski script uyumluluğu içindir; aktif UI/CLI kullanmaz.
- **`data.resample_to_granularity(df, granularity)`** — granularity ∈ {"Gün","Ay","Yıl"};
  Gün = NO-OP (aynı df), Ay = `resample("ME").mean()`, Yıl = `resample("YE").mean()`.
  ÇAĞRI SIRASI: `add_features` → `train_test_split` → `resample_to_granularity` (her split ayrı).
  `config.GRANULARITY_OPTIONS` tek kaynak; `config.MIN_POINTS` alt sınır koruması.
- **`factory.build_env(..., episode_clean=False)`** — `episode_clean` flag'ini env ctor'a
  iletir; default False → golden-güvenli. `_episode_idx` sıfırdan başlar; `_risky_returns`
  gürültüyü `not episode_clean or _episode_idx >= 1` koşuluyla ekler (ilk episode temiz).
- **`DiscretePortfolioEnv` pencere cap kuralı** — ctor'da `n = len(prices)`;
  `cap = min(preset, max(1, n//3))`; `self.window = self.mom_window = self.minvol_window = cap`
  (coarse'da). Daily'de cap > preset olamaz → davranış değişmez.

## 5. Kararlar Günlüğü (Decision Log)

> Yeni kararlar buraya tarih + gerekçe ile eklenir. (Ör. "TD3 eklendi — hoca tavsiyesi,
> sürekli kontrol; SAC ile kıyas için." / "PPO forecast feature almaz — v2 ablation kararı.")

- _(2026-06-22)_ V12: vade/granülerlik yerine tek N-seans dönem-sonu modeli; her adım rebalans, self-financing ağırlık sürüklenmesi, fold-yerel makro scaler, provenance ve checkpoint format v2. Kullanıcı onayıyla golden değişecek.

- _(seed)_ TD3 sürekli kontrol için eklendi (hoca tavsiyesi); eylem 29-softmax.
- _(seed)_ PPO forecast feature'ı dışlar (v2 ablation) → boyutu DQN/SAC/TD3'ten 28 düşük.
- _(seed)_ Rigor modülleri (DSR/PBO/CSCV/MC) sibling `Reinforcement Learning Final/` (PARS)
  referansından port ediliyor; PARS kanonik teslim DEĞİL, referans.
- _(2026-06-19)_ **Durum boyutu kanonik = 397 (DQN/SAC/TD3) / 369 (PPO)** — makro 4 feature
  (`MacroConfig.enabled`) dahil. 393/365 = makro-öncesi V5 tabanı. Docstring/config/README/
  DOKUMANTASYON hizalandı (yalnız metin; davranış değişmedi).
- _(2026-06-19)_ **Golden-master V8'e re-baseline edildi (kullanıcı onaylı).** V8 slippage
  terimi DQN'i baseline-altına itti (CAGR +%22.4 → −%15.5; FinalNAV 1.74 → 0.63).
  `metrics_baseline.csv` + `navs_aligned_baseline.csv` güncellendi; V7 arşivi
  `golden/v7_metrics_baseline.csv`'de korunuyor. 101 test yeşil.
- _(2026-06-19)_ §9.7 RL-pedagojik metrikleri UI Tab 2'ye eklendi; §8.3'e statik V8 sonuç
  tabloları gömüldü; §8.4'e "V7-referans, geçerli olan V8" notu eklendi.
- _(2026-06-19)_ **UI'da parametrik episode sayısı + kontrol edilebilir fiyat-gürültüsü σ.**
  `build_env`'e opsiyonel `price_noise_std` (None → `EnvConfig` default, golden-güvenli);
  sidebar'da `n_episodes` (1–1000) + σ slider; `train_generator(n_iters=n_episodes)`. Fiyat-
  gürültüsü mekanizması zaten vardı (env-yerel RNG, train-only, **her episode farklı
  realizasyon**) — bu değişiklik onu UI'dan parametrize/görünür kılar. 107 test yeşil.
- _(2026-06-19)_ **Tam parametrik ödül/ceza + train/test tarih seçimi.** (a) Gizli 6 ödül
  param'ı (`w_dsr, w_cvar, dsr_eta, cvar_alpha, regime_beta, cvar_amp`) `reward_overrides` →
  `build_env` → `RewardEngine` boyunca UI'a açıldı. (b) **2 yeni OPT-IN terim** (default KAPALI,
  yeni RNG yok → golden-güvenli): kazanç-çarpanı ödülü `w_gain·max(0,nav−gain_floor)·(1+
  w_gain_speed·(1−step_frac))` ve iflas-timing cezası `bankruptcy_penalty·(1+w_ruin_timing·
  (1−step_frac))`. (c) `config.DataConfig` + `data.py` tarih-parametrik (`download_bist(start,end)`,
  `train_test_split(split)`); UI'da train başlangıç/ayırım/bitiş seçici + sızıntı doğrulama.
  Yeni `odul-ceza-tasarimcisi` ajanı eklendi (16. ajan). **131 test yeşil; golden DEĞİŞMEDİ.**
  `terms` dict'e additive: `gain_bonus`, `ruin_timing_mult`.
- _(2026-06-19)_ **Akademik revizyon (A+B+C) — REPRODÜKSİYON DÜZELTMESİ (önemli).** Hakem
  eleştirisi üzerine: (1) Eski V8 golden bu makinede REPRODÜKLENMİYORDU — `main.py` seed=42'de
  PPO/SAC golden'ı birebir verdi (veri kanonik-eşdeğer) ama DQN patolojik kararsız (0.63/1.28/2.63).
  Golden **reprodüklenebilir `main.py` çıktısına** yeniden donduruldu (iki ardışık koşu max fark
  0.0 → GERÇEK reprodüksiyon; eski hali statik-CSV karşılaştırıyordu). Yeni kanonik: DQN 1.28,
  PPO 3.91, SAC 5.50, TD3 4.83. (2) **Çoklu-seed (5 seed)**: DQN Sharpe 0.46±0.22 (CV~%47,
  kararsız), SAC 2.09±0.02 (en stabil), TD3 2.05±0.12, PPO 1.71±0.05 → tek-seed yetersizliği
  nicel kanıt. (3) Genişletilmiş baseline: **MinVariance (Sharpe 2.30 / NAV 7.84) RL'i her
  metrikte geçiyor**. (4) `scripts/multiseed.py` (kanonik `train.py`'yi DOĞRUDAN çağırır →
  seed=42 = main.py bit-aynı) + `scripts/reward_sensitivity.py` eklendi. (5) Tier-A doc:
  Markov→"yaklaşık MDP", §2b Veri Metodolojisi & Survivorship, adalet kriteri, dürüst tez
  manşeti, V9→Ek B. **Dürüst tez (keskin):** RL (en iyi SAC) naif 1/N ile risk-ayarlıda başa
  baş ama klasik risk-bazlı optimize edicileri (MinVar) ne Sharpe ne NAV'da geçemez.
- _(2026-06-19)_ **Veri kurtarma:** Cache budanırsa (`data/prices.parquet`), `results/bist30_prices.csv`
  (tam 2015-2024) → parquet geri yüklenir (`pd.read_csv → to_parquet`). Golden bu tam veriden üretildi.
- _(2026-06-19)_ **Tier-C ek analizler (forecast eval + cost realism).** (1) `scripts/forecast_eval.py`:
  CNN-LSTM tahmincisi gerçek BIST'te öngörü gücü GÖSTERMİYOR (yön %47.8<%50, corr~0.003); düşük
  RMSE shrink-to-zero'dan (forecast≈zero baseline). Çıkarım: forecast feature bir düzenlileştirici
  giriş, öngörü kanalı değil (V4 ablation kazancı muhtemelen regularization; forecast-kapalı A/B
  önerilir). (2) `scripts/cost_sensitivity.py`: BIST maliyet modeli (komisyon 10-20bps + BSMV %5 +
  spread); **η (5-15bps tek-yön) GERÇEKÇİ**, kalibrasyon değişmez. SAC turnover 0.011 → maliyet-bağışık
  (drag@50bps %1.35); DQN çöker (Sharpe 0.42→−0.46), PPO/Momentum bozulur → SAC'ın düşük-turnover'ı
  gerçek net-edge. İkisi golden-güvenli (yeni script, RNG sırası değişmez). 159 test yeşil.
- _(2026-06-19)_ **V11 — Adil-karşılaştırma + nakit-faizi revizyonu (hakem #1–#8 + 5 yeni ister).**
  Golden 5. kez re-baseline; **191 test yeşil**; determinizm ≤1e-6. Sekiz adalet düzeltmesi:
  **C1** metrik tarih HİZALAMA + ortak-pencere `NAV[0]=1` yeniden-tabanlama (RL+baseline AYNI gün
  sayısı; `train.py`+`extra_baselines.py`) — FinalNAV adil pencere büyümesi; **C2** maliyetli
  baseline'lar (η=0.0010 simetrik; `utils/baselines.py`); **C3** başarı=`success_vs_benchmark` (EW),
  CLI'daki `nav>1` override kalktı; **C4** PPO eval deterministik (mu, sample değil); **C5**
  `--allow-synthetic` koruması (`main.py`); **C6** walk-forward'a macro/regime; **C7** reward
  `log(1+net)` (tx_cost çift-sayım giderildi); **C8** app caption TD3. Yeni isterler: **nakit faizi**
  (`EnvConfig.cash_annual_rate=0.40` PARAMETRİK; `config.cash_daily_rate()`; env nakit varlığı +
  `cash_riskfree` baseline gerçek faiz kazanır); **vade gün-limiti** (HORIZON_PRESETS
  min/max_days/train_max_steps: short 1/30, medium 30/90, long 90/360; eğitim episode=train_max_steps,
  eval=tam test dönemi); **model liste UI** (`{algo}_{horizon}_{adaptive}.pt` + sidebar selectbox;
  CLI save da `model_path` kullanır); **metrics NaN-guard** (sıfır-vol/sıfır-DD → Sharpe/Calmar NaN,
  yalnız CashRiskFree). **KİLİT BULGU:** adalet düzeltmeleri RL'i GÜÇLENDİRDİ (önce RL maliyet öderken
  baseline ödemiyordu + RL nakiti %0 kazanıyordu = haksız ceza). Yeni adil kanonik: SAC 2.141 / TD3
  2.151 Sharpe → InverseVol(2.16)/RiskParity(2.14) düzeyinde, EqualWeight(2.09) + risksiz hurdle
  (NAV 2.5x) üstünde; MinVar(2.32) hâlâ önde ama makas kapandı. DQN zayıf (0.484). Altın/dolar
  tradeable YAPILMADI (makro-feature kaldı — kullanıcı kararı).

- _(2026-06-21)_ **Adım granülerliği (Gün/Ay/Yıl) — UI-only resample (Approach 1).**
  `add_features` hiç değişmez (2554-günlük tam seri); ardından `data.resample_to_granularity`
  seçilen granülerliğe resample eder (Ay=`ME`, Yıl=`YE`, Gün=NO-OP). Neden Approach 1: daily
  V11 bit-aynı (golden korunur) + NaN imkânsız + resample atomik nokta → sızıntısız split.
  Yalnız UI yolu (`ui/services._load_data`); CLI/golden etkilenmez. `config.GRANULARITY_OPTIONS`
  / `MIN_POINTS` tek kaynak.
- _(2026-06-21)_ **Coarse env pencere üçlüsü cap — adversarial gotcha yakalandı.**
  Ay (~120) / Yıl (~10) granülerlikte `window`, `mom_window`, `minvol_window` ÜÇÜ DE
  `min(preset, max(1, n//3))` ile cap'lenir. YALNIZ `window` cap'lenirse `feat_tensor
  [t-minvol_window:t]` negatif index → boş slice → NaN ağırlık/NAV (DQN + ay/yıl +
  medium/long). Daily'de cap etkisiz (window=20, minvol=60, mom=20; cap=851). Regresyon:
  `tests/test_granularity.py::test_discrete_env_coarse_all_actions_nan_free`.
- _(2026-06-21)_ **episode_clean opt-in (golden-güvenli deseni).**
  `EnvConfig.episode_clean=False` default; `_episode_idx` reset'te artar; gürültü
  `not episode_clean or _episode_idx>=1` koşulunda eklenir → ilk episode orijinal/temiz,
  2+ gürültülü. CLI/golden bu flag'i kullanmaz → V11 bit-aynı. UI `_make_env(train)`
  `episode_clean=True` açar. Ödül opt-in deseniyle aynı felsefe.

- _(2026-06-22)_ **Ajan sistemi kapsamlı yeniden-yapı (kontrol / roster / veri düzlemi).**
  (a) **Kontrol**: `CLAUDE.md` deterministik yönlendirme cascade'i (yol-sahipliği + C1–C4 tie-breaker),
  öncelik kafesi (P0 invariant > P1 rubrik > P2 RL > P3 istatistik > P4 finans > P5 UI; `test-muhendisi`
  SERT-DUR, `proje-rubrik-bekcisi` DANIŞMAN-VETO), R1–R9 refleks yaşam döngüsü; orkestratör dağıtım-planı şeması.
  (b) **Roster 16→17**: yeni `dagitim-tekrarlanabilirlik-uzmani` (HF Space deploy/repro; "Space GERÇEK veriyle
  çalışır, sentetik-GBM NaN'e DÜŞMEZ" invariant'ı). Reward dikişi (`env/reward.py`→ödül-ceza; `agents/*.py`→rl),
  forecaster ikili-sahip çözümü (`forecast/forecaster.py`→veri; politika-girdisi→rl), finans 4-lens keskinleştirme
  (birleştirme REDDEDİLDİ), script sahipliği (analitik→kantitatif, çalıştırma→backend); her ajana OWNS/DEĞİL/DEVRET.
  (c) **Veri düzlemi**: bu dosya §0 blackboard + §7 guard defteri + `derived:config` bloğuna kavuştu. Yeni guard:
  `tests/test_agent_roster.py` (5) + `tests/test_config_single_source.py` (4) = 9 assertion, hepsi yeşil.
  **Golden DEĞİŞMEDİ** — yalnız `.md`/`docs`/`tests` (runtime/RNG'ye dokunulmadı). Tetik: ekteki ChatGPT "framework"
  analizinin (kategori hatası + MARL halüsinasyonu) reddi → platform-doğru, akademik-kapsam-disiplinli yeniden-yapı.

## 6. Bilinen Riskler / Açık Konular

- Ortam pyarrow/yfinance'a bağlı; bazı kabuklarda eksik olabilir (golden env-kilitli).
- Golden artık **V11 referansı** (adil-karşılaştırma + nakit-faizi; 2026-06-19'da bu ortamda
  donduruldu, ≤1e-6 reprodüklenebilir); nöral satırlar torch/numpy sürümüne duyarlı — başka
  ortamda kayarsa kanonik ortamda yeniden dondur.
- **Nakit faizi varsayımı (V11):** `cash_annual_rate=0.40` parametrik; bu, evalüasyonda nakit
  hurdle'ını (NAV ~2.5x) belirler. Sharpe NOMİNAL'dir (rf=0); rf-hurdle CashRiskFree satırında
  ayrıca gösterilir. Oran UI/config'ten değiştirilebilir → sonuçlar buna duyarlıdır.
- `KOZAA.IS`, `KOZAL.IS` evrenden hariç (28 hisse).
