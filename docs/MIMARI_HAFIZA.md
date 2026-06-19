# MİMARİ HAFIZA — BIST 28 Portföy RL

> Bu dosya `mimari-hafiza-koruyucusu` ajanının **canlı kayıt defteridir**. Alt-ajanlar
> durumsuzdur; projenin mimari belleği burada yaşar. Her büyük değişiklikten sonra
> güncellenir. **Kaynak-doğruluk**: bu dosyadaki bir iddia kodla çelişiyorsa kod kazanır;
> çelişkiyi düzelt.

**Proje**: BIST 28 portföy-yönetimi RL · UYİK 2026 bildirisi / `RL_FinalProje.pdf` teslimi
**Kanonik kök**: `kod/` · **Son güncelleme**: 2026-06-19 (akademik revizyon: reprodüklenebilir golden + multi-seed + genişletilmiş baseline)

---

## 1. Paket Yapısı (üst seviye)

```
kod/
├── app.py              # Streamlit UI girişi (core.trainer/rollout tüketir)
├── main.py             # CLI orkestratörü: train.run() + plots.run()
├── train.py            # CLI eğitim + backtest sürücüsü
├── plots.py            # Yayın figürleri (matplotlib)
├── data.py             # BIST 28 yfinance indirme + parquet cache
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
   →  forecast/forecaster.py ('forecast' feature, train-only)  →  env/portfolio_env.py (MDP state)
   →  agents/* (DQN/PPO/SAC/TD3)  →  core/trainer.py (generator eğitim)
   →  core/rollout.py (deterministik eval)  →  utils/metrics.py
   →  plots.py (CLI figürler) / ui/* (interaktif)  →  results/ + figures/
```

## 3. KIRILMAZ İnvariant'lar (her değişiklikte koru)

1. **Golden-master ≤1e-6** — `tests/golden/metrics_baseline.csv`, `navs_aligned_baseline.csv`.
   Eğitim/eval'de **RNG çağrı sırası** değişmemeli; aksi hâlde golden test kırılır.
2. **Sızıntısızlık (leak-safety)** — tüm feature'lar causal (yalnız ≤t). `TrainScaler` ve
   `forecaster` **YALNIZ** train kümesinde fit edilir, test'e uygulanır ama orada fit edilmez.
   `tests/test_features.py`, `test_price_noise.py` bunu kilitler.
3. **Tek-kaynak config** — tüm hiperparametre/preset `config.py` (özellikle `HORIZON_PRESETS`).
   Sihirli sayıları kodun içine gömme; config'e bağla.
4. **Tek komut UI** — `streamlit run app.py` tek başına çalışır.
5. **CLI ↔ UI ortak çekirdek** — `core/trainer.py` & `core/rollout.py` her ikisini de besler;
   mantık çatallanmamalı.
6. **Determinizm** — `seed=42`; `torch.manual_seed` + `np.random.seed`; CPU.

## 4. Önemli Sözleşmeler (contracts)

- **`core.trainer.train_*()`** generator'dır; her adımda dict yield eder:
  `{"algo","iter","reward","nav","loss","success","actions","agent","env"}`. UI ve CLI bu
  sözleşmeye bağımlıdır — anahtar adlarını değiştirme.
- **`agent.act_eval(state)`** deterministik eylem döndürür (eval/rollout için).
- **`config.HORIZON_PRESETS`** = {short, medium, long} → (η, λ, τ, γ, rebalans, pencereler).

## 5. Kararlar Günlüğü (Decision Log)

> Yeni kararlar buraya tarih + gerekçe ile eklenir. (Ör. "TD3 eklendi — hoca tavsiyesi,
> sürekli kontrol; SAC ile kıyas için." / "PPO forecast feature almaz — v2 ablation kararı.")

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

## 6. Bilinen Riskler / Açık Konular

- Ortam pyarrow/yfinance'a bağlı; bazı kabuklarda eksik olabilir (golden env-kilitli).
- Golden artık **V8 referansı** (2026-06-19'da bu ortamda donduruldu); nöral satırlar
  torch/numpy sürümüne duyarlı — başka ortamda kayarsa kanonik ortamda yeniden dondur.
- `KOZAA.IS`, `KOZAL.IS` evrenden hariç (28 hisse).
