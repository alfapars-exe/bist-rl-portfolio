# MİMARİ HAFIZA — BIST 28 Portföy RL

> Bu dosya `mimari-hafiza-koruyucusu` ajanının **canlı kayıt defteridir**. Alt-ajanlar
> durumsuzdur; projenin mimari belleği burada yaşar. Her büyük değişiklikten sonra
> güncellenir. **Kaynak-doğruluk**: bu dosyadaki bir iddia kodla çelişiyorsa kod kazanır;
> çelişkiyi düzelt.

**Proje**: BIST 28 portföy-yönetimi RL · UYİK 2026 bildirisi / `RL_FinalProje.pdf` teslimi
**Kanonik kök**: `kod/` · **Son güncelleme**: (ilk seed — kurulumla birlikte)

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

- _(seed)_ TD3 sürekli kontrol için eklendi (hoca tavsiyesi); state ℝ³⁹³, eylem 29-softmax.
- _(seed)_ PPO durumu ℝ³⁶⁵: forecast feature'ı dışlar (v2 ablation).
- _(seed)_ Rigor modülleri (DSR/PBO/CSCV/MC) sibling `Reinforcement Learning Final/` (PARS)
  referansından port ediliyor; PARS kanonik teslim DEĞİL, referans.

## 6. Bilinen Riskler / Açık Konular

- Ortam pyarrow/yfinance'a bağlı; bazı kabuklarda eksik olabilir (golden env-kilitli).
- `KOZAA.IS`, `KOZAL.IS` evrenden hariç (28 hisse).
