# BIST 28 Portföy Yönetimi — İnteraktif Pekiştirmeli Öğrenme Demosu

UYİK 2026 bildirisi için hazırlanan DQN/PPO/SAC portföy RL ortamının
**Streamlit tabanlı interaktif UI**'ı. Ajan eğitimini ve test dönemini
adım adım gözleyebildiğiniz, vade preset'leri (Kısa/Orta/Uzun) ve
adaptif ödül şekillendirici içeren bir demo uygulama.

## Öne Çıkanlar

- **Evren**: 28 BIST hissesi (`KOZAA.IS` ve `KOZAL.IS` hariç), 2015-01-01 → 2024-12-31
- **Durum (v2)**: ℝ³⁹³ = 13 özellik (12 teknik + 1 CNN-LSTM forecast) × 28 hisse + 29 ağırlık (nakit dâhil), train-only z-score
- **Eylem (DQN)**: 6 şablon (Nakit, Eşit, Top-3 Mom, Top-5 Mom, Ters-Vol, Min-Vol)
- **Eylem (PPO/SAC)**: 29-boyutlu softmax (sürekli)
- **Ödül**: `log(1+w·r) − η_t·‖Δw‖₁ − λ_t·max(0, DD−τ_t)` — 4 terim ayrı ayrı raporlanır
- **Vade Preset'leri**: Kısa / Orta / Uzun — (η, λ, τ, γ, rebalans frekansı) değişir
- **Adaptif Şekillendirici**: EWMA rolling vol + turnover'a göre katsayıları anlık ölçekler
- **Framework**: PyTorch (tüm ajanlar)

## v2 Değişiklikleri (RL-in-finance literatürüyle hizalı)

| Değişiklik | Gerekçe / kaynak |
|-----------|-------------------|
| Zengin gözlem: 5 → 13 feature (MACD, Bollinger %b/bant, ROC, mom60, vol60, EMA-uzaklık + forecast) | FinRL standart TA seti (arXiv:2011.09607) |
| CNN-LSTM forecast: bir-adım getiri tahmini state'e (predict-then-optimize) | LSTM→PPO hibrit (arXiv:2511.17963) |
| Diferansiyel Sharpe ödülü: online risk-ayarlı terim | Moody & Saffell; çok-ödül (arXiv:2511.11481) |
| Rastgele-başlangıç ortam + env-yerel RNG → çeşitli, çok-episode | overfitting/genelleme (Velay 2023, arXiv:2306.10950) |
| Walk-forward doğrulama: `python main.py --walkforward` | backtest overfitting (Liu 2022, arXiv:2209.05559) |

> **Sızıntısızlık**: tüm feature'lar causal (yalnız ≤t); forecaster YALNIZ train'de fit;
> walk-forward fold-yerel ölçeklenir. `tests/` bunları kilitler (leak-safety testleri).
> **Not**: Aşağıdaki S1/S2 raporlama bölümleri v1 (ℝ¹⁶⁹) tasarımını anlatır; v2 bunu
> yukarıdaki tabloyla genişletir.

## Kurulum

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

CPU üzerinde çalışacak şekilde test edilmiştir. `torch` CPU build'i varsayılan olarak kurulur.

## Çalıştırma

### İnteraktif UI (ana deliverable)

```bash
streamlit run app.py
```

Tarayıcı otomatik açılır (http://localhost:8501). Sidebar'dan:
1. **Ajan** seçin (DQN/PPO/SAC)
2. **Vade** seçin (Kısa/Orta/Uzun)
3. **Adaptif ödül** toggle'ını ayarlayın
4. "Veriyi Yükle" → sonra sırayla Tab 2 → Tab 3 → Tab 4

### CLI Pipeline (9 yayın figürü)

```bash
python main.py                   # tam akış
python main.py --skip-data       # veri cache'i varsa
python main.py --skip-train      # sadece çizim
python main.py --skip-plots      # sadece eğitim
```

## UI Sekmeleri

| Sekme | İçerik |
|-------|--------|
| 📐 Veri & MDP | 28 ticker, MDP tuple, vade preset tablosu, adaptif formül açıklaması |
| 🎓 Eğitim | Canlı ödül/kazanç/başarı/loss eğrileri, progress bar, session cache |
| 🎬 Test (Adım-Adım) | Oynat/Durdur/İleri-Geri + slider; durum, Q-değerleri, ağırlık pastası, ödül terimleri, adaptif katsayıların mini zaman serisi |
| 📊 Karşılaştırma | Metrik tablosu (CAGR, Sharpe, Sortino, MaxDD, Calmar, Vol, FinalNAV, Turnover), NAV çok-çizgili, ağırlık ısı haritası, DQN aksiyon dağılımı, adaptif katsayılar |

## Dosya Yapısı

```
kod/
├── app.py                      # Streamlit UI (ana giriş) — core.trainer/rollout tüketir
├── main.py                     # CLI orkestratörü (train.run() + plots.run())
├── train.py                    # CLI eğitim + backtest sürücüsü (prepare_data + run)
├── plots.py                    # 9 matplotlib figürü (run())
├── data.py                     # BIST 28 indirme + parquet cache
├── config.py                   # Merkezi hiperparametreler + HORIZON_PRESETS (tek kaynak)
├── pyproject.toml              # Paketleme + pytest yapılandırması
├── requirements.txt
├── README.md
├── data/prices.parquet         # yfinance cache (ilk çalıştırmada oluşur)
├── core/                       # Eğitim/eval çekirdeği — CLI + UI ortak (Faz 3)
│   ├── trainer.py              # generator tabanlı eğitim (DQN/PPO/SAC) + dispatch
│   ├── rollout.py              # ajan-agnostik evaluate (act_eval)
│   └── walkforward.py          # v2: walk-forward doğrulama (overfitting kontrolü)
├── forecast/                   # v2: CNN-LSTM bir-adım getiri tahmincisi
│   └── forecaster.py           # predict-then-optimize; 'forecast' feature (train-only fit)
├── env/
│   ├── __init__.py
│   └── portfolio_env.py        # MDP env + AdaptiveRewardShaper (HORIZON_PRESETS → config)
├── agents/
│   ├── __init__.py
│   ├── base.py                 # BaseAgent arayüzü (act_eval)
│   ├── common.py               # ReplayBuffer, mlp, get_device, set_seed
│   ├── dqn.py                  # PyTorch DQN (169→256→128→6, Huber, lr=1e-3)
│   ├── ppo.py                  # PyTorch PPO (GAE, clipped surrogate)
│   └── sac.py                  # PyTorch SAC (twin-Q, tanh-squashed Gaussian)
├── utils/
│   ├── __init__.py
│   ├── features.py             # add_features + TrainScaler (z-score)
│   ├── baselines.py            # EW, BuyHold, MeanVar
│   ├── metrics.py              # CAGR/Sharpe/Sortino/MDD + success_vs_benchmark
│   └── portfolio_tl.py         # NAV→TL/lot/işlem-logu türetimi (UI katmanı)
├── tests/                      # pytest: invariants + golden-master regresyon
│   └── golden/                 # dondurulmuş metrics_baseline.csv (≤1e-6 gate)
├── results/                    # CSV'ler
└── figures/                    # PNG'ler
```

## Hiperparametreler (DQN Varsayılanları — Prompt Spec)

| Parametre | Değer |
|---|---|
| Ağ mimarisi | `Linear(169, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 6)` |
| Öğrenme oranı | 1e-3 (Adam) |
| Kayıp | Huber (δ=1.0) |
| Replay buffer | 50.000, uniform, batch 64 |
| Target sync | Her 500 adım (hard update) |
| ε-greedy | 1.0 → 0.05, 10.000 adımda lineer decay |
| γ | 0.99 (orta vade) / 0.95 (kısa) / 0.995 (uzun) |
| Seed | 42 (tekrar üretilebilir) |

## Vade Preset'leri

| | Kısa | Orta | Uzun |
|---|---|---|---|
| Rebalans (gün) | 1 | 5 | 20 |
| Momentum pencere | 5 | 20 | 60 |
| Min-vol pencere | 20 | 60 | 120 |
| η base | 0.0015 | 0.0010 | 0.0005 |
| λ base | 0.25 | 0.50 | 1.00 |
| τ base | 0.03 | 0.05 | 0.08 |
| γ | 0.95 | 0.99 | 0.995 |

## Adaptif Ödül Şekillendirici

`AdaptiveRewardShaper` (env/portfolio_env.py) her adımda EWMA ile:
- `vol_ewma` (portföy getirisinin mutlak değeri)
- `turnover_ewma` (`‖Δw‖₁`)

değerlerini günceller ve katsayıları ölçekler:

```
η_t    = η_base  × max(1, turnover_ewma / turnover_target)
λ_t    = λ_base  × (1 + max(0, vol_ratio − 1))
τ_t    = τ_base  × max(0.7, 1 / vol_ratio)
```

Böylece çok işlem yapan ajan kendini cezalandırır, volatil rejimde DD cezası
sertleşir ve eşik daralır. Sidebar'daki toggle kapatıldığında katsayılar sabit
kalır — A/B karşılaştırması için.

## Kabul Kriterleri Karşılığı

- ✅ Tek komut: `streamlit run app.py`
- ✅ MDP 5 bileşeni kodda + UI'da Tab 1'de
- ✅ DQN'in 4 temel bileşeni (replay, target, ε-greedy, MLP 256-128)
- ✅ 3 zorunlu eğitim grafiği (kümülatif ödül, kazanç, başarı)
- ✅ Test sekmesinde adım-adım ajan hareketleri + Q-değerleri + ödül dekompozisyonu
- ✅ Metrikler tablo + CSV indir butonu
- ✅ 2 raporlama sorusu (aşağıda)
- ✅ Kısa/Orta/Uzun vade ödül politikaları
- ✅ Adaptif ödül ve ceza

---

# Raporlama Soruları

## S1 — Durum Temsili Neden Bu Şekilde?

Durum vektörümüz 169 boyutludur: 28 hisse × 5 teknik öznitelik + 29 boyutlu
önceki ağırlık vektörü (28 riskli varlık + nakit). Bu boyut, **bilgi zenginliği
ile öğrenme verimliliği arasında bilinçli bir denge**dir. 5 özellik (log getiri,
5g/20g hareketli değişim, 20g volatilite, RSI-14) kısa vadeli ivme, orta vadeli
trend ve risk rejimini eşzamanlı yakalar; daha uzun LSTM/CNN mimarilerinin
ihtiyaç duyduğu ham fiyat tarihçesini feature engineering ile özetler.
Önceki ağırlıkların duruma eklenmesi MDP'nin **Markov özelliğini korumak için
kritiktir**: işlem maliyeti `‖w_t − w_{t−1}‖₁`'e bağlı olduğundan mevcut
ağırlık bilinmeden optimal eylem tanımlanamaz. 169 boyut, 256-128 hidden
MLP ile 15 dakika altında CPU üzerinde eğitilebilir büyüklüktedir; daha
büyük ham pencere (ör. 60-günlük fiyat dizisi) durum boyutunu 1.700+'a
çıkarır ve model eğitilemez hâle gelir. **Veri sızıntısına karşı** tüm
özellikler `TrainScaler` ile yalnızca 2015–2021 train kümesinin
istatistikleriyle z-skorlanır; aynı ortalama/std test (2022–2024)
dönemine **uygulanır ama orada fit edilmez**. RSI ve hareketli
pencerelerin hepsi yalnızca geçmişe bakar, ileri-bakış yoktur. Böylece
ajan, kâğıttaki iddiayı gerçek piyasaya transfer edilebilecek gerçekçi
bilgilerle eğitilir.

## S2 — Ödül Fonksiyonunun Davranış Etkileri

Ödül üç terimden oluşur: `r_t = log(1+w·r) − η_t·‖Δw‖₁ − λ_t·max(0,
DD−τ_t)`. **log getiri terimi** ajanı uzun vadede geometrik büyümeye
yönlendirir; basit aritmetik getiriden farklı olarak kümülatif kayıpları
daha ağır cezalandırır (log convavity). **İşlem maliyeti (η·‖Δw‖)** ajanın
sürekli yeniden-tartım yaparak sömürü-keşif gürültüsüne düşmesini önler;
η büyüdükçe ajan daha istikrarlı hold stratejilerine kayar. **Drawdown
cezası (λ·max(0, DD−τ))** sadece eşik aşıldığında aktifleşir — böylece
küçük dalgalanmalar cezasız kalır, fakat %5+ düşüşlerde risk yönetimi
zorlanır. Vade preset'lerimiz bu trade-off'u öyküleştirir: Kısa vade
(η=0.0015, λ=0.25, τ=0.03) günlük rebalansta agresif çalışan ajana
yüksek işlem maliyeti koyar ama DD'yi rahat bırakır; Uzun vade
(η=0.0005, λ=1.0, τ=0.08) aylık rebalansla neredeyse işlem yapmayan
ama büyük DD'den şiddetle kaçınan bir profil çizer. **Adaptif
şekillendirici** (EWMA ile rolling vol ve turnover izleyen) statik
katsayı seçiminin en zayıf yanını — piyasa rejimi değiştiğinde aynı
kalan hiperparametreleri — çözer: yüksek-vol rejimde λ_t büyür ve τ_t
daralır, böylece ajan türbülansta nakite kayar; aşırı trade eden
ajan turnover_ewma şişirip η_t'yi kendi kendine büyüterek
*self-regulating* biçimde sakinleşir.

---

## Notlar

- `seed=42` sabit; PyTorch CPU'da deterministik davranış için
  `torch.manual_seed` + `np.random.seed` çağrıları mevcuttur.
- yfinance erişimi yoksa kod sentetik GBM üretir; deney yine çalışır,
  sadece sonuçlar farklı olur.
- Test dönemi trajectory'si session_state'te saklanır; ajanı yeniden
  eğitmeden tekrar tekrar incelenebilir.

**Yazar**: Harun Benli — UYİK 2026 Bildirisi
