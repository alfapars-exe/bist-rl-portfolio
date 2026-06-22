# BIST 28 Portföy Yönetimi — Pekiştirmeli Öğrenme (Final Proje Dokümantasyonu)

> Bu belge, uygulamanın **mimarisini, çalışma mantığını, kullanılan teknolojileri ve
> algoritmaları** tek yerde toplar ve aynı zamanda **PEKİŞTİRMELİ ÖĞRENME Final Projesi**
> şablonunun (PDF) tüm zorunlu başlıklarını (§9.1–§9.9) karşılar. Rapor ve sunum bu
> belgeden türetilebilir.

> **V12 uygulama notu (2026-06-22):** Aktif kod yolu vade presetleri ve
> Gün/Ay/Yıl ortalama granülerliği yerine `step_days=N` kullanır. Her gözlem
> ardışık N BIST seansının dönem-sonu kapanışıdır; ajan her adımda rebalans yapar,
> ağırlıklar fiyat sonrası self-financing biçimde sürüklenir ve UI/CLI aynı
> `RunSpec`/`BacktestResult` sözleşmesini paylaşır. Belgedeki V11 preset tabloları
> tarihsel deney açıklaması olarak korunmuştur.

**Proje adı:** BIST 28 hissesi üzerinde derin pekiştirmeli öğrenme ile portföy yönetimi
**Ajan(lar):** Tek ajan — dört farklı algoritmayla (DQN / PPO / SAC / TD3) bağımsız eğitilir ve karşılaştırılır
**Arayüz:** Streamlit (canlı eğitim + adım-adım test oynatma + karşılaştırma)
**Tohum:** `SEED = 42` (deterministik; golden-master 1e-6 toleransta kilitli)

---

## 0. Hızlı Bakış

| Konu | Değer |
|---|---|
| Problem türü | Sürekli kontrol — portföy tahsisi (continuous control / allocation) |
| Evren | BIST 30'dan 28 hisse (KOZAA.IS, KOZAL.IS hariç) |
| Veri aralığı | 2015-01-01 → 2024-12-31, train/test ayrımı 2022-01-01 |
| Durum uzayı | ℝ³⁹⁷ (DQN/SAC: 13 öznitelik×28 + **4 makro** + 29 ağırlık) · ℝ³⁶⁹ (PPO: 12×28 + 4 makro + 29) |
| Eylem uzayı | Ayrık 6 şablon (DQN) **veya** sürekli ℝ²⁹ → softmax simpleks (PPO/SAC/TD3) |
| Ödül | `log(1+w·r) − η·‖Δw‖₁ − λ·max(0,DD−τ) − iflas + w_dsr·DSR − κ·CVaR` (adaptif + **rejim-amplified**) |
| Makro rejim (V6) | faiz (getiri eğrisi), dolar (USD/TRY), altın (gram-altın/TL) + bileşik rejim skoru |
| Fiyat gürültüsü (V8) | Eğitimde gerçekleşen getiriye slippage (`σ=0.001`) — anti-ezber, hocanın şartı; eval'de kapalı |
| Nakit faizi (V10) | Nakit varlık risksiz faiz kazanır (`cash_annual_rate=0.40`; günlük `(1+r)^(1/252)-1`); ajan fırsat maliyetini içselleştirir |
| Algoritmalar | DQN (ayrık), PPO (sürekli on-policy), SAC (sürekli off-policy stokastik), **TD3 (sürekli off-policy deterministik — hocanın tavsiyesi)** |
| Tahmin katmanı | CNN-LSTM bir-adım getiri tahmincisi (predict-then-optimize, sadece DQN/SAC/TD3) |
| Adil karşılaştırma (V11) | Re-base hizalama + maliyetli baseline + nakit %40 faiz + PPO deterministik eval + 8 metodoloji düzeltmesi |
| Doğrulama | 101 pytest + golden-master regresyon (1e-6) + walk-forward (3 kat) + **titizlik: Deflated/Probabilistic Sharpe + PBO + Monte-Carlo stres** (López de Prado) |
| Teknolojiler | Python 3.10–3.12, PyTorch (CPU), Streamlit, Plotly, matplotlib, pandas, NumPy, yfinance |

> **Ana bulgu (dürüst tez, v12 re-baseline; `tests/golden/metrics_baseline.csv`):**
> TD3 (Sharpe 2.203 / NAV 2.728) ve SAC (Sharpe 2.174 / NAV 2.729) EqualWeight
> (Sharpe 2.185 / NAV 2.830) ile risk-ayarlıda başa baş; MeanVar en yüksek NAV (2.929).
> **DQN bu konfigürasyonda NAV≈0.01, Sharpe≈−6.59 (pratik iflas); gizlenmez.**
> Eski V11 değerleri (SAC 5.595, DQN 1.348 vb.) v12 re-baseline ile geçersizdir.
> V11 adil-karşılaştırma düzeltmeleri (re-base hizalama, maliyetli baseline, nakit faizi)
> korunmaya devam etmektedir; yalnız sayısal sonuçlar yeniden temellendirilmiştir.

---

## 1. Giriş (Rapor §9.1)

> **Ana bulgu (dürüst tez, v12 re-baseline; seed=42):** TD3 (Sharpe 2.203 / NAV 2.728)
> ve SAC (Sharpe 2.174 / NAV 2.729) EqualWeight (Sharpe 2.185 / NAV 2.830) ile
> risk-ayarlıda başa baş; MeanVar en yüksek NAV (2.929). **DQN bu konfigürasyonda
> NAV≈0.01 (pratik iflas); bu v12 modelinin DQN sonucudur, gizlenmez.**
> Eski V11 sayıları (SAC 5.595, DQN 1.348 vb.) v12 re-baseline ile geçersizdir.
> V11 adil-karşılaştırma düzeltmeleri (re-base hizalama, maliyetli baseline, nakit %40
> faizi) korunmaktadır; yalnız sayısal sonuçlar yeniden temellendirilmiştir.

**Problem neden önemli?** Portföy yönetimi, sınırlı sermayeyi zaman içinde değişen riskli
varlıklara dağıtma problemidir. Her gün piyasa yeni bilgi üretir; yatırımcı işlem maliyeti,
düşüş (drawdown) riski ve getiri arasında sürekli bir denge kurmak zorundadır. Bu, doğası
gereği **ardışık karar verme** (sequential decision-making) problemidir.

**RL ile çözülmesi neden anlamlı?** Klasik yöntemler (eşit ağırlık, al-tut, ortalama-varyans)
ya statiktir ya da tek-adımlık optimizasyon yapar; işlem maliyetini, rejim değişimini ve
düşüş cezasını **çok-adımlı** bir amaç içinde birlikte optimize etmez. Pekiştirmeli öğrenme,
ödül fonksiyonuna bu terimleri koyarak ajanın **uzun vadeli iskonto edilmiş ödülü** doğrudan
maksimize etmesini sağlar — tam da portföy yönetiminin doğasına uyan formülasyon budur.

**Ajan hangi kararı öğreniyor?** Ajan her rebalans gününde, gözlediği duruma (teknik
göstergeler + bir-adım getiri tahmini + mevcut ağırlıklar) bakarak **yeni portföy ağırlık
vektörünü** seçer. Öğrendiği politika: "hangi piyasa koşulunda, ne kadar işlem yaparak,
hangi varlıklara ne ağırlık vermeliyim ki düşüşten korunup risk-ayarlı getiriyi
büyüteyim?"

**Bu çalışmanın katkısı:** Tek bir makro rejim skorunu hem duruma (V6) hem de ödüle
(V7 CVaR kuyruk amplifikasyonu) bağlayan birleşik bir tasarım sunması ve López de Prado
titizliğiyle (DSR/PBO) RL'in pasif baseline'ı dürüstçe geçemediğini gösteren bir BIST
portföy-RL referansı olmasıdır.

---

## 2. Problem Tanımı (Rapor §9.2 / PDF §2)

| Başlık | Tanım |
|---|---|
| **Problem adı** | BIST 28 portföy tahsisi |
| **Problem senaryosu** | Ajan, 28 BIST hissesinden oluşan bir piyasada her gün durumu gözler, yeni ağırlık vektörü seçer, ertesi günün getirisiyle NAV'ı güncellenir, ödül alır ve yeni duruma geçer |
| **Ajan / ajanlar** | **Tek ajan.** Üç algoritma (DQN/PPO/SAC) aynı problem için ayrı ayrı eğitilip karşılaştırılır |
| **Görev hedefi** | Test döneminde (2022–2024) risk-ayarlı getiriyi (Sharpe/Sortino) ve nihai NAV'ı, klasik baseline'ları geçecek şekilde büyütmek |
| **Başlangıç koşulları** | Episode başında portföy %100 nakit (`w = [0,…,0,1]`), NAV=1.0. Eğitimde rastgele başlangıç penceresi (`random_start`), değerlendirmede sabit başlangıç |
| **Değişkenlik** | Eğitimde her episode farklı tarih penceresinden başlar (çeşitlilik → genelleme). Piyasa getirileri stokastiktir; vade preset'i (kısa/orta/uzun) rebalans frekansını ve ödül katsayılarını değiştirir |
| **Başarı ölçütü** | Episode sonunda ajan NAV'ı eşit-ağırlık benchmark NAV'ını geçtiyse "başarılı" (`success_vs_benchmark`); test döneminde Sharpe/Sortino/CAGR baseline üstü |
| **Başarısızlık durumu** | NAV iflas eşiğinin (varsayılan 0.01) altına düşerse episode "iflas" ile biter ve büyük ceza uygulanır |

---

## 2b. Veri Metodolojisi ve Sınırlılıklar

Bu bölüm, deneysel sonuçların yorumlanması için kritik olan veri tasarımı kararlarını ve açık sınırlılıkları belgeler.

### Evren seçimi ve survivorship bias

Evren, bugünkü BIST 30 bileşenlerinden KOZAA.IS ve KOZAL.IS çıkarılarak oluşturulmuştur (28 hisse). Bu seçim yönteminde **survivorship bias** riski mevcuttur: 2015–2024 döneminde BIST 30 endeksinden çıkan, askıya alınan veya delist olan hisseler evrene dahil edilmemiştir. Gerçek bir portföy yöneticisi bu hisselere de maruz kalırdı; bunların dışarıda bırakılması tarihsel performans tahminini iyimser yönde çarpıtabilir. Bu, açık ve dürüst bir metodolojik sınırlılıktır.

KOZAA.IS ve KOZAL.IS'in hariç tutulma gerekçesi: `data.py` satır 17'deki yorum `"KOZAA.IS ve KOZAL.IS prompt gereği hariç tutuldu (28 hisse)"` olarak belgelenmiştir — bu iki ticker proje şartı (sınav ödevi tanımı) gereği kapsam dışıdır.

### Fiyat verisi ve düzeltmeler

- **Kaynak:** yfinance kütüphanesi (`data.py:87–90`), `auto_adjust=True` parametresiyle indirilir. Bu parametre temettü ve hisse bölünmesi (stock split) etkisini geriye dönük olarak ayarlı kapanış fiyatlarına yansıtır; ham fiyat yerine **düzeltilmiş kapanış** kullanılır.
- **Cache:** İlk başarılı indirmede `data/prices.parquet` olarak kaydedilir (pyarrow). Sonraki çalıştırmalar cache'ten okur; `use_cache=False` ile zorla yeniden indirilir.
- **Sentetik GBM fallback:** yfinance erişilemez olduğunda (`data.py:113–121`) BIST istatistiklerine kalibre edilmiş çok değişkenli GBM (Geometric Brownian Motion) üretilir. Sentetik veri cache'e **yazılmaz** (cache zehirlenmesi önlemi, `data.py:125–130`). Sentetik veriden elde edilen sonuçlar yalnızca demo amaçlıdır; gerçek BIST fiyatı değildir.

### Eksik veri işleme

- **Sütun düşürme:** `dropna(thresh=int(0.9 * len(px)))` — bir hissenin toplam veri noktalarının %90'ından fazlası eksikse o hisse düşürülür (`data.py:95`).
- **Doldurma:** `ffill().bfill()` — kalan eksik değerler önce ileri, sonra geri doldurulur.
- **Kısmi indirme koruması:** yfinance bazı ticker'ları getirememişse (`data.py:104–111`) eksik hisseler sentetik GBM ile tamamlanır; böylece evren boyutu (N=28) ve state vektörü boyutu (397) sabit kalır. Bu durum `RuntimeWarning` ile kullanıcıya bildirilir.

### İşlem yapılabilirlik basitleştirmeleri (açık sınırlılıklar)

Modelde aşağıdaki gerçek piyasa etkileri **modellenmemiştir**:

| Basitleştirme | Gerçek piyasadaki karşılığı | Etkisi |
|---|---|---|
| Bid-ask spread | Alış/satış fiyatı farkı | Gerçek işlem maliyeti η·‖Δw‖₁'den yüksek olabilir |
| Likidite kısıtı | Büyük emirlerin fiyatı hareket ettirmesi (market impact) | Küçük portföylerde ihmal edilebilir, büyük fonlarda kritik |
| Fiyat limitleri (devre kesici) | BIST günlük %10 tavan/taban | Kriz günlerinde gerçekleştirilemez emirler |
| BSMV / damga vergisi | İşlem başına %0.2 BSMV | Yüksek turnover'da ek maliyet |
| Lot kısıtı | Minimum işlem birimi (lot=1 hisse) | Küçük portföylerde tahsis hassasiyeti kaybolur |
| Slippage (V8) | Eğitimde Gauss gürültüsü σ=0.001 eklendi (`EnvConfig.price_noise_std`) | Kısmi: anti-ezber amaçlı, gerçek likidite modeli değil |

Test dönemi yalnızca 2022–2024 (≈3 yıl, tek kesim) kullanılmıştır. Farklı piyasa rejimleri (2008, 2013 BIST çöküşü, pandemi) test setine dahil değildir; walk-forward (3 kat) bu sınırlamayı kısmen giderir ancak tam olarak çözmez.

---

## 3. Problem Kısıtları (PDF §3)

| Kısıt türü | Bu projedeki karşılığı |
|---|---|
| **Simpleks kısıtı** | Ağırlıklar softmax'tan geçer → `wᵢ ≥ 0`, `Σwᵢ = 1` (kısa satış yok, kaldıraç yok) |
| **İşlem (turnover) kısıtı** | Her rebalansta `‖Δw‖₁` kadar işlem maliyeti ödülden düşülür ve NAV'ı azaltır (η katsayısı) |
| **Rebalans frekansı kısıtı** | Ağırlık yalnızca rebalans günlerinde değişir (kısa=1, orta=5, uzun=20 günde bir); ara günlerde önceki ağırlık korunur |
| **Düşüş (drawdown) kısıtı** | Tepe-değerden düşüş τ eşiğini aşınca λ ile cezalandırılır (risk yönetimi) |
| **Güvenlik / iflas kısıtı** | NAV < `bankruptcy_nav` → episode biter, ek ceza (`bankruptcy_penalty`) |
| **Zaman kısıtı (v12)** | Episode uzunluğu vade preset'inden bağımsızdır. Eğitim env = `len(px_tr)` adım (seçili tarih aralığının tamamı); eval env = `len(px_te)` adım (`max_steps=10_000` üst tavanıyla). Eski `train_max_steps` (Kısa 30, Orta 90, Uzun 360) tarihsel uyumluluk için `HORIZON_PRESETS`'te korunmuştur; aktif kod yolunda kullanılmamaktadır |
| **Ayrık eylem kısıtı (DQN)** | DQN yalnızca 6 önceden tanımlı şablondan birini seçebilir; aralık-dışı indeks `ValueError` ile reddedilir |
| **Sızıntısızlık kısıtı** | Tüm öznitelikler yalnız geçmişe bakar; ölçekleyici (z-score) ve tahminci yalnız train'de fit edilir, test'e uygulanır ama orada fit edilmez |

---

## 4. Amaç Fonksiyonu (Rapor §9.4 / PDF §4)

**Genel RL amacı:**

```
max_π  E_π [ Σ_{t=0}^{T} γ^t · r_t ]
```

**Probleme özel amaç** (ödül terimleri açık yazılırsa):

```
max_π  E_π [ Σ_t γ^t ( log(1 + wₜ·r_{t+1})           ← risk-getiri (log büyüme)
                       − ηₜ·‖Δwₜ‖₁                    ← işlem maliyeti
                       − λₜ·max(0, DDₜ − τₜ)          ← düşüş cezası
                       − C_iflas·𝟙[NAV<NAV_min]       ← iflas cezası
                       + w_dsr·DSRₜ                   ← çevrim-içi risk-ayarı (Diferansiyel Sharpe)
                       − κₜ·CVaRₜ ) ]                 ← rejim-amplified kuyruk-riski cezası (V7)
```

Yani ajan; **log-büyümeyi ve risk-ayarlı getiriyi artırmaya**, **işlem maliyetini, düşüşü ve
iflas riskini azaltmaya** çalışır. η, λ, τ katsayıları sabit değildir — `AdaptiveRewardShaper`
tarafından piyasa rejimine (rolling volatilite & turnover EWMA) göre anlık ölçeklenir.

| Sembol | Anlam |
|---|---|
| `wₜ` | t anındaki portföy ağırlık vektörü (29-boyut: 28 hisse + nakit) |
| `r_{t+1}` | t+1'deki varlık getirileri |
| `Δwₜ` | ağırlık değişimi (turnover ölçüsü) |
| `DDₜ` | tepe-değerden düşüş oranı |
| `ηₜ, λₜ, τₜ` | adaptif işlem / düşüş-ceza / düşüş-eşik katsayıları |
| `DSRₜ` | Diferansiyel Sharpe (Moody & Saffell çevrim-içi Sharpe türevi) |
| `γ` | iskonto faktörü (kısa 0.95 / orta 0.99 / uzun 0.995) |

---

## 5. MDP Formülasyonu (Rapor §9.3 / PDF §5)

### 5.1. Durum Uzayı (𝒮)

Durum vektörü `sₜ`, her hisse için bir öznitelik bloğu ile mevcut portföy ağırlıklarının
birleşimidir:

```
sₜ = [ z-skorlu teknik öznitelikler (F × 28) ‖ makro rejim (4) ‖ mevcut ağırlıklar (29) ]
```

- **F = 13** (DQN/SAC): 12 teknik öznitelik + 1 CNN-LSTM getiri tahmini → `13×28 + 4 makro + 29 = 397`
- **F = 12** (PPO): forecast özelliği hariç (ablation kararı) → `12×28 + 4 makro + 29 = 369`

**12 teknik öznitelik** (`config.FEATURES`, hepsi yalnız geçmişe bakar, ölçek-bağımsız):
`logret, ma5, ma20, vol20, rsi, macd_hist, bb_pctb, bb_bw, roc10, mom60, vol60, ema_dist`

**4 makro rejim özniteliği (V6)** (`config.MacroConfig`, causal, train-only z-score):
`regime` (bileşik omurga: VIX+S&P) · `slope` (**faiz**: TNX−IRX) · `usd_try_mom` (**dolar**) ·
`gold_tl_mom` (**altın**: gram-altın/TL). Ham `regime` ayrıca V7 ödül amplifikasyonunu besler.

| Soru (PDF §5.1) | Cevap |
|---|---|
| Ajan hangi bilgileri gözlüyor? | Teknik göstergeler (momentum/trend/volatilite/RSI/MACD/Bollinger), bir-adım getiri tahmini, mevcut ağırlıklar |
| Markov özelliğini sağlıyor mu? | **Finansal piyasa tam gözlemlenebilir bir Markov ortamı DEĞİLDİR.** Problem, geçmiş teknik göstergeler + portföy ağırlıklarıyla **yaklaşık bir MDP** (partially observable market process approximated as an MDP) olarak kurulmuştur. İşlem maliyeti `‖wₜ−w_{t−1}‖₁`'e bağlı olduğundan **mevcut ağırlığın state'e dahil edilmesi zorunludur**; geçmiş fiyat tarihçesi kayan-pencere göstergelerle özetlenerek Markov yaklaşımı güçlendirilir. Gizli makro/likidite dinamikleri, piyasa mikroyapısı ve sürü davranışı gibi gözlemlenemeyen etkenler state'te temsil edilmemiştir; bu bir açık sınırlılıktır. |
| Eksik bilgi var mı? | Ham fiyat tarihçesi yerine özetlenmiş göstergeler kullanılır; **V6'dan önce makro rejim eksikti** (faiz/dolar/altın) → eklendi. Gizli piyasa durumu (likidite, kurumsal akım, jeopolitik risk) modellenmemiştir. |
| Durum vektörü kaç boyutlu? | 397 (DQN/SAC) / 369 (PPO) — makro blok dahil |
| Görsel girdi var mı? | Hayır — durum sayısal öznitelik vektörüdür |

Tüm öznitelikler `utils.features.TrainScaler` ile **yalnız train istatistikleriyle**
z-skorlanır; aynı ortalama/std test dönemine uygulanır → veri sızıntısı yok.

**Parametrik tarih seçimi (`config.DataConfig`):** Train başlangıcı, train/test ayrım tarihi
ve test bitiş tarihi `config.py`'deki `DataConfig` dataclass'ında tek kaynaktan yönetilir.
`data.py` modül sabitleri (`START`, `END`, `SPLIT`) bu dataclass'a işaret eder; UI'daki
tarih seçiciler bu değerleri `download_bist(start, end)` ve `train_test_split(split)` ile
çalışma-zamanında enjekte eder. Varsayılan: `start="2015-01-01"`, `end="2024-12-31"`,
`train_end="2022-01-01"`. Sızıntı-güvenlidir: scaler ve forecaster yalnız `train < split`
verisinde fit edilir.

### 5.2. Eylem Uzayı (𝒜)

- **Ayrık (DQN):** 6 portföy şablonu — `{Nakit, Eşit Ağırlık, Top-3 Momentum, Top-5 Momentum,
  Ters Volatilite, Min Volatilite}`. Ajan şablon indeksini seçer; şablon o günkü piyasa
  istatistiklerinden ağırlık üretir.
- **Sürekli (PPO/SAC):** `aₜ ∈ ℝ²⁹`, `softmax(aₜ)` ile portföy simpleksine projeksiyon
  (28 hisse + nakit). PPO Gaussian politikadan örnekler, SAC tanh-sıkıştırılmış Gaussian kullanır.

### 5.3. Geçiş Yapısı (𝒫)

`P(s_{t+1} | sₜ, aₜ)` — **stokastik**, piyasa tarafından belirlenir:

1. `aₜ` → `wₜ` (rebalans günüyse softmax/şablon; değilse `w_{t−1}` korunur)
2. Gerçek getiriler gelir: `r_vec = (p_{t+1} − pₜ)/pₜ`
3. NAV güncellenir: `NAV ← NAV · (1 + wₜ·r_vec − işlem maliyeti)`
4. `s_{t+1}` = bir sonraki günün öznitelikleri ⊕ yeni ağırlıklar

**Per-episode noise — anti-ezber veri artırımı (opt-in, v12):**

`PortfolioEnv(episode_data_fn=...)` ile her `reset()` çağrısında fiyat + feature verisinin
UNIFORM-noise'lu yeni bir versiyonuna swap yapılabilir. Mekanizma:

- Episode 0 (ilk): `episode_data_fn(0)` → **orijinal** fiyatlar (gürültüsüz referans).
- Episode ≥1: `episode_data_fn(i)` → `core.episodes.make_noisy_prices(prices, noise_std, seed=seed+i)`
  → log-getirilere bağımsız **uniform** gürültü `[-σ, +σ]` eklenir, başlangıç fiyatı korunur.
  Her episode farklı tohum → farklı yol, aynı istatistik band. Gauss yerine Uniform seçilmesinin
  nedeni: aykırı-değer üretmez, deterministik aralık garantisi sağlar (`core/episodes.py:54`).
- Swap episode sınırında gerçekleşir — her `(s, a, r, s')` geçişi tek bir veri setinden gelir
  → Bellman hedefi tutarlı.

**Off-policy (DQN/SAC/TD3) replay buffer episode'lar arası KORUNUR:** Buffer, farklı
noise'lu veri setlerinin transition'larını biriktirerek karıştırır. Bu **amaçlıdır** —
bir data augmentation / anti-ezber stratejisidir; ajan farklı gürültü realizasyonlarından
gelen deneyimleri karıştırarak öğrenir. Her transition kendi veri setinden geliyor olduğu
için Bellman tutarlılığı korunur.

**PPO** (on-policy): Her episode rollout buffer'ı tüketilir ve sıfırlanır; episode sınırında
dataset swap yapıldığında GAE boundary maskesi (`self.Boundary`) bootstrap'ı keser —
farklı noise'lu episode'lar arası avantaj geçişmez.

**Golden-güvenlik:** `episode_data_fn=None` (CLI/golden default) → env mevcut prices'ı
olduğu gibi kullanır → bit-aynı davranış. UI'da "Eğitimde her episode = noise'lu veri
seti" toggle (varsayılan **açık**) `core.factory.build_env(episode_data_fn=...)` üzerinden
enjekte edilir. Kaynak: `core/episodes.py` (make_noisy_prices), `ui/sidebar.py:101–107`,
`core/factory.py:154`.

**Episode-clean (opt-in, ek katman):** `PortfolioEnv(episode_clean=True)` aynı zamanda
`_episode_idx` sayacını artırır; `idx=0` → gürültüsüz orijinal, `idx≥1` → farklı
realizasyon. **CLI ve golden testlerde default `False`** → V11 bit-aynı davranış korunur.
UI'da "1. iterasyon orijinal veri (anti-ezber)" checkbox (varsayılan **açık**).
`core.factory.build_env(episode_clean=...)` ile enjekte edilir (`core/factory.py:153`).

### 5.4. Ödül Fonksiyonu (`env/reward.py → RewardEngine`)

`rₜ = R(sₜ, aₜ, s_{t+1})` aşağıdaki terimlerin toplamıdır (hesap sırası `step()` ile birebir aynı):

| Olay / terim | Ödüle etkisi |
|---|---|
| Log-büyüme `log(1 + gross_port_r)` | **+** (ana getiri sinyali) |
| İşlem maliyeti `ηₜ·‖Δw‖₁` | **−** (fazla işlemi caydırır) |
| Düşüş cezası `λₜ·max(0, DD − τₜ)` | **−** (yüksek düşüşü cezalandırır) |
| İflas cezası (NAV<eşik) | **−** büyük sabit (≈10, timing-amplified opsiyonel) |
| Diferansiyel Sharpe `w_dsr·DSR` | **±** (çevrim-içi risk-ayarı) |
| CVaR kuyruk cezası `κ·CVaR` (V7) | **−** (krizde rejim ile amplify: `κ=w_cvar·(1+β·max(0,regime))^a`) |
| Kazanç-çarpanı ödülü `gain_bonus` (V9, opt-in) | **+** (NAV eşik üstü; varsayılan KAPALI) |

`AdaptiveRewardShaper`: `ηₜ = η·max(1, turnover_ewma/turnover_hedef)`,
`λₜ = λ·(1 + max(0, vol_oranı−1))`, `τₜ = τ·max(0.7, 1/vol_oranı)`.

#### Tam parametrik ödül/ceza kontrolleri

Tüm katsayılar `config.py`'deki `RewardConfig`, `EnvConfig` ve `HORIZON_PRESETS`
üzerinden tek kaynaktan yönetilir; UI bunları `core/factory.build_env(reward_overrides=...)`
ile çalışma-zamanında ortama enjekte eder.

**Mevcut ödül/ceza parametreleri (UI'da sidebar):**

| Parametre | Config kaynağı | Varsayılan | Açıklama |
|-----------|----------------|-----------|----------|
| `η`, `λ`, `τ` | `HORIZON_PRESETS[vade]` | kısa: 0.0015/0.25/0.03 · orta: 0.0010/0.50/0.05 · uzun: 0.0005/1.00/0.08 | İşlem, drawdown ceza/eşik (vadeye göre) |
| `bankruptcy_nav` | `EnvConfig.bankruptcy_nav` | 0.01 | İflas NAV eşiği |
| `bankruptcy_penalty` | `EnvConfig.bankruptcy_penalty` | 10.0 | Düz iflas ceza büyüklüğü |
| `vol_target` | `EnvConfig.vol_target` | 0.02 | Adaptif şekillendirici vol hedefi |
| `turnover_target` | `EnvConfig.turnover_target` | 0.05 | Adaptif şekillendirici işlem hedefi |
| `ema_alpha` | `EnvConfig.ema_alpha` | 0.05 | EWMA güncelleme oranı |
| `w_dsr` | `RewardConfig.w_dsr` | 0.05 | DSR ağırlığı (0 → kapalı) |
| `dsr_eta` | `RewardConfig.dsr_eta` | 0.01 | DSR EWMA oranı |
| `w_cvar` | `RewardConfig.w_cvar` | 0.06 | CVaR baz ağırlığı (0 → kapalı) |
| `cvar_alpha` | `RewardConfig.cvar_alpha` | 0.05 | CVaR kuyruk seviyesi (%5) |
| `regime_beta` | `RewardConfig.regime_beta` | 1.0 | Kriz amplifikasyon gücü β |
| `cvar_amp` | `RewardConfig.cvar_amp` | 1.0 | Rejim amplifikasyon üsteli a |

> **V9 — Opt-in deneysel terimler:** Kanonik ödül formülüne dahil olmayan iki ek terim (`gain_bonus`, `ruin_timing`) `RewardConfig`'te varsayılan `0.0` (kapalı) ile tanımlanmıştır; golden ≤1e-6 toleransı korunur. Ayrıntılar için bkz. **Ek B: Deneysel (Opt-in) Ödül Terimleri**.

### 5.5. Bölüm Sonlandırma Koşulları

| Koşul | Sonuç |
|---|---|
| Veri sonu (`t ≥ n_days − 1`) | `done` (nötr/başarı — NAV'a göre) |
| İflas (`NAV < bankruptcy_nav`, varsayılan 0.01) | `done` — **başarısız** + ceza |
| Maksimum adım (`step_count ≥ max_steps`, 252) | `trunc` (zaman aşımı) |

---

## 6. Kullanılan Algoritmalar (Rapor §9.5 / PDF §6)

**Dört algoritma** seçilmiştir çünkü problem **hem ayrık hem sürekli** formüle edilebilir;
böylece "ayrık şablon mu, sürekli tahsis mi daha iyi?" ve "stokastik (SAC) mı, deterministik
(TD3) sürekli politika mı?" soruları deneysel olarak yanıtlanır. **TD3, ders ekibinin sürekli-
eylem problemleri için açıkça tavsiye ettiği** (RL_12) algoritmadır.

| Problem türü | Seçilen yöntem | Gerekçe |
|---|---|---|
| Ayrık eylemli (6 şablon) | **DQN** (replay + hedef ağ + ε-greedy + Huber) | Düşük-boyutlu ayrık eylem; değer-tabanlı öğrenme verimli |
| Sürekli, on-policy | **PPO** (clipped surrogate + GAE) | Simpleks üzerinde kararlı politika gradyanı; örnek-verimli on-policy |
| Sürekli, off-policy (stokastik) | **SAC** (çift-Q + entropi düzenlemesi) | Keşif-sömürü dengesi; off-policy örnek verimliliği |
| Sürekli, off-policy (deterministik) | **TD3** (çift-Q min + gecikmeli politika + hedef yumuşatma) | DDPG'nin aşırı-tahmin/kararsızlık sorunlarını üç hileyle giderir; **hocanın tavsiyesi** |

### Ağ mimarileri ve algoritma yapılandırması (doğrulanmış)

Aşağıdaki tablo dört algoritmanın **yapılandırma ve işlevsel** olarak denetlenmiş bilgilerini
özetler. Tüm sayılar `config.py` dataclass'larından (`DQNConfig`, `PPOConfig`, `SACConfig`,
`TD3Config`) alınmıştır; bağlantı `core/factory.py:build_agent` üzerinden `hp.get(key,
Config.default)` deseniyle sağlanır (golden-güvenli: kullanıcı müdahalesi olmadan bit-aynı).

**DQN** (vanilla DQN, ayrık 6-şablon `DiscretePortfolioEnv`)

- Ağ: `QNetwork` — `MLP[s → 256 → 128 → 6]` (ReLU), 6 ham Q-değeri çıktısı (`agents/dqn.py`)
- Replay: uniform deque, kapasite 50.000 (FIFO; `DQNConfig.buffer_size`)
- Hedef ağ: hard-update her `target_update=500` **gradyan adımında** (`step_count % 500 == 0`)
- Keşif: ε-greedy, **env-adımına** bağlı lineer decay — `eps_start=1.0 → eps_end=0.05`, `eps_decay=10_000` adım
- Kayıp: Huber(δ=1.0); gradyan kırpma max_norm=10 (`agents/dqn.py:116`)
- **Tasarım tercihi:** Double-DQN uygulanmamıştır (vanilla); hedef ağ `q_target.max()` ile
  hesaplandığından overestimation bias içerebilir — bilinçli basitlik kararı

**PPO** (on-policy, sürekli, Gaussian politika)

- Ağ: `PolicyNet` — `MLP[s → 256 → 128]` (Tanh) → μ başlığı + öğrenilen `log_std` `Parameter`
  (`log_std_init=-0.5`); `ValueNet` — `MLP[s → 256 → 128 → 1]` (Tanh). **Policy ve Value ağları
  ayrıdır**; ayrı optimizer (lr_p=3e-4 / lr_v=1e-3) (`agents/ppo.py`)
- Rollout: liste tabanlı (replay buffer YOK — on-policy doğrudur); her güncelleme rollout tamamen
  tüketilir, sonra sıfırlanır
- GAE(λ=0.95): tam-rollout avantaj hesabı; normalize edilir (`adv = (adv - adv.mean()) / (adv.std() + 1e-8)`)
- Clipped surrogate: clip=0.2; n_epochs=6 (hat değeri — ctor default'u 8; `config.py` docstring);
  entropi bonusu ent_coef=0.005; mini-batch=128
- KL ölçümü: `(lp_old - logp).mean()` yalnız telemetri olarak döner; **KL erken-durdurma YOK**
  — clip zaten güven-bölgesi rolünü üstlenir (bilinçli tasarım tercihi)
- Eval: deterministik μ (`act_eval` — `agents/ppo.py:93`)

**SAC** (off-policy, sürekli, tanh-sıkıştırılmış Gaussian)

- Ağ: `GaussianPolicy` — MLP trunk (ReLU) → μ başlığı + log_std başlığı → tanh-squash;
  TWIN-Q: `QNet` × 2 + soft-target × 2 (`agents/sac.py`)
- Replay: 50.000 (`SACConfig.buffer_size`); batch=128
- Q-hedef: `min(Q1_t, Q2_t) − α·logπ`; reparametrize örnekleme + tanh-squash logp düzeltmesi
  (`logp − log(1 − a² + 1e-6).sum()`)
- Soft-target: τ=0.01 (`SACConfig.tau`)
- **Tasarım tercihi:** `alpha=0.05` SABİT — otomatik entropi ayarı (hedef-entropi bazlı α güncellemesi)
  uygulanmamıştır; keşif-sömürü dengesi ortam koşullarına göre otomatik ölçeklenmez

**TD3** (off-policy, sürekli, deterministik; hocanın tavsiyesi)

- Ağ: `Actor` — `MLP[s → 256 → 128 → action_dim]` (tanh çıktı, deterministik);
  TWIN kritikler: `Critic` × 2 + hedef kopyaları × 3 (`agents/td3.py`)
- Replay: 50.000 (`TD3Config.buffer_size`); batch=128
- Üç TD3 hilesi: **(1)** twin critics + clipped-double Q-min hedef (aşırı-tahmin azaltma);
  **(2)** GECİKMELİ politika güncellemesi: `policy_delay=2` — actor her 2 critic adımda bir
  güncellenir (`self._it % self.policy_delay == 0`); **(3)** hedef-politika yumuşatma:
  `policy_noise=0.2`, `noise_clip=0.5` — hedef aksiyona kırpılmış gürültü eklenir
- Keşif: `expl_noise=0.1` (eğitimde aksiyona eklenir; eval'de 0 — `act_eval`)
- Soft-target: τ=0.005 (`TD3Config.tau`) — SAC'tan (0.01) daha yavaş güncelleme

**Ortak:** Tüm hiperparametreler `config.py`'de tek-kaynak; `core/factory.py:build_agent`
`hp.get(key, Config.default)` ile kablolu — UI widget'ı default'a bırakılırsa eğitim
CLI/golden ile bit-aynı. Yukarıda işaretlenen noktalar (Double-DQN, KL erken-durdurma,
sabit-α) **bilinçli tasarım tercihleri** olarak belgelenmiştir.

### Hiperparametreler (`config.py` — hat-etkin değerler)

| | DQN | PPO | SAC | TD3 |
|---|---|---|---|---|
| Gizli katman | (256, 128) | (256, 128) | (256, 128) | (256, 128) |
| Öğrenme oranı | lr=1e-3 | lr_p=3e-4, lr_v=1e-3 | lr_pi=3e-4, lr_q=5e-4 | lr_pi=lr_q=3e-4 |
| γ (iskonto) | 0.99 | 0.99 | 0.99 | 0.99 |
| Batch | 64 | 128 | 128 | 128 |
| Replay buffer | 50.000 (uniform FIFO) | — (on-policy, liste) | 50.000 | 50.000 |
| Aktivasyon | ReLU | Tanh | ReLU | ReLU |
| Keşif | ε: 1.0→0.05 (10k env-adım) | entropi ent_coef=0.005 | entropi α=0.05 (sabit) | expl_noise=0.1 (eval'de 0) |
| Diğer | target_update=500 (gradyan adımı), Huber δ=1.0, grad-clip 10 | clip=0.2, λ_GAE=0.95, n_epochs=6, rollout=400, log_std_init=−0.5 | τ=0.01, tanh-squash logp düzeltme | τ=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2 |
| Eğitim uzunluğu | 12 episode | 24 güncelleme | 8 episode × 600 adım | 8 episode × 600 adım |

Ortak: `SEED=42`, CPU-PyTorch, train-only z-score, `random_start` eğitim çeşitliliği.

### Uygulama notları (dürüst sınırlar)

- **PPO eval deterministik:** PPO değerlendirmede `act_eval` politika ortalamasını (μ) kullanır
  — stokastik örnekleme değil. Bu V11 adil-karşılaştırma düzeltmelerinden biridir (C4).
- **SAC sabit α:** SAC entropi katsayısı `α=0.05` sabit tutulmuştur; otomatik entropi ayarı
  (hedef-entropi bazlı α güncellemesi) uygulanmamıştır. Bilinçli basitlik kararı; keşif-sömürü
  dengesi ortam koşullarına göre otomatik ölçeklenmez.
- **DQN standart (vanilla) DQN:** Double-DQN uygulanmamıştır; hedef ağ `q_target.max()` ile
  hesaplanır → overestimation bias içerebilir. DDQN etkisi bu ortamda deneysel olarak test
  edilmemiştir.
- **PPO KL monitörü, erken-durdurma değil:** KL `(lp_old - logp).mean()` yalnız telemetri
  olarak `train()` return dict'ine girer; clip=0.2 zaten güven-bölgesi rolünü üstlenir.
- **Algoritma karşılaştırmasında adalet kriteri (dürüst sınır):** Bu çalışmadaki algoritma
  karşılaştırması **eşit env-step veya eşit wall-clock bazında değildir**. DQN 12 episode,
  PPO 24 güncelleme (rollout=400 adım/güncelleme), SAC/TD3 8 episode × 600 adım
  (`config.py TrainConfig`). Aynı çevre-adımı bütçesiyle karşılaştırma yapılmadığından
  performans farkları hem öğrenme verimliliğini hem de bütçe asimetrisini yansıtıyor olabilir.
  Eşit-bütçe (equal env-step budget) karşılaştırması gelecek iş olarak belirtilir.

---

## 7. State ve Reward Geliştirme Süreci (Rapor §9.6 / PDF §8) — **EN KRİTİK BÖLÜM**

İlk tasarım yeterli olmadı; ödül ve durum, gözlemlenen sorunlara göre iteratif geliştirildi.
Projenin gerçek evrimi (V1→V8) aşağıdaki ≥3 iterasyon tablosuna eşlenir:

| İter. | State tasarımı | Reward tasarımı | Gözlenen problem | Yapılan düzeltme | Sonuç |
|---|---|---|---|---|---|
| **V1** | 5 teknik öznitelik (logret, ma5, ma20, vol20, rsi) + mevcut ağırlıklar | log-getiri − η·turnover − λ·DD | Ajan zayıf sinyalle benchmark'ı geçemiyor, aşırı işlem yapıyor | Durum bilgisi yetersiz ve sabit ceza katsayıları rejime uymuyor | Temel hat kuruldu |
| **V2** | + 7 teknik öznitelik (macd_hist, bb_pctb, bb_bw, roc10, mom60, vol60, ema_dist) → 12 öznitelik | — | Tek-ölçek momentum/volatilite yeterli ayrım vermiyor | Çoklu-ölçek trend/risk göstergeleri eklendi; durum 169→365 boyut | Daha zengin ayrım, daha stabil |
| **V3** | + adaptif shaper hedefleri (vol/turnover EWMA) | η, λ, τ **adaptif** (rejime göre ölçeklenir) + Diferansiyel Sharpe terimi | Sabit ceza volatil rejimde ya çok sert ya çok gevşek; risk-ayarı yok | `AdaptiveRewardShaper` + çevrim-içi `DifferentialSharpe` (w_dsr·DSR) | Rejime-duyarlı ceza, risk-ayarlı ödül |
| **V4** | + CNN-LSTM **forecast** özelliği (bir-adım getiri tahmini) → 393 boyut | — | Ajan yalnız geçmişe bakıyor; ileri-görü yok (predict-then-optimize eksik) | Train-only fit CNN-LSTM tahmini state'e eklendi. **Ablation:** DQN/SAC'a yaradı, PPO'ya zarar verdi → forecast yalnız (DQN, SAC) | DQN/SAC'ta belirgin iyileşme |
| **V5** | (değişmez) | (değişmez) | Tek test dönemine aşırı-uyum riski; genelleme ölçülemiyor | **Walk-forward** doğrulama (3 kat, fold-yerel ölçekleme) + eğitimde **random-start** pencere | Düşük fold-arası std = stabil genelleme |
| **V6** | + **makro rejim** bloğu (4 yalın öznitelik): `regime` omurgası + `slope` (faiz: TNX−IRX) + `usd_try_mom` (dolar) + `gold_tl_mom` (altın). 393→397 / 365→369 boyut | (değişmez) | Ajan makro rejim körüydü (krizde geç tepki); BIST'in baskın sürücüsü USD/TRY ve risk-on/off state'te yoktu | Tek mühendislik **rejim skoru** (`tanh(vix_rel+4·(−spx_dd)−0.10)`) state'e eklendi; train-only z-score (leak-safe) | Ajan **daha savunmacı** (DQN MaxDD −53%→−41%) ama makro algı *tek başına* getiriyi düşürdü (Sharpe ↓) — algıyı kullanan ödül eksikti |
| **V7** | (V6 omurgasını tüketir) | + **rejim-amplified CVaR** kuyruk cezası: `κ·CVaR`, `κ=w_cvar·(1+β·max(0,regime))^a`; ileri-parametrik `CVaR≈vol_ewma·φ(z_α)/α`; vade-bağlı `w_cvar` | V6'da ajan rejimi *görüyordu* ama ödülde karşılığı yoktu; ortalama iyi olsa da kuyruk (kriz) davranışı zayıftı | Aynı rejim skoru kuyruk cezasını krizde **otomatik sertleştirir** (omurga 2. kez kullanıldı) | **En zayıf ajan dönüştü**: DQN Sharpe 0.36→**0.78** (2×+), Calmar 0.15→**0.67**, MaxDD daha da düştü. PPO/SAC zaten optimale yakın → küçük değişim |
| **V8** | (değişmez) | (değişmez — değişiklik geçiş yapısında 𝒫) | Ajan eğitim fiyatlarını **ezberleyebiliyordu**; gerçekte "al" dediğinde tam o fiyattan alınmaz (slippage). Hoca finansal projede **gürültüyü açıkça şart koştu** (9. Hafta): *"al dediğinde alınmıyor, yukarıdan alırsın… hem gerçekçi olur HEM EZBERİ ÖNLER."* | Gerçekleşen riskli getiriye env-yerel RNG ile küçük Gauss **slippage** (`σ=0.001`) eklendi. **Yalnız eğitimde** (`random_start=True`); eval'de kapalı → golden eval determinizmi korunur (yalnız öğrenilen politika değişir, ölçüm deterministik kalır) | Anti-ezber düzenlileştirme: ajan tek bir fiyat-patikasına aşırı-uyamaz, daha sağlam (robust) politika öğrenir. Davranış-değiştiren iterasyon → 4 ajan satırı kanonik ortamda yeniden baseline'lanır |

> PDF en az **3 iterasyon** ister; yukarıdaki tablo gerçek **8 aşamalı** evrimi kapsar. Çekirdek
> öğrenme döngüsü: *gözlemle → eksiği gerekçelendir → state/reward'ı geliştir → ölç.* **V6→V7
> dersi (dürüst):** makro *algı* (V6) tek başına yetmedi; algıyı *kullanan ödül* (V7) eklenince
> en zayıf ajan belirgin iyileşti — "rejim skoru = omurga" (bir kez üret, iki kez kullan).

### 7b. v12 Mühendislik Değişiklikleri (Episode/Adım Modeli + NaN-Güvenliği)

v12, state/reward tasarımına değil **MDP zaman adımı sözleşmesine** ve **veri sağlamlığına**
odaklanan bir mühendislik revizyonudur. Golden ≤1e-6 korunur.

| Değişiklik | Dosya/Kaynak | Etkisi |
|---|---|---|
| **Tek-adım modeli** (`StepDefaults`, `DEFAULTS`, `STEP_DAYS_MAX`) | `config.py` | Vade preset'i episode uzunluğunu değil yalnız pencere/rebalans/ödül parametrelerini belirler. `step_days=1` (default) = günlük; n≥2 = N-günlük blok |
| **Episode = tam tarih aralığı** | `ui/services.py`, `ui/sidebar.py` | Eğitim env `len(px_tr)` adım; eval env `len(px_te)` adım (`max_steps=10_000`). Eski horizon slider kaldırıldı |
| **`resample_to_step_days(df, n)`** | `data.py` | n=1 no-op; n≥2 N-günlük blok-ortalama. Leak-safe: add_features (günlük) → split → resample sırası |
| **`core/contracts.py`** (`RunSpec`, `DataProvenance`, `BacktestResult`) | `core/contracts.py` | Model+veri kimliği/provenance sözleşmeleri; UI ve CLI aynı `RunSpec`/`BacktestResult` paylaşır |
| **`core/episodes.py`** (`make_noisy_prices`, `evaluate_noise_episodes`, `summarize_episodes`) | `core/episodes.py` | Gürültü-artırımlı çoklu-episode değerlendirmesi. `make_noisy_prices`: log-getirilere uniform `[-σ,+σ]` gürültü → yeni fiyat serisi (başlangıç fiyatı korunur). `force_price_noise=False` default → golden bit-aynı |
| **`force_price_noise`** (env parametresi) | `env/portfolio_env.py`, `core/factory.py` | Eval'de gürültü açar (çoklu-episode için); default kapalı → golden korunur |
| **`_risky_returns` `np.nan_to_num`** | `env/portfolio_env.py` | Halt/eksik gün → 0 getiri; NaN NAV zincirini kırar. Temiz veride no-op → golden korunur |
| **`_sanitize_prices()`** | `data.py` | Cache-okuma NaN temizliği: `ffill().bfill()` + tamamen-boş sütun düşürme |
| **yfinance tz-aware → tz-naive normalize** | `data.py` | `px.index.tz_localize(None).normalize()` — MIXED-boş frame hatasını önler (HF Space fix) |
| **`validate_train_range()`** | `config.py` | Resample sonrası yetersiz train noktası varsa sessiz NaN yerine dostça hata |
| **`gamma`/`mom_window`/`minvol_window` açık override** | `env/portfolio_env.py`, `core/factory.py` | `None` → preset (golden bit-aynı); UI/CLI açık değer geçebilir |

---

## 8. Deneysel Sonuçlar (Rapor §9.7)

`python main.py` çalıştırması `results/` altına CSV'leri (golden `metrics.csv` + `rigor_metrics.csv`)
ve `figures/` altına 13 figürü üretir.

### 8.1. Metrikler

| Metrik | Nerede üretilir | Açıklama |
|---|---|---|
| **Episode return** | `*_curve.csv` (`reward`), F7 | Her episode/güncelleme toplam çevre ödülü |
| **Moving average return** | F10 (hareketli ortalama) | Öğrenme eğilimini düzleştirir |
| **Başarı oranı** | `training_diagnostics.csv` (`success_rate`) | EW benchmark'ı geçen episode oranı |
| **Ortalama adım** | `training_diagnostics.csv` (`avg_steps`) | Episode başına ortalama adım (erken iflas → düşük) |
| **Ceza sayısı** | `training_diagnostics.csv` (`bankrupt_count`, `mean_tx_cost`) | İflas eden episode sayısı + ortalama işlem maliyeti |
| **Test performansı** | `metrics.csv`, F4/F5 | CAGR, Sharpe, Sortino, MaxDD, Calmar, Volatility, FinalNAV, Turnover |

### 8.2. Figürler (`figures/`)

| Figür | İçerik |
|---|---|
| F1 | Kümülatif NAV (RL ajanları vs baseline'lar, test dönemi) |
| F2 | Rolling drawdown (%) |
| F3 | 60-gün rolling Sharpe |
| F4 | Test metrikleri bar (CAGR / Sharpe / MaxDD) |
| F5 | Risk-getiri düzlemi (Sharpe izo-çizgileriyle) |
| F6 | DQN/PPO/SAC/TD3 günlük ağırlık ısı haritaları |
| F7 | Eğitim eğrileri (DQN ödül, PPO kayıplar, SAC & TD3 NAV) |
| F8 | MDP diyagramı (ajan↔ortam döngüsü) |
| F9 | Dört ajanın mimari özeti |
| **F10** | **Hareketli ortalama episode getirisi** (öğrenme eğilimi — PDF §9.7) |
| **F11** | **Deflated & Probabilistic Sharpe + PBO** (çoklu-deneme düzeltmeli — López de Prado) |
| **F12** | **Monte-Carlo stres** (1-yıl ileri terminal getiri dağılımı + VaR/CVaR) |
| **F13** | **Nominal (TL) vs Reel (USD) NAV** (lira illüzyonu — §9.9) |

### 8.3. Test metrikleri (Reprodüklenebilir kanonik, seed=42; iki ardışık main.py max fark 0.0)

Değerler `results/metrics.csv`'den alınmıştır; `tests/golden/metrics_baseline.csv` içinde 1e-6
toleransta kilitlidir. Her refactor sonrası `python main.py` ile yeniden üretilip doğrulanır.
Her iterasyon kendi referansıyla saklanır: `golden/v5_…`, `v6_…`, `v7_metrics_baseline.csv`.
Reprodüksiyon düzeltmesi sonrası golden artık gerçek anlamda reprodüklenebilirdir: aynı ortamda
iki ardışık `python main.py` çalıştırmasının tüm metriklerde maksimum farkı 0.0'dır.

**Kanonik v12 re-baseline (seed=42; `tests/golden/metrics_baseline.csv`; 7 strateji):**

> **ESKİ V11 SAYILARI GEÇERSİZDİR.** Aşağıdaki tablo v12 re-baseline sonuçlarıdır;
> golden artık 7 strateji içerir (RiskParity/InverseVol/MinVariance/Momentum/CashRiskFree
> golden'dan çıkmıştır). V11 metodoloji düzeltmeleri (re-base hizalama, maliyetli baseline,
> nakit %40 faiz) korunmakta; yalnız sayısal değerler yeniden temellendirilmiştir.

| Strateji | CAGR | Sharpe | Sortino | MaxDD | Calmar | FinalNAV | Turnover |
|---|---|---|---|---|---|---|---|
| DQN | −0.967 | −6.590 | −6.887 | −0.991 | −0.976 | 0.010 | 0.875 |
| PPO | 0.754 | 1.677 | 2.639 | −0.263 | 2.870 | 2.149 | 0.163 |
| SAC | 1.091 | 2.174 | 3.485 | −0.262 | 4.163 | 2.729 | 0.084 |
| TD3 | 1.091 | 2.203 | 3.591 | −0.247 | 4.422 | 2.728 | 0.054 |
| BuyHold | 1.135 | 2.140 | 3.439 | −0.269 | 4.225 | 2.807 | 0.006 |
| EqualWeight | 1.148 | 2.185 | 3.504 | −0.263 | 4.364 | 2.830 | 0.023 |
| MeanVar | 1.203 | 2.048 | 3.273 | −0.283 | 4.253 | 2.929 | 0.033 |

> **v12 okuma notu:** Tablo `tests/golden/metrics_baseline.csv`'den alınmıştır (2022–2024,
> seed=42). TD3 (Sharpe 2.203) ve SAC (Sharpe 2.174) EqualWeight (2.185) ile başa baş;
> MeanVar en yüksek NAV (2.929). **DQN NAV≈0.01 (pratik iflas): bu v12 modelinin
> DQN sonucudur, gizlenmez.** En yüksek NAV: MeanVar (2.929) / en yüksek Sharpe:
> TD3 (2.203). Golden 7 strateji içerir; eski RiskParity/InverseVol/MinVariance/
> Momentum/CashRiskFree golden'dan çıkmıştır. Metodoloji ayrıntıları için bkz. §8.9.

**Eğitim özet tablosu (`results/training_diagnostics.csv`):**

| Ajan | Episode sayısı | Ort. return | Hareketli ort. return (son) | Başarı oranı | Ort. adım | DD ceza adımı | Ort. işlem maliyeti |
|---|---|---|---|---|---|---|---|
| DQN | 12 | −9.33 | −8.35 | 0.167 | 252 | 545 | 0.00206 |
| PPO | 24 | −5.33 | −5.63 | 0.958 | 400 | 292 | 0.00044 |
| SAC | 8 | −5.55 | −10.70 | 0.875 | 600 | 250 | 1.9×10⁻⁵ |
| TD3 | 8 | −1.81 | −7.63 | 1.000 | 600 | 257 | 5.5×10⁻⁵ |

**Titizlik (rigor) katmanı (PARS referans ağacından port, golden-güvenli raporlama).**
`scripts/rigor_analysis.py`, deterministik eval NAV'larını okuyup `results/rigor_metrics.csv`
üretir — golden `metrics.csv`'ye **dokunmaz** (ayrı dosya). Hero metrikler:
- **Deflated Sharpe (DSR)** ve **Probabilistic Sharpe (PSR)** — Bailey & López de Prado (2014);
  çoklu-deneme (V1→V8 + ajanlar = `n_trials`) altında gözlenen Sharpe'ın *tesadüf olmama*
  olasılığı. Sağlam ajanları (SAC/TD3 ~0.90) fluke'tan (zayıf DQN ~0.00) **ayırır**.
- **PBO (Probability of Backtest Overfitting)** — CSCV (López de Prado et al. 2017); IS-en-iyi
  config'in OOS-medyan-altı olma oranı. **Yüksek PBO = RL ajanları pasif baseline'ı OOS'ta
  sağlam geçemiyor** uyarısı (dürüst, §9.9).
- **Reel (USD-bazlı) NAV** — nominal TL NAV ÷ (USD/TRY normalize) → lira illüzyonunu niceler.
- **Monte-Carlo stres** — durağan blok bootstrap (Politis-Romano) + Student-t (ağır kuyruk) →
  en iyi RL ajanın 1-yıl ileri terminal dağılımı + VaR/CVaR + P(zarar), P(>%20 düşüş).

### 8.4. İterasyon karşılaştırması — V5 → V6 → V7 (test dönemi, DQN)

> **TARİHSEL ARŞİV:** Aşağıdaki tablo `tests/golden/v7_metrics_baseline.csv` kaynaklı eski
> golden snapshot'larından (v5/v6/v7) gelmektedir. Geçerli reprodüklenebilir sonuçlar §8.3'tedir.
> Reprodüksiyon düzeltmesinden sonra DQN'in tek-seed kararsızlığı (§8.5 multi-seed) bu tarihsel
> değerlerin birer tek-gerçekleşme (single realization) olduğunu göstermektedir; seed değişiminde
> aynı rakamlar tutmayabilir (CV~%47). Meşru iterasyon kaydı olarak korunmaktadır, fakat
> **sayısal iddia için §8.3 kanonik tablosuna başvurunuz.**
>
> **Önemli not — V7 referans ölçümü:** Aşağıdaki tablo `tests/golden/v7_metrics_baseline.csv`
> kaynaklı bir **V7 iterasyon-referansının** anlık görüntüsüdür. V8 anti-ezber slippage terimi
> (`σ=0.001`, yalnız eğitimde) DQN davranışını köklü biçimde değiştirdi; reprodüksiyon
> düzeltmesi sonrası geçerli DQN sonucu FinalNAV 1.279, CAGR +9.4% (§8.3).
> V7 tarihi silinmemeli — meşru iterasyon kaydıdır; fakat **geçerli (GEÇERLİ) sonuçlar §8.3'te
> verilmektedir**. Bu tablo yalnızca state/reward geliştirme döngüsünün ölçülen yönünü gösterir.

State/reward geliştirme döngüsünün (§7) **ölçülen** etkisi. En zayıf ajan DQN, makro+CVaR
iterasyonlarından en çok faydalanan; PPO/SAC zaten optimale yakın olduğundan az değişti.

| Metrik (DQN) | V5 (teknik) | V6 (+makro algı) | V7 (+CVaR ödülü) |
|---|---|---|---|
| Sharpe | +0.57 | +0.36 | **+0.78** |
| Sortino | +0.84 | +0.53 | **+1.12** |
| MaxDD | −53% | −41% | **−34%** |
| Calmar | +0.28 | +0.15 | **+0.67** |
| CAGR | +14.9% | +6.3% | **+22.4%** |

**Dürüst okuma:** V6 makro *algısı* tek başına ajanı daha savunmacı yaptı (MaxDD ↓) ama
getiriyi düşürdü — algının ödülde karşılığı yoktu. V7'de aynı rejim skoru kuyruk cezasını
krizde sertleştirince DQN hem düşüşü azalttı hem getiriyi belirgin artırdı (Sharpe 2×+). Bu,
"rejim skoru = omurga" tezinin doğrulanmasıdır. Tek test dönemi sınırlı; gerçek hakem
`python main.py --walkforward` (fold-arası CVaR/MaxDD stabilitesi).

### 8.5. Çoklu-Seed Sağlamlık (5 seed: 42–46)

Tek bir seed sonucu tesadüf eseri iyi (ya da kötü) çıkabilir. Algoritmaların gerçek kararlılığını
ölçmek için 5 bağımsız seed (42, 43, 44, 45, 46) üzerinde tam eğitim + test döngüsü çalıştırılmış;
ort ± std olarak raporlanmıştır. Kaynak: `results/multiseed_summary.csv`.

**Çoklu-seed performans tablosu (V11 adil kanonik, 5 seed 42–46, 2022–2024 test dönemi):**

> **Not:** Aşağıdaki çoklu-seed tablosu V11 koşullarında üretilmiştir; kanonik tek-seed
> sonuçlar için v12 re-baseline tablosu (§8.3) geçerlidir. Çoklu-seed analizi algoritmaların
> **göreceli** kararlılığını ölçmek için korunmuştur; mutlak NAV değerleri V11 baseline'ına
> aittir.

| Algoritma | Sharpe ort ± std | FinalNAV ort ± std | CAGR ort ± std | MaxDD ort ± std |
|---|---|---|---|---|
| DQN | 0.903 ± 0.408 | 2.204 ± 0.908 | 0.316 ± 0.191 | −0.410 ± 0.072 |
| PPO | 2.045 ± 0.044 | 5.284 ± 0.211 | 0.838 ± 0.027 | −0.258 ± 0.007 |
| SAC | 2.123 ± 0.017 | 5.619 ± 0.079 | 0.880 ± 0.010 | −0.254 ± 0.001 |
| TD3 | 2.122 ± 0.125 | 5.748 ± 0.575 | 0.894 ± 0.072 | −0.253 ± 0.010 |

DQN per-seed FinalNAV (V11): 1.37 / 2.47 / 3.65 / 1.58 / 1.96 (CV ~%45). v12 kanonik seed=42'de DQN NAV≈0.01 (pratik iflas); §8.3'e bakınız.

**Yorum:**

- **DQN patolojik kararsız:** Sharpe CV ≈ %45 (std 0.408 / ort 0.903). Per-seed FinalNAV
  1.37–3.65 arasında geniş bir bant; kanonik seed=42 (1.348) bu dağılımın alt ucundadır. DQN
  tek-seed sonuçları yapısal olarak güvenilmez. Ayrık şablon eylem uzayı + standart DQN'in
  BIST ortamında yüksek varyanslı bir politika ürettiğinin kanıtıdır.
- **SAC en stabil:** Sharpe std yalnızca 0.017 (CV ~%0.8). FinalNAV std 0.079 — seed değişiminde
  pratik olarak aynı sonuç. Entropi düzenlemesi ve off-policy öğrenme bu stabilitenin kaynağıdır.
- **TD3 orta kararlılık:** Sharpe std 0.116, FinalNAV std 0.650 — makul ama SAC'tan yüksek.
  Deterministik politika bazı seed'lerde iyi, bazılarında zayıf yerel minimuma takılıyor.
- **PPO kararlı:** Std düşük (Sharpe 0.044), SAC ile başa baş düzeye yaklaştı (ort 2.045 vs 2.123).
- **Algoritma sıralaması std'lerle ayrışıyor:** SAC ≈ TD3 > PPO >> DQN. Tek-seed sıralama (TD3 > SAC
  > PPO >> DQN) çoklu-seed ortalamasında da korunmaktadır; DQN'in zayıflığı tek-gerçekleşme değil, yapısal.

### 8.6. Ödül Katsayı Duyarlılık Analizi (OAT — One-At-a-Time)

Ödül fonksiyonunun 5 temel katsayısı birer birer ±değiştirilerek SAC (seed=42) üzerinde
etkisi ölçülmüştür. Kaynak: `results/sensitivity.csv`.

**Sharpe yayılımı (V11, her katsayı için maks−min fark; SAC seed=42):**

| Katsayı | Test değerleri | Sharpe aralığı (maks−min) | Yorumu |
|---|---|---|---|
| `eta_base` (işlem cezası) | 0.0005 / 0.001 / 0.002 | 0.014 | En geniş yayılım; işlem maliyeti en duyarlı eksen |
| `lambda_base` (DD cezası) | 0.25 / 0.50 / 1.00 | 0.016 | Dar yayılım; DD cezası geniş platoda |
| `tau_base` (DD eşiği) | 0.03 / 0.05 / 0.08 | 0.002 | Pratik fark yok |
| `w_dsr` (Diferansiyel Sharpe) | 0.0 / 0.05 / 0.10 | 0.030 | DSR hafif iyileştiriyor, kırılgan değil |
| `w_cvar` (CVaR ağırlığı) | 0.0 / 0.06 / 0.12 | 0.009 | En dar yayılım; CVaR terimi sağlam |

**Özet:** Tüm katsayılarda maksimum Sharpe yayılımı 0.030 (<%1.5 baz metriğe göre). Ödül
fonksiyonu kırılgan bir tepe noktasına değil, **geniş ve sağlam bir platoya** oturmaktadır.
Bu, ödülün test performansına uyarlanmadığının (reward hacking / test sızıntısının olmadığının)
nicel kanıtıdır. İnsan seçimi yapılan katsayıların gerçek kalibrasyona duyarlılığı düşüktür;
ajanın öğrendiği politika, bu aralıktaki herhangi bir katsayı setiyle üretilebilir.

### 8.7. Forecast Tahminci Doğruluğu (Bağımsız Değerlendirme)

Kaynak: `scripts/forecast_eval.py`; BIST 2022–2024 test kümesi (749 gün × 28 hisse). Tahminci
`train.py` ile özdeş sızıntı-güvenli akışla değerlendirilmiştir: `TrainScaler` ve CNN-LSTM ağırlıkları
yalnız train veriyle fit edilmiş, test hissesi kaynağa dokunulmamıştır (seed=42).

| Tahminci | RMSE | MAE | Yön Doğruluğu % | Bir-Adım Corr |
|---|---|---|---|---|
| Forecast (CNN-LSTM) | 0.0307 | 0.0220 | 47.77 | 0.0169 |
| Naif persistence (r_{t-1}) | 0.0425 | 0.0308 | 50.21 | 0.0314 |
| Zero (=0 tahmini) | 0.0306 | 0.0220 | — | — |

**Dürüst yorum:** Forecast, hata büyüklüğünde (RMSE/MAE) persistence'ı %27.8 geçmektedir; ancak
bu üstünlük gerçek sinyalden değil, neredeyse-sıfır (shrink-to-mean) çıktısından kaynaklanmaktadır.
Kanıt: zero-baseline RMSE (0.0306) ≈ forecast RMSE (0.0307) — model ortalamaya yakın çıkış üreterek
büyük hata yapıyor görünmekten kaçınmaktadır. Gerçek sinyal göstergelerinde forecast persistence'ın
**altındadır:** yön doğruluğu %47.77 (yazı-tura şansının altında), bir-adım korelasyon ≈ 0.003–0.017.

**Çıkarım:** Finansal bir-adım getiri tahmini doğası gereği neredeyse imkansızdır (EMH-yakını piyasa).
Forecast feature, state'e bir "öngörü kanalı" değil, vol-kalibre bir düzenleyici girdi olarak girmektedir.
V4 ablation'daki DQN/SAC iyileşmesi muhtemelen tahmin doğruluğundan değil, ek düzenlileştirme
etkisinden kaynaklanmaktadır (hipotez; forecast'i kapatıp A/B doğrulaması önerilir). Bu dürüst negatif
bulgu, abartısız bilim anlayışının parçasıdır.

### 8.8. İşlem-Maliyeti Gerçekliği (BIST Komisyon/BSMV/Spread)

Kaynak: `scripts/cost_sensitivity.py` (gerçek çalıştırma → `results/cost_sensitivity.csv`).

**Maliyet modeli (teyitli 2024–2025):** Aracı komisyonu binde 1–2 = 10–20 bps tek-yön
(Garanti BBVA 31.05.2024; İKON Menkul 2025) + BSMV komisyon üzerine %5 + spread/slippage ~2–10 bps.
Round-trip senaryolar c ∈ {0, 10, 20, 50} bps tek-yön (kanonik ödül η=10 bps üstündeki EK maliyet).

**V11 Maliyet Simetrisi Notu:** V11'den itibaren kanonik değerlendirmede maliyet **simetriktir** —
RL ajanları ve baseline'lar aynı η=10 bps temel maliyetle hesaplanmaktadır. `cost_sensitivity.py`
bu temel üstüne EK maliyet seviyelerine (0/10/20/50 bps) duyarlılığı gösterir. Önceki asimetrik
tasarımda (RL maliyet öder, baseline ödemez) RL aleyhine yapısal bir ceza vardı; V11 bunu kaldırdı.

**η Gerçekçilik Hükmü:** η tek-yön ‖Δw‖₁'e uygulanır → η=0.0010 ≈ 10 bps tek-yön ≈ 20 bps
round-trip. Preset değerleri — kısa 0.0015 (~30 bps RT) / orta 0.0010 (~20 bps RT) / uzun 0.0005
(~10 bps RT) — gerçek BIST maliyetinin merkezinde yer almakta; yeniden kalibrasyon gerekmemektedir.

**Net Sharpe — EK Maliyet Duyarlılık Tablosu (V11, kanonik η=10 bps üstüne):**

| Strateji | Turnover (tek-yön) | Sharpe (v12 kanonik) | Sharpe @+20 bps EK | Sharpe @+50 bps EK | Drag @50 bps |
|---|---|---|---|---|---|
| SAC | 0.084 | 2.174 | ~2.149 | ~2.132 | ~%1.94 |
| TD3 | 0.054 | 2.203 | ~2.181 | ~2.163 | ~%1.82 |
| MeanVar | 0.033 | 2.048 | ~2.005 | ~1.944 | ~%5.1 |
| PPO | 0.163 | 1.677 | ~1.571 | ~1.413 | ~%15.7 |
| DQN | 0.875 | −6.590 | (daha negatif) | (daha negatif) | — |
| EqualWeight / BuyHold | 0.006–0.023 | 2.140–2.185 | (hafif düşüş) | (hafif düşüş) | ~%0–%1 |

> **Not:** v12 turnover değerleri `tests/golden/metrics_baseline.csv`'den alınmıştır.
> Eski V11 Sharpe değerleri (SAC 2.141, TD3 2.151 vb.) artık geçerli değildir.

**Bulgu (v12):** SAC ve TD3 düşük turnover'ları (0.084 / 0.054) sayesinde EK maliyet altında
kararlı kalmaktadır. DQN v12'de zaten pratik iflas konumundadır (NAV≈0.01); maliyet analizi
anlamsızdır. PPO yüksek turnover (0.163) nedeniyle 50 bps EK'de Sharpe ~1.41'e geriler.
MeanVar turnover 0.033 ile maliyet-dayanıklı konumunu korur. SAC "öğrenilmiş düşük-turnover"
politikasıyla v12'de de en savunulabilir aktif ajan konumundadır.

**Metot uyarısı:** Bu analiz `results/weights_*` + `navs_aligned` kaynaklı günlük-seri
yeniden-bileşikleme ile yapılmış **göreli** bir değerlendirmedir. Sağlam bulgu turnover→maliyet-drag
ilişkisidir, mutlak NAV değil.

**Model sınırları:** Lineer maliyet varsayımı (piyasa etkisi modellenmemiş), T+2 valör ihmal, lot/tick
yuvarlaması dışarıda, kısmi-fill yok.

### 8.9. Adil-Karşılaştırma Metodolojisi (V11 Düzeltmeleri)

V11 revizyonu, RL ile klasik baseline'ların **aynı koşullarda** değerlendirilmesini sağlayan 8 metodoloji
düzeltmesini kapsar. Bu düzeltmeler RL'i yapay olarak güçlendirmek için değil, önceki asimetrik
tasarımdan kaynaklanan haksız cezayı kaldırmak için uygulanmıştır.

| Kod | Düzeltme | Neden gerekli |
|-----|----------|---------------|
| **C1** | **Metrik tarih hizalama + re-base** — RL ve baseline NAV dizileri aynı başlangıç tarihine kırpılır, NAV[0]=1 yeniden normalize edilir | Önceki tasarımda RL ve baseline farklı başlangıç noktalarından ölçülüyordu; CAGR/Calmar karşılaştırması anlamsızlaşıyordu |
| **C2** | **Maliyetli baseline** — η=10 bps temel maliyet RL ve tüm aktif baseline'lara (MeanVar, RiskParity vb.) eşit uygulanır | Önceki tasarım RL'den işlem maliyeti keserken baseline'lar sıfır maliyetle hesaplanıyordu; asimetrik ceza |
| **C3** | **Başarı kriteri = EW-benchmark** — episode başarısı eşit-ağırlık (EW) benchmark NAV'ına göre tanımlanır | BuyHold tek-hisse ağırlıklı olduğundan çeşitlendirilmiş portföyler için daha adil referans EW'dir |
| **C4** | **PPO deterministik eval** — değerlendirmede politika ortalaması (μ) kullanılır, Gaussian örnekleme değil | Stokastik örnekleme eval varyansını artırıyor, karşılaştırmayı gürültülü yapıyordu |
| **C5** | **`--allow-synthetic` bayrağı** — yfinance erişilemezse sentetik GBM ile çalışmaya izin verir | Pipeline yfinance hatası nedeniyle çöküyor ve kıyaslama yapılamıyordu |
| **C6** | **Walk-forward makro/rejim** — her fold kendi makro z-score'uyla normalizasyon | Makro özniteliklerin tam dönem istatistiğiyle normalize edilmesi geleceğe sızıntı riski taşıyordu |
| **C7** | **Ödül log(1+net)** — işlem maliyeti düşüldükten sonraki net getiri üzerinden log-büyüme hesabı | Brüt getiri üzerinden ödül, ajan maliyet bilgisini tam yansıtmıyordu |
| **C8** | **Grafik açıklamaları (caption)** — her figür metodoloji notunu içerir (kanonik vs EK-maliyet ayrımı) | Okuyucunun tablodaki sayıların hangi maliyet varsayımıyla üretildiğini net görmesi gerekir |

**Temel mesaj:** V11 düzeltmeleri RL'i güçlendirdi çünkü önceki tasarım RL aleyhine yapısal asimetri
içeriyordu — RL maliyet öder, baseline ödemez; RL nakiti %0 kazanır, gerçek risksiz faiz yok. Bu
asimetri kaldırılınca RL (özellikle TD3/SAC) risk-bazlı optimize edicilerle başa baş seviyeye geldi.

### 8.10. Action Tasarımı ve Nakit Faizi (Fırsat Maliyeti)

**Portföy kararı:** Ajan her rebalans gününde 29-boyutlu softmax çıktısıyla (28 hisse + 1 nakit)
portföy ağırlık vektörünü seçer. Bu karar üç unsuru kapsar: (1) hangi hisselere ne ağırlık, (2)
alım/satım işleminin büyüklüğü (ağırlık değişimi `‖Δw‖₁`), (3) nakit tutma oranı.

**Nakit artık gerçek risksiz faiz kazanıyor (V10):** `EnvConfig.cash_annual_rate = 0.40` (Türkiye
2022–2024 mevduat/repo düzeyi ~%40, `config.py` satır 137). Günlük oran bileşik tutarlılıkla:

```
cash_daily_rate = (1 + 0.40)^(1/252) - 1  ≈  0.001328  (%0.13/gün)
```

Bu oran sabit (RNG çağrısı yok) olduğundan golden RNG sırası korunur; yalnızca NAV bileşimi değişir.

**Fırsat maliyeti içselleştirme:** Ajan nakit tuttuğunda risksiz getiri kazanır; hisse tuttuğunda
piyasa getirisi eksi işlem maliyeti kazanır. Dolayısıyla ajan örtük olarak "hisse mi nakit mi daha
kârlı?" sorusunu her adımda çözmektedir — gerçek bir fırsat maliyeti (opportunity cost) hesabı.

**CashRiskFree baseline:** Tüm portföyü nakitte tutan strateji; risksiz hurdle görevi görür.
V11 kanonik tabloda FinalNAV 2.506 (CAGR %39.9) idi. **v12 golden'da CashRiskFree strateji
yoktur** (7 strateji: DQN/PPO/SAC/TD3/BuyHold/EqualWeight/MeanVar). v12 kanonik'te TD3/SAC
NAV≈2.728–2.729 ile BuyHold (2.807) ve EqualWeight (2.830) hurdle'ını geçememektedir; ancak
DQN (NAV≈0.01) dışındaki tüm sürekli ajanlar varlığa yatırım yapmanın nakit tutmaktan değer
kattığını göstermektedir.

**Altın ve dolar makro-özellik olarak:** `gold_tl_mom` ve `usd_try_mom` state vektörüne (V6 makro
bloğu) **rejim sinyali** olarak girer. Bu varlıklar portföyde alınamaz — kapsam kararı olarak
tradeable yapılmamıştır (evren = 28 BIST hissesi + nakit). Altın/dolar momenti ajanın piyasa
rejimini algılamasını sağlar, doğrudan pozisyon alamaz. Gelecek iş: çok-varlık-sınıfı evren.

---

## 9. Mimari ve Çalışma Mantığı

### 9.1. Paket yapısı (SOLID geçişi sonrası)

```
kod/
├── app.py                # İnce Streamlit giriş noktası (main + sekme dispatch + uyumluluk re-export)
├── main.py               # CLI orkestratör: veri → eğitim → figürler (+ --walkforward)
├── train.py              # DataBundle; prepare_data/train_dqn/ppo/sac/td3/evaluate/run
├── data.py               # BIST verisi indirme + sentetik GBM fallback + cache + train/test split
│                         #   · _sanitize_prices() — cache NaN temizliği (v12)
│                         #   · resample_to_step_days(df, n) — N-günlük blok-ortalama (v12)
│                         #   · resample_to_granularity(df, granularity) — aylık/yıllık
│                         #   · yfinance tz-aware → tz-naive normalize + emniyet ağı (v12)
├── config.py             # TEK yapılandırma kaynağı (SEED, HORIZON_PRESETS, FEATURES, *Config dataclass)
│                         #   · StepDefaults / DEFAULTS / STEP_DAYS_MAX (v12 tek-adım modeli)
│                         #   · validate_train_range() — dejenere-config NaN koruması (v12)
│                         #   · GRANULARITY_OPTIONS / GRANULARITY_MIN_POINTS
├── plots.py              # 13 figür (matplotlib, F1–F13)
├── scripts/rigor_analysis.py  # DSR/PBO/Monte-Carlo stres/reel-NAV (golden-güvenli)
├── utils/deflated_sharpe.py   # Deflated/Probabilistic Sharpe + CSCV-PBO (López de Prado)
├── utils/stress_mc.py         # Monte-Carlo stres (Student-t + blok bootstrap)
├── agents/
│   ├── base.py           # BaseAgent (act_eval) + SupportsQValues Protocol (ISP)
│   ├── common.py         # mlp, ReplayBuffer (+ torch_utils re-export)
│   ├── dqn.py · ppo.py · sac.py · td3.py   # dört ajan
├── core/
│   ├── trainer.py        # generator-tabanlı ortak eğitim döngüsü + _TRAINERS registry dispatch (OCP)
│   ├── rollout.py        # değerlendirme (evaluate)
│   ├── walkforward.py    # walk-forward doğrulama
│   ├── factory.py        # build_agent / build_env — step_days/gamma/mom/minvol override (v12)
│   ├── contracts.py      # RunSpec / DataProvenance / BacktestResult sözleşmeleri (v12)
│   ├── episodes.py       # make_noisy_prices / evaluate_noise_episodes / summarize_episodes (v12)
│   ├── persistence.py    # save_agent(name,saved_at) / load_agent / named_model_path / list kayıt
│   └── features.py       # select_features (forecast-filtreleme, DRY)
├── env/
│   ├── portfolio_env.py  # PortfolioEnv — step_days / force_price_noise / nan_to_num (v12)
│   │                     #   gamma / mom_window / minvol_window açık override (None→preset)
│   └── reward.py         # AdaptiveRewardShaper + DifferentialSharpe + RewardEngine (SRP)
├── forecast/forecaster.py  # CNN-LSTM bir-adım getiri tahmincisi (train-only fit)
├── utils/
│   ├── features.py       # add_features + TrainScaler (z-score)
│   ├── metrics.py        # CAGR/Sharpe/Sortino/MaxDD/Calmar/Turnover + training_diagnostics
│   ├── baselines.py      # equal_weight / buy_and_hold_index / mean_variance
│   ├── portfolio_tl.py   # NAV→TL/lot/işlem-logu türetimi (UI katmanı)
│   └── torch_utils.py    # get_device / set_seed (nötr; DIP)
├── ui/                   # Streamlit paketi (SRP)
│   ├── state.py          # episode_clean=True (UI default) dahil session başlangıç değerleri
│   ├── services.py       # evaluate_noise_episodes_ui / _run_trace_loop (v12 gürültü-episode)
│   │                     #   list_saved_models / save_trained_agent / resample akışı
│   ├── charts.py · sidebar.py   # step_days kaydırıcı + episode seçici (v12)
│   └── tabs/ (mdp · train · test · compare)
└── tests/                # pytest + golden-master (1e-6); 7 strateji v12 re-baseline
```

### 9.2. Çalışma mantığı (uçtan uca akış)

**CLI (`python main.py`):**
1. `data.download_bist()` → 28 hisse fiyat matrisi (cache/yfinance/sentetik)
2. `utils.features.add_features()` → 12 teknik öznitelik
3. `forecast.build_forecast_feature()` → CNN-LSTM tahmini (train-only fit)
4. `TrainScaler` → train-only z-score; train/test ayrımı → `DataBundle`
5. `core.factory.build_env/build_agent` → ortam + ajan; `core.trainer.train` generator döngüsü
6. `core.rollout.evaluate` → test backtest; `utils.metrics.summary` → metrikler
7. CSV'ler (`results/`) + figürler (`plots.run`)

**UI (`streamlit run app.py`):** Aynı `core` çekirdeğini paylaşır. Sekmeler: **Veri & MDP**
(formülasyon), **Eğitim** (canlı eğri + ⏹ durdur), **Test** (adım-adım oynatma), **Karşılaştırma**
(metrik tablosu + NAV + ısı haritası). Eğitim generator'ı CLI ile **aynı** telemetri dict'ini üretir.

### 9.3. Tasarım ilkeleri (SOLID)

- **SRP:** Ödül hesabı env'den ayrı `RewardEngine`'de; UI tek dosyadan `ui/` paketine bölündü.
- **OCP:** Ajan/eğitici seçimi `if/elif` yerine registry (`AGENT_BUILDERS`, `_TRAINERS`) — yeni
  algoritma eklemek mevcut kodu değiştirmez.
- **DIP:** `get_device/set_seed` nötr `utils.torch_utils`'te; `forecast→agents` ters bağımlılığı kırıldı.
- **ISP:** DQN'e özgü `q_values` introspeksiyonu `SupportsQValues` Protocol'üyle resmileştirildi.
- **DRY:** forecast-filtreleme ve fabrika tek kaynağa indirildi (`core.features`, `core.factory`).

**`core/factory.py` + `config.py` parametrik kablolama notu:**
`build_agent(algo, state_dim, action_dim, hp, seed)` fonksiyonu `hp.get(key, Config.default)`
deseniyle çalışır: `hp` dict'inde bir anahtar yoksa ilgili `Config` dataclass sabitine düşer.
Bu sayede UI'da yeni bir widget eklemek `factory.py` veya ajan constructor'larını
değiştirmeden yeterlidir — sidebar `hp["key"] = widget_value` yazar, factory bunu okur.
`hidden` parametresi tuple gerektirdiğinden `hp.get("hidden", DQNConfig.hidden)` döndürdükten
sonra `tuple(...)` ile sarılır (`core/factory.py:35,52,66,79`). Widget default'u
`Config.hidden` sabitine bağlandığında kullanıcı dokunmazsa ajan CLI/golden ile bit-aynı
çalışır (RNG tüketimi değişmez → golden ≤1e-6 korunur).

### 9.4. Teknoloji yığını

| Katman | Teknoloji |
|---|---|
| Dil | Python 3.10 / 3.11 / 3.12 |
| Derin öğrenme | PyTorch (CPU) — ajanlar + CNN-LSTM tahminci |
| Sayısal | NumPy, pandas, SciPy |
| Veri | yfinance (indirme) + parquet cache (pyarrow) + sentetik GBM fallback |
| Arayüz | Streamlit + Plotly (canlı/etkileşimli) |
| Statik figür | matplotlib |
| Test / kalite | pytest, pytest-cov, golden-master regresyon, SonarCloud (CI) |
| Determinizm | tek `SEED=42`, train-only ölçekleme, env-yerel RNG |

---

## 10. Arayüz ve Görsel Sunum (Rapor §9.8 / PDF §7)

`streamlit run app.py` ile açılır. PDF §7'nin istediği tüm bileşenler mevcuttur:

| PDF §7 bileşeni | Uygulamadaki karşılığı |
|---|---|
| **Eğitim başlatma düğmesi** | Eğitim sekmesi → "Eğit" / "Yeniden Eğit" (type=primary) |
| **Durdurma / devam** | Her iterasyon sonunda **⏹ Eğitimi Durdur**; durdurulan ajan session'a kaydedilir. **▶ Devam Et** aynı ajanla (ağırlık+optimizer+buffer) kaldığı yerden sürdürür; "Yeniden Eğit" sıfırdan başlatır |
| **Test düğmesi** | Test sekmesi → "Test dönemini çalıştır (rollout + trajectory)" |
| **Görsel ortam (ajan hareketi)** | Test sekmesi adım-adım oynatma (⏮ ◀ ▶ ▶▶ + kaydırıcı): portföy ağırlık pastası, ağırlık ısı haritası, işlem logu, adım-adım NAV/TL — ajanın "hareketi" = portföy tahsisinin zaman içindeki değişimi |
| **Performans grafiği (return vs episode)** | Eğitim sekmesi canlı eğrileri: kümülatif ödül, kazanç, başarı (0/1), loss |
| **Hiperparametre alanları** | Sidebar: öğrenme oranı, ε-decay/clip/α, batch, rollout, vade (γ), adaptif toggle, ödül katsayıları (η/λ/τ/iflas) |

Ek paneller: Q-değeri çubuğu (DQN), ödül-terim dekompozisyonu, adaptif katsayı zaman serileri
(ηₜ/λₜ/τₜ), TL bazlı kâr/zarar ve kümülatif işlem geçmişi.

**Model kalıcılığı (sunum):** Sidebar'daki "Eğitilmiş modeli kaydet" / "Kaydedilmiş modeli yükle"
ile eğitilmiş ajan diske (`models/{algo}_{horizon}_{adaptive}.pt` şemasıyla) yazılır/okunur —
sunumda yeniden eğitmeden Test çalıştırılabilir. Kaydedilmiş modeller bir selectbox listesiyle
seçilir; `python main.py` dört ajanı otomatik kaydeder.

**V11 UI eklentileri:**
- **Nakit faiz alanı:** Sidebar'da `cash_annual_rate` (yıllık %, varsayılan %40) doğrudan girilebilir;
  günlük bileşik oran `(1+r)^(1/252)-1` formülüyle otomatik türetilir ve ortama enjekte edilir.
- **Vade gün-limiti slider:** Seçilen vadeye ait `min_days`–`max_days` aralığında (Kısa 1–30,
  Orta 30–90, Uzun 90–360) kullanıcı episode uzunluğunu ince ayarla seçebilir; `train_max_steps`
  olarak ortama iletilir.
- **Kaydedilmiş model listesi (selectbox):** `models/` altındaki `.pt` dosyaları `{algo}_{horizon}_{adaptive}.pt`
  şemasıyla listelenir; kullanıcı açılır listeden önceden eğitilmiş bir modeli seçip doğrudan
  Test sekmesine geçebilir.

**Yeni UI özellikleri (sonraki commit'ler):**

- **Parametrik hiperparametreler — "Gelişmiş" expander (v12 sidebar):** Her algoritma dalında
  ana kontrollerin altında `st.sidebar.expander("🔧 Gelişmiş hiperparametreler")` açılır.
  Açık kontroller algo × parametre olarak:

  | Algo | Ana kontroller | Gelişmiş expander (`🔧`) |
  |------|----------------|--------------------------|
  | DQN | lr, eps_decay, batch_size, target_update | hidden (256×128 / 128×64 / 512×256), buffer_size, eps_start, eps_end, huber_delta |
  | PPO | rollout_len, lr_p, lr_v, clip, ent_coef, batch_size, n_epochs | hidden, lam (GAE λ), log_std_init |
  | SAC | lr_pi, lr_q, alpha, tau, batch_size | hidden, buffer_size |
  | TD3 | lr_pi, lr_q, policy_noise, expl_noise, tau, batch_size | hidden, noise_clip, policy_delay, buffer_size |

  Tüm widget default'ları `config.py` sabitine birebir bağlıdır (ör. `DQNConfig.hidden`,
  `PPOConfig.lam`, `TD3Config.noise_clip`). **Golden-güvenlik:** Kullanıcı hiçbir widget'a
  dokunmazsa `build_agent` aldığı `hp` dict'i config sabit değeriyle örtüşür → eğitim
  CLI/golden ile bit-aynı. Yalnız `ui/sidebar.py` değişti; `core/factory.py` ve ajan
  constructor'ları değişmedi — `hp.get(key, Config.default)` deseni sayesinde widget
  eklemek yeterli oldu. Kaynak: `ui/sidebar.py:387–564`, `core/factory.py:32–88`.

- **Episode-clean (1. iterasyon orijinal veri):** Sidebar'da "1. iterasyon orijinal veri
  (anti-ezber)" checkbox'ı (varsayılan **açık**). Açıkken 1. episode gürültüsüz orijinal
  fiyatlarla, 2.–N. episodeler her biri farklı `N(0,σ)` gürültü realizasyonuyla çalışır.
  Kapalıyken (CLI/golden modu) tüm episodeler V11 bit-aynı davranışla gürültülü kalır.
  Kaynak: `ui/sidebar.py:92–99`, `env/portfolio_env.py:94`, `core/factory.py:153`.

- **Adım granülerliği — step_days (v12):** Sidebar'da `step_days` kaydırıcısı
  (`config.STEP_DAYS_MAX=252`; kaynak: `config.py`). Seçilen N değerinde pipeline:
  (1) fiyatlar ve feature'lar **her zaman günlük** hesaplanır (`add_features` DEĞİŞMEZ);
  (2) `data.resample_to_step_days(df, n)` — n=1 no-op (golden-güvenli), n≥2 N-günlük
  blok-ortalama (`df.resample(f"{n}D").mean().ffill()`); (3) env her adımda N seans
  karşılığı getiriyi işler. `validate_train_range()` resample sonrası yetersiz nokta
  varsa kullanıcıya dostça hata verir (sessiz NaN yerine). `config.DEFAULTS`
  (`StepDefaults`) = step_days=1 + medium-preset türevi parametreler (tek kaynak).
  Eski `resample_to_granularity` (aylık/yıllık) hâlâ mevcuttur; aktif UI yolu
  `step_days` kontrolünü kullanır. Kaynak: `config.py` (StepDefaults/STEP_DAYS_MAX),
  `data.py` (resample_to_step_days), `ui/sidebar.py`, `ui/services.py`.

- **Model kaydet/yükle (isim + tarih):** Eğitilen ajan kullanıcı-verilen ada ve kayıt
  tarihiyle (`saved_at` ISO, saniye hassasiyeti) diske yazılır.
  `core.persistence.named_model_path(name)` → `models/{güvenli_isim}.pt` yolunu döner;
  `save_agent(agent, algo, path, name=..., saved_at=...)` meta alanlarını checkpoint'e
  gömer. `list_saved_models()` (`ui/services.py:54`) `models/*.pt` dosyalarını tarayıp
  meta okur; sonuçlar sidebar'da `saved_at`'e göre sıralı selectbox'ta görünür. Eski
  `{algo}_{horizon}_{adaptive}.pt` şeması geriye uyumlu olarak listede kalmaya devam eder.
  Kaynak: `core/persistence.py:50–100`, `ui/services.py:35–68`, `ui/sidebar.py:517–555`.

- **Gürültü-artırımlı çoklu-episode (v12 opt-in):** Test sekmesinde "🎲 Gürültü-artırımlı
  çoklu episode" paneli. Episode sayısı + σ kaydırıcı; "Çalıştır" düğmesiyle `core.episodes.
  evaluate_noise_episodes` sırayla N tam-aralık episode çalıştırır (episode 0 = orijinal,
  1..N = farklı tohumlu gürültü). Sonuçlar: NAV(TL) çok-çizgili grafik + özet tablo (Final
  NAV ort/std, ort. getiri, zarar olasılığı) + **episode seçici** → seçilen episode için
  adım-adım detay tablosu (Gün#/Tarih/Aksiyon/Nakit TL/Portföy TL/Adım P&L/Kümülatif P&L/
  Kümülatif %/Komisyon TL/Δturnover/Tutulan hisse — ana test tablosuyla aynı sütunlar).
  `force_price_noise=True` ile env eval'de gürültü açar; default `False` → golden bit-aynı.
  Kaynak: `core/episodes.py` (make_noisy_prices/evaluate_noise_episodes/summarize_episodes),
  `ui/services.py` (evaluate_noise_episodes_ui/_run_trace_loop), `ui/tabs/test.py`.

### Arayüz ekran görüntüleri

**1) Veri & MDP sekmesi** — evren, MDP tuple, vade preset'leri, adaptif ödül açıklaması:

![Veri & MDP sekmesi](docs/screenshots/01_mdp.png)

**2) Eğitim sekmesi (canlı)** — return-vs-episode eğrileri (kümülatif ödül / kazanç / başarı /
loss), canlı throughput metrikleri ve **⏹ Eğitimi Durdur** / **▶ Devam Et** kontrolleri:

![Eğitim sekmesi](docs/screenshots/02_train.png)

**3) Test sekmesi (adım-adım ajan hareketi)** — ⏮◀▶▶▶ oynatma + kaydırıcı, portföy ağırlık
pastası, durum öznitelikleri, ödül dekompozisyonu, TL bazlı P&L; ajanın "hareketi" = portföy
tahsisinin zaman içindeki değişimi. Solda **💾 Model kaydet/yükle** paneli:

![Test sekmesi — adım-adım oynatma](docs/screenshots/03_test.png)

**4) Karşılaştırma sekmesi** — metrik tablosu, test dönemi NAV eğrileri, ağırlık ısı haritası,
DQN aksiyon dağılımı:

![Karşılaştırma sekmesi](docs/screenshots/04_compare.png)

> Ekran görüntüleri `streamlit run app.py` üzerinden alınmıştır; "test animasyonu" Test
> sekmesindeki ▶ oynatma ile canlı izlenir.

---

## 11. Tartışma (Rapor §9.9)

> **Ana bulgu (dürüst tez, v12 re-baseline; seed=42; §8.3 sayısal dayanak):**
> TD3 (Sharpe 2.203 / NAV 2.728) ve SAC (Sharpe 2.174 / NAV 2.729) EqualWeight
> (Sharpe 2.185 / NAV 2.830) ile risk-ayarlıda başa baş; MeanVar en yüksek NAV (2.929).
> **DQN bu konfigürasyonda NAV≈0.01, Sharpe≈−6.59 (pratik iflas); bu v12 modelinin
> DQN sonucudur, gizlenmez.** Eski V11 sayıları (SAC 5.595 vs BuyHold 5.751 vb.)
> v12 re-baseline ile geçersizdir. V11 metodoloji düzeltmeleri (§8.9) korunmaktadır.

- **İlk state tasarımı neden yetersizdi?** 5 teknik öznitelik tek-ölçekli sinyal veriyordu;
  çoklu-ölçek trend/volatilite ve ileri-görü (forecast) olmadan ajan rejim ayrımı yapamıyordu.
- **İlk reward tasarımı neden yetersizdi?** Sabit η/λ/τ, volatil rejimde ya aşırı ya yetersiz
  cezalandırıyordu; risk-ayarlı (Sharpe türevi) terim yoktu → ajan ödül uğruna riskli davranıyordu.
- **En kritik düzeltme?** İki taşıyıcı: (1) **adaptif ödül şekillendirme + Diferansiyel Sharpe**
  (rejime-duyarlı, risk-ayarlı ödül), (2) **CNN-LSTM forecast özelliği** (predict-then-optimize) —
  ablation forecast'ı yalnız DQN/SAC'a vermeyi gerektirdi.
- **Ajan hangi davranışı öğrendi?** Düşük-turnover, düşüş-bilinçli tahsis; volatil dönemde
  nakit/ters-volatilite ağırlıklı, sakin dönemde momentum ağırlıklı davranış.
- **Ajan nerede başarısız kaldı?** Somut bulgular (`results/metrics.csv`, §8.1):
  - **DQN v12'de pratik iflas:** NAV≈0.01, Sharpe≈−6.59 — bu v12 modelinin DQN sonucudur,
    gizlenmez. V11 çoklu-seed (CV ~%45, Sharpe 0.903±0.408) DQN'in yapısal kararsızlığını
    gösteriyordu; v12'de bu kararsızlık daha da belirgin biçimde tezahür etti. Ayrık şablon
    tasarımı bu ortamda yüksek varyans ve kırılganlık üretmektedir.
  - **Sürekli ajanlar (SAC/TD3) EqualWeight ile başa baş:** v12 kanonik'te TD3 Sharpe 2.203,
    SAC Sharpe 2.174, EqualWeight 2.185 — pratik fark yok. NAV'da EqualWeight (2.830) ve
    BuyHold (2.807) sürekli ajanların hafif önünde. MeanVar en yüksek NAV (2.929).
  - **Risk-ayarlıda sonuç:** EqualWeight (2.185) TD3/SAC (2.203/2.174) ile başa baş. Golden'dan
    çıkan RiskParity/InverseVol/MinVariance ile karşılaştırma v12'de mevcut değildir. Bu sonuç
    DeMiguel et al. (2009) ile tutarlıdır: naif 1/N çeşitlendirmeyi ham Sharpe'ta yenmek zordur.
  - Ani rejim kırılmalarında tepki gecikmeli; PPO forecast özelliğinden faydalanamadı (on-policy
    + dağılım kayması).
- **Ezberi nasıl önledik (hocanın şartı)?** Hoca finansal projede gürültüyü açıkça şart koştu:
  *"al dediğinde alınmıyor, yukarıdan alırsın… hem gerçekçi olur HEM EZBERİ ÖNLER."* **V8**'de
  gerçekleşen getiriye eğitim-içi slippage (`σ=0.001`, env-yerel RNG) eklendi — ajan tek bir
  fiyat-patikasını ezberleyemez, daha sağlam (robust) politika öğrenir. Eval'de gürültü **kapalı**
  olduğundan golden ölçüm determinizmi korunur (yalnız öğrenilen politika değişir).
- **Reprodüklenebilirlik:** Reprodüksiyon düzeltmesi sonrası golden artık gerçek anlamda
  reprodüklenebilirdir — aynı ortamda iki ardışık `python main.py` çalıştırmasının tüm
  metriklerde maksimum farkı 0.0'dır. Eski golden statik CSV'ye karşı karşılaştırma yapıyordu;
  yeni yaklaşım canlı deterministik üretim + 1e-6 toleranslı regresyon testidir. Çoklu-seed
  analizi (§8.5) tek-seed güvenilirliğini algoritmik olarak niceler.
- **Lira illüzyonu — nominal kazanç ne kadarı gerçek? (§9.9)** Test döneminde (2022–2024) tüm
  stratejiler ve baseline'lar yüksek **nominal TL** getiri gösterir (BuyHold NAV ≈ 6.7×). Ancak bu
  dönemde USD/TRY ~13'ten ~35'e yükseldi (≈2.7× devalüasyon) ve enflasyon yüksekti. **Nominal NAV'ın
  büyük kısmı satın-alma-gücü artışı değil, para biriminin değer kaybıdır** — "lira illüzyonu". Hoca
  bunu da işaret etti: *"başlangıç paranı o yılın değerine göre normalize et."* Bu artık
  **uygulandı** (`utils.metrics.real_nav` + F13): NAV, başlangıç USD/TRY'ye normalize edilerek
  **reel (USD-bazlı) NAV** olarak raporlanır. Sonuç çarpıcı — nominal ≈ 5.5× kazanç reel bazda
  ≈ 2.1×'e iner: **nominal getirinin ~%60'ı satın-alma-gücü değil, TL'nin değer kaybıdır.** Ajanlar
  arası **göreli** sıralama para biriminden bağımsızdır (hepsi aynı TL evreni) → değişmez; mutlak
  kazanç yorumu reel bazda yapılır. Golden-güvenli (yalnız raporlama; eğitim/eval/ödül değişmez).
- **Strateji gerçekten sağlam mı, yoksa backtest aşırı-uyumu mu? (titizlik katmanı)** "Yüksek
  Sharpe" yanıltıcı olabilir — çoklu-deneme (V1→V8 + 4 ajan) altında bir strateji şans eseri iyi
  görünebilir. Bunu **Deflated Sharpe (DSR)** ve **Probabilistic Sharpe (PSR)** ile (López de Prado)
  test ettik: sağlam ajanlar (SAC/TD3 DSR ~0.90) ile fluke (DQN DSR ~0.00) **net ayrışıyor**.
  Dahası **PBO (Probability of Backtest Overfitting, CSCV)** yüksek çıkıyor — bu *dürüst* bir bulgu:
  RL ajanları pasif baseline'ı (Eşit Ağırlık/BuyHold) test döneminde **sağlam biçimde geçemiyor**.
  Bu, RL'in piyasayı "yendiği" iddiasını **abartmaktan kaçınmamızı** sağlar; literatürle de tutarlı
  (DeMiguel et al. 2009: naif 1/N çeşitlendirmeyi ham Sharpe'ta yenmek zordur). **Monte-Carlo stres**
  (blok bootstrap + Student-t ağır kuyruk) en iyi RL ajanın 1-yıl ileri VaR/CVaR + felaket olasılığını
  niceler.
- **Problem daha karmaşık olsaydı ne eklenirdi?** Bu projede zaten eklenenler: **makro rejim** (V6),
  **rejim-amplified CVaR** (V7), **TD3** (sürekli-deterministik ajan), **fiyat gürültüsü** (V8),
  **nakit risksiz faizi** (V10, `cash_annual_rate=0.40`), **adil-karşılaştırma metodolojisi** (V11,
  8 düzeltme), ve **titizlik katmanı** — Deflated/Probabilistic Sharpe + PBO + Monte-Carlo stres +
  reel-NAV (PARS referans ağacından port). Bundan sonrası: çok-varlık-sınıfı evren (altın/dolar
  tradeable), BSMV'yi ödüle katma, reel-NAV'ı doğrudan ödüle katma, otomatik entropi (SAC-auto-α).
- **Tier-C analiz scriptleri (§8.7–§8.8):** `scripts/forecast_eval.py` (CNN-LSTM tahminci
  bağımsız değerlendirmesi, 749 gün × 28 hisse) ve `scripts/cost_sensitivity.py` (BIST
  komisyon/BSMV/spread senaryoları → `results/cost_sensitivity.csv`) bu proje kapsamında
  eklenmiştir. Her ikisi de sızıntı-güvenli akışı (train-only fit, seed=42) miras alır ve
  golden-güvenlidir (eğitim/eval değişmez, yalnız raporlama analizi).

---

## 12. Çalıştırma ve Teslim (PDF §10–§11)

### Kurulum
```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[dev]"                              # pyproject bağımlılıkları + pytest
```

### Komutlar
```bash
streamlit run app.py            # arayüz (eğitim + test + karşılaştırma)
python main.py                  # tam akış: veri → eğitim → backtest → titizlik → 13 figür
python scripts/rigor_analysis.py # yalnız titizlik katmanı (DSR/PBO/stres/reel-NAV)
python main.py --skip-data      # cache varsa veriyi atla
python main.py --walkforward    # walk-forward genelleme doğrulaması (3 kat)
pytest -q                       # 101 test
pytest -m "not slow"            # hızlı yerel döngü (UI smoke hariç)
```

### Determinizm / doğrulama
- `SEED=42` her yerde sabit → tekrar-üretilebilir.
- `tests/test_golden_regression.py`: `results/metrics.csv` ↔ `tests/golden/metrics_baseline.csv`
  karşılaştırması (1e-6). Davranış-koruyan her değişiklik bu kapıdan geçer.
- CI (`.github/workflows/ci.yml`): Python 3.10/3.11/3.12 matrisi + SonarCloud (3.12'de).

### Teslim bileşenleri
- **Final raporu:** Bu `DOKUMANTASYON.md` (PDF §9 başlıklarıyla hizalı) çıktısı.
- **Python kodları:** Tüm kaynak `kod/` altında; rapor ekine eklenir.

---

## 13. Dosya Rehberi (hızlı referans)

| Soru | Dosya |
|---|---|
| MDP / ortam mekaniği | `env/portfolio_env.py` |
| Ödül matematiği | `env/reward.py` (`RewardEngine`) |
| Ajan ağları | `agents/dqn.py`, `agents/ppo.py`, `agents/sac.py` |
| Eğitim döngüsü | `core/trainer.py` |
| Ajan/ortam kurulumu | `core/factory.py` |
| Hiperparametreler | `config.py` |
| Metrikler | `utils/metrics.py` |
| CLI akış | `train.py`, `main.py` |
| Arayüz | `app.py`, `ui/` |
| Figürler | `plots.py` → `figures/` |
| Testler | `tests/` |

---

## Ek B: Deneysel (Opt-in) Ödül Terimleri (V9)

Bu ek, ana ödül formülüne dahil olmayan iki deneysel terimi belgeler. Her iki terim de `RewardConfig`'te varsayılan `0.0` (kapalı) ile tanımlanmıştır; `w_*` alanları sıfır iken hesaplama sonuca sıfır katkı yapar, kanonik ödül bit-aynı kalır ve `tests/golden/` ≤1e-6 toleransı korunur. Yeni RNG çağrısı yoktur; yalnız mevcut `nav`, `step_count` ve `max_steps` değerleri kullanılır.

Kaynak: `env/reward.py` `RewardEngine.compute()` (satır 157–223) ve `config.py` `RewardConfig` (satır 152–166). Opt-in terimler `terms` sözlüğüne `gain_bonus` ve `ruin_timing_mult` anahtarlarıyla eklenmektedir (UI panelleri ve `test_env` bu anahtarları okur).

### B.1. Kazanç-çarpanı ödülü (`gain_bonus`)

NAV belirlenen `gain_floor` eşiğini aştığında doğrusal bonus; isteğe bağlı hız faktörü erken büyümeyi kayırır (nav=2 → w_gain @ gain_floor=1; nav=3 → 2·w_gain):

```
step_frac  = step_count / max_steps                          # 0 → 1
gain_bonus = w_gain · max(0, nav − gain_floor)
             · (1 + w_gain_speed · (1 − step_frac))
```

| Parametre | `RewardConfig` alanı | Varsayılan | Davranışsal etki |
|-----------|----------------------|-----------|-----------------|
| `w_gain` | `RewardConfig.w_gain` | 0.0 | Kazanç büyüklük ölçeği (0 → kapalı) |
| `gain_floor` | `RewardConfig.gain_floor` | 1.0 | NAV eşiği; üstü ödüllenir |
| `w_gain_speed` | `RewardConfig.w_gain_speed` | 0.0 | Erken-kazanç hız faktörü (0 → hızdan bağımsız) |

### B.2. İflas-timing cezası (`ruin_timing`)

Erken iflas daha sert cezalandırılır; `w_ruin_timing=0` durumunda düz `bankruptcy_penalty` değeri korunur:

```
ruin_timing_mult = 1 + w_ruin_timing · (1 − step_frac)
ruin_pen         = bankruptcy_penalty · ruin_timing_mult   (bankrupt ise, yoksa 0)
```

| Parametre | `RewardConfig` alanı | Varsayılan | Davranışsal etki |
|-----------|----------------------|-----------|-----------------|
| `w_ruin_timing` | `RewardConfig.w_ruin_timing` | 0.0 | Erken-iflas ceza ölçeği (0 → düz ceza) |
