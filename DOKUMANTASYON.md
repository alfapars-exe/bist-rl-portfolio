# BIST 28 Portföy Yönetimi — Pekiştirmeli Öğrenme (Final Proje Dokümantasyonu)

> Bu belge, uygulamanın **mimarisini, çalışma mantığını, kullanılan teknolojileri ve
> algoritmaları** tek yerde toplar ve aynı zamanda **PEKİŞTİRMELİ ÖĞRENME Final Projesi**
> şablonunun (PDF) tüm zorunlu başlıklarını (§9.1–§9.9) karşılar. Rapor ve sunum bu
> belgeden türetilebilir.

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
| Algoritmalar | DQN (ayrık), PPO (sürekli on-policy), SAC (sürekli off-policy stokastik), **TD3 (sürekli off-policy deterministik — hocanın tavsiyesi)** |
| Tahmin katmanı | CNN-LSTM bir-adım getiri tahmincisi (predict-then-optimize, sadece DQN/SAC/TD3) |
| Doğrulama | 101 pytest + golden-master regresyon (1e-6) + walk-forward (3 kat) + **titizlik: Deflated/Probabilistic Sharpe + PBO + Monte-Carlo stres** (López de Prado) |
| Teknolojiler | Python 3.10–3.12, PyTorch (CPU), Streamlit, Plotly, matplotlib, pandas, NumPy, yfinance |

---

## 1. Giriş (Rapor §9.1)

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

## 3. Problem Kısıtları (PDF §3)

| Kısıt türü | Bu projedeki karşılığı |
|---|---|
| **Simpleks kısıtı** | Ağırlıklar softmax'tan geçer → `wᵢ ≥ 0`, `Σwᵢ = 1` (kısa satış yok, kaldıraç yok) |
| **İşlem (turnover) kısıtı** | Her rebalansta `‖Δw‖₁` kadar işlem maliyeti ödülden düşülür ve NAV'ı azaltır (η katsayısı) |
| **Rebalans frekansı kısıtı** | Ağırlık yalnızca rebalans günlerinde değişir (kısa=1, orta=5, uzun=20 günde bir); ara günlerde önceki ağırlık korunur |
| **Düşüş (drawdown) kısıtı** | Tepe-değerden düşüş τ eşiğini aşınca λ ile cezalandırılır (risk yönetimi) |
| **Güvenlik / iflas kısıtı** | NAV < `bankruptcy_nav` → episode biter, ek ceza (`bankruptcy_penalty`) |
| **Zaman kısıtı** | Episode en fazla `max_steps` (varsayılan 252 ≈ 1 iş yılı) sürer |
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
| Markov özelliğini sağlıyor mu? | Evet — işlem maliyeti `‖wₜ−w_{t−1}‖₁`'e bağlı olduğundan **mevcut ağırlık state'e dahildir**; geçmiş, kayan-pencere göstergelerle özetlenir |
| Eksik bilgi var mı? | Ham fiyat tarihçesi yerine özetlenmiş göstergeler kullanılır; **V6'dan önce makro rejim eksikti** (faiz/dolar/altın) → eklendi |
| Durum vektörü kaç boyutlu? | 397 (DQN/SAC) / 369 (PPO) — makro blok dahil |
| Görsel girdi var mı? | Hayır — durum sayısal öznitelik vektörüdür |

Tüm öznitelikler `utils.features.TrainScaler` ile **yalnız train (2015–2021) istatistikleriyle**
z-skorlanır; aynı ortalama/std test dönemine uygulanır → veri sızıntısı yok.

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

### 5.4. Ödül Fonksiyonu (`env/reward.py → RewardEngine`)

`rₜ = R(sₜ, aₜ, s_{t+1})` aşağıdaki terimlerin toplamıdır (hesap sırası `step()` ile birebir aynı):

| Olay / terim | Ödüle etkisi |
|---|---|
| Log-büyüme `log(1 + gross_port_r)` | **+** (ana getiri sinyali) |
| İşlem maliyeti `ηₜ·‖Δw‖₁` | **−** (fazla işlemi caydırır) |
| Düşüş cezası `λₜ·max(0, DD − τₜ)` | **−** (yüksek düşüşü cezalandırır) |
| İflas cezası (NAV<eşik) | **−** büyük sabit (≈10) |
| Diferansiyel Sharpe `w_dsr·DSR` | **±** (çevrim-içi risk-ayarı) |
| CVaR kuyruk cezası `κ·CVaR` (V7) | **−** (krizde rejim ile amplify: `κ=w_cvar·(1+β·max(0,regime))^a`) |

`AdaptiveRewardShaper`: `ηₜ = η·max(1, turnover_ewma/turnover_hedef)`,
`λₜ = λ·(1 + max(0, vol_oranı−1))`, `τₜ = τ·max(0.7, 1/vol_oranı)`.

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

### Ağ mimarileri

- **DQN** — `QNetwork`: `MLP[s → 256 → 128 → 6]` (ReLU), Huber kaybı, hedef ağ (hard sync),
  uniform replay buffer, ε-greedy keşif.
- **PPO** — `Actor`: `MLP[s → 256 → 128]` (Tanh) → μ başlığı + öğrenilen `log_std`; `Critic`:
  `MLP[s → 256 → 128 → 1]` (Tanh). GAE avantajı, clipped surrogate + entropi bonusu.
- **SAC** — `Actor`: tanh-sıkıştırılmış Gaussian; **çift** Q-eleştirmen `MLP[s+a → 256 → 128 → 1]`
  (ReLU) + hedef ağlar; entropi katsayısı α; yumuşak güncelleme (τ).
- **TD3** — `Actor`: `MLP[s → 256 → 128 → 29]` (tanh çıktı, **deterministik**); **çift** eleştirmen
  `MLP[s+a → 256 → 128 → 1]` (ReLU) + hedef ağlar. Üç TD3 hilesi: (1) twin critics + **min-Q**
  hedefi (aşırı-tahmin azaltma), (2) **gecikmeli** politika güncellemesi (`policy_delay=2`),
  (3) **hedef-politika yumuşatma** (clamped gürültü). Keşif eğitimde aksiyona eklenen Gauss
  gürültüsüyle; eval'de deterministik (`act_eval`). Arayüz SAC ile birebir → aynı off-policy
  eğitim döngüsünü (`core.trainer.train_td3`) paylaşır.

### Hiperparametreler (`config.py` — hat-etkin değerler)

| | DQN | PPO | SAC | TD3 |
|---|---|---|---|---|
| Gizli katman | (256, 128) | (256, 128) | (256, 128) | (256, 128) |
| Öğrenme oranı | lr=1e-3 | lr_p=3e-4, lr_v=1e-3 | lr_pi=3e-4, lr_q=5e-4 | lr_pi=lr_q=3e-4 |
| γ (iskonto) | 0.99 | 0.99 | 0.99 | 0.99 |
| Batch | 64 | 128 | 128 | 128 |
| Replay buffer | 50.000 | — (on-policy) | 50.000 | 50.000 |
| Keşif | ε: 1.0→0.05 (10k adım) | entropi 0.005 | entropi α=0.05 | expl_noise=0.1 (eval'de 0) |
| Diğer | target_update=500, Huber δ=1.0 | clip=0.2, λ_GAE=0.95, n_epochs=6, rollout=400 | τ=0.01 | τ=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2 |
| Eğitim uzunluğu | 12 episode | 24 güncelleme | 8 episode × 600 adım | 8 episode × 600 adım |

Ortak: `SEED=42`, CPU-PyTorch, train-only z-score, `random_start` eğitim çeşitliliği.

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

### 8.3. Test metrikleri (golden baseline, seed=42 deterministik)

Değerler `tests/golden/metrics_baseline.csv` içinde 1e-6 toleransta kilitlidir; her refactor
sonrası `python main.py` ile yeniden üretilip doğrulanır. (Güncel değerler için `results/metrics.csv`.)
Her iterasyon kendi referansıyla saklanır: `golden/v5_…`, `v6_…`, `v7_metrics_baseline.csv`.

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

---

## 9. Mimari ve Çalışma Mantığı

### 9.1. Paket yapısı (SOLID geçişi sonrası)

```
kod/
├── app.py                # İnce Streamlit giriş noktası (main + sekme dispatch + uyumluluk re-export)
├── main.py               # CLI orkestratör: veri → eğitim → figürler (+ --walkforward)
├── train.py              # DataBundle; prepare_data/train_dqn/ppo/sac/evaluate/run
├── data.py               # BIST verisi indirme + sentetik GBM fallback + cache + train/test split
├── config.py             # TEK yapılandırma kaynağı (SEED, HORIZON_PRESETS, FEATURES, *Config dataclass)
├── plots.py              # 13 figür (matplotlib, F1–F13)
├── scripts/rigor_analysis.py  # DSR/PBO/Monte-Carlo stres/reel-NAV (golden-güvenli)
├── utils/deflated_sharpe.py   # Deflated/Probabilistic Sharpe + CSCV-PBO (López de Prado)
├── utils/stress_mc.py         # Monte-Carlo stres (Student-t + blok bootstrap)
├── agents/
│   ├── base.py           # BaseAgent (act_eval) + SupportsQValues Protocol (ISP)
│   ├── common.py         # mlp, ReplayBuffer (+ torch_utils re-export)
│   ├── dqn.py · ppo.py · sac.py   # üç ajan
├── core/
│   ├── trainer.py        # generator-tabanlı ortak eğitim döngüsü + _TRAINERS registry dispatch (OCP)
│   ├── rollout.py        # değerlendirme (evaluate)
│   ├── walkforward.py    # walk-forward doğrulama
│   ├── factory.py        # build_agent / build_env (AGENT_BUILDERS registry, OCP+DRY)
│   └── features.py       # select_features (forecast-filtreleme, DRY)
├── env/
│   ├── portfolio_env.py  # PortfolioEnv + DiscretePortfolioEnv (MDP mekaniği)
│   └── reward.py         # AdaptiveRewardShaper + DifferentialSharpe + RewardEngine (SRP)
├── forecast/forecaster.py  # CNN-LSTM bir-adım getiri tahmincisi (train-only fit)
├── utils/
│   ├── features.py       # add_features + TrainScaler (z-score)
│   ├── metrics.py        # CAGR/Sharpe/Sortino/MaxDD/Calmar/Turnover + training_diagnostics
│   ├── baselines.py      # equal_weight / buy_and_hold_index / mean_variance
│   ├── portfolio_tl.py   # NAV→TL/lot/işlem-logu türetimi (UI katmanı)
│   └── torch_utils.py    # get_device / set_seed (nötr; DIP)
├── ui/                   # Streamlit paketi (SRP)
│   ├── state.py · services.py · charts.py · sidebar.py
│   └── tabs/ (mdp · train · test · compare)
└── tests/                # 101 test + golden-master (1e-6)
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

**💾 Model kalıcılığı (sunum):** Sidebar'daki **💾 Eğitilmiş modeli kaydet** / **📂 Kaydedilmiş
modeli yükle** ile eğitilmiş ajan diske (`models/{algo}.pt`) yazılır/okunur — sunumda yeniden
eğitmeden Test çalıştırılabilir. `python main.py` üç ajanı otomatik kaydeder.

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

- **İlk state tasarımı neden yetersizdi?** 5 teknik öznitelik tek-ölçekli sinyal veriyordu;
  çoklu-ölçek trend/volatilite ve ileri-görü (forecast) olmadan ajan rejim ayrımı yapamıyordu.
- **İlk reward tasarımı neden yetersizdi?** Sabit η/λ/τ, volatil rejimde ya aşırı ya yetersiz
  cezalandırıyordu; risk-ayarlı (Sharpe türevi) terim yoktu → ajan ödül uğruna riskli davranıyordu.
- **En kritik düzeltme?** İki taşıyıcı: (1) **adaptif ödül şekillendirme + Diferansiyel Sharpe**
  (rejime-duyarlı, risk-ayarlı ödül), (2) **CNN-LSTM forecast özelliği** (predict-then-optimize) —
  ablation forecast'ı yalnız DQN/SAC'a vermeyi gerektirdi.
- **Ajan hangi davranışı öğrendi?** Düşük-turnover, düşüş-bilinçli tahsis; volatil dönemde
  nakit/ters-volatilite ağırlıklı, sakin dönemde momentum ağırlıklı davranış.
- **Ajan nerede başarısız kaldı?** Ani rejim kırılmalarında (ör. şok günleri) tepki gecikmeli;
  PPO forecast özelliğinden faydalanamadı (on-policy + dağılım kayması).
- **Ezberi nasıl önledik (hocanın şartı)?** Hoca finansal projede gürültüyü açıkça şart koştu:
  *"al dediğinde alınmıyor, yukarıdan alırsın… hem gerçekçi olur HEM EZBERİ ÖNLER."* **V8**'de
  gerçekleşen getiriye eğitim-içi slippage (`σ=0.001`, env-yerel RNG) eklendi — ajan tek bir
  fiyat-patikasını ezberleyemez, daha sağlam (robust) politika öğrenir. Eval'de gürültü **kapalı**
  olduğundan golden ölçüm determinizmi korunur (yalnız öğrenilen politika değişir).
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
  **rejim-amplified CVaR** (V7), **TD3** (sürekli-deterministik ajan), **fiyat gürültüsü** (V8), ve
  **titizlik katmanı** — Deflated/Probabilistic Sharpe + PBO + Monte-Carlo stres + reel-NAV (PARS
  referans ağacından port). Bundan sonrası: çok-varlık-sınıfı evren, nakit faiz geliri + BSMV'yi
  *ödüle* katma (davranış-değiştiren — şu an yok), reel-NAV'ı doğrudan ödüle katma.

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
