# SOLID Kapsamlı Refactoring — Tasarım Dokümanı

- **Tarih:** 2026-05-31
- **Dal:** `refactor/solid-pass` (← `ci/sonarcloud-coverage`)
- **Kapsam:** Tüm kod tabanı, kapsamlı SOLID geçişi (Yaklaşım B), 7 faz
- **Değişmez kural:** Davranış birebir korunur. `tests/` (özellikle golden-master 1e-6) güvenlik ağıdır.

---

## 1. Bağlam

BIST 28 portföy RL projesi (DQN/PPO/SAC + Streamlit UI). Kod zaten birçok refactoring fazı (Faz 1-5, v2-v5) geçirmiş ve büyük ölçüde modüler. Bu çalışma, kalan SOLID ihlallerini — sorumluluğun yoğunlaştığı yerlerde — gideren, davranış-koruyan bir geçiştir. Amaç **gereksiz soyutlama eklemek değil**, mevcut iyi mimariyi tamamlamaktır.

## 2. Hedefler / Hedef olmayanlar

**Hedefler**
- `app.py`'nin (1151 satır) SRP ihlalini bir `ui/` paketine bölerek gidermek.
- UI↔CLI arasındaki fabrika/forecast-filtreleme tekrarını `core/`'da birleştirmek (DRY).
- `isinstance`/`if algo==` dağıtımını registry tabanlı hale getirmek (OCP).
- Ödül hesabını env'den ayrı bir `RewardEngine`'e çıkarmak (SRP).
- `get_device`/`set_seed`'i nötr konuma taşımak; `forecast→agents` ters bağımlılığını kırmak (DIP).
- DQN'e özgü `q_values` introspeksiyonunu `SupportsQValues` Protocol ile resmileştirmek (ISP).
- `train.py` modül-global state'ini açık `DataBundle`'a çevirmek (testability/DIP).

**Hedef olmayanlar**
- `core/trainer.py`'nin off-policy/on-policy ayrımını birleştirmek (bilinçli tasarım; sızıntılı soyutlama riski — DOKUNULMAZ).
- Sayısal davranışı değiştirecek herhangi bir şey (golden-master 1e-6 kalır).
- Yeni özellik, algoritma veya bağımlılık eklemek.

## 3. SOLID Bulguları (inceleme özeti)

| Bulgu | Konum | İlke | Faz |
|---|---|---|---|
| `app.py` 7+ sorumluluk | `app.py` | SRP | P5 |
| forecast-filtreleme + fabrika tekrarı | `app.py:117`, `train.py:66`, `train.py:55` | DRY | P2, P3 |
| `isinstance` dispatch | `core/trainer.py:147` | OCP | P4 |
| `if algo==` ajan fabrikası | `app.py:135` | OCP | P3 |
| Ödül hesabı `step()` içinde | `env/portfolio_env.py:258` | SRP | P7 |
| `forecast → agents.common` | `forecast/forecaster.py:20` | DIP | P1 |
| `train.py` global state | `train.py:31,37` | DIP/test | P6 |
| `hasattr(agent,"q_values")` | `app.py:190` | ISP | P5 |

**Bilinçli olarak değiştirilmeyenler (yazarın kararı savunulabilir):** trainer off/on-policy ayrımı; `DiscretePortfolioEnv(PortfolioEnv)` step imza farkı (env tipi her zaman ajan tipiyle eşleşir).

## 4. Hedef yapı

```
kod/
├── app.py              # İNCE entry: main() + sekme dispatch + uyumluluk re-export'ları
├── train.py            # DataBundle; _feats_for shim kalır
├── main.py, plots.py, data.py, config.py   (≈ değişmez)
├── core/
│   ├── trainer.py      # registry dispatch (isinstance YOK; TypeError korunur)
│   ├── rollout.py, walkforward.py          (değişmez)
│   ├── factory.py      # 🆕 build_agent / build_env
│   └── features.py     # 🆕 select_features
├── env/
│   ├── portfolio_env.py # step() → RewardEngine; shaper re-export
│   └── reward.py        # 🆕 AdaptiveRewardShaper + DifferentialSharpe (taşındı) + RewardEngine
├── agents/
│   ├── base.py          # + SupportsQValues Protocol
│   ├── common.py        # get_device/set_seed re-export (compat)
│   └── dqn.py, ppo.py, sac.py               (değişmez)
├── forecast/forecaster.py # utils.torch_utils'ten import
├── utils/
│   ├── torch_utils.py   # 🆕 get_device, set_seed
│   └── features.py, metrics.py, baselines.py, portfolio_tl.py (değişmez)
└── ui/                  # 🆕 paket
    ├── state.py         # _init_state, _agent_key, env_rebalance_hint
    ├── services.py      # _load_data, train_generator, evaluate_with_trace, _compute_test_tl_snaps
    ├── charts.py        # _weights_pie, _q_bar, _reward_bar, _state_top_features, _horizon_preset_table
    ├── sidebar.py       # sidebar_controls, _sidebar_reward_editor
    └── tabs/
        ├── mdp.py · train.py · test.py · compare.py
```

## 5. Yeni soyutlamaların API'leri

### `utils/torch_utils.py` (DIP)
- `set_seed(seed: int) -> None` — `agents/common.py`'den gövde **birebir** taşınır.
- `get_device(device: str | None = None) -> torch.device` — birebir taşınır.
- `agents/common.py`: `from utils.torch_utils import get_device, set_seed` (re-export; `mlp`, `ReplayBuffer` yerinde kalır).
- `forecast/forecaster.py`: `from utils.torch_utils import get_device, set_seed`.

### `core/features.py` (DRY)
```python
def select_features(feats: dict, algo: str) -> dict:
    """forecast feature'ı yalnız ForecastConfig.forecast_agents'taki ajanlara verir."""
```
- `train._feats_for(feats, algo)` → `return select_features(feats, algo)` (shim; test edilir).
- **P2 anında** doğrudan çağıranlar: `train._feats_for` (shim) + monolitik `app.py`'nin satır-içi filtresi (`app.py:117`).
- **Nihai tüketiciler** (sonraki fazlarda): `core/factory.build_env` (P3), `ui/services` (P5).

### `core/factory.py` (OCP + DRY)
```python
AGENT_BUILDERS: dict[str, Callable]   # "DQN"|"PPO"|"SAC" -> builder
def build_agent(algo, state_dim, action_dim, hp=None, *, seed=SEED) -> BaseAgent
def build_env(algo, prices, feats, *, horizon, adaptive, max_steps,
              random_start, seed, reward_overrides=None) -> PortfolioEnv
```
- Discrete↔continuous seçimi + `select_features` burada tek noktada.
- `app._make_agent(algo, sd, ad, hp)` → `return build_agent(algo, sd, ad, hp)` (shim; test edilir).
- DİKKAT: UI ve CLI farklı `max_steps` kullanır (UI SAC=1200, CLI SAC=600). Bunlar **birleştirilmez**; çağıran taraf geçirir.

### `core/trainer.py` (OCP)
```python
_TRAINERS: dict[type, Callable]   # {DQNAgent: train_dqn, PPOAgent: train_ppo, SACAgent: train_sac}
def train(agent, env, *, n_iters=None, rollout_len=400, ew_nav=None):
    gen = _TRAINERS.get(type(agent))
    if gen is None: raise TypeError(f"Bilinmeyen ajan tipi: {type(agent).__name__}")
    ...
```
- `isinstance` zinciri kalkar; bilinmeyen tip → `TypeError` (test edilir).

### `env/reward.py` (SRP) — EN RİSKLİ
- `AdaptiveRewardShaper`, `DifferentialSharpe` → buraya **birebir** taşınır.
- `env/portfolio_env.py`: `from env.reward import AdaptiveRewardShaper, DifferentialSharpe, RewardEngine` (shaper'lar re-export).
```python
@dataclass
class RewardOutcome:
    total: float; nav: float; peak: float; port_r_net: float; terms: dict

class RewardEngine:
    def __init__(self, shaper, dsharpe, w_dsr, bankruptcy_nav, bankruptcy_penalty): ...
    def reset(self): self.shaper.reset(); self.dsharpe.reset()
    def compute(self, *, gross_port_r, delta_w_l1, nav, peak) -> RewardOutcome: ...
```
- `compute()` aritmetiğin **sırası** `step()`'teki ile birebir aynı olmalı:
  `shaper.update_and_shape → tx_cost → nav*=(1+port_r_net) → max(nav,0) → bankrupt → peak → dd → log_r → dd_penalty → dsharpe.update(port_r_net) → total`.
- `terms` dict anahtarları aynen korunur: `log_return, tx_cost, drawdown_penalty, total, dsr, dsr_term, eta_t, lambda_t, tau_t, vol_ewma, turnover_ewma, dd, gross_port_r, delta_w_l1, bankruptcy_penalty, bankrupt`.
- `step()` artık: w_new/delta_w_l1/gross_port_r hesapla → `outcome = self.reward.compute(...)` → `self.nav, self.peak = outcome.nav, outcome.peak` → history'leri yaz → `outcome.total` döndür.

### `agents/base.py` (ISP)
```python
@runtime_checkable
class SupportsQValues(Protocol):
    def q_values(self, s: np.ndarray) -> np.ndarray: ...
```
- `ui/services.evaluate_with_trace`: `hasattr(agent,"q_values")` → `isinstance(agent, SupportsQValues)`.

### `train.py` (testability/DIP)
```python
@dataclass
class DataBundle:
    px; px_tr; px_te; feats_tr; feats_te; scaler
def prepare_data() -> DataBundle      # global yerine döndürür
def train_dqn(bundle, ...); train_ppo(bundle, ...); train_sac(bundle, ...); evaluate(bundle, agent, algo, ...)
```
- `run()` bundle'ı açıkça iletir; `np.random.seed(SEED)` ve çağrı sırası korunur.
- `_feats_for` shim'i kalır (test_env onu çağırır).

## 6. Uyumluluk dikişleri (testler yeşil kalır)

| Test kilidi | Dosya | Korunma |
|---|---|---|
| `app._make_agent("DQN",169,6,{})` + config default'ları | `test_config_wiring.py:18` | app.py shim → `build_agent` |
| `train._feats_for(feats, algo)` | `test_env.py:108` | train.py shim → `select_features` |
| `train(object(), None)` → `TypeError` | `test_trainer.py:61` | registry miss → TypeError |
| Telemetri dict anahtarları (DQN/PPO/SAC) | `test_trainer.py` | generator'lar değişmez |
| `reward["reward"]` == Σ env total (1e-6) | `test_trainer.py:36` | reward semantiği değişmez |
| Ödül özdeşliği + `reward_terms` anahtarları | `test_env.py:35,62` | RewardEngine birebir aritmetik |
| `DifferentialSharpe` env'den import; ilk=0, clip | `test_env.py:117` | env/reward re-export |
| `HORIZON_PRESETS is config.HORIZON_PRESETS` | `test_config_wiring.py:10` | import korunur |
| `evaluate` dönüş anahtarları | `test_rollout.py:29` | rollout değişmez |
| `walk_forward` imza + factory(sd,ad,seed) | `test_walkforward.py` | walkforward değişmez |
| `add_features` sıra==FEATURES; `TrainScaler` | `test_features.py` | utils.features değişmez |
| `build_forecast_feature` imza + warmup | `test_forecaster.py` | forecaster imza değişmez |
| `AppTest.from_file("app.py")` istisnasız render | `test_app_smoke.py` | app.py streamlit entry kalır |
| `SAFE_MODULES` import sağlığı | `test_imports.py` | yeni modüller eklenir |

## 7. Fazlı inşa sırası (artan risk; her faz golden-kapılı)

| # | Faz | Risk | Dokunulan | Doğrulama |
|---|---|---|---|---|
| **0** | Yeşil temel | — | — | `pytest -q` + `python main.py` → `test_golden_regression` 1e-6 |
| **1** | `torch_utils` (DIP) | 🟢 | utils, agents/common, forecast | `pytest -q` (+ `test_imports`) |
| **2** | `core/features` (DRY) | 🟢 | core, train(shim), ui-öncesi app | `test_env::per_agent_forecast` + `pytest -q` |
| **3** | `core/factory` (OCP+DRY) | 🟢 | core, app(shim), train | `test_config_wiring` + `pytest -q` |
| **4** | trainer registry (OCP) | 🟢 | core/trainer | `test_trainer` (TypeError) + `pytest -q` |
| **5** | `app.py`→`ui/` + `SupportsQValues` (SRP+ISP) | 🟡 | app, ui/*, agents/base | `test_app_smoke` + `test_config_wiring` + `pytest -q` |
| **6** | `train.py` DataBundle (test) | 🟡 | train | `python main.py` golden + `test_env` |
| **7** | env `RewardEngine` (SRP) | 🔴 | env/reward, env/portfolio_env | `test_env` + `python main.py` golden — **en son** |

Golden-yolu fazları (3,4,6,7): `pytest -q` sonrası ek olarak `python main.py` çalıştır + `test_golden_regression` 1e-6 doğrula.

## 8. Doğrulama stratejisi

- **Baseline (P0):** Çalışan `.venv` ile tüm suite yeşil + `python main.py` golden 1e-6 eşleşmeli. Hiçbir faz bu sağlanmadan başlamaz.
- **Her faz sonrası:** `pytest -q` yeşil. Golden-yolu fazlarında ek golden kontrolü.
- **Final:** Tam `pytest` + `python main.py` golden + streamlit smoke (`test_app_smoke`).
- **CI paritesi:** CI Python 3.12 + `pip install -e ".[dev]"` + `pytest --cov`. Yerel `.venv` (3.10) ile çalışılır; final öncesi `SAFE_MODULES` ve yeni `ui`/`core` modüllerinin import edilebilirliği garanti edilir.

## 9. Riskler & geri-alma

- **P7 (RewardEngine):** En yüksek risk, en düşük göreli kazanç (shaper'lar zaten ayrı). Golden oynarsa **yalnız P7 geri alınır**; P0-P6 kalıcı yeşildir.
- **P5 (app split):** Hacimli ama mekanik; `test_app_smoke` ilk render'ı kapısı.
- **Genel:** Her faz tek commit; golden kırılırsa o commit revert edilir. Strangler-fig: her adımda sistem tam çalışır.

## 10. Açık varsayımlar

- `.venv` çalışır durumda ve bağımlılıklar kurulu (P0'da doğrulanır).
- `tests/golden/metrics_baseline.csv` mevcut ve geçerli (deterministik seed=42 baseline).
- Yeni modüller `tests/test_imports.py::SAFE_MODULES` listesine eklenecek (test genişletmesi; davranış değil).

## 11. SonarCloud baseline bulguları (PR #2)

PR #2'de SonarCloud **Quality Gate PASSED** ✅. Raporlanan 8 "new issue"nun tamamı `env/portfolio_env.py`'de — `9cd6c13` ("env/ paketini izlemeye al") commit'i bu dosyayı izlemeye aldığı için yüzeye çıkan **önceden var olan** kod kokularıdır; bu oturumun (saf markdown) commit'i sıfır issue üretti. GitHub inline yorumu yoktur (resolve edilecek thread yok). Gate geçtiğinden bloke edici değildir.

Planlanan refactoring bunların çoğunu **doğal olarak** kapatır (ad-hoc yama yerine golden-kapılı fazlar içinde):

| Satır | Düzey | Plan kapsamı |
|---|---|---|
| 147 (`PortfolioEnv.__init__`, ~20 parametre) | warning | **P7**: ödül parametreleri (`eta/lambda/tau/vol/turnover/ema/w_dsr/dsr_eta/bankruptcy*`) `RewardEngine`'e taşınır → ctor parametre sayısı düşer. **P3**: `build_env` tek kurulum noktası |
| 131, 132 (`DifferentialSharpe.update`) | warning | **P7**: `DifferentialSharpe` → `env/reward.py` |
| 194 (`__init__` gövdesi) | warning | **P7**: ctor yeniden düzenlenir |
| 258, 292, 306 (`step`, reward_terms/info dict'leri) | warning | **P7**: ödül hesabı `RewardEngine`'e çıkar |
| 329 (`DiscretePortfolioEnv._discrete_to_logits`) | warning | (hedef dışı; küçük) |
| **214** (`_reset_state` rastgele-başlangıç ternary) | **failure** | **P7 içinde açıkça incelenir** — DİKKAT: `_reset_state` RNG/golden-duyarlı; YALNIZ davranış-koruyan düzeltme |

> **Şimdi ad-hoc dokunulmaz:** (a) brainstorming gate'i (spec onayı öncesi implementasyon yok); (b) `env/portfolio_env.py` golden-master'ın en duyarlı dosyası — koku temizliği davranışı değiştirip 1e-6'yı kırabilir. Doğru yer: golden-kapılı fazlar.

## 12. PR review geri bildirimi (Copilot, PR #2)

| # | Konum | Geri bildirim | Disposition |
|---|---|---|---|
| 1 | `.github/workflows/ci.yml` | Fork PR'larda `SONAR_TOKEN` expose edilmez → scan adımı patlar | ✅ **Düzeltildi**: scan adımı `push` veya ayni-repo PR ile gate'lendi; fork PR'lar yine test/coverage alır. CI dosyası refactoring kapsamı dışı, golden riski yok. |
| 2 | `env/portfolio_env.py:354` (`DiscretePortfolioEnv.step`) | Aralık-dışı `action_idx` kabul ediliyor → tüm logit `-1e6` → softmax near-uniform portföy (sessiz hatalı davranış) | ⏭ **P7'ye ertelendi**: env davranış değişikliği + golden-duyarlı dosya. P7 `DiscretePortfolioEnv.step`'i yeniden düzenlerken `0 <= a_idx < n_discrete` doğrulaması eklenir (golden-kapılı). DQN her zaman geçerli indeks ürettiğinden golden etkilenmez. |
