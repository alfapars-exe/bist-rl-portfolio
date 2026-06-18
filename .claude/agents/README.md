# Ajan Yapısı — BIST 28 Portföy RL

Bu klasör, projenin **Claude Code alt-ajanlarını** (`.claude/agents/*.md`) içerir. Ana
oturum, bir görevi ilgili ajanın `description`'ına göre **otomatik yönlendirir**; ayrıca
`@ajan-adi` ile elle de çağırabilirsin. Roster ve refleks kuralları için kökteki
[`CLAUDE.md`](../../CLAUDE.md)'ye de bak.

## Roster (15 ajan)

| Ajan | Ne zaman | Model |
|------|----------|-------|
| **orkestrator-planlayici** | Büyük/çok-disiplinli hedefi parçalara böl, ajanlara dağıt | opus |
| **rl-arastirma-muhendisi** | DQN/PPO/SAC/TD3, ödül şekillendirme, state tasarımı, eğitim kararlılığı | opus |
| **backend-muhendisi** | `core/`, `main.py`, `train.py`, `config.py`, pipeline & persistence | sonnet |
| **frontend-muhendisi** | Streamlit `app.py`/`ui/`/`plots.py` — 4 sekme, grafikler | sonnet |
| **veri-muhendisi** | `data.py`, `utils/features.py`, makro; **sızıntısızlık** | sonnet |
| **urun-yoneticisi** | Kapsam, kabul kriterleri, kullanıcı akışı, önceliklendirme | opus |
| **test-muhendisi** | pytest, golden-master ≤1e-6, leak-safety, determinizm | sonnet |
| **fon-yoneticisi** | Mandat/ister değerlendirme, baseline kıyas, risk iştahı | opus |
| **finansal-regulasyon-uzmani** | BIST komisyon/BSMV/takas/lot — işlem maliyeti gerçekliği | opus |
| **finans-uzmani** | Finansal "neden", makro rejim, terim doğruluğu | opus |
| **kantitatif-strateji-uzmani** | DSR/PBO/CSCV, Monte-Carlo, walk-forward, VaR/CVaR | opus |
| **akademisyen-degerlendirici** | Rapor §9.1–§9.9 akademik kalite & literatür | opus |
| **proje-rubrik-bekcisi** | `RL_FinalProje.pdf` ister-uyum denetimi (salt-okunur) | opus |
| **mimari-hafiza-koruyucusu** | `docs/MIMARI_HAFIZA.md` canlı mimari bellek | sonnet |
| **dokumantasyon-yazari** | README/DOKUMANTASYON/KOD_EKI üretim & senkron | sonnet |

## Örnek tetiklemeler

- "PPO entropy katsayısını ayarla, ajan keşfetmiyor" → **rl-arastirma-muhendisi**
- "Tab 3'e Q-değeri bar grafiği ekle" → **frontend-muhendisi**
- "Bu Sharpe overfitting mi, DSR/PBO bak" → **kantitatif-strateji-uzmani**
- "Bir hisse almanın gerçek maliyeti ne, η gerçekçi mi" → **finansal-regulasyon-uzmani**
- "Teslim PDF isterlerini karşılıyor mu" → **proje-rubrik-bekcisi**
- "Şu büyük refactor'u baştan sona planla" → **orkestrator-planlayici**

## Refleks kuralları (ana oturum uygular)

- Kaynak kod değişti → **test-muhendisi** (ilgili testler + golden).
- Büyük mimari değişiklik/karar → **mimari-hafiza-koruyucusu** (`docs/MIMARI_HAFIZA.md`).
- Teslim öncesi → **proje-rubrik-bekcisi** + **akademisyen-degerlendirici**.
- Kod değişikliği dokümana yansıyor → **dokumantasyon-yazari**.

## Kapasite sınırları (önemli)

- **Alt-ajanlar başka alt-ajan çağıramaz** ve **durumsuzdur** (her çağrı temiz bağlam).
  Bu yüzden `orkestrator-planlayici` *dağıtım planı* üretir (ana oturum uygular), ve
  `mimari-hafiza-koruyucusu` belleği **`docs/MIMARI_HAFIZA.md` dosyasında** tutar.
- **En az ayrıcalık**: salt-değerlendiren ajanlara (`proje-rubrik-bekcisi`,
  `akademisyen-degerlendirici`, `fon-yoneticisi`, `finans-uzmani`) `Edit`/`Bash` verilmedi —
  kanonik teslimi yanlışlıkla bozmasınlar diye.

## Düzenleme

Her dosya YAML frontmatter (`name` = dosya adı, `description`, `tools`, `model`) + Türkçe
sistem promptu taşır. Yeni ajan eklerken bu README'yi ve kök `CLAUDE.md` roster'ını güncelle.
