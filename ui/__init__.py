"""Streamlit UI paketi — SOLID P5 (SRP).

app.py'nin 7+ sorumlulugu (state, veri servisi, grafikler, sidebar, 4 sekme)
tek dosyada topluyordu. Bu paket sorumluluklari ayirir:

  ui/state.py     : session_state varsayilanlari + anahtar yardimcilari
  ui/services.py  : veri yukleme, env/ajan kurulumu, egitim/test servisleri
  ui/charts.py    : saf grafik/tablo ureticileri (UI-durumsuz)
  ui/sidebar.py   : kontrol paneli
  ui/tabs/        : 4 sekme (mdp, train, test, compare)

app.py ince giris noktasi olarak kalir ve eski API'yi re-export eder.
"""
