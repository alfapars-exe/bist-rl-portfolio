"""BIST 30 veri indirme ve ön işleme (28 hisse, KOZAA + KOZAL hariç).

- Günlük ayarlı kapanış fiyatlarını yfinance ile indirir.
- İlk çalıştırmada `data/prices.parquet` olarak cache'ler.
- Ağ yoksa BIST istatistiklerine kalibre edilmiş sentetik GBM fallback devreye girer.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import DataConfig, MacroConfig

# BIST 30 tickers — KOZAA.IS ve KOZAL.IS prompt gereği hariç tutuldu (28 hisse).
BIST28 = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS",
    "EREGL.IS", "FROTO.IS", "GARAN.IS", "HEKTS.IS", "ISCTR.IS",
    "KCHOL.IS", "KRDMD.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
    "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS",
    "TOASO.IS", "TUPRS.IS", "VAKBN.IS", "YKBNK.IS", "ENKAI.IS",
    "DOHOL.IS", "TTKOM.IS", "MGROS.IS",
]
# Geriye uyumluluk
BIST30 = BIST28

# Geriye uyum: modül sabitleri DataConfig'e işaret eder (tek kaynak).
# Bu satırları import eden mevcut kod (main.py, app.py, testler) kırılmaz.
_dc = DataConfig()
START = _dc.start      # "2015-01-01"
END   = _dc.end        # "2024-12-31"
SPLIT = _dc.train_end  # "2022-01-01"

BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
RESULTS_DIR  = BASE_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = DATA_DIR / "prices.parquet"
MACRO_PARQUET = DATA_DIR / "macro_raw.parquet"


def download_bist(tickers=BIST28, start=None, end=None,
                  use_cache: bool = True) -> pd.DataFrame:
    """28 hisselik (T, N) ayarlı kapanış fiyat matrisi döner.

    Parametreler
    ------------
    start : str | None
        Başlangıç tarihi (dahil). None → DataConfig.start ("2015-01-01").
    end : str | None
        Bitiş tarihi (dahil). None → DataConfig.end ("2024-12-31").

    Önce parquet cache'i dener; varsa verilen [start, end] aralığına dilimler.
    Cache uyumsuzsa veya yoksa yfinance'tan indirir. yfinance başarısızsa BIST-
    benzeri sentetik GBM üretir (deney yine çalışır).
    """
    _dc_local = DataConfig()
    if start is None:
        start = _dc_local.start
    if end is None:
        end = _dc_local.end

    if use_cache and PARQUET_PATH.exists():
        try:
            px = pd.read_parquet(PARQUET_PATH)
            px.index = pd.to_datetime(px.index)
            expected = set(tickers)
            if expected.issubset(set(px.columns)) and len(px) > 500:
                # Cache tüm aralığı tutabilir; istenen [start, end]'e dilimle.
                px_slice = px[list(tickers)]
                px_slice = px_slice.loc[
                    (px_slice.index >= pd.Timestamp(start)) &
                    (px_slice.index <= pd.Timestamp(end))
                ]
                if len(px_slice) > 100:
                    return px_slice
            print("[INFO] cache uyumsuz veya dilim boş, yeniden indiriliyor ...")
        except Exception as exc:
            print(f"[WARN] parquet okunamadı ({exc!r}); yeniden indiriliyor")

    synthetic = False
    try:
        import yfinance as yf
        data = yf.download(
            tickers, start=start, end=end,
            auto_adjust=True, progress=False, threads=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            px = data["Close"].copy()
        else:
            px = data[["Close"]].copy()
        px = px.dropna(axis=1, thresh=int(0.9 * len(px)))
        px = px.ffill().bfill().dropna()
        if px.shape[1] < 10:
            raise RuntimeError("Too few tickers returned")
        px = px[[c for c in tickers if c in px.columns]]
    except Exception as exc:
        # Sessiz yutma yok: stderr'e gorunur uyari (CI loglari + kullanici).
        warnings.warn(
            f"yfinance basarisiz ({exc!r}); SENTETIK GBM verisi uretiliyor — "
            "bu GERCEK BIST fiyati DEGIL, sonuclar yalnizca demo amaclidir!",
            RuntimeWarning, stacklevel=2,
        )
        print(f"[WARN] yfinance başarısız ({exc!r}); sentetik BIST verisi üretiliyor")
        px = _synthetic_bist(tickers, start, end)
        synthetic = True
        px.attrs["synthetic"] = True   # programatik kaynak izi (provenance)

    if synthetic:
        # KRITIK: sentetik veri CACHE'E YAZILMAZ. Onceki surum yaziyordu;
        # bir kez ag hatasi -> sonraki TUM calistirmalar cache'ten sessizce
        # sahte veri okuyordu (cache zehirlenmesi).
        print("[WARN] sentetik veri cache'e yazılmadı; ağ gelince gerçek veri indirilecek")
        return px

    try:
        px.to_parquet(PARQUET_PATH)
    except Exception as exc:
        print(f"[WARN] parquet yazılamadı ({exc!r}); CSV fallback")
        px.to_csv(DATA_DIR / "prices.csv")
    return px


def _synthetic_bist(tickers, start, end) -> pd.DataFrame:
    """BIST istatistiklerine kalibre edilmiş çok değişkenli GBM seti."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start=start, end=end)
    T = len(idx)
    N = len(tickers)
    mu    = rng.normal(0.18, 0.08, size=N) / 252.0
    sigma = np.clip(rng.normal(0.40, 0.10, size=N), 0.2, 0.8) / np.sqrt(252.0)
    base = rng.normal(0, 1, size=(N, 4))
    C = np.corrcoef(base @ rng.normal(0, 1, size=(4, N)) + 0.5 * rng.normal(0, 1, (N, N)))
    C = 0.6 * C + 0.4 * np.eye(N)
    L = np.linalg.cholesky(C)
    Z = rng.normal(0, 1, size=(T, N)) @ L.T
    logret = mu + sigma * Z
    crash_idx = (idx >= "2020-03-05") & (idx <= "2020-03-30")
    logret[crash_idx] += rng.normal(-0.015, 0.02, size=(crash_idx.sum(), N))
    rally = (idx >= "2021-06-01") & (idx <= "2022-12-31")
    logret[rally] += 0.0012
    price = 100.0 * np.exp(np.cumsum(logret, axis=0))
    return pd.DataFrame(price, index=idx, columns=tickers)


# ---------------------------------------------------------------------
# v6: Makro rejim verisi (faiz/dolar/altin) — egzojen, BIST takvimine hizali.
# ---------------------------------------------------------------------
def download_macro(series=tuple(MacroConfig.series), start=None, end=None,
                   use_cache: bool = True) -> pd.DataFrame:
    """Ham makro seri matrisi (T, M): VIX, S&P, faiz (TNX/IRX), USDTRY, altin (GC=F).

    Parametreler
    ------------
    start : str | None
        Başlangıç tarihi (dahil). None → DataConfig.start ("2015-01-01").
    end : str | None
        Bitiş tarihi (dahil). None → DataConfig.end ("2024-12-31").

    parquet cache -> yfinance -> sentetik fallback. Cache varsa [start, end]'e
    dilimler. Sentetik veri CACHE'E YAZILMAZ (download_bist ile ayni zehirlenme
    korumasi)."""
    _dc_local = DataConfig()
    if start is None:
        start = _dc_local.start
    if end is None:
        end = _dc_local.end

    series = list(series)
    if use_cache and MACRO_PARQUET.exists():
        try:
            mc = pd.read_parquet(MACRO_PARQUET)
            mc.index = pd.to_datetime(mc.index)
            have = [s for s in series if s in mc.columns]
            if len(have) >= 3 and len(mc) > 500:
                mc_slice = mc[have]
                mc_slice = mc_slice.loc[
                    (mc_slice.index >= pd.Timestamp(start)) &
                    (mc_slice.index <= pd.Timestamp(end))
                ]
                if len(mc_slice) > 100:
                    return mc_slice
        except Exception as exc:
            print(f"[WARN] makro cache okunamadı ({exc!r}); yeniden indiriliyor")

    synthetic = False
    try:
        import yfinance as yf
        data = yf.download(series, start=start, end=end, auto_adjust=True,
                           progress=False, threads=True)
        mc = (data["Close"].copy() if isinstance(data.columns, pd.MultiIndex)
              else data[["Close"]].copy())
        mc = mc.ffill().bfill().dropna(axis=1, how="all")
        mc = mc[[s for s in series if s in mc.columns]]
        if mc.shape[1] < 3:
            raise RuntimeError("Too few macro series returned")
    except Exception as exc:
        warnings.warn(
            f"yfinance makro basarisiz ({exc!r}); SENTETIK makro uretiliyor — "
            "gercek piyasa verisi DEGIL!", RuntimeWarning, stacklevel=2)
        print(f"[WARN] yfinance makro başarısız ({exc!r}); sentetik makro üretiliyor")
        mc = _synthetic_macro(series, start, end)
        synthetic = True

    if synthetic:
        print("[WARN] sentetik makro cache'e yazılmadı")
        return mc
    try:
        mc.to_parquet(MACRO_PARQUET)
    except Exception as exc:
        print(f"[WARN] makro parquet yazılamadı ({exc!r})")
    return mc


def _synthetic_macro(series, start, end) -> pd.DataFrame:
    """Determinist sentetik makro (offline + testler). TL sürekli zayıflar (gerçekçi)."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(start=start, end=end)
    T = len(idx)
    vix    = np.clip(18 + 9 * np.abs(np.cumsum(rng.normal(0, 0.25, T)) % 3.0), 9.0, 80.0)
    gspc   = 1500.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, T)))
    tnx    = np.clip(3.0 + np.cumsum(rng.normal(0, 0.02, T)), 0.5, 6.0)
    irx    = np.clip(tnx - 0.5 + rng.normal(0, 0.1, T), 0.05, None)
    usdtry = 2.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, T)))   # TL erir
    gold   = 1300.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, T)))  # altın USD/oz
    full = {"^VIX": vix, "^GSPC": gspc, "^TNX": tnx, "^IRX": irx,
            "USDTRY=X": usdtry, "GC=F": gold}
    df = pd.DataFrame(full, index=idx)
    return df[[s for s in series if s in df.columns]]


def align_macro(macro_raw: pd.DataFrame, index) -> pd.DataFrame:
    """Makroyu BIST işlem takvimine (index) reindex + ffill/bfill (causal)."""
    return macro_raw.reindex(index).ffill().bfill()


def train_test_split(df: pd.DataFrame, split=None):
    """DataFrame'i train (< split) ve test (>= split) olarak ikiye böler.

    Parametreler
    ------------
    split : str | None
        Bölünme tarihi. None → DataConfig.train_end ("2022-01-01").
        Train kesinlikle test'ten önce gelir; sızıntı yoktur.
    """
    if split is None:
        split = DataConfig().train_end
    return df[df.index < split], df[df.index >= split]


if __name__ == "__main__":
    px = download_bist()
    print(f"Prices shape: {px.shape}, {px.index[0].date()} → {px.index[-1].date()}")
    print(f"Tickers ({len(px.columns)}): {list(px.columns)}")
    px.to_csv(RESULTS_DIR / "bist30_prices.csv")
    # Feature CSV'leri artık utils.features tarafından üretilir; CLI için orada da kopyasını yaz
    try:
        from utils.features import add_features
        feats = add_features(px)
        for name, f in feats.items():
            f.to_csv(RESULTS_DIR / f"feat_{name}.csv")
        print("Saved prices & features.")
    except ImportError:
        print("Saved prices only (utils.features not yet available).")
