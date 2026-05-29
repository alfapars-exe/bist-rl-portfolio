"""BIST 30 veri indirme ve ön işleme (28 hisse, KOZAA + KOZAL hariç).

- Günlük ayarlı kapanış fiyatlarını yfinance ile indirir.
- İlk çalıştırmada `data/prices.parquet` olarak cache'ler.
- Ağ yoksa BIST istatistiklerine kalibre edilmiş sentetik GBM fallback devreye girer.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

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

START = "2015-01-01"
END   = "2024-12-31"
SPLIT = "2022-01-01"

BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
RESULTS_DIR  = BASE_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = DATA_DIR / "prices.parquet"


def download_bist(tickers=BIST28, start=START, end=END,
                  use_cache: bool = True) -> pd.DataFrame:
    """28 hisselik (T, N) ayarlı kapanış fiyat matrisi döner.

    Önce parquet cache'i dener; yoksa yfinance'tan indirir. yfinance erişimi
    başarısızsa BIST-benzeri sentetik GBM seti üretir (deney yine çalışır).
    """
    if use_cache and PARQUET_PATH.exists():
        try:
            px = pd.read_parquet(PARQUET_PATH)
            px.index = pd.to_datetime(px.index)
            expected = set(tickers)
            if expected.issubset(set(px.columns)) and len(px) > 500:
                return px[list(tickers)]
            print("[INFO] cache uyumsuz, yeniden indiriliyor ...")
        except Exception as exc:
            print(f"[WARN] parquet okunamadı ({exc!r}); yeniden indiriliyor")

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
        print(f"[WARN] yfinance başarısız ({exc!r}); sentetik BIST verisi üretiliyor")
        px = _synthetic_bist(tickers, start, end)

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


def train_test_split(df: pd.DataFrame, split=SPLIT):
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
