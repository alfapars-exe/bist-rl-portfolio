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
from core.contracts import DataProvenance

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


def _cache_covers(index, start, end) -> bool:
    idx = pd.DatetimeIndex(index)
    requested = pd.bdate_range(start=start, end=end)
    if not len(idx) or not len(requested) or idx.min() > requested[0]:
        return False
    # Exchange holidays are not represented by pandas' generic business-day
    # calendar. Permit at most two nominal business days at the right boundary,
    # while still rejecting materially truncated caches.
    missing_tail = len(pd.bdate_range(idx.max(), requested[-1], inclusive="right"))
    return missing_tail <= 2


def _with_provenance(df: pd.DataFrame, *, source: str, provider: str,
                     start, end, reason: str = "", missing=()) -> pd.DataFrame:
    df.attrs["synthetic"] = source == "synthetic"
    df.attrs["provenance"] = DataProvenance(
        source=source, provider=provider, reason=reason,
        missing_tickers=tuple(missing), requested_start=str(start), requested_end=str(end),
    ).to_dict()
    return df


def _sanitize_prices(px: pd.DataFrame) -> pd.DataFrame:
    """Fiyat matrisini NaN'dan arindir: ileri/geri doldur + tamamen-bos sutunu dus.

    Gercek BIST verisinde halt/eksik gunler NaN birakabilir; tek bir NaN getiri
    env'de NAV'i zehirleyebilir (v12 env ctor NaN'i reddeder de). Cache OKUMA yolu
    onceden temizlenmiyordu -> zehirli/eksik cache'e karsi savunma. Temiz veride no-op.
    """
    px = px.ffill().bfill()
    all_nan = px.columns[px.isna().all()]
    if len(all_nan):
        px = px.drop(columns=list(all_nan))
    return px


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
            if (expected.issubset(set(px.columns)) and len(px) > 500
                    and _cache_covers(px.index, start, end)):
                # Cache tüm aralığı tutabilir; istenen [start, end]'e dilimle.
                px_slice = _sanitize_prices(px[list(tickers)])
                px_slice = px_slice.loc[
                    (px_slice.index >= pd.Timestamp(start)) &
                    (px_slice.index <= pd.Timestamp(end))
                ]
                if len(px_slice) > 100:
                    return _with_provenance(px_slice, source="real", provider="parquet-cache",
                                            start=start, end=end)
            print("[INFO] cache uyumsuz veya dilim boş, yeniden indiriliyor ...")
        except Exception as exc:
            print(f"[WARN] parquet okunamadı ({exc!r}); yeniden indiriliyor")

    synthetic = False
    try:
        import yfinance as yf
        inclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download(
            tickers, start=start, end=inclusive_end,
            auto_adjust=True, progress=False, threads=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            px = data["Close"].copy()
        else:
            px = data[["Close"]].copy()
        # yfinance bazen tz-aware / time-bilesenli index doner; sentetik-doldurma reindex'i
        # (naive bdate_range) eslesmezse TUM sentetik sutunlar NaN -> son dropna frame'i
        # BOSALTIR (HF Space MIXED-bos hatasi). Index'i tz-naive GUNE normalize et.
        _idx = pd.to_datetime(px.index)
        px.index = (_idx.tz_localize(None) if _idx.tz is not None else _idx).normalize()
        px = px.dropna(axis=1, thresh=int(0.9 * len(px)))
        px = px.ffill().dropna()
        if px.shape[1] < 10 or len(px) < 100:
            raise RuntimeError("Too few tickers returned")
        px = px[[c for c in tickers if c in px.columns]]
        # Evren degismezligi: yfinance kismi dondurduyse (bazi ticker'lar eksik /
        # >%10 NaN -> dropna ile dustu) eksikleri sentetik ile doldurup TAM BIST28'i
        # garanti et. Aksi halde N degisir -> UI ASSET_NAMES (sabit 29) ile uyumsuzluk
        # + kayitli ajan state-dim (397) ile uyumsuzluk olur.
        miss = [c for c in tickers if c not in px.columns]
        if miss:
            warnings.warn(
                f"{len(miss)} ticker yfinance'tan gelmedi; sentetik ile dolduruldu: "
                f"{miss}", RuntimeWarning, stacklevel=2)
            synth = _synthetic_bist(miss, start, end).reindex(px.index).ffill()
            for c in miss:
                px[c] = synth[c].to_numpy()
        px = px[list(tickers)].ffill().dropna()
        # EMNIYET AGI: islenmis frame bos/cok-kisa veya NaN'li ise (index eslesmezligi vb.)
        # bozuk MIXED frame'i DONDURME -> except'e dusur (CSV/sentetik fallback gecerli veri verir).
        if len(px) < 100 or bool(px.isna().any().any()):
            raise RuntimeError(f"islenmis yfinance verisi gecersiz ({len(px)} satir)")
        if miss:
            _with_provenance(px, source="mixed", provider="yfinance+synthetic",
                             start=start, end=end, reason="missing yfinance tickers", missing=miss)
        else:
            _with_provenance(px, source="real", provider="yfinance", start=start, end=end)
    except Exception as exc:
        # SENTETIGE DUSMEDEN ONCE: kanonik results/bist30_prices.csv (tam 2015-2024 GERCEK
        # BIST verisi). Bu dosya .gitignore'da DEGIL -> HF Space'e yuklenir (data/*.parquet
        # cache'i gitignore-aware upload nedeniyle Space'e GIDEMEZ). Ag/cache yoksa bunu kullan.
        csv_fb = RESULTS_DIR / "bist30_prices.csv"
        if use_cache and csv_fb.exists():
            try:
                pxc = pd.read_csv(csv_fb, index_col=0, parse_dates=True)
                if (set(tickers).issubset(set(pxc.columns))
                        and _cache_covers(pxc.index, start, end)):
                    pxc = pxc[list(tickers)]
                    pxc = pxc.loc[(pxc.index >= pd.Timestamp(start)) &
                                  (pxc.index <= pd.Timestamp(end))].ffill().dropna()
                    if len(pxc) > 100:
                        print("[INFO] yfinance yok; results/bist30_prices.csv (GERCEK BIST) kullanildi")
                        return _with_provenance(pxc, source="real", provider="csv-cache",
                                                start=start, end=end)
            except Exception as exc_csv:
                print(f"[WARN] bist30_prices.csv okunamadi ({exc_csv!r})")
        # Sessiz yutma yok: stderr'e gorunur uyari (CI loglari + kullanici).
        warnings.warn(
            f"yfinance basarisiz ({exc!r}); SENTETIK GBM verisi uretiliyor — "
            "bu GERCEK BIST fiyati DEGIL, sonuclar yalnizca demo amaclidir!",
            RuntimeWarning, stacklevel=2,
        )
        print(f"[WARN] yfinance başarısız ({exc!r}); sentetik BIST verisi üretiliyor")
        px = _synthetic_bist(tickers, start, end)
        synthetic = True
        _with_provenance(px, source="synthetic", provider="synthetic-gbm",
                         start=start, end=end, reason=repr(exc))

    if synthetic:
        # KRITIK: sentetik veri CACHE'E YAZILMAZ. Onceki surum yaziyordu;
        # bir kez ag hatasi -> sonraki TUM calistirmalar cache'ten sessizce
        # sahte veri okuyordu (cache zehirlenmesi).
        print("[WARN] sentetik veri cache'e yazılmadı; ağ gelince gerçek veri indirilecek")
        return px
    if px.attrs.get("provenance", {}).get("source") == "mixed":
        print("[WARN] karma gercek/sentetik veri cache'e yazilmadi")
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
            if len(have) >= 3 and len(mc) > 500 and _cache_covers(mc.index, start, end):
                mc_slice = mc[have]
                mc_slice = mc_slice.loc[
                    (mc_slice.index >= pd.Timestamp(start)) &
                    (mc_slice.index <= pd.Timestamp(end))
                ]
                if len(mc_slice) > 100:
                    return _with_provenance(mc_slice, source="real", provider="parquet-cache",
                                            start=start, end=end)
        except Exception as exc:
            print(f"[WARN] makro cache okunamadı ({exc!r}); yeniden indiriliyor")

    synthetic = False
    try:
        import yfinance as yf
        inclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.download(series, start=start, end=inclusive_end, auto_adjust=True,
                           progress=False, threads=True)
        mc = (data["Close"].copy() if isinstance(data.columns, pd.MultiIndex)
              else data[["Close"]].copy())
        mc = mc.ffill().dropna(axis=1, how="all")
        mc = mc[[s for s in series if s in mc.columns]]
        if mc.shape[1] < 3 or len(mc) < 100:
            raise RuntimeError("Too few macro series returned")
        missing = [s for s in series if s not in mc.columns]
        _with_provenance(mc, source="mixed" if missing else "real", provider="yfinance",
                         start=start, end=end,
                         reason="missing macro series" if missing else "", missing=missing)
    except Exception as exc:
        # SENTETIGE DUSMEDEN ONCE: results/macro_raw.csv (gercek makro; gitignore'da DEGIL
        # -> HF Space'te bulunur). Ag/cache yoksa bunu kullan.
        csv_fb = RESULTS_DIR / "macro_raw.csv"
        if use_cache and csv_fb.exists():
            try:
                mcc = pd.read_csv(csv_fb, index_col=0, parse_dates=True)
                have = [s for s in series if s in mcc.columns]
                if len(have) >= 3:
                    mcc = mcc[have].loc[(mcc.index >= pd.Timestamp(start)) &
                                        (mcc.index <= pd.Timestamp(end))].ffill()
                    if len(mcc) > 100 and _cache_covers(mcc.index, start, end):
                        print("[INFO] yfinance yok; results/macro_raw.csv (GERCEK makro) kullanildi")
                        return _with_provenance(mcc, source="real", provider="csv-cache",
                                                start=start, end=end)
            except Exception as exc_csv:
                print(f"[WARN] macro_raw.csv okunamadi ({exc_csv!r})")
        warnings.warn(
            f"yfinance makro basarisiz ({exc!r}); SENTETIK makro uretiliyor — "
            "gercek piyasa verisi DEGIL!", RuntimeWarning, stacklevel=2)
        print(f"[WARN] yfinance makro başarısız ({exc!r}); sentetik makro üretiliyor")
        mc = _synthetic_macro(series, start, end)
        synthetic = True
        _with_provenance(mc, source="synthetic", provider="synthetic-macro",
                         start=start, end=end, reason=repr(exc))

    if synthetic:
        print("[WARN] sentetik makro cache'e yazılmadı")
        return mc
    if mc.attrs.get("provenance", {}).get("source") == "mixed":
        print("[WARN] eksik serili karma makro veri cache'e yazilmadi")
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


def resample_to_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Fiyat/feature/makro DataFrame'ini istenen adım granülerliğine indirgeer.

    Parametreler
    ------------
    df : pd.DataFrame
        DatetimeIndex'li DataFrame (fiyat, feature veya makro).
    granularity : str
        "daily"   -> df'i AYNEN döndür (NO-OP; golden-güvenli, RNG sırası korunur).
        "monthly" -> aylık ortalama (pandas ME kuralı).
        "yearly"  -> yıllık ortalama (pandas YE kuralı).

    Dönüş
    ------
    pd.DataFrame
        DatetimeIndex korunur. "daily" dışında .ffill().bfill() uygulanır
        (boş dönem NaN'larına karşı güvenlik).

    Not: feature'lar her zaman günlük hesaplanır (add_features DEĞİŞMEZ).
    Bu fonksiyon yalnızca downstream adımda (env/UI) granülerliği indirger.
    """
    if granularity == "daily":
        # NO-OP: aynı nesneyi döndür — golden testleri etkilemez, RNG sırası korunur.
        return df

    rule_map = {
        "monthly": "ME",   # month-end
        "yearly":  "YE",   # year-end
    }
    rule = rule_map.get(granularity)
    if rule is None:
        raise ValueError(
            f"Geçersiz granularity={granularity!r}; "
            f"geçerli seçenekler: {('daily', 'monthly', 'yearly')}"
        )

    resampled = df.resample(rule).mean()
    # Boş ay/yıl periyotlarında oluşabilecek NaN'lara karşı güvenlik.
    resampled = resampled.ffill()
    return resampled


def resample_to_step_days(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """N-günlük blok-ortalama resample (v12 tek-adım modeli — granülerliğin yerine).

    n=1  -> df'i AYNEN döndür (NO-OP; golden-güvenli, RNG sırası korunur).
    n>=2 -> df.resample(f"{n}D").mean().ffill().bfill() (boş blok NaN koruması).

    ÇAĞRI SIRASI (leak-safe): add_features (GÜNLÜK, tam seri) -> train_test_split
    -> resample_to_step_days (HER split AYRI). Resample atomik nokta üretir -> train/test
    sınırı blok-hizalı, karışma yok. Feature'lar HER ZAMAN günlük hesaplanır (add_features DEĞİŞMEZ).
    DatetimeIndex korunur (env `self.dates = prices.index` için).
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"step_days en az 1 olmali; {n} geldi")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample_to_step_days DatetimeIndex gerektirir")
    if n == 1:
        df.attrs["session_counts"] = tuple([1] * len(df))
        df.attrs["step_days"] = 1
        return df
    if len(df) == 0:
        out = df.copy()
        out.attrs["session_counts"] = ()
        out.attrs["step_days"] = n
        return out

    # Consecutive exchange sessions, not calendar-day buckets. Each selected row
    # is a tradable period-end observation. Keep the trailing partial period.
    positions = list(range(n - 1, len(df), n))
    counts = [n] * len(positions)
    if not positions or positions[-1] != len(df) - 1:
        previous = positions[-1] + 1 if positions else 0
        positions.append(len(df) - 1)
        counts.append(len(df) - previous)
    out = df.iloc[positions].copy()
    out.attrs.update(df.attrs)
    out.attrs["session_counts"] = tuple(int(x) for x in counts)
    out.attrs["step_days"] = n
    return out


def align_macro(macro_raw: pd.DataFrame, index) -> pd.DataFrame:
    """Makroyu BIST işlem takvimine (index) reindex + ffill/bfill (causal)."""
    out = macro_raw.reindex(index).ffill()
    out.attrs.update(macro_raw.attrs)
    return out


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
    train = df[df.index < split].copy()
    test = df[df.index >= split].copy()
    train.attrs.update(df.attrs)
    test.attrs.update(df.attrs)
    return train, test


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
