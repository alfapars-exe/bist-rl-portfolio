"""Özellik mühendisliği ve eğitim-seti z-score ölçekleyicisi.

Durum vektörü için her hisseden 12 teknik öznitelik üretilir (hepsi Close-türevli,
ölçek-bağımsız ve yalnız-geçmişe-bakar):
  - logret    : günlük log getiri
  - ma5/ma20  : 5g / 20g yüzde değişim
  - vol20/vol60 : 20g / 60g logret standart sapması
  - rsi       : 14 günlük RSI (0-1 ölçekli)
  - macd_hist : MACD histogramı (EMA12-EMA26-sinyal9), close ile normalize
  - bb_pctb   : merkezli Bollinger %b (20g): (close-SMA)/(2*std) = 2*%b-1, ~[-1,1]
  - bb_bw     : Bollinger bant genişliği (20g)
  - roc10     : 10g değişim
  - mom60     : 60g momentum
  - ema_dist  : EMA50'ye göreli uzaklık

`TrainScaler` istatistikleri YALNIZCA eğitim kümesinde fit eder;
test dönemine aynı istatistikler uygulanır → veri sızıntısı yok.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def add_features(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Close (T,N) matrisinden zenginleştirilmiş teknik feature dict'i döner (ham).

    Anahtar sırası `config.FEATURES` ile aynıdır. Tüm göstergeler yalnız-geçmişe
    bakar (rolling/ewm/pct_change/diff — causal) → ileri-bakış (lookahead) yok.
    """
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices bos olmayan DatetimeIndex'li DataFrame olmali")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("prices pozitif ve sonlu olmali")
    close = prices
    logret = np.log(close).diff().fillna(0.0)
    ma5    = close.pct_change(5).fillna(0.0)
    ma20   = close.pct_change(20).fillna(0.0)
    vol20  = logret.rolling(20).std().fillna(0.0)
    delta  = close.diff()
    up     = delta.clip(lower=0).rolling(14).mean()
    dn     = (-delta.clip(upper=0)).rolling(14).mean()
    rsi    = (100 - 100 / (1 + up / (dn + 1e-9))).fillna(50) / 100.0

    # --- v2 yeni göstergeler (Close-türevli, ölçek-bağımsız) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = ((macd - signal) / close).fillna(0.0)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_pctb = ((close - sma20) / (2 * std20 + 1e-9)).fillna(0.0)
    bb_bw   = ((4 * std20) / (sma20 + 1e-9)).fillna(0.0)
    roc10   = close.pct_change(10).fillna(0.0)
    mom60   = close.pct_change(60).fillna(0.0)
    vol60   = logret.rolling(60).std().fillna(0.0)
    ema50   = close.ewm(span=50, adjust=False).mean()
    ema_dist = ((close - ema50) / (ema50 + 1e-9)).fillna(0.0)

    return {"logret": logret, "ma5": ma5, "ma20": ma20, "vol20": vol20, "rsi": rsi,
            "macd_hist": macd_hist, "bb_pctb": bb_pctb, "bb_bw": bb_bw, "roc10": roc10,
            "mom60": mom60, "vol60": vol60, "ema_dist": ema_dist}


class TrainScaler:
    """Özellik-bazında (her feature tipi × her hisse) z-score ölçekleyici.

    fit(): eğitim DataFrame'lerinden mean/std hesaplar
    transform(): aynı istatistiklerle test DataFrame'lerini ölçekler
    fit_transform(): tek adımda yapar
    """

    def __init__(self, clip: float = 8.0):
        self.means: Dict[str, pd.Series] = {}
        self.stds:  Dict[str, pd.Series] = {}
        self.clip = clip
        self.fitted = False

    def fit(self, feats: Dict[str, pd.DataFrame]) -> "TrainScaler":
        if not feats or any(df.empty for df in feats.values()):
            raise ValueError("TrainScaler.fit bos feature matrisi kabul etmez")
        for k, df in feats.items():
            self.means[k] = df.mean(axis=0)
            self.stds[k]  = df.std(axis=0).replace(0, 1.0)
        self.fitted = True
        return self

    def transform(self, feats: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if not self.fitted:
            raise RuntimeError("TrainScaler.fit() önce çağrılmalı")
        out: Dict[str, pd.DataFrame] = {}
        for k, df in feats.items():
            mu = self.means[k]
            sd = self.stds[k]
            z = (df - mu) / sd
            z = z.clip(lower=-self.clip, upper=self.clip)
            out[k] = z.fillna(0.0)
        return out

    def fit_transform(self, feats_train: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        self.fit(feats_train)
        return self.transform(feats_train)
