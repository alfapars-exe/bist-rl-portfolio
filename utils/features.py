"""Özellik mühendisliği ve eğitim-seti z-score ölçekleyicisi.

Durum vektörü için her hisseden 5 teknik öznitelik üretilir:
  - logret : günlük log getiri
  - ma5    : 5 günlük yüzde değişim
  - ma20   : 20 günlük yüzde değişim
  - vol20  : 20 günlük logret standart sapması
  - rsi    : 14 günlük RSI (Wilder; 0-1 aralığında ölçeklenmiş)

`TrainScaler` istatistikleri YALNIZCA eğitim kümesinde fit eder;
test dönemine aynı istatistikler uygulanır → veri sızıntısı yok.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def add_features(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """28 hisselik fiyat matrisinden 5 özellik DataFrame'i döner (ham, standardize değil)."""
    logret = np.log(prices).diff().fillna(0.0)
    ma5    = prices.pct_change(5).fillna(0.0)
    ma20   = prices.pct_change(20).fillna(0.0)
    vol20  = logret.rolling(20).std().fillna(0.0)
    delta  = prices.diff()
    up     = delta.clip(lower=0).rolling(14).mean()
    dn     = (-delta.clip(upper=0)).rolling(14).mean()
    rsi    = (100 - 100 / (1 + up / (dn + 1e-9))).fillna(50) / 100.0
    return dict(logret=logret, ma5=ma5, ma20=ma20, vol20=vol20, rsi=rsi)


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
