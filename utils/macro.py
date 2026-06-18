"""Makro rejim özellik mühendisliği — v6 (causal, leak-safe, YALIN omurga).

Ham egzojen makro panelini (VIX, S&P, faiz TNX/IRX, USDTRY, altın GC=F) küçük,
standart bir **rejim-koşullandırma** vektörüne çevirir → RL state'ine eklenir.
Tüm dönüşümler yalnız-geçmişe bakar (causal). Standardizasyon (`MacroScaler`)
YALNIZ train penceresinde fit edilir, test'e aynı uygulanır → sızıntı yok
(`utils.features.TrainScaler` ile birebir desen).

4 yalın öznitelik (kullanıcının "faiz, dolar, altın" üçlüsü + bileşik rejim):
  - regime      : tanh(vix_rel + 4·(−spx_dd) − 0.10) ∈ (−1,1)  [OMURGA — V7 amplify eder]
  - slope       : 10Y − 13W faiz farkı (getiri eğrisi eğimi)    [FAİZ]
  - usd_try_mom : USD/TRY 20g momentum                           [DOLAR]
  - gold_tl_mom : (altın×USDTRY = gram-altın/TL) 20g momentum    [ALTIN]

Gerekçe: makro-kör RL tahsisçisi tek fiyat-yoluna aşırı uyar, rejimler arası
başarısız olur. Makro rejimine koşullanmak ajanın *ortamı* öğrenmesini sağlar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MacroConfig


def add_macro_features(macro_raw: pd.DataFrame,
                       mom_window: int = MacroConfig.mom_window) -> pd.DataFrame:
    """Ham makro panel (T, seri) -> yalın makro öznitelikleri (T, 4).

    Kolon sırası `config.MacroConfig.features` ile birebir. Eksik kaynak seri
    nazikçe 0'a düşer (degrade gracefully)."""
    def col(name: str) -> pd.Series:
        if name in macro_raw.columns:
            return macro_raw[name].astype(float)
        return pd.Series(0.0, index=macro_raw.index)

    vix = col("^VIX")
    spx = col("^GSPC")
    tnx = col("^TNX")
    irx = col("^IRX")
    usdtry = col("USDTRY=X")
    gold = col("GC=F")

    feats: dict[str, pd.Series] = {}

    # OMURGA: bileşik causal rejim skoru ∈ (−1,1) — sakin < 0 < kriz.
    vix_rel = (vix / vix.rolling(252, min_periods=20).median() - 1.0).fillna(0.0)
    spx_dd  = (spx / spx.rolling(252, min_periods=20).max() - 1.0).fillna(0.0)  # <=0
    feats["regime"] = np.tanh(1.0 * vix_rel + 4.0 * (-spx_dd) - 0.10)

    # FAİZ: getiri eğrisi eğimi (10Y − 13W); resesyon/risk proxy.
    feats["slope"] = (tnx - irx).fillna(0.0)

    # DOLAR: USD/TRY momentum (TL zayıflaması).
    feats["usd_try_mom"] = usdtry.pct_change(mom_window).fillna(0.0)

    # ALTIN: gram-altın/TL ≈ altın(USD/oz)×USDTRY momentum (oran → sabit düşer).
    gold_tl = gold * usdtry
    feats["gold_tl_mom"] = gold_tl.pct_change(mom_window).fillna(0.0)

    df = pd.DataFrame(feats, index=macro_raw.index)
    cols = [c for c in MacroConfig.features if c in df.columns]
    return df[cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)


class MacroScaler:
    """Makro öznitelik matrisi için train-only z-score (tek DataFrame).

    `utils.features.TrainScaler` ile aynı sözleşme; tek matris üzerinde çalışır.
    'regime' zaten ∈(−1,1) olsa da tutarlılık için o da z-skorlanır (state için);
    ödül amplifikasyonu HAM regime'i ayrıca kullanır (bkz. train.prepare_data)."""

    def __init__(self, clip: float = 8.0):
        self.mean: pd.Series | None = None
        self.std: pd.Series | None = None
        self.clip = clip
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "MacroScaler":
        self.mean = df.mean(axis=0)
        self.std = df.std(axis=0).replace(0, 1.0)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("MacroScaler.fit() önce çağrılmalı")
        z = (df - self.mean) / self.std
        return z.clip(lower=-self.clip, upper=self.clip).fillna(0.0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
