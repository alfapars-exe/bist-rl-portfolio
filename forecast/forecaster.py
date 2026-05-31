"""CNN-LSTM bir-adim getiri tahmincisi — predict-then-optimize (Faz V4).

Train-only fit edilir; her (varlik, t) icin t+1 log-getirisini t'de biten
pencereden tahmin eder. Bu tahmin state'e 'forecast' feature'i olarak eklenir
(ajan t'de t+1 hakkinda bir ON-GORUYU gorur — gerçek gelecek degil, modelin
ciktisi). Literatur: hibrit LSTM->PPO (arXiv:2511.17963), predict-then-optimize
(arXiv:2501.17992).

SIZINTISIZLIK (kritik): tahmin penceresi yalniz <=t; model YALNIZ train'de fit
edilir, test'i hic gormez. Determinizm: set_seed(seed) ile fit tekrar-uretilebilir
(golden gecerli kalir). test_forecaster bunlari kilitler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from agents.common import get_device, set_seed


class ReturnForecaster(nn.Module):
    """Conv1d -> LSTM -> Linear; (B, window, 1) -> (B,) bir-adim getiri tahmini."""

    def __init__(self, window: int = 20, conv_ch: int = 16, hidden: int = 32):
        super().__init__()
        self.conv = nn.Conv1d(1, conv_ch, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(conv_ch, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x.transpose(1, 2)))   # (B, conv_ch, window)
        out, _ = self.lstm(h.transpose(1, 2))          # (B, window, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)    # (B,)


def _make_windows(logret: np.ndarray, window: int):
    """logret (T,N) -> X (M,window,1), y (M,); ornek (t,j) icin y = logret[t+1,j]."""
    T, N = logret.shape
    xs, ys = [], []
    for j in range(N):
        col = logret[:, j]
        for t in range(window - 1, T - 1):
            xs.append(col[t - window + 1: t + 1])
            ys.append(col[t + 1])
    x = np.asarray(xs, dtype=np.float32)[:, :, None]
    y = np.asarray(ys, dtype=np.float32)
    return x, y


def _logret(prices: pd.DataFrame) -> np.ndarray:
    return np.log(prices).diff().fillna(0.0).values.astype(np.float32)


def build_forecast_feature(prices_full: pd.DataFrame, prices_train: pd.DataFrame,
                           window: int = 20, conv_ch: int = 16, hidden: int = 32,
                           epochs: int = 4, lr: float = 1e-3, batch: int = 256,
                           seed: int = 42) -> pd.DataFrame:
    """YALNIZ train'de fit; tüm seri için bir-adım getiri tahmini (T,N) döner.

    prices_train ile fit edilir (sizinti yok); prices_full uzerinde causal tahmin
    uretilir (her t icin pencere <=t). Donen DataFrame feats['forecast'] olur.
    """
    set_seed(seed)                          # fit'i tekrar-uretilebilir kil (golden)
    device = get_device()
    lr_tr = _logret(prices_train)
    lr_full = _logret(prices_full)

    x, y = _make_windows(lr_tr, window)
    model = ReturnForecaster(window, conv_ch, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.MSELoss()
    x_all = torch.as_tensor(x, device=device)
    y_all = torch.as_tensor(y, device=device)
    n = x_all.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            b = perm[s: s + batch]
            opt.zero_grad()
            loss_fn(model(x_all[b]), y_all[b]).backward()
            opt.step()

    # Causal tahmin: t'deki feature = t+1 ongorusu (pencere t'de biter).
    model.eval()
    T, N = lr_full.shape
    out = np.zeros((T, N), dtype=np.float32)
    if T > window - 1:
        idx = list(range(window - 1, T))
        with torch.no_grad():
            for j in range(N):
                col = lr_full[:, j]
                W = np.stack([col[t - window + 1: t + 1] for t in idx]).astype(np.float32)[:, :, None]
                pred = model(torch.as_tensor(W, device=device)).cpu().numpy()
                out[idx, j] = pred
    return pd.DataFrame(out, index=prices_full.index, columns=prices_full.columns)
