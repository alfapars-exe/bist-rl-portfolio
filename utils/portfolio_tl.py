"""TL cinsinden portföy telemetrisi — env'in NAV ve ağırlık çıktılarını
gerçek TL bakiyesine, lot sayısına, alım-satım logu'na ve holding-days'e çevirir.

Env'i değiştirmeden, tamamen UI katmanında türetim yapar.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

HOLDING_EPS = 0.005  # w[i] >= bu eşik ise "tutuluyor" say


def compute_tl_step(nav: float, w_now: np.ndarray, w_prev: np.ndarray,
                    prices_t: np.ndarray, initial_capital: float,
                    prev_portfolio_tl: float, holding_days_prev: np.ndarray,
                    tx_cost_rate: float = 0.0,
                    target_weights: np.ndarray | None = None,
                    holding_period_days: int = 1,
                    eps: float = HOLDING_EPS) -> Dict[str, np.ndarray | float]:
    """Tek adımın TL türevlerini hesapla.

    Args:
        nav: env.nav (scalar)
        w_now: adım sonu ağırlık vektörü (N+1,)
        w_prev: adım öncesi ağırlık (N+1,)
        prices_t: o adımdaki risk'li varlık fiyat vektörü (N,)
        initial_capital: kullanıcının girdiği TL bakiyesi
        prev_portfolio_tl: önceki adımın portföy TL'si (step P&L için)
        holding_days_prev: (N,) — önceki adım sonu holding gün sayıları
        tx_cost_rate: env'in bu adım için uyguladığı tx_cost oranı (reward_terms["tx_cost"])
        eps: holding sayımı için ağırlık eşiği
    """
    w_now = np.asarray(w_now, dtype=np.float64).reshape(-1)
    w_prev = np.asarray(w_prev, dtype=np.float64).reshape(-1)
    prices_t = np.asarray(prices_t, dtype=np.float64).reshape(-1)
    safe_px = np.maximum(prices_t, 1e-9)

    N_risky = prices_t.shape[0]
    portfolio_tl = float(initial_capital) * float(nav)
    cash_tl      = portfolio_tl * float(w_now[-1])
    asset_tl     = portfolio_tl * w_now[:N_risky]
    shares       = asset_tl / safe_px
    trade_w = (w_now if target_weights is None
               else np.asarray(target_weights, dtype=np.float64).reshape(-1))
    trade_tl     = portfolio_tl * (trade_w[:N_risky] - w_prev[:N_risky])
    trade_shares = trade_tl / safe_px
    commission_tl = portfolio_tl * float(tx_cost_rate)

    holding_days_prev = np.asarray(holding_days_prev, dtype=np.int64).reshape(-1)
    holding_days = np.where(w_now[:N_risky] >= eps,
                            holding_days_prev + max(1, int(holding_period_days)), 0)

    return dict(
        portfolio_tl=portfolio_tl,
        cash_tl=cash_tl,
        asset_tl=asset_tl,
        shares=shares,
        trade_tl=trade_tl,
        trade_shares=trade_shares,
        commission_tl=commission_tl,
        holding_days=holding_days,
        step_pnl_tl=portfolio_tl - float(prev_portfolio_tl),
        cum_pnl_tl=portfolio_tl - float(initial_capital),
        w_now=w_now.copy(),
    )


def compute_tl_series(nav_hist: List[float],
                      weight_hist: List[np.ndarray],
                      prices_matrix: np.ndarray,
                      initial_capital: float,
                      tx_cost_rates: List[float] | None = None,
                      t_start: int = 0) -> List[Dict]:
    """Bir episod/trace için adım-adım TL dict listesi döner.

    Args:
        nav_hist: [nav_0=1.0, nav_1, nav_2, ...] — env.nav_history (len = steps+1)
        weight_hist: [w_0, w_1, ...] — env.weight_history (len = steps+1)
        prices_matrix: shape (T, N_risky) — env.prices tamamı
        initial_capital: başlangıç TL bakiyesi
        tx_cost_rates: her adımın tx_cost oranı listesi (len = steps); None ise 0
        t_start: weight_hist[i]'ye karşılık gelen prices_matrix indeks başlangıcı
                 (env'de env.reset sonrası t = max(window,21), step_count=0).
                 weight_hist[0] reset sonrası w, weight_hist[k] env.t = t_start+k
                 sonrasındaki ağırlık.
    """
    steps = len(nav_hist) - 1
    N_risky = prices_matrix.shape[1]
    if tx_cost_rates is None:
        tx_cost_rates = [0.0] * steps
    rows: List[Dict] = []
    holding_days_prev = np.zeros(N_risky, dtype=np.int64)
    prev_portfolio_tl = float(initial_capital) * float(nav_hist[0])
    for k in range(1, steps + 1):
        t_idx = min(t_start + k, prices_matrix.shape[0] - 1)
        snap = compute_tl_step(
            nav=nav_hist[k],
            w_now=weight_hist[k],
            w_prev=weight_hist[k - 1],
            prices_t=prices_matrix[t_idx],
            initial_capital=initial_capital,
            prev_portfolio_tl=prev_portfolio_tl,
            holding_days_prev=holding_days_prev,
            tx_cost_rate=tx_cost_rates[k - 1],
        )
        rows.append(snap)
        holding_days_prev = snap["holding_days"]
        prev_portfolio_tl = snap["portfolio_tl"]
    return rows


def build_portfolio_table(tickers: List[str], snap: Dict,
                          include_cash: bool = True,
                          weight_threshold: float = HOLDING_EPS,
                          w_prev: np.ndarray | None = None) -> pd.DataFrame:
    """Adım anının portföy panosu: hisse, ağırlık, lot, TL değeri, holding days,
    bu adım Δ TL ve Durum (YENİ/TUTULUYOR/ÇIKIŞ/NAKİT).

    snap: compute_tl_step / compute_tl_series elemanı + 'w' (mevcut ağırlık vektörü).
    w_prev: önceki adımın ağırlık vektörü (Durum sütunu için gerekli).
    """
    w_raw = snap.get("w_now")
    if w_raw is None:
        # Fallback: asset_tl ve cash_tl'den ağırlık tahmin et
        tot = snap["portfolio_tl"] or 1.0
        w_risky = snap["asset_tl"] / tot
        w_cash = snap["cash_tl"] / tot
        w = np.concatenate([w_risky, [w_cash]])
    else:
        w = np.asarray(w_raw).reshape(-1)
    N_risky = len(snap["asset_tl"])
    w_prev_arr = (np.asarray(w_prev).reshape(-1) if w_prev is not None
                  else np.zeros_like(w))
    rows = []
    # Tutulan veya bu adımda çıkılan hisseler — hepsini göster
    for i in range(N_risky):
        weight = float(w[i])
        prev_w = float(w_prev_arr[i]) if i < len(w_prev_arr) else 0.0
        is_in  = weight >= weight_threshold
        was_in = prev_w >= weight_threshold
        if not (is_in or was_in):
            continue
        if is_in and not was_in:
            status = "YENİ"
        elif is_in and was_in:
            status = "TUTULUYOR"
        else:  # was_in and not is_in
            status = "ÇIKIŞ"
        rows.append({
            "Hisse": tickers[i],
            "Durum": status,
            "Ağırlık": round(weight, 4),
            "Lot": round(float(snap["shares"][i]), 3),
            "TL Değeri": round(float(snap["asset_tl"][i]), 2),
            "Tutuluyor (gün)": int(snap["holding_days"][i]),
            "Bu adım Δ TL": round(float(snap["trade_tl"][i]), 2),
        })
    # Durum önceliği: ÇIKIŞ → YENİ → TUTULUYOR, sonra ağırlığa göre
    status_order = {"ÇIKIŞ": 0, "YENİ": 1, "TUTULUYOR": 2}
    rows.sort(key=lambda r: (status_order.get(r["Durum"], 9), -r["Ağırlık"]))
    if include_cash:
        rows.append({
            "Hisse": "NAKİT",
            "Durum": "—",
            "Ağırlık": round(float(w[-1]), 4),
            "Lot": None,
            "TL Değeri": round(float(snap["cash_tl"]), 2),
            "Tutuluyor (gün)": None,
            "Bu adım Δ TL": None,
        })
    return pd.DataFrame(rows)


def build_cumulative_trade_log(tickers: List[str], tl_snaps: List[Dict],
                               dates: List[str],
                               threshold_tl: float = 1.0,
                               up_to_step: int | None = None) -> pd.DataFrame:
    """Episod/test başından `up_to_step` dahil (None ise hepsi) tüm al-sat kayıtları."""
    end_k = len(tl_snaps) if up_to_step is None else min(up_to_step + 1, len(tl_snaps))
    rows = []
    for k in range(end_k):
        snap = tl_snaps[k]
        trade_tl = np.asarray(snap["trade_tl"])
        trade_sh = np.asarray(snap["trade_shares"])
        for i in range(len(trade_tl)):
            tt = float(trade_tl[i])
            if abs(tt) < threshold_tl:
                continue
            rows.append({
                "Adım": k,
                "Tarih": dates[k] if k < len(dates) else "",
                "Hisse": tickers[i],
                "İşlem": "ALIŞ" if tt > 0 else "SATIŞ",
                "Lot": round(float(trade_sh[i]), 3),
                "TL": round(tt, 2),
            })
    return pd.DataFrame(rows)


def build_trade_log(tickers: List[str], snap: Dict,
                    threshold_tl: float = 1.0) -> pd.DataFrame:
    """Bu adımda yapılan alış/satışlar — |trade_tl| > eşik olanlar."""
    N_risky = len(snap["trade_tl"])
    rows = []
    for i in range(N_risky):
        tt = float(snap["trade_tl"][i])
        if abs(tt) < threshold_tl:
            continue
        lots = float(snap["trade_shares"][i])
        rows.append({
            "Hisse": tickers[i],
            "İşlem": "ALIŞ" if tt > 0 else "SATIŞ",
            "Lot": round(lots, 3),
            "TL": round(tt, 2),
        })
    rows.sort(key=lambda r: -abs(r["TL"]))
    return pd.DataFrame(rows)


def step_rows_for_training(curve_snaps: List[Dict], dates: np.ndarray,
                           action_names: List[str] | None = None,
                           action_indices: List[int] | None = None,
                           reward_terms_list: List[Dict] | None = None,
                           initial_capital: float = 0.0) -> pd.DataFrame:
    """Eğitim episodunun tüm adımları için tablo — 252 satır.

    Args:
        curve_snaps: compute_tl_series çıktısı
        dates: env.dates[t_start+1 : t_start+1+steps]
        action_names: adım başına aksiyon adı listesi (DQN) veya None
        action_indices: adım başına aksiyon idx (DQN) veya None
        reward_terms_list: adım başına reward_terms (delta_w_l1 için)
        initial_capital: yüzde hesabı için
    """
    rows = []
    for k, snap in enumerate(curve_snaps):
        action_name = ""
        if action_names and k < len(action_names):
            action_name = action_names[k]
        elif action_indices and k < len(action_indices):
            action_name = f"a={action_indices[k]}"
        delta_w = 0.0
        if reward_terms_list and k < len(reward_terms_list):
            delta_w = float(reward_terms_list[k].get("delta_w_l1", 0.0))
        held = int(np.sum(np.asarray(snap["holding_days"]) > 0))
        cum_pct = 100.0 * (snap["cum_pnl_tl"] / initial_capital) if initial_capital else 0.0
        dstr = str(pd.Timestamp(dates[k]).date()) if k < len(dates) else ""
        rows.append({
            "Gün #": k + 1,
            "Tarih": dstr,
            "Aksiyon": action_name,
            "Nakit TL": round(snap["cash_tl"], 2),
            "Portföy TL": round(snap["portfolio_tl"], 2),
            "Adım P&L": round(snap["step_pnl_tl"], 2),
            "Kümülatif P&L": round(snap["cum_pnl_tl"], 2),
            "Kümülatif %": round(cum_pct, 3),
            "Komisyon TL": round(snap["commission_tl"], 2),
            "Δ turnover": round(delta_w, 4),
            "Tutulan hisse": held,
        })
    return pd.DataFrame(rows)
