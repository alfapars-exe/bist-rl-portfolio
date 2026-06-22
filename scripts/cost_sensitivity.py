"""İşlem maliyeti duyarlılık analizi — net-edge testi (GOLDEN-GUVENLI, gozlemsel).

Soru: Gercekci BIST islem maliyeti (komisyon + BSMV + spread/slippage) altinda
duusk-turnover stratejiler (ozellikle SAC, gunluk tek-yon turnover ~0.011) yuksek-
turnover stratejilere (Momentum ~0.163, DQN ~0.238) karsi NET avantaj kazaniyor mu?
Sira nasil degisiyor?

MALIYET MODELI (kaynak: aracilik komisyon tarifeleri + Tebli, 2024-2025; bkz. rapor):
  - Aracilik komisyonu: binde 1-3.2 = ~10-32 bps tek-yon (serbest tarife, kurumdan
    kuruma degisir). Tipik retail ~15-20 bps.
  - BSMV: komisyon uzerine %5 (komisyonu 1.05x'e cikarir).
  - Spread/slippage: ~2-10 bps tek-yon (likit BIST-30 hisseleri).
  -> Tek-yon efektif maliyet ~ komisyon*1.05 + slippage ≈ 10-25+ bps; round-trip iki kat.
  Senaryolar (tek-yon, ‖Δw‖₁'e uygulanir): c ∈ {0, 10, 20, 50} bps.

η ESLEMESI: odul terimi η·‖Δw‖₁ tek-yon turnover'a uygulanir -> η (ondalik) = c/10000.
  η=0.0010 ≈ 10 bps tek-yon ≈ 20 bps round-trip. Preset'ler: kisa 0.0015(~15bps/30RT),
  orta 0.0010(~10bps/20RT), uzun 0.0005(~5bps/10RT). c=10-20 bps senaryolari preset
  bandinin merkezinde -> η gercekci.

HESAP (golden-guvenli — egitim yolunu degistirmez, np.random YOK):
  - RL ajanlari: results/backtest_{DQN,PPO,SAC,TD3}.csv icindeki gross_return ve
    gerceklesen tek-yon turnover kullanilir.
  - Baseline'lar results/bist30_prices.csv test doneminde maliyetsiz yeniden uretilir.
  - Tum stratejiler ortak degerleme tarihleri uzerinde hizalanir. c=0 maliyetsiz
    brut senaryodur; adaptif maliyet iceren kanonik metrics.csv ile ayni olmasi beklenmez.
  - Her c icin: rets_net_t = rets_gross_t - (c/10000)*‖Δw_t‖₁ ; NAV yeniden bilesik;
    Sharpe/MaxDD/CAGR utils/metrics.py ile yeniden hesap.
  - results/cost_sensitivity.csv: satir=strateji, sutun=her c icin FinalNAV + Sharpe.

MODEL SINIRLARI (rapor akademik-sinirlar bolumune): lineer maliyet (piyasa-etkisi /
derinlik / kalici-impact yok), T+2 valor nakit-zamanlamasi modellenmez, lot/tick
yuvarlamasi ihmal, kismi-fill (partial fill) yok, komisyon basamakli/minimum-ucret yok.

Tamamen GOZLEMSEL: hicbir egitim/eval/odul sayisal yolunu degistirmez; golden korunur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RES = BASE / "results"

from utils.metrics import sharpe, max_drawdown, TRADING_DAYS  # noqa: E402
from utils.baselines import (  # noqa: E402
    equal_weight, buy_and_hold_index, mean_variance, inverse_volatility,
    risk_parity, min_variance, momentum, cash_riskfree,
)

SPLIT = "2022-01-01"                 # data.py train_test_split ile ayni
COST_BPS = [0, 10, 20, 50]           # tek-yon round-trip senaryolari (bps)
RL_ALGOS = ["DQN", "PPO", "SAC", "TD3"]


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------
def turnover_series(W: np.ndarray) -> np.ndarray:
    """Gunluk tek-yon turnover ‖Δw_t‖₁ — len(W)-1 uzunlugunda; bas satira 0 eklenir
    ki getiri dizisiyle (t=0 dahil) hizalansin. ‖Δw_0‖₁ = ilk konuma giris maliyeti
    HARIC tutulur (rollout/metrics turnover tanimiyla tutarli: yalniz diff)."""
    d = np.abs(np.diff(W, axis=0)).sum(axis=1)
    return np.concatenate([[0.0], d])     # t=0'da degisim yok varsayimi


def recompute(rets_gross: np.ndarray, turn: np.ndarray, c_bps: float) -> dict:
    """c bps tek-yon maliyet uygulanmis net metrikler. NAV her zaman 1.0'dan baslar."""
    c = c_bps / 10_000.0
    rets_net = rets_gross - c * turn
    nav = np.cumprod(1.0 + rets_net)
    metric_rets = rets_net[1:] if len(rets_net) > 1 and abs(rets_net[0]) < 1e-15 else rets_net
    elapsed_periods = max(len(nav) - 1, 1)
    return dict(
        FinalNAV=float(nav[-1]),
        Sharpe=float(sharpe(metric_rets)),
        CAGR=float(nav[-1] ** (TRADING_DAYS / elapsed_periods) - 1.0),
        MaxDD=float(max_drawdown(nav)),
        annual_drag=float(np.mean(turn) * c * TRADING_DAYS),  # ~ gunluk_turnover * c * 252
        mean_turnover=float(np.mean(turn)),
    )


# ---------------------------------------------------------------------------
# Veri yukleme
# ---------------------------------------------------------------------------
def load_rl():
    """RL ajanlari: navs_aligned'dan gross getiri, weights_*.csv'den turnover.
    navs_aligned RL navs'lerini min_len'e kirpar; weights tam test uzunlugunda.
    Hizalama: ikisini de SON L gune kirp (L = min(len(nav)-? , len(turn)))."""
    out = {}
    for algo in RL_ALGOS:
        trace_path = RES / f"backtest_{algo}.csv"
        if not trace_path.exists():
            raise FileNotFoundError(
                f"{trace_path.name} yok; gross getiri net NAV'dan guvenle turetilemez. "
                "Yeni train.run() ile sonuclari yeniden uretin.")
        trace = pd.read_csv(trace_path)
        out[algo] = dict(
            dates=pd.DatetimeIndex(pd.to_datetime(trace["date"])),
            rets=trace["gross_return"].to_numpy(float),
            turn=trace["turnover"].to_numpy(float),
        )
    return out


def load_baselines():
    """Baseline weights results/'ta yok -> utils/baselines.py'yi TEST donemi fiyatlariyla
    deterministik yeniden uret (train.py'nin px_te'siyle ayni cikti -> c=0 sanity).
    np.random cagrilmaz (tum baseline'lar determinist)."""
    px = pd.read_csv(RES / "bist30_prices.csv", index_col=0, parse_dates=True)
    px_te = px[px.index >= SPLIT]
    specs = {
        "BuyHold":     lambda: buy_and_hold_index(px_te, eta=0.0),
        "EqualWeight": lambda: equal_weight(px_te, eta=0.0),
        "MeanVar":     lambda: mean_variance(px_te, lookback=120, rebalance=20, eta=0.0),
        "RiskParity":  lambda: risk_parity(px_te, lookback=120, rebalance=20, eta=0.0),
        "InverseVol":  lambda: inverse_volatility(px_te, lookback=60, rebalance=20, eta=0.0),
        "MinVariance": lambda: min_variance(px_te, lookback=120, rebalance=20, eta=0.0),
        "Momentum":    lambda: momentum(px_te, lookback=60, rebalance=20, top_k=5, eta=0.0),
        "CashRiskFree": lambda: cash_riskfree(px_te),
    }
    out = {}
    for name, fn in specs.items():
        d = fn()
        rets = np.asarray(d["rets"], dtype=float)
        W = d.get("weights")
        if W is None:                       # BuyHold: weights yok, alip-tut -> turnover ~0
            turn = np.zeros(len(rets))
        else:
            turn = turnover_series(np.asarray(W, dtype=float))
        L = min(len(rets), len(turn))
        dates = pd.DatetimeIndex(d.get("dates", px_te.index))[-L:]
        out[name] = dict(dates=dates, rets=rets[-L:], turn=turn[-L:])
    return out


# ---------------------------------------------------------------------------
# Calistir
# ---------------------------------------------------------------------------
def run():
    strategies = {}
    strategies.update(load_rl())
    strategies.update(load_baselines())

    common_dates = None
    for d in strategies.values():
        dates = pd.DatetimeIndex(d["dates"])
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    if common_dates is None or common_dates.empty:
        raise ValueError("Stratejiler arasinda ortak degerleme tarihi bulunamadi.")
    common_dates = common_dates.sort_values()
    for name, d in strategies.items():
        frame = pd.DataFrame(
            {"rets": d["rets"], "turn": d["turn"]},
            index=pd.DatetimeIndex(d["dates"]),
        )
        frame = frame.loc[~frame.index.duplicated(keep="last")].reindex(common_dates)
        if frame.isna().any().any():
            raise ValueError(f"{name}: ortak tarih hizalamasinda eksik deger olustu.")
        d["dates"] = common_dates
        d["rets"] = frame["rets"].to_numpy(float)
        d["turn"] = frame["turn"].to_numpy(float)

    rows = []
    per_c = {c: {} for c in COST_BPS}
    for name, d in strategies.items():
        rec = {"Strateji": name, "Turnover_1way": float(np.mean(d["turn"]))}
        for c in COST_BPS:
            r = recompute(d["rets"], d["turn"], c)
            per_c[c][name] = r
            rec[f"NAV_c{c}"] = r["FinalNAV"]
            rec[f"Sharpe_c{c}"] = r["Sharpe"]
        rec["AnnualDrag_c20"] = per_c[20][name]["annual_drag"]
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("Strateji")
    # Sutun sirasi: Turnover, sonra her c icin NAV+Sharpe, sonra drag
    cols = ["Turnover_1way"]
    for c in COST_BPS:
        cols += [f"NAV_c{c}", f"Sharpe_c{c}"]
    cols += ["AnnualDrag_c20"]
    df = df[cols]
    df.to_csv(RES / "cost_sensitivity.csv")

    # ---- Rapor ----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("=" * 90)
    print("ISLEM-MALIYETI DUYARLILIK ANALIZI  (golden-guvenli, gozlemsel)")
    print("=" * 90)
    print(f"Maliyet senaryolari (tek-yon, bps): {COST_BPS}   |  eta = c/10000")
    print(f"Ortak test donemi: {common_dates[0].date()} - {common_dates[-1].date()} "
          f"({len(common_dates)} degerleme tarihi)")
    print()

    # (c) maliyet-duyarlilik tablosu
    print("--- (c) MALIYET-DUYARLILIK TABLOSU: strateji x c -> FinalNAV / Sharpe ---")
    show = df.copy()
    print(show.to_string())
    print()

    print("--- c=0 NOTU ---")
    print("c=0 maliyetsiz brut senaryodur. Kanonik metrics.csv adaptif/gerceklesen "
          "islem maliyetlerini icerdigi icin sayisal eslesme beklenmez.")
    print()

    # (d) SAC net-edge + siralama degisimi
    print("--- (d) SIRALAMA: net Sharpe'a gore (yuksek->dusuk), her c icin ---")
    order_by_c = {}
    for c in COST_BPS:
        ranking = sorted(per_c[c].items(), key=lambda kv: kv[1]["Sharpe"], reverse=True)
        order_by_c[c] = [n for n, _ in ranking]
    # yan yana sirali liste
    hdr = "".join(f"c={c:<3}bps".ljust(20) for c in COST_BPS)
    print(f"{'Sira':<6}{hdr}")
    n_show = len(strategies)
    for i in range(n_show):
        line = f"{i+1:<6}"
        for c in COST_BPS:
            name = order_by_c[c][i]
            sh = per_c[c][name]["Sharpe"]
            line += f"{name}({sh:.2f})".ljust(20)
        print(line)
    print()

    # SAC vs yuksek-turnover net Sharpe karsilastirmasi
    print("--- SAC (dusuk-turnover) vs yuksek-turnover stratejiler: net Sharpe @ c ---")
    focus = [s for s in ["SAC", "TD3", "Momentum", "DQN", "MinVariance", "PPO"]
             if s in per_c[0]]
    print(f"{'Strateji':<13}{'Turn_1way':>11}" + "".join(f"{'Sh_c'+str(c):>10}" for c in COST_BPS)
          + f"{'Drag_c20':>11}{'Drag_c50':>11}")
    for name in focus:
        t = per_c[0][name]["mean_turnover"]
        line = f"{name:<13}{t:>11.4f}"
        for c in COST_BPS:
            line += f"{per_c[c][name]['Sharpe']:>10.3f}"
        line += f"{per_c[20][name]['annual_drag']:>11.4f}{per_c[50][name]['annual_drag']:>11.4f}"
        print(line)
    print()

    # Sayisal bulgu ozeti
    sac_sh = {c: per_c[c]["SAC"]["Sharpe"] for c in COST_BPS}
    print("--- BULGU OZETI ---")
    print(f"SAC net Sharpe:   c=0 -> {sac_sh[0]:.3f},  c=20 -> {sac_sh[20]:.3f},  "
          f"c=50 -> {sac_sh[50]:.3f}   (drag@50bps = {per_c[50]['SAC']['annual_drag']*100:.2f}%/yil)")
    for hi in ["Momentum", "DQN"]:
        if hi in per_c[0]:
            hi_sh = {c: per_c[c][hi]["Sharpe"] for c in COST_BPS}
            print(f"{hi:<9} net Sharpe:  c=0 -> {hi_sh[0]:.3f},  c=20 -> {hi_sh[20]:.3f},  "
                  f"c=50 -> {hi_sh[50]:.3f}   (drag@50bps = {per_c[50][hi]['annual_drag']*100:.2f}%/yil)")
    # SAC'in yuksek-turnover'a karsi yer degistirmesi
    for hi in ["Momentum", "DQN"]:
        if hi not in per_c[0]:
            continue
        rank0 = order_by_c[0].index("SAC") - order_by_c[0].index(hi)
        rank50 = order_by_c[50].index("SAC") - order_by_c[50].index(hi)
        print(f"SAC vs {hi}: sira-farki (SAC_sira - {hi}_sira)  c=0 -> {rank0:+d},  c=50 -> {rank50:+d}  "
              f"(negatif = SAC ustte)")
    print("=" * 90)
    print(f"Yazildi: {RES / 'cost_sensitivity.csv'}")
    return df


if __name__ == "__main__":
    run()
