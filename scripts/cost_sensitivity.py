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

HESAP (golden-guvenli — core/rollout & golden & metrics.csv & train DOKUNULMAZ, np.random YOK):
  - RL ajanlari: results/weights_{DQN,PPO,SAC,TD3}.csv -> gunluk tek-yon turnover
    ‖Δw_t‖₁ = |diff(W)|.sum(axis=1); gross getiriler results/navs_aligned.csv'den (pct_change).
  - Baseline'lar: utils/baselines.py'yi results/bist30_prices.csv TEST donemi (>=2022-01-01)
    alt kumesi ile deterministik yeniden uret -> weights trajektorisi -> turnover.
    c=0 satiri results/metrics.csv ile sanity-eslesir (train.py ayni px_te'yi kullanir).
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

from utils.metrics import sharpe, max_drawdown, cagr, TRADING_DAYS  # noqa: E402
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
    return dict(
        FinalNAV=float(nav[-1]),
        Sharpe=float(sharpe(rets_net)),
        CAGR=float(cagr(nav)),
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
    navs = pd.read_csv(RES / "navs_aligned.csv", index_col=0)
    out = {}
    for algo in RL_ALGOS:
        nav = navs[algo].to_numpy(dtype=float)
        rets_gross = np.concatenate([[0.0], np.diff(nav) / nav[:-1]])  # pct_change, t=0 -> 0
        W = pd.read_csv(RES / f"weights_{algo}.csv").to_numpy(dtype=float)
        turn = turnover_series(W)
        # navs_aligned kirpilmis olabilir -> son L gune hizala
        L = min(len(rets_gross), len(turn))
        out[algo] = dict(rets=rets_gross[-L:], turn=turn[-L:])
    return out


def load_baselines():
    """Baseline weights results/'ta yok -> utils/baselines.py'yi TEST donemi fiyatlariyla
    deterministik yeniden uret (train.py'nin px_te'siyle ayni cikti -> c=0 sanity).
    np.random cagrilmaz (tum baseline'lar determinist)."""
    px = pd.read_csv(RES / "bist30_prices.csv", index_col=0, parse_dates=True)
    px_te = px[px.index >= SPLIT]
    specs = {
        "BuyHold":     lambda: buy_and_hold_index(px_te),
        "EqualWeight": lambda: equal_weight(px_te),
        "MeanVar":     lambda: mean_variance(px_te, lookback=120, rebalance=20),
        "RiskParity":  lambda: risk_parity(px_te, lookback=120, rebalance=20),
        "InverseVol":  lambda: inverse_volatility(px_te, lookback=60, rebalance=20),
        "MinVariance": lambda: min_variance(px_te, lookback=120, rebalance=20),
        "Momentum":    lambda: momentum(px_te, lookback=60, rebalance=20, top_k=5),
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
        out[name] = dict(rets=rets[-L:], turn=turn[-L:])
    return out


# ---------------------------------------------------------------------------
# Calistir
# ---------------------------------------------------------------------------
def run():
    strategies = {}
    strategies.update(load_rl())
    strategies.update(load_baselines())

    # c=0 sanity: metrics.csv ile FinalNAV/Sharpe eslesmesi (toleransli rapor)
    met = pd.read_csv(RES / "metrics.csv", index_col=0)

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
    print(f"Maliyet senaryolari (tek-yon, bps): {COST_BPS}   |  η_ondalik = c/10000")
    print(f"Test donemi: index >= {SPLIT}  ({len(next(iter(strategies.values()))['rets'])} gun)")
    print()

    # (c) maliyet-duyarlilik tablosu
    print("--- (c) MALIYET-DUYARLILIK TABLOSU: strateji x c -> FinalNAV / Sharpe ---")
    show = df.copy()
    print(show.to_string())
    print()

    # c=0 sanity
    print("--- c=0 SANITY (cost_sensitivity vs results/metrics.csv) ---")
    print(f"{'Strateji':<13}{'NAV_c0':>12}{'metrics.NAV':>14}{'dNAV':>11}"
          f"{'Sh_c0':>10}{'metrics.Sh':>12}{'dSh':>10}")
    max_dnav = max_dsh = 0.0
    for name in strategies:
        if name not in met.index:
            continue
        nav0 = per_c[0][name]["FinalNAV"]
        sh0 = per_c[0][name]["Sharpe"]
        mnav = float(met.loc[name, "FinalNAV"])
        msh = float(met.loc[name, "Sharpe"])
        dnav, dsh = nav0 - mnav, sh0 - msh
        max_dnav = max(max_dnav, abs(dnav))
        max_dsh = max(max_dsh, abs(dsh))
        print(f"{name:<13}{nav0:>12.4f}{mnav:>14.4f}{dnav:>+11.4f}"
              f"{sh0:>10.4f}{msh:>12.4f}{dsh:>+10.4f}")
    print(f"  -> max |dNAV|={max_dnav:.4f}, max |dSharpe|={max_dsh:.4f}  "
          f"(RL: navs_aligned min_len kirpmasi -> kucuk sapma beklenir; "
          f"baseline: px_te birebir -> ~0)")
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
