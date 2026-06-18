"""Generate all figures for the paper and the presentation."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"
FIG  = BASE / "figures"; FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
    "savefig.dpi": 200,
})
PAL = {"DQN":"#d62728", "PPO":"#1f77b4", "SAC":"#2ca02c", "TD3":"#17becf",
       "BuyHold":"#7f7f7f", "EqualWeight":"#ff7f0e", "MeanVar":"#9467bd"}
# RL ajanlari (kalin/duz cizgi); baseline'lar ince/kesik. TD3 4. ajan olarak eklendi.
RL_AGENTS = ["DQN", "PPO", "SAC", "TD3"]


def run():
    # ---------- F1: Cumulative NAV ----------
    navs = pd.read_csv(RES/"navs_aligned.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in navs.columns:
        ax.plot(navs.index, navs[c], label=c, color=PAL.get(c, None),
                lw=1.8 if c in RL_AGENTS else 1.2,
                ls="-" if c in RL_AGENTS else "--")
    ax.set_ylabel("Portföy Değeri (NAV, başlangıç=1)")
    ax.set_xlabel("Tarih"); ax.set_title("BIST 30 — RL vs Klasik Stratejiler (Test dönemi)")
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout(); plt.savefig(FIG/"f1_cumulative_nav.png"); plt.close()
    print("f1 ok")

    # ---------- F2: Drawdown ----------
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in navs.columns:
        peak = navs[c].cummax()
        dd = (navs[c] - peak) / peak * 100
        ax.fill_between(navs.index, dd, 0, alpha=0.25, color=PAL.get(c, None))
        ax.plot(navs.index, dd, label=c, color=PAL.get(c, None), lw=1.2)
    ax.set_ylabel("Drawdown (%)"); ax.set_title("Rolling Drawdown")
    ax.legend(ncol=3, fontsize=9); plt.tight_layout()
    plt.savefig(FIG/"f2_drawdown.png"); plt.close()
    print("f2 ok")

    # ---------- F3: Rolling 60d Sharpe ----------
    rets = navs.pct_change().dropna()
    roll_sh = rets.rolling(60).apply(lambda x: np.sqrt(252) * x.mean() / (x.std() + 1e-9))
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in roll_sh.columns:
        ax.plot(roll_sh.index, roll_sh[c], label=c, color=PAL.get(c, None),
                lw=1.5 if c in RL_AGENTS else 1.0,
                ls="-" if c in RL_AGENTS else "--")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("60-Gün Rolling Sharpe"); ax.set_title("Koşullu Risk-Getiri Dengesi")
    ax.legend(ncol=2, fontsize=9); plt.tight_layout()
    plt.savefig(FIG/"f3_rolling_sharpe.png"); plt.close()
    print("f3 ok")

    # ---------- F4: Metrics bar ----------
    met = pd.read_csv(RES/"metrics.csv", index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(axes, ["CAGR", "Sharpe", "MaxDD"],
                              ["Yıllık Getiri (CAGR)", "Sharpe Oranı", "Maksimum Drawdown"]):
        colors = [PAL.get(k, "#333") for k in met.index]
        ax.bar(met.index, met[col], color=colors, edgecolor="k", lw=0.5)
        ax.set_title(title); ax.tick_params(axis='x', rotation=25)
    plt.suptitle("Test Dönemi Performans Karşılaştırması", y=1.02)
    plt.tight_layout(); plt.savefig(FIG/"f4_metrics_bar.png", bbox_inches="tight"); plt.close()
    print("f4 ok")

    # ---------- F5: Risk-Return scatter (collision-free labels) ----------
    OFF = {
        "DQN":         (10,  14),
        "PPO":         (10, -18),
        "SAC":         (14,   8),
        "MeanVar":     (14,  16),
        "BuyHold":     (-60, 30),
        "EqualWeight": (-85, -10),
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    xs_all, ys_all = [], []
    for k in met.index:
        x = met.loc[k, "Volatility"] * 100
        y = met.loc[k, "CAGR"] * 100
        xs_all.append(x); ys_all.append(y)
        ax.scatter(x, y, s=170, color=PAL.get(k, "#333"), edgecolor="k",
                   lw=1.1, zorder=5, label=k)

    for k in met.index:
        x = met.loc[k, "Volatility"] * 100
        y = met.loc[k, "CAGR"] * 100
        dx, dy = OFF.get(k, (8, 8))
        ax.annotate(
            k, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=10, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                      ec=PAL.get(k, "#333"), lw=1.0, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color=PAL.get(k, "#555"),
                            lw=0.9, alpha=0.75,
                            shrinkA=0, shrinkB=6),
            zorder=6,
        )

    ax.set_xlabel("Yıllık Volatilite (%)")
    ax.set_ylabel("Yıllık Getiri (CAGR, %)")
    ax.set_title("Risk-Getiri Düzlemi")

    xmax = max(xs_all) * 1.15
    ymax = max(ys_all) * 1.25
    xs_line = np.linspace(0.1, xmax, 100)
    for sh in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ax.plot(xs_line, sh * xs_line, ls=":", color="gray", lw=0.8, alpha=0.6)
        yend = sh * xmax
        if yend < ymax:
            ax.text(xmax, yend, f"  Sharpe={sh}", fontsize=8, color="gray", va="center")
        else:
            xend = (ymax - 2) / sh
            ax.text(xend, ymax - 2, f"Sharpe={sh}", fontsize=8, color="gray", ha="center", va="bottom")

    ax.set_xlim(left=0, right=xmax * 1.08)
    ax.set_ylim(bottom=min(0, min(ys_all)) - 5, top=ymax)
    plt.tight_layout(); plt.savefig(FIG/"f5_risk_return.png", bbox_inches="tight"); plt.close()
    print("f5 ok")

    # ---------- F6: PPO / SAC / DQN / TD3 weights heatmap ----------
    for algo in ["PPO", "SAC", "DQN", "TD3"]:
        W = pd.read_csv(RES/f"weights_{algo}.csv")
        # sample every N days, transpose for display
        step = max(1, len(W) // 200)
        Wp = W.iloc[::step].T
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(Wp.values, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=min(0.3, Wp.values.max() * 1.05))
        ax.set_yticks(range(Wp.shape[0])); ax.set_yticklabels(Wp.index, fontsize=7)
        ax.set_xticks([0, Wp.shape[1] - 1]); ax.set_xticklabels([f"t=0", f"t={len(W)}"])
        ax.set_title(f"{algo} — Günlük Portföy Ağırlıkları (Test dönemi)")
        plt.colorbar(im, ax=ax, label="Ağırlık")
        plt.tight_layout(); plt.savefig(FIG/f"f6_weights_{algo}.png"); plt.close()
    print("f6 ok")

    # ---------- F7: Training curves ----------
    dqn_c = pd.read_csv(RES/"dqn_curve.csv")
    ppo_c = pd.read_csv(RES/"ppo_curve.csv")
    sac_c = pd.read_csv(RES/"sac_curve.csv")
    td3_c = pd.read_csv(RES/"td3_curve.csv")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(dqn_c["episode"], dqn_c["reward"], "o-", color=PAL["DQN"])
    axes[0].set_title("DQN — Epizot Ödülü"); axes[0].set_xlabel("Epizot"); axes[0].set_ylabel("Kümülatif Ödül")
    ax2 = axes[0].twinx()
    if "train_nav" in dqn_c.columns:
        ax2.plot(dqn_c["episode"], dqn_c["train_nav"], "s--", color="k", alpha=0.6, label="Tren NAV")
        ax2.set_ylabel("Tren NAV")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ppo_c["update"], -ppo_c["p_loss"], "o-", color=PAL["PPO"], label="surrogate obj")
    axes[1].plot(ppo_c["update"], ppo_c["v_loss"], "s--", color="k", alpha=0.5, label="value loss")
    axes[1].set_title("PPO — Surrogate & Value Loss"); axes[1].set_xlabel("Güncelleme")
    axes[1].legend(fontsize=9)

    axes[2].plot(sac_c["episode"], sac_c["train_nav"], "o-", color=PAL["SAC"])
    axes[2].set_title("SAC — Epizot Sonu NAV"); axes[2].set_xlabel("Epizot"); axes[2].set_ylabel("NAV (tren)")

    axes[3].plot(td3_c["episode"], td3_c["train_nav"], "o-", color=PAL["TD3"])
    axes[3].set_title("TD3 — Epizot Sonu NAV"); axes[3].set_xlabel("Epizot"); axes[3].set_ylabel("NAV (tren)")
    plt.tight_layout(); plt.savefig(FIG/"f7_training_curves.png"); plt.close()
    print("f7 ok")

    # ---------- F8: MDP diagram ----------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    agent_box = patches.FancyBboxPatch((0.8, 1.8), 2.4, 1.3, boxstyle="round,pad=0.1",
                                        fc="#cde", ec="#246", lw=1.5)
    env_box   = patches.FancyBboxPatch((8.8, 1.8), 2.4, 1.3, boxstyle="round,pad=0.1",
                                        fc="#ffe5c8", ec="#a64", lw=1.5)
    ax.add_patch(agent_box); ax.add_patch(env_box)
    ax.text(2.0, 2.45, "AJAN\n(DQN/PPO/SAC/TD3)", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(10.0, 2.45, "ORTAM\n(BIST 30 Piyasa)", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.annotate("", xy=(8.8, 2.7), xytext=(3.2, 2.7),
                arrowprops=dict(arrowstyle="->", lw=2, color="#246"))
    ax.text(6.0, 2.95, "a_t = π(s_t)   (portföy ağırlıkları)", ha="center", fontsize=10, color="#246")
    ax.annotate("", xy=(3.2, 2.2), xytext=(8.8, 2.2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#a64"))
    ax.text(6.0, 1.95, "s_{t+1},  r_t = log(1+w_t·r_{t+1}) − λ·DD − η·|Δw|",
            ha="center", fontsize=10, color="#a64")
    ax.text(6.0, 4.5, "Portföy Yönetimi MDP Formülasyonu",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(6.0, 0.7, "S: 13 özellik × 28 hisse + mevcut ağırlıklar = 393 boyut        "
            "A: softmax(29-boyutlu simpleks)        γ = 0.99        T ≈ 760 gün/bölüm",
            ha="center", fontsize=9, color="#555")
    plt.tight_layout(); plt.savefig(FIG/"f8_mdp.png"); plt.close()
    print("f8 ok")

    # ---------- F9: Algo architecture sketch ----------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (name, desc) in zip(axes, [
        ("DQN", "s -> MLP(64,64) -> Q(s,a)\nayrik eylem: 6 portfoy sablonu\nTD hedefi + hedef ag"),
        ("PPO", "s -> policy -> Normal(mu, sigma) -> softmax(w)\nGAE avantaji\nclipped surrogate loss"),
        ("SAC", "s -> policy -> tanh(Normal)\ncift-Q elestirmen\nentropi-duzenlenmis amac"),
        ("TD3", "s -> Actor -> tanh (deterministik)\ncift-Q min + gecikmeli politika\nhedef-politika yumusatma")
    ]):
        ax.axis("off")
        ax.text(0.5, 0.88, name, ha="center", fontsize=20, fontweight="bold",
                color=PAL[name])
        ax.text(0.5, 0.5, desc, ha="center", fontsize=11, va="center")
        rect = patches.FancyBboxPatch((0.05, 0.08), 0.9, 0.84, boxstyle="round,pad=0.02",
                                       fc="none", ec=PAL[name], lw=2)
        ax.add_patch(rect)
    plt.suptitle("Dört Ajanın Mimari Özeti", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG/"f9_arch.png", bbox_inches="tight"); plt.close()
    print("f9 ok")

    # ---------- F10: Moving-average episode return (PDF §9.7 ogrenme egilimi) ----------
    from utils.metrics import moving_average
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for algo, c in [("DQN", dqn_c), ("PPO", ppo_c), ("SAC", sac_c), ("TD3", td3_c)]:
        if "reward" not in c.columns:
            continue
        r = c["reward"].to_numpy(dtype=float)
        ax.plot(range(len(r)), r, color=PAL[algo], alpha=0.30, lw=1.0)
        ma = moving_average(r, 5)
        if ma.size:
            x_ma = range(len(r) - ma.size, len(r))
            ax.plot(list(x_ma), ma, color=PAL[algo], lw=2.2, label=f"{algo} (MA-5)")
    ax.set_title("Hareketli Ortalama Episode Getirisi (öğrenme eğilimi)")
    ax.set_xlabel("İterasyon (episode / güncelleme)"); ax.set_ylabel("Çevre Ödülü Σr")
    ax.legend(fontsize=9); plt.tight_layout()
    plt.savefig(FIG/"f10_moving_avg_return.png"); plt.close()
    print("f10 ok")

    # ===== Titizlik (rigor) figürleri — scripts/rigor_analysis.py çıktıları (golden-güvenli) =====
    # ---------- F11: Deflated & Probabilistic Sharpe (López de Prado) + PBO ----------
    rp = RES/"rigor_metrics.csv"
    if rp.exists():
        rig = pd.read_csv(rp, index_col=0)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = np.arange(len(rig.index)); wd = 0.38
        ax.bar(x - wd/2, rig["PSR"], wd, label="PSR", color="#1f77b4", edgecolor="k", lw=0.5)
        ax.bar(x + wd/2, rig["DSR"], wd, label="DSR (deflated)", color="#d62728", edgecolor="k", lw=0.5)
        ax.axhline(0.95, ls="--", color="gray", lw=1)
        ax.text(len(x) - 0.5, 0.965, "0.95 eşiği", fontsize=8, color="gray", ha="right")
        ax.set_xticks(x); ax.set_xticklabels(rig.index, rotation=25); ax.set_ylim(0, 1.05)
        ax.set_title("Probabilistic & Deflated Sharpe (çoklu-deneme düzeltmeli) — López de Prado")
        ax.set_ylabel("Olasılık"); ax.legend(fontsize=9, loc="lower right")
        sp = RES/"rigor_summary.csv"
        if sp.exists():
            s = pd.read_csv(sp, index_col=0, header=None).iloc[:, 0]
            try:
                ax.text(0.01, 0.93, f"PBO = {float(s.get('PBO')):.2f}", transform=ax.transAxes,
                        fontsize=11, fontweight="bold", color="#555")
            except (TypeError, ValueError):
                pass
        plt.tight_layout(); plt.savefig(FIG/"f11_deflated_sharpe.png"); plt.close()
        print("f11 ok")

    # ---------- F12: Monte-Carlo stres — terminal getiri dağılımı + VaR/CVaR ----------
    mp = RES/"rigor_mc_terminal.csv"
    if mp.exists():
        mc = pd.read_csv(mp)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for col, color, lbl in [("block_bootstrap", "#2ca02c", "Blok bootstrap"),
                                ("student_t", "#9467bd", "Student-t (ağır kuyruk)")]:
            if col in mc.columns:
                d = mc[col].dropna().to_numpy() * 100
                ax.hist(d, bins=60, alpha=0.5, color=color, label=lbl, density=True)
                ax.axvline(np.quantile(d, 0.05), ls="--", color=color, lw=1.2)   # %5 VaR
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title("Monte-Carlo Stres — 1-yıl ileri terminal getiri dağılımı (en iyi RL ajan)")
        ax.set_xlabel("Terminal getiri (%)  ·  kesik çizgi = %5 VaR")
        ax.set_ylabel("Yoğunluk"); ax.legend(fontsize=9)
        plt.tight_layout(); plt.savefig(FIG/"f12_mc_stress.png"); plt.close()
        print("f12 ok")

    # ---------- F13: Nominal (TL) vs Reel (USD-bazlı) NAV — lira illüzyonu (§9.9) ----------
    nr = RES/"navs_real.csv"
    if nr.exists():
        real = pd.read_csv(nr, index_col=0, parse_dates=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        for c in [a for a in RL_AGENTS if a in navs.columns and a in real.columns]:
            ax.plot(navs.index, navs[c], color=PAL.get(c), lw=1.9, label=f"{c} nominal (TL)")
            ax.plot(real.index, real[c], color=PAL.get(c), lw=1.4, ls="--", label=f"{c} reel (USD)")
        ax.axhline(1.0, color="k", lw=0.5)
        ax.set_title("Nominal (TL) vs Reel (USD-bazlı) NAV — Lira İllüzyonu (§9.9)")
        ax.set_ylabel("NAV (başlangıç=1)"); ax.set_xlabel("Tarih"); ax.legend(fontsize=8, ncol=2)
        plt.tight_layout(); plt.savefig(FIG/"f13_real_nav.png"); plt.close()
        print("f13 ok")

    print(f"\nAll figures saved in {FIG}")


if __name__ == "__main__":
    run()
