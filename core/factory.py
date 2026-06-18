"""Ajan + ortam fabrikasi — SOLID P3 (OCP + DRY).

UI (app.py) ve CLI (train.py) ajan/ortam kurulumunu ayri ayri kopyaliyordu;
ayrica `if algo ==` zinciri yeni algoritma eklemeyi mevcut kodu degistirmeye
zorluyordu (OCP ihlali). Artik:

  - AGENT_BUILDERS: "DQN"|"PPO"|"SAC" -> builder registry'si. Yeni algoritma =
    yeni kayit; mevcut kod degismez (OCP).
  - build_agent: hp dict'i config default'lariyla birlestirip ajani kurar.
  - build_env  : discrete<->continuous secimi + select_features + odul
    override'lari TEK noktada.

DAVRANIS KORUNUR: parametre birlestirme sirasi onceki app._make_agent ve
train.make_env ile birebir ayni; ajan ctor'lari ayni sirayla cagrilir (RNG
tuketimi degismez). Golden-master <=1e-6 bunu dogrular.

DIKKAT: UI ve CLI farkli max_steps kullanir (orn. UI SAC=1200, CLI SAC=600).
Bunlar bilincli olarak BIRLESTIRILMEZ; max_steps'i cagiran taraf gecirir.
"""
from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from agents import DQNAgent, PPOAgent, SACAgent, TD3Agent
from config import SEED, DQNConfig, EnvConfig, PPOConfig, RewardConfig, SACConfig, TD3Config
from core.features import select_features
from env.portfolio_env import DiscretePortfolioEnv, PortfolioEnv


def _build_dqn(state_dim: int, action_dim: int, hp: dict, seed: int) -> DQNAgent:
    return DQNAgent(
        state_dim, action_dim,
        hidden=tuple(hp.get("hidden", DQNConfig.hidden)),
        lr=hp.get("lr", DQNConfig.lr),
        eps_decay=hp.get("eps_decay", DQNConfig.eps_decay),
        batch_size=hp.get("batch_size", DQNConfig.batch_size),
        target_update=hp.get("target_update", DQNConfig.target_update),
        seed=seed,
    )


def _build_ppo(state_dim: int, action_dim: int, hp: dict, seed: int) -> PPOAgent:
    return PPOAgent(
        state_dim, action_dim,
        hidden=tuple(hp.get("hidden", PPOConfig.hidden)),
        lr_p=hp.get("lr_p", PPOConfig.lr_p), lr_v=hp.get("lr_v", PPOConfig.lr_v),
        clip=hp.get("clip", PPOConfig.clip), ent_coef=hp.get("ent_coef", PPOConfig.ent_coef),
        batch_size=hp.get("batch_size", PPOConfig.batch_size),
        n_epochs=hp.get("n_epochs", PPOConfig.n_epochs),
        seed=seed,
    )


def _build_sac(state_dim: int, action_dim: int, hp: dict, seed: int) -> SACAgent:
    return SACAgent(
        state_dim, action_dim,
        hidden=tuple(hp.get("hidden", SACConfig.hidden)),
        lr_pi=hp.get("lr_pi", SACConfig.lr_pi), lr_q=hp.get("lr_q", SACConfig.lr_q),
        alpha=hp.get("alpha", SACConfig.alpha), tau=hp.get("tau", SACConfig.tau),
        batch_size=hp.get("batch_size", SACConfig.batch_size), seed=seed,
    )


def _build_td3(state_dim: int, action_dim: int, hp: dict, seed: int) -> TD3Agent:
    return TD3Agent(
        state_dim, action_dim,
        hidden=tuple(hp.get("hidden", TD3Config.hidden)),
        lr_pi=hp.get("lr_pi", TD3Config.lr_pi), lr_q=hp.get("lr_q", TD3Config.lr_q),
        gamma=hp.get("gamma", TD3Config.gamma), tau=hp.get("tau", TD3Config.tau),
        policy_noise=hp.get("policy_noise", TD3Config.policy_noise),
        noise_clip=hp.get("noise_clip", TD3Config.noise_clip),
        policy_delay=hp.get("policy_delay", TD3Config.policy_delay),
        expl_noise=hp.get("expl_noise", TD3Config.expl_noise),
        batch_size=hp.get("batch_size", TD3Config.batch_size), seed=seed,
    )


# OCP: yeni algoritma eklemek = bu registry'ye kayit eklemek.
AGENT_BUILDERS: Dict[str, Callable] = {
    "DQN": _build_dqn,
    "PPO": _build_ppo,
    "SAC": _build_sac,
    "TD3": _build_td3,
}


def build_agent(algo: str, state_dim: int, action_dim: int,
                hp: dict | None = None, *, seed: int = SEED):
    """Registry uzerinden ajan kurar; hp eksik anahtarlarda config default'a duser."""
    builder = AGENT_BUILDERS.get(algo)
    if builder is None:
        raise ValueError(f"Bilinmeyen algoritma: {algo!r} "
                         f"(kayitli: {sorted(AGENT_BUILDERS)})")
    return builder(state_dim, action_dim, hp or {}, seed)


def build_env(algo: str, prices: pd.DataFrame, feats: dict, *,
              horizon: str = "medium", adaptive: bool = True,
              max_steps: int, random_start: bool = False, seed: int = SEED,
              reward_overrides: dict | None = None,
              price_noise_std: float | None = None,
              macro=None, regime=None) -> PortfolioEnv:
    """Tek ortam kurulum noktasi: discrete<->continuous secimi + feature secimi.

    reward_overrides (UI'nin reward_cfg'i): None/eksik anahtarlar env'in preset
    default'larina duser — onceki app._make_env mapping'i ile birebir ayni.
    """
    cfg = reward_overrides or {}
    cls = DiscretePortfolioEnv if algo == "DQN" else PortfolioEnv
    return cls(
        prices, select_features(feats, algo),
        horizon=horizon, adaptive=adaptive, max_steps=max_steps,
        random_start=random_start, seed=seed,
        eta_base=cfg.get("eta_base"),
        lambda_base=cfg.get("lambda_base"),
        tau_base=cfg.get("tau_base"),
        vol_target=float(cfg.get("vol_target", EnvConfig.vol_target)),
        turnover_target=float(cfg.get("turnover_target", EnvConfig.turnover_target)),
        ema_alpha=float(cfg.get("ema_alpha", EnvConfig.ema_alpha)),
        bankruptcy_nav=cfg.get("bankruptcy_nav"),
        bankruptcy_penalty=cfg.get("bankruptcy_penalty"),
        price_noise_std=(EnvConfig.price_noise_std if price_noise_std is None else float(price_noise_std)),
        # Mevcut 6 odul param'i parametrik akisa acilir — eksik/None anahtar config
        # default'una duser (golden-guvenli; eta_base/bankruptcy_penalty deseni ile ayni).
        w_dsr=float(cfg.get("w_dsr", RewardConfig.w_dsr)),
        dsr_eta=float(cfg.get("dsr_eta", RewardConfig.dsr_eta)),
        w_cvar=cfg.get("w_cvar"),   # None -> env'de config default*cvar_factor (golden-guvenli)
        cvar_alpha=float(cfg.get("cvar_alpha", RewardConfig.cvar_alpha)),
        regime_beta=float(cfg.get("regime_beta", RewardConfig.regime_beta)),
        cvar_amp=float(cfg.get("cvar_amp", RewardConfig.cvar_amp)),
        # v9: OPT-IN kazanc-carpani + iflas-timing (default 0/kapali -> golden bit-ayni)
        w_gain=float(cfg.get("w_gain", 0.0)),
        gain_floor=float(cfg.get("gain_floor", 1.0)),
        w_gain_speed=float(cfg.get("w_gain_speed", 0.0)),
        w_ruin_timing=float(cfg.get("w_ruin_timing", 0.0)),
        macro=macro, regime=regime,   # v6: makro rejim blogu + ham regime (V7)
    )
