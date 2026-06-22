"""Granülerlik (Gün/Ay/Yil) ozelligini kilitleyen testler.

`data.resample_to_granularity(df, granularity)` ve kisa-veri adaptasyonu:
  - "daily"   -> NO-OP (ayni nesne, golden-guvenli).
  - "monthly" -> resample("ME").mean().ffill().bfill() -> satir azalir, NaN yok.
  - "yearly"  -> resample("YE").mean()...             -> cok daha az satir, NaN yok.

Kisa-veri env adaptasyonu:
  - window = min(max(window, minvol_window), n_days//3) -> coarse veride sigacak sekilde.
  - lo = min(max(window,21), n_days-2) -> gecerli baslangic noktasi.
  - Crash/NaN OLMAMALI.

Test senaryolari:
  (a) resample_to_granularity("daily") is df  -> NO-OP.
      monthly/yearly: satir azalir, isna==0, index DatetimeIndex.
  (b) GOLDEN-GUVENLIK (daily): PortfolioEnv(horizon="medium") -> window==60, t==60 (V11-exact).
  (c) Kisa-veri (yearly/monthly): env reset+done'a kadar step -> crash yok, nav NaN degil.
"""
import numpy as np
import pandas as pd
import pytest

from data import resample_to_granularity
from env.portfolio_env import PortfolioEnv, DiscretePortfolioEnv
from utils.features import add_features


# ---------------------------------------------------------------------------
# Yardimci: gercekci buyuklukta sentetik piyasa (gunluk)
# ---------------------------------------------------------------------------

def _daily_market(seed=0, n=400, k=5):
    """n gunluk, k hisseli kapanish fiyat DataFrame'i ve feature dict'i."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(k)]
    )
    feats = add_features(prices)
    return prices, feats


def _daily_market_long(seed=0, k=5):
    """~2000 gunluk (2015-2022) piyasa: yearly resample icin yeterli yil sayisi saglar."""
    return _daily_market(seed=seed, n=2000, k=k)


# ---------------------------------------------------------------------------
# (a) resample_to_granularity: davranis testleri
# ---------------------------------------------------------------------------

def test_daily_granularity_is_noop():
    """resample_to_granularity(df, "daily") ayni nesneyi (is) dondurmeli."""
    prices, _ = _daily_market()
    result = resample_to_granularity(prices, "daily")
    assert result is prices, (
        "resample_to_granularity('daily') NO-OP: ayni nesneyi dondurmeli (is)"
    )


def test_monthly_granularity_reduces_rows():
    """'monthly': satir sayisi gunluk'ten az olmali (~aylik frekansta)."""
    prices, _ = _daily_market(n=400)
    monthly = resample_to_granularity(prices, "monthly")
    assert len(monthly) < len(prices), (
        f"monthly ({len(monthly)}) < daily ({len(prices)}) olmali"
    )
    # ~400 is gunu ~ 19 ay -> 15-22 arasi beklenir
    assert 10 <= len(monthly) <= 25, (
        f"~400 gunluk veriden beklenen aylik satir: 10-25; {len(monthly)} geldi"
    )


def test_monthly_granularity_no_nan():
    """'monthly' resample sonrasi NaN olmamali."""
    prices, _ = _daily_market(n=400)
    monthly = resample_to_granularity(prices, "monthly")
    assert monthly.isna().sum().sum() == 0, "monthly: NaN deger olmamali"


def test_monthly_granularity_datetime_index():
    """'monthly' resample sonrasi index DatetimeIndex olmali."""
    prices, _ = _daily_market(n=400)
    monthly = resample_to_granularity(prices, "monthly")
    assert isinstance(monthly.index, pd.DatetimeIndex), (
        "monthly: index DatetimeIndex olmali"
    )


def test_yearly_granularity_reduces_rows():
    """'yearly': satir sayisi gunluk'ten cok daha az olmali (~yillik frekansta)."""
    # 2000 gunluk (~8 yil) veri kullan ki yeterli yillik satir olussin
    prices, _ = _daily_market_long()
    yearly = resample_to_granularity(prices, "yearly")
    assert len(yearly) < len(prices), (
        f"yearly ({len(yearly)}) < daily ({len(prices)}) olmali"
    )
    # ~2000 is gunu ~ 8 yil -> 7-9 yillik satir beklenir
    assert 5 <= len(yearly) <= 12, (
        f"~2000 gunluk veriden beklenen yillik satir: 5-12; {len(yearly)} geldi"
    )


def test_yearly_granularity_no_nan():
    """'yearly' resample sonrasi NaN olmamali."""
    prices, _ = _daily_market_long()
    yearly = resample_to_granularity(prices, "yearly")
    assert yearly.isna().sum().sum() == 0, "yearly: NaN deger olmamali"


def test_yearly_granularity_datetime_index():
    """'yearly' resample sonrasi index DatetimeIndex olmali."""
    prices, _ = _daily_market_long()
    yearly = resample_to_granularity(prices, "yearly")
    assert isinstance(yearly.index, pd.DatetimeIndex), (
        "yearly: index DatetimeIndex olmali"
    )


def test_invalid_granularity_raises():
    """Gecersiz granularity degeri ValueError yukseltmeli."""
    prices, _ = _daily_market(n=100)
    with pytest.raises(ValueError, match="granularity"):
        resample_to_granularity(prices, "weekly")


def test_feature_dataframes_monthly_no_nan():
    """Feature dict'indeki her DataFrame monthly resample sonrasi NaN olmamali."""
    prices, feats = _daily_market(n=400)
    for feat_name, feat_df in feats.items():
        monthly = resample_to_granularity(feat_df, "monthly")
        nan_count = monthly.isna().sum().sum()
        assert nan_count == 0, (
            f"Feature '{feat_name}' monthly resample sonrasi {nan_count} NaN iceriyor"
        )


def test_feature_dataframes_yearly_no_nan():
    """Feature dict'indeki her DataFrame yearly resample sonrasi NaN olmamali."""
    prices, feats = _daily_market_long()
    for feat_name, feat_df in feats.items():
        yearly = resample_to_granularity(feat_df, "yearly")
        nan_count = yearly.isna().sum().sum()
        assert nan_count == 0, (
            f"Feature '{feat_name}' yearly resample sonrasi {nan_count} NaN iceriyor"
        )


# ---------------------------------------------------------------------------
# (b) GOLDEN-GUVENLIK: gunluk veri -> window==60, t==60 (V11-exact)
# ---------------------------------------------------------------------------

def test_daily_window_equals_60_medium_horizon():
    """Gunluk veri (n>=2554), horizon='medium' -> window=min(max(20,60), n//3)=60 (V11-exact)."""
    # Yeterince uzun gunluk veri (V11 BIST gibi ~2554 gun)
    rng = np.random.default_rng(0)
    n = 600   # 600 gunluk: n//3=200 -> min(60,200)=60 -> V11-ayni
    idx = pd.bdate_range("2019-01-01", periods=n)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 5)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(5)]
    )
    feats = add_features(prices)
    env = PortfolioEnv(prices, feats, horizon="medium", seed=42)
    assert env.window == 60, (
        f"horizon='medium', n={n}: window={env.window} olmali 60 (V11-exact; "
        f"min(max(20,60), {n}//3=={n//3}) = min(60,{n//3}) = 60)"
    )


def test_daily_reset_t_equals_window_eval():
    """random_start=False eval modunda reset sonrasi t==lo=max(window,21)==60 (V11-exact)."""
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.bdate_range("2019-01-01", periods=n)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, 5)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(5)]
    )
    feats = add_features(prices)
    env = PortfolioEnv(prices, feats, horizon="medium", random_start=False, seed=42)
    env.reset()
    # lo_raw = max(window=60, 21) = 60; lo = min(60, n-2=598) = 60
    expected_t = min(max(env.window, 21), max(1, n - 2))
    assert env.t == expected_t, (
        f"reset() sonrasi t={env.t}, beklenen={expected_t} (V11-exact)"
    )
    assert env.t == 60, f"t=={env.t} olmali 60 (V11 golden degeri)"


# ---------------------------------------------------------------------------
# (c) Kisa-veri crash/NaN yok: yearly ve monthly
# ---------------------------------------------------------------------------

def _run_until_done(env, max_iter=500):
    """Env'i done/trunc'a kadar rastgele aksiyonlarla calistir."""
    rng = np.random.default_rng(77)
    env.reset()
    for i in range(max_iter):
        a = rng.normal(0, 1, env.action_dim).astype(np.float32)
        obs, reward, done, trunc, info = env.step(a)
        assert not np.isnan(env.nav), f"Adim {i}: NAV NaN oldu!"
        assert not np.any(np.isnan(obs)), f"Adim {i}: obs NaN iceriyor!"
        if done or trunc:
            return info
    return None


def test_yearly_short_data_no_crash():
    """Gunluk yukle -> add_features -> yearly resample -> PortfolioEnv -> crash/NaN yok.

    Not: yearly resample icin yeterli yil sayisi gerekir (en az 4-5 yil = ~1000+ gun).
    2000 gunluk (~8 yil) veri kullanilir -> 8 yillik satir -> env gecerli calisir.
    """
    prices, feats = _daily_market_long(k=5)

    p_year = resample_to_granularity(prices, "yearly")
    f_year = {k: resample_to_granularity(v, "yearly") for k, v in feats.items()}

    assert p_year.isna().sum().sum() == 0, "yearly prices: NaN olmamali"
    assert len(p_year) >= 4, f"Yeterli yillik satir gerekmeli (>=4), {len(p_year)} geldi"

    env = PortfolioEnv(
        p_year, f_year,
        horizon="short",
        random_start=True,
        max_steps=90,
        seed=42,
    )
    _run_until_done(env)
    assert not np.isnan(env.nav), "yearly env: son NAV NaN olmamali"


def test_monthly_short_data_no_crash():
    """Gunluk yukle -> add_features -> monthly resample -> PortfolioEnv -> crash/NaN yok."""
    prices, feats = _daily_market(n=400, k=5)

    p_mon = resample_to_granularity(prices, "monthly")
    f_mon = {k: resample_to_granularity(v, "monthly") for k, v in feats.items()}

    assert p_mon.isna().sum().sum() == 0, "monthly prices: NaN olmamali"

    env = PortfolioEnv(
        p_mon, f_mon,
        horizon="short",
        random_start=True,
        max_steps=90,
        seed=42,
    )
    _run_until_done(env)
    assert not np.isnan(env.nav), "monthly env: son NAV NaN olmamali"


def test_yearly_env_window_capped():
    """Yearly veride window n_days//3 ile caplenip env kuruluyor olmali."""
    prices, feats = _daily_market_long(k=5)
    p_year = resample_to_granularity(prices, "yearly")
    f_year = {k: resample_to_granularity(v, "yearly") for k, v in feats.items()}

    n_days = len(p_year)
    env = PortfolioEnv(p_year, f_year, horizon="medium", seed=42)
    # window = min(max(20, 60), max(1, n_days//3)) = min(60, n_days//3)
    expected_window = min(60, max(1, n_days // 3))
    assert env.window == expected_window, (
        f"yearly n_days={n_days}: window={env.window}, beklenen={expected_window}"
    )
    # window <= n_days garantisi (gecerli slice)
    assert env.window <= n_days, f"window ({env.window}) <= n_days ({n_days}) olmali"


def test_monthly_env_window_capped():
    """Kisa monthly veride window n_days//3 ile caplenip env kuruluyor olmali."""
    prices, feats = _daily_market(n=400, k=5)
    p_mon = resample_to_granularity(prices, "monthly")
    f_mon = {k: resample_to_granularity(v, "monthly") for k, v in feats.items()}

    n_days = len(p_mon)
    env = PortfolioEnv(p_mon, f_mon, horizon="medium", seed=42)
    expected_window = min(60, max(1, n_days // 3))
    assert env.window == expected_window, (
        f"monthly n_days={n_days}: window={env.window}, beklenen={expected_window}"
    )
    assert env.window <= n_days, f"window ({env.window}) <= n_days ({n_days}) olmali"


def test_yearly_full_rollout_nav_finite():
    """Yearly env'de tam episode boyunca NAV sonlu ve pozitif olmali."""
    prices, feats = _daily_market_long(k=5)
    p_year = resample_to_granularity(prices, "yearly")
    f_year = {k: resample_to_granularity(v, "yearly") for k, v in feats.items()}

    env = PortfolioEnv(p_year, f_year, horizon="short", random_start=True,
                       max_steps=90, seed=42)
    info = _run_until_done(env)
    assert np.isfinite(env.nav), f"Yearly env: NAV={env.nav} sonlu olmali"
    assert env.nav > 0, f"Yearly env: NAV={env.nav} pozitif olmali"


def test_monthly_full_rollout_nav_finite():
    """Monthly env'de tam episode boyunca NAV sonlu ve pozitif olmali."""
    prices, feats = _daily_market(n=400, k=5)
    p_mon = resample_to_granularity(prices, "monthly")
    f_mon = {k: resample_to_granularity(v, "monthly") for k, v in feats.items()}

    env = PortfolioEnv(p_mon, f_mon, horizon="short", random_start=True,
                       max_steps=90, seed=42)
    info = _run_until_done(env)
    assert np.isfinite(env.nav), f"Monthly env: NAV={env.nav} sonlu olmali"
    assert env.nav > 0, f"Monthly env: NAV={env.nav} pozitif olmali"


def test_granularity_config_options():
    """config.GRANULARITY_OPTIONS ve GRANULARITY_MIN_POINTS var ve dogru tiplerde olmali."""
    from config import GRANULARITY_OPTIONS, GRANULARITY_MIN_POINTS
    assert "daily" in GRANULARITY_OPTIONS
    assert "monthly" in GRANULARITY_OPTIONS
    assert "yearly" in GRANULARITY_OPTIONS
    assert isinstance(GRANULARITY_MIN_POINTS, dict)
    assert "daily" in GRANULARITY_MIN_POINTS
    assert "monthly" in GRANULARITY_MIN_POINTS
    assert "yearly" in GRANULARITY_MIN_POINTS


# ---------------------------------------------------------------------------
# (d) REGRESYON (adversarial bulgu): DiscretePortfolioEnv (DQN) + coarse + medium/long
#     window cap'lenip mom_window/minvol_window cap'lenmediginde _discrete_to_logits
#     feat_tensor[t-mvw:t] NEGATIF/BOS slice -> NaN agirlik/NAV uretiyordu. mom/minvol
#     da cap'lenip slice basi max(0,.) ile clamp'lendi. Tum coarse/horizon/aksiyon NaN-suz.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("granularity", ["monthly", "yearly"])
@pytest.mark.parametrize("horizon", ["short", "medium", "long"])
def test_discrete_env_coarse_all_actions_nan_free(granularity, horizon):
    """DQN (DiscreteEnv) coarse veride TUM 6 discrete aksiyonu dolasarak NaN uretmemeli."""
    prices, feats = _daily_market_long(seed=7)
    pg = resample_to_granularity(prices, granularity)
    fg = {k: resample_to_granularity(v, granularity) for k, v in feats.items()}
    env = DiscretePortfolioEnv(pg, fg, horizon=horizon, random_start=True,
                               max_steps=90, seed=7)
    s, _ = env.reset()
    # mom/minvol pencereleri veriye sigdirilmis olmali (cap'siz NaN'in koku buydu)
    assert env.mom_window <= max(1, env.n_days // 3)
    assert env.minvol_window <= max(1, env.n_days // 3)
    done = trunc = False
    n = 0
    while not (done or trunc) and n < 300:
        a = n % env.n_discrete                  # 6 discrete aksiyonu sirayla dolas
        s, r, done, trunc, info = env.step(a)
        n += 1
        assert not np.isnan(env.nav), f"{granularity}/{horizon} adim {n}: NAV NaN"
        assert not np.isnan(s).any(), f"{granularity}/{horizon} adim {n}: obs NaN"
        assert not np.isnan(env.w).any(), f"{granularity}/{horizon} adim {n}: agirlik NaN"
    assert np.isfinite(env.nav) and env.nav > 0


def test_discrete_env_daily_windows_unchanged():
    """GOLDEN-GUVENLIK: DiscreteEnv daily yolda mom/minvol/window V11-exact (degismedi)."""
    prices, feats = _daily_market_long(seed=0)
    env = DiscretePortfolioEnv(prices, feats, horizon="medium",
                               random_start=False, seed=42)
    env.reset()
    assert env.window == 60 and env.mom_window == 20 and env.minvol_window == 60, (
        f"daily medium V11-exact bekleniyor (60,20,60); "
        f"({env.window},{env.mom_window},{env.minvol_window}) geldi"
    )
    assert env.t == 60, f"daily medium reset t==60 (V11); {env.t} geldi"
