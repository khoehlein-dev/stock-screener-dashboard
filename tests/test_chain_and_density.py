import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from stock_screener.analytics.chain import (
    SmileSurface,
    analyse_chain,
    clean_chain,
    fit_smile,
    implied_forward,
    interp_forward,
)
from stock_screener.analytics.qlpricing import MarketEnv, VolSurface, bs_price
from stock_screener.analytics.risk_neutral import risk_metrics, risk_neutral_density
from stock_screener.data.provider import CHAIN_COLUMNS

AS_OF = dt.date(2026, 9, 17)
R, Q, S = 0.04, 0.01, 100.0


def flat_chain(sigma=0.3, expiries=(30, 60, 90), with_iv=True):
    rows = []
    for d in expiries:
        exp = AS_OF + dt.timedelta(days=d)
        t = d / 365
        for K in np.arange(60, 145, 5.0):
            for typ in ("call", "put"):
                rows.append(
                    {
                        "contract": f"O:{typ}{d}{K}",
                        "type": typ,
                        "strike": K,
                        "expiry": pd.Timestamp(exp),
                        "price": bs_price(typ, S, K, t, R, Q, sigma),
                        "vwap": None,
                        "volume": 10,
                        "open_interest": 100,
                        "iv": sigma if with_iv else None,
                        "delta": None,
                        "gamma": None,
                        "theta": None,
                        "vega": None,
                        "last_updated": pd.Timestamp(AS_OF) + pd.Timedelta(hours=16),
                        "underlying_price": S,
                        "price_source": "close",
                    }
                )
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


def test_clean_chain_filters_and_solves_iv():
    env = MarketEnv(AS_OF, R)
    chain = flat_chain(with_iv=False)
    chain.loc[0, "open_interest"] = 0
    chain.loc[0, "volume"] = 0  # illiquid -> dropped
    chain.loc[1, "price"] = (
        0.01  # below intrinsic (deep ITM put? row 1 is a put at K=60: OTM) -> keep
    )
    cleaned = clean_chain(chain, env, S, Q)
    assert chain.loc[0, "contract"] not in set(cleaned["contract"])
    assert (cleaned["iv_src"] == "quantlib").all()
    assert cleaned["iv"].between(0.29, 0.31).mean() > 0.9


def test_implied_forward_from_parity():
    env = MarketEnv(AS_OF, R)
    chain = clean_chain(flat_chain(), env, S, Q)
    exp_df = chain[chain["expiry"] == AS_OF + dt.timedelta(days=30)]
    t = float(exp_df["t"].iloc[0])
    f, src = implied_forward(exp_df, S, t, R, Q)
    assert src == "parity"
    assert f == pytest.approx(S * math.exp((R - Q) * t), rel=1e-6)


def test_smile_fit_recovers_quadratic():
    env = MarketEnv(AS_OF, R)
    exp = AS_OF + dt.timedelta(days=30)
    t = env.year_fraction(exp)
    F = S * math.exp((R - Q) * t)
    rows = []
    for K in np.arange(70, 131, 2.5):
        k = math.log(K / F)
        iv = 0.25 - 0.2 * k + 0.6 * k * k
        typ = "call" if K >= F else "put"
        rows.append({"type": typ, "strike": K, "iv": iv, "open_interest": 100, "volume": 5, "t": t})
    fit = fit_smile(pd.DataFrame(rows), exp, t, F, Q)
    assert fit is not None
    a, b, c = fit.coef
    assert (a, b, c) == pytest.approx((0.25, -0.2, 0.6), abs=1e-6)
    assert fit.rmse < 1e-8
    # C1 flattening: slope decays beyond the observed range
    far = float(fit.iv(fit.k_max + 2.0))
    edge = float(fit.iv(fit.k_max))
    assert far - edge < abs(b + 2 * c * fit.k_max) * fit.lam * 1.01


def test_density_matches_lognormal_for_flat_vol():
    env = MarketEnv(AS_OF, R)
    sigma, d = 0.3, 30
    exp = [AS_OF + dt.timedelta(days=x) for x in (30, 60, 90)]
    strikes = S * np.exp(np.linspace(-2.5, 2.5, 121))
    surf = VolSurface(env, exp, strikes, np.full((121, 3), sigma))
    t = d / 365
    F = S * math.exp((R - Q) * t)
    res = risk_neutral_density(surf, S, F, t, R)
    st = res.stats
    assert st["mean_price"] == pytest.approx(F, rel=1e-4)
    ln_std = math.sqrt(math.exp(sigma**2 * t) - 1) * F / S
    assert st["std_ret"] == pytest.approx(ln_std, rel=2e-3)
    m = risk_metrics(res)
    sd = sigma * math.sqrt(t)
    exact = norm.cdf((math.log(0.9 * S / F) + 0.5 * sd**2) / sd)
    assert m["p_loss_10"] == pytest.approx(exact, abs=5e-4)
    assert res.clipped_mass < 1e-6
    assert m["es_5"] < m["var_5"] < 0


def test_analyse_chain_end_to_end_flat():
    env = MarketEnv(AS_OF, R)
    an = analyse_chain(flat_chain(0.3), env, S, Q)
    assert an.surface is not None
    assert an.metrics["iv_30"] == pytest.approx(0.3, abs=2e-3)
    assert an.metrics["iv_90"] == pytest.approx(0.3, abs=2e-3)
    assert abs(an.metrics["skew_25d"]) < 1e-3
    assert an.metrics["pc_oi_ratio"] == pytest.approx(1.0)
    assert an.metrics["n_expiries"] == 3


def test_analyse_chain_synthetic_provider(provider):
    env = MarketEnv(AS_OF, R)
    chain = provider.get_option_chain("MSFT")
    p = provider.params("MSFT")
    spot = p["s0"]
    an = analyse_chain(chain, env, spot, p["div_yield"])
    assert an.metrics["iv_30"] == pytest.approx(p["sigma"] + p["vrp"], abs=0.01)
    # negative smile slope -> puts richer than calls
    assert an.metrics["skew_25d"] > 0
    assert an.metrics["forward_source"] == "parity"
    assert an.metrics["implied_carry"] == pytest.approx(-p["div_yield"], abs=0.01)


def test_smile_surface_agrees_with_quantlib_surface(provider):
    env = MarketEnv(AS_OF, R)
    p = provider.params("AAPL")
    an = analyse_chain(provider.get_option_chain("AAPL"), env, p["s0"], p["div_yield"])
    assert isinstance(an.surface, SmileSurface) and an.ql_surface is not None
    for days in (20, 45, 75, 150):
        t = days / 365
        f = interp_forward(an.fits, p["s0"], t, R, p["div_yield"])
        for K in (0.9 * f, f, 1.1 * f):
            # bilinear QuantLib grid vs smooth smile: agree to a few bp of vol
            assert an.ql_surface.vol(t, K) == pytest.approx(an.surface.vol(t, K), abs=5e-4)
    assert an.metrics["ql_surface_diff"] < 5e-4


def test_smile_surface_time_interpolation_is_linear_in_variance():
    from stock_screener.analytics.chain import SmileFit

    f1 = SmileFit(AS_OF + dt.timedelta(days=30), 30 / 365, 100.0, (0.30, 0.0, 0.0), -1, 1, 10, 0.0)
    f2 = SmileFit(AS_OF + dt.timedelta(days=90), 90 / 365, 100.0, (0.20, 0.0, 0.0), -1, 1, 10, 0.0)
    surf = SmileSurface([f2, f1])
    tm = 60 / 365
    expected = math.sqrt((0.09 * f1.t + 0.04 * f2.t) / 2 / tm)
    assert surf.vol(tm, 100.0) == pytest.approx(expected, rel=1e-9)
    assert surf.vol(10 / 365, 100.0) == pytest.approx(0.30)
    assert surf.vol(2.0, 100.0) == pytest.approx(0.20)
