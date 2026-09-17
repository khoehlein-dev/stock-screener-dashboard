import datetime as dt
import math

import numpy as np
import pytest
from scipy.stats import norm

from stock_screener.analytics.qlpricing import (
    MarketEnv,
    VolSurface,
    bs_delta,
    bs_greeks,
    bs_price,
    implied_vol,
)

AS_OF = dt.date(2026, 9, 17)


def _bs_closed_form(typ, S, K, t, r, q, s):
    d1 = (math.log(S / K) + (r - q + 0.5 * s * s) * t) / (s * math.sqrt(t))
    d2 = d1 - s * math.sqrt(t)
    if typ == "call":
        return S * math.exp(-q * t) * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    return K * math.exp(-r * t) * norm.cdf(-d2) - S * math.exp(-q * t) * norm.cdf(-d1)


@pytest.mark.parametrize("typ", ["call", "put"])
def test_bs_price_matches_closed_form(typ):
    p = bs_price(typ, 100, 95, 0.25, 0.04, 0.01, 0.3)
    assert p == pytest.approx(_bs_closed_form(typ, 100, 95, 0.25, 0.04, 0.01, 0.3), rel=1e-10)


def test_put_call_parity():
    S, K, t, r, q, s = 120.0, 110.0, 0.5, 0.03, 0.02, 0.4
    c = bs_price("call", S, K, t, r, q, s)
    p = bs_price("put", S, K, t, r, q, s)
    assert c - p == pytest.approx(S * math.exp(-q * t) - K * math.exp(-r * t), rel=1e-10)


def test_greeks_consistency():
    g = bs_greeks("call", 100, 100, 0.25, 0.04, 0.0, 0.3)
    fwd = 100 * math.exp(0.04 * 0.25)
    assert g["delta"] == pytest.approx(bs_delta("call", fwd, 100, 0.25, 0.3, 0.0), rel=1e-10)
    # finite-difference delta
    h = 1e-3
    fd = (
        bs_price("call", 100 + h, 100, 0.25, 0.04, 0.0, 0.3)
        - bs_price("call", 100 - h, 100, 0.25, 0.04, 0.0, 0.3)
    ) / (2 * h)
    assert g["delta"] == pytest.approx(fd, abs=1e-6)
    assert g["gamma"] > 0 and g["vega"] > 0 and g["theta"] < 0


@pytest.mark.parametrize("american", [False, True])
def test_implied_vol_round_trip(american):
    env = MarketEnv(AS_OF, 0.04)
    expiry = AS_OF + dt.timedelta(days=45)
    t = env.year_fraction(expiry)
    price = bs_price("call", 100, 105, t, 0.04, 0.01, 0.35)  # OTM call: no early exercise value
    iv = implied_vol(env, "call", 100, 105, expiry, price, q=0.01, american=american)
    assert iv == pytest.approx(0.35, abs=2e-3 if american else 1e-6)


def test_implied_vol_rejects_below_intrinsic():
    env = MarketEnv(AS_OF, 0.04)
    expiry = AS_OF + dt.timedelta(days=45)
    assert implied_vol(env, "call", 100, 80, expiry, 5.0) is None


def test_surface_interpolates_total_variance_and_extrapolates_flat():
    env = MarketEnv(AS_OF, 0.04)
    e1, e2 = AS_OF + dt.timedelta(days=30), AS_OF + dt.timedelta(days=90)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vols = np.column_stack([np.full(5, 0.30), np.full(5, 0.20)])
    surf = VolSurface(env, [e1, e2], strikes, vols)
    t1, t2 = env.year_fraction(e1), env.year_fraction(e2)
    tm = 0.5 * (t1 + t2)
    expected = math.sqrt((0.09 * t1 + 0.04 * t2) / 2 / tm)
    assert surf.vol(tm, 100) == pytest.approx(expected, rel=1e-6)
    assert surf.vol(t1, 40) == pytest.approx(0.30, rel=1e-6)  # constant strike extrapolation
    assert surf.vol(t1 / 2, 100) == pytest.approx(0.30, rel=1e-6)  # flat before first expiry
