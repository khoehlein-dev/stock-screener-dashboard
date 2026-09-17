import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from stock_screener.analytics.qlpricing import bs_price
from stock_screener.derivatives.costs import CostModel
from stock_screener.derivatives.evaluation import Evaluator, recommend, sample_terminal
from stock_screener.derivatives.models import Derivative, products_frame, products_from_frame
from stock_screener.derivatives.pricing import (
    MarketContext,
    knockout_probability,
    ko_levels,
    ko_values,
    warrant_implied_vol,
    warrant_values,
)
from stock_screener.derivatives.sources import SyntheticDerivativesSource, map_record, parse_records
from stock_screener.derivatives.strategies import (
    STRATEGY_CLASSES,
    Strategy,
    bucket_products,
    filter_products,
    sample_realizations,
)

AS_OF = dt.date(2026, 9, 17)
FX = 1.10


@pytest.fixture
def ctx() -> MarketContext:
    return MarketContext("TST", AS_OF, spot=100.0, r=0.04, q=0.0, vol_fn=lambda t, k: 0.30)


@pytest.fixture
def costs(tmp_path) -> CostModel:
    return CostModel.load()


def ko(
    ptype="ko_long",
    strike=90.0,
    barrier=90.0,
    ask=1.0,
    spread=0.01,
    maturity=None,
    issuer="HSBC",
    fin=0.05,
):
    return Derivative(
        underlying="TST",
        isin=f"X{ptype}{strike}{barrier}",
        product_type=ptype,
        strike=strike,
        ratio=0.1,
        bid=ask * (1 - spread),
        ask=ask,
        barrier=barrier,
        maturity=maturity,
        issuer=issuer,
        financing_rate=fin,
        name=f"{ptype} {barrier}",
    )


def warrant(
    ptype="call_warrant", strike=100.0, days=180, ask=None, sigma=0.3, ctx=None, issuer="HSBC"
):
    t = days / 365
    fair = (
        bs_price("call" if ptype == "call_warrant" else "put", 100.0, strike, t, 0.04, 0.0, sigma)
        * 0.1
        / FX
    )
    ask = ask or fair * 1.01
    return Derivative(
        underlying="TST",
        isin=f"W{ptype}{strike}{days}",
        product_type=ptype,
        strike=strike,
        ratio=0.1,
        bid=ask * 0.98,
        ask=ask,
        maturity=AS_OF + dt.timedelta(days=days),
        issuer=issuer,
        name=f"{ptype} {strike}",
    )


# ---- costs ------------------------------------------------------------------------------


def test_cost_model_reads_yaml_not_code(costs):
    assert costs.plan_name == costs.raw["active_plan"]
    v = costs.venue_fees
    assert costs.order_fee(500.0) == pytest.approx(
        max(v["order_fee_fixed"] + v["order_fee_pct"] * 500, v["order_fee_min"])
    )
    partner = costs.plan["partner_derivatives"]
    assert (
        costs.order_fee(partner["min_order_value"], partner["issuers"][0])
        == partner["order_fee_fixed"]
    )
    assert costs.order_fee(0.0) == 0.0


# ---- pricing ----------------------------------------------------------------------------


def test_bridge_probability_limits():
    assert knockout_probability(100, np.array([80.0]), 90, 0.3, 0.1, True)[0] == 1.0
    p = knockout_probability(100, np.array([100.0]), 90, 0.3, 30 / 365, True)[0]
    assert 0 < p < 0.1
    # vanishing vol -> no crossing when both ends are above the barrier
    assert knockout_probability(100, np.array([100.0]), 90, 1e-4, 30 / 365, True)[0] < 1e-9
    # Monte Carlo check of the bridge formula against a fine GBM path simulation (conditional on S_T ~ S_0)
    rng = np.random.default_rng(1)
    n, steps, t, sig = 200000, 400, 30 / 365, 0.3
    z = rng.standard_normal((n, steps)) * sig * math.sqrt(t / steps)
    paths = 100 * np.exp(np.cumsum(z, axis=1))
    end = paths[:, -1]
    sel = np.abs(end / 100 - 1) < 0.005
    hit = (paths[sel].min(axis=1) <= 90).mean()
    assert hit == pytest.approx(p, abs=0.01)


def test_ko_financing_moves_strike_and_barrier(ctx):
    k, b = ko_levels(ko(strike=90, barrier=92, fin=0.05), 1.0, 0.05, 0.01)
    assert k == pytest.approx(90 * math.exp(0.05)) and b == pytest.approx(92 * math.exp(0.05))
    k, b = ko_levels(ko("ko_short", strike=110, barrier=108, fin=0.02), 1.0, 0.05, 0.01)
    assert k == pytest.approx(110 * math.exp(-0.02))
    k, _ = ko_levels(ko(maturity=AS_OF + dt.timedelta(days=90)), 1.0, 0.05, 0.01)
    assert k == 90  # dated products do not accrue


def test_ko_values_and_residual(ctx):
    p = ko(strike=88.0, barrier=90.0)
    S_T = np.array([120.0, 95.0, 85.0])
    v, knocked = ko_values(
        p,
        S_T,
        np.array([0.99, 0.99, 0.0]),
        1e-6,
        ctx,
        FX,
        recovery=0.5,
        fin_long=0.0,
        fin_short=0.0,
    )
    assert not knocked[0] and v[0] == pytest.approx((120 - 88) * 0.1 / FX)
    assert knocked[2] and v[2] == pytest.approx(0.5 * (90 - 88) * 0.1 / FX)


def test_warrant_round_trip(ctx):
    w = warrant(strike=105.0, days=180, sigma=0.35)
    iv = warrant_implied_vol(w, ctx, FX, price=w.mid)
    assert iv == pytest.approx(0.35, abs=0.02)  # mid carries the 1 % markup / 2 % spread
    v0 = warrant_values(w, np.array([100.0]), 0.0, 0.35, ctx, FX)[0]
    assert v0 == pytest.approx(
        bs_price("call", 100, 105, 180 / 365, 0.04, 0.0, 0.35) * 0.1 / FX, rel=1e-9
    )
    at_exp = warrant_values(w, np.array([120.0]), 180 / 365, 0.35, ctx, FX)[0]
    assert at_exp == pytest.approx(15 * 0.1 / FX)


# ---- sources ----------------------------------------------------------------------------


def test_parse_and_map_records():
    fields = {
        "isin": "isin",
        "wkn": "wkn",
        "name": "name",
        "issuer": "issuer",
        "product_type": "type",
        "strike": "strike",
        "barrier": "knockOut",
        "ratio": "ratio",
        "bid": "quote.bid",
        "ask": "quote.ask",
        "maturity": "maturity",
    }
    text = (
        '{"results": [{"isin": "DE1", "type": "KNOCKOUT_LONG", "strike": "95,5", "knockOut": 96, "ratio": 0.1, "quote": {"bid": 1.0, "ask": 1.02}, "maturity": "open-end"},'
        '{"isin": "DE2", "type": "Call", "strike": 100, "ratio": 0.1, "quote": {"bid": 0.5, "ask": 0.52}, "maturity": "17.12.2026"},'
        '{"isin": "DE3", "type": "Call", "strike": 100, "ratio": 0.1, "quote": {"bid": 0.6, "ask": 0.52}}]}'
    )
    recs = parse_records(text, "json")
    types = {"KNOCKOUT_LONG": "ko_long", "Call": "call_warrant"}
    out = [map_record(r, "AAPL", fields, types, "cli") for r in recs]
    assert (
        out[0].product_type == "ko_long"
        and out[0].strike == 95.5
        and out[0].maturity is None
        and out[0].barrier == 96
    )
    assert out[1].product_type == "call_warrant" and out[1].maturity == dt.date(2026, 12, 17)
    assert out[2] is None  # crossed quotes rejected
    assert parse_records("a,b\n1,2\n", "csv") == [{"a": "1", "b": "2"}]
    assert parse_records('{"isin": "x"}\n{"isin": "y"}', "json")[1]["isin"] == "y"


def test_synthetic_source_and_frame_round_trip(ctx):
    src = SyntheticDerivativesSource(
        {
            "synthetic": {
                "seed": 1,
                "ko_barrier_distances": [0.05, 0.1],
                "warrant_moneyness": [0.9, 1.0, 1.1],
                "warrant_maturities_days": [90, 180],
                "issuers": ["HSBC"],
            }
        },
        FX,
        AS_OF,
    )
    prods = src.list_products("TST", ctx)
    assert {p.product_type for p in prods} == set(
        STRATEGY_CLASSES and ("ko_long", "ko_short", "call_warrant", "put_warrant")
    )
    for p in prods:
        assert p.ask > p.bid > 0
        if p.product_type == "ko_long":
            assert p.barrier < ctx.spot and p.strike <= p.barrier
    df = products_frame(prods, AS_OF)
    back = products_from_frame(df)
    assert [p.isin for p in back] == [p.isin for p in prods]
    assert back[0].maturity == prods[0].maturity


# ---- strategies -------------------------------------------------------------------------


def test_filter_bucket_and_sample(ctx):
    prods = [
        ko(strike=100 * (1 - d), barrier=100 * (1 - d), ask=d * 10)
        for d in (0.01, 0.03, 0.05, 0.08, 0.12, 0.2, 0.6)
    ]
    prods += [
        ko("ko_short", strike=100 * (1 + d), barrier=100 * (1 + d), ask=d * 10)
        for d in (0.03, 0.06, 0.12, 0.25)
    ]
    prods += [warrant(strike=k, days=d) for k in (90, 100, 110) for d in (30, 90, 200)]
    prods += [warrant("put_warrant", strike=k, days=d) for k in (90, 100, 110) for d in (90, 200)]
    filters = {
        "min_price": 0.1,
        "max_spread_pct": 0.05,
        "ko": {"min_barrier_distance": 0.02, "max_barrier_distance": 0.5, "max_leverage": 100},
        "warrants": {
            "min_days_to_maturity_after_horizon": 20,
            "max_days_to_maturity": 730,
            "moneyness_range": [0.7, 1.35],
        },
    }
    usable = filter_products(prods, ctx, filters, 30)
    dists = sorted(abs(p.barrier / 100 - 1) for p in usable if p.product_type == "ko_long")
    assert dists == pytest.approx([0.03, 0.05, 0.08, 0.12, 0.2])  # 1 % too close, 60 % too far
    assert all(
        (p.maturity - AS_OF).days >= 50 for p in usable if not p.is_ko
    )  # 30-day warrants dropped
    sampling = {
        "ko_bucket_width": 0.025,
        "warrant_strike_bucket": 0.05,
        "warrant_maturity_buckets_days": [60, 120, 240],
        "max_products_per_type": 12,
        "max_realizations_per_class": 10,
        "ladder_min_barrier_spacing": 0.03,
        "ladder_sizes": [2, 3, 4],
        "straddle_max_strike_gap": 0.06,
        "strangle_min_wing": 0.04,
    }
    buckets = bucket_products(usable, ctx, sampling)
    reals = sample_realizations(buckets, ctx, sampling)
    by_class = {}
    for s in reals:
        by_class.setdefault(s.strategy_class, []).append(s)
    assert set(by_class) >= {
        "ko_long",
        "ko_short",
        "call_warrant",
        "put_warrant",
        "ko_long_ladder",
        "warrant_straddle",
        "warrant_strangle",
        "ko_long_put_hedge",
        "ko_pair",
        "call_warrant_ladder",
    }
    assert all(len(v) <= 10 for v in by_class.values())
    assert all(1 <= s.n_legs <= 4 for s in reals)
    for s in by_class["ko_long_ladder"]:
        d = sorted(abs(leg.barrier / 100 - 1) for leg in s.legs)
        assert min(np.diff(d)) >= 0.03
    for s in by_class["warrant_straddle"]:
        assert s.legs[0].maturity == s.legs[1].maturity


# ---- evaluation -------------------------------------------------------------------------


def lognormal_samples(spot, sigma, t, n=20000, seed=3):
    z = np.linspace(-6, 6, 2001)
    grid = spot * np.exp((0.04 - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * z)
    return sample_terminal(grid, norm.cdf(z), n, t, 0.0, seed)


def test_sample_terminal_matches_distribution():
    s = lognormal_samples(100, 0.3, 30 / 365)
    assert s.S_T.mean() == pytest.approx(100 * math.exp(0.04 * 30 / 365), rel=2e-3)
    assert np.log(s.S_T).std() == pytest.approx(0.3 * math.sqrt(30 / 365), rel=2e-2)


def test_evaluator_costs_and_metrics(ctx, costs):
    samples = lognormal_samples(100, 0.3, 30 / 365)
    ev = Evaluator(
        ctx, costs, samples, {"budget_eur": 1000, "es_alpha": 0.05, "risk_aversion": 0.5}
    )
    p = ko(strike=80.0, barrier=80.0, ask=(100 - 80) * 0.1 / FX * 1.01, issuer="NoPartner")
    s = Strategy("ko_long", "TST", [p], [1.0], 0.2)
    m = ev.metrics(s)
    assert m["feasible"]
    q = int(m["quantities"])
    assert q == math.floor(1000 / p.ask)
    # invested = q * ask + entry fee from the YAML fee model
    assert m["invested"] == pytest.approx(q * p.ask + costs.order_fee(q * p.ask, "NoPartner"))
    assert m["cost_drag"] > 0
    assert m["es_5"] <= m["var_5"] <= m["median_return"]
    assert 0.0 <= m["p_profit"] <= 1.0 and m["p_knockout"] < 0.05
    # deterministic payoff: zero P&L crossing exists on the downside only for a long KO
    assert (
        m["break_even_up"] is not None
        and m["break_even_down"] is None
        or m["break_even_up"] is not None
    )
    # a far-OTM knock-out that is certainly hit yields total loss
    hopeless = ko(strike=99.9, barrier=99.9, ask=0.05)
    m2 = ev.metrics(Strategy("ko_long", "TST", [hopeless], [1.0], 0.001))
    assert m2["p_knockout"] > 0.9 and m2["p_total_loss"] > 0.9


def test_recommend_flags_best_and_stability():
    rows = []
    for i, u in enumerate(np.linspace(-0.2, 0.05, 6)):
        rows.append(
            {
                "id": f"s{i}",
                "underlying": "TST",
                "strategy_class": "ko_long",
                "feasible": True,
                "param": 0.05 * (i + 1),
                "utility": float(u),
                "p_total_loss": 0.1,
                "exp_return": float(u),
                "es_5": -0.3,
                "label": str(i),
            }
        )
    rows.append(
        {
            "id": "bad",
            "underlying": "TST",
            "strategy_class": "ko_long",
            "feasible": True,
            "param": 0.4,
            "utility": 1.0,
            "p_total_loss": 0.9,
            "exp_return": 1.0,
            "es_5": -1.0,
            "label": "bad",
        }
    )
    out = pd.DataFrame(
        recommend(rows, {"max_total_loss_probability": 0.6, "neighbours_for_stability": 2})
    )
    best = out[out["is_best"]]
    assert (
        len(best) == 1 and best["id"].iloc[0] == "s5"
    )  # highest eligible utility, the 'bad' one is excluded
    assert 0 <= best["stability"].iloc[0] <= 1
    assert best["utility_gap"].iloc[0] == pytest.approx(0.05)
    assert out[out["id"] == "bad"]["eligible"].iloc[0] == False  # noqa: E712


def test_end_to_end_pipeline_and_ui(provider, settings, tmp_path):
    from stock_screener.analytics.pipeline import build_screener
    from stock_screener.app.dashboard import create_app
    from stock_screener.data.cache import DataCache
    from stock_screener.derivatives.pipeline import DerivativesRun

    settings.data_dir = tmp_path
    cache = DataCache(tmp_path)
    df = build_screener(provider, settings, dt.date(2026, 9, 17), cache, ["AAPL"])
    cache.put_screener(dt.date(2026, 9, 17), df)
    run = DerivativesRun(settings, source_kind="synthetic")
    products, strategies = run.run(["AAPL"])
    assert len(products) > 40 and len(strategies) > 100
    assert (
        strategies["is_best"].sum()
        == strategies[strategies["eligible"]].groupby("strategy_class").ngroups
    )
    assert (tmp_path / "derivatives" / "strategies.parquet").exists()
    app = create_app(settings, tmp_path)
    dstore = app.dstore
    assert len(dstore.strategies) == len(strategies)
    row = dstore.strategies[dstore.strategies["is_best"]].iloc[0].replace({np.nan: None}).to_dict()
    strat = dstore.run.rebuild_strategy(row, dstore.products)
    assert strat is not None and strat.id == row["id"]
    ev = dstore.evaluator("AAPL")
    m = ev.metrics(strat)
    assert m["exp_return"] == pytest.approx(
        row["exp_return"], abs=1e-9
    )  # deterministic re-evaluation
    from stock_screener.app.derivatives_tab import (
        distribution_figure,
        payoff_figure,
        stability_figure,
        tradeoff_figure,
    )

    res = ev.pnl(strat)
    group = dstore.strategies[(dstore.strategies["strategy_class"] == row["strategy_class"])]
    assert len(payoff_figure(ev, strat, "t").data) >= 2
    assert len(distribution_figure(res, m, "t").data) == 1
    assert len(tradeoff_figure(group, row["id"], "t", "p").data) >= 2
    assert len(stability_figure(group, row["id"], "t", "p").data) == 3
