"""Strategy classes and the heuristic sampling of their realizations.

A *strategy class* is a template ("buy one long knock-out", "KO long ladder",
"warrant straddle", ...). A *realization* is a concrete choice of products for
one underlying. All strategies are buy-only (retail investors cannot write
warrants or certificates) and use at most four instruments.

The search space is bounded by:

* product filters (``filters`` in ``config/derivatives.yaml``),
* bucketing: per product type only the cheapest-spread product per bucket
  (barrier distance for KOs, moneyness × maturity for warrants) survives,
* per-class combinatorial constraints (barrier spacing, strike gaps, sizes),
* a hard cap on realizations per class (evenly spaced along the class
  parameter, so the retained set still spans the whole range).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .models import Derivative
from .pricing import MarketContext


@dataclass
class StrategyClass:
    key: str
    label: str
    description: str
    param_name: str  # primary parameter used for stability / tradeoff charts
    max_legs: int


STRATEGY_CLASSES: dict[str, StrategyClass] = {
    c.key: c
    for c in [
        StrategyClass(
            "ko_long",
            "Long knock-out",
            "Buy one long KO certificate (bullish, leveraged, path-dependent)",
            "barrier distance",
            1,
        ),
        StrategyClass(
            "ko_short",
            "Short knock-out",
            "Buy one short KO certificate (bearish, leveraged, path-dependent)",
            "barrier distance",
            1,
        ),
        StrategyClass(
            "call_warrant",
            "Call warrant",
            "Buy one call warrant (bullish, limited loss, vega/theta exposure)",
            "moneyness (K/S)",
            1,
        ),
        StrategyClass(
            "put_warrant",
            "Put warrant",
            "Buy one put warrant (bearish, limited loss)",
            "moneyness (K/S)",
            1,
        ),
        StrategyClass(
            "ko_long_ladder",
            "KO long ladder",
            "2–4 long KOs with staggered barriers (reduces all-or-nothing knock-out risk)",
            "mean barrier distance",
            4,
        ),
        StrategyClass(
            "ko_short_ladder",
            "KO short ladder",
            "2–4 short KOs with staggered barriers",
            "mean barrier distance",
            4,
        ),
        StrategyClass(
            "warrant_straddle",
            "Warrant straddle",
            "Call + put warrant with (near-)equal strikes, same maturity (volatility bet)",
            "days to maturity",
            2,
        ),
        StrategyClass(
            "warrant_strangle",
            "Warrant strangle",
            "OTM call + OTM put warrant, same maturity (cheaper volatility bet)",
            "wing width",
            2,
        ),
        StrategyClass(
            "ko_long_put_hedge",
            "KO long + put hedge",
            "Long KO with a put warrant limiting the downside",
            "barrier distance",
            2,
        ),
        StrategyClass(
            "ko_short_call_hedge",
            "KO short + call hedge",
            "Short KO with a call warrant limiting the upside risk",
            "barrier distance",
            2,
        ),
        StrategyClass(
            "ko_pair",
            "KO long + short",
            "Long and short KO with symmetric barriers (leveraged range breakout)",
            "mean barrier distance",
            2,
        ),
        StrategyClass(
            "call_warrant_ladder",
            "Call warrant ladder",
            "2–3 call warrants with different strikes, same maturity",
            "mean moneyness",
            3,
        ),
    ]
}


@dataclass
class Strategy:
    strategy_class: str
    underlying: str
    legs: list[Derivative]
    weights: list[float]
    param: float
    label: str = ""
    id: str = field(default="")

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.underlying}:{self.strategy_class}:" + "+".join(
                leg.isin for leg in self.legs
            )
        if not self.label:
            self.label = " + ".join(f"{leg.name or leg.isin}" for leg in self.legs)

    @property
    def n_legs(self) -> int:
        return len(self.legs)


# --------------------------------------------------------------------------------------
# filtering and bucketing
# --------------------------------------------------------------------------------------


def barrier_distance(p: Derivative, spot: float) -> float:
    b = p.barrier if p.barrier is not None else p.strike
    return abs(b / spot - 1)


def filter_products(
    products: list[Derivative], ctx: MarketContext, filters: dict, horizon_days: int
) -> list[Derivative]:
    out = []
    ko_f, w_f = filters.get("ko", {}), filters.get("warrants", {})
    for p in products:
        if p.ask < float(filters.get("min_price", 0)) or p.spread_pct > float(
            filters.get("max_spread_pct", 1)
        ):
            continue
        if p.is_ko:
            d = barrier_distance(p, ctx.spot)
            if not (
                float(ko_f.get("min_barrier_distance", 0))
                <= d
                <= float(ko_f.get("max_barrier_distance", 1))
            ):
                continue
            if p.leverage and p.leverage > float(ko_f.get("max_leverage", 1e9)):
                continue
            # a long must have barrier below spot, a short above
            b = p.barrier if p.barrier is not None else p.strike
            if (p.product_type == "ko_long" and b >= ctx.spot) or (
                p.product_type == "ko_short" and b <= ctx.spot
            ):
                continue
            if p.maturity is not None and (p.maturity - ctx.as_of).days < horizon_days:
                continue
        else:
            if p.maturity is None:
                continue
            dtm = (p.maturity - ctx.as_of).days
            if dtm < horizon_days + int(
                w_f.get("min_days_to_maturity_after_horizon", 0)
            ) or dtm > int(w_f.get("max_days_to_maturity", 10**6)):
                continue
            lo, hi = w_f.get("moneyness_range", [0, 10])
            if not (lo <= p.strike / ctx.spot <= hi):
                continue
        out.append(p)
    return out


def bucket_products(
    products: list[Derivative], ctx: MarketContext, sampling: dict
) -> dict[str, list[Derivative]]:
    """Per product type: cheapest-spread product per bucket, capped and sorted by parameter."""
    by_type: dict[str, dict[tuple, Derivative]] = {}
    mat_edges = sampling.get("warrant_maturity_buckets_days", [60, 120, 240, 400, 800])
    for p in products:
        if p.is_ko:
            key = (
                round(
                    barrier_distance(p, ctx.spot) / float(sampling.get("ko_bucket_width", 0.025))
                ),
            )
        else:
            dtm = (p.maturity - ctx.as_of).days
            mb = int(np.searchsorted(mat_edges, dtm))
            key = (
                mb,
                round(p.strike / ctx.spot / float(sampling.get("warrant_strike_bucket", 0.05))),
            )
        bucket = by_type.setdefault(p.product_type, {})
        if key not in bucket or p.spread_pct < bucket[key].spread_pct:
            bucket[key] = p
    cap = int(sampling.get("max_products_per_type", 12))
    out: dict[str, list[Derivative]] = {}
    for t, bucket in by_type.items():
        items = list(bucket.values())
        items.sort(
            key=lambda p: (
                barrier_distance(p, ctx.spot)
                if p.is_ko
                else ((p.maturity - ctx.as_of).days, p.strike)
            )
        )
        if len(items) > cap:
            idx = np.linspace(0, len(items) - 1, cap).round().astype(int)
            items = [items[i] for i in sorted(set(idx))]
        out[t] = items
    return out


# --------------------------------------------------------------------------------------
# realization sampling per class
# --------------------------------------------------------------------------------------


def _cap(reals: list[Strategy], cap: int) -> list[Strategy]:
    if len(reals) <= cap:
        return reals
    reals = sorted(reals, key=lambda s: s.param)
    idx = np.linspace(0, len(reals) - 1, cap).round().astype(int)
    return [reals[i] for i in sorted(set(idx))]


def sample_realizations(
    buckets: dict[str, list[Derivative]],
    ctx: MarketContext,
    sampling: dict,
    classes: list[str] | None = None,
) -> list[Strategy]:
    S = ctx.spot
    kol, kos = buckets.get("ko_long", []), buckets.get("ko_short", [])
    calls, puts = buckets.get("call_warrant", []), buckets.get("put_warrant", [])
    cap = int(sampling.get("max_realizations_per_class", 80))
    spacing = float(sampling.get("ladder_min_barrier_spacing", 0.03))
    sizes = [int(s) for s in sampling.get("ladder_sizes", [2, 3, 4])]
    out: list[Strategy] = []
    want = set(classes or STRATEGY_CLASSES)

    def add(cls: str, reals: list[Strategy]):
        if cls in want:
            out.extend(_cap(reals, cap))

    add(
        "ko_long",
        [Strategy("ko_long", ctx.ticker, [p], [1.0], barrier_distance(p, S)) for p in kol],
    )
    add(
        "ko_short",
        [Strategy("ko_short", ctx.ticker, [p], [1.0], barrier_distance(p, S)) for p in kos],
    )
    add(
        "call_warrant",
        [Strategy("call_warrant", ctx.ticker, [p], [1.0], p.strike / S) for p in calls],
    )
    add(
        "put_warrant", [Strategy("put_warrant", ctx.ticker, [p], [1.0], p.strike / S) for p in puts]
    )

    def ladders(cls: str, prods: list[Derivative]) -> list[Strategy]:
        reals = []
        ds = [(barrier_distance(p, S), p) for p in prods]
        for n in sizes:
            if n > 4:
                continue
            for combo in itertools.combinations(ds, n):
                dist = [d for d, _ in combo]
                if min(np.diff(dist)) < spacing:
                    continue
                legs = [p for _, p in combo]
                reals.append(Strategy(cls, ctx.ticker, legs, [1 / n] * n, float(np.mean(dist))))
        return reals

    add("ko_long_ladder", ladders("ko_long_ladder", kol))
    add("ko_short_ladder", ladders("ko_short_ladder", kos))

    gap = float(sampling.get("straddle_max_strike_gap", 0.06))
    wing = float(sampling.get("strangle_min_wing", 0.04))
    straddles, strangles = [], []
    for c in calls:
        for p in puts:
            if c.maturity != p.maturity:
                continue
            kc, kp = c.strike / S, p.strike / S
            if abs(kc - kp) <= gap and abs(kc - 1) <= gap and abs(kp - 1) <= gap:
                straddles.append(
                    Strategy(
                        "warrant_straddle",
                        ctx.ticker,
                        [c, p],
                        [0.5, 0.5],
                        float((c.maturity - ctx.as_of).days),
                    )
                )
            if kc >= 1 + wing and kp <= 1 - wing:
                strangles.append(
                    Strategy("warrant_strangle", ctx.ticker, [c, p], [0.5, 0.5], float(kc - kp))
                )
    add("warrant_straddle", straddles)
    add("warrant_strangle", strangles)

    hedges = []
    for k in kol:
        # hedge with the put closest to ATM among the shortest maturities
        cands = sorted(
            puts, key=lambda p: (abs(p.strike / S - 0.97), (p.maturity - ctx.as_of).days)
        )[:2]
        for p in cands:
            hedges.append(
                Strategy(
                    "ko_long_put_hedge", ctx.ticker, [k, p], [0.7, 0.3], barrier_distance(k, S)
                )
            )
    add("ko_long_put_hedge", hedges)
    hedges = []
    for k in kos:
        cands = sorted(
            calls, key=lambda c: (abs(c.strike / S - 1.03), (c.maturity - ctx.as_of).days)
        )[:2]
        for c in cands:
            hedges.append(
                Strategy(
                    "ko_short_call_hedge", ctx.ticker, [k, c], [0.7, 0.3], barrier_distance(k, S)
                )
            )
    add("ko_short_call_hedge", hedges)

    pairs = []
    for k1 in kol:
        d1 = barrier_distance(k1, S)
        for k2 in kos:
            d2 = barrier_distance(k2, S)
            if abs(d1 - d2) <= spacing:
                pairs.append(
                    Strategy("ko_pair", ctx.ticker, [k1, k2], [0.5, 0.5], float((d1 + d2) / 2))
                )
    add("ko_pair", pairs)

    cl = []
    by_mat: dict = {}
    for c in calls:
        by_mat.setdefault(c.maturity, []).append(c)
    for _, group in by_mat.items():
        group = sorted(group, key=lambda c: c.strike)
        for n in (2, 3):
            for combo in itertools.combinations(group, n):
                strikes = [c.strike / S for c in combo]
                if min(np.diff(strikes)) < 0.03:
                    continue
                cl.append(
                    Strategy(
                        "call_warrant_ladder",
                        ctx.ticker,
                        list(combo),
                        [1 / n] * n,
                        float(np.mean(strikes)),
                    )
                )
    add("call_warrant_ladder", cl)
    return out
