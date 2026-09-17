"""Cross-sectional composite scores (percentile-rank blends across the universe).

Scores are 0–100 conveniences for sorting; they are not forecasts. Weights are
kept explicit so they can be audited and changed.
"""

from __future__ import annotations

import pandas as pd

SCORES = {
    # name: [(column, weight, higher_is_better)]
    "momentum_score": [
        ("mom_60", 1.0, True),
        ("mom_120", 1.0, True),
        ("trend_score", 1.5, True),
        ("rsi_14", 0.5, True),
        ("volume_trend", 0.5, True),
    ],
    "quality_score": [
        ("roe", 1.0, True),
        ("net_margin", 1.0, True),
        ("revenue_growth", 1.0, True),
        ("debt_to_equity", 0.5, False),
        ("fcf_yield", 0.5, True),
    ],
    "value_score": [
        ("pe", 1.0, False),
        ("ev_ebitda", 1.0, False),
        ("ps", 0.5, False),
        ("fcf_yield", 1.0, True),
    ],
    # higher risk score = riskier
    "risk_score": [
        ("p_loss_10", 1.5, True),
        ("es_5", 1.0, False),
        ("iv_30", 1.0, True),
        ("rv_60", 0.5, True),
        ("max_dd_1y", 0.5, False),
        ("rn_skew", 0.5, False),
    ],
}


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, spec in SCORES.items():
        total = pd.Series(0.0, index=df.index)
        weight = pd.Series(0.0, index=df.index)
        for col, w, hib in spec:
            if col not in df:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() < 2:
                continue
            pct = s.rank(pct=True, ascending=hib) * 100
            mask = pct.notna()
            total[mask] += w * pct[mask]
            weight[mask] += w
        df[name] = (total / weight.where(weight > 0)).round(1)
    if "momentum_score" in df and "risk_score" in df:
        parts = [c for c in ("momentum_score", "quality_score", "value_score") if c in df]
        inv_risk = 100 - df["risk_score"]
        blend = df[parts].mean(axis=1, skipna=True) * 0.7 + inv_risk * 0.3
        df["screener_score"] = blend.round(1)
    return df
