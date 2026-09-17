"""Dash application: AG Grid screener table with configurable columns, filters,
presets, CSV export and a per-ticker detail panel.

The app reads only local artefacts written by ``screener refresh`` (parquet
files under ``data_dir``); it never calls the market-data API itself.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

from ..analytics.chain import SmileFit
from ..config import Settings
from ..data.cache import DataCache
from . import derivatives_tab
from .columns import BY_KEY, COLUMNS, GROUPS, ColumnSpec
from .presets import PRESETS

# palette (light surface) — see docs/PLAN.md §6 and the dataviz reference palette
PAL = {
    "s1": "#2a78d6",
    "s2": "#eb6834",
    "s3": "#1baf7a",
    "s4": "#eda100",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "good": "#006300",
    "bad": "#d03b3b",
    "neutral": "#f0efec",
}

FORMATTERS = {
    "pct": "params.value == null ? '' : d3.format('+.1%')(params.value)",
    "pct2": "params.value == null ? '' : d3.format('+.2%')(params.value)",
    "vol": "params.value == null ? '' : d3.format('.1%')(params.value)",
    "num": "params.value == null ? '' : d3.format(',.1f')(params.value)",
    "num2": "params.value == null ? '' : d3.format(',.2f')(params.value)",
    "int": "params.value == null ? '' : d3.format(',.0f')(params.value)",
    "money": "params.value == null ? '' : d3.format('$.3s')(params.value).replace('G','B')",
    "score": "params.value == null ? '' : d3.format('.0f')(params.value)",
    "text": "params.value == null ? '' : params.value",
}


# --------------------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------------------


class ScreenerStore:
    def __init__(self, data_dir: Path):
        self.cache = DataCache(data_dir)
        self.df: pd.DataFrame = pd.DataFrame()
        self.as_of: dt.date | None = None
        self.reload()

    def reload(self) -> None:
        df = self.cache.get_screener()
        if df is None:
            self.df = pd.DataFrame(columns=["ticker"])
            self.as_of = None
            return
        self.df = df
        self.as_of = (
            dt.date.fromisoformat(str(df["as_of"].iloc[0])) if "as_of" in df and len(df) else None
        )

    def records(self, df: pd.DataFrame) -> list[dict]:
        clean = df.replace({np.nan: None})
        return clean.to_dict("records")

    def bars(self, ticker: str) -> pd.DataFrame | None:
        return self.cache.get_frame(self.as_of, "bars", ticker) if self.as_of else None

    def chain(self, ticker: str) -> pd.DataFrame | None:
        return self.cache.get_chain_summary(self.as_of, ticker) if self.as_of else None

    def fits(self, ticker: str) -> list[SmileFit]:
        df = self.cache.get_frame(self.as_of, "fits", ticker) if self.as_of else None
        if df is None or df.empty:
            return []
        return [
            SmileFit(
                expiry=pd.Timestamp(r.expiry).date(),
                t=r.t,
                forward=r.forward,
                coef=(r.a, r.b, r.c),
                k_min=r.k_min,
                k_max=r.k_max,
                n=int(r.n),
                rmse=r.rmse,
                forward_source=r.forward_source,
            )
            for r in df.itertuples()
        ]

    def density(self, ticker: str) -> pd.DataFrame | None:
        return self.cache.get_density(self.as_of, ticker) if self.as_of else None


# --------------------------------------------------------------------------------------
# grid helpers
# --------------------------------------------------------------------------------------


def column_def(spec: ColumnSpec) -> dict:
    d: dict = {
        "field": spec.key,
        "headerName": spec.label,
        "headerTooltip": spec.tooltip,
        "valueFormatter": {"function": FORMATTERS[spec.fmt]},
        "filter": "agTextColumnFilter" if spec.fmt == "text" else "agNumberColumnFilter",
        "sortable": True,
        "resizable": True,
    }
    if spec.fmt != "text":
        d["type"] = "rightAligned"
        d["filterParams"] = {"buttons": ["reset", "apply"], "closeOnApply": True}
    if spec.width:
        d["width"] = spec.width
    if spec.key == "ticker":
        d.update(
            {"pinned": "left", "cellStyle": {"fontWeight": 600}, "filter": "agTextColumnFilter"}
        )
    elif spec.sign != 0 and spec.fmt in ("pct", "pct2", "num2", "vol"):
        pos, neg = (PAL["good"], PAL["bad"]) if spec.sign > 0 else (PAL["bad"], PAL["good"])
        d["cellStyle"] = {
            "styleConditions": [
                {"condition": "params.value > 0", "style": {"color": pos}},
                {"condition": "params.value < 0", "style": {"color": neg}},
            ]
        }
    elif spec.fmt == "score":
        d["cellStyle"] = {
            "styleConditions": [
                {"condition": "params.value >= 75", "style": {"backgroundColor": "#cde2fb"}},
                {"condition": "params.value < 25", "style": {"backgroundColor": PAL["neutral"]}},
            ]
        }
    return d


def column_defs(keys: list[str], available: set[str]) -> list[dict]:
    return [
        column_def(BY_KEY[k]) for k in keys if k in BY_KEY and (k in available or k == "ticker")
    ]


def apply_global_filters(
    df: pd.DataFrame, sectors, min_mcap, min_dvol, min_oi, only_options, only_fund
) -> pd.DataFrame:
    out = df
    if sectors and "sector" in out:
        out = out[out["sector"].isin(sectors)]
    if min_mcap and "market_cap" in out:
        out = out[out["market_cap"].fillna(0) >= float(min_mcap)]
    if min_dvol and "dollar_volume_20" in out:
        out = out[out["dollar_volume_20"].fillna(0) >= float(min_dvol)]
    if min_oi and "open_interest" in out:
        out = out[out["open_interest"].fillna(0) >= float(min_oi)]
    if only_options and "iv_30" in out:
        out = out[out["iv_30"].notna()]
    if only_fund and "has_fundamentals" in out:
        out = out[out["has_fundamentals"].fillna(False).astype(bool)]
    return out


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def _base_layout(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": PAL["ink"]}, "x": 0.01},
        height=height,
        margin={"l": 50, "r": 16, "t": 40, "b": 40},
        paper_bgcolor=PAL["surface"],
        plot_bgcolor=PAL["surface"],
        font={"color": PAL["ink2"], "size": 12},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"size": 11}},
    )
    fig.update_xaxes(gridcolor=PAL["grid"], linecolor=PAL["axis"], zeroline=False)
    fig.update_yaxes(gridcolor=PAL["grid"], linecolor=PAL["axis"], zeroline=False)
    return fig


def empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font={"color": PAL["muted"]})
    return _base_layout(fig, "", 240)


def price_figure(bars: pd.DataFrame, ticker: str) -> go.Figure:
    b = bars.iloc[-252:]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04
    )
    fig.add_trace(
        go.Scatter(x=b.index, y=b["close"], name="Close", line={"color": PAL["s1"], "width": 2}),
        1,
        1,
    )
    s50 = bars["close"].rolling(50).mean().iloc[-252:]
    s200 = bars["close"].rolling(200).mean().iloc[-252:]
    fig.add_trace(
        go.Scatter(x=b.index, y=s50, name="SMA 50", line={"color": PAL["s2"], "width": 1.5}), 1, 1
    )
    fig.add_trace(
        go.Scatter(x=b.index, y=s200, name="SMA 200", line={"color": PAL["s3"], "width": 1.5}), 1, 1
    )
    fig.add_trace(go.Bar(x=b.index, y=b["volume"], name="Volume", marker_color=PAL["axis"]), 2, 1)
    _base_layout(fig, f"{ticker} — price, moving averages and volume (1y)", 380)
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Shares", row=2, col=1, tickformat="~s")
    return fig


def smile_figure(chain: pd.DataFrame, fits: list[SmileFit], spot: float, ticker: str) -> go.Figure:
    if chain is None or chain.empty or not fits:
        return empty_fig("No option chain data")
    targets = [30, 60, 90]
    chosen: list[SmileFit] = []
    for d in targets:
        f = min(fits, key=lambda f: abs(f.t * 365 - d))
        if f not in chosen:
            chosen.append(f)
    colors = [PAL["s1"], PAL["s2"], PAL["s3"]]
    fig = go.Figure()
    for f, col in zip(chosen, colors, strict=False):
        pts = chain[chain["expiry"].astype(str) == str(f.expiry)]
        label = f"{f.expiry:%d %b %Y} ({round(f.t * 365)}d)"
        fig.add_trace(
            go.Scatter(
                x=pts["strike"] / spot - 1,
                y=pts["iv"],
                mode="markers",
                name=f"{label} contracts",
                marker={"color": col, "size": 7, "opacity": 0.55},
                showlegend=False,
                hovertemplate="K/S-1 %{x:.1%}<br>IV %{y:.1%}<extra>" + label + "</extra>",
            )
        )
        kk = np.linspace(min(f.k_min, -0.02) - 0.05, max(f.k_max, 0.02) + 0.05, 120)
        strikes = f.forward * np.exp(kk)
        fig.add_trace(
            go.Scatter(
                x=strikes / spot - 1,
                y=f.iv(kk),
                mode="lines",
                name=label,
                line={"color": col, "width": 2},
            )
        )
    fig.add_vline(x=0, line={"color": PAL["muted"], "dash": "dot", "width": 1})
    _base_layout(fig, f"{ticker} — implied volatility smile (fit vs. contracts)", 320)
    fig.update_xaxes(title_text="Strike / spot − 1", tickformat="+.0%")
    fig.update_yaxes(title_text="Implied vol", tickformat=".0%")
    fig.update_layout(hovermode="closest")
    return fig


def term_figure(fits: list[SmileFit], rv20: float | None, ticker: str) -> go.Figure:
    if not fits:
        return empty_fig("No option chain data")
    x = [round(f.t * 365) for f in fits]
    y = [f.atm_iv for f in fits]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="ATM IV",
            line={"color": PAL["s1"], "width": 2},
            marker={"size": 8},
            hovertemplate="%{x}d: %{y:.1%}<extra></extra>",
        )
    )
    if rv20:
        fig.add_hline(
            y=rv20,
            line={"color": PAL["s2"], "width": 1.5, "dash": "dash"},
            annotation_text=f"RV 20d {rv20:.1%}",
            annotation_position="top right",
        )
    _base_layout(fig, f"{ticker} — ATM implied volatility term structure", 320)
    fig.update_xaxes(title_text="Days to expiry")
    lo = min(y + ([rv20] if rv20 else [])) * 0.9
    hi = max(y + ([rv20] if rv20 else [])) * 1.1
    fig.update_yaxes(title_text="Implied vol", tickformat=".0%", range=[lo, hi])
    return fig


def density_figure(
    dens: pd.DataFrame, spot: float, ticker: str, thresholds=(0.05, 0.10, 0.20)
) -> go.Figure:
    if dens is None or dens.empty:
        return empty_fig("No risk-neutral density")
    r = dens["strike"].to_numpy() / spot - 1
    f = dens["density"].to_numpy() * spot  # density of the return
    mask = (r > -0.6) & (r < 0.6)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=r[mask],
            y=f[mask],
            mode="lines",
            name="RN density",
            line={"color": PAL["s1"], "width": 2},
            hovertemplate="return %{x:+.1%}<extra></extra>",
        )
    )
    loss = mask & (r <= -thresholds[1])
    fig.add_trace(
        go.Scatter(
            x=r[loss],
            y=f[loss],
            mode="lines",
            fill="tozeroy",
            name=f"loss ≥ {thresholds[1]:.0%}",
            line={"color": PAL["bad"], "width": 0},
            fillcolor="rgba(208,59,59,0.25)",
        )
    )
    gain = mask & (r >= thresholds[1])
    fig.add_trace(
        go.Scatter(
            x=r[gain],
            y=f[gain],
            mode="lines",
            fill="tozeroy",
            name=f"gain ≥ {thresholds[1]:.0%}",
            line={"color": PAL["good"], "width": 0},
            fillcolor="rgba(0,99,0,0.2)",
        )
    )
    fig.add_vline(x=0, line={"color": PAL["muted"], "dash": "dot", "width": 1})
    _base_layout(fig, f"{ticker} — 30-day risk-neutral return distribution", 320)
    fig.update_xaxes(title_text="30-day return", tickformat="+.0%")
    fig.update_yaxes(title_text="Density", showticklabels=False)
    fig.update_layout(hovermode="x")
    return fig


def stats_card(row: pd.Series) -> html.Div:
    def fmt(key, f="{:+.1%}"):
        v = row.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "–"
        try:
            return f.format(v)
        except (ValueError, TypeError):
            return str(v)

    items = [
        ("Price", fmt("price", "{:,.2f}")),
        ("IV 30 / RV 20", f"{fmt('iv_30', '{:.1%}')} / {fmt('rv_20', '{:.1%}')}"),
        ("Expected move 30d", fmt("expected_move", "±{:.1%}")),
        ("P(loss ≥ 10%)", fmt("p_loss_10", "{:.1%}")),
        ("P(gain ≥ 10%)", fmt("p_gain_10", "{:.1%}")),
        ("VaR 5% / ES 5%", f"{fmt('var_5')} / {fmt('es_5')}"),
        ("Skew 25Δ", fmt("skew_25d", "{:+.1%}")),
        ("Term slope", fmt("term_slope", "{:+.1%}")),
        ("RSI 14", fmt("rsi_14", "{:.0f}")),
        ("Trend", fmt("trend_score", "{:+.2f}")),
        ("Ret 3m / 1y", f"{fmt('ret_3m')} / {fmt('ret_1y')}"),
        ("P/E · EV/EBITDA", f"{fmt('pe', '{:.1f}')} · {fmt('ev_ebitda', '{:.1f}')}"),
        ("Open interest", fmt("open_interest", "{:,.0f}")),
        ("Fit RMSE", fmt("fit_rmse", "{:.2%}")),
    ]
    tiles = [
        html.Div(
            [html.Div(k, className="tile-label"), html.Div(v, className="tile-value")],
            className="tile",
        )
        for k, v in items
    ]
    warn = row.get("warnings")
    extra = (
        html.Div(f"Warnings: {warn}", className="warn") if isinstance(warn, str) and warn else None
    )
    return html.Div(
        [html.Div(tiles, className="tiles"), extra]
        if extra
        else [html.Div(tiles, className="tiles")]
    )


# --------------------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------------------

CSS = """
body { background: #f9f9f7; color: #0b0b0b; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; }
.header { display:flex; align-items:baseline; gap:16px; padding: 12px 16px 6px; border-bottom: 1px solid #e1e0d9; background:#fcfcfb; }
.header h1 { font-size: 18px; margin: 0; }
.status { color:#52514e; font-size: 12px; }
.layout { display:grid; grid-template-columns: 290px 1fr; gap: 12px; padding: 12px 0; }
.main-tabs { padding: 0 16px; }
.main-tabs .tab { padding: 8px 14px !important; font-size: 14px; }
table.classes { border-collapse: collapse; font-size: 12px; margin: 6px 0 10px; }
table.classes th, table.classes td { border-bottom: 1px solid #e1e0d9; padding: 4px 10px 4px 0; text-align: left; vertical-align: top; }
.sidebar { background:#fcfcfb; border:1px solid #e1e0d9; border-radius:8px; padding:12px; font-size: 13px; max-height: calc(100vh - 90px); overflow:auto; }
.sidebar h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color:#898781; margin: 12px 0 6px; }
.sidebar label { display:flex; align-items:center; gap:6px; margin: 2px 0; }
.sidebar .flabel { display:block; margin-top:6px; color:#52514e; }
.main { min-width: 0; }
.toolbar { display:flex; gap:8px; align-items:center; margin-bottom: 8px; flex-wrap: wrap; }
.toolbar input { padding: 6px 8px; border:1px solid #c3c2b7; border-radius: 6px; min-width: 240px; }
button.btn { padding: 6px 10px; border:1px solid #c3c2b7; background:#fcfcfb; border-radius:6px; cursor:pointer; }
.detail { margin-top: 12px; background:#fcfcfb; border:1px solid #e1e0d9; border-radius:8px; padding: 12px; }
.tiles { display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; margin-bottom: 8px; }
.tile { border:1px solid #e1e0d9; border-radius:6px; padding: 8px; background:#fcfcfb; }
.tile-label { font-size: 11px; color:#898781; }
.tile-value { font-size: 15px; font-weight: 600; }
.warn { color:#d03b3b; font-size: 12px; margin-top: 4px; }
.charts { display:grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.note { font-size: 11px; color:#898781; margin-top: 6px; }
details summary { cursor:pointer; font-weight: 600; }
@media (max-width: 1100px) { .layout { grid-template-columns: 1fr; } .charts { grid-template-columns: 1fr; } }
"""


def create_app(settings: Settings | None = None, data_dir: Path | None = None) -> Dash:
    settings = settings or Settings()
    store = ScreenerStore(Path(data_dir or settings.data_dir))
    dstore = derivatives_tab.DerivativesStore(settings, Path(data_dir or settings.data_dir))
    app = Dash(__name__, title="Stock Screener", suppress_callback_exceptions=True)
    app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1("Stock Screener"),
                    html.Span(_status_text(store, settings), className="status", id="status"),
                ],
                className="header",
            ),
            dcc.Tabs(
                id="main-tabs",
                value="screener",
                className="main-tabs",
                children=[
                    dcc.Tab(label="Screener", value="screener", children=_layout(store, settings)),
                    dcc.Tab(
                        label="Derivatives (KO certificates & warrants)",
                        value="derivatives",
                        children=html.Div(
                            derivatives_tab.layout(dstore), style={"padding": "8px 16px"}
                        ),
                    ),
                ],
            ),
        ]
    )
    _register_callbacks(app, store, settings)
    derivatives_tab.register_callbacks(app, dstore)
    app.store = store  # type: ignore[attr-defined]
    app.dstore = dstore  # type: ignore[attr-defined]
    return app


def _status_text(store: ScreenerStore, settings: Settings) -> str:
    df = store.df
    n_opt = int(df["iv_30"].notna().sum()) if "iv_30" in df else 0
    if not store.as_of:
        return "No data yet — run `screener refresh` (or `screener demo`)."
    return (
        f"Data as of {store.as_of} · {len(df)} tickers · {n_opt} with option analytics · "
        f"provider: {settings.resolved_provider()} · r = {settings.risk_free_rate:.2%}"
    )


def _layout(store: ScreenerStore, settings: Settings) -> html.Div:
    df = store.df
    sectors = sorted(s for s in df.get("sector", pd.Series(dtype=str)).dropna().unique() if s)
    group_checklists = []
    for g in GROUPS:
        cols = [c for c in COLUMNS if c.group == g]
        group_checklists.append(
            html.Details(
                [
                    html.Summary(g),
                    dcc.Checklist(
                        id={"type": "colgroup", "group": g},
                        options=[
                            {"label": c.label, "value": c.key, "title": c.tooltip} for c in cols
                        ],
                        value=[c.key for c in cols if c.default],
                        labelStyle={"display": "flex", "alignItems": "center", "gap": "6px"},
                        inputStyle={"margin": "0"},
                    ),
                ],
                open=(g == "Identification"),
            )
        )
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Column preset"),
                            dcc.Dropdown(
                                id="preset",
                                options=[{"label": k, "value": k} for k in PRESETS],
                                value="Overview",
                                clearable=False,
                            ),
                            html.H3("Global filters"),
                            html.Label("Sector", className="flabel"),
                            dcc.Dropdown(
                                id="f-sector",
                                options=[{"label": s, "value": s} for s in sectors],
                                multi=True,
                                placeholder="All sectors",
                            ),
                            html.Label("Min market cap", className="flabel"),
                            dcc.Dropdown(
                                id="f-mcap",
                                value=0,
                                clearable=False,
                                options=[
                                    {"label": "Any", "value": 0},
                                    {"label": "≥ $300M", "value": 3e8},
                                    {"label": "≥ $2B", "value": 2e9},
                                    {"label": "≥ $10B", "value": 1e10},
                                    {"label": "≥ $50B", "value": 5e10},
                                    {"label": "≥ $200B", "value": 2e11},
                                ],
                            ),
                            html.Label("Min avg $ volume (20d)", className="flabel"),
                            dcc.Dropdown(
                                id="f-dvol",
                                value=0,
                                clearable=False,
                                options=[
                                    {"label": "Any", "value": 0},
                                    {"label": "≥ $10M", "value": 1e7},
                                    {"label": "≥ $50M", "value": 5e7},
                                    {"label": "≥ $250M", "value": 2.5e8},
                                    {"label": "≥ $1B", "value": 1e9},
                                ],
                            ),
                            html.Label("Min option open interest", className="flabel"),
                            dcc.Dropdown(
                                id="f-oi",
                                value=0,
                                clearable=False,
                                options=[
                                    {"label": "Any", "value": 0},
                                    {"label": "≥ 1k", "value": 1e3},
                                    {"label": "≥ 10k", "value": 1e4},
                                    {"label": "≥ 100k", "value": 1e5},
                                    {"label": "≥ 1M", "value": 1e6},
                                ],
                            ),
                            dcc.Checklist(
                                id="f-flags",
                                options=[
                                    {
                                        "label": " Only tickers with option analytics",
                                        "value": "options",
                                    },
                                    {"label": " Only tickers with fundamentals", "value": "fund"},
                                ],
                                value=[],
                                labelStyle={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "6px",
                                },
                            ),
                            html.H3("Columns"),
                            html.Div(group_checklists),
                            html.Div(
                                "Per-column filters and multi-column sort (shift-click) are available in the table header. "
                                "Option-derived expectations are risk-neutral: they embed risk premia.",
                                className="note",
                            ),
                        ],
                        className="sidebar",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Input(
                                        id="quick",
                                        placeholder="Quick search (ticker, name, sector)…",
                                        debounce=True,
                                    ),
                                    html.Button("Export CSV", id="export", className="btn"),
                                    html.Button("Reload data", id="reload", className="btn"),
                                    html.Span(id="rowcount", className="status"),
                                ],
                                className="toolbar",
                            ),
                            dag.AgGrid(
                                id="grid",
                                rowData=[],
                                columnDefs=[],
                                defaultColDef={
                                    "sortable": True,
                                    "filter": True,
                                    "resizable": True,
                                    "floatingFilter": False,
                                    "minWidth": 90,
                                    "wrapHeaderText": True,
                                    "autoHeaderHeight": True,
                                },
                                dashGridOptions={
                                    "rowSelection": "single",
                                    "animateRows": False,
                                    "tooltipShowDelay": 300,
                                    "enableCellTextSelection": True,
                                    "suppressMenuHide": True,
                                    "rowHeight": 30,
                                },
                                columnSize="autoSize",
                                style={"height": "58vh", "width": "100%"},
                                csvExportParams={"fileName": "screener.csv"},
                            ),
                            html.Div(
                                id="detail",
                                className="detail",
                                children=html.Div("Select a row to see details.", className="note"),
                            ),
                        ],
                        className="main",
                    ),
                ],
                className="layout",
            ),
            dcc.Store(id="reload-token", data=0),
        ]
    )


def _register_callbacks(app: Dash, store: ScreenerStore, settings: Settings) -> None:
    from dash import ALL

    @app.callback(
        Output({"type": "colgroup", "group": ALL}, "value"),
        Input("preset", "value"),
        State({"type": "colgroup", "group": ALL}, "id"),
        prevent_initial_call=True,
    )
    def apply_preset(preset, ids):
        keys = set(PRESETS.get(preset, []))
        return [[c.key for c in COLUMNS if c.group == i["group"] and c.key in keys] for i in ids]

    @app.callback(
        Output("grid", "rowData"),
        Output("grid", "columnDefs"),
        Output("rowcount", "children"),
        Output("status", "children"),
        Input({"type": "colgroup", "group": ALL}, "value"),
        Input("f-sector", "value"),
        Input("f-mcap", "value"),
        Input("f-dvol", "value"),
        Input("f-oi", "value"),
        Input("f-flags", "value"),
        Input("reload", "n_clicks"),
    )
    def update_grid(col_values, sectors, mcap, dvol, oi, flags, _n):
        if ctx.triggered_id == "reload":
            store.reload()
        df = store.df
        keys = ["ticker"] + [k for vals in col_values for k in vals if k != "ticker"]
        # keep registry order
        order = {c.key: i for i, c in enumerate(COLUMNS)}
        keys = sorted(dict.fromkeys(keys), key=lambda k: order.get(k, 1e9))
        flags = flags or []
        f = apply_global_filters(df, sectors, mcap, dvol, oi, "options" in flags, "fund" in flags)
        cols = column_defs(keys, set(df.columns))
        n_opt = int(df["iv_30"].notna().sum()) if "iv_30" in df else 0
        status = (
            f"Data as of {store.as_of} · {len(df)} tickers · {n_opt} with option analytics · "
            f"provider: {settings.resolved_provider()} · r = {settings.risk_free_rate:.2%}"
        )
        return store.records(f), cols, f"{len(f)} of {len(df)} rows", status

    @app.callback(
        Output("grid", "dashGridOptions"), Input("quick", "value"), State("grid", "dashGridOptions")
    )
    def quick_filter(text, opts):
        opts = dict(opts or {})
        opts["quickFilterText"] = text or ""
        return opts

    @app.callback(
        Output("grid", "exportDataAsCsv"), Input("export", "n_clicks"), prevent_initial_call=True
    )
    def export_csv(_n):
        return True

    @app.callback(Output("detail", "children"), Input("grid", "selectedRows"))
    def detail(selected):
        if not selected:
            return no_update
        ticker = selected[0]["ticker"]
        rows = store.df[store.df["ticker"] == ticker]
        if rows.empty:
            return html.Div("No data", className="note")
        row = rows.iloc[0]
        spot = float(row.get("price") or row.get("chain_spot") or 0) or None
        bars = store.bars(ticker)
        chain = store.chain(ticker)
        fits = store.fits(ticker)
        dens = store.density(ticker)
        figs = [
            price_figure(bars, ticker)
            if bars is not None and len(bars) > 30
            else empty_fig("No price history"),
            smile_figure(chain, fits, spot, ticker) if spot else empty_fig("No spot"),
            term_figure(fits, row.get("rv_20"), ticker),
            density_figure(dens, spot, ticker, tuple(settings.loss_thresholds))
            if spot
            else empty_fig("No spot"),
        ]
        return html.Div(
            [
                html.H3(
                    f"{ticker} · {row.get('name', '')} · {row.get('sector', '')}",
                    style={"margin": "0 0 8px"},
                ),
                stats_card(row),
                html.Div(
                    [dcc.Graph(figure=f, config={"displayModeBar": False}) for f in figs],
                    className="charts",
                ),
                html.Div(
                    "Risk-neutral quantities are derived from the fitted implied-volatility surface via "
                    "Breeden–Litzenberger (QuantLib Black pricing). They embed risk premia and are not "
                    "real-world forecasts.",
                    className="note",
                ),
            ]
        )
