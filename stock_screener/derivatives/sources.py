"""Product sources: Scalable Capital CLI adapter, exported-file adapter, synthetic generator.

The CLI adapter only *queries* products. Trading is deliberately not wired up:
the dashboard never places orders.

ASSUMPTION (documented in ``config/derivatives.yaml``): the CLI's sub-command,
flags and output keys are configurable because the tool's interface could not
be verified when this module was written. Use ``screener derivatives probe`` to
check the mapping against the real tool.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .models import PRODUCT_TYPES, Derivative
from .pricing import MarketContext, warrant_price_from_vol

log = logging.getLogger(__name__)


class DerivativesSource(Protocol):
    name: str

    def list_products(self, ticker: str, ctx: MarketContext | None = None) -> list[Derivative]: ...


# --------------------------------------------------------------------------------------
# parsing helpers shared by the CLI and file adapters
# --------------------------------------------------------------------------------------


def _get_path(d: dict, path: str | None):
    if not path:
        return None
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        v = v.replace(",", ".").replace("%", "").strip()
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _date(v) -> dt.date | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("open-end", "open end", "openend", "null", "none", "unlimited"):
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y%m%d"):
        try:
            return dt.datetime.strptime(s[: len(fmt) + 2] if "T" in fmt else s[:10], fmt).date()
        except ValueError:
            continue
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def parse_records(text: str, output_format: str) -> list[dict]:
    """JSON array / object with results / JSON lines / CSV -> list of dicts."""
    text = text.strip()
    if not text:
        return []
    if output_format == "csv":
        return list(csv.DictReader(io.StringIO(text)))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
    if isinstance(data, dict):
        for key in ("results", "data", "items", "products", "derivatives"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return data if isinstance(data, list) else []


def map_record(
    rec: dict,
    ticker: str,
    fields: dict,
    type_values: dict,
    source: str,
    default_type: str | None = None,
) -> Derivative | None:
    typ_raw = _get_path(rec, fields.get("product_type"))
    ptype = type_values.get(str(typ_raw), None) if typ_raw is not None else None
    ptype = (
        ptype
        or (str(typ_raw).lower() if str(typ_raw).lower() in PRODUCT_TYPES else None)
        or default_type
    )
    strike = _num(_get_path(rec, fields.get("strike")))
    bid, ask = _num(_get_path(rec, fields.get("bid"))), _num(_get_path(rec, fields.get("ask")))
    ratio = _num(_get_path(rec, fields.get("ratio"))) or 1.0
    isin = _get_path(rec, fields.get("isin")) or _get_path(rec, fields.get("wkn")) or ""
    if ptype not in PRODUCT_TYPES or strike is None or bid is None or ask is None or not isin:
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    return Derivative(
        underlying=ticker,
        isin=str(isin),
        product_type=ptype,
        strike=strike,
        ratio=ratio,
        bid=bid,
        ask=ask,
        barrier=_num(_get_path(rec, fields.get("barrier"))),
        maturity=_date(_get_path(rec, fields.get("maturity"))),
        wkn=str(_get_path(rec, fields.get("wkn")) or ""),
        name=str(_get_path(rec, fields.get("name")) or ""),
        issuer=str(_get_path(rec, fields.get("issuer")) or ""),
        leverage=_num(_get_path(rec, fields.get("leverage"))),
        financing_rate=_num(_get_path(rec, fields.get("financing_rate"))),
        currency=str(_get_path(rec, fields.get("currency")) or "EUR"),
        quote_time=(
            str(_get_path(rec, fields.get("quote_time")))
            if _get_path(rec, fields.get("quote_time"))
            else None
        ),
        source=source,
        extra={},
    )


# --------------------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------------------


class ScalableCliSource:
    """Runs the Scalable Capital CLI (query only) once per underlying and product type."""

    name = "cli"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cli = cfg["cli"]
        self.underlying_map = cfg.get("underlying_map") or {}
        self.warnings: list[str] = []

    def _run(self, ticker: str, ptype: str) -> str:
        underlying = self.underlying_map.get(ticker, ticker)
        type_arg = (self.cli.get("type_args") or {}).get(ptype, ptype)
        cmd = [
            str(part).format(underlying=underlying, type=type_arg, ticker=ticker)
            for part in self.cli["command"]
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=int(self.cli.get("timeout_seconds", 60)),
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"CLI executable not found: {cmd[0]!r}. Configure `source.cli.command` in "
                f"config/derivatives.yaml or use the file/synthetic source."
            ) from None
        if proc.returncode != 0:
            self.warnings.append(
                f"{ticker}/{ptype}: CLI exit {proc.returncode}: {proc.stderr.strip()[:200]}"
            )
            return ""
        return proc.stdout

    def list_products(self, ticker: str, ctx: MarketContext | None = None) -> list[Derivative]:
        out: list[Derivative] = []
        seen: set[str] = set()
        for ptype in PRODUCT_TYPES:
            text = self._run(ticker, ptype)
            for rec in parse_records(text, self.cli.get("output_format", "json")):
                d = map_record(
                    rec,
                    ticker,
                    self.cli["fields"],
                    self.cli.get("type_values") or {},
                    "cli",
                    default_type=ptype,
                )
                if d is not None and d.isin not in seen:
                    seen.add(d.isin)
                    out.append(d)
        return out


class FileSource:
    """Reads exported CLI output (``<TICKER>.json`` / ``.csv``) using the same field mapping."""

    name = "file"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.directory = Path(cfg["file"]["directory"])
        self.fields = cfg["cli"]["fields"]
        self.type_values = cfg["cli"].get("type_values") or {}
        self.warnings: list[str] = []

    def list_products(self, ticker: str, ctx: MarketContext | None = None) -> list[Derivative]:
        out: list[Derivative] = []
        for path in sorted(self.directory.glob(f"{ticker}.*")):
            fmt = "csv" if path.suffix.lower() == ".csv" else "json"
            for rec in parse_records(path.read_text(encoding="utf-8"), fmt):
                d = map_record(rec, ticker, self.fields, self.type_values, f"file:{path.name}")
                if d is not None:
                    out.append(d)
        if not out:
            self.warnings.append(f"{ticker}: no product file under {self.directory}")
        return out


class SyntheticDerivativesSource:
    """Generates a realistic product shelf for an underlying from its market context.

    * Open-end KO longs/shorts at configured barrier distances (barrier == strike for
      classic turbos, or barrier 2 % above strike for mini futures), priced at
      intrinsic value plus a small premium.
    * Warrants on a strike/maturity grid priced with Black-Scholes at the surface
      vol plus an issuer vol markup.
    """

    name = "synthetic"

    def __init__(
        self, cfg: dict, fx_eur_usd: float, as_of: dt.date, issuer_defaults: dict | None = None
    ):
        self.cfg = cfg.get("synthetic", {})
        self.fx = fx_eur_usd
        self.as_of = as_of
        self.issuer_defaults = issuer_defaults or {}
        self.warnings: list[str] = []

    def list_products(self, ticker: str, ctx: MarketContext | None = None) -> list[Derivative]:
        if ctx is None:
            return []
        rng = np.random.default_rng(int(self.cfg.get("seed", 7)) + sum(map(ord, ticker)))
        issuers = self.cfg.get("issuers", ["HSBC"])
        S = ctx.spot
        out: list[Derivative] = []
        ratio = 0.1 if S < 200 else 0.01 if S > 1000 else 0.1
        # knock-outs
        for i, dist in enumerate(self.cfg.get("ko_barrier_distances", [0.05, 0.1, 0.2])):
            for side in ("ko_long", "ko_short"):
                for variant in ("turbo", "mini"):
                    if variant == "mini" and i % 2:
                        continue
                    issuer = issuers[(i + (side == "ko_short")) % len(issuers)]
                    if side == "ko_long":
                        barrier = S * (1 - dist)
                        strike = barrier if variant == "turbo" else barrier * 0.97
                        intrinsic = (S - strike) * ratio / self.fx
                    else:
                        barrier = S * (1 + dist)
                        strike = barrier if variant == "turbo" else barrier * 1.03
                        intrinsic = (strike - S) * ratio / self.fx
                    premium = intrinsic * (0.004 + 0.02 * math.exp(-dist / 0.05)) + 0.01
                    mid = intrinsic + premium
                    spread = max(0.01, mid * (0.004 + rng.uniform(0, 0.006)))
                    fin = float(
                        self.issuer_defaults.get("ko_financing_rate_long", 0.05)
                    ) + rng.uniform(-0.01, 0.01)
                    out.append(
                        Derivative(
                            underlying=ticker,
                            isin=f"DE000SYN{side[3].upper()}{i:02d}{variant[0].upper()}{ticker[:3]}",
                            product_type=side,
                            strike=round(strike, 2),
                            ratio=ratio,
                            bid=round(mid - spread / 2, 3),
                            ask=round(mid + spread / 2, 3),
                            barrier=round(barrier, 2),
                            maturity=None,
                            wkn=f"SY{side[3].upper()}{i:02d}{variant[0].upper()}",
                            issuer=issuer,
                            name=f"{'Turbo' if variant == 'turbo' else 'Mini Future'} {'Long' if side == 'ko_long' else 'Short'} {ticker} KO {barrier:.0f}",
                            leverage=round(S * ratio / self.fx / mid, 2),
                            financing_rate=round(fin, 4),
                            currency="EUR",
                            quote_time=f"{self.as_of}T17:30:00",
                            source="synthetic",
                        )
                    )
        # warrants
        for m_days in self.cfg.get("warrant_maturities_days", [90, 180]):
            maturity = self.as_of + dt.timedelta(days=int(m_days))
            t = m_days / 365.0
            for j, mny in enumerate(self.cfg.get("warrant_moneyness", [0.9, 1.0, 1.1])):
                K = round(S * mny, 1)
                for side in ("call_warrant", "put_warrant"):
                    issuer = issuers[(j + m_days // 45) % len(issuers)]
                    sigma = ctx.vol(t, K) * (1.0 + rng.uniform(0.02, 0.08))  # issuer vol markup
                    fair = warrant_price_from_vol(
                        side, S, K, t, ctx.r, ctx.q, sigma, ratio, self.fx
                    )
                    if fair < 0.02:
                        continue
                    spread = max(0.01, fair * (0.008 + rng.uniform(0, 0.012)))
                    out.append(
                        Derivative(
                            underlying=ticker,
                            isin=f"DE000SYN{'C' if side[0] == 'c' else 'P'}{m_days:03d}{j}{ticker[:3]}",
                            product_type=side,
                            strike=K,
                            ratio=ratio,
                            bid=round(fair - spread / 2, 3),
                            ask=round(fair + spread / 2, 3),
                            barrier=None,
                            maturity=maturity,
                            wkn=f"SY{'C' if side[0] == 'c' else 'P'}{m_days:03d}{j}",
                            issuer=issuer,
                            name=f"{'Call' if side[0] == 'c' else 'Put'} {ticker} {K:g} {maturity:%b %Y}",
                            leverage=round(S * ratio / self.fx / fair, 2),
                            financing_rate=None,
                            currency="EUR",
                            quote_time=f"{self.as_of}T17:30:00",
                            source="synthetic",
                        )
                    )
        return out


def make_source(
    cfg: dict,
    fx_eur_usd: float,
    as_of: dt.date,
    issuer_defaults: dict | None = None,
    kind: str | None = None,
) -> DerivativesSource:
    src = cfg["source"]
    kind = kind or src.get("kind", "synthetic")
    if kind == "cli":
        return ScalableCliSource(src)
    if kind == "file":
        return FileSource(src)
    return SyntheticDerivativesSource(src, fx_eur_usd, as_of, issuer_defaults)
