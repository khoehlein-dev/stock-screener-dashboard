"""Data model for leveraged retail derivatives."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field

import pandas as pd

PRODUCT_TYPES = ("ko_long", "ko_short", "call_warrant", "put_warrant")
KO_TYPES = ("ko_long", "ko_short")
WARRANT_TYPES = ("call_warrant", "put_warrant")

PRODUCT_COLUMNS = [
    "underlying",
    "isin",
    "wkn",
    "name",
    "issuer",
    "product_type",
    "strike",
    "barrier",
    "ratio",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "maturity",
    "days_to_maturity",
    "leverage",
    "financing_rate",
    "currency",
    "quote_time",
    "source",
]


@dataclass
class Derivative:
    underlying: str  # screener ticker
    isin: str
    product_type: str  # ko_long | ko_short | call_warrant | put_warrant
    strike: float  # in underlying currency (USD for US stocks)
    ratio: float  # underlying units per certificate (e.g. 0.1)
    bid: float  # EUR
    ask: float  # EUR
    barrier: float | None = None  # knock-out level (KO only), underlying currency
    maturity: dt.date | None = None  # None = open-end
    wkn: str = ""
    name: str = ""
    issuer: str = ""
    leverage: float | None = None
    financing_rate: float | None = None  # p.a., KO open-end financing (long: added to strike)
    currency: str = "EUR"
    quote_time: str | None = None
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else float("nan")

    @property
    def is_ko(self) -> bool:
        return self.product_type in KO_TYPES

    @property
    def is_call_like(self) -> bool:
        return self.product_type in ("ko_long", "call_warrant")

    def days_to_maturity(self, as_of: dt.date) -> int | None:
        return None if self.maturity is None else (self.maturity - as_of).days

    def to_row(self, as_of: dt.date) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        d["mid"] = self.mid
        d["spread_pct"] = self.spread_pct
        d["days_to_maturity"] = self.days_to_maturity(as_of)
        d["maturity"] = self.maturity.isoformat() if self.maturity else None
        return d


def products_frame(products: list[Derivative], as_of: dt.date) -> pd.DataFrame:
    if not products:
        return pd.DataFrame(columns=PRODUCT_COLUMNS)
    df = pd.DataFrame([p.to_row(as_of) for p in products])
    return df[[c for c in PRODUCT_COLUMNS if c in df.columns]]


def products_from_frame(df: pd.DataFrame) -> list[Derivative]:
    out = []
    for r in df.to_dict("records"):
        mat = r.get("maturity")
        out.append(
            Derivative(
                underlying=r["underlying"],
                isin=r["isin"],
                product_type=r["product_type"],
                strike=float(r["strike"]),
                ratio=float(r["ratio"]),
                bid=float(r["bid"]),
                ask=float(r["ask"]),
                barrier=None if pd.isna(r.get("barrier")) else float(r["barrier"]),
                maturity=None
                if (mat is None or (isinstance(mat, float) and pd.isna(mat)) or mat == "")
                else dt.date.fromisoformat(str(mat)[:10]),
                wkn=r.get("wkn") or "",
                name=r.get("name") or "",
                issuer=r.get("issuer") or "",
                leverage=None if pd.isna(r.get("leverage")) else float(r["leverage"]),
                financing_rate=None
                if pd.isna(r.get("financing_rate"))
                else float(r["financing_rate"]),
                currency=r.get("currency") or "EUR",
                quote_time=r.get("quote_time"),
                source=r.get("source") or "",
            )
        )
    return out
