"""Trading-cost model loaded from ``config/costs.yaml`` (nothing hard-coded)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_COSTS_PATH = Path(__file__).resolve().parents[2] / "config" / "costs.yaml"
DEFAULT_DERIVATIVES_PATH = Path(__file__).resolve().parents[2] / "config" / "derivatives.yaml"


def load_yaml(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class CostModel:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str | None = None) -> CostModel:
        return cls(load_yaml(path or DEFAULT_COSTS_PATH))

    # ---- accessors ------------------------------------------------------------------
    @property
    def plan_name(self) -> str:
        return self.raw.get("active_plan", "free_broker")

    @property
    def plan(self) -> dict:
        return self.raw["plans"][self.plan_name]

    @property
    def venue(self) -> str:
        return self.raw.get("venue", "gettex")

    @property
    def venue_fees(self) -> dict:
        return self.plan["venues"][self.venue]

    @property
    def fx_eur_usd(self) -> float:
        return float(self.raw.get("fx", {}).get("eur_usd", 1.0))

    @property
    def exit(self) -> dict:
        return self.raw.get("exit", {})

    @property
    def issuer_defaults(self) -> dict:
        return self.raw.get("issuer_defaults", {})

    @property
    def tax_rate(self) -> float:
        t = self.raw.get("tax", {})
        return float(t.get("rate", 0.0)) if t.get("apply") else 0.0

    # ---- fees -----------------------------------------------------------------------
    def order_fee(self, order_value: float, issuer: str | None = None) -> float:
        """Fee for one executed order of ``order_value`` EUR."""
        if order_value <= 0:
            return 0.0
        partner = self.plan.get("partner_derivatives") or {}
        if (
            issuer
            and partner
            and issuer in partner.get("issuers", [])
            and partner.get("venue", self.venue) == self.venue
            and order_value >= float(partner.get("min_order_value", 0))
        ):
            return float(partner.get("order_fee_fixed", 0.0))
        v = self.venue_fees
        free_above = v.get("free_above_order_value")
        if free_above is not None:
            if order_value >= float(free_above):
                return 0.0
            return float(v.get("small_order_fee", v.get("order_fee_fixed", 0.0)))
        fee = (
            float(v.get("order_fee_fixed", 0.0)) + float(v.get("order_fee_pct", 0.0)) * order_value
        )
        return max(fee, float(v.get("order_fee_min", 0.0)))

    def round_trip_fee(
        self, buy_value: float, sell_value: float, issuer: str | None = None
    ) -> float:
        return self.order_fee(buy_value, issuer) + self.order_fee(sell_value, issuer)

    def describe(self) -> str:
        v = self.venue_fees
        return (
            f"{self.raw.get('meta', {}).get('broker', 'broker')} / plan '{self.plan_name}' / {self.venue}: "
            f"fixed {v.get('order_fee_fixed', 0)} EUR + {100 * float(v.get('order_fee_pct', 0)):.3f} %"
            + (
                f", free ≥ {v['free_above_order_value']} EUR"
                if v.get("free_above_order_value")
                else ""
            )
            + (" (values not verified)" if not self.raw.get("meta", {}).get("verified") else "")
        )
