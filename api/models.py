from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Quote:
    code: str
    time: str
    price: int
    volume: int = 0
    value: int = 0
    size: int = 0
    open: int = 0
    high: int = 0
    low: int = 0
    isin: str = ""


@dataclass(frozen=True)
class Order:
    code: str
    qty: int
    side: Side
    price: int | None = None
    hoga: str = "1"
    account: str = ""
    account_password: str = ""
    product: str = "01"
    market: str = "1"
    condition: str = "0"


@dataclass(frozen=True)
class OrderResult:
    order_no: str
    code: str = ""
    msg_code: str = ""
    msg1: str = ""
    msg2: str = ""
    msg3: str = ""
