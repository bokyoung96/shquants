from __future__ import annotations

from argparse import Namespace

from api.models import Quote
from api.orders import build_order
from api.quotes import format_quote


def test_format_quote_outputs_plain_csv_line():
    quote = Quote(code="005930", time="093001", price=71000, volume=1200)

    assert format_quote(quote) == "005930,093001,71000,1200"


def test_build_order_uses_short_clear_names():
    args = Namespace(
        side="buy",
        code="A005930",
        qty=3,
        price=71000,
        hoga="0",
        account="acc",
        account_password="pwd",
    )

    order = build_order(args)

    assert order.side == "buy"
    assert order.code == "A005930"
    assert order.qty == 3
    assert order.price == 71000
    assert order.hoga == "0"
    assert order.account == "acc"
    assert order.account_password == "pwd"
