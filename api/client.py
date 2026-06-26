from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from api.config import IndiConfig
from api.control import Control
from api.models import Order, OrderResult, Quote


QuoteHandler = Callable[[Quote], None]
OrderHandler = Callable[[OrderResult], None]


@dataclass
class Request:
    name: str
    kind: str
    code: str = ""


class Indi:
    def __init__(self, tr: object, real: object, config: IndiConfig) -> None:
        self.tr = tr if isinstance(tr, Control) else Control(tr)
        self.real = real if isinstance(real, Control) else Control(real)
        self.config = config
        self.requests: dict[int, Request] = {}
        self.quotes: dict[str, list[QuoteHandler]] = {}
        self.orders: dict[int, OrderHandler] = {}

        self.tr.on("ReceiveData", self.on_data)
        self.tr.on("ReceiveSysMsg", self.on_sys)
        self.real.on("ReceiveRTData", self.on_rt)

    def connect(self) -> bool:
        self.tr.start()
        self.real.start()
        if self.tr.state() == 0:
            return True

        if not self.config.user or not self.config.password:
            raise RuntimeError("iIndi is not connected and config has no login credentials")

        return self.tr.login(
            self.config.user,
            self.config.password,
            self.config.cert_password,
            self.config.starter,
        )

    def subscribe_quote(self, code: str, handler: QuoteHandler) -> int:
        code = clean_code(code)
        self.quotes.setdefault(code, []).append(handler)

        self.tr.query("SC")
        self.tr.set(0, code)
        rqid = self.tr.request_rt()
        self.requests[rqid] = Request(name="SC", kind="quote", code=code)
        return rqid

    def unsubscribe_quote(self, code: str = "") -> None:
        code = clean_code(code) if code else ""
        self.real.stop_rt("SC", code)
        self._drop_quote_requests(code)
        if code:
            self.quotes.pop(code, None)
        else:
            self.quotes.clear()

    def buy(self, code: str, qty: int, price: int | None = None, hoga: str = "1") -> int:
        return self.order(Order(code=code, qty=qty, price=price, side="buy", hoga=hoga))

    def sell(self, code: str, qty: int, price: int | None = None, hoga: str = "1") -> int:
        return self.order(Order(code=code, qty=qty, price=price, side="sell", hoga=hoga))

    def order(self, order: Order, handler: OrderHandler | None = None) -> int:
        validate_order(order)
        account = order.account or self.config.account
        password = order.account_password or self.config.account_password
        if not account or not password:
            raise ValueError("order requires account and account_password")

        code = clean_code(order.code)
        side = "2" if order.side == "buy" else "1"

        self.tr.query("SABA101U1")
        self.tr.set(0, account)
        self.tr.set(1, order.product)
        self.tr.set(2, password)
        self.tr.set(5, "0")
        self.tr.set(6, "00")
        self.tr.set(7, side)
        self.tr.set(8, "A" + code)
        self.tr.set(9, order.qty)
        if order.price is not None:
            self.tr.set(10, order.price)
        self.tr.set(11, order.market)
        self.tr.set(12, order.hoga)
        self.tr.set(13, order.condition)
        self.tr.set(14, "0")
        self.tr.set(21, "Y")

        rqid = self.tr.request()
        self.requests[rqid] = Request(name="SABA101U1", kind="order", code=code)
        if handler is not None:
            self.orders[rqid] = handler
        return rqid

    def on_data(self, *args: object) -> None:
        ctrl, rqid = split_args(self.tr, args)
        request = self.requests.pop(rqid, None)
        if request is None:
            return

        if request.kind == "quote":
            quote = read_quote(ctrl)
            for handler in self.quotes.get(request.code, []):
                handler(quote)
            if request.code in self.quotes:
                self.real.request_rt("SC", request.code)
            return

        if request.kind == "order":
            result = read_order(ctrl, request.code)
            handler = self.orders.pop(rqid, None)
            if handler is not None:
                handler(result)

    def on_rt(self, *args: object) -> None:
        ctrl, real_type = split_args(self.real, args)
        if str(real_type) != "SC":
            return
        quote = read_quote(ctrl)
        for handler in self.quotes.get(quote.code, []):
            handler(quote)

    def on_sys(self, *_args: object) -> None:
        return

    def _drop_quote_requests(self, code: str = "") -> None:
        self.requests = {
            rqid: request
            for rqid, request in self.requests.items()
            if request.kind != "quote" or (code and request.code != code)
        }


def split_args(default_ctrl: Control, args: tuple[object, ...]) -> tuple[Control, int | str]:
    if len(args) == 2:
        ctrl = args[0] if isinstance(args[0], Control) else Control(args[0])
        return ctrl, args[1]  # type: ignore[return-value]
    if len(args) == 1:
        return default_ctrl, args[0]  # type: ignore[return-value]
    raise TypeError(f"unexpected callback args: {args!r}")


def clean_code(code: str) -> str:
    text = str(code).strip().upper()
    return text[1:] if text.startswith("A") else text


def validate_order(order: Order) -> None:
    if not clean_code(order.code):
        raise ValueError("order code is required")
    if order.qty <= 0:
        raise ValueError("order qty must be positive")
    if order.price is not None and order.price < 0:
        raise ValueError("order price must be non-negative")
    if order.hoga not in {"0", "1", "X", "Y"}:
        raise ValueError("order hoga must be one of 0, 1, X, Y")
    if order.market not in {"1"}:
        raise ValueError("order market must be one of 1")
    if order.condition not in {"0", "3", "4"}:
        raise ValueError("order condition must be one of 0, 3, 4")


def read_quote(ctrl: Control) -> Quote:
    return Quote(
        isin=ctrl.get(0),
        code=clean_code(ctrl.get(1)),
        time=ctrl.get(2),
        price=to_int(ctrl.get(3)),
        volume=to_int(ctrl.get(7)),
        value=to_int(ctrl.get(8)),
        size=to_int(ctrl.get(9)),
        open=to_int(ctrl.get(10)),
        high=to_int(ctrl.get(11)),
        low=to_int(ctrl.get(12)),
    )


def read_order(ctrl: Control, code: str) -> OrderResult:
    return OrderResult(
        order_no=ctrl.get(0),
        code=code,
        msg_code=ctrl.get(2),
        msg1=ctrl.get(3),
        msg2=ctrl.get(4),
        msg3=ctrl.get(5),
    )


def to_int(value: str) -> int:
    text = str(value).strip().replace(",", "")
    if not text:
        return 0
    return int(float(text))
