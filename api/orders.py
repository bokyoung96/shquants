from __future__ import annotations

import argparse
import time

from api import Order, OrderResult, make


def build_order(args: argparse.Namespace) -> Order:
    return Order(
        side=args.side,
        code=args.code,
        qty=args.qty,
        price=args.price,
        hoga=args.hoga,
        account=args.account,
        account_password=args.account_password,
    )


def format_result(result: OrderResult) -> str:
    return ",".join(
        [result.code, result.order_no, result.msg_code, result.msg1, result.msg2, result.msg3]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a Shinhan iIndi stock order.")
    parser.add_argument("--side", choices=["buy", "sell"], required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--qty", type=int, required=True)
    parser.add_argument("--price", type=int)
    parser.add_argument("--hoga", default="1")
    parser.add_argument("--account", default="")
    parser.add_argument("--account-password", default="")
    parser.add_argument("--config", default="api/config.json")
    parser.add_argument("--send", action="store_true", help="Actually send the order.")
    parser.add_argument("--wait", type=float, default=3.0)
    args = parser.parse_args()

    order = build_order(args)
    if not args.send:
        print(
            "dry-run "
            f"side={order.side} code={order.code} qty={order.qty} "
            f"price={order.price} hoga={order.hoga}"
        )
        return

    client = make(args.config)
    client.connect()
    rqid = client.order(order, lambda result: print(format_result(result), flush=True))
    print(f"sent rqid={rqid}")
    time.sleep(args.wait)


if __name__ == "__main__":
    main()
