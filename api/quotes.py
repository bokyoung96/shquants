from __future__ import annotations

import argparse
import time

from api import Quote, make


def format_quote(quote: Quote) -> str:
    return f"{quote.code},{quote.time},{quote.price},{quote.volume}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Subscribe to Shinhan iIndi stock ticks.")
    parser.add_argument("codes", nargs="+")
    parser.add_argument("--config", default="api/config.json")
    args = parser.parse_args()

    client = make(args.config)
    client.connect()
    for code in args.codes:
        client.subscribe_quote(code, lambda quote: print(format_quote(quote), flush=True))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        client.unsubscribe_quote()


if __name__ == "__main__":
    main()
