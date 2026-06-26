from __future__ import annotations

import argparse

from api import make


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect to Shinhan iIndi.")
    parser.add_argument("--config", default="api/config.json")
    args = parser.parse_args()

    client = make(args.config)
    ok = client.connect()
    print("connected" if ok else "failed")


if __name__ == "__main__":
    main()
