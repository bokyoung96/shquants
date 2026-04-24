from __future__ import annotations

import json
from dataclasses import asdict

from backtesting.run import BacktestRunner


def main() -> None:
    report = BacktestRunner().run_resolved_cli(preset_id="kospi200_semiannual_floatcap")
    payload = {
        "config": asdict(report.config),
        "summary": report.summary,
        "output_dir": None if report.output_dir is None else str(report.output_dir),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
