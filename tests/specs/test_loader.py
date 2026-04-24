import json
from pathlib import Path

import pytest

from backtesting.specs import load_execution_spec


def test_load_execution_spec_from_json(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "strategy": "momentum",
                "schedule": {"kind": "named", "name": "monthly"},
                "weight_source": {"kind": "strategy"},
            }
        ),
        encoding="utf-8",
    )

    spec = load_execution_spec(path)

    assert spec.start == "2024-01-01"
    assert spec.schedule.name == "monthly"
    assert spec.weight_source.kind == "strategy"


def test_load_execution_spec_rejects_yaml_without_parser_support(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.yaml"
    path.write_text("start: 2024-01-01\nend: 2024-12-31\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML spec loading is not available without an approved YAML dependency"):
        load_execution_spec(path)


def test_load_execution_spec_rejects_non_boolean_flags(tmp_path: Path) -> None:
    path = tmp_path / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end": "2024-12-31",
                "use_k200": "false",
                "allow_fractional": "false",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="use_k200 must be a boolean"):
        load_execution_spec(path)
