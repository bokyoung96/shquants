from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ValidationHolding:
    ticker: str
    ticker_raw: str
    name: str
    quantity: float
    amount: float
    weight: float


@dataclass(frozen=True, slots=True)
class ValidationSnapshot:
    as_of: str
    equity_holdings: list[ValidationHolding]
    cash: dict[str, object]


@dataclass(frozen=True, slots=True)
class ValidationFixture:
    schema_version: str
    source_type: str
    etf_code: str
    etf_code_raw: str
    etf_name: str
    index_code: str
    source: dict[str, object]
    snapshots: list[ValidationSnapshot]


def load_validation_fixtures(path: Path) -> list[ValidationFixture]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [_fixture_from_mapping(item) for item in payload.get("fixtures", []) if isinstance(item, Mapping)]


def fixture_from_mapping(item: Mapping[str, object]) -> ValidationFixture:
    return _fixture_from_mapping(item)


def _fixture_from_mapping(item: Mapping[str, object]) -> ValidationFixture:
    return ValidationFixture(
        schema_version=str(item.get("schema_version", "")),
        source_type=str(item.get("source_type", "")),
        etf_code=str(item.get("etf_code", "")),
        etf_code_raw=str(item.get("etf_code_raw", "")),
        etf_name=str(item.get("etf_name", "")),
        index_code=str(item.get("index_code", "")),
        source=dict(_mapping(item.get("source"))),
        snapshots=[
            ValidationSnapshot(
                as_of=str(snapshot.get("as_of", "")),
                equity_holdings=[
                    ValidationHolding(
                        ticker=str(holding.get("ticker", "")),
                        ticker_raw=str(holding.get("ticker_raw", "")),
                        name=str(holding.get("name", "")),
                        quantity=float(holding.get("quantity", 0.0) or 0.0),
                        amount=float(holding.get("amount", 0.0) or 0.0),
                        weight=float(holding.get("weight", 0.0) or 0.0),
                    )
                    for holding in snapshot.get("equity_holdings", [])
                    if isinstance(holding, Mapping)
                ],
                cash=dict(_mapping(snapshot.get("cash"))),
            )
            for snapshot in item.get("snapshots", [])
            if isinstance(snapshot, Mapping)
        ],
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
