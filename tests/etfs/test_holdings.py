import json
from pathlib import Path

from etfs.common.holdings import ValidationFixture, load_validation_fixtures


def test_load_validation_fixtures_hydrates_provider_neutral_holdings_model(tmp_path: Path) -> None:
    path = tmp_path / "validation_fixtures.json"
    path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "schema_version": "1.0",
                        "source_type": "etf_portfolio_component_xlsx",
                        "etf_code": "123456",
                        "etf_code_raw": "A123456",
                        "etf_name": "Example ETF",
                        "index_code": "FI00.EXAMPLE",
                        "source": {"path": "holdings.xlsx"},
                        "snapshots": [
                            {
                                "as_of": "2026-06-16",
                                "equity_holdings": [
                                    {
                                        "ticker": "000001",
                                        "ticker_raw": "A000001",
                                        "name": "Example Security",
                                        "quantity": 10,
                                        "amount": 1000,
                                        "weight": 0.25,
                                    }
                                ],
                                "cash": {"weight": 0.001},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    fixtures = load_validation_fixtures(path)

    assert fixtures == [
        ValidationFixture(
            schema_version="1.0",
            source_type="etf_portfolio_component_xlsx",
            etf_code="123456",
            etf_code_raw="A123456",
            etf_name="Example ETF",
            index_code="FI00.EXAMPLE",
            source={"path": "holdings.xlsx"},
            snapshots=[
                fixtures[0].snapshots[0],
            ],
        )
    ]
    assert fixtures[0].snapshots[0].equity_holdings[0].ticker == "000001"
    assert fixtures[0].snapshots[0].cash == {"weight": 0.001}
