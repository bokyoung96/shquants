import json
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook

from etfs.fnguide.validation import (
    build_target_weight_validation_results,
    build_validation_results,
    parse_validation_workbook,
    write_target_weight_validation_results,
    write_validation_fixtures,
    write_validation_results,
)


def test_parse_validation_workbook_normalizes_equity_and_cash_rows(tmp_path: Path) -> None:
    workbook_path = _write_validation_workbook(tmp_path)

    fixture = parse_validation_workbook(workbook_path, index_code_by_etf={"0167A0": "FI00.WLT.KSS"})

    assert fixture.etf_code == "0167A0"
    assert fixture.source["source_type"] == "etf_portfolio_component_xlsx"
    assert len(fixture.snapshots) == 1
    snapshot = fixture.snapshots[0]
    assert snapshot.as_of == "2026-05-29"
    assert [holding.ticker for holding in snapshot.equity_holdings] == ["005930", "000660"]
    assert snapshot.equity_holdings[0].weight == 0.25
    assert snapshot.cash["weight"] == 0.002


def test_parse_validation_workbook_accepts_normal_korean_header(tmp_path: Path) -> None:
    workbook_path = _write_korean_validation_workbook(tmp_path)

    fixture = parse_validation_workbook(workbook_path, index_code_by_etf={"0167A0": "FI00.WLT.NMS"})

    assert fixture.etf_code == "0167A0"
    assert fixture.etf_name == "SOL AI반도체TOP2플러스"
    assert fixture.index_code == "FI00.WLT.NMS"
    assert [snapshot.as_of for snapshot in fixture.snapshots] == ["2026-06-15", "2026-06-16"]
    latest = fixture.snapshots[-1]
    assert [holding.ticker for holding in latest.equity_holdings] == ["009150", "000660"]
    assert latest.equity_holdings[0].name == "삼성전기"
    assert latest.equity_holdings[0].weight == 0.2575
    assert latest.cash == {"name": "원화현금", "quantity": 10.0, "amount": 10.0, "weight": 0.0013}


def test_write_validation_fixtures_outputs_json_payload(tmp_path: Path) -> None:
    workbook_path = _write_validation_workbook(tmp_path)

    output_path = write_validation_fixtures(
        [workbook_path],
        tmp_path,
        index_code_by_etf={"0167A0": "FI00.WLT.KSS"},
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.name == "validation_fixtures.json"
    assert payload["fixtures"][0]["snapshots"][0]["equity_holdings"][0]["ticker"] == "005930"


def test_build_validation_results_checks_constituent_count_against_specs(tmp_path: Path) -> None:
    workbook_path = _write_validation_workbook(tmp_path)
    fixture = parse_validation_workbook(workbook_path, index_code_by_etf={"0167A0": "FI00.WLT.KSS"})
    specs = [
        {
            "index_code": "FI00.WLT.KSS",
            "selection": {"total_constituents": 2},
        }
    ]

    results = build_validation_results([fixture], specs)

    assert results[0].validation_type == "etf_holdings_constituents"
    assert results[0].status == "passed"
    assert results[0].checks["cash_excluded"] == "passed"
    assert results[0].metrics["official_equity_count"] == 2


def test_write_validation_results_uses_fixture_and_specs_json(tmp_path: Path) -> None:
    workbook_path = _write_validation_workbook(tmp_path)
    fixtures_path = write_validation_fixtures(
        [workbook_path],
        tmp_path,
        index_code_by_etf={"0167A0": "FI00.WLT.KSS"},
    )
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps({"indices": [{"index_code": "FI00.WLT.KSS", "selection": {"total_constituents": 2}}]}),
        encoding="utf-8",
    )

    results_path = write_validation_results(fixtures_path, specs_path, tmp_path)

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert results_path.name == "validation_results.json"
    assert payload["results"][0]["status"] == "passed"


def test_build_target_weight_validation_results_compares_target_weights_to_holdings(tmp_path: Path) -> None:
    workbook_path = _write_validation_workbook(tmp_path)
    fixture = parse_validation_workbook(workbook_path, index_code_by_etf={"0167A0": "FI00.WLT.KSS"})
    target_payload = {
        "results": [
            {
                "index_code": "FI00.WLT.KSS",
                "as_of": "2026-05-29",
                "target_weights": [
                    {"security_code": "A005930", "target_weight": 0.25},
                    {"security_code": "A000660", "target_weight": 0.748},
                ],
            }
        ]
    }

    results = build_target_weight_validation_results(target_payload, [fixture], weight_tolerance=0.0)

    assert len(results) == 1
    assert results[0].validation_type == "target_weights_vs_etf_holdings"
    assert results[0].status == "passed"
    assert results[0].checks == {"constituent_membership": "passed", "weight_tolerance": "passed"}
    assert results[0].metrics["common_constituent_count"] == 2
    assert results[0].metrics["max_abs_weight_difference"] == 0.0


def test_build_target_weight_validation_results_reports_membership_and_weight_differences(tmp_path: Path) -> None:
    workbook_path = _write_validation_workbook(tmp_path)
    fixture = parse_validation_workbook(workbook_path, index_code_by_etf={"0167A0": "FI00.WLT.KSS"})
    target_payload = {
        "results": [
            {
                "index_code": "FI00.WLT.KSS",
                "as_of": "2026-05-29",
                "target_weights": [
                    {"security_code": "A005930", "target_weight": 0.20},
                    {"security_code": "A123456", "target_weight": 0.80},
                ],
            }
        ]
    }

    results = build_target_weight_validation_results(target_payload, [fixture], weight_tolerance=0.01)

    assert results[0].status == "failed"
    assert results[0].checks == {"constituent_membership": "failed", "weight_tolerance": "failed"}
    assert results[0].metrics["max_abs_weight_difference"] == 0.05
    assert results[0].differences == [
        {"type": "missing_in_official_holdings", "security_code": "123456", "target_weight": 0.8},
        {"type": "extra_in_official_holdings", "security_code": "000660", "official_weight": 0.748},
        {
            "type": "weight_difference",
            "security_code": "005930",
            "target_weight": 0.2,
            "official_weight": 0.25,
            "difference": -0.05,
        },
    ]


def test_write_target_weight_validation_results_uses_fixture_and_target_weight_json(tmp_path: Path) -> None:
    workbook_path = _write_validation_workbook(tmp_path)
    fixtures_path = write_validation_fixtures(
        [workbook_path],
        tmp_path,
        index_code_by_etf={"0167A0": "FI00.WLT.KSS"},
    )
    target_weights_path = tmp_path / "target_weights.json"
    target_weights_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "index_code": "FI00.WLT.KSS",
                        "as_of": "2026-05-29",
                        "target_weights": [
                            {"security_code": "A005930", "target_weight": 0.25},
                            {"security_code": "A000660", "target_weight": 0.748},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = write_target_weight_validation_results(
        fixtures_path,
        target_weights_path,
        tmp_path,
        weight_tolerance=0.0,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.name == "target_weight_validation.json"
    assert payload["results"][0]["status"] == "passed"


def _write_validation_workbook(tmp_path: Path) -> Path:
    path = tmp_path / "validation_A0167A0.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append([None, "Last Refresh: 2026-06-15 15:21:49"])
    ws.append(["ETF Portfolio Component"])
    ws.append(["ETF", "A0167A0"])
    ws.append(["Frequency2", "Monthly"])
    ws.append(["Term", 20260101, "Current(20260612)"])
    ws.append(["Date", "ETF코드", "ETF명", "구성종목코드", "구성종목", "주식수(계약수)", "금액", "금액기준 구성비중(%)"])
    ws.append([datetime(2026, 5, 29), "A0167A0", "SOL AI반도체TOP2플러스", "A005930", "삼성전자", 10, 1000, 25.0])
    ws.append([datetime(2026, 5, 29), "A0167A0", "SOL AI반도체TOP2플러스", "A000660", "SK하이닉스", 10, 1000, 74.8])
    ws.append([datetime(2026, 5, 29), "A0167A0", "SOL AI반도체TOP2플러스", None, "원화현금", 10, 2, 0.2])
    wb.save(path)
    return path


def _write_korean_validation_workbook(tmp_path: Path) -> Path:
    path = tmp_path / "pdf_A0167A0.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Refresh", "Last Updated: 2026-06-17 10:17:54"])
    ws.append(["ETF 구성종목"])
    ws.append(["ETF", "A0167A0", "SOL AI반도체TOP2플러스"])
    ws.append(["출력주기", "일간", "오름차순"])
    ws.append(["조회기간", 20260317, "최근일자(20260616)"])
    ws.append(["날짜", "ETF코드", "ETF명", "구성종목코드", "구성종목", "주식수(계약수)", "금액", "금액기준 구성비중(%)"])
    ws.append([datetime(2026, 6, 15), "A0167A0", "SOL AI반도체TOP2플러스", "A009150", "삼성전기", 161, 332177000, 25.95])
    ws.append([datetime(2026, 6, 15), "A0167A0", "SOL AI반도체TOP2플러스", None, "원화현금", 10, 10, 0.10])
    ws.append([datetime(2026, 6, 16), "A0167A0", "SOL AI반도체TOP2플러스", "A009150", "삼성전기", 161, 329728000, 25.75])
    ws.append([datetime(2026, 6, 16), "A0167A0", "SOL AI반도체TOP2플러스", "A000660", "SK하이닉스", 133, 316806000, 24.74])
    ws.append([datetime(2026, 6, 16), "A0167A0", "SOL AI반도체TOP2플러스", None, "원화현금", 10, 10, 0.13])
    wb.save(path)
    return path
