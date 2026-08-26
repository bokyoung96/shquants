# EMP008 리서치 인수인계 패키지

이 폴더는 원본 `shquants` 저장소와 분리해서 사용할 수 있는 EMP008 실행 패키지다. 데이터 변환부터 factor weight 생성과 백테스트까지 이 폴더 안에서 수행한다.

## 1. 폴더 구성

```text
emp008_research/
├─ run.py                  # 가장 간단한 통합 실행 파일
├─ emp008/                 # factor 계산과 target weight 생성
├─ backtest/               # target weight 백테스트
├─ data/raw/               # 원본 CSV/엑셀
├─ data/parquet/           # 실행에 사용하는 parquet
├─ scripts/                # 단계별 실행 스크립트
├─ experiments/            # factor set catalog와 실험 메모
├─ tests/                  # 검증 테스트
└─ results/                # 실행 결과
```

`.venv/`, `__pycache__/`, `.pytest_cache/`는 복사하지 않아도 된다.

## 2. 처음 한 번만 할 일

PowerShell에서 이 폴더로 이동한 뒤 의존 패키지를 설치한다.

```powershell
cd emp008_research
uv sync
```

Python 3.12 이상과 `uv`가 필요하다.

## 3. 데이터 준비

`data/raw/`에 필요한 CSV와 엑셀 파일을 넣는다. parquet이 이미 최신이면 변환을 건너뛰어도 된다.

raw 데이터를 parquet으로 변환하려면:

```powershell
uv run python scripts/convert_data.py `
  --raw-dir data/raw `
  --parquet-dir data/parquet
```

raw 파일을 수정한 뒤에는 parquet도 다시 변환해야 한다.

## 4. 가장 쉬운 실행 방법

`run.py` 아래쪽의 `SETTINGS`만 수정한다.

```python
SETTINGS = RunSettings(
    start="2019-12-30",
    end="2026-06-30",
    factor_set="production_core",
    sector_neutral_dataset="wi26",
    convert_raw_to_parquet=False,
    run_backtest=True,
    fill_mode="close",
    capital=100_000_000.0,
    fee=0.0002,
    sell_tax=0.0015,
    slippage=0.0005,
    allow_fractional=True,
    run_name="my_emp008_run",
)
```

그다음 실행한다.

```powershell
uv run python run.py
```

실행 순서는 다음과 같다.

```text
parquet 로드 → factor 계산 → target weights 생성 → 백테스트 → 결과 저장
```

결과는 `results/<run_name>/`에 저장된다.

전체 canonical factor set을 동일 조건으로 한 번에 실행하고, 모델별 성과 파일과
PNG 차트를 만들려면:

```powershell
uv run python scripts/run_all_models.py
```

이 명령은 `results/all_models/<factor_set>/` 아래에 가중치, 백테스트,
`plots/cumulative_return.png`, `plots/drawdown.png`, `model_summary.json`을
저장하고, 전체 비교표를 `results/all_models/comparison_summary.csv`에 저장한다.

## 5. 주요 설정

| 설정 | 의미 |
|---|---|
| `start`, `end` | 분석 기간 |
| `factor_set` | 사용할 factor 조합 |
| `sector_neutral_dataset` | `wi26` 또는 `wics` |
| `convert_raw_to_parquet` | raw를 parquet으로 다시 변환할지 여부 |
| `run_backtest` | 백테스트까지 실행할지 여부 |
| `fill_mode` | 체결 가격 방식. 기본은 `close` |
| `capital` | 초기 투자금 |
| `fee` / `sell_tax` / `slippage` | 거래비용 조건 |
| `allow_fractional` | 소수점 주식 허용 여부 |
| `run_name` | 결과 폴더 이름 |

기본 factor weight는 모두 `1.0`이므로 factor set 안의 factor가 equal weight로 적용된다.

## 6. Factor set

자세한 목록은 [experiments/factor_sets.md](experiments/factor_sets.md)에 있다.

| 이름 | 구성 | 용도 |
|---|---|---|
| `origin` | size, momentum-12m, dividend-yield-fy0 | 원본 EMP008 reference |
| `origin_add` | price-high, earnings, dividend, retail, value, size | 6팩터 production baseline |
| `production_core` | size, momentum-12m, earnings, value | 현재 4팩터 equal-weight core |
| `research_*` | 연구용 조합 | factor 연구 |
| `reference_*` | 원본/reference 변형 | 재현 및 비교 |

`mfbt` 같은 예전 이름은 지원하지 않는다. catalog에 적힌 canonical 이름만 사용한다.

## 7. 단계별 실행

가중치만 생성:

```powershell
uv run python scripts/generate_weights.py `
  --parquet-dir data/parquet `
  --output-root results
```

생성된 `target_weights.csv`로 백테스트:

```powershell
uv run python scripts/run_backtest.py `
  --data-dir data/parquet `
  --weights-csv results/production_core/weights/target_weights.csv `
  --output-dir results/production_core/backtest
```

## 8. 결과 파일

```text
results/<run_name>/
├─ weights/
│  ├─ target_weights.csv
│  ├─ target_weights.parquet
│  ├─ active_weights.parquet
│  └─ diagnostics.parquet
└─ backtest/
   ├─ summary.json
   └─ series/
      ├─ equity.csv
      ├─ returns.csv
      ├─ turnover.csv
      └─ weights.parquet
```

`target_weights.csv`는 날짜별 목표 포트폴리오이고, `backtest/series/equity.csv`는 백테스트 자산 곡선이다.

## 9. 검증

```powershell
uv run python -m pytest tests -q
```

원본과의 parity 검증 내용은 [VERIFICATION.md](VERIFICATION.md)에 기록돼 있다.

## 10. 주의사항

- `data/raw/`는 반드시 함께 전달한다.
- `data/parquet/`는 raw에서 재생성할 수 있지만, 바로 실행하려면 함께 전달하는 편이 편하다.
- raw 파일을 바꿨다면 parquet 변환 후 가중치를 다시 생성한다.
- `results/`를 삭제해도 코드와 데이터는 손상되지 않으며 같은 설정으로 다시 만들 수 있다.
- 현재 `run.py`의 기본 실행은 `production_core` equal-weight 모델이다.
