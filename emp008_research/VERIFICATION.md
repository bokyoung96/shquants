# 검증 기록

이 패키지는 `emp008_research` 폴더 내부의 코드와 `data/parquet`만 사용해
가중치 생성과 백테스트를 수행할 수 있다.

## 자동 테스트

```text
14 tests passed
```

검증 명령:

```powershell
uv run python -m pytest tests -q
```

## 원본 parity 검증

원본 `backtesting/strategies/emp008`과 동일한 데이터, 기간, factor set,
거래비용 조건으로 비교했다. 현재 `production_core`와 연구용 4팩터 실행은
원본과 동일한 target weights 및 백테스트 결과를 생성한다.

대표 검증 조건:

- 기간: 2019-12-30 ~ 2026-06-30
- 섹터 중립화: WI26
- 위험모형: factor-idio
- expected alpha: 최근 36개월 산술평균
- 체결: close
- 수수료: 0.0002
- 매도세: 0.0015
- 슬리피지: 0.0005
- fractional: 허용

검증 결과는 `results/<run_name>/`에 생성되며, 코드를 수정하면 같은 조건으로
다시 실행해 재검증할 수 있다.
