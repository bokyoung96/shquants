# MFBT EMP008 멀티팩터 제안서 디자인

## Communication job

팀 내 제안서 독자가 복잡한 수식 없이 EMP008의 초과수익 아이디어, 팩터 구성,
포트폴리오 구성 과정을 이해하고 다음 논의를 시작할 수 있게 한다.

## Scope

- 참조 자료: `backtesting/strategies/emp008/제안서_270890.pdf`의 멀티팩터 모델 관련 내용만 사용한다.
- 구현 근거: `backtesting/strategies/emp008/mfbt_emp008*.py`와 README의 현재 동작을 반영한다.
- 최대 4장, 16:9 PowerPoint.
- 흰 배경과 블루 포인트를 사용하는 기관 제안서 스타일.
- 복잡한 최적화 수식과 성과 수치는 본문에서 생략하고, 의사결정에 필요한 개념과 제약만 설명한다.

## Narrative

1. 단일 신호가 아니라 여러 설명력 있는 신호를 결합해야 하는 이유를 제시한다.
2. EMP008이 실제로 사용하는 여섯 가지 팩터를 역할 중심으로 소개한다.
3. 원천 데이터가 투자비중으로 바뀌는 프로세스를 한 줄의 흐름으로 보여준다.
4. benchmark-relative, sector-neutral, tracking-error 관리라는 포트폴리오 결과와 산출물을 정리한다.

## Slide content

### 1. 멀티팩터 모델은 여러 수익 원천을 하나의 포트폴리오로 결합한다

KOSPI200 개별종목을 팩터별로 평가하고, 설명력이 있는 신호를 결합한 뒤,
사전 설정한 위험 한도 안에서 투자비중으로 변환한다. 참조 PDF의 구조를
EMP008의 benchmark-relative 목적에 맞게 간결한 3단계로 표현한다.

### 2. EMP008은 가격·이익·배당·수급·가치·규모를 함께 본다

여섯 팩터를 복잡한 정의 대신 투자 해석으로 설명한다. 가격 모멘텀, 이익 모멘텀,
배당수익률, 섹터 대비 개인수급, FCF 기반 가치, 로그 시가총액을 포함한다.
팩터 간 상호보완성이 핵심이라는 메시지를 유지한다.

### 3. 원천 데이터는 표준화와 위험 추정을 거쳐 목표비중이 된다

데이터 수집 → 월말 팩터 신호 → 유니버스/float 시가총액 기준 표준화 → 섹터 active
exposure → 과거 수익률로 팩터 위험 추정 → benchmark 대비 active weight 최적화의
6단계 흐름으로 보여준다.

### 4. 결과는 작은 tilting으로 설명 가능한 benchmark-relative 포트폴리오다

KOSPI200을 기준으로 비중을 더하거나 덜고, sector neutral을 지키며, 연 70bp
tracking-error 예산 안에서 최적화한다. 최종 산출물은 target weights, active weights,
diagnostics, backtest/report로 정리한다. 이는 참조 PDF의 Barra 기반 위험관리 메시지를
EMP008의 현재 설정과 연결한다.

## Visual direction

- 16:9, white background, dark navy text, blue accent, restrained gray structure.
- Native PowerPoint shapes and editable text only; no invented performance chart.
- Use one dominant composition per slide and avoid dense dashboard-like cards.
- Keep titles at least 35pt and body text at least 16pt.

## Verification

- Final deck has no more than four slides.
- All visible claims trace to the reference PDF or current EMP008 code/README.
- Render every slide and inspect for clipping, overlap, and unexpected wrapping.
- Run slide overflow checks before delivery.
