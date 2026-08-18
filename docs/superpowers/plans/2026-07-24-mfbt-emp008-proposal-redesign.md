# MFBT EMP008 Proposal Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the EMP008 proposal as a verified six-slide PowerPoint that preserves Shinhan brand chrome, explains the proposed factor methodology accurately, and grounds the Samsung Electronics/SK hynix relative-positioning explanation in saved optimizer evidence.

**Architecture:** A single plain JavaScript ES module under an external scratch workspace will create the deck with `@oai/artifact-tool`. Shared helpers own brand chrome, typography, image embedding, and slide export; each slide function owns one narrative claim. Four generated raster visuals are embedded once each, while titles, factor rules, evidence, footers, and page numbers remain editable PowerPoint objects.

**Tech Stack:** Node.js ES modules, `@oai/artifact-tool`, PowerPoint `.pptx`, presentation rendering and QA scripts from the installed Presentations skill.

---

### Task 1: Initialize the isolated presentation workspace

**Files:**
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/019f91cc-8881-7553-9922-3830b3f18945/mfbt-emp008-redesign/tmp/source-notes.txt`
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/019f91cc-8881-7553-9922-3830b3f18945/mfbt-emp008-redesign/tmp/qa/qa-ledger.txt`
- Read: `backtesting/strategies/emp008/assets/design-v2/*.png`
- Read: `backtesting/strategies/emp008/제안서_270890.pdf`

- [ ] **Step 1: Create the scratch directory tree**

Run:

```powershell
$root = 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\019f91cc-8881-7553-9922-3830b3f18945\mfbt-emp008-redesign\tmp'
New-Item -ItemType Directory -Force -Path $root, "$root\slides", "$root\preview", "$root\layout", "$root\assets", "$root\qa" | Out-Null
```

Expected: all six directories exist outside the repository.

- [ ] **Step 2: Initialize artifact-tool resolution**

Run:

```powershell
node 'C:\Users\CHECK\.codex\plugins\cache\openai-primary-runtime\presentations\26.630.12135\skills\presentations\container_tools\setup_artifact_tool_workspace.mjs' --workspace 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\019f91cc-8881-7553-9922-3830b3f18945\mfbt-emp008-redesign\tmp'
```

Expected: the scratch workspace contains a package manifest and can resolve `@oai/artifact-tool`.

- [ ] **Step 3: Validate the four visual assets and evidence inputs**

Run:

```powershell
Get-ChildItem 'backtesting\strategies\emp008\assets\design-v2\*.png' | Select-Object Name,Length
Test-Path 'results\emp008_runs\mfbt_emp008_wics_neutral\weights\active_weights.parquet'
Test-Path 'results\emp008_runs\mfbt_emp008_wics_neutral\factor_attribution\factor_attribution.xlsx'
```

Expected: four non-empty PNGs and two `True` values.

### Task 2: Author the six-slide artifact-tool builder

**Files:**
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/019f91cc-8881-7553-9922-3830b3f18945/mfbt-emp008-redesign/tmp/build_emp008_deck.mjs`
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/019f91cc-8881-7553-9922-3830b3f18945/mfbt-emp008-redesign/tmp/source-notes.txt`

- [ ] **Step 1: Define deck constants and reusable brand helpers**

The builder must use this geometry and palette:

```js
const W = 1280;
const H = 886;
const C = {
  cyan: '#00A7E1', blue: '#1769E0', navy: '#173B67', ink: '#20252B',
  muted: '#68717D', line: '#D9DEE5', pale: '#F4F7FA', white: '#FFFFFF'
};
```

Implement `addText`, `addRule`, `addImage`, `addChrome`, `addTitle`, and `writeBlob`. `addChrome` must preserve the source-deck header `운용전략`, a thin rule at y=86, a footer rule at y=818, the cyan circular brand marker, `신한자산운용`, a short internal-use note, and the page number.

- [ ] **Step 2: Implement slide 1 and slide 2**

Use the exact narrative titles:

```js
const titles = [
  '여섯 개 투자 질문이 하나의 액티브 비중으로 수렴합니다',
  '서로 다른 실패 가능성을 보완하도록 여섯 팩터를 결합합니다'
];
```

Slide 1 embeds `01-signal-to-active-weight.png` and labels the six questions: 상승 추세, 이익 개선, 배당의 질, 업종 수급, 저평가, 규모 효과. Slide 2 embeds `02-factor-architecture.png` and groups the factors as 가격·구조, 펀더멘털, 수급. No repeated rounded-card grid is allowed.

- [ ] **Step 3: Implement factor-method slides 3 and 4**

Use three horizontal editorial rows per slide. Each row must contain an editable factor name, a compact `제안 산정 방식`, and an editable `채택 사유`.

Slide 3 content:

```js
const methodA = [
  ['배당수익률', '배당수익률 Q1~Q5 점수 + 3년 연속 배당액 증액 프리미엄', '주가 하락 때문에 높아진 단순 배당수익률을 좋은 신호로 오인하지 않습니다.'],
  ['시가총액', '기존 점수 유지 · 지수 비중 10% 이상 종목은 Neutral 고정', '한국 시장의 대형주 쏠림이 규모 팩터를 지배하지 않도록 합니다.'],
  ['주가모멘텀', '현재가 ÷ 52주 신고가 > 80%이면 1, 아니면 0', '분위보다 신고가 접근 여부가 강한 상승 종목을 더 직접적으로 식별합니다.']
];
```

Slide 4 content:

```js
const methodB = [
  ['이익모멘텀', '영업이익 12MF의 1개월 상향률을 분위 점수화', '분모는 절대값을 사용하고, 영업이익 1,000억원 미만이면서 성장률 50% 초과인 종목은 배제합니다.'],
  ['밸류', 'FCF ÷ TEV 분위 · TEV=시가총액+이자발생부채-당좌자산', '주주 귀속 현금흐름이 크고 기업가치가 낮은 종목을 포착합니다.'],
  ['수급', '252일 누적 개인수급 - 업종 평균을 계산한 뒤 부호 반전', '개별 종목 노이즈보다 지속성이 높은 업종 단위 자금 흐름을 사용합니다.']
];
```

Both slides must visibly label the formulas as `제안 산정 방식` so they are not confused with every detail of the current code implementation.

- [ ] **Step 4: Implement optimization slide 5 and relative-positioning slide 6**

Slide 5 embeds `03-risk-optimization.png` and shows the editable flow `팩터 익스포저 → 36개월 기대 알파 → 위험 공분산 → 목표 비중`. Its four constraints are active-weight sum 0, sector-active exposure 0, annualized tracking error 70bp, and final weight at least 0.

Slide 6 embeds `04-semiconductor-active-tilt.png` and must contain these evidence statements:

```js
const pairEvidence = [
  '43개월 중 반대 방향 31개월',
  '닉스 롱·삼전 언더 16회 | 삼전 롱·닉스 언더 15회',
  '2026.06.01 액티브 비중: 삼성전자 -0.046% | SK하이닉스 +0.066%'
];
```

The interpretation must say that the pattern is a benchmark-relative position created by sector neutrality, covariance, and the risk budget—not an absolute short or a permanent single-name view.

### Task 3: Build and inspect the first complete deck

**Files:**
- Create: `backtesting/strategies/emp008/outputs/mfbt_emp008_multifactor_proposal.pptx`
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/019f91cc-8881-7553-9922-3830b3f18945/mfbt-emp008-redesign/tmp/preview/slide-01.png` through `slide-06.png`
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/019f91cc-8881-7553-9922-3830b3f18945/mfbt-emp008-redesign/tmp/preview/deck-montage.webp`

- [ ] **Step 1: Run the builder**

Run:

```powershell
node 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\019f91cc-8881-7553-9922-3830b3f18945\mfbt-emp008-redesign\tmp\build_emp008_deck.mjs'
```

Expected: six PNG previews, six layout JSON files, one montage, and the final PPTX.

- [ ] **Step 2: Run structural checks**

Run:

```powershell
.\.venv\Scripts\python.exe 'C:\Users\CHECK\.codex\plugins\cache\openai-primary-runtime\presentations\26.630.12135\skills\presentations\container_tools\slides_test.py' 'backtesting\strategies\emp008\outputs\mfbt_emp008_multifactor_proposal.pptx'
```

Expected: no slide-canvas overflow errors.

- [ ] **Step 3: Inspect the montage and all six full-size slides**

Use `view_image` on the montage, then inspect every slide PNG at full size. Record clipping, unintended overlap, weak contrast, misleading crop, title wrapping, and footer inconsistency in `tmp/qa/qa-ledger.txt`.

### Task 4: Revise, re-render, and complete the delivery audit

**Files:**
- Modify: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/019f91cc-8881-7553-9922-3830b3f18945/mfbt-emp008-redesign/tmp/build_emp008_deck.mjs`
- Modify: `backtesting/strategies/emp008/outputs/mfbt_emp008_multifactor_proposal.pptx`
- Modify: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/019f91cc-8881-7553-9922-3830b3f18945/mfbt-emp008-redesign/tmp/qa/qa-ledger.txt`

- [ ] **Step 1: Fix every recorded visual defect**

Adjust layout or shorten copy; do not resolve density by shrinking body text below 14pt. Keep all one-line titles on one line and preserve equal left/right outer margins.

- [ ] **Step 2: Rebuild and rerun structural checks**

Run the builder and `slides_test.py` again. Expected: all six slides render and structural checks pass.

- [ ] **Step 3: Validate content evidence**

Run:

```powershell
@'
import pandas as pd
a = pd.read_parquet('results/emp008_runs/mfbt_emp008_wics_neutral/weights/active_weights.parquet')
latest = a.iloc[-1][['A005930','A000660']]
opposite = (a['A005930'] * a['A000660'] < 0).sum()
print(a.index[-1].date(), latest.to_dict(), int(opposite))
'@ | .\.venv\Scripts\python.exe -
```

Expected: `2026-06-01`, values approximately `-0.0004604` and `0.0006553`, and `31` opposite-sign months.

- [ ] **Step 4: Confirm the final artifact**

Run:

```powershell
Get-Item 'backtesting\strategies\emp008\outputs\mfbt_emp008_multifactor_proposal.pptx' | Select-Object FullName,Length,LastWriteTime
```

Expected: a non-empty, newly written PPTX ready for delivery.
