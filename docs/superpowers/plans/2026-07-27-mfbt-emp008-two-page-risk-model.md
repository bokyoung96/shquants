# MFBT EMP008 Two-Page Risk Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the supplied Shinhan PowerPoint template while modernizing slide 1 and replacing the blank slide 2 with a clear Barra-style expected-return, covariance, and portfolio-optimization flow.

**Architecture:** Use the supplied PPTX as the only visual source. Inspect and duplicate source slides with the template-following scripts, then import the starter deck with `@oai/artifact-tool` and edit inherited objects by source element ID. Preserve source slides 3–12 unchanged, render the full deck, and run overflow, placeholder, and template-fidelity checks before delivery.

**Tech Stack:** Node.js ES modules, `@oai/artifact-tool`, bundled presentation template-following scripts, PowerPoint `.pptx`

---

## File structure

- Source template: `backtesting/strategies/emp008/제안서_멀티팩터모델.pptx`
- Design specification: `docs/superpowers/specs/2026-07-27-mfbt-emp008-two-page-risk-model-design.md`
- Implementation plan: `docs/superpowers/plans/2026-07-27-mfbt-emp008-two-page-risk-model.md`
- External scratch workspace: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model`
- Scratch editor module: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/edit-emp008-deck.mjs`
- Final presentation: `backtesting/strategies/emp008/outputs/제안서_멀티팩터모델_개편본.pptx`

### Task 1: Initialize the artifact-tool workspace and refresh the template inventory

**Files:**
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/source-template.pptx`
- Refresh: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/template-inspect/`

- [ ] **Step 1: Copy the Unicode-named source deck to an ASCII scratch path**

```powershell
$source = Resolve-Path -LiteralPath 'C:\Users\CHECK\Documents\GitHub\shquants\backtesting\strategies\emp008\제안서_멀티팩터모델.pptx'
$scratch = 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\source-template.pptx'
Copy-Item -LiteralPath $source.Path -Destination $scratch -Force
```

- [ ] **Step 2: Initialize the bundled artifact-tool package from the user runtime directory**

Run from `C:\Users\CHECK` so the helper resolves `C:\Users\CHECK\.cache\codex-runtimes\codex-primary-runtime`.

```powershell
node 'C:\Users\CHECK\.codex\plugins\cache\openai-primary-runtime\presentations\26.630.12135\skills\presentations\container_tools\setup_artifact_tool_workspace.mjs' --workspace 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp'
```

Expected: the command prints the scratch `tmp` directory and exits successfully.

- [ ] **Step 3: Inspect all 12 source slides**

```powershell
$env:Path = 'C:\Program Files\Git\usr\bin;' + $env:Path
node 'C:\Users\CHECK\.codex\plugins\cache\openai-primary-runtime\presentations\26.630.12135\skills\presentations\template_following_scripts\inspect_template_deck.mjs' --workspace 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp' --pptx 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\source-template.pptx'
```

Expected evidence:

- `template-manifest.json` reports `slideCount: 12`.
- `source-slide-01.png` through `source-slide-12.png` exist.
- Slide 1 contains the original factor-model flow and slide 2 is blank.

### Task 2: Create and validate the template audit, frame map, and starter deck

**Files:**
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/template-audit.txt`
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/template-frame-map.json`
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/deviation-log.txt`
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/template-starter.pptx`

- [ ] **Step 1: Record the template audit**

Write the following decisions to `template-audit.txt`:

```text
Source deck: 12 slides, 1280x720, white background.
Brand system: #002060/#003366 navy, #336699 blue, #BDD3E9 light blue, restrained gray panels.
Typography: inherited Shinhan fonts and source sizes; do not substitute a new theme.
Reusable frame: source slide 1 provides section title, subtitle, navy topic bar, three-stage workflow, footer, and page marker.
Output slide 1: duplicate source slide 1 and edit inherited objects in place.
Output slide 2: duplicate source slide 1, keep the same brand frame, rewrite inherited workflow objects, delete only explicitly mapped surplus objects, and allow only bounded additions inside the original workflow body.
Output slides 3-12: duplicate source slides 3-12 without edits.
Source slide 2: omit because it is blank and supplies no editable content frame.
```

- [ ] **Step 2: Create the 12-slide frame map**

The map must contain:

```json
{
  "outputSlides": [
    {
      "outputSlide": 1,
      "sourceSlide": 1,
      "narrativeRole": "factor research and expected-return workflow",
      "reuseMode": "duplicate-slide",
      "editTargets": [
        { "action": "rewrite-and-reposition", "shapeIds": ["sh/ex0nuxc3", "sh/3itgfmlk", "sh/lwval87e", "sh/e9cvixcz", "sh/b6xkf6dc", "sh/il0rqd4v", "sh/tcj6tova", "sh/t47y1kre", "sh/oza1gvy9", "sh/bulwval8", "sh/i9s36hcr"] }
      ]
    },
    {
      "outputSlide": 2,
      "sourceSlide": 1,
      "narrativeRole": "Barra-style expected-return covariance and optimization workflow",
      "reuseMode": "duplicate-slide",
      "editTargets": [
        { "action": "rewrite-and-reposition", "shapeIds": ["sh/ex0nuxc3", "sh/3itgfmlk", "sh/lwval87e", "sh/e9cvixcz", "sh/b6xkf6dc", "sh/il0rqd4v", "sh/tcj6tova", "sh/t47y1kre", "sh/oza1gvy9", "sh/bulwval8", "sh/i9s36hcr"] },
        { "action": "delete", "shapeIds": ["sh/3q5o7qpg", "sh/sneloj2t", "sh/l87ed8ba", "sh/0rqd4vmt", "sh/byhgn29g", "sh/idonyd0z"] },
        { "action": "add", "newPrimitiveAllowed": true, "zone": { "left": 212, "top": 178, "width": 870, "height": 333 }, "reason": "The inherited three-stage frame needs one additional branch connector and two outcome labels to show expected return and covariance as parallel optimizer inputs.", "mustNotOverlapInherited": true }
      ]
    },
    { "outputSlide": 3, "sourceSlide": 3, "narrativeRole": "preserve-only", "reuseMode": "duplicate-slide", "editTargets": [] },
    { "outputSlide": 4, "sourceSlide": 4, "narrativeRole": "preserve-only", "reuseMode": "duplicate-slide", "editTargets": [] },
    { "outputSlide": 5, "sourceSlide": 5, "narrativeRole": "preserve-only", "reuseMode": "duplicate-slide", "editTargets": [] },
    { "outputSlide": 6, "sourceSlide": 6, "narrativeRole": "preserve-only", "reuseMode": "duplicate-slide", "editTargets": [] },
    { "outputSlide": 7, "sourceSlide": 7, "narrativeRole": "preserve-only", "reuseMode": "duplicate-slide", "editTargets": [] },
    { "outputSlide": 8, "sourceSlide": 8, "narrativeRole": "preserve-only", "reuseMode": "duplicate-slide", "editTargets": [] },
    { "outputSlide": 9, "sourceSlide": 9, "narrativeRole": "preserve-only", "reuseMode": "duplicate-slide", "editTargets": [] },
    { "outputSlide": 10, "sourceSlide": 10, "narrativeRole": "preserve-only", "reuseMode": "duplicate-slide", "editTargets": [] },
    { "outputSlide": 11, "sourceSlide": 11, "narrativeRole": "preserve-only", "reuseMode": "duplicate-slide", "editTargets": [] },
    { "outputSlide": 12, "sourceSlide": 12, "narrativeRole": "preserve-only", "reuseMode": "duplicate-slide", "editTargets": [] }
  ],
  "omittedSourceSlides": [
    { "sourceSlide": 2, "reason": "Blank slide with no usable inherited content frame" }
  ]
}
```

- [ ] **Step 3: Validate the frame map**

```powershell
node 'C:\Users\CHECK\.codex\plugins\cache\openai-primary-runtime\presentations\26.630.12135\skills\presentations\template_following_scripts\validate_template_plan.mjs' --workspace 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp' --map 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\template-frame-map.json'
```

Expected: zero `fail` findings.

- [ ] **Step 4: Prepare the starter deck and previews**

```powershell
$env:Path = 'C:\Program Files\Git\usr\bin;' + $env:Path
node 'C:\Users\CHECK\.codex\plugins\cache\openai-primary-runtime\presentations\26.630.12135\skills\presentations\template_following_scripts\prepare_template_starter_deck.mjs' --workspace 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp' --pptx 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\source-template.pptx' --map 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\template-frame-map.json' --out 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\template-starter.pptx' --preview-dir 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\template-starter-preview' --layout-dir 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\template-starter-layout' --contact-sheet 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\template-starter-contact-sheet.png'
```

Expected: a 12-slide starter deck with source slide 1 duplicated into positions 1 and 2.

### Task 3: Edit slides 1 and 2 with artifact-tool

**Files:**
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/edit-emp008-deck.mjs`
- Create: `backtesting/strategies/emp008/outputs/제안서_멀티팩터모델_개편본.pptx`

- [ ] **Step 1: Implement source-element targeting and inherited-style edits**

Use numeric source element IDs on each duplicated slide:

```js
function shapeById(slide, id) {
  const shape = slide.shapes.items.find((item) => String(item.id) === String(id));
  if (!shape) throw new Error(`Missing inherited shape ${id}`);
  return shape;
}

function setText(slide, id, text) {
  const shape = shapeById(slide, id);
  shape.text = text;
  return shape;
}

function removeShapes(slide, ids) {
  for (const id of ids) shapeById(slide, id).delete();
}
```

- [ ] **Step 2: Modernize slide 1 without changing its 3-stage structure**

Keep the title and subtitle. Rewrite the topic bar and stage outputs with the following visible copy:

```js
setText(slide1, 58, "팩터모델 활용 흐름도");
setText(slide1, 42, "팩터 Clustering & Develop");
setText(slide1, 88, "국내 팩터 풀 관리");
setText(slide1, 46, "Quant Portfolio 구성");
setText(slide1, 45, "신한 멀티팩터 모델");
setText(slide1, 50, "팩터별 기대수익률 추정");
setText(slide1, 51, "종목별 기대수익률 산출");
setText(slide1, 52, "위험모형 및 최적화 연계");
setText(slide1, 56, "설명 가능한 팩터\n중복을 줄인 안정적 조합\n데이터 품질과 지속성 검증");
setText(slide1, 62, "유의성 검증\nFactor pool 편입\n성과 지속성 모니터링");
setText(slide1, 59, "[멀티팩터 모델 운용 원칙]\n설명 가능한 팩터를 선별하고, 중복성과 데이터 품질을 검증해 종목별 기대수익률로 변환");
```

Update the five inherited center rows without adding a sixth row:

```text
Value      FCF/TEV
Size       ln(시가총액)
Momentum   가격 모멘텀 · 이익 모멘텀
Yield      배당수익률
Flow       개인 수급
```

This keeps all six current EMP008 factors visible while preserving the inherited panel count and spacing.

- [ ] **Step 3: Rewrite slide 2 as the Barra-style estimation and optimization flow**

Use the inherited three panels as `입력 및 팩터 수익률 추정`, `기대수익률·공분산 추정`, and `Portfolio Optimization`. The visible copy must be:

```text
Barra Risk Model 기반 기대수익률·공분산 추정 및 포트폴리오 최적화

종목별 팩터 노출도 + 과거 종목 수익률
→ 월별 팩터 수익률·종목 고유 수익률 추정

기대수익률 추정
최근 36개월 평균 팩터 수익률 × 종목별 팩터 노출도
→ 종목별 기대수익률

분산·공분산 추정
팩터 공분산 + 종목 고유 분산
→ 종목 공분산 행렬

종목별 기대수익률 + 종목 공분산 행렬
→ 운용 제약조건 반영
→ 종목별 최적 비중
```

The lower inherited gray box must list only the implemented constraints:

```text
액티브 비중 합계 0 · 업종 중립 · 추적오차 한도 · 최종 종목 비중 0 이상
```

Do not include CAPM, formulas, performance claims, or promotional copy.

- [ ] **Step 4: Export previews, layout JSON, montage, and final PPTX**

```js
for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(`${previewDir}/${stem}.png`, await presentation.export({ slide, format: "png", scale: 2 }));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${layoutDir}/${stem}.layout.json`, await layout.text());
}
await writeBlob(`${qaDir}/final-montage.webp`, await presentation.export({ format: "webp", montage: true, scale: 1 }));
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(finalPptx);
```

Expected: the final PPTX exists and contains 12 slides.

### Task 4: Render and visually inspect every slide

**Files:**
- Inspect: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/preview/final/slide-01.png`
- Inspect: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/preview/final/slide-02.png`
- Inspect: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/qa/final-montage.webp`

- [ ] **Step 1: Review slides 1 and 2 at full size**

Verify:

- slide 1 still reads as the original three-stage workflow;
- all six current EMP008 factors are visible;
- slide 2 gives expected return and covariance equal visual weight;
- both estimates visibly converge into portfolio optimization;
- no CAPM or emotional headline remains;
- no title or banner wraps to two lines;
- the navy, blue, and light-blue palette remains dominant.

- [ ] **Step 2: Review the full-deck montage**

Expected: slides 3–12 match the source deck and no global theme or master change has altered them.

- [ ] **Step 3: Iterate until all visual defects are resolved**

For each defect, edit only the affected inherited object or bounded insertion, rerun `edit-emp008-deck.mjs`, and re-render. Do not lower font sizes below the inherited template hierarchy to make text fit; shorten copy instead.

### Task 5: Run structural and fidelity verification

**Files:**
- Verify: `backtesting/strategies/emp008/outputs/제안서_멀티팩터모델_개편본.pptx`
- Create: `C:/Users/CHECK/AppData/Local/Temp/codex-presentations/manual-emp008-risk-model/tmp/qa/verification.txt`

- [ ] **Step 1: Check slide count, required copy, and forbidden copy**

Inspect the final deck and assert:

```text
slide count = 12
required: 기대수익률, 공분산, Portfolio Optimization, 최근 36개월
forbidden: CAPM
```

- [ ] **Step 2: Run the template fidelity checker**

```powershell
node 'C:\Users\CHECK\.codex\plugins\cache\openai-primary-runtime\presentations\26.630.12135\skills\presentations\template_following_scripts\check_template_fidelity.mjs' --workspace 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp' --starter-pptx 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\template-starter.pptx' --final-pptx 'C:\Users\CHECK\Documents\GitHub\shquants\backtesting\strategies\emp008\outputs\제안서_멀티팩터모델_개편본.pptx' --map 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\template-frame-map.json' --starter-layout-dir 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\template-starter-layout' --final-layout-dir 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp\layout\final' --edit-dir 'C:\Users\CHECK\AppData\Local\Temp\codex-presentations\manual-emp008-risk-model\tmp'
```

Expected: zero fidelity failures.

- [ ] **Step 3: Inspect final PPTX XML for empty placeholders**

Use `unzip -p` or the bundled verifier to check every `ppt/slides/slide*.xml`. Any `<p:sp>` containing `<p:ph>` must have non-empty text or be explicitly deleted by the edit plan.

Expected: no unresolved title, date, footer, slide-number, or body placeholders.

- [ ] **Step 4: Record completion evidence**

Write `verification.txt` with the final slide count, required/forbidden copy result, visual QA result, fidelity-check result, and the final PPTX path.

- [ ] **Step 5: Commit the plan before implementation and do not commit the generated PPTX unless requested**

```powershell
git add -- 'docs/superpowers/plans/2026-07-27-mfbt-emp008-two-page-risk-model.md'
git commit -m 'Make the EMP008 deck redesign executable and verifiable' -m 'The plan preserves the supplied template, maps every output slide to a source frame, and defines exact artifact-tool and QA steps.' -m 'Constraint: Use the supplied PPTX as the sole visual source' -m 'Confidence: high' -m 'Scope-risk: narrow' -m 'Tested: Plan coverage, placeholder, and path review' -m 'Not-tested: Final deck rendering is deferred to execution'
```
