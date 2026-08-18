# Buy-Sidecar Event Journey Visual Design

## Goal

Replace the current three-panel analytical chart with one story-first figure that makes the operational conclusion readable within a few seconds.

## Approved Direction

Use the selected **Event Journey** direction. The figure should explain the sequence `activation -> A+3 entry -> release -> R+3 exit` before showing secondary comparisons.

## Information Hierarchy

1. Conclusion headline: `매수 사이드카: 발동 +3분 진입, 해제 +3분 청산`.
2. Context line: 15 events, data through 2026-06-15, first-minute-boundary-open fill convention.
3. Main timeline:
   - activation at minute 0;
   - five-minute program-bid halt;
   - robust entry window A+1 through A+4;
   - operational entry marker A+3;
   - release at minute 5;
   - conservative exit window R+2 through R+3;
   - operational exit marker R+3.
4. Primary metrics: gross mean +0.872%, after-10bp mean +0.772%, gross wins 15/15.
5. Supporting evidence:
   - entry comparison A+0, A+3, A+5;
   - holding-risk comparison for R+3, same-day close, and next-day close.
6. Footer: provisional result, n=15, no true OOS, 1-minute OHLC execution approximation.

## Visual System

- Canvas: 16:9, high-resolution PNG.
- Background: near-white neutral, not cream or dark navy.
- Text: charcoal with zero letter-spacing; Korean-capable system font.
- Event-halt band: coral.
- Robust/selected trading window: teal and green.
- Secondary/risk information: cool gray and restrained red.
- Avoid legends where direct labels can be used.
- Avoid a dense heatmap, large confidence band, and three equal-weight panels.

## Data Contract

The visualization consumes the existing `canonical_summary`, `execution_summary`, and `holding_summary` frames. It must not alter research calculations, rule selection, output CSVs, or the report conclusion.

## Acceptance Criteria

- The headline, entry marker, release marker, and exit marker are visible without zooming.
- A viewer can identify A+3 and R+3 without reading axes or a legend.
- The robust windows A+1~4 and R+2~3 remain visible.
- Gross, net-of-10bp, and win-count metrics are displayed exactly once.
- Sample and OOS limitations remain on the figure.
- The generated image is nonblank, at least 1600x900, and contains no clipped or overlapping labels.
- Existing sidecar research tests continue to pass.
