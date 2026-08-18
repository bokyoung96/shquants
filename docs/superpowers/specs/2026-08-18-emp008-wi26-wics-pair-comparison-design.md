# EMP008 WI26-WICS Pair Comparison Design

## Goal

Compare WI26 and WICS without mixing factor-weight choices by pairing candidates with identical IDs and plotting their cumulative relative-return gap.

## Outputs

- `candidate_pair_cumulative_gap.png`: a 3x3 grid, one panel per common candidate, showing the full-period cumulative `WICS - WI26` relative-return gap in basis points with calendar-year separators.
- `yearly_pair_cumulative_gap.png`: a 4x2 grid, one panel per year, resetting to zero at each year start and showing all common candidate gaps.
- `yearly_pair_end_gap_bp.csv`: each candidate's year-end gap for numeric verification.

The gap is computed as the WICS candidate's cumulative benchmark-relative return minus the paired WI26 candidate's cumulative benchmark-relative return. Positive values favor WICS; negative values favor WI26. Both runs must have identical benchmark returns and common dates.

## Structure

Create a focused report module under `backtesting/strategies/emp008/reports/`. It loads the two existing `daily_returns.csv` files, validates pairing and benchmark identity, builds full-period and yearly frames, renders the two figures, and writes a manifest. Unit tests cover pairing, benchmark validation, annual reset, and gap direction.

The historical unsuffixed `factor_weight_grid_search` directory is byte-identical to the files shared with `factor_weight_grid_search_wi26`; the WI26 directory has four additional deliverables. Rewrite stale embedded paths in the WI26 text artifacts, verify no remaining references, then remove the redundant unsuffixed directory.

## Verification

- Unit tests for frame construction and validation.
- Run the report against the saved WI26 and WICS artifacts.
- Confirm the year-end CSV reproduces the known 2023 WI26 advantage for all nine candidates.
- Open both PNGs and visually verify titles, zero lines, legends, and subplot layout.
