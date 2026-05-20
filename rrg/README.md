# Advanced 3D RRG

This package builds a three-dimensional Relative Rotation Graph for KOSPI200 WICS large sectors.

It is not a trading strategy and it does not attempt to reproduce proprietary JdK RRG formulas. It uses transparent log-relative-strength derivatives:

```text
RS_t = sector_index_t / benchmark_index_t
LRS_t = log(RS_t)
MOM_t = LRS_t - LRS_{t-k}
ACC_t = MOM_t - MOM_{t-k}
```

The 3D phase-space axes are:

- `x`: centered log relative strength
- `y`: relative momentum
- `z`: normalized acceleration, `ACC z-score`

## Example

```bash
uv run python -m rrg.examples --start 2020-01-02 --end 2026-03-25 --output results/rrg/advanced_rrg_3d.html
```

The example reuses the existing backtesting data stack and requires these parquet datasets:

- `qw_adj_c`
- `qw_BM`
- `qw_k200_yn`
- `qw_wics_sec_big`
- `qw_mktcap`

## Interpretation

- High RS, high MOM, high ACC: relative leadership is strengthening.
- High RS, high MOM, low ACC: leadership may be exhausting.
- Low RS, low MOM, high ACC: early recovery candidate.
- Low RS, low MOM, low ACC: relative weakness is still worsening.

Use the plot as a sector leadership transition monitor, not as a standalone buy/sell signal.
