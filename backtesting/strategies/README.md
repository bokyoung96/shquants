# Strategy Registry Notes

## index_alpha_tilt_consensus_revision_oi_beta

Index-management alpha tilt strategy for KOSPI200-like mandates.

### Description
- Base: K200 membership weighted by available `qw_mktcap`.
- Alpha inputs: EPS/OP forward revision, foreign+institution-vs-retail order imbalance, and beta-adjusted price momentum.
- Construction: long-only benchmark overlay with active-share, stock-level, and sector-level caps.
- Intended use: keep index participation while tilting toward names with improving consensus, supportive order imbalance, and high-beta momentum confirmation.

### Implementation notes
- Python strategy id: `index_alpha_tilt_consensus_revision_oi_beta`
- Source file: `backtesting/strategies/index_alpha_tilt_consensus_revision_oi_beta.py`
- Uses `qw_mktcap` rather than `qw_mktcap_flt` because the current raw/parquet data set does not include `qw_mktcap_flt`.

## Removed low-output strategies

- `consensus_beta_persistence_concentrated_longonly` was removed after the local 2023-01-02..2026-04-15 evaluation produced 0 nonzero-weight days, 0.0 total return, and 0.0 turnover.
- `revision_asymmetric_relay_hedge_ls` was removed after the local 2026-01-02..2026-04-15 smoke evaluation produced 0.0 total return.
- `revision_minparam_v02` was removed after the local 2026-01-02..2026-04-15 smoke evaluation produced 0.0 total return.
- `revision_oi_beta_momo_gate_ls` was removed after the local 2026-01-02..2026-04-15 smoke evaluation produced 0 nonzero-weight days, 0.0 total return, and 0.0 gross exposure.
- `revision_oi_high_beta_momentum_ls` was removed after the local 2026-01-02..2026-04-15 smoke evaluation produced -0.067% total return with only 1 nonzero-weight day.
- `revision_oi_soft_beta_tilt_momentum_ls` was removed after the local 2026-01-02..2026-04-15 smoke evaluation produced -0.067% total return with only 1 nonzero-weight day.
- `revision_oi_state_conditioned_beta_gate_ls` was removed after the local 2026-01-02..2026-04-15 smoke evaluation produced only 3.07% total return.
- `revision_oi_state_conditioned_short_squeeze_beta_cap_ls` was removed as a low-return duplicate of the remaining short-squeeze exclusion idea.
- `revision_oi_state_conditioned_short_squeeze_beta_exclusion_ls` was removed as a low-return duplicate of the remaining short-squeeze exclusion idea.
