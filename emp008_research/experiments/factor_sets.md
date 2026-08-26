# Factor-set catalog

The canonical names below are accepted by the weight-generation entrypoint.
They are ordered from the original reference model to the current production
core. A factor set defines the factors; unless a run supplies explicit factor
weights, the factors are equally weighted.

| Canonical name | Role | Factors |
| --- | --- | --- |
| `origin` | Original reference | size, momentum-12m, dividend-yield-fy0 |
| `origin_add` | Production baseline | price-to-252d-high, earnings-momentum, dividend-yield-ttm, retail-flow, value, size |
| `production_core` | Production core (equal weight) | size, momentum-12m, earnings-momentum, value |
| `research_12_1m_momentum` | Research variant | momentum-12-1m, earnings-momentum, dividend-yield-ttm, value, size |
| `research_positivity_momentum` | Research variant | positivity-momentum, earnings-momentum, dividend-yield-ttm, retail-flow, value, size |
| `research_origin_small_cap_rule` | Research variant | production factors with origin small-cap alpha-direction rule |
| `reference_origin` | Reference reproduction | size, momentum-12m, dividend-yield-fy0 |
| `reference_origin_ttm_dividend` | Reference reproduction | size, momentum-12m, dividend-yield-ttm |
| `reference_origin_12_1m` | Reference reproduction | size, momentum-12-1m, dividend-yield-fy0 |
| `research_size_only` | Research variant | size |
| `research_size_momentum_12m` | Research variant | size, momentum-12m |
| `research_size_momentum_12_1m` | Research variant | size, momentum-12-1m |
| `research_size_momentum_high` | Research variant | size, price-to-252d-high |
| `research_size_earnings_momentum` | Research variant | size, earnings-momentum |
| `research_size_retail_flow` | Research variant | size, retail-flow |
| `research_size_value_fcf_tev` | Research variant | size, value (FCF/TEV) |
| `research_size_momentum_earnings_value` | Research variant | size, momentum-12m, earnings-momentum, value |
| `research_size_value_dividend_fy0` | Research variant | size, value, dividend-yield-fy0 |
| `research_size_value_dividend_ttm` | Research variant | size, value, dividend-yield-ttm |
| `diagnostic_all_factors` | Diagnostics only | all registered factors |

Only the canonical names above are accepted by the handoff configuration. The
former `mfbt` naming and short factor-set aliases have been removed.
