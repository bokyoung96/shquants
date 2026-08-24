# Factor-set catalog

The canonical names below are the names accepted by the weight-generation
CLI. `production_core` is the only operator-facing production set.

| Canonical name | Role | Factors |
| --- | --- | --- |
| `production_core` | Production | price-to-252d-high, earnings-momentum, dividend-yield-ttm, retail-flow, value, size |
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

Legacy names (`mfbt`, `adjust`, `origin`, and the former `size_*` names) are
accepted only by the Python compatibility parser. New configuration and CLI
usage should use the canonical names above.
