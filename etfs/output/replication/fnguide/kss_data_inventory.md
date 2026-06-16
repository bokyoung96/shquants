# FnGuide AI Semiconductor TOP2 Plus Index data inventory

- Index: FI00.WLT.KSS
- Replication readiness: missing_required_data
- Methodology status: methodology_verified

## Requirements

| Requirement | Status | Notes |
| --- | --- | --- |
| price_snapshot | available | Local price history supports direct price-momentum derivation. |
| price_momentum | derivable | Derive price momentum from the local price snapshot once the rebalance window is fixed. |
| float_market_cap_snapshot | available | Free-float market cap is needed for top-2 ranking and residual weighting. |
| sector_classification | available | Local sector labels can support semiconductor screening, but not provider theme confirmation. |
| theme_membership | external_required | Full replication still needs FnGuide or provider-confirmed theme membership evidence. |
| sales_momentum | external_required | The methodology depends on official sales-momentum inputs that are not inferable from current local files. |
| composite_score | external_required | Composite ranking inputs remain provider-controlled unless the official scoring formula and values are supplied. |
| issuer_holdings_snapshot | available | ETF issuer holdings are useful proxy evidence but cannot prove full index replication on their own. |
| corporate_actions | missing | Corporate action history is needed to keep the security universe and weights aligned through rebalance dates. |
| official_bucket_assignments | external_required | Official bucket assignments are required to confirm which names land in top-2, momentum, and fill buckets. |
| official_target_weights | external_required | Official target weights remain the primary replication and validation evidence for full fidelity. |
