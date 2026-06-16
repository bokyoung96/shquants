# FnGuide AI Semiconductor TOP2 Plus Index data inventory

- Index: FI00.WLT.KSS
- Calculation readiness: missing_calculation_inputs
- Replication proven: False
- Methodology status: methodology_verified

## Requirements

| Requirement | Usage | Status | Notes |
| --- | --- | --- | --- |
| price_snapshot | calculation_input | available | Local price history supports direct price-momentum derivation. |
| price_momentum | calculation_input | derivable | Derive price momentum from the local price snapshot once the rebalance window is fixed. |
| float_market_cap_snapshot | calculation_input | available | Free-float market cap is needed for top-2 ranking and residual weighting. |
| semiconductor_classification_snapshot | calculation_input | available | Classification snapshot used to build the semiconductor selection universe; this is not an official constituent list. |
| sales_momentum | calculation_input | missing | Sales momentum input required by the KSS momentum bucket selection. |
| composite_score | calculation_input | missing | Composite score or enough component inputs to calculate it are required for ranking. |
| issuer_holdings_snapshot | validation_evidence | available | ETF issuer holdings are useful proxy evidence but cannot prove full index replication on their own. |
| corporate_actions | calculation_input | missing | Corporate action history is needed to keep the security universe and weights aligned through rebalance dates. |
| official_bucket_assignments | validation_evidence | missing | Official bucket assignments validate the calculated bucket output; they are not calculation inputs. |
| official_target_weights | validation_evidence | missing | Official target weights validate calculated weights; they are not required to calculate unknown constituents. |
