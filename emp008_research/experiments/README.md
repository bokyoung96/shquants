# EMP008 experiments

Research drivers and their manifests live here. Generated CSV, Parquet, image,
and workbook outputs belong under `../results/`, not in source directories.

The production handoff path is deliberately short: `emp008/` calculates target
weights and `backtest/` evaluates those weights. Add a dated experiment folder
here only when changing factor sets, timing, or optimizer settings; keep the
exact CLI arguments and source data manifest in that folder.
