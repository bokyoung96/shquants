import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from root import ROOT

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_index_weekly')
runs = {
    'BM KOSPI200': None,
    'Balanced weekly': base / 'kospi200_index_bs_opf_band_balanced_weekly_20260429_095701',
    'Aggressive weekly': base / 'kospi200_index_bs_opf_band_aggressive_weekly_20260429_100048',
}

series = {}
bm = pd.read_parquet(ROOT.parquet_path / 'qw_BM.parquet')['IKS200'].loc['2015-01-01':'2026-04-28'].dropna()
series['BM KOSPI200'] = bm / bm.iloc[0]

for label, path in runs.items():
    if path is None:
        continue
    eq = pd.read_csv(path / 'series' / 'equity.csv', parse_dates=['date']).set_index('date')['equity']
    series[label] = eq / eq.iloc[0]

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
colors = {
    'BM KOSPI200': '#111111',
    'Balanced weekly': '#2ca02c',
    'Aggressive weekly': '#d62728',
}
linestyles = {
    'BM KOSPI200': '--',
    'Balanced weekly': '-',
    'Aggressive weekly': '-',
}

for label, eq in series.items():
    axes[0].plot(eq.index, eq.values, label=label, linewidth=2.1 if label == 'BM KOSPI200' else 2.2, color=colors[label], linestyle=linestyles[label])
    dd = eq / eq.cummax() - 1.0
    axes[1].plot(dd.index, dd.values, label=label, linewidth=1.9 if label == 'BM KOSPI200' else 2.0, color=colors[label], linestyle=linestyles[label])

axes[0].set_title('Weekly BS + OPF Variants vs Benchmark, Equity Curves (Normalized to 1.0)')
axes[0].set_ylabel('Normalized Equity')
axes[0].grid(True, alpha=0.25)
axes[0].legend(loc='upper left')

axes[1].set_title('Drawdown')
axes[1].set_ylabel('Drawdown')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.25)
axes[1].axhline(0.0, color='black', linewidth=1, alpha=0.5)

plt.tight_layout()
out = base / 'bs_opf_weekly_balanced_aggressive_vs_bm_combo.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
