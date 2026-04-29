import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from root import ROOT

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_quantile')
run = base / 'quantile_breakout_monthly_20260429_104412'
bm = pd.read_parquet(ROOT.parquet_path / 'qw_BM.parquet')['IKS200'].loc['2015-01-01':'2026-04-28'].dropna()
bm = bm / bm.iloc[0]
eq = pd.read_csv(run / 'series' / 'equity.csv', parse_dates=['date']).set_index('date')['equity']
eq = eq / eq.iloc[0]

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios':[2,1]})
axes[0].plot(bm.index, bm.values, color='black', linestyle='--', linewidth=2.0, label='BM KOSPI200')
axes[0].plot(eq.index, eq.values, color='#d62728', linewidth=2.2, label='Quantile Breakout Monthly')
axes[0].set_title('Quantile Breakout Monthly vs Benchmark')
axes[0].set_ylabel('Normalized Equity')
axes[0].grid(True, alpha=0.25)
axes[0].legend(loc='upper left')

bm_dd = bm / bm.cummax() - 1.0
st_dd = eq / eq.cummax() - 1.0
axes[1].plot(bm_dd.index, bm_dd.values, color='black', linestyle='--', linewidth=1.8, label='BM drawdown')
axes[1].plot(st_dd.index, st_dd.values, color='#d62728', linewidth=2.0, label='Strategy drawdown')
axes[1].axhline(0.0, color='black', linewidth=1, alpha=0.5)
axes[1].set_title('Drawdown')
axes[1].set_ylabel('Drawdown')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.25)
axes[1].legend(loc='lower left')

plt.tight_layout()
out = base / 'quantile_breakout_monthly_vs_bm_combo.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
