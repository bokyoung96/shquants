import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_quantile')
df = pd.read_csv(base / 'v2_filter_variant_navs.csv', parse_dates=['date']).set_index('date')

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios':[2,1]})
axes[0].plot(df.index, df['bm_100'], color='black', linestyle='--', linewidth=1.8, label='BM 100%')
axes[0].plot(df.index, df['v2_base'], color='#d62728', linewidth=2.0, label='V2 base')
axes[0].plot(df.index, df['v2_trend_filter'], color='#1f77b4', linewidth=2.0, label='V2 + trend filter')
axes[0].plot(df.index, df['v2_vol_cap'], color='#2ca02c', linewidth=2.0, label='V2 + vol cap')
axes[0].set_title('V2 Overlay Variants vs Benchmark')
axes[0].set_ylabel('Normalized Equity')
axes[0].grid(True, alpha=0.25)
axes[0].legend(loc='upper left')

for col, color, label, ls in [
    ('bm_100', 'black', 'BM DD', '--'),
    ('v2_base', '#d62728', 'V2 base DD', '-'),
    ('v2_trend_filter', '#1f77b4', 'Trend filter DD', '-'),
    ('v2_vol_cap', '#2ca02c', 'Vol cap DD', '-'),
]:
    dd = df[col] / df[col].cummax() - 1.0
    axes[1].plot(df.index, dd, color=color, linestyle=ls, linewidth=1.7, label=label)

axes[1].axhline(0.0, color='black', linewidth=1, alpha=0.5)
axes[1].set_title('Drawdown')
axes[1].set_ylabel('Drawdown')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.25)
axes[1].legend(loc='lower left', ncol=2)

plt.tight_layout()
out = base / 'v2_trend_vol_compare.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
