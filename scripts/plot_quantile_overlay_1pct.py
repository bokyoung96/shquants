import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_quantile')
df = pd.read_csv(base / 'quantile_breakout_overlay_1pct.csv', parse_dates=['date']).set_index('date')

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios':[2,1]})
axes[0].plot(df.index, df['bm_nav'], color='black', linestyle='--', linewidth=2.0, label='BM 100%')
axes[0].plot(df.index, df['overlay_nav'], color='#d62728', linewidth=2.2, label='BM 100% + max 1% overlay')
axes[0].set_title('KOSPI200 100% vs KOSPI200 100% + Quantile Breakout Overlay (Max 1%)')
axes[0].set_ylabel('Normalized Equity')
axes[0].grid(True, alpha=0.25)
axes[0].legend(loc='upper left')

bm_dd = df['bm_nav'] / df['bm_nav'].cummax() - 1.0
overlay_dd = df['overlay_nav'] / df['overlay_nav'].cummax() - 1.0
axes[1].plot(df.index, bm_dd, color='black', linestyle='--', linewidth=1.8, label='BM drawdown')
axes[1].plot(df.index, overlay_dd, color='#d62728', linewidth=2.0, label='Overlay drawdown')
axes[1].axhline(0.0, color='black', linewidth=1, alpha=0.5)
axes[1].set_title('Drawdown')
axes[1].set_ylabel('Drawdown')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.25)
axes[1].legend(loc='lower left')

plt.tight_layout()
out = base / 'quantile_breakout_overlay_max1pct.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
