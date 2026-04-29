import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_quantile')
v2 = pd.read_csv(base / 'quantile_breakout_overlay_10pct.csv', parse_dates=['date']).set_index('date')
v3 = pd.read_csv(base / 'v3_overlay_10pct.csv', parse_dates=['date']).set_index('date')

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios':[2,1]})
axes[0].plot(v2.index, v2['bm_nav'], color='black', linestyle='--', linewidth=1.8, label='BM 100%')
axes[0].plot(v2.index, v2['overlay_nav'], color='#d62728', linewidth=2.0, label='V2 overlay 10%')
axes[0].plot(v3.index, v3['v3_nav'], color='#1f77b4', linewidth=2.0, label='V3 overlay 10%')
axes[0].set_title('V2 vs V3 Overlay on BM 100%')
axes[0].set_ylabel('Normalized Equity')
axes[0].grid(True, alpha=0.25)
axes[0].legend(loc='upper left')

bm_dd = v2['bm_nav'] / v2['bm_nav'].cummax() - 1.0
v2_dd = v2['overlay_nav'] / v2['overlay_nav'].cummax() - 1.0
v3_dd = v3['v3_nav'] / v3['v3_nav'].cummax() - 1.0
axes[1].plot(v2.index, bm_dd, color='black', linestyle='--', linewidth=1.5, label='BM DD')
axes[1].plot(v2.index, v2_dd, color='#d62728', linewidth=1.8, label='V2 DD')
axes[1].plot(v3.index, v3_dd, color='#1f77b4', linewidth=1.8, label='V3 DD')
axes[1].axhline(0.0, color='black', linewidth=1, alpha=0.5)
axes[1].set_title('Drawdown')
axes[1].set_ylabel('Drawdown')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.25)
axes[1].legend(loc='lower left')

plt.tight_layout()
out = base / 'v2_v3_overlay_compare.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
