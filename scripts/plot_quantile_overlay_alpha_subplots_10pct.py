import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_quantile')
df = pd.read_csv(base / 'quantile_breakout_overlay_10pct.csv', parse_dates=['date']).set_index('date')
periods = [
    ('2015-01-01', '2026-04-28', 'From 2015'),
    ('2020-01-01', '2026-04-28', 'From 2020'),
    ('2025-01-01', '2026-04-28', 'From 2025'),
]

fig, axes = plt.subplots(3, 1, figsize=(15, 14), sharex=False)
for ax, (start, end, title) in zip(axes, periods):
    sub = df.loc[start:end].copy()
    bm = sub['bm_nav'] / sub['bm_nav'].iloc[0]
    ov = sub['overlay_nav'] / sub['overlay_nav'].iloc[0]
    alpha = sub['alpha_overlay']

    ax.plot(sub.index, bm, color='black', linestyle='--', linewidth=1.9, label='BM 100%')
    ax.plot(sub.index, ov, color='#d62728', linewidth=2.1, label='BM 100% + max 10% overlay')
    ax.set_title(f'{title}, Equity with Alpha Overlay')
    ax.set_ylabel('Normalized Equity')
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.step(sub.index, alpha, where='post', color='#1f77b4', linewidth=1.8, label='Alpha overlay step')
    ax2.scatter(sub.index[:: max(1, len(sub)//60)], alpha.iloc[:: max(1, len(sub)//60)], c=alpha.iloc[:: max(1, len(sub)//60)], cmap='Blues', s=18, alpha=0.85)
    ax2.set_ylabel('Alpha overlay')
    ax2.set_ylim(-0.005, max(0.105, alpha.max() * 1.15 if len(sub) else 0.105))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if ax is axes[0]:
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

axes[-1].set_xlabel('Date')
plt.tight_layout()
out = base / 'quantile_breakout_overlay_alpha_subplots_10pct.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
