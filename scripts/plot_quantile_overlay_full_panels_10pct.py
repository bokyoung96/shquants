import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_quantile')
df = pd.read_csv(base / 'quantile_breakout_overlay_10pct.csv', parse_dates=['date']).set_index('date')
df['bm_dd'] = df['bm_nav'] / df['bm_nav'].cummax() - 1.0
df['overlay_dd'] = df['overlay_nav'] / df['overlay_nav'].cummax() - 1.0
df['extra_weight'] = df['alpha_overlay']
periods = [
    ('2015-01-01', '2026-04-28', 'From 2015'),
    ('2020-01-01', '2026-04-28', 'From 2020'),
    ('2025-01-01', '2026-04-28', 'From 2025'),
]

fig, axes = plt.subplots(3, 1, figsize=(16, 15), sharex=False)
for ax, (start, end, title) in zip(axes, periods):
    sub = df.loc[start:end].copy()
    bm = sub['bm_nav'] / sub['bm_nav'].iloc[0]
    ov = sub['overlay_nav'] / sub['overlay_nav'].iloc[0]

    ax.plot(sub.index, bm, color='black', linestyle='--', linewidth=1.8, label='BM 100%')
    ax.plot(sub.index, ov, color='#d62728', linewidth=2.0, label='BM 100% + max 10% overlay')
    ax.set_title(f'{title}, Equity / Drawdown / Alpha / Extra Weight')
    ax.set_ylabel('Equity')
    ax.grid(True, alpha=0.25)

    ax_dd = ax.twinx()
    ax_dd.spines['right'].set_position(('outward', 45))
    ax_dd.plot(sub.index, sub['overlay_dd'], color='#9467bd', linewidth=1.5, alpha=0.9, label='Overlay DD')
    ax_dd.plot(sub.index, sub['bm_dd'], color='#7f7f7f', linestyle=':', linewidth=1.2, alpha=0.9, label='BM DD')
    ax_dd.set_ylabel('Drawdown')
    ax_dd.set_ylim(min(sub['overlay_dd'].min(), sub['bm_dd'].min()) * 1.1, 0.05)

    ax_alpha = ax.twinx()
    ax_alpha.spines['right'].set_position(('outward', 90))
    ax_alpha.step(sub.index, sub['alpha_overlay'], where='post', color='#1f77b4', linewidth=1.6, label='Alpha')
    ax_alpha.plot(sub.index, sub['extra_weight'], color='#2ca02c', linewidth=1.4, alpha=0.9, label='Extra weight')
    ax_alpha.set_ylabel('Alpha / Extra')
    ax_alpha.set_ylim(-0.005, max(0.105, sub['alpha_overlay'].max() * 1.2 if len(sub) else 0.105))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_dd.get_legend_handles_labels()
    lines3, labels3 = ax_alpha.get_legend_handles_labels()
    if ax is axes[0]:
        ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', ncol=2)

axes[-1].set_xlabel('Date')
plt.tight_layout()
out = base / 'quantile_breakout_overlay_full_panels_10pct.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
