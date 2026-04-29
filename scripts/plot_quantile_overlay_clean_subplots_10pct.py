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

fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=False, gridspec_kw={'height_ratios': [2,1,2,1,2,1]})
for i, (start, end, title) in enumerate(periods):
    sub = df.loc[start:end].copy()
    bm = sub['bm_nav'] / sub['bm_nav'].iloc[0]
    ov = sub['overlay_nav'] / sub['overlay_nav'].iloc[0]

    ax_top = axes[i*2]
    ax_bot = axes[i*2 + 1]

    ax_top.plot(sub.index, bm, color='black', linestyle='--', linewidth=1.8, label='BM 100%')
    ax_top.plot(sub.index, ov, color='#d62728', linewidth=2.0, label='BM 100% + max 10% overlay')
    ax_top.set_title(f'{title}, Equity with Alpha / Extra Weight')
    ax_top.set_ylabel('Equity')
    ax_top.grid(True, alpha=0.25)

    ax_top_r = ax_top.twinx()
    ax_top_r.step(sub.index, sub['alpha_overlay'], where='post', color='#1f77b4', linewidth=1.5, label='Alpha')
    ax_top_r.plot(sub.index, sub['extra_weight'], color='#2ca02c', linewidth=1.3, alpha=0.9, label='Extra weight')
    ax_top_r.set_ylabel('Alpha / Extra')
    ax_top_r.set_ylim(-0.005, max(0.105, sub['alpha_overlay'].max() * 1.2 if len(sub) else 0.105))

    l1, lb1 = ax_top.get_legend_handles_labels()
    l2, lb2 = ax_top_r.get_legend_handles_labels()
    ax_top.legend(l1 + l2, lb1 + lb2, loc='upper left', ncol=2)

    ax_bot.plot(sub.index, sub['bm_dd'], color='black', linestyle='--', linewidth=1.5, label='BM DD')
    ax_bot.plot(sub.index, sub['overlay_dd'], color='#9467bd', linewidth=1.8, label='Overlay DD')
    ax_bot.axhline(0.0, color='black', linewidth=1, alpha=0.5)
    ax_bot.set_title(f'{title}, Drawdown')
    ax_bot.set_ylabel('Drawdown')
    ax_bot.grid(True, alpha=0.25)
    ax_bot.legend(loc='lower left')
    if i == len(periods) - 1:
        ax_bot.set_xlabel('Date')

plt.tight_layout()
out = base / 'quantile_breakout_overlay_clean_subplots_10pct.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
