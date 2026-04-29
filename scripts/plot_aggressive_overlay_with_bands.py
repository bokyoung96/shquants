import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_index_weekly/aggressive_band_debug.csv')
df = pd.read_csv(csv_path, parse_dates=['date']).set_index('date')
df = df.loc['2020-01-01':'2023-12-31'].copy()
df['exposure_ffill'] = df['exposure'].ffill().fillna(0.0)

def band_to_num(s):
    return s.map({'L': 0, 'M': 1, 'H': 2})

df['bs_band_num'] = band_to_num(df['bs_band'])
df['ofs_band_num'] = band_to_num(df['ofs_band'])

fig, ax1 = plt.subplots(figsize=(15, 8))
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 55))

ax1.plot(df.index, df['price'], color='black', linewidth=2.0, label='KOSPI200')

colors = {0.0: '#d3d3d3', 0.25: '#a6cee3', 0.5: '#1f78b4', 0.75: '#33a02c', 1.0: '#e31a1c'}
for exp, color in colors.items():
    mask = df['exposure_ffill'] == exp
    if mask.any():
        ax1.fill_between(df.index, df['price'].min()*0.98, df['price'].max()*1.02, where=mask, color=color, alpha=0.10, transform=ax1.get_xaxis_transform(), label=f'exposure={exp}')

buy_up = df['exposure_ffill'].diff().fillna(0) > 0
sell_down = df['exposure_ffill'].diff().fillna(0) < 0
ax1.scatter(df.index[buy_up], df.loc[buy_up, 'price'], marker='^', s=55, color='#d62728', label='Exposure up')
ax1.scatter(df.index[sell_down], df.loc[sell_down, 'price'], marker='v', s=45, color='#1f77b4', label='Exposure down')

ax2.step(df.index, df['exposure_ffill'], where='post', color='#2ca02c', linewidth=2.1, label='Exposure')
ax2.set_ylim(-0.05, 1.05)
ax2.set_ylabel('Exposure')

ax3.plot(df.index, df['bs_band_num'], color='#6a3d9a', linewidth=1.4, alpha=0.9, label='BS band')
ax3.plot(df.index, df['ofs_band_num'], color='#ff7f00', linewidth=1.4, alpha=0.9, label='OFS band')
ax3.set_ylim(-0.2, 2.2)
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(['L', 'M', 'H'])
ax3.set_ylabel('Band state')

ax1.set_title('Aggressive Weekly Overlay, Price + Exposure + BS/OFS Band State (2020-2023)')
ax1.set_ylabel('KOSPI200')
ax1.grid(True, alpha=0.25)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
seen = set()
all_lines, all_labels = [], []
for line, label in list(zip(lines1, labels1)) + list(zip(lines2, labels2)) + list(zip(lines3, labels3)):
    if label not in seen:
        all_lines.append(line)
        all_labels.append(label)
        seen.add(label)
ax1.legend(all_lines, all_labels, loc='upper left', ncol=2)

plt.tight_layout()
out = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_index_weekly/aggressive_weekly_overlay_with_bands_2020_2023.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
