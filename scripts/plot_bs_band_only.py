import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_index_weekly/aggressive_band_debug.csv')
df = pd.read_csv(csv_path, parse_dates=['date']).set_index('date')
df = df.loc['2020-01-01':'2023-12-31'].copy()

def band_to_num(s):
    return s.map({'L': 0, 'M': 1, 'H': 2})

df['bs_band_num'] = band_to_num(df['bs_band'])

fig, ax1 = plt.subplots(figsize=(15, 7))
ax2 = ax1.twinx()

ax1.plot(df.index, df['price'], color='black', linewidth=2.0, label='KOSPI200')

for band, color, alpha in [('L', '#d7191c', 0.10), ('M', '#fdae61', 0.08), ('H', '#1a9641', 0.10)]:
    mask = df['bs_band'] == band
    if mask.any():
        ax1.fill_between(df.index, df['price'].min()*0.98, df['price'].max()*1.02, where=mask, color=color, alpha=alpha, transform=ax1.get_xaxis_transform(), label=f'BS band={band}')

ax2.step(df.index, df['bs_band_num'], where='post', color='#1f77b4', linewidth=2.0, label='BS band state')
ax2.set_ylim(-0.2, 2.2)
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['L', 'M', 'H'])
ax2.set_ylabel('BS band')

ax1.set_title('BS Band Only, Aggressive Weekly Example (2020-2023)')
ax1.set_ylabel('KOSPI200')
ax1.grid(True, alpha=0.25)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
seen = set()
all_lines, all_labels = [], []
for line, label in list(zip(lines1, labels1)) + list(zip(lines2, labels2)):
    if label not in seen:
        all_lines.append(line)
        all_labels.append(label)
        seen.add(label)
ax1.legend(all_lines, all_labels, loc='upper left', ncol=2)

plt.tight_layout()
out = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_index_weekly/bs_band_only_2020_2023.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
