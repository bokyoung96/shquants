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

bs_band_num = band_to_num(df['bs_band'])
ofs_band_num = band_to_num(df['ofs_band'])

fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.3, 1.3, 1.1]})

axes[0].plot(df.index, df['price'], color='#111111', linewidth=2.0)
axes[0].set_title('Aggressive Weekly Strategy Example, 2020-2023')
axes[0].set_ylabel('KOSPI200')
axes[0].grid(True, alpha=0.25)

axes[1].plot(df.index, df['bs'], color='#1f77b4', linewidth=1.8, label='BS')
axes[1].scatter(df.index, bs_band_num, c=bs_band_num, cmap='Blues', s=10, alpha=0.45, label='BS band')
axes[1].set_ylabel('BS')
axes[1].set_yticks([0, 1, 2])
axes[1].set_yticklabels(['L', 'M', 'H'])
axes[1].grid(True, alpha=0.25)
axes[1].legend(loc='upper left')

axes[2].plot(df.index, df['ofs'], color='#d62728', linewidth=1.8, label='OFS')
axes[2].scatter(df.index, ofs_band_num, c=ofs_band_num, cmap='Reds', s=10, alpha=0.45, label='OFS band')
axes[2].set_ylabel('OFS')
axes[2].set_yticks([0, 1, 2])
axes[2].set_yticklabels(['L', 'M', 'H'])
axes[2].grid(True, alpha=0.25)
axes[2].legend(loc='upper left')

axes[3].step(df.index, df['exposure'].fillna(method='ffill').fillna(0.0), where='post', color='#2ca02c', linewidth=2.0)
axes[3].set_ylabel('Exposure')
axes[3].set_xlabel('Date')
axes[3].set_ylim(-0.05, 1.05)
axes[3].grid(True, alpha=0.25)

plt.tight_layout()
out = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_index_weekly/aggressive_weekly_band_exposure_2020_2023.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
