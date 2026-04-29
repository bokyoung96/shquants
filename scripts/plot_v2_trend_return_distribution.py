import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_quantile')
rets = pd.read_csv(base / 'v2_trend_filter_returns.csv', parse_dates=['date']).set_index('date').iloc[:,0]

sk = float(skew(rets, bias=False))
ku = float(kurtosis(rets, fisher=True, bias=False))
mean = float(rets.mean())
std = float(rets.std())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(rets, bins=60, color='#1f77b4', alpha=0.85)
axes[0].axvline(mean, color='red', linestyle='--', linewidth=1.8, label=f'mean={mean:.4f}')
axes[0].set_title('Return Distribution, V2 + Trend Filter')
axes[0].set_xlabel('Return')
axes[0].set_ylabel('Count')
axes[0].grid(True, alpha=0.2)
axes[0].legend()

axes[1].boxplot(rets, vert=True)
axes[1].set_title('Boxplot')
axes[1].grid(True, alpha=0.2)
axes[1].set_ylabel('Return')

fig.suptitle(f'Skew={sk:.3f}, Kurtosis={ku:.3f}, Std={std:.4f}', y=1.02)
plt.tight_layout()
out = base / 'v2_trend_filter_return_distribution.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
