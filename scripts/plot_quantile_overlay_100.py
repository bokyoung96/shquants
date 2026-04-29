import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from root import ROOT

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_quantile')
bm = pd.read_parquet(ROOT.parquet_path / 'qw_BM.parquet')['IKS200'].loc['2015-01-01':'2026-04-28'].dropna()
bm_ret = bm.pct_change().fillna(0.0)
exp = pd.read_csv(base / 'quantile_breakout_monthly_exposure.csv', parse_dates=['date']).set_index('date')['exposure']
exp = exp.reindex(bm.index).ffill().fillna(0.0)

# baseline 100%
bm_nav = (1.0 + bm_ret).cumprod()
# overlay: 100% + strategy exposure, capped at 2.0 gross
overlay_ret = bm_ret * (1.0 + exp)
overlay_nav = (1.0 + overlay_ret).cumprod()

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios':[2,1]})
axes[0].plot(bm_nav.index, bm_nav.values, color='black', linestyle='--', linewidth=2.0, label='BM 100%')
axes[0].plot(overlay_nav.index, overlay_nav.values, color='#d62728', linewidth=2.2, label='BM 100% + Quantile Overlay')
axes[0].set_title('KOSPI200 100% vs KOSPI200 100% + Quantile Breakout Overlay')
axes[0].set_ylabel('Normalized Equity')
axes[0].grid(True, alpha=0.25)
axes[0].legend(loc='upper left')

bm_dd = bm_nav / bm_nav.cummax() - 1.0
overlay_dd = overlay_nav / overlay_nav.cummax() - 1.0
axes[1].plot(bm_dd.index, bm_dd.values, color='black', linestyle='--', linewidth=1.8, label='BM drawdown')
axes[1].plot(overlay_dd.index, overlay_dd.values, color='#d62728', linewidth=2.0, label='Overlay drawdown')
axes[1].axhline(0.0, color='black', linewidth=1, alpha=0.5)
axes[1].set_title('Drawdown')
axes[1].set_ylabel('Drawdown')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.25)
axes[1].legend(loc='lower left')

plt.tight_layout()
out = base / 'quantile_breakout_overlay_on_bm100.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(out)
