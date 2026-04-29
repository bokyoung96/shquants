import json
from pathlib import Path
import pandas as pd
from root import ROOT

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_quantile')
bm = pd.read_parquet(ROOT.parquet_path / 'qw_BM.parquet')['IKS200'].loc['2015-01-01':'2026-04-28'].dropna()
ret = bm.pct_change().fillna(0.0)
base_exp = pd.read_csv(base / 'quantile_breakout_monthly_exposure.csv', parse_dates=['date']).set_index('date')['exposure']
base_exp = base_exp.reindex(bm.index).ffill().fillna(0.0)

trend_ma = bm.rolling(120, min_periods=60).mean()
trend_mult = pd.Series(1.0, index=bm.index)
trend_mult = trend_mult.where(bm > trend_ma, 0.5)

rv20 = ret.rolling(20, min_periods=10).std() * (252 ** 0.5)
vol_pct = rv20.rolling(252, min_periods=126).rank(pct=True)
# rolling rank shortcut is awkward, so use apply
vol_pct = rv20.rolling(252, min_periods=126).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
vol_mult = pd.Series(1.0, index=bm.index)
vol_mult = vol_mult.where(vol_pct < 0.8, 0.5)

variants = {
    'v2_base': base_exp,
    'v2_trend_filter': base_exp * trend_mult,
    'v2_vol_cap': base_exp * vol_mult,
}

def stats(nav):
    r = nav.pct_change().fillna(0.0)
    years = len(r) / 252
    return {
        'cagr': float(nav.iloc[-1] ** (1 / years) - 1),
        'mdd': float((nav / nav.cummax() - 1.0).min()),
        'sharpe': float((r.mean() / r.std()) * (252 ** 0.5)) if r.std() > 0 else 0.0,
        'final_nav': float(nav.iloc[-1]),
    }

rows = {}
nav_df = pd.DataFrame(index=bm.index)
nav_df['bm_100'] = (1.0 + ret).cumprod()
for name, exp in variants.items():
    nav = (1.0 + ret * (1.0 + 0.10 * exp)).cumprod()
    nav_df[name] = nav
    rows[name] = stats(nav)
rows['bm_100'] = stats(nav_df['bm_100'])

nav_df.to_csv(base / 'v2_filter_variant_navs.csv', index_label='date')
(base / 'v2_filter_variant_stats.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(rows, ensure_ascii=False, indent=2))
