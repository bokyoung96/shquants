import json
from pathlib import Path
import pandas as pd
from root import ROOT

# Fixed v2 spec
# strategy: kospi200_index_bs_opf_quantile_breakout
# schedule: weekly
# overlay cap: 10%
# no parameter retuning during this run

base = Path('/mnt/c/Users/AI_Quant/Documents/GitHub/shquants/results/backtests_bsopf_quantile')
bm = pd.read_parquet(ROOT.parquet_path / 'qw_BM.parquet')['IKS200'].dropna()
exp = pd.read_csv(base / 'quantile_breakout_monthly_exposure.csv', parse_dates=['date']).set_index('date')['exposure']
exp = exp.reindex(bm.index).ffill().fillna(0.0)
ret = bm.pct_change().fillna(0.0)
overlay_ret = ret * (1.0 + 0.10 * exp)
nav_bm = (1.0 + ret).cumprod()
nav_overlay = (1.0 + overlay_ret).cumprod()

splits = {
    'IS_2015_2020': ('2015-01-01', '2020-12-31'),
    'OOS_2021_2026': ('2021-01-01', '2026-04-28'),
}


def stats(nav):
    r = nav.pct_change().fillna(0.0)
    years = len(r) / 252
    cagr = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    mdd = float((nav / nav.cummax() - 1.0).min())
    sharpe = float((r.mean() / r.std()) * (252 ** 0.5)) if r.std() > 0 else 0.0
    final = float(nav.iloc[-1])
    return {'cagr': float(cagr), 'mdd': mdd, 'sharpe': sharpe, 'final_nav': final}

results = {}
for name, (start, end) in splits.items():
    bm_sub = nav_bm.loc[start:end]
    ov_sub = nav_overlay.loc[start:end]
    results[name] = {
        'bm': stats(bm_sub / bm_sub.iloc[0]),
        'overlay': stats(ov_sub / ov_sub.iloc[0]),
    }

out = {'spec_locked': True, 'strategy': 'kospi200_index_bs_opf_quantile_breakout', 'schedule': 'weekly', 'overlay_cap': 0.10, 'splits': results}
out_path = base / 'quantile_v2_is_oos_results.json'
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))
