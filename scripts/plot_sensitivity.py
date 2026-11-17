"""Step 2 figures and table.

usage: python plot_sensitivity.py
For every domain with out/sens_<domain>.json:
  out/sens_<domain>.png  - top: how often each (along-trace, elevation) position holds MS across all
                           variant x threshold combinations; bottom: the same for OH.
                           1.0 = inside in every defensible geometry, low values = setting-dependent.
Writes out/sensitivity_table.csv and out/sensitivity_summary.json (P10/P50/P90 of volume per domain
and target over all combinations, plus the volume swing due to each kind of setting).
"""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
files = sorted(glob.glob(os.path.join(OUT, 'sens_*.json')))
allrows, summ = [], []
for f in files:
    js = json.load(open(f)); dom = js['domain']
    df = pd.DataFrame(js['rows']); df['domain'] = dom; allrows.append(df)
    df['family'] = df.variant.str.split('_').str[0]
    for tgt, g in df.groupby('target'):
        ref = g[(g.variant == 'reference') & (g.threshold == 0.5)].volume_Mm3.iloc[0]
        s = dict(domain=dom, target=tgt, reference_Mm3=ref,
                 P10=round(g.volume_Mm3.quantile(.1), 3), P50=round(g.volume_Mm3.quantile(.5), 3), P90=round(g.volume_Mm3.quantile(.9), 3),
                 min=round(g.volume_Mm3.min(), 3), max=round(g.volume_Mm3.max(), 3))
        # swing from each family at threshold 0.5, and from threshold alone on the reference fit
        for fam, h in g[g.threshold == 0.5].groupby('family'):
            if fam != 'reference':
                s[f'swing_{fam}_pct'] = round(100 * (h.volume_Mm3.max() - h.volume_Mm3.min()) / ref, 1)
        r = g[g.variant == 'reference']
        s['swing_threshold_pct'] = round(100 * (r.volume_Mm3.max() - r.volume_Mm3.min()) / ref, 1)
        s['min_jaccard_vs_reference'] = round(g.get('overlap_with_reference_jaccard', pd.Series([np.nan])).min(), 3)
        summ.append(s)

    z = np.load(os.path.join(OUT, f'sens_{dom}_freq.npz'))
    n = float(z['n_comb']); gs, gd = z['gs'], z['gd']
    fig, ax = plt.subplots(2, 1, figsize=(11, 9))
    for a, tgt, cmap in zip(ax, ('MS', 'OH'), ('Reds', 'Blues')):
        col = z[tgt].max(2) / n          # share of combinations with shell anywhere across the horizon
        col = np.where(col > 0, col, np.nan)
        im = a.pcolormesh(gs, gd, col.T, cmap=cmap, vmin=0, vmax=1, shading='nearest')
        a.set_aspect('equal'); a.set_xlabel('along the ore-horizon trace (m)'); a.set_ylabel('elevation relative to mean (m)')
        a.set_title(f'{dom}: share of {int(n)} defensible geometries that place {tgt} here')
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'sens_{dom}.png'), dpi=110); plt.close(fig)

if allrows:
    pd.concat(allrows).to_csv(os.path.join(OUT, 'sensitivity_table.csv'), index=False)
    json.dump(summ, open(os.path.join(OUT, 'sensitivity_summary.json'), 'w'), indent=1)
    print(pd.DataFrame(summ).to_string(index=False))
