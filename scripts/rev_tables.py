"""All revision-2 numbers for the manuscript, from the model outputs.

usage: python rev_tables.py <out_dir> <supp_dir>
Writes <supp_dir>/S5_scenario_results.csv and prints every number used in the text and tables.
"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from unfold import Unfold

OUT, SUP = Path(sys.argv[1]), Path(sys.argv[2]); SUP.mkdir(parents=True, exist_ok=True)
DOMS = ['FarWest', 'West', 'East', 'FarEast']
DOM = {'FarWest': (369500, 370500), 'West': (370500, 372050), 'East': (372050, 373250), 'FarEast': (373250, 374300)}
AB = {'indicator': 'I', 'signed': 'S', 'surfaces': 'U', 'lenses': 'L'}
REF = {'indicator': 'threshold_0.5', 'signed': 'reference', 'surfaces': 'reference', 'lenses': 'reference'}
GT_GEOMS = {'indicator': ['threshold_0.3', 'threshold_0.5', 'threshold_0.7', 'ranges_x0.75', 'ranges_x1.25']}
P = lambda *a: print(*a, flush=True)

# ------------------------------------------------------------------ volumes by construction and variant
rows = []
for d in DOMS:
    for f in (OUT / f'contacts_{d}.json', OUT / f'lenses_{d}.json'):
        for r in json.load(open(f))['rows']:
            rows.append(dict(domain=d, construction=r['construction'], variant=r['variant'], volume=r['volume_Mm3']))
V = pd.DataFrame(rows).drop_duplicates(['domain', 'construction', 'variant'], keep='last')
T = V.groupby(['construction', 'variant']).volume.sum().unstack(0)
P('=== total MS volume (Mm3) by variant\n', T.round(3).to_string())
for con in ('signed', 'surfaces', 'lenses'):
    r = T.loc[REF[con], con]
    P(f'{con}: contacts +/-1 m {T.loc["contacts_in_1m", con]:.2f}-{T.loc["contacts_out_1m", con]:.2f}'
      f' ({100*(T.loc["contacts_in_1m", con]/r-1):+.0f}% / {100*(T.loc["contacts_out_1m", con]/r-1):+.0f}%)')
r = T.loc['threshold_0.5', 'indicator']
P(f'indicator: thresholds {T.loc["threshold_0.7","indicator"]:.2f}-{T.loc["threshold_0.3","indicator"]:.2f} ({100*(T.loc["threshold_0.7","indicator"]/r-1):+.0f}% / {100*(T.loc["threshold_0.3","indicator"]/r-1):+.0f}%)')
for con in T.columns:
    vv = T[con].dropna(); P(f'{con}: ranges {T.loc["ranges_x0.75", con]:.2f}-{T.loc["ranges_x1.25", con]:.2f};'
                           f' other (excl. contacts/threshold/ranges/withheld) {vv.drop([i for i in vv.index if i.startswith(("contacts","threshold","ranges","without"))], errors="ignore").agg(["min","max"]).round(2).tolist()}')
P('=== range sensitivity by lens group (% vs reference)')
for con in T.columns:
    for d in DOMS:
        x = V[(V.domain == d) & (V.construction == con)].set_index('variant').volume
        if REF[con] in x and 'ranges_x0.75' in x:
            P(f'  {con:9s} {d:8s} {100*(x["ranges_x0.75"]/x[REF[con]]-1):+5.0f}% {100*(x["ranges_x1.25"]/x[REF[con]]-1):+5.0f}%  ref {x[REF[con]]:.3f}')
P('reference by lens group:\n', V[V.apply(lambda r: r.variant == REF[r.construction], axis=1)].pivot(index='domain', columns='construction', values='volume').loc[DOMS].round(3).to_string())

# ------------------------------------------------------------------ pairwise overlap of reference geometries
P('=== pairwise overlap (footprint Jaccard; 3D with one cell = 2.5 m tolerance across)')
ov = []
for d in DOMS:
    z = np.load(OUT / f'contacts_{d}_masks.npz'); names = list(z['names']); N = int(z['n_cells'])
    shp = (len(z['gs']), len(z['gd']), len(z['gn']))
    M = {c: np.unpackbits(z[f'm{names.index(c + "|" + v)}'])[:N].astype(bool).reshape(shp) for c, v in REF.items()}
    keys = list(M)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            A, B = M[a], M[b]; fa, fb = A.any(2), B.any(2)
            jf = (fa & fb).sum() / max((fa | fb).sum(), 1)
            dil = lambda X: X | np.roll(X, 1, 2) | np.roll(X, -1, 2)
            j3 = ((A & dil(B)).sum() + (B & dil(A)).sum()) / max(A.sum() + B.sum(), 1)
            ov.append(dict(domain=d, pair=f'{AB[a]}-{AB[b]}', footprint=round(jf, 2), tol25=round(j3, 2)))
OV = pd.DataFrame(ov); P(OV.pivot(index='domain', columns='pair', values='footprint').loc[DOMS].to_string())
P(OV.pivot(index='domain', columns='pair', values='tol25').loc[DOMS].to_string())
P('footprint range all pairs', OV.footprint.min(), OV.footprint.max(), '; vs indicator', OV[OV.pair.str.startswith('I')].footprint.min(), OV[OV.pair.str.startswith('I')].footprint.max())
OV.to_csv(SUP / 'overlap_pairs.csv', index=False)

# ------------------------------------------------------------------ data honouring of the reference geometries
P('=== data honouring (coded composites, cell containing the composite centre)')
comp = pd.read_csv(OUT / 'composites.csv'); U = Unfold(comp); z0 = float(comp.Z.mean())
L = U(comp[['X', 'Y', 'Z']].to_numpy()); comp['s'], comp['z'], comp['n'] = L[:, 0], L[:, 1] - z0, L[:, 2]
hon = []
for d in DOMS:
    z = np.load(OUT / f'contacts_{d}_masks.npz'); names = list(z['names']); N = int(z['n_cells'])
    gs, gd, gn = z['gs'], z['gd'], z['gn']; shp = (len(gs), len(gd), len(gn))
    c = comp[(comp.X >= DOM[d][0]) & (comp.X < DOM[d][1]) & comp.form.notna()]
    i = np.round((c.s - gs[0]) / 10).astype(int); j = np.round((c.z - gd[0]) / 10).astype(int); k = np.round((c.n - gn[0]) / 2.5).astype(int)
    ok = ((i >= 0) & (i < shp[0]) & (j >= 0) & (j < shp[1]) & (k >= 0) & (k < shp[2])).to_numpy()
    ms = (c.f_ms >= 0.5).to_numpy()
    for con, v in REF.items():
        m = np.unpackbits(z[f'm{names.index(con + "|" + v)}'])[:N].astype(bool).reshape(shp)
        ins = np.zeros(len(c), bool); ins[ok] = m[i[ok], j[ok], k[ok]]
        md = m | np.roll(m, 1, 2) | np.roll(m, -1, 2)
        ins2 = np.zeros(len(c), bool); ins2[ok] = md[i[ok], j[ok], k[ok]]
        hon.append(dict(domain=d, construction=con, n=len(c), n_ms=int(ms.sum()), ms_in=int((ms & ins).sum()), nonms_in=int((~ms & ins).sum()),
                        ms_in_tol=int((ms & ins2).sum())))
H = pd.DataFrame(hon).groupby('construction')[['n', 'n_ms', 'ms_in', 'nonms_in', 'ms_in_tol']].sum()
H['MS_inside_tol25_pct'] = 100 * H.ms_in_tol / H.n_ms
H['MS_inside_pct'] = 100 * H.ms_in / H.n_ms; H['nonMS_outside_pct'] = 100 * (1 - H.nonms_in / (H.n - H.n_ms))
H['overall_pct'] = 100 * (H.ms_in + (H.n - H.n_ms - H.nonms_in)) / H.n; H['shell_cells_from_MS_pct'] = 100 * H.ms_in / (H.ms_in + H.nonms_in)
P(H.round(1).to_string())

# ------------------------------------------------------------------ withheld draws (three constructions)
dr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / 'draws_*.csv')))])
g = dr.groupby(['draw', 'construction'])[['all_Mm3', 'without_Mm3', 'cols_0', 'changed_0', 'cols_25', 'changed_25', 'cols_50', 'changed_50']].sum()
g['dv'] = 100 * (g.without_Mm3 / g.all_Mm3 - 1)
for b in ('0', '25', '50'): g['p' + b] = 100 * g['changed_' + b] / g['cols_' + b]
P('=== withheld draws (all lens groups)\n', g[['dv', 'cols_0', 'p0', 'cols_25', 'p25', 'cols_50', 'p50']].round(1).to_string())
P('draw holes withheld that cross a lens group:', dr.groupby(['draw', 'domain']).n_withheld_crossing.first().unstack().to_string())
pd.concat([dr]).to_csv(SUP / 'withheld_draw_results.csv', index=False)

# ------------------------------------------------------------------ grade-tonnage (all geometries carried to grades)
gt = pd.read_csv(OUT / 'gt_table.csv')
keep_v = lambda r: (r.variant != 'without_frozen') and (r.construction != 'indicator' or r.variant in GT_GEOMS['indicator'])
gt = gt[gt.apply(keep_v, axis=1)]
G = gt[gt.density_factor == 1.0].groupby(['construction', 'variant', 'zn_cutoff']).agg(
    Mt=('tonnes_Mt', 'sum'), Zn_kt=('Zn_kt', 'sum'), Mt_reached=('tonnes_reached_Mt', 'sum'),
    vol=('volume_Mm3', 'sum')).reset_index()
gz = gt[gt.density_factor == 1.0].copy(); gz['ZnT'] = gz.Zn_pct.fillna(0) * gz.tonnes_Mt
gz['unr_T'] = gz.share_unreached.fillna(0) * gz.tonnes_Mt
agg = gz.groupby(['construction', 'variant', 'zn_cutoff'])[['ZnT', 'tonnes_Mt', 'unr_T']].sum()
G = G.merge((agg.ZnT / agg.tonnes_Mt).rename('Zn_pct').reset_index(), on=['construction', 'variant', 'zn_cutoff'])
G = G.merge((agg.unr_T / agg.tonnes_Mt).rename('share_unreached').reset_index(), on=['construction', 'variant', 'zn_cutoff'])
G['n_geoms'] = 1
G.round(4).to_csv(SUP / 'S5_scenario_results.csv', index=False)
P('number of geometries carried to grades:', G[G.zn_cutoff == 4].groupby('construction').size().to_dict(), 'total', (G.zn_cutoff == 4).sum())
for cut in (0, 4, 8):
    x = G[G.zn_cutoff == cut]; r = x[x.apply(lambda q: q.variant == REF[q.construction], axis=1)]
    P(f'--- cut-off {cut}% Zn: references Mt {r.Mt.min():.1f}-{r.Mt.max():.1f}, Zn {r.Zn_pct.min():.1f}-{r.Zn_pct.max():.1f}, Zn kt {r.Zn_kt.min():.0f}-{r.Zn_kt.max():.0f};'
      f' all {len(x)}: Mt {x.Mt.min():.1f}-{x.Mt.max():.1f}, Zn {x.Zn_pct.min():.1f}-{x.Zn_pct.max():.1f}, Zn kt {x.Zn_kt.min():.0f}-{x.Zn_kt.max():.0f}')
    if cut == 4:
        P('  refs:\n', r[['construction', 'Mt', 'Mt_reached', 'Zn_pct', 'Zn_kt', 'share_unreached']].round(3).to_string())
        P(f'  CV across all {len(x)}: tonnes {100*x.Mt.std()/x.Mt.mean():.0f}%, Zn kt {100*x.Zn_kt.std()/x.Zn_kt.mean():.0f}%, Zn grade {100*x.Zn_pct.std()/x.Zn_pct.mean():.1f}%')
        for con, y in x.groupby('construction'):
            P(f'  {con:9s} n={len(y):2d} Mt {y.Mt.min():.2f}-{y.Mt.max():.2f} CV {100*y.Mt.std()/y.Mt.mean():.0f}%  Zn {y.Zn_pct.min():.2f}-{y.Zn_pct.max():.2f} CV {100*y.Zn_pct.std()/y.Zn_pct.mean():.1f}%'
              f'  unreached share {y.share_unreached.min():.3f}-{y.share_unreached.max():.3f}  Mt excl. unreached {y.Mt_reached.min():.2f}-{y.Mt_reached.max():.2f}')
        P('  extremes:\n', x.sort_values('Mt').iloc[[0, -1]][['construction', 'variant', 'Mt', 'Mt_reached', 'Zn_pct', 'share_unreached']].round(3).to_string())
