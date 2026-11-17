"""Normative density for every assay interval, then length-weighted onto the 2 m composites (out/composites_density.csv).
Pb as galena, Zn as sphalerite, Cu as chalcopyrite, Ba as barite, remaining S as pyrite, rest felsic gangue 2.75 t/m3.
Intervals without S: density from base metals, fitted separately on MS and non-MS intervals that have S."""
import sys, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
a = pd.read_csv(src, low_memory=False).rename(columns={'Drillhole ID': 'hole', 'From': 'f', 'To': 't'})
n = lambda c: pd.to_numeric(a[c], errors='coerce')
Zn, Pb, Cu, S, Ba = n('Zn_pct'), n('Pb_pct'), n('Cu_pct'), n('S_pct'), n('BA_PCT')
z, p, c, b = Zn.fillna(0), Pb.fillna(0), Cu.fillna(0), Ba.fillna(0)
w = pd.DataFrame(dict(gal=p / .866, sph=z / .671, cpy=c / .346, bar=b / .588)) / 100
Srem = (S - 100 * (w.gal * .134 + w.sph * .329 + w.cpy * .349 + w.bar * .137)).clip(lower=0)
w['py'] = Srem / .534 / 100
w = w.div(np.maximum(w.sum(axis=1), 1), axis=0); g = (1 - w.sum(axis=1)).clip(lower=0)
rho = dict(gal=7.58, sph=4.05, cpy=4.19, bar=4.48, py=5.01)
a['den'] = 1 / (sum(w[k] / rho[k] for k in rho) + g / 2.75)
a.loc[S.isna(), 'den'] = np.nan
a['base'] = z + p + c
comp = pd.read_csv(out + '/composites.csv')
a = a[a.hole.isin(set(comp.hole)) & (a.t > a.f)]
# composite: length-weighted density of overlapping intervals (with S only) and base metals
rows = []
A = a.groupby('hole')
for h, cg in comp.groupby('hole'):
    if h not in A.groups:
        continue
    ag = A.get_group(h)
    for i, r in cg.iterrows():
        ov = (np.minimum(ag.t, r.t) - np.maximum(ag.f, r.f)).clip(lower=0)
        m = (ov > 0) & ag.den.notna()
        if m.any():
            rows.append((i, float(np.average(ag.den[m], weights=ov[m]))))
d = pd.Series(dict(rows), dtype=float)
comp['den_S'] = d
base = comp[['Zn_pct', 'Pb_pct', 'Cu_pct']].fillna(0).sum(axis=1)
ms = comp.f_ms >= 0.5
coef = {}
for k, sel in (('MS', ms), ('other', ~ms)):
    m = sel & comp.den_S.notna() & comp.Zn_pct.notna()
    X = np.column_stack([np.ones(m.sum()), base[m]]); beta = np.linalg.lstsq(X, comp.den_S[m], rcond=None)[0]
    res = comp.den_S[m] - X @ beta
    coef[k] = (beta, int(m.sum()), float(1 - res.var() / comp.den_S[m].var()), float(res.std()))
    fill = sel & comp.den_S.isna()
    comp.loc[fill, 'den'] = beta[0] + beta[1] * base[fill]
comp['den'] = comp.den_S.combine_first(comp.get('den'))
comp.loc[comp.den.isna(), 'den'] = 2.75
comp[['hole', 'f', 't', 'den', 'den_S']].to_csv(out + '/composites_density.csv', index=False)
for k, (beta, nn, r2, sd) in coef.items():
    print(f'{k}: den = {beta[0]:.3f} + {beta[1]:.4f} * (Zn+Pb+Cu)   n={nn}  R2={r2:.2f}  rmse={sd:.3f}')
print('MS composites: density mean %.2f, with S %d of %d' % (comp.den[ms].mean(), comp.den_S[ms].notna().sum(), ms.sum()))
