import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
from unfold import Unfold
OUT = Path('out'); ip = pd.read_csv(sys.argv[1])
comp = pd.read_csv(OUT / 'composites.csv'); U = Unfold(comp); z0 = float(comp.Z.mean())
DOM = {'FarWest': (369500, 370500), 'West': (370500, 372050), 'East': (372050, 373250), 'FarEast': (373250, 374300)}
rows = []
for dom, (x0, x1) in DOM.items():
    z = np.load(OUT / f'contacts_{dom}_masks.npz'); names = list(z['names']); n = int(z['n_cells'])
    gs, gd, gn = z['gs'], z['gd'], z['gn']
    G = np.stack(np.meshgrid(gs, gd, gn, indexing='ij'), -1).reshape(-1, 3)
    m = np.zeros(n, bool)
    for k in ('indicator|threshold_0.5', 'signed|reference', 'surfaces|reference', 'lenses|reference'):
        m |= np.unpackbits(z[f'm{names.index(k)}'])[:n].astype(bool)
    oh = np.load(OUT / f'{dom}_grid.npz')['OH'].reshape(-1) >= 0
    p = ip[(ip.GridE >= x0) & (ip.GridE < x1)].copy()
    L = U(p[['GridE', 'GridN', 'Elev']].to_numpy(float)); L[:, 1] -= z0
    p['d_ms'] = cKDTree(G[m]).query(L)[0]; p['d_oh'] = cKDTree(G[oh]).query(L)[0]
    p['dom'] = dom; rows.append(p)
p = pd.concat(rows)
bins = [0, 25, 50, 100, 200, 1e9]; lab = ['0-25', '25-50', '50-100', '100-200', '>200']
p['bin_ms'] = pd.cut(p.d_ms, bins, labels=lab, right=False); p['bin_oh'] = pd.cut(p.d_oh, bins, labels=lab, right=False)
t1 = p.groupby('bin_ms', observed=True).agg(n=('InvIP', 'size'), ip_median=('InvIP', 'median'), ip_p75=('InvIP', lambda v: v.quantile(.75)), res_median=('InvRes', 'median')).round(2)
t2 = p.groupby('bin_oh', observed=True).agg(n=('InvIP', 'size'), ip_median=('InvIP', 'median'), ip_p75=('InvIP', lambda v: v.quantile(.75)), res_median=('InvRes', 'median')).round(2)
t3 = p[p.d_ms < 50].groupby('dom').agg(n=('InvIP', 'size'), ip_median=('InvIP', 'median')).round(2)
t4 = p[p.d_ms >= 200].groupby('dom').agg(n=('InvIP', 'size'), ip_median=('InvIP', 'median')).round(2)
print('IP vs distance to modelled MS (m)\n', t1, '\nIP vs distance to OH shell (m)\n', t2, '\nwithin 50 m of MS by domain\n', t3, '\n>200 m by domain\n', t4)
print('IP points within the four lens groups:', len(p))
pd.concat([t1.assign(ref='MS'), t2.assign(ref='OH')]).to_csv(OUT / 'step6_ip_distance.csv')
