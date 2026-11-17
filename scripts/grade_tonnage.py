"""Step 5: ordinary kriging of Zn, Pb, Cu and density inside every MS geometry, and grade-tonnage per geometry.

usage: python grade_tonnage.py <out_dir> [domain ...]
Reads out/composites.csv, out/composites_density.csv, out/contacts_<domain>_masks.npz.
Hard MS boundary: only MS composites (f_ms >= 0.5) inform MS cells. One set of variograms (fitted once on all MS
composites in the unfolded frame) is used for every geometry, so differences come from geometry alone.
Writes out/gt_variograms.json, out/gt_table.csv (domain x construction x variant x cut-off) and out/gt_cells_<domain>.npz.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from unfold import Unfold

OUT = Path(sys.argv[1])
DOMS = sys.argv[2:] or ['FarWest', 'West', 'East', 'FarEast']
DOM = {'FarWest': (369500, 370500), 'West': (370500, 372050), 'East': (372050, 373250), 'FarEast': (373250, 374300)}
CUTS = [0, 2, 4, 6, 8, 10]
VARS = ['Zn_pct', 'Pb_pct', 'Cu_pct', 'den']
CELL = 10 * 10 * 2.5

comp = pd.read_csv(OUT / 'composites.csv')
den = pd.read_csv(OUT / 'composites_density.csv')
comp = comp.merge(den[['hole', 'f', 'den']], on=['hole', 'f'], how='left')
U = Unfold(comp); z0 = float(comp.Z.mean())
L = U(comp[['X', 'Y', 'Z']].to_numpy()); L[:, 1] -= z0
comp['s'], comp['z'], comp['n'] = L[:, 0], L[:, 1], L[:, 2]
ms = comp[(comp.f_ms >= 0.5) & comp.Zn_pct.notna()].copy()
for v in ('Zn_pct', 'Pb_pct', 'Cu_pct'):
    cap = ms[v].quantile(0.995); ms[v] = ms[v].clip(upper=cap)           # top cut at the 99.5th percentile
ms['den'] = ms.den.fillna(ms.den.median())
ms = ms[ms[['Pb_pct', 'Cu_pct']].notna().all(axis=1)]
print('MS composites with grades:', len(ms))


# ------------------------------------------------------------------ variograms: spherical, nugget + 1 structure,
# anisotropic ranges along s, z (in plane) and n (across); fitted on directional experimental semivariograms
def sph(h, c0, c1, a):
    h = np.minimum(h / a, 1.0)
    return c0 + c1 * (1.5 * h - 0.5 * h ** 3)


P = ms[['s', 'z', 'n']].to_numpy()
tree = cKDTree(P)
pairs = tree.query_pairs(150.0, output_type='ndarray')
pairs = pairs[np.random.default_rng(1).permutation(len(pairs))[:3_000_000]]
D = P[pairs[:, 1]] - P[pairs[:, 0]]
ad = np.abs(D)
dirs = {'s': (ad[:, 0] > 2 * np.hypot(ad[:, 1], ad[:, 2])) & (ad[:, 2] < 10),
        'z': (ad[:, 1] > 2 * np.hypot(ad[:, 0], ad[:, 2])) & (ad[:, 2] < 10),
        'n': (np.hypot(ad[:, 0], ad[:, 1]) < 5)}
lags = {'s': np.arange(5, 155, 10.0), 'z': np.arange(5, 155, 10.0), 'n': np.arange(1, 31, 2.0)}
vg = {}
for v in VARS:
    x = ms[v].to_numpy(); g2 = 0.5 * (x[pairs[:, 1]] - x[pairs[:, 0]]) ** 2
    exp = {}
    for k, m in dirs.items():
        h = np.linalg.norm(D[m], axis=1); gm = g2[m]; e = lags[k]; w = e[1] - e[0]
        b = np.digitize(h, e - w / 2) - 1
        rows = [(e[i], gm[b == i].mean(), int((b == i).sum())) for i in range(len(e)) if (b == i).sum() >= 30]
        exp[k] = np.array(rows)
    var = float(np.var(x))

    def resid(p):
        c0, c1, a_s, a_z, a_n = p
        r = []
        for k, a in (('s', a_s), ('z', a_z), ('n', a_n)):
            if len(exp[k]):
                r.append(np.sqrt(exp[k][:, 2]) * (sph(exp[k][:, 0], c0, c1, a) - exp[k][:, 1]) / var)
        return np.concatenate(r)
    var = max(var, 1e-9)
    fit = least_squares(resid, [0.2 * var, 0.8 * var, 60, 60, 8], bounds=([0, 0, 10, 10, 2], [2 * var, 3 * var, 300, 300, 40]))
    c0, c1, a_s, a_z, a_n = fit.x
    vg[v] = dict(nugget=float(c0), sill=float(c1), range_s=float(a_s), range_z=float(a_z), range_n=float(a_n), variance=var,
                 experimental={k: exp[k].round(4).tolist() for k in exp})
    print(f'{v}: nugget {c0:.3g} sill {c1:.3g} ranges s {a_s:.0f} z {a_z:.0f} n {a_n:.1f} m (var {var:.3g})')
json.dump(vg, open(OUT / 'gt_variograms.json', 'w'), indent=1)


# ------------------------------------------------------------------ ordinary kriging with 24 neighbours (min 3)
def ok(v, Pd, xd, Pg, nmax=24, nmin=3):
    p = vg[v]; R = np.array([p['range_s'], p['range_z'], p['range_n']])
    c0, c1 = p['nugget'], p['sill']
    cov = lambda h: np.where(h == 0, c0 + c1, c1 * (1 - np.minimum(h, 1) * (1.5 - 0.5 * np.minimum(h, 1) ** 2)))
    tr = cKDTree(Pd / R)
    dist, idx = tr.query(Pg / R, k=nmax, distance_upper_bound=1.5)
    est = np.full(len(Pg), np.nan)
    for i in range(len(Pg)):
        ok_ = np.isfinite(dist[i])
        if ok_.sum() < nmin:
            continue
        j = idx[i][ok_]; Q = Pd[j] / R
        H = np.linalg.norm(Q[:, None] - Q[None], axis=2)
        k = len(j)
        A = np.ones((k + 1, k + 1)); A[:k, :k] = cov(H) + np.eye(k) * 1e-6 * (c0 + c1); A[k, k] = 0
        rhs = np.ones(k + 1); rhs[:k] = cov(dist[i][ok_])
        try:
            w = np.linalg.solve(A, rhs)[:k]
        except np.linalg.LinAlgError:
            continue
        if np.abs(w).max() > 10:                     # ill-conditioned (near-duplicate samples): fall back to inverse distance
            w = 1 / np.maximum(dist[i][ok_], 1e-3) ** 2; w = w / w.sum()
        est[i] = np.clip(w @ xd[j], xd[j].min(), xd[j].max())
    return est


rows = []
for dom in DOMS:
    zf = np.load(OUT / f'contacts_{dom}_masks.npz')
    names = list(zf['names']); ncell = int(zf['n_cells'])
    gs, gd, gn = zf['gs'], zf['gd'], zf['gn']
    M = {nm: np.unpackbits(zf[f'm{i}'])[:ncell].astype(bool) for i, nm in enumerate(names)}
    union = np.zeros(ncell, bool)
    for nm, m in M.items():
        union |= m
    G = np.stack(np.meshgrid(gs, gd, gn, indexing='ij'), -1).reshape(-1, 3)
    Pg = G[union]
    x0, x1 = DOM[dom]
    md = ms[(ms.X >= x0 - 100) & (ms.X < x1 + 100)]                     # neighbours may come from just across the domain edge
    Pd = md[['s', 'z', 'n']].to_numpy()
    est = {v: ok(v, Pd, md[v].to_numpy(), Pg) for v in VARS}
    # cells kriging cannot reach (no MS composite within 1.5 ranges): domain mean, flagged
    unreached = np.isnan(est['Zn_pct'])
    for v in VARS:
        est[v] = np.where(np.isnan(est[v]), md[v].mean(), est[v])
    np.savez_compressed(OUT / f'gt_cells_{dom}.npz', union=np.packbits(union), **{v: est[v].astype(np.float32) for v in VARS},
                        unreached=unreached)
    print(f'{dom}: {union.sum()} cells kriged, {unreached.sum()} beyond reach', flush=True)
    for nm, m in M.items():
        sel = m[union]
        constr, variant = nm.split('|')
        for dfac in (1.0, 0.95):
            t = CELL * np.clip(est['den'][sel], 2.6, 4.6) * dfac
            for cut in CUTS:
                k = est['Zn_pct'][sel] >= cut
                T = t[k].sum()
                rows.append(dict(domain=dom, construction=constr, variant=variant, density_factor=dfac, zn_cutoff=cut,
                                 volume_Mm3=round(float(k.sum() * CELL / 1e6), 4), tonnes_Mt=round(float(T / 1e6), 4),
                                 Zn_pct=round(float((t[k] * est['Zn_pct'][sel][k]).sum() / T), 3) if T > 0 else None,
                                 Pb_pct=round(float((t[k] * est['Pb_pct'][sel][k]).sum() / T), 3) if T > 0 else None,
                                 Cu_pct=round(float((t[k] * est['Cu_pct'][sel][k]).sum() / T), 3) if T > 0 else None,
                                 Zn_kt=round(float((t[k] * est['Zn_pct'][sel][k]).sum() / 100 / 1e3), 2),
                                 share_unreached=round(float(unreached[sel][k].mean()), 3) if k.any() else None,
                                 tonnes_reached_Mt=round(float(t[k & ~unreached[sel]].sum() / 1e6), 4),
                                 Zn_pct_reached=round(float((t[k & ~unreached[sel]] * est['Zn_pct'][sel][k & ~unreached[sel]]).sum()
                                                            / max(t[k & ~unreached[sel]].sum(), 1e-9)), 3) if (k & ~unreached[sel]).any() else None))
pd.DataFrame(rows).to_csv(OUT / 'gt_table.csv', index=False)
print('wrote', OUT / 'gt_table.csv')
