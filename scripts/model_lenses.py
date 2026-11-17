"""Fourth construction: correlated lenses. Logged MS intercepts are linked into separate lenses, each with its own
width and centre surfaces, so stacked lenses stay apart (revision 2).

usage: python model_lenses.py <domain>
Run after model_contacts.py; appends 'lenses|<variant>' masks to out/contacts_<domain>_masks.npz.

1. Intercepts: in each hole, coded MS composites separated by more than 2.5 m of non-MS down-hole start a new
   intercept (the definition used for the count of separate intercepts). Width = sum of thickness factor x length.
2. Correlation: intercepts of different holes within 50 m (in s, z) are linked when their centres across the horizon
   agree within 3 m + 0.06 m per metre of distance; most consistent first; a lens takes at most one intercept per
   hole, and every link must agree with the lens's local centre. Every intercept is used (single-hole lenses too).
3. Each lens: centre = least-squares plane (>= 3 holes) or mean, plus an interpolated residual; width interpolated
   from its intercepts plus zero widths at holes with logged codes that cross its position without MS within 3 m.
   Kernel and ranges as the other constructions (generalised Cauchy, 80 m in plane); evaluated within 60 m of the
   lens's intercepts. A cell is MS where |n - centre| <= w / 2 and w >= w_min (1 m), as for the contact surfaces.
Variants: reference; contacts 1 m out / in (each intercept's width +/- 2 m); ranges x0.75 / x1.25; without the
withheld holes (correlation redone without them).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent
src = open(HERE / 'model_contacts.py', encoding='utf-8').read()
exec(src[:src.index('# ------------------------------------------------------------------ indicator')])
from scipy.spatial import cKDTree

R_LINK, N_TOL, N_SLOPE, GAP, NEAR, W_MIN, ZERO_TOL = 50.0, 3.0, 0.06, 2.5, 60.0, 1.0, 3.0
cc = c[coded].copy()


def intercepts(cd):
    rows = []
    for h, g in cd.groupby('hole'):
        m = g[g.ms]
        if not len(m): continue
        ids = np.cumsum(np.concatenate([[False], (m.f.values[1:] - m.t.values[:-1]) > GAP]))
        for k, q in m.groupby(ids):
            w = (q.t - q.f).to_numpy()
            rows.append(dict(hole=str(h), s=np.average(q.s, weights=w), z=np.average(q.z, weights=w),
                             n=np.average(q.n, weights=w), tw=float((q.across * w).sum()), frozen=bool(q.frozen.iloc[0])))
    return pd.DataFrame(rows)


def correlate(I):
    P2 = I[['s', 'z']].to_numpy(); nn = I.n.to_numpy(); hole = I.hole.to_numpy(); N = len(I)
    pairs = cKDTree(P2).query_pairs(R_LINK, output_type='ndarray')
    pairs = pairs[hole[pairs[:, 0]] != hole[pairs[:, 1]]]
    dist = np.hypot(*(P2[pairs[:, 0]] - P2[pairs[:, 1]]).T)
    cost = np.abs(nn[pairs[:, 0]] - nn[pairs[:, 1]]) / (N_TOL + N_SLOPE * dist)
    k = cost <= 1.0; pairs, dist, cost = pairs[k], dist[k], cost[k]
    parent = np.arange(N); mem = {i: [i] for i in range(N)}; hol = {i: {hole[i]} for i in range(N)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def local_ok(A, B):
        for X_, Y_ in ((A, B), (B, A)):
            if len(X_) < 2: continue
            tr = cKDTree(P2[X_]); Xa = np.array(X_)
            for j in Y_:
                idx = tr.query_ball_point(P2[j], R_LINK)
                if not idx: continue
                loc = np.median(nn[Xa[idx]]); dd = np.hypot(*(P2[Xa[idx]] - P2[j]).T).mean()
                if abs(nn[j] - loc) > N_TOL + N_SLOPE * dd + 1.0: return False
        return True

    for e in np.lexsort((dist, cost)):
        a, b = find(pairs[e, 0]), find(pairs[e, 1])
        if a == b or hol[a] & hol[b] or not local_ok(mem[a], mem[b]): continue
        if len(mem[a]) < len(mem[b]): a, b = b, a
        parent[b] = a; mem[a] += mem.pop(b); hol[a] |= hol.pop(b)
    return np.array([find(i) for i in range(N)])


def rbf2(P, v, R):
    m = d.NativeExactRBFImplicitModel('gen_cauchy', 1.0, 0.05); m.set_drift('none')
    Q = P / R; m.fit(np.column_stack([Q, np.zeros(len(Q))]).tolist(), np.asarray(v, float).tolist()); return m


def pr2(m, P, R):
    Q = P / R; return np.asarray(m.predict(np.column_stack([Q, np.zeros(len(Q))]).tolist()))


G2 = np.stack(np.meshgrid(gs, gd, indexing='ij'), -1).reshape(-1, 2)
tree_cc = cKDTree(cc[['s', 'z']].to_numpy())


def build(I, R=80.0, dw=0.0, data=cc):
    lab = correlate(I); I = I.assign(lens=lab)
    inside = np.zeros(shape, bool); n_lens = 0
    tree_d = cKDTree(data[['s', 'z']].to_numpy())
    for lid, M in I.groupby('lens'):
        Pm = M[['s', 'z']].to_numpy(); n_lens += 1
        if len(M) >= 3:
            A = np.column_stack([np.ones(len(M)), Pm - Pm.mean(0)]); coef = np.linalg.lstsq(A, M.n.to_numpy(), rcond=None)[0]
            trend = lambda Q, coef=coef, mu=Pm.mean(0): np.column_stack([np.ones(len(Q)), Q - mu]) @ coef
        else:
            mu_n = float(M.n.mean()); trend = lambda Q, mu_n=mu_n: np.full(len(Q), mu_n)
        mc = rbf2(Pm, M.n.to_numpy() - trend(Pm), R) if len(M) > 1 else None
        cen = (lambda Q: trend(Q) + pr2(mc, Q, R)) if mc is not None else trend
        # zeros: holes near the lens, logged at its position, without MS within ZERO_TOL across
        idx = sorted(set(j for p in Pm for j in tree_d.query_ball_point(p, NEAR)))
        cand = data.iloc[idx]; cand = cand[~cand.hole.astype(str).isin(set(M.hole))]
        zp = []
        if len(cand):
            dn_ = np.abs(cand.n.to_numpy() - cen(cand[['s', 'z']].to_numpy())); cand = cand.assign(dn=dn_)
            for h, g in cand[cand.dn <= ZERO_TOL].groupby('hole'):
                if g.ms.any(): continue
                j = g.dn.values.argmin(); zp.append((g.s.iloc[j], g.z.iloc[j]))
        Zp = np.array(zp) if zp else np.zeros((0, 2))
        w = np.where(M.tw > 0, np.clip(M.tw + dw, 0, None), 0.0)
        Pw = np.vstack([Pm, Zp]); vw = np.concatenate([w, np.zeros(len(Zp))])
        sel = cKDTree(Pm).query(G2, distance_upper_bound=NEAR)[0] < np.inf
        if not sel.any(): continue
        Q = G2[sel]
        W = pr2(rbf2(Pw, vw, R), Q, R) if len(Pw) > 1 else np.full(len(Q), vw[0])
        C = cen(Q)
        ii = np.nonzero(sel)[0]; i_s, i_d = np.unravel_index(ii, shape[:2])
        on = W >= W_MIN
        cell = (np.abs(gn[None, :] - C[on, None]) <= W[on, None] / 2)
        inside[i_s[on], i_d[on], :] |= cell
    return inside.reshape(-1), n_lens


I_all = intercepts(cc)
print(name, 'MS intercepts', len(I_all), 'in', I_all.hole.nunique(), 'holes', flush=True)
for nm, kw in (('reference', {}), ('contacts_out_1m', dict(dw=2.0)), ('contacts_in_1m', dict(dw=-2.0)),
               ('ranges_x0.75', dict(R=60.0)), ('ranges_x1.25', dict(R=100.0))):
    m, nl = build(I_all, **kw); record('lenses', nm, m, dict(n_lenses=nl))
I_nf = intercepts(cc[~cc.frozen]); m, nl = build(I_nf, data=cc[~cc.frozen]); record('lenses', 'without_frozen', m, dict(n_lenses=nl))

# append to the masks file and write a summary
zf = dict(np.load(OUT / f'contacts_{name}_masks.npz'))
names = [x for x in zf['names'] if not str(x).startswith('lenses|')]
keep = {f'm{i}': zf[f'm{list(zf["names"]).index(x)}'] for i, x in enumerate(names)}
for k, v in masks.items():
    keep[f'm{len(names)}'] = v; names.append(k)
np.savez_compressed(OUT / f'contacts_{name}_masks.npz', gs=gs, gd=gd, gn=gn, z0=z0, n_cells=len(G), names=np.array(names), **keep)
ref = {k: np.unpackbits(keep[f'm{names.index(k)}'])[:len(G)].astype(bool) for k in ('indicator|threshold_0.5', 'signed|reference', 'surfaces|reference', 'lenses|reference')}
jac = {f'{a.split("|")[0]}-{b.split("|")[0]}': round(float((ref[a] & ref[b]).sum() / max((ref[a] | ref[b]).sum(), 1)), 3)
       for i, a in enumerate(ref) for b in list(ref)[i + 1:]}
json.dump(dict(domain=name, intercepts=int(len(I_all)), rows=rows, jaccard=jac), open(OUT / f'lenses_{name}.json', 'w'), indent=1)
print('jaccard', jac, flush=True)
