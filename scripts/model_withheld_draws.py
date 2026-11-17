"""Withheld-hole draws: is the 'data-limited beyond 25 m' result specific to one withheld set?

usage: python model_withheld_draws.py <domain>
Five further withheld sets are drawn like the original one: 50 m x 50 m collar blocks are taken in random order
(seeds 1-5) until at least 274 logged holes are withheld. For each draw the indicator, signed-distance and
contact-surface references are refitted without those holes. Writes out/draws_<domain>.csv.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

import deeprbf_recovered as d
from unfold import Unfold
from scipy.spatial import cKDTree

OUT = Path(__file__).with_name('out')
DOM = {'FarWest': (369500, 370500), 'West': (370500, 372050), 'East': (372050, 373250), 'FarEast': (373250, 374300)}
print('DeepRBF', d.__version__, d.__file__, flush=True)
name = sys.argv[1]
x0, x1 = DOM[name]
CAP = 20.0
R0 = np.array([80.0, 80.0, 10.0])
ELONG = np.array([80.0 * 2 ** 0.5, 80.0 / 2 ** 0.5, 10.0])
frozen = set(pd.read_csv(Path(__file__).with_name('frozen_holes.csv'), comment='#').hole_id.astype(str))


def rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    if axis == 'n':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot2(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def rbf(P, v):
    P = np.asarray(P, float)
    if P.shape[1] == 2:                                  # 2D surfaces: the engine takes 3D points
        P = np.column_stack([P, np.zeros(len(P))])
    m = d.NativeExactRBFImplicitModel('gen_cauchy', 1.0, 0.05)
    m.set_drift('none')
    t0 = time.perf_counter(); m.fit(P.tolist(), np.asarray(v, float).tolist())
    return m, time.perf_counter() - t0


comp_all = pd.read_csv(OUT / 'composites.csv')
ct_all = pd.read_csv(OUT / 'contacts.csv')
U = Unfold(comp_all)
z0 = float(comp_all.Z.mean())


def to_local(X):
    S = U(X); S[:, 1] -= z0
    return S


# ------------------------------------------------------------------ composites of the domain, in the unfolded frame
c = comp_all[(comp_all.X >= x0) & (comp_all.X < x1)].copy()
L = to_local(c[['X', 'Y', 'Z']].to_numpy())
c['s'], c['z'], c['n'] = L[:, 0], L[:, 1], L[:, 2]
c = c.sort_values(['hole', 'f']).reset_index(drop=True)
c['ms'] = c.f_ms >= 0.5
# thickness factor |u . N| (drilling direction against the local lens normal from the MS contacts), floored at 0.25
from truewidth import true_across
_ce = ct_all[ct_all.kind == 'MS_edge']
c['across'] = true_across(c, to_local(_ce[['X', 'Y', 'Z']].to_numpy()))
c['frozen'] = c.hole.astype(str).isin(frozen)
coded = c.form.notna().to_numpy()

# signed distance (across the horizon) to the nearest MS contact of the same hole
ce = ct_all[(ct_all.kind == 'MS_edge') & (ct_all.X >= x0) & (ct_all.X < x1)].copy()
LE = to_local(ce[['X', 'Y', 'Z']].to_numpy()); ce['s'], ce['z'], ce['n'] = LE[:, 0], LE[:, 1], LE[:, 2]
ce['frozen'] = ce.hole.astype(str).isin(frozen)
mid = ((c.f + c.t) / 2).to_numpy()
sd = np.full(len(c), -CAP)
edges = {h: g.depth.to_numpy() for h, g in ce.groupby('hole')}
for h, idx in c.groupby('hole').indices.items():
    if h in edges:
        dd = np.abs(mid[idx][:, None] - edges[h][None, :]).min(1) * c.across.to_numpy()[idx]
        sd[idx] = np.where(c.ms.to_numpy()[idx], dd, -dd)
    else:
        sd[idx] = np.where(c.ms.to_numpy()[idx], CAP, -CAP)
c['sd'] = np.clip(sd, -CAP, CAP)

# same point set as the indicator fits: coded composites, thinned beyond 40 m from the horizon
keep = coded & ((np.abs(c.n.to_numpy()) <= 40) | (np.arange(len(c)) % 4 == 0))
P3 = c.loc[keep, ['s', 'z', 'n']].to_numpy()
pin = c.loc[keep]

# ------------------------------------------------------------------ grid (same as model_unfold.py)
Lc = P3
lo, hi = Lc.min(0), Lc.max(0)
gs = np.arange(lo[0], hi[0], 10.0); gd = np.arange(lo[1], hi[1], 10.0); gn = np.arange(-60, 60.01, 2.5)
G = np.stack(np.meshgrid(gs, gd, gn, indexing='ij'), -1).reshape(-1, 3)
near = cKDTree(Lc[:, :2]).query(G[:, :2], distance_upper_bound=60)[0] < np.inf
Gn = G[near]
shape = (len(gs), len(gd), len(gn)); cell = 10 * 10 * 2.5
masks, rows = {}, []


def record(constr, variant, inside, extra=None):
    th = inside.reshape(shape).sum(2) * 2.5
    r = dict(construction=constr, variant=variant, volume_Mm3=round(float(inside.sum() * cell / 1e6), 4),
             thickness_p50_p90=[float(np.percentile(th[th > 0], q)) for q in (50, 90)] if (th > 0).any() else None)
    r.update(extra or {})
    masks[f'{constr}|{variant}'] = np.packbits(inside)
    rows.append(r)
    print(f'{constr:9s} {variant:16s} vol {r["volume_Mm3"]:.4f}  {extra or ""}', flush=True)


def field3(P, v, R, Q=np.eye(3)):
    m, fs = rbf((P @ Q.T) / R, v)
    out = np.zeros(len(G)); out[near] = np.asarray(m.predict(((Gn @ Q.T) / R).tolist()))
    it = m.iterative_info() or {}
    return out, dict(fit_s=round(fs, 1), hmatrix_fallback=it.get('hmatrix_fallback'), sampled_rel_residual=it.get('sampled_rel_residual'))



# ------------------------------------------------------------------ the draws (global, so every domain sees the same sets)
hl = pd.read_csv(OUT / 'holes.csv')
hl = hl[hl.HOLE_ID.isin(set(comp_all.hole))]
blk = (np.floor(hl.X_MGA / 50).astype(int).astype(str) + '_' + np.floor(hl.Y_MGA / 50).astype(int).astype(str)).to_numpy()
ub = np.unique(blk)
DRAWS = {'original': frozen}
for seed in range(1, 6):
    order = np.random.default_rng(seed).permutation(ub); chosen = set()
    for b in order:
        chosen |= set(hl.HOLE_ID[blk == b].astype(str))
        if len(chosen) >= 274:
            break
    DRAWS[f'draw{seed}'] = chosen

yi = pin.ms.to_numpy().astype(float)
PE = ce[['s', 'z', 'n']].to_numpy()
Psd = np.vstack([P3, PE]); vsd = np.concatenate([pin.sd.to_numpy() + CAP, np.full(len(PE), CAP)])
hole_sd = np.concatenate([pin.hole.astype(str).to_numpy(), ce.hole.astype(str).to_numpy()])
# ------------------------------------------------------------------ contact surfaces: thickness and centre per hole
hrows = []
for h, g in c[coded].groupby('hole'):
    g = g.sort_values('f')
    m = g[g.ms]
    if len(m):
        t = float((m.across * (m.t - m.f)).sum())
        w = (m.t - m.f).to_numpy()
        hrows.append(dict(hole=h, s=np.average(m.s, weights=w), z=np.average(m.z, weights=w), n=np.average(m.n, weights=w),
                          t=t, nint=int(((m.f.diff() > 2.5).sum()) + 1)))
    else:
        k = np.abs(g.n.to_numpy()).argmin()
        if abs(g.n.iloc[k]) <= 15:                       # crosses the horizon without MS
            hrows.append(dict(hole=h, s=g.s.iloc[k], z=g.z.iloc[k], n=g.n.iloc[k], t=0.0, nint=0))
H = pd.DataFrame(hrows); H['frozen'] = H.hole.astype(str).isin(frozen)
H.to_csv(OUT / f'contacts_{name}_holes.csv', index=False)
G2 = np.stack(np.meshgrid(gs, gd, indexing='ij'), -1).reshape(-1, 2)
# surfaces are evaluated only within 60 m (along s, z) of a hole that crosses the horizon
near2 = cKDTree(H[['s', 'z']].to_numpy()).query(G2, distance_upper_bound=60)[0] < np.inf
NN = np.broadcast_to(gn, shape)


def surfaces(Hh, R2=np.array([80.0, 80.0]), Q2=np.eye(2), dt=0.0, tmins=(1.0,)):
    P = (Hh[['s', 'z']].to_numpy() @ Q2.T) / R2
    tt = np.where(Hh.t > 0, np.clip(Hh.t + dt, 0, None), 0.0)
    mt, f1 = rbf(P, tt); mc, f2 = rbf(P, Hh.n.to_numpy())
    Pg = (G2[near2] @ Q2.T) / R2
    T = np.zeros(len(G2)); C = np.zeros(len(G2))
    Pg = np.column_stack([Pg, np.zeros(len(Pg))])
    T[near2] = np.asarray(mt.predict(Pg.tolist())); C[near2] = np.asarray(mc.predict(Pg.tolist()))
    T = T.reshape(shape[:2]); C = C.reshape(shape[:2])
    out = {}
    for tm in tmins:
        Tm = np.where(T >= tm, T, 0.0)
        out[tm] = ((Tm[:, :, None] > 0) & (np.abs(NN - C[:, :, None]) <= Tm[:, :, None] / 2)).reshape(-1)
    return out, dict(fit_s=round(f1 + f2, 2), n_holes=int(len(Hh)), n_ms_holes=int((Hh.t > 0).sum()))



G2c = np.stack(np.meshgrid(gs, gd, indexing='ij'), -1)
ref = {}
v, _ = field3(P3, yi, R0); ref['indicator'] = v >= 0.5
v, _ = field3(Psd, vsd, R0); ref['signed'] = v >= CAP
ref['surfaces'] = surfaces(H)[0][1.0]
out = []
for dn, hs in DRAWS.items():
    w_i = ~pin.hole.astype(str).isin(hs).to_numpy(); w_s = ~np.isin(hole_sd, list(hs)); w_h = ~H.hole.astype(str).isin(hs).to_numpy()
    alt = {}
    v, _ = field3(P3[w_i], yi[w_i], R0); alt['indicator'] = v >= 0.5
    v, _ = field3(Psd[w_s], vsd[w_s], R0); alt['signed'] = v >= CAP
    alt['surfaces'] = surfaces(H[w_h])[0][1.0]
    dev = H[w_h]; d_dev = cKDTree(dev[['s', 'z']].to_numpy()).query(G2c.reshape(-1, 2))[0].reshape(G2c.shape[:2]) if len(dev) else np.full(G2c.shape[:2], np.inf)
    for con in ref:
        a = ref[con].reshape(shape); b = alt[con].reshape(shape)
        ca, cb = a.any(2), b.any(2); ch = (a != b).any(2); any_ = ca | cb
        r = dict(domain=name, draw=dn, construction=con, n_withheld_crossing=int((~w_h).sum()),
                 all_Mm3=round(a.sum() * cell / 1e6, 4), without_Mm3=round(b.sum() * cell / 1e6, 4),
                 jaccard=round(float((a & b).sum() / max((a | b).sum(), 1)), 3))
        for lo, hi in ((0, 25), (25, 50), (50, 1e9)):
            m = any_ & (d_dev >= lo) & (d_dev < hi)
            r[f'cols_{lo}'] = int(m.sum()); r[f'changed_{lo}'] = int((ch & m).sum())
        out.append(r)
    print(dn, [(o['construction'], o['without_Mm3']) for o in out[-3:]], flush=True)
pd.DataFrame(out).to_csv(OUT / f'draws_{name}.csv', index=False)
print('wrote', OUT / f'draws_{name}.csv', flush=True)
