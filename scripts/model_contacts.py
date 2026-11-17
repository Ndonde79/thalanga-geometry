"""Step 3 (+ step 4 fits): massive sulphide (MS) from contacts, compared with the indicator, for one lens domain.

usage: python model_contacts.py <domain>

Three constructions of MS on the same unfolded grid (s along trace, z, n across; 10 x 10 x 2.5 m):
  indicator  - plain 0/1 MS indicator (as step 2), read at a threshold.
  signed     - signed distance to the nearest logged MS contact in the same hole (+ inside, - outside), measured
               across the horizon, capped at +/-CAP m and offset by +CAP so that far from data the field falls to 0
               (outside). The MS contact points themselves are data at the boundary value. Shell = field >= CAP.
  surfaces   - per hole: MS true thickness t and centre n_c of the MS intercepts; holes that cross the horizon
               without MS give t = 0. t(s,z) and n_c(s,z) are interpolated in 2D; MS = |n - n_c| <= t/2 where t >= t_min.
Variants (all holes in every one, except 'without_frozen'):
  indicator: threshold 0.3/0.5/0.7 (reference fit), ranges x0.75 / x1.25, without_frozen
  signed:    reference, ranges x0.75 / x1.25, across range 7.5 / 12.5 m, 2:1 elongation (strike, dip, plunge +/-45),
             dip +/-10 deg, contacts moved 1 m out / in (shell level CAP-1 / CAP+1), without_frozen
  surfaces:  reference (t_min 1 m), t_min 0.5 / 2 m, ranges x0.75 / x1.25, 2:1 elongation (strike, dip, plunge +/-45),
             contacts moved 1 m out / in (t +/- 2 m per intercept), without_frozen
Writes out/contacts_<domain>.json and out/contacts_<domain>_masks.npz (every variant's MS mask, bit-packed).
"""
import json, sys, time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
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


# ------------------------------------------------------------------ indicator
yi = pin.ms.to_numpy().astype(float)
v, info = field3(P3, yi, R0)
for thr in (0.3, 0.5, 0.7):
    record('indicator', f'threshold_{thr}', v >= thr, info)
for nm, R in (('ranges_x0.75', R0 * 0.75), ('ranges_x1.25', R0 * 1.25)):
    v, info = field3(P3, yi, R); record('indicator', nm, v >= 0.5, info)
nf = ~pin.frozen.to_numpy()
v, info = field3(P3[nf], yi[nf], R0); record('indicator', 'without_frozen', v >= 0.5, info)

# ------------------------------------------------------------------ signed distance (+ contacts as boundary points)
PE = ce[['s', 'z', 'n']].to_numpy()
Psd = np.vstack([P3, PE]); vsd = np.concatenate([pin.sd.to_numpy() + CAP, np.full(len(PE), CAP)])
fr_sd = np.concatenate([pin.frozen.to_numpy(), ce.frozen.to_numpy()])
SDV = [('reference', R0, np.eye(3)), ('ranges_x0.75', R0 * 0.75, np.eye(3)), ('ranges_x1.25', R0 * 1.25, np.eye(3)),
       ('across_7.5m', np.array([80, 80, 7.5]), np.eye(3)), ('across_12.5m', np.array([80, 80, 12.5]), np.eye(3)),
       ('elong_strike', ELONG, np.eye(3)), ('elong_dip', ELONG[[1, 0, 2]], np.eye(3)),
       ('elong_plunge+45', ELONG, rot('n', 45)), ('elong_plunge-45', ELONG, rot('n', -45)),
       ('dip_+10', R0, rot('s', 10)), ('dip_-10', R0, rot('s', -10))]
for nm, R, Q in SDV:
    v, info = field3(Psd, vsd, R, Q)
    record('signed', nm, v >= CAP, info)
    if nm == 'reference':
        record('signed', 'contacts_out_1m', v >= CAP - 1, info)
        record('signed', 'contacts_in_1m', v >= CAP + 1, info)
v, info = field3(Psd[~fr_sd], vsd[~fr_sd], R0); record('signed', 'without_frozen', v >= CAP, info)

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


o, info = surfaces(H, tmins=(0.5, 1.0, 2.0))
record('surfaces', 'reference', o[1.0], info); record('surfaces', 'tmin_0.5m', o[0.5], info); record('surfaces', 'tmin_2m', o[2.0], info)
for nm, R2, Q2 in (('ranges_x0.75', np.array([60.0, 60.0]), np.eye(2)), ('ranges_x1.25', np.array([100.0, 100.0]), np.eye(2)),
                   ('elong_strike', ELONG[:2], np.eye(2)), ('elong_dip', ELONG[[1, 0]], np.eye(2)),
                   ('elong_plunge+45', ELONG[:2], rot2(45)), ('elong_plunge-45', ELONG[:2], rot2(-45))):
    o, info = surfaces(H, R2, Q2); record('surfaces', nm, o[1.0], info)
for nm, dt in (('contacts_out_1m', 2.0), ('contacts_in_1m', -2.0)):
    o, info = surfaces(H, dt=dt); record('surfaces', nm, o[1.0], info)
o, info = surfaces(H[~H.frozen]); record('surfaces', 'without_frozen', o[1.0], info)

# ------------------------------------------------------------------ overlap with each construction's reference
refs = {'indicator': 'indicator|threshold_0.5', 'signed': 'signed|reference', 'surfaces': 'surfaces|reference'}
for r in rows:
    a = np.unpackbits(masks[refs[r['construction']]])[:len(G)].astype(bool)
    b = np.unpackbits(masks[f"{r['construction']}|{r['variant']}"])[:len(G)].astype(bool)
    r['jaccard_vs_own_reference'] = round(float((a & b).sum() / max((a | b).sum(), 1)), 4)
a = np.unpackbits(masks['indicator|threshold_0.5'])[:len(G)].astype(bool)
cross = {}
for k in ('signed|reference', 'surfaces|reference'):
    b = np.unpackbits(masks[k])[:len(G)].astype(bool)
    cross[k] = round(float((a & b).sum() / max((a | b).sum(), 1)), 4)

json.dump(dict(domain=name, cap_m=CAP, n_points=int(len(P3)), n_contacts=int(len(PE)), n_frozen_points=int(pin.frozen.sum()),
               jaccard_reference_vs_indicator=cross, rows=rows), open(OUT / f'contacts_{name}.json', 'w'), indent=1)
np.savez_compressed(OUT / f'contacts_{name}_masks.npz', gs=gs, gd=gd, gn=gn, z0=z0, n_cells=len(G),
                    names=np.array(list(masks)), **{f'm{i}': v for i, v in enumerate(masks.values())})
print('wrote', OUT / f'contacts_{name}.json', flush=True)
