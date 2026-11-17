"""Step 2: geometric sensitivity of the MS and OH shells for one lens domain.

usage: python model_sensitivity.py <domain>

Refits the reference model (gen_cauchy indicator, unfolded frame, ranges 80/80/10 m) under
settings a modeller could reasonably have chosen, and reads each fit at three thresholds:
  ranges x0.75 and x1.25 (all axes), across-horizon range 7.5 and 12.5 m,
  in-plane 2:1 elongation along strike, down dip, or plunging +/-45 deg,
  dip +/-10 deg (rotation in the elevation / across plane),
  indicator threshold 0.3 / 0.5 / 0.7 on every fit.
The indicator is fitted as plain 0/1, so far from data the field falls to 0 (outside).
All holes are used in every variant: this is the spread of defensible geometries, not a
hold-out test.  Writes out/sens_<domain>.json (volumes, thicknesses) and
out/sens_<domain>_freq.npz (for each grid node, how many of the variant x threshold
combinations put it inside the shell).
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import deeprbf_recovered as d
from unfold import Unfold

OUT = Path(__file__).with_name('out')
DOM = {'FarWest': (369500, 370500), 'West': (370500, 372050), 'East': (372050, 373250), 'FarEast': (373250, 374300)}
print('DeepRBF', d.__version__, d.__file__, flush=True)
name = sys.argv[1]
x0, x1 = DOM[name]
R0 = np.array([80.0, 80.0, 10.0])
ELONG = np.array([80.0 * 2 ** 0.5, 80.0 / 2 ** 0.5, 10.0])
THR = (0.3, 0.5, 0.7)


def rot(axis, deg):
    """Rotation of local coords (s, z, n). axis 'n' = plunge (s-z plane), axis 's' = dip (z-n plane)."""
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    if axis == 'n':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


VARIANTS = [
    ('reference', R0, np.eye(3)),
    ('ranges_x0.75', R0 * 0.75, np.eye(3)),
    ('ranges_x1.25', R0 * 1.25, np.eye(3)),
    ('across_7.5m', np.array([80, 80, 7.5]), np.eye(3)),
    ('across_12.5m', np.array([80, 80, 12.5]), np.eye(3)),
    # in-plane elongation 2:1 (same area as 80 x 80): long axis along strike, down dip, or plunging +/-45 deg
    ('elong_strike', ELONG, np.eye(3)),
    ('elong_dip', ELONG[[1, 0, 2]], np.eye(3)),
    ('elong_plunge+45', ELONG, rot('n', 45)),
    ('elong_plunge-45', ELONG, rot('n', -45)),
    ('dip_+10', R0, rot('s', 10)),
    ('dip_-10', R0, rot('s', -10)),
]

comp = pd.read_csv(OUT / 'composites.csv')
c = comp[(comp.X >= x0) & (comp.X < x1) & comp.form.notna()].copy()
U = Unfold(comp)
z0 = float(comp.Z.mean())


def to_local(X):
    S = U(X); S[:, 1] -= z0
    return S


L = to_local(c[['X', 'Y', 'Z']].to_numpy())
far = np.abs(L[:, 2]) > 40
keep = ~far | (np.arange(len(c)) % 4 == 0)
c, L = c[keep], L[keep]

lo, hi = L.min(0), L.max(0)
gs = np.arange(lo[0], hi[0], 10.0); gd = np.arange(lo[1], hi[1], 10.0); gn = np.arange(-60, 60.01, 2.5)
G = np.stack(np.meshgrid(gs, gd, gn, indexing='ij'), -1).reshape(-1, 3)
near = cKDTree(L[:, :2]).query(G[:, :2], distance_upper_bound=60)[0] < np.inf
Gn = G[near]
shape = (len(gs), len(gd), len(gn)); cell = 10 * 10 * 2.5

rows = []
freq = {t: np.zeros(len(G), np.int16) for t in ('MS', 'OH')}
ref_inside = {}
for vname, R, Q in VARIANTS:
    F = (L @ Q.T) / R; FG = (Gn @ Q.T) / R
    for tgt, col in [('MS', 'f_ms'), ('OH', 'f_OH')]:
        y = (c[col].to_numpy() >= 0.5).astype(float)
        m = d.NativeExactRBFImplicitModel('gen_cauchy', 1.0, 0.05)
        m.set_drift('none')
        t0 = time.perf_counter(); m.fit(F.tolist(), y.tolist()); fit_s = time.perf_counter() - t0
        it = m.iterative_info() or {}
        acc = float(((np.asarray(m.predict(F.tolist())) >= 0.5) == (y >= 0.5)).mean())
        v = np.zeros(len(G)); v[near] = np.asarray(m.predict(FG.tolist()))
        for thr in THR:
            inside = v >= thr
            freq[tgt] += inside
            th = inside.reshape(shape).sum(2) * 2.5
            r = dict(variant=vname, target=tgt, threshold=thr, volume_Mm3=round(float(inside.sum() * cell / 1e6), 4),
                     thickness_p50_p90=[float(np.percentile(th[th > 0], q)) for q in (50, 90)] if (th > 0).any() else None,
                     fit_s=round(fit_s, 1), honour_accuracy=round(acc, 4),
                     hmatrix_fallback=it.get('hmatrix_fallback'), sampled_rel_residual=it.get('sampled_rel_residual'))
            if vname == 'reference' and thr == 0.5:
                ref_inside[tgt] = inside
            if tgt in ref_inside:
                a, b = ref_inside[tgt], inside
                r['overlap_with_reference_jaccard'] = round(float((a & b).sum() / max((a | b).sum(), 1)), 4)
            rows.append(r)
        print(f'{vname:14s} {tgt}  fit {fit_s:5.1f}s  acc {acc:.4f}  vol@0.5 {rows[-2]["volume_Mm3"]}', flush=True)

n_comb = len(VARIANTS) * len(THR)
json.dump(dict(domain=name, n_points=int(len(c)), combinations=n_comb, rows=rows), open(OUT / f'sens_{name}.json', 'w'), indent=1)
np.savez_compressed(OUT / f'sens_{name}_freq.npz', gs=gs, gd=gd, gn=gn, z0=z0, n_comb=n_comb,
                    **{t: f.reshape(shape) for t, f in freq.items()})
print('wrote', OUT / f'sens_{name}.json')
