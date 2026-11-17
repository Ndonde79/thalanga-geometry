"""First-pass reference geometry for one lens domain: MS and OH indicator shells with DeepRBF (gen_cauchy, H-matrix).

usage: python model_domain.py <domain> [ranges_strike ranges_dip ranges_normal]
Frame from a plane fitted to the domain's OH contacts; plunge ignored in this first pass.
Writes out/<domain>_*.npz/png/json.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

import deeprbf_recovered as d
from unfold import Unfold

OUT = Path(__file__).with_name('out')
DOM = {'FarWest': (369500, 370500), 'West': (370500, 372050), 'East': (372050, 373250), 'FarEast': (373250, 374300)}
print('DeepRBF', d.__version__, d.__file__, flush=True)
name = sys.argv[1]
R = np.array([float(v) for v in sys.argv[2:5]]) if len(sys.argv) >= 5 else np.array([80.0, 80.0, 10.0])
x0, x1 = DOM[name]

comp = pd.read_csv(OUT / 'composites.csv')
ct = pd.read_csv(OUT / 'contacts.csv')
c = comp[(comp.X >= x0) & (comp.X < x1) & comp.form.notna()].copy()
cc = ct[(ct.X >= x0) & (ct.X < x1) & ct.kind.isin(['OH_top', 'OH_base'])]

# ------------------------------------------------------------------ unfolded frame (s along trace, z, n across)
U = Unfold(comp)
o = np.zeros(3); B = np.eye(3); strike = float('nan'); dip = float('nan')
z0 = float(comp.Z.mean())


def to_local(X):
    S = U(X); S[:, 1] -= z0
    return S


def frame(X):
    return to_local(X) / R


X = c[['X', 'Y', 'Z']].to_numpy(); L = to_local(X)
# thin far from the ore horizon: keep all within 40 m of the plane, 1 in 4 beyond
far = np.abs(L[:, 2]) > 40
keep = ~far | (np.arange(len(c)) % 4 == 0)
c, X, L = c[keep], X[keep], L[keep]
F = L / R
info = dict(domain=name, n_points=int(len(c)), frame='unfolded (s along ore-horizon trace, z, n across)', ranges_m=R.tolist(),
            ms_fraction=round(float((c.f_ms >= 0.5).mean()), 4), oh_fraction=round(float(c.f_OH.mean()), 4))

M = d.NativeExactRBFImplicitModel
models = {}
for tgt, col in [('MS', 'f_ms'), ('OH', 'f_OH')]:
    y = (c[col].to_numpy() >= 0.5).astype(float)
    m = M('gen_cauchy', 1.0, 0.05)
    m.set_drift('none')
    t = time.perf_counter()
    m.fit(F.tolist(), y.tolist())   # plain 0/1 indicator: far from data the field falls to 0 (outside)
    info[f'{tgt}_fit_s'] = round(time.perf_counter() - t, 1)
    info[f'{tgt}_backend'] = m.solver_backend()
    it = m.iterative_info()
    if it:
        info[f'{tgt}_iter'] = {k: round(float(v), 6) for k, v in it.items() if k in ('products', 'sampled_rel_residual', 'hmatrix', 'hmatrix_fallback')}
    yhat = np.asarray(m.predict(F.tolist()))
    info[f'{tgt}_honour_accuracy'] = round(float(((yhat >= 0.5) == (y >= 0.5)).mean()), 4)
    models[tgt] = m
    print(tgt, info[f'{tgt}_fit_s'], 's', info.get(f'{tgt}_iter'), flush=True)

# ------------------------------------------------------------------ evaluate on a frame-aligned grid (10 x 10 x 2.5 m)
lo, hi = L.min(0), L.max(0)
gs = np.arange(lo[0], hi[0], 10.0); gd = np.arange(lo[1], hi[1], 10.0); gn = np.arange(-60, 60.01, 2.5)
G = np.stack(np.meshgrid(gs, gd, gn, indexing='ij'), -1).reshape(-1, 3)
# only nodes within 60 m (in strike/dip) of a composite: skip empty space
from scipy.spatial import cKDTree
near = cKDTree(L[:, :2]).query(G[:, :2], distance_upper_bound=60)[0] < np.inf
Gn = G[near]
t = time.perf_counter()
val = {}
for tgt, m in models.items():
    val[tgt] = np.full(len(G), -0.5); val[tgt][near] = np.asarray(m.predict((Gn / R).tolist())) - 0.5   # stored as indicator - 0.5 (shell at 0)
info['grid_nodes_evaluated'] = int(near.sum()); info['grid_predict_s'] = round(time.perf_counter() - t, 1)
cell = 10 * 10 * 2.5
for tgt in models:
    inside = val[tgt] >= 0
    info[f'{tgt}_volume_Mm3'] = round(float(inside.sum() * cell / 1e6), 3)
    thick = inside.reshape(len(gs), len(gd), len(gn)).sum(2) * 2.5
    info[f'{tgt}_true_thickness_m_p50_p90'] = [float(np.percentile(thick[thick > 0], q)) for q in (50, 90)] if (thick > 0).any() else None
np.savez_compressed(OUT / f'{name}_grid.npz', gs=gs, gd=gd, gn=gn, o=o, B=B, R=R, z0=z0, frame='unfold', **{k: v.reshape(len(gs), len(gd), len(gn)).astype(np.float32) for k, v in val.items()})
json.dump(info, open(OUT / f'{name}_info.json', 'w'), indent=1)
print(json.dumps(info, indent=1))
