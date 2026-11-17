"""Re-evaluate the three reference MS constructions as continuous fields on a fine grid, for smooth implicit shells.

usage: python model_fine_ep.py <domain>

Fits exactly as model_contacts.py (same points, kernel, ranges, regularisation), then evaluates each reference field on a
fine frame grid: 2.5 m along s, 2.5 m along z, 0.5 m across the horizon (n), in the columns where any construction is
near the shell. Fields saved (float16):
  ind  - MS indicator field; shell at 0.5
  sd   - signed-distance field (offset by CAP); shell at CAP
  sur  - contact-surface field min(T - t_min, T/2 - |n - C|); shell at 0 (T = interpolated width, C = centre)
Also writes volumes: fine-cell counts (5 x 5 x 0.5 m) against the 10 x 10 x 2.5 m counts used in the paper.
Writes out/fine_<domain>.npz and out/fine_<domain>.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
src = open(HERE / 'model_contacts.py', encoding='utf-8').read()
# set-up from model_contacts.py: data, unfolded frame, composites, signed distances, point sets, coarse grid
exec(src[:src.index('# ------------------------------------------------------------------ indicator')])
# hole table for the contact surfaces (same code, without rewriting the csv)
blk = src[src.index('hrows = []'):src.index("H.to_csv(")]
exec(blk)
H = pd.DataFrame(hrows)

T_MIN = 1.0
t_all = time.perf_counter()
# ------------------------------------------------------------------ fits (reference settings)
m_ind, f1 = rbf(P3 / R0, pin.ms.to_numpy().astype(float))
PE = ce[['s', 'z', 'n']].to_numpy()
Psd = np.vstack([P3, PE]); vsd = np.concatenate([pin.sd.to_numpy() + CAP, np.full(len(PE), CAP)])
m_sd, f2 = rbf(Psd / R0, vsd)
R2 = np.array([80.0, 80.0])
PH = H[['s', 'z']].to_numpy() / R2
m_t, f3 = rbf(PH, np.where(H.t > 0, H.t, 0.0)); m_c, f4 = rbf(PH, H.n.to_numpy())
m_tp, _ = rbf(PH, np.where(H.t > 0, np.clip(H.t + 2.0, 0, None), 0.0)); m_tm, _ = rbf(PH, np.where(H.t > 0, np.clip(H.t - 2.0, 0, None), 0.0))
print(f'fits {f1:.1f} {f2:.1f} {f3 + f4:.1f} s', flush=True)


def pred(m, Q, chunk=400000):
    out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        out[i:i + chunk] = np.asarray(m.predict(Q[i:i + chunk].tolist()))
    return out


def pred2(m, Q2):
    return pred(m, np.column_stack([Q2, np.zeros(len(Q2))]))


# ------------------------------------------------------------------ coarse pass (paper grid) to find active columns
ci = np.zeros(len(G)); cs_ = np.zeros(len(G))
ci[near] = pred(m_ind, Gn / R0); cs_[near] = pred(m_sd, Gn / R0)
G2 = np.stack(np.meshgrid(gs, gd, indexing='ij'), -1).reshape(-1, 2)
near2 = cKDTree(H[['s', 'z']].to_numpy()).query(G2, distance_upper_bound=60)[0] < np.inf
Tc = np.zeros(len(G2)); Cc = np.zeros(len(G2))
Tc[near2] = pred2(m_t, G2[near2] / R2); Cc[near2] = pred2(m_c, G2[near2] / R2)
Tc = Tc.reshape(shape[:2]); Cc = Cc.reshape(shape[:2])
NN = np.broadcast_to(gn, shape)
coarse = dict(indicator=(ci >= 0.5), signed=(cs_ >= CAP),
              surfaces=((Tc[:, :, None] >= T_MIN) & (np.abs(NN - Cc[:, :, None]) <= Tc[:, :, None] / 2)).reshape(-1))
Tcp = np.zeros(len(G2)); Tcm = np.zeros(len(G2))
Tcp[near2] = pred2(m_tp, G2[near2] / R2); Tcm[near2] = pred2(m_tm, G2[near2] / R2)
Tcp = Tcp.reshape(shape[:2]); Tcm = Tcm.reshape(shape[:2])
sur_c = lambda TT: ((TT[:, :, None] >= T_MIN) & (np.abs(NN - Cc[:, :, None]) <= TT[:, :, None] / 2)).reshape(-1)
coarse.update({'indicator_0.3': ci >= 0.3, 'indicator_0.7': ci >= 0.7, 'signed_out1': cs_ >= CAP - 1, 'signed_in1': cs_ >= CAP + 1,
               'surfaces_out1': sur_c(Tcp), 'surfaces_in1': sur_c(Tcm)})
vol_coarse = {k: round(float(v.sum() * cell / 1e6), 4) for k, v in coarse.items()}
print('coarse volumes (paper grid)', vol_coarse, flush=True)

col = ((ci.reshape(shape) >= 0.2).any(2) | (cs_.reshape(shape) >= CAP - 4).any(2) | (Tc >= 0.5) | (Tcp >= 0.5))
# dilate one coarse cell along s and z
dil = col.copy()
for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
    dil |= np.roll(np.roll(col, a, 0), b, 1)
col = dil

# ------------------------------------------------------------------ fine grid: 5 x 5 x 0.5 m, aligned with the coarse cells
fs = np.repeat(gs, 4) + np.tile([-3.75, -1.25, 1.25, 3.75], len(gs))
fz = np.repeat(gd, 4) + np.tile([-3.75, -1.25, 1.25, 3.75], len(gd))
fn = np.arange(gn[0] - 1.0, gn[-1] + 1.0 + 1e-9, 0.5)       # covers the coarse cells' extent across the horizon
fcol = np.repeat(np.repeat(col, 4, 0), 4, 1)
I, J = np.nonzero(fcol)
print('active fine columns', len(I), 'of', fcol.size, '; points', len(I) * len(fn), flush=True)
SZ = np.column_stack([fs[I], fz[J]])

shp = (len(fs), len(fz), len(fn))
ind = np.zeros(shp, np.float32); sdf = np.zeros(shp, np.float32); sur = np.full(shp, -10.0, np.float32)
# 2D surfaces on the fine columns
T2 = np.zeros(len(I)); C2 = np.zeros(len(I))
ok2 = cKDTree(H[['s', 'z']].to_numpy()).query(SZ, distance_upper_bound=60)[0] < np.inf
T2[ok2] = pred2(m_t, SZ[ok2] / R2); C2[ok2] = pred2(m_c, SZ[ok2] / R2)
sur[I, J, :] = np.minimum(T2[:, None] - T_MIN, T2[:, None] / 2 - np.abs(fn[None, :] - C2[:, None]))
sur[I[~ok2], J[~ok2], :] = -10.0
Tp = np.zeros(len(I)); Tm = np.zeros(len(I)); Tp[ok2] = pred2(m_tp, SZ[ok2] / R2); Tm[ok2] = pred2(m_tm, SZ[ok2] / R2)
fsur = lambda TT: int(((TT[:, None] >= T_MIN) & (np.abs(fn[None, :] - C2[:, None]) <= TT[:, None] / 2)).sum())
# 3D fields only in a band across the horizon: coarse cells near any shell, +/- 4 m, per (dilated) coarse column
near3 = (ci.reshape(shape) >= 0.1) | (cs_.reshape(shape) >= CAP - 6)
band = np.where(near3.any(2), 0, 1)
kk = np.arange(shape[2])
nlo = np.where(near3.any(2), np.where(near3, kk, 10 ** 6).min(2), 10 ** 6).astype(float)
nhi = np.where(near3.any(2), np.where(near3, kk, -1).max(2), -1).astype(float)
lo_n = np.full(col.shape, np.inf); hi_n = np.full(col.shape, -np.inf)
ok = near3.any(2)
lo_n[ok] = gn[nlo[ok].astype(int)] - 1.25 - 4.0; hi_n[ok] = gn[nhi[ok].astype(int)] + 1.25 + 4.0
for a_, b_ in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
    lo_n = np.minimum(lo_n, np.roll(np.roll(lo_n, a_, 0), b_, 1)); hi_n = np.maximum(hi_n, np.roll(np.roll(hi_n, a_, 0), b_, 1))
flo = np.repeat(np.repeat(lo_n, 4, 0), 4, 1)[I, J]; fhi = np.repeat(np.repeat(hi_n, 4, 0), 4, 1)[I, J]
sel = (fn[None, :] >= flo[:, None]) & (fn[None, :] <= fhi[:, None])        # (columns, n) to evaluate
print('band points', int(sel.sum()), flush=True)
t0 = time.perf_counter()
ci_, cn_ = np.nonzero(sel)
CH = 300000
for k in range(0, len(ci_), CH):
    a2, n2 = ci_[k:k + CH], cn_[k:k + CH]
    Q = np.column_stack([fs[I[a2]], fz[J[a2]], fn[n2]])
    ind[I[a2], J[a2], n2] = pred(m_ind, Q / R0)
    sdf[I[a2], J[a2], n2] = pred(m_sd, Q / R0)
print(f'fine prediction {time.perf_counter() - t0:.0f} s', flush=True)

fcell = 2.5 * 2.5 * 0.5
vol_fine = dict(indicator=round(float((ind >= 0.5).sum() * fcell / 1e6), 4), signed=round(float((sdf >= CAP).sum() * fcell / 1e6), 4),
                surfaces=round(float((sur >= 0).sum() * fcell / 1e6), 4),
                **{'indicator_0.3': round(float((ind >= 0.3).sum() * fcell / 1e6), 4), 'indicator_0.7': round(float((ind >= 0.7).sum() * fcell / 1e6), 4),
                   'signed_out1': round(float((sdf >= CAP - 1).sum() * fcell / 1e6), 4), 'signed_in1': round(float((sdf >= CAP + 1).sum() * fcell / 1e6), 4),
                   'surfaces_out1': round(fsur(Tp) * fcell / 1e6, 4), 'surfaces_in1': round(fsur(Tm) * fcell / 1e6, 4)})
print('fine volumes', vol_fine, flush=True)
(lambda *a, **k: None)(OUT / f'fine_ep_{name}.npz', fs=fs, fz=fz, fn=fn, z0=z0, cap=CAP,
                    ind=ind.astype(np.float16), sd=sdf.astype(np.float16), sur=sur.astype(np.float16))
json.dump(dict(domain=name, coarse_cell_m=[10, 10, 2.5], fine_cell_m=[2.5, 2.5, 0.5], volume_coarse_Mm3=vol_coarse,
               volume_fine_Mm3=vol_fine, active_fine_columns=int(len(I)), total_s=round(time.perf_counter() - t_all, 1)),
          open(OUT / f'fine_ep_{name}.json', 'w'), indent=1)
print('wrote', OUT / f'fine_ep_{name}.npz', flush=True)
