"""Physical volume of frame cells (revision 2).

The unfolded frame maps (s, z, n) to MGA by X = c(s) + n * N(s), Z = z + z0, with c(s) the centreline at arc length s
and N(s) its horizontal unit normal. A frame cell of nominal size ds x dz x dn has physical volume
ds dz dn |det J|, with |det J| = |1 - kappa(s) n| for curvature kappa (sign set by the normal's direction).
J is evaluated numerically from the same inverse mapping used to export solids.
usage: python physvol.py <out_dir>   -> out/physical_volumes.csv (every mask in contacts_<dom>_masks.npz)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(sys.argv[1])
cl = np.genfromtxt(OUT / 'centreline.csv', delimiter=',', names=True)
cx, cy, cs = cl['cx'], cl['cy'], cl['cs']; tx, ty = np.gradient(cx, cs), np.gradient(cy, cs)


def to_world(s, n):
    fx = np.interp(s, cs, cx); fy = np.interp(s, cs, cy)
    lo, hi = s < cs[0], s > cs[-1]
    fx[lo] = cx[0] + (s[lo] - cs[0]) * tx[0]; fy[lo] = cy[0] + (s[lo] - cs[0]) * ty[0]
    fx[hi] = cx[-1] + (s[hi] - cs[-1]) * tx[-1]; fy[hi] = cy[-1] + (s[hi] - cs[-1]) * ty[-1]
    ux, uy = np.interp(s, cs, tx), np.interp(s, cs, ty); L = np.hypot(ux, uy); ux, uy = ux / L, uy / L
    nx, ny = -uy, ux; f = ny < 0; nx[f], ny[f] = -nx[f], -ny[f]
    return fx + n * nx, fy + n * ny


def jac(s, n, h=1.0):
    """|det| of d(X, Y)/d(s, n) (Z = z, so the full Jacobian reduces to this 2 x 2)."""
    xa, ya = to_world(s + h, n); xb, yb = to_world(s - h, n)
    xc, yc = to_world(s, n + h); xd, yd = to_world(s, n - h)
    return np.abs(((xa - xb) * (yc - yd) - (ya - yb) * (xc - xd)) / (4 * h * h))


rows = []
for dom in ('FarWest', 'West', 'East', 'FarEast'):
    z = np.load(OUT / f'contacts_{dom}_masks.npz'); names = list(z['names']); N = int(z['n_cells'])
    gs, gd, gn = z['gs'], z['gd'], z['gn']
    S, _, Nn = np.meshgrid(gs, gd, gn, indexing='ij'); J = jac(S.reshape(-1).astype(float), Nn.reshape(-1).astype(float))
    cell = (gs[1] - gs[0]) * (gd[1] - gd[0]) * (gn[1] - gn[0])
    for i, nm in enumerate(names):
        m = np.unpackbits(z[f'm{i}'])[:N].astype(bool)
        rows.append(dict(domain=dom, geometry=nm, frame_Mm3=m.sum() * cell / 1e6, physical_Mm3=(J[m] * cell).sum() / 1e6))
df = pd.DataFrame(rows); df['diff_pct'] = 100 * (df.physical_Mm3 / df.frame_Mm3 - 1)
df.round(5).to_csv(OUT / 'physical_volumes.csv', index=False)
tot = df.groupby('geometry')[['frame_Mm3', 'physical_Mm3']].sum(); tot['diff_pct'] = 100 * (tot.physical_Mm3 / tot.frame_Mm3 - 1)
print(tot.round(3).to_string())
print('per lens group, reference geometries:')
print(df[df.geometry.isin(['indicator|threshold_0.5', 'signed|reference', 'surfaces|reference', 'lenses|reference'])].round(3).to_string())
