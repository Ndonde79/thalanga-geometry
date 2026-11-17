"""Steps 3-4-6 analysis on the workstation outputs (run in the sandbox).

usage: python checks.py <out_dir> <collar_csv> <ip_csv> <paulick_dir>
  - step 3 table: volume per construction and variant, swing per family     -> out/step3_table.csv, step3_summary.csv
  - step 4: reference vs without_frozen per construction; where the geometry changes against distance to drilling
                                                                            -> out/step4_withheld.csv, step4_by_distance.csv, fig_step4_<dom>.png
  - step 6: IP chargeability/resistivity inside and around the MS and OH shells -> out/step6_ip.csv
            Paulick (1999) sections: our reference shells on the same section lines -> fig_paulick_<k>.png
"""
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from unfold import Unfold

OUT = Path(sys.argv[1]); COLLAR = sys.argv[2]; IPF = sys.argv[3]; PDIR = Path(sys.argv[4])
DOM = {'FarWest': (369500, 370500), 'West': (370500, 372050), 'East': (372050, 373250), 'FarEast': (373250, 374300)}
comp = pd.read_csv(OUT / 'composites.csv')
U = Unfold(comp); z0 = float(comp.Z.mean())
INK, INK2 = '#1f2328', '#6e7781'


def local(X):
    S = U(np.asarray(X, float)); S[:, 1] -= z0
    return S


def load(dom):
    z = np.load(OUT / f'contacts_{dom}_masks.npz')
    names = list(z['names']); n = int(z['n_cells']); sh = (len(z['gs']), len(z['gd']), len(z['gn']))
    M = {nm: np.unpackbits(z[f'm{i}'])[:n].astype(bool).reshape(sh) for i, nm in enumerate(names)}
    g = np.load(OUT / f'{dom}_grid.npz')
    oh = g['OH'] >= 0 if g['OH'].shape == sh else None
    return z['gs'], z['gd'], z['gn'], M, oh


def lookup(gs, gd, gn, L):
    i = np.rint((L[:, 0] - gs[0]) / 10).astype(int); j = np.rint((L[:, 1] - gd[0]) / 10).astype(int)
    k = np.rint((L[:, 2] - gn[0]) / 2.5).astype(int)
    ok = (i >= 0) & (i < len(gs)) & (j >= 0) & (j < len(gd)) & (k >= 0) & (k < len(gn))
    return i, j, k, ok


D = {dom: load(dom) for dom in DOM}
REFS = {'indicator': 'indicator|threshold_0.5', 'signed': 'signed|reference', 'surfaces': 'surfaces|reference', 'lenses': 'lenses|reference'}

# ------------------------------------------------------------------ step 3 table
rows = []
for dom in DOM:
    js = json.load(open(OUT / f'contacts_{dom}.json'))
    for r in js['rows'] + json.load(open(OUT / f'lenses_{dom}.json'))['rows']:
        rows.append(dict(domain=dom, **{k: r.get(k) for k in ('construction', 'variant', 'volume_Mm3', 'jaccard_vs_own_reference')}))
t3 = pd.DataFrame(rows); t3.to_csv(OUT / 'step3_table.csv', index=False)
fam = lambda v: v.split('_')[0] if not v.startswith('threshold') else 'threshold'
t3['family'] = t3.variant.map(fam)
summ = []
for (dom, con), g in t3[t3.variant != 'without_frozen'].groupby(['domain', 'construction']):
    ref = g[g.variant.isin(['threshold_0.5', 'reference'])].volume_Mm3.iloc[0]
    s = dict(domain=dom, construction=con, reference_Mm3=ref, min_Mm3=g.volume_Mm3.min(), max_Mm3=g.volume_Mm3.max())
    for f, h in g.groupby('family'):
        if f in ('reference',):
            continue
        hh = pd.concat([h, g[g.variant.isin(['threshold_0.5', 'reference'])]]) if f != 'threshold' else h
        s[f'swing_{f}_pct'] = round(100 * (hh.volume_Mm3.max() - hh.volume_Mm3.min()) / ref, 1)
    summ.append(s)
s3 = pd.DataFrame(summ); s3.to_csv(OUT / 'step3_summary.csv', index=False)
tot = t3[t3.variant != 'without_frozen'].groupby(['construction', 'variant']).volume_Mm3.sum().unstack(0)
tot.to_csv(OUT / 'step3_totals.csv')
print(s3.to_string(index=False)); print(tot.round(3).to_string())

# ------------------------------------------------------------------ step 4: withheld holes
w_rows, dist_rows = [], []
for dom in DOM:
    gs, gd, gn, M, _ = D[dom]
    H = pd.read_csv(OUT / f'contacts_{dom}_holes.csv')
    dev = H[~H.frozen]; frz = H[H.frozen]
    G2 = np.stack(np.meshgrid(gs, gd, indexing='ij'), -1)
    d_dev = cKDTree(dev[['s', 'z']].to_numpy()).query(G2.reshape(-1, 2))[0].reshape(G2.shape[:2])
    d_frz = cKDTree(frz[['s', 'z']].to_numpy()).query(G2.reshape(-1, 2))[0].reshape(G2.shape[:2]) if len(frz) else np.full(G2.shape[:2], np.inf)
    fig, axs = plt.subplots(len(REFS), 1, figsize=(10, 3.5 * len(REFS)))
    for ax, (con, ref) in zip(axs, REFS.items()):
        a, b = M[ref], M[f'{con}|without_frozen']
        va, vb = a.sum() * 250 / 1e6, b.sum() * 250 / 1e6
        w_rows.append(dict(domain=dom, construction=con, all_holes_Mm3=round(va, 4), without_frozen_Mm3=round(vb, 4),
                           change_pct=round(100 * (vb - va) / va, 1) if va else None,
                           jaccard=round(float((a & b).sum() / max((a | b).sum(), 1)), 3),
                           frozen_holes_crossing=int(len(frz)), frozen_with_ms=int((frz.t > 0).sum())))
        col_a, col_b = a.any(2), b.any(2)
        changed = (a != b).any(2); any_ = col_a | col_b
        for lo, hi in ((0, 25), (25, 50), (50, 100), (100, 1e9)):
            m = any_ & (d_dev >= lo) & (d_dev < hi)
            dist_rows.append(dict(domain=dom, construction=con, dist_to_development_hole_m=f'{lo}-{hi if hi < 1e9 else "inf"}',
                                  columns=int(m.sum()), changed_share=round(float(changed[m].mean()), 3) if m.any() else None,
                                  near_frozen_share=round(float((d_frz[m] < 25).mean()), 3) if m.any() else None))
        img = np.full(col_a.shape, np.nan)
        img[col_a & col_b] = 1; img[col_a & ~col_b] = 0; img[~col_a & col_b] = 3
        img[col_a & col_b & changed] = 2
        ax.pcolormesh(gs, gd, img.T, cmap=matplotlib.colors.ListedColormap(['#c0392b', '#d0d7de', '#f0b429', '#2e86de']),
                      norm=matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4), shading='nearest')
        ax.scatter(dev.s, dev.z, s=3, c=INK2, lw=0); ax.scatter(frz.s, frz.z, s=14, marker='x', c=INK, lw=0.8)
        ax.set_aspect('equal'); ax.set_title(f'{dom}, {con}: all holes vs without the frozen holes (x)', loc='left', fontsize=9, color=INK)
        ax.set_ylabel('elevation rel. mean (m)')
    axs[-1].set_xlabel('along the ore-horizon trace (m)')
    fig.text(0.01, 0.005, 'grey: MS in both; amber: both, thickness changed; red: lost without the frozen holes; blue: gained', fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(OUT / f'fig_step4_{dom}.png', dpi=110); plt.close(fig)
pd.DataFrame(w_rows).to_csv(OUT / 'step4_withheld.csv', index=False)
pd.DataFrame(dist_rows).to_csv(OUT / 'step4_by_distance.csv', index=False)
print(pd.DataFrame(w_rows).to_string(index=False))
print(pd.DataFrame(dist_rows).to_string(index=False))

# ------------------------------------------------------------------ step 6a: IP inside / around the shells
ip = pd.read_csv(IPF)
res = []
for dom, (x0, x1) in DOM.items():
    gs, gd, gn, M, oh = D[dom]
    p = ip[(ip.GridE >= x0) & (ip.GridE < x1)]
    if not len(p):
        continue
    L = local(p[['GridE', 'GridN', 'Elev']].to_numpy())
    i, j, k, ok = lookup(gs, gd, gn, L)
    ms_any = np.zeros(len(p), bool); ms_col = np.zeros(len(p), bool); in_oh = np.zeros(len(p), bool)
    ms_any[ok] = M[REFS['indicator']][i[ok], j[ok], k[ok]] | M[REFS['signed']][i[ok], j[ok], k[ok]] | M[REFS['surfaces']][i[ok], j[ok], k[ok]]
    col = M[REFS['indicator']].any(2) | M[REFS['signed']].any(2) | M[REFS['surfaces']].any(2)
    ms_col[ok] = col[i[ok], j[ok]]
    if oh is not None:
        in_oh[ok] = oh[i[ok], j[ok], k[ok]]
    an = np.abs(L[:, 2])
    cls = np.where(ms_any, 'in MS shell', np.where(in_oh, 'in OH shell', np.where(ms_col & (an <= 30), 'within 30 m of MS (across)',
                   np.where(an <= 60, 'ore-horizon window, no MS', 'away from ore horizon'))))
    for c_ in np.unique(cls):
        q = p[cls == c_]
        res.append(dict(domain=dom, zone=c_, n=len(q), ip_median=round(q.InvIP.median(), 2), ip_p75=round(q.InvIP.quantile(.75), 2),
                        res_median=round(q.InvRes.median(), 0)))
ipr = pd.DataFrame(res); ipr.to_csv(OUT / 'step6_ip.csv', index=False); print(ipr.to_string(index=False))

# ------------------------------------------------------------------ step 6b: Paulick (1999) sections
col = pd.read_csv(COLLAR, low_memory=False)
col['key'] = col.HOLE_ID.astype(str).str.replace(r'^TH0*', 'TH', regex=True)
SEC = {1: (['TH62C', 'TH40', 'TH37', 'TH410A', 'TH410', 'TH148'], 'Alteration Far West Section 1 - Paulick 1999.tif'),
       2: (['TH28', 'TH29', 'TH41', 'TH41A', 'TH245', 'TH243A', 'TH247'], 'Alteration Thalanga West Section 2 - Paulick 1999.tif'),
       3: (['TH5', 'TH334', 'TH10', 'TH238', 'TH112'], 'Alteration Thalanga Central Section 3 - Paulick 1999.tif'),
       4: (['TH18', 'TH3', 'TH38', 'TH412A', 'TH412B'], 'Alteration Thalanga Central Section 4 - Paulick 1999.tif'),
       5: (['TH382A', 'TH401', 'TH85A', 'TH85', 'TH270', 'TH488C'], 'Alteration Thalanga East Section 5 - Paulick 1999.tif'),
       6: (['TH402', 'TH61', 'TH144B', 'TH394'], 'Alteration Thalanga East Section 6 - Paulick 1999.tif')}
traces = comp.groupby('hole')
sec_rows = []
for k, (holes, img) in SEC.items():
    cc = col[col.key.isin(holes)]
    XY = cc[['X_MGA', 'Y_MGA']].to_numpy(float)
    c0 = XY.mean(0); u = np.linalg.svd(XY - c0)[2][0]
    if u[1] < 0:
        u = -u                                                            # section runs towards north
    t = np.arange(-350, 350.1, 2.5); zz = np.arange(-700, 420.1, 2.5)
    T, Z = np.meshgrid(t, zz, indexing='ij')
    X = np.column_stack([c0[0] + T.ravel() * u[0], c0[1] + T.ravel() * u[1], Z.ravel()])
    dom = next(d_ for d_, (a, b) in DOM.items() if a <= c0[0] < b)
    gs, gd, gn, M, oh = D[dom]
    L = local(X); i, j, kk, ok = lookup(gs, gd, gn, L)
    layers = {}
    for con, ref in REFS.items():
        v = np.zeros(len(X), bool); v[ok] = M[ref][i[ok], j[ok], kk[ok]]; layers[con] = v.reshape(T.shape)
    vo = np.zeros(len(X), bool)
    if oh is not None:
        vo[ok] = oh[i[ok], j[ok], kk[ok]]
    fig = plt.figure(figsize=(13, 7.5))
    ax = fig.add_axes([0.05, 0.08, 0.5, 0.85])
    ax.contourf(t, zz, vo.reshape(T.shape).T.astype(float), levels=[0.5, 1.5], colors=['#cfe3f5'])
    for (con, v), colr in zip(layers.items(), ('#2b2b2b', '#c0392b', '#f0b429', '#7b3fa0')):
        if v.any():
            ax.contour(t, zz, v.T.astype(float), levels=[0.5], colors=[colr], linewidths=1.4)
    # drillholes on the section (within 25 m) with logged MS
    near = comp[(np.abs((comp.X - c0[0]) * u[1] - (comp.Y - c0[1]) * u[0]) < 25)]
    tt = (near.X - c0[0]) * u[0] + (near.Y - c0[1]) * u[1]
    ax.scatter(tt, near.Z, s=0.6, c=INK2, lw=0)
    m = near.f_ms >= 0.5; ax.scatter(tt[m], near.Z[m], s=5, c='k', lw=0)
    ax.set_xlim(-350, 350); ax.set_ylim(-500, 420); ax.set_aspect('equal')
    ax.set_xlabel(f'distance along section (m; direction {np.degrees(np.arctan2(u[0], u[1])) % 360:.0f}°)'); ax.set_ylabel('Z (m, MGA)')
    ax.set_title(f'Section {k} ({dom}): OH shell (blue fill); MS indicator (black), signed distance (red), surfaces (amber), lenses (purple); '
                 f'logged MS dots', loc='left', fontsize=8.5, color=INK)
    try:
        im = Image.open(PDIR / img).convert('RGB'); im.thumbnail((900, 1200))
        ax2 = fig.add_axes([0.58, 0.03, 0.4, 0.94]); ax2.imshow(im); ax2.axis('off')
        ax2.set_title(f'Paulick (1999) section {k} (massive sulphide in black)', fontsize=8.5, color=INK)
    except Exception as e:
        print('image', e)
    fig.savefig(OUT / f'fig_paulick_{k}.png', dpi=100); plt.close(fig)
    f2, a2 = plt.subplots(figsize=(5.2, 6.4))
    a2.contourf(t, zz, vo.reshape(T.shape).T.astype(float), levels=[0.5, 1.5], colors=['#cfe3f5'])
    for (con, v), colr in zip(layers.items(), ('#2b2b2b', '#c0392b', '#f0b429', '#7b3fa0')):
        if v.any(): a2.contour(t, zz, v.T.astype(float), levels=[0.5], colors=[colr], linewidths=1.3)
    a2.scatter(tt, near.Z, s=0.8, c='#9aa4ad', lw=0); a2.scatter(tt[m], near.Z[m], s=7, c='k', lw=0)
    ext = [v.any(0).nonzero()[0] for v in layers.values() if v.any()]
    _any = np.zeros_like(next(iter(layers.values())), bool)
    for v in layers.values(): _any |= v
    if _any.any():
        _xt = t[_any.any(1)]; _zz = zz[_any.any(0)]
        xc = 0.5 * (_xt.min() + _xt.max()); a2.set_xlim(xc - 150, xc + 150); a2.set_ylim(_zz.min() - 60, _zz.max() + 60)
    a2.set_aspect('equal')
    from matplotlib.lines import Line2D
    a2.legend([Line2D([], [], color=c_, lw=1.5) for c_ in ('#2b2b2b', '#c0392b', '#f0b429', '#7b3fa0')] + [plt.Rectangle((0, 0), 1, 1, color='#cfe3f5'), Line2D([], [], ls='', marker='o', ms=3, color='k')],
              ['Indicator', 'Signed distance', 'Contact surfaces', 'Correlated lenses', 'Ore horizon (indicator)', 'Logged MS composite'], fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    a2.set_xlabel(f'Distance along section line (m; azimuth {np.degrees(np.arctan2(u[0], u[1])) % 360:.0f}°)'); a2.set_ylabel('Elevation (m, MGA datum of the database)')
    a2.set_title(f'Section {k} line of Paulick (1999), {dom}', loc='left', fontsize=9, color=INK)
    for sp in ('top', 'right'): a2.spines[sp].set_visible(False)
    f2.savefig(OUT / f'fig_section_{k}.png', dpi=300, bbox_inches='tight'); plt.close(f2)
    sec_rows.append(dict(section=k, domain=dom, holes_found=int(len(cc)), centre_E=round(c0[0]), centre_N=round(c0[1]),
                         ms_area_m2={con: float(v.sum() * 6.25) for con, v in layers.items()}, oh_area_m2=float(vo.sum() * 6.25),
                         ms_z_range={con: ([float(zz[v.any(0)].min()), float(zz[v.any(0)].max())] if v.any() else None) for con, v in layers.items()}))
json.dump(sec_rows, open(OUT / 'step6_sections.json', 'w'), indent=1)
print(json.dumps(sec_rows, indent=1))
