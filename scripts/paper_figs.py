"""Figures 1-6, 8 and 10 of the paper (Figs 7 and 9 come from figs.py). usage: python paper_figs.py <drilling_dir>"""
import sys, json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
sys.path.insert(0, '.')
from unfold import Unfold
INK, INK2, GRID = '#1f2328', '#57606a', '#d0d7de'
COL = {'indicator': '#0072B2', 'signed': '#D55E00', 'surfaces': '#009E73', 'lenses': '#7b3fa0'}
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'font.family': 'DejaVu Sans'})
import os
F = 'out/paper_figs/'; os.makedirs(F, exist_ok=True)
DATA = sys.argv[1]   # folder with ThalangaML_Geology.csv and ThalangaML_Assays.csv
DOM = {'Far West': (369500, 370500), 'West': (370500, 372050), 'East': (372050, 373250), 'Far East': (373250, 374300)}
KEY = {'Far West': 'FarWest', 'West': 'West', 'East': 'East', 'Far East': 'FarEast'}
comp = pd.read_csv('out/composites.csv'); U = Unfold(comp)
def clean(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)

# ---------------- Fig 1: collars, ore-horizon trace, lens groups
h = pd.read_csv('out/holes.csv')
ms = comp[comp.f_ms >= 0.5]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.scatter(h.X_MGA, h.Y_MGA, s=1.5, c='#9aa4ad', lw=0, label='Drillhole collar (2,131)')
ax.scatter(ms.X, ms.Y, s=1.2, c='#2b2b2b', lw=0, label='Massive sulphide composite')
ax.plot(U.cx, U.cy, color='#D55E00', lw=1.6, label='Ore-horizon trace (unfolding centreline)')
for i, (n, (a, b)) in enumerate(DOM.items()):
    ax.axvspan(a, b, color=['#eef3f8', '#ffffff'][i % 2], zorder=-1)
    ax.text((a + b) / 2, 7752350, n, ha='center', fontsize=8.5, color=INK)
ax.set_xlim(369300, 374500); ax.set_ylim(7749500, 7752500); ax.set_aspect('equal')
ax.set_xlabel('Easting (m, GDA94 MGA Zone 55)'); ax.set_ylabel('Northing (m)')
ax.ticklabel_format(useOffset=False, style='plain'); ax.legend(loc='lower left', fontsize=7, frameon=False, markerscale=4)
clean(ax); fig.tight_layout(); fig.savefig(F + 'fig01_location.png', dpi=300); plt.close(fig)

# ---------------- Fig 3: workflow (two rows)
steps = [('1  Data audit', 'desurvey, 2 m composites,\ncolumn-shift correction,\nnormative density'),
         ('2  Unfolded frame', 's along the ore-horizon trace,\nz elevation,\nn across the horizon'),
         ('3  Four constructions', 'indicator, signed distance,\ncontact surfaces, correlated\nlenses (gen. Cauchy kernel)'),
         ('4  Sensitivity design', 'threshold, contacts \u00b11 m,\nranges, anisotropy, dip,\nminimum width'),
         ('5  Withheld holes', 'refits without six withheld\nsets: where geometry is\nsensitive to data'),
         ('6  Grades and tonnage', 'ordinary kriging inside\nevery geometry;\ngrade\u2013tonnage bands')]
fig, ax = plt.subplots(figsize=(7.2, 3.1)); ax.axis('off'); ax.set_xlim(0, 3); ax.set_ylim(0, 2)
pos = [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]
for (t, d), (x, y) in zip(steps, pos):
    ax.add_patch(FancyBboxPatch((x + 0.05, y + 0.08), 0.86, 0.82, boxstyle='round,pad=0.01,rounding_size=0.04', fc='#f6f8fa', ec=INK2, lw=0.8))
    ax.text(x + 0.48, y + 0.72, t, ha='center', va='center', fontsize=8.5, weight='bold', color=INK)
    ax.text(x + 0.48, y + 0.36, d, ha='center', va='center', fontsize=6.6, color=INK2, linespacing=1.35)
for a_, b_ in (((0.91, 1.49), (1.05, 1.49)), ((1.91, 1.49), (2.05, 1.49)), ((2.48, 1.08), (2.48, 0.9)), ((2.05, 0.49), (1.91, 0.49)), ((1.05, 0.49), (0.91, 0.49))):
    ax.annotate('', xy=b_, xytext=a_, arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
fig.tight_layout(); fig.savefig(F + 'fig03_workflow.png', dpi=300); plt.close(fig)

# ---------------- helpers for mask thickness maps
def masks(dom):
    z = np.load(f'out/contacts_{dom}_masks.npz'); names = list(z['names']); n = int(z['n_cells'])
    sh = (len(z['gs']), len(z['gd']), len(z['gn']))
    get = lambda k: np.unpackbits(z[f'm{names.index(k)}'])[:n].astype(bool).reshape(sh)
    return z['gs'], z['gd'], get
def thick(m):
    t = m.sum(2) * 2.5; return np.where(t > 0, t, np.nan)

# ---------------- Fig 6: three constructions, West
gs, gd, get = masks('West')
c = comp[(comp.X >= 370500) & (comp.X < 372050) & (comp.f_ms >= 0.5)]
L = U(c[['X', 'Y', 'Z']].to_numpy()); L[:, 1] -= comp.Z.mean()
fig, axs = plt.subplots(4, 1, figsize=(7.2, 9.6), sharex=True)
for ax, (k, lab) in zip(axs, [('indicator|threshold_0.5', 'Indicator (threshold 0.5)'), ('signed|reference', 'Signed distance to contacts'),
                              ('surfaces|reference', 'Contact surfaces (one sheet per hole, $w_{min}$ 1 m)'), ('lenses|reference', 'Correlated lenses ($w_{min}$ 1 m)')]):
    im = ax.pcolormesh(gs, gd, thick(get(k)).T, cmap='Reds', vmin=0, vmax=20, shading='nearest')
    ax.scatter(L[:, 0], L[:, 1], s=0.4, c='k', lw=0)
    ax.set_aspect('equal'); ax.set_ylabel('Elevation rel. mean (m)'); ax.set_title(lab, loc='left', fontsize=9, color=INK); clean(ax)
    ax.set_xlim(1000, 2750); ax.set_ylim(-450, 230)
axs[-1].set_xlabel('Distance along the ore-horizon trace (m)')
cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.02, extend='max'); cb.set_label('MS width across the horizon (m); colour saturates at 20 m')
fig.savefig(F + 'fig06_constructions_west.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- Fig 5: share of defensible indicator geometries, West and East (MS)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.6))
for ax, d in zip(axs, ['West', 'East']):
    z = np.load(f'out/sens_{d}_freq.npz'); n = float(z['n_comb'])
    v = z['MS'].max(2) / n; v = np.where(v > 0, v, np.nan)
    im = ax.pcolormesh(z['gs'], z['gd'], v.T, cmap='Reds', vmin=0, vmax=1, shading='nearest')
    ax.set_aspect('equal'); ax.set_title(f'{d}: share of {int(n)} geometries placing massive sulphide in each column', loc='left', fontsize=9, color=INK)
    ax.set_ylabel('Elevation rel. mean (m)'); clean(ax)
    ok = ~np.isnan(v); ii = np.where(ok.any(1))[0]; jj = np.where(ok.any(0))[0]
    ax.set_xlim(z['gs'][ii[0]] - 30, z['gs'][ii[-1]] + 30); ax.set_ylim(z['gd'][jj[0]] - 30, z['gd'][jj[-1]] + 30)
axs[-1].set_xlabel('Distance along the ore-horizon trace (m)')
cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.02); cb.set_label('Share of geometries')
fig.savefig(F + 'fig05_sensitivity_share.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- Fig 4: reference MS and OH thickness, four lens groups
fig, axs = plt.subplots(4, 2, figsize=(7.2, 8.6))
for r, (name, d) in enumerate(KEY.items()):
    g = np.load(f'out/{d}_grid.npz')
    for cidx, (k, cm, vmax) in enumerate((('MS', 'Reds', 15), ('OH', 'Blues', 60))):
        ax = axs[r, cidx]; t = (g[k] >= 0).sum(2) * 2.5; t = np.where(t > 0, t, np.nan)
        im = ax.pcolormesh(g['gs'], g['gd'], t.T, cmap=cm, vmin=0, vmax=vmax, shading='nearest')
        ok = ~np.isnan(t)
        if ok.any():
            ii = np.where(ok.any(1))[0]; jj = np.where(ok.any(0))[0]
            ax.set_xlim(g['gs'][ii[0]] - 20, g['gs'][ii[-1]] + 20); ax.set_ylim(g['gd'][jj[0]] - 20, g['gd'][jj[-1]] + 20)
        ax.set_aspect('equal'); clean(ax); ax.tick_params(labelsize=7)
        ax.set_title(f'{name}: {"massive sulphide" if k == "MS" else "ore horizon"}', loc='left', fontsize=8.5, color=INK)
        if r == 3: ax.set_xlabel('Along trace (m)')
        if cidx == 0: ax.set_ylabel('Elevation (m)')
    if r == 0:
        pass
fig.tight_layout(rect=(0, 0.05, 1, 1))
for cidx, (cm, vmax, lab) in enumerate((('Reds', 15, 'MS width across the horizon (m)'), ('Blues', 60, 'OH width across the horizon (m)'))):
    cax = fig.add_axes([0.1 + 0.48 * cidx, 0.025, 0.35, 0.012])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, vmax)), cax=cax, orientation='horizontal', extend='max').set_label(lab + f' (saturates at {vmax} m)', fontsize=8)
fig.savefig(F + 'fig04_reference_width.png', dpi=300); plt.close(fig)

# ---------------- Fig 2: the column-shift audit
import glob
G = os.path.join(DATA, 'ThalangaML_Geology.csv')
A = os.path.join(DATA, 'ThalangaML_Assays.csv')
g = pd.read_csv(G, low_memory=False); a = pd.read_csv(A, low_memory=False)
valid = {'FV', 'FW', 'HV', 'HW', 'OH', 'EX', 'IN'}; grain = {'FS', 'CS', 'MW', 'VFS', 'VCS'}
sh = (g.FORMATION.notna() & ~g.FORMATION.isin(valid)) | pd.to_numeric(g.TEXTURE, errors='coerce').notna() | g.LITH1_legacy.isin(grain) | g.Lith1_Validated.isin(grain)
naive = g.Lith1_Validated.fillna(g.LITH1_legacy).isin(['MS', 'SM'])
fixed = g.Lith1_Validated.fillna(g.LITH1_legacy).where(~sh, g.LITH2_legacy).isin(['MS', 'SM'])
g['grp'] = np.where(naive & sh & ~fixed, 'False MS\n(shifted rows)', np.where(fixed, 'Logged MS\n(corrected)', None))
a['mid'] = (a.From + a.To) / 2; a['S'] = pd.to_numeric(a.S_pct, errors='coerce')
a['base'] = sum(pd.to_numeric(a[c], errors='coerce').fillna(0) for c in ('Zn_pct', 'Pb_pct', 'Cu_pct'))
gg = g[g.grp.notna()].copy(); gg['mid'] = (gg.From + gg.To) / 2
m = pd.merge_asof(a.sort_values('mid'), gg[['Drillhole ID', 'From', 'To', 'grp']].sort_values('From'), left_on='mid', right_on='From', by='Drillhole ID')
m = m[m.grp.notna() & (m.mid < m.To_y)]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
cats = ['False MS\n(shifted rows)', 'Logged MS\n(corrected)']
for ax, col, lab in ((axs[0], 'S', 'S (%)'), (axs[1], 'base', 'Zn + Pb + Cu (%)')):
    data = [m.loc[m.grp == k, col].dropna().to_numpy() for k in cats]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK))
    for p, cc in zip(bp['boxes'], ('#D55E00', '#0072B2')): p.set_facecolor(cc); p.set_alpha(0.35)
    ax.set_xticks([1, 2]); ax.set_xticklabels([f'{k}\nn={len(d)}' for k, d in zip(cats, data)], fontsize=7)
    ax.set_ylabel(lab); clean(ax); ax.grid(axis='y', color=GRID, lw=0.5)
axs[2].bar([0, 1], [2.93, 0.10], color=['#D55E00', '#0072B2'], alpha=0.8, width=0.6)
for x, v in ((0, 2.93), (1, 0.10)): axs[2].text(x, v + 0.05, f'{v:.2f}', ha='center', fontsize=8)
axs[2].set_xticks([0, 1]); axs[2].set_xticklabels(['As\nrecorded', 'Corrected'], fontsize=7); axs[2].set_ylabel('Far West MS volume (Mm³)')
clean(axs[2])
fig.tight_layout(); fig.savefig(F + 'fig02_log_audit.png', dpi=300); plt.close(fig)
print('audit n', m.groupby('grp').agg(S=('S', 'median'), base=('base', 'median'), n=('S', 'size')).round(2).to_dict())

# ---------------- Fig 8: withheld holes, East (three constructions)
gs, gd, get = masks('East')
H = pd.read_csv('out/contacts_East_holes.csv'); dev = H[~H.frozen]; frz = H[H.frozen]
fig, axs = plt.subplots(4, 1, figsize=(7.2, 9.2), sharex=True)
cmap = matplotlib.colors.ListedColormap(['#c0392b', '#d0d7de', '#f0b429', '#2e86de'])
for ax, (con, ref, lab) in zip(axs, [('indicator', 'indicator|threshold_0.5', 'Indicator'), ('signed', 'signed|reference', 'Signed distance'),
                                     ('surfaces', 'surfaces|reference', 'Contact surfaces'), ('lenses', 'lenses|reference', 'Correlated lenses')]):
    A_, B_ = get(ref), get(f'{con}|without_frozen'); ca, cb_ = A_.any(2), B_.any(2); ch = (A_ != B_).any(2)
    img = np.full(ca.shape, np.nan); img[ca & cb_] = 1; img[ca & cb_ & ch] = 2; img[ca & ~cb_] = 0; img[~ca & cb_] = 3
    ax.pcolormesh(gs, gd, img.T, cmap=cmap, norm=matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4), shading='nearest')
    ax.scatter(dev.s, dev.z, s=2.5, c=INK2, lw=0); ax.scatter(frz.s, frz.z, s=10, marker='x', c=INK, lw=0.6)
    ax.set_aspect('equal'); clean(ax); ax.set_ylabel('Elevation rel. mean (m)'); ax.set_title(lab, loc='left', fontsize=9, color=INK)
    ax.set_xlim(2700, 4000); ax.set_ylim(-350, 200)
axs[-1].set_xlabel('Distance along the ore-horizon trace (m)')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
axs[0].legend([Patch(color='#d0d7de'), Patch(color='#f0b429'), Patch(color='#c0392b'), Patch(color='#2e86de'),
               Line2D([], [], marker='.', ls='', color=INK2), Line2D([], [], marker='x', ls='', color=INK)],
              ['MS in both', 'MS in both, thickness changed', 'Lost without withheld holes', 'Gained', 'Remaining hole', 'Withheld hole'],
              fontsize=6.5, ncol=3, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.42))
fig.savefig(F + 'fig08_withheld_east.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# ---------------- Fig 10: the reference shells on the lines of Paulick (1999) sections 2 and 5 (published sections not reproduced)
ims = [Image.open(f'out/fig_section_{k}.png') for k in (2, 5)]
W = sum(i.size[0] for i in ims); Hh = max(i.size[1] for i in ims)
c_ = Image.new('RGB', (W, Hh), 'white'); x = 0
for i in ims: c_.paste(i, (x, 0)); x += i.size[0]
c_.save(F + 'fig10_sections.png')
for f in ('fig07_what_moves_boundary.png', 'fig09_grade_tonnage.png'):
    Image.open('out/' + f).save(F + f)
print('done')
