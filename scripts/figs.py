import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
INK, INK2, GRID = '#1f2328', '#57606a', '#d0d7de'
COL = {'indicator': '#0072B2', 'signed': '#D55E00', 'surfaces': '#009E73', 'lenses': '#7b3fa0'}
LAB = {'indicator': 'Indicator', 'signed': 'Signed distance to contacts', 'surfaces': 'Contact surfaces (one sheet)', 'lenses': 'Correlated lenses'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2})
# ---- Fig 7: what moves the boundary (total MS volume, all four domains)
t = pd.read_csv('out/step3_table.csv'); t = t[t.variant != 'without_frozen']
tot = t.groupby(['construction', 'variant']).volume_Mm3.sum().reset_index()
tot['family'] = np.where(tot.variant.str.startswith('threshold'), 'threshold', tot.variant.str.split('_').str[0])
NAMES = {'threshold': 'Threshold 0.3–0.7', 'contacts': 'Contacts ±1 m', 'ranges': 'Ranges ×0.75–1.25', 'tmin': 'Minimum width 0.5–2 m',
         'elong': '2:1 elongation (4 directions)', 'across': 'Across range 7.5–12.5 m', 'dip': 'Dip ±10°'}
fig, ax = plt.subplots(figsize=(7.2, 4.8))
y = 0; ticks = []
for fam in ['threshold', 'contacts', 'tmin', 'ranges', 'elong', 'across', 'dip']:
    for con in ['indicator', 'signed', 'surfaces', 'lenses']:
        g = tot[(tot.construction == con) & ((tot.family == fam) | tot.variant.isin(['threshold_0.5', 'reference']))]
        if not (tot[(tot.construction == con)].family == fam).any():
            continue
        ref = tot[(tot.construction == con) & tot.variant.isin(['threshold_0.5', 'reference'])].volume_Mm3.iloc[0]
        lo, hi = g.volume_Mm3.min(), g.volume_Mm3.max()
        ax.plot([lo, hi], [y, y], color=COL[con], lw=5, solid_capstyle='butt')
        ax.plot(ref, y, '|', color=INK, ms=10, mew=1.5)
        ticks.append((y, NAMES[fam] if con == min([c for c in COL if (tot[tot.construction == c].family == fam).any()], key=list(COL).index) else ''))
        y -= 1
    y -= 0.6
ax.set_yticks([a for a, _ in ticks]); ax.set_yticklabels([b for _, b in ticks])
ax.set_xlabel('Massive sulphide volume, all four lens groups (Mm³)'); ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], color=COL[c], lw=5) for c in COL] + [Line2D([], [], color=INK, marker='|', ls='', ms=10, mew=1.5)],
          [LAB[c] for c in COL] + ['Reference'], loc='lower right', fontsize=8, frameon=False)
ax.set_title('What moves the massive sulphide boundary', loc='left', color=INK, fontsize=10)
fig.tight_layout(); fig.savefig('out/fig07_what_moves_boundary.png', dpi=200); plt.close(fig)
# ---- Fig 9: tonnes, contained Zn and mean Zn grade above cut-off across geometries (three charts, no shared axis)
gt = pd.read_csv('out/gt_table.csv'); gt = gt[(gt.density_factor == 1) & (gt.variant != 'without_frozen')]
gt = gt[(gt.construction != 'indicator') | gt.variant.isin(['threshold_0.3', 'threshold_0.5', 'threshold_0.7', 'ranges_x0.75', 'ranges_x1.25'])]
gt['ZnT'] = gt.Zn_pct.fillna(0) * gt.tonnes_Mt
g = gt.groupby(['construction', 'variant', 'zn_cutoff']).agg(Mt=('tonnes_Mt', 'sum'), Zn_kt=('Zn_kt', 'sum'), ZnT=('ZnT', 'sum')).reset_index()
g['Zn'] = g.ZnT / g.Mt
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
for ax, col, ylab, ttl in ((axs[0], 'Mt', 'Tonnes (Mt)', 'Tonnes'), (axs[1], 'Zn_kt', 'Contained Zn (kt)', 'Contained zinc'),
                           (axs[2], 'Zn', 'Mean Zn grade (%)', 'Mean zinc grade')):
    for con in COL:
        x = g[g.construction == con]
        b = x.groupby('zn_cutoff')[col].agg(['min', 'max'])
        r = x[x.variant.isin(['threshold_0.5', 'reference'])].set_index('zn_cutoff')[col]
        ax.fill_between(b.index, b['min'], b['max'], color=COL[con], alpha=0.15, lw=0)
        ax.plot(r.index, r.values, color=COL[con], lw=1.8, label=LAB[con])
    ax.set_xlabel('Zn cut-off (%)'); ax.set_ylabel(ylab); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    if col != 'Zn': ax.set_ylim(bottom=0)
    ax.set_title(ttl + ' above cut-off', loc='left', color=INK, fontsize=9)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
axs[0].legend(fontsize=6.5, frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig('out/fig09_grade_tonnage.png', dpi=200); plt.close(fig)
print('ok')
