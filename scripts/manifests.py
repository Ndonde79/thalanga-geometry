"""Supplementary manifests (revision 2).

usage: python manifests.py <drilling_dir> <out_dir> <supp_dir>
Writes to <supp_dir>:
  S2_log_corrections.csv   every geology row read as column-shifted: original fields, how it was read before and
                           after the correction, and which rule flagged it
  S3_scenarios.csv         every geometry: construction, variant and its settings
  S4_withheld_sets.csv     the six withheld-hole sets (original + seeds 1-5) and the rule that drew them
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

SRC, OUT, SUP = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); SUP.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ S2: column-shift corrections (same rule as prep.py)
collar = pd.read_csv(SRC / 'ThalangaML_collar.csv')
survey = pd.read_csv(SRC / 'ThalangaML_survey.csv').rename(columns={'Drillhole ID': 'hole'})
geol = pd.read_csv(SRC / 'ThalangaML_Geology.csv').rename(columns={'Drillhole ID': 'hole', 'From': 'f', 'To': 't'})
keep = collar[collar.HOLE_TYPE.isin(['DD', 'RC', 'PC'])]
keep = set(keep[keep.HOLE_ID.isin(survey.hole)].HOLE_ID)
geol = geol[geol.hole.isin(keep)].copy()
FORM = {'FV', 'FW', 'HV', 'HW', 'OH', 'EX', 'IN'}
GRAIN = {'FS', 'CS', 'MW', 'VFS', 'VCS'}
r1 = geol.FORMATION.notna() & ~geol.FORMATION.isin(FORM)
r2 = pd.to_numeric(geol.TEXTURE, errors='coerce').notna()
r3 = geol.LITH1_legacy.isin(GRAIN) | geol.Lith1_Validated.isin(GRAIN)
shifted = r1 | r2 | r3
before = geol.Lith1_Validated.fillna(geol.LITH1_legacy)
after = before.where(~shifted, geol.LITH2_legacy)
S = geol.loc[shifted, ['hole', 'f', 't', 'FORMATION', 'TEXTURE', 'LITH1_legacy', 'Lith1_Validated', 'LITH2_legacy']].copy()
S['lithology_read_before'] = before[shifted]; S['lithology_read_after'] = after[shifted]
S['MS_before'] = before[shifted].isin(['MS', 'SM']); S['MS_after'] = after[shifted].isin(['MS', 'SM'])
S['rule'] = np.select([r1[shifted] & r2[shifted], r1[shifted] & r3[shifted], r1[shifted], r2[shifted], r3[shifted]],
                      ['FORMATION not a code + numeric TEXTURE', 'FORMATION not a code + grain-size code in LITH1',
                       'FORMATION not a code', 'numeric TEXTURE', 'grain-size code in LITH1'], 'other')
S['formation_after'] = 'uncoded'
S.to_csv(SUP / 'S2_log_corrections.csv', index=False)
print('S2 rows', len(S), 'holes', S.hole.nunique(), '; MS before', int(S.MS_before.sum()), 'after', int(S.MS_after.sum()))

# ------------------------------------------------------------------ S3: scenario manifest
R = '80 x 80 x 10 m (s, z, n)'
rows = []
for v, th, rng, rot in (('reference', '0.3 / 0.5 / 0.7', R, '-'), ('ranges_x0.75', '0.3 / 0.5 / 0.7', '60 x 60 x 7.5 m', '-'),
                        ('ranges_x1.25', '0.3 / 0.5 / 0.7', '100 x 100 x 12.5 m', '-'), ('across_7.5m', '0.3 / 0.5 / 0.7', '80 x 80 x 7.5 m', '-'),
                        ('across_12.5m', '0.3 / 0.5 / 0.7', '80 x 80 x 12.5 m', '-'), ('elong_strike', '0.3 / 0.5 / 0.7', '113 x 57 x 10 m', 'long axis along s'),
                        ('elong_dip', '0.3 / 0.5 / 0.7', '57 x 113 x 10 m', 'long axis along z'), ('elong_plunge+45', '0.3 / 0.5 / 0.7', '113 x 57 x 10 m', 'long axis +45 deg in s-z'),
                        ('elong_plunge-45', '0.3 / 0.5 / 0.7', '113 x 57 x 10 m', 'long axis -45 deg in s-z'), ('dip_+10', '0.3 / 0.5 / 0.7', R, '+10 deg in z-n'),
                        ('dip_-10', '0.3 / 0.5 / 0.7', R, '-10 deg in z-n')):
    rows.append(dict(set='indicator sensitivity (MS and OH; 11 fits x 3 thresholds = 33)', construction='indicator', variant=v,
                     threshold_or_level=th, ranges=rng, rotation=rot, data='all holes'))
for v, lev, rng, rot, data in (('threshold_0.3', '0.3', R, '-', 'all'), ('threshold_0.5', '0.5 (reference)', R, '-', 'all'),
                               ('threshold_0.7', '0.7', R, '-', 'all'), ('ranges_x0.75', '0.5', '60 x 60 x 7.5 m', '-', 'all'),
                               ('ranges_x1.25', '0.5', '100 x 100 x 12.5 m', '-', 'all'), ('without_frozen', '0.5', R, '-', 'original withheld set removed')):
    rows.append(dict(set='grade-tonnage geometries', construction='indicator', variant=v, threshold_or_level=lev, ranges=rng, rotation=rot, data=data))
for v, lev, rng, rot, data in (('reference', 'field >= 20 (contact value)', R, '-', 'all'), ('contacts_out_1m', 'field >= 19', R, '-', 'all'),
                               ('contacts_in_1m', 'field >= 21', R, '-', 'all'), ('ranges_x0.75', '20', '60 x 60 x 7.5 m', '-', 'all'),
                               ('ranges_x1.25', '20', '100 x 100 x 12.5 m', '-', 'all'), ('across_7.5m', '20', '80 x 80 x 7.5 m', '-', 'all'),
                               ('across_12.5m', '20', '80 x 80 x 12.5 m', '-', 'all'), ('elong_strike', '20', '113 x 57 x 10 m', 'along s', 'all'),
                               ('elong_dip', '20', '57 x 113 x 10 m', 'along z', 'all'), ('elong_plunge+45', '20', '113 x 57 x 10 m', '+45 deg', 'all'),
                               ('elong_plunge-45', '20', '113 x 57 x 10 m', '-45 deg', 'all'), ('dip_+10', '20', R, '+10 deg in z-n', 'all'),
                               ('dip_-10', '20', R, '-10 deg in z-n', 'all'), ('without_frozen', '20', R, '-', 'original withheld set removed')):
    rows.append(dict(set='grade-tonnage geometries', construction='signed', variant=v, threshold_or_level=lev, ranges=rng, rotation=rot, data=data))
for v, lev, rng, rot, data in (('reference', 'w >= 1 m', '80 x 80 m (s, z)', '-', 'all'), ('tmin_0.5m', 'w >= 0.5 m', '80 x 80 m', '-', 'all'),
                               ('tmin_2m', 'w >= 2 m', '80 x 80 m', '-', 'all'), ('ranges_x0.75', 'w >= 1 m', '60 x 60 m', '-', 'all'),
                               ('ranges_x1.25', 'w >= 1 m', '100 x 100 m', '-', 'all'), ('elong_strike', 'w >= 1 m', '113 x 57 m', 'along s', 'all'),
                               ('elong_dip', 'w >= 1 m', '57 x 113 m', 'along z', 'all'), ('elong_plunge+45', 'w >= 1 m', '113 x 57 m', '+45 deg', 'all'),
                               ('elong_plunge-45', 'w >= 1 m', '113 x 57 m', '-45 deg', 'all'), ('contacts_out_1m', 'w + 2 m per intercept', '80 x 80 m', '-', 'all'),
                               ('contacts_in_1m', 'w - 2 m per intercept', '80 x 80 m', '-', 'all'), ('without_frozen', 'w >= 1 m', '80 x 80 m', '-', 'original withheld set removed')):
    rows.append(dict(set='grade-tonnage geometries', construction='surfaces', variant=v, threshold_or_level=lev, ranges=rng, rotation=rot, data=data))
for v, lev, rng, data in (('reference', 'w >= 1 m', '80 x 80 m (s, z)', 'all'), ('contacts_out_1m', 'w + 2 m per intercept', '80 x 80 m', 'all'),
                          ('contacts_in_1m', 'w - 2 m per intercept', '80 x 80 m', 'all'), ('ranges_x0.75', 'w >= 1 m', '60 x 60 m', 'all'),
                          ('ranges_x1.25', 'w >= 1 m', '100 x 100 m', 'all'), ('without_frozen', 'w >= 1 m', '80 x 80 m', 'original withheld set removed')):
    rows.append(dict(set='grade-tonnage geometries', construction='lenses', variant=v, threshold_or_level=lev, ranges=rng, rotation='-', data=data))
pd.DataFrame(rows).to_csv(SUP / 'S3_scenarios.csv', index=False)
print('S3 rows', len(rows))

# ------------------------------------------------------------------ S4: withheld sets (same rule as model_withheld_draws.py)
comp = pd.read_csv(OUT / 'composites.csv'); hl = pd.read_csv(OUT / 'holes.csv'); hl = hl[hl.HOLE_ID.isin(set(comp.hole))]
frozen = pd.read_csv(OUT.parent / 'frozen_holes.csv', comment='#').hole_id.astype(str) if (OUT.parent / 'frozen_holes.csv').exists() else None
blk = (np.floor(hl.X_MGA / 50).astype(int).astype(str) + '_' + np.floor(hl.Y_MGA / 50).astype(int).astype(str)).to_numpy()
ub = np.unique(blk); out = []
if frozen is not None:
    out += [dict(set='original', seed='', hole_id=h) for h in sorted(frozen)]
for seed in range(1, 6):
    order = np.random.default_rng(seed).permutation(ub); chosen = set()
    for b in order:
        chosen |= set(hl.HOLE_ID[blk == b].astype(str))
        if len(chosen) >= 274: break
    out += [dict(set=f'draw{seed}', seed=seed, hole_id=h) for h in sorted(chosen)]
W = pd.DataFrame(out); W.to_csv(SUP / 'S4_withheld_sets.csv', index=False)
print('S4', W.groupby('set').size().to_dict(), '; pool of logged holes', len(hl))
