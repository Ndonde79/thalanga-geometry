"""Thalanga data preparation for the geometry paper (all holes, no split).

Inputs: Red River Resources Thalanga ML database as released through the GSQ (collar, survey, geology, assays).
Outputs in out/: holes.csv, composites.csv (2 m), contacts.csv, prep_report.json.

Conventions found in the data: dip positive = down (all surface collars have dips 45-90), negative = up holes
from underground; azimuths above 360 are wrapped.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(sys.argv[1])
OUT = Path(__file__).with_name('out'); OUT.mkdir(exist_ok=True)
COMP = 2.0
rep = {}

collar = pd.read_csv(SRC / 'ThalangaML_collar.csv')
survey = pd.read_csv(SRC / 'ThalangaML_survey.csv').rename(columns={'Drillhole ID': 'hole', 'Depth': 'depth', 'Dip': 'dip', 'Azimuth': 'azi'})
geol = pd.read_csv(SRC / 'ThalangaML_Geology.csv').rename(columns={'Drillhole ID': 'hole', 'From': 'f', 'To': 't'})
assay = pd.read_csv(SRC / 'ThalangaML_Assays.csv', low_memory=False).rename(columns={'Drillhole ID': 'hole', 'From': 'f', 'To': 't'})

rep['collars_all'] = len(collar)
rep['hole_types_all'] = collar.HOLE_TYPE.value_counts().to_dict()
keep = collar[collar.HOLE_TYPE.isin(['DD', 'RC', 'PC'])].copy()
keep = keep[keep.HOLE_ID.isin(survey.hole)]
rep['holes_kept'] = len(keep)

# ---------------------------------------------------------------- desurvey (minimum curvature)
survey['azi'] = survey.azi % 360.0
survey = survey.dropna(subset=['depth', 'dip', 'azi']).sort_values(['hole', 'depth']).drop_duplicates(['hole', 'depth'])


def unit(dip, azi):
    inc = np.radians(dip)               # dip positive = down
    a = np.radians(azi)
    return np.array([np.cos(inc) * np.sin(a), np.cos(inc) * np.cos(a), -np.sin(inc)])


dropped_stations = 0
traj = {}
for h, row in keep.set_index('HOLE_ID').iterrows():
    s = survey[survey.hole == h]
    st = s[['depth', 'dip', 'azi']].to_numpy(float)
    # drop stations whose direction jumps > 20 deg over < 30 m from both neighbours (implausible deviation)
    if len(st) >= 3:
        ok = np.ones(len(st), bool)
        U = np.array([unit(d, a) for _, d, a in st])
        for i in range(1, len(st) - 1):
            a1 = np.degrees(np.arccos(np.clip(U[i] @ U[i - 1], -1, 1)))
            a2 = np.degrees(np.arccos(np.clip(U[i] @ U[i + 1], -1, 1)))
            a0 = np.degrees(np.arccos(np.clip(U[i - 1] @ U[i + 1], -1, 1)))
            if a1 > 20 and a2 > 20 and a0 < 10 and st[i + 1, 0] - st[i - 1, 0] < 60:
                ok[i] = False
        dropped_stations += int((~ok).sum())
        st = st[ok]
    if st[0, 0] > 0:
        st = np.vstack([[0.0, st[0, 1], st[0, 2]], st])
    if st[-1, 0] < row.DEPTH:
        st = np.vstack([st, [row.DEPTH, st[-1, 1], st[-1, 2]]])
    P = [np.array([row.X_MGA, row.Y_MGA, row.Z_MGA], float)]
    for i in range(1, len(st)):
        md = st[i, 0] - st[i - 1, 0]
        u1, u2 = unit(st[i - 1, 1], st[i - 1, 2]), unit(st[i, 1], st[i, 2])
        beta = np.arccos(np.clip(u1 @ u2, -1, 1))
        rf = 1.0 if beta < 1e-9 else 2 / beta * np.tan(beta / 2)
        P.append(P[-1] + md / 2 * (u1 + u2) * rf)
    traj[h] = (st[:, 0], np.array(P))
rep['survey_stations_dropped'] = dropped_stations


def xyz(h, depth):
    d, P = traj[h]
    depth = np.clip(np.asarray(depth, float), d[0], d[-1])
    return np.column_stack([np.interp(depth, d, P[:, k]) for k in range(3)])


# ---------------------------------------------------------------- geology codes
FORM = {'FV': 'FV', 'FW': 'FV', 'HV': 'HV', 'HW': 'HV', 'OH': 'OH', 'EX': 'OH', 'IN': 'IN'}
geol = geol[geol.hole.isin(traj)].copy()
geol['form'] = geol.FORMATION.map(FORM)
# Some legacy logs (mostly W45 / Far West, e.g. TH784, TH801) are shifted one or
# more columns to the right: free text sits in FORMATION, grain-size codes
# (FS/MS/CS = fine/medium/coarse) sit in LITH1_legacy and the lithology code sits
# in LITH2_legacy.  Read as-is, "MS" (medium grained) becomes massive sulphide;
# those intervals assay ~0.5% S and ~0.05% Zn+Pb+Cu.  Detect the shifted rows and
# take the lithology from LITH2_legacy instead; their formation stays uncoded.
VALID_FORM = set(FORM)
GRAIN = {'FS', 'CS', 'MW', 'VFS', 'VCS'}
shifted = ((geol.FORMATION.notna() & ~geol.FORMATION.isin(VALID_FORM))
           | pd.to_numeric(geol.TEXTURE, errors='coerce').notna()
           | geol.LITH1_legacy.isin(GRAIN) | geol.Lith1_Validated.isin(GRAIN))
lith = geol.Lith1_Validated.fillna(geol.LITH1_legacy).where(~shifted, geol.LITH2_legacy)
geol['ms'] = lith.isin(['MS', 'SM']).astype(float)
rep['geology_rows_column_shifted'] = int(shifted.sum())
rep['geology_holes_with_shifted_rows'] = int(geol.loc[shifted, 'hole'].nunique())
geol.loc[geol.form.isna() & geol.ms.eq(1), 'form'] = 'OH'
rep['geology_intervals'] = len(geol)
rep['formation_coded_fraction'] = round(float(geol.form.notna().mean()), 4)
rep['holes_logged'] = int(geol.hole.nunique())

# ---------------------------------------------------------------- 2 m composites (length-weighted fractions, grades)
for el in ['Zn_pct', 'Pb_pct', 'Cu_pct']:
    assay[el] = pd.to_numeric(assay[el], errors='coerce').clip(lower=0)
assay = assay[assay.hole.isin(traj)]

rows = []
for h, gh in geol.groupby('hole'):
    gh = gh.sort_values('f')
    ah = assay[assay.hole == h].sort_values('f')
    top, bot = gh.f.min(), gh.t.max()
    edges = np.arange(np.floor(top / COMP) * COMP, bot + COMP, COMP)
    for a, b in zip(edges[:-1], edges[1:]):
        g = gh[(gh.t > a) & (gh.f < b)]
        if g.empty:
            continue
        w = (np.minimum(g.t, b) - np.maximum(g.f, a)).clip(lower=0).to_numpy()
        if w.sum() < 0.5 * COMP:
            continue
        fr = {k: float(w[(g.form == k).to_numpy()].sum() / w.sum()) for k in ['HV', 'OH', 'FV', 'IN']}
        coded = sum(fr.values())
        r = dict(hole=h, f=a, t=b, coded=coded, f_ms=float((w * g.ms.to_numpy()).sum() / w.sum()), **{'f_' + k: v for k, v in fr.items()})
        aa = ah[(ah.t > a) & (ah.f < b)]
        if not aa.empty:
            wa = (np.minimum(aa.t, b) - np.maximum(aa.f, a)).clip(lower=0).to_numpy()
            for el in ['Zn_pct', 'Pb_pct', 'Cu_pct']:
                v = aa[el].to_numpy(); m = np.isfinite(v) & (wa > 0)
                r[el] = float((v[m] * wa[m]).sum() / wa[m].sum()) if wa[m].sum() >= 0.5 * COMP else np.nan
        rows.append(r)
comp = pd.DataFrame(rows)
P = np.vstack([xyz(h, (g.f + g.t).to_numpy() / 2) for h, g in comp.groupby('hole', sort=False)])
comp[['X', 'Y', 'Z']] = P
comp['form'] = comp[['f_HV', 'f_OH', 'f_FV', 'f_IN']].idxmax(axis=1).str[2:].where(comp.coded >= 0.5)
comp.to_csv(OUT / 'composites.csv', index=False)
rep['composites'] = len(comp)
rep['composites_by_form'] = comp.form.value_counts(dropna=False).to_dict()
rep['composites_ms_majority'] = int((comp.f_ms >= 0.5).sum())
rep['composites_with_zn'] = int(comp.Zn_pct.notna().sum())

# ---------------------------------------------------------------- contacts along each hole (formation changes)
cts = []
for h, gh in geol[geol.form.isin(['HV', 'OH', 'FV'])].groupby('hole'):
    gh = gh.sort_values('f')
    prev = None
    for _, r in gh.iterrows():
        if prev is not None and r.form != prev.form and r.f - prev.t <= 2.0:
            pair = {prev.form, r.form}
            kind = 'OH_top' if pair == {'HV', 'OH'} else 'OH_base' if pair == {'OH', 'FV'} else 'HV_FV'
            cts.append(dict(hole=h, depth=(prev.t + r.f) / 2, kind=kind, above=prev.form, below=r.form))
        prev = r
    # massive sulphide contacts
ms = []
for h, gh in geol.groupby('hole'):
    gh = gh.sort_values('f'); v = gh.ms.to_numpy(); f = gh.f.to_numpy(); t = gh.t.to_numpy()
    for i in range(1, len(gh)):
        if v[i] != v[i - 1] and f[i] - t[i - 1] <= 2.0:
            cts.append(dict(hole=h, depth=(t[i - 1] + f[i]) / 2, kind='MS_edge', above='MS' if v[i - 1] else 'x', below='MS' if v[i] else 'x'))
ct = pd.DataFrame(cts).sort_values(['hole', 'depth'], kind='stable').reset_index(drop=True)
ct[['X', 'Y', 'Z']] = np.vstack([xyz(h, g.depth.to_numpy()) for h, g in ct.groupby('hole', sort=False)])
ct.to_csv(OUT / 'contacts.csv', index=False)
rep['contacts'] = ct.kind.value_counts().to_dict()

holes = keep[keep.HOLE_ID.isin(geol.hole)][['HOLE_ID', 'X_MGA', 'Y_MGA', 'Z_MGA', 'DEPTH', 'HOLE_TYPE', 'PROSPECT']]
holes.to_csv(OUT / 'holes.csv', index=False)
rep['extent_m'] = {k: [round(float(comp[k].min()), 1), round(float(comp[k].max()), 1)] for k in 'XYZ'}
json.dump(rep, open(OUT / 'prep_report.json', 'w'), indent=1)
print(json.dumps(rep, indent=1))
