"""Reproduce every result and figure of the paper.

usage:  python run_all.py <drilling_dir> <ip_csv> <paulick_dir>

  drilling_dir  folder with ThalangaML_collar.csv, ThalangaML_survey.csv, ThalangaML_Geology.csv, ThalangaML_Assays.csv
  ip_csv        'Dipole-Dipole IP-Res Inversion Point data.csv'
  paulick_dir   folder with the six 'Alteration ... - Paulick 1999.tif' sections
All three come from GSQ dataset DS100103 ('Thalanga 3D Model Source Data' ZIP). DeepRBF 3.5.0 must be importable
(module deeprbf_recovered). Outputs go to scripts/out.
"""
import subprocess, sys, time
from pathlib import Path

S = Path(__file__).with_name('scripts')
drill, ip, paulick = sys.argv[1:4]
DOMS = ['FarWest', 'West', 'East', 'FarEast']


def run(*args):
    t = time.time(); print('>>', ' '.join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=S, check=True)
    print(f'   {time.time() - t:.0f} s', flush=True)


run('prep.py', drill)                                         # step 0: audit, desurvey, composites, contacts
for d in DOMS:                                                # step 1: reference geometry (indicator, MS and OH)
    run('model_unfold.py', d); run('plot_domain.py', d)
run('summary.py')
for d in DOMS:                                                # step 2: indicator sensitivity (33 geometries per group)
    run('model_sensitivity.py', d)
run('plot_sensitivity.py')
for d in DOMS:                                                # steps 3-4: contact constructions, variants, withheld holes
    run('model_contacts.py', d)
for d in DOMS:                                                # step 3b: correlated lenses (fourth construction)
    run('model_lenses.py', d)
for d in DOMS:                                                # step 4b: five further withheld-hole draws
    run('model_withheld_draws.py', d)
run('physvol.py', 'out')                                       # physical vs frame volume (Jacobian)
collar = str(Path(drill) / 'ThalangaML_collar.csv')
run('comp_density.py', str(Path(drill) / 'ThalangaML_Assays.csv'), 'out')   # normative density
run('grade_tonnage.py', 'out')                                 # step 5: kriging and grade-tonnage
run('checks.py', 'out', collar, ip, paulick)                   # tables for steps 3, 4, 6
run('ip_dist.py', ip)                                          # IP against distance to modelled MS
run('manifests.py', drill, 'out', 'out/supplement')           # Supplementary Tables S2-S4
run('rev_tables.py', 'out', 'out/supplement')                 # Tables 3-7, S5
run('figs.py'); run('paper_figs.py', drill)                    # figures
# optional, slow (hours on 2 cores): fine-grid endpoint check, Table 9
# for d in DOMS: run('model_fine_ep.py', d)
print('done: results in scripts/out, paper figures in scripts/out/paper_figs')
