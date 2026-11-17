# thalanga-geometry

Code for the paper *Boundary placement and tonnage sensitivity in implicit models of thin massive sulphide lenses: the Thalanga VHMS deposit, Queensland* (R.W. Ndonde, submitted to *Natural Resources Research*).

The scripts rebuild every number, table and figure in the paper from the public Thalanga drillhole database: the data audit, four implicit constructions of the massive sulphide lenses (indicator, signed distance, one contact sheet per hole, correlated lenses), their sensitivity to defensible modelling choices, refits without six blocked sets of withheld drillholes, and ordinary kriging and grade–tonnage inside every geometry.

## Data

Nothing from the deposit is stored here. Download GSQ dataset **DS100103**, *NEQ Deposit Atlas – Thalanga* (Geological Survey of Queensland Open Data Portal, Creative Commons Attribution licence), resource *Thalanga 3D Model Source Data* (ZIP). The scripts need from it:

| Input | Path in the ZIP |
|---|---|
| Collars, surveys, geology, assays | `Local/Drilling/ThalangaML_{collar,survey,Geology,Assays}.csv` |
| IP-resistivity inversion | `Local/Geophysics/IP-Resistivity/Dipole-Dipole IP-Res Inversion Point data.csv` |
| Paulick (1999) sections | `Local/Geology/Alteration * - Paulick 1999.tif` |

`scripts/frozen_holes.csv` lists the original 274 withheld drillholes (hole IDs only), chosen in 50 m collar blocks; the five further sets are drawn by `model_withheld_draws.py` (seeds 1–5) and listed in Supplementary Table S4.

## Requirements

- Python 3.11 with numpy, pandas, scipy, matplotlib and pillow (`requirements.txt`)
- DeepRBF 3.5.0 (BSD 3-Clause; https://doi.org/10.5281/zenodo.24268959), with its Python module `deeprbf_recovered` on the path

## Run

```
python run_all.py <drilling_dir> <ip_csv> <paulick_dir>
```

Results go to `scripts/out`, paper figures to `scripts/out/paper_figs`. The full run takes under an hour on a desktop workstation; every step can also be run on its own (see `run_all.py`).

## What each script does

| Step | Script | Paper |
|---|---|---|
| Audit, desurvey, 2 m composites, contacts | `prep.py` (column-shift correction included) | §2.2–2.3, Fig. 2 |
| Unfolded frame | `unfold.py` | §3.1 |
| Reference indicator geometry, MS and OH | `model_unfold.py`, `plot_domain.py`, `summary.py` | §4.1, Table 3, Fig. 4 |
| Indicator sensitivity (33 geometries per lens group) | `model_sensitivity.py`, `plot_sensitivity.py` | §4.2, Fig. 5 |
| True thickness factor \|u·N\| | `truewidth.py` | §3.1 |
| Signed distance and contact surfaces, variants, withheld-hole refit | `model_contacts.py` | §3.3–3.5, §4.3–4.4, Figs 6–8 |
| Correlated lenses, variants, withheld-hole refit | `model_lenses.py` | §3.3, §4.3–4.4 |
| Physical (Jacobian) volume | `physvol.py` | §3.1 |
| Fine-grid endpoint check (slow) | `model_fine_ep.py` | §4.3, Table 9 |
| Five further withheld-hole draws | `model_withheld_draws.py` | §3.5, §4.4 |
| Normative density | `comp_density.py` | §2.4 |
| Ordinary kriging and grade–tonnage | `grade_tonnage.py` | §4.5, Table 7, Fig. 9 |
| Summary tables, Paulick sections, IP | `checks.py`, `ip_dist.py` | §4.6, Table 8, Fig. 10 |
| Paper tables and Supplementary Tables S2–S5 | `rev_tables.py`, `manifests.py` | Tables 3–7, S2–S5 |
| Figures | `figs.py` (Figs 7, 9), `paper_figs.py` (Figs 1–6, 8, 10) | |

## Notes

- About 6,500 legacy geology rows in the database are shifted one or more columns to the right. Read as recorded, the grain-size code *MS* (medium grained) becomes massive sulphide. `prep.py` detects these rows and reads the rock code from the next field; without it, Far West massive sulphide is overstated about thirty-fold.
- Indicators are fitted as plain 0/1 so that far from data the field falls to *outside*; a centred indicator (y − 0.5) grows false bodies where drilling is sparse.
- Run on Windows by double-click: set `PATH` to include the Python environment's `Library\bin` folders (as an activated conda prompt does), or numpy's MKL may fail to load.

## Supplementary tables

`supplement/` holds Supplementary Tables S1 (calculation specification), S2 (log-correction manifest), S3 (scenarios), S4 (withheld-hole sets) and S5 (per-scenario results).

## Licence and citation

BSD 3-Clause (see `LICENSE`). Please cite the paper and this repository (`CITATION.cff`).
