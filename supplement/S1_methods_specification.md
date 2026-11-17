# Supplementary Table S1 — Calculation specification

All scripts are in the code repository (`scripts/`). The order of the runs is in `run_all.py`.

## Data and preparation (`prep.py`)
- **Source:** GSQ DS100103 files ThalangaML_collar, survey, Geology and Assays.
- **Holes:** 2,131 diamond, RC and percussion holes. Excluded are sludge holes, costeans and holes without a hole type.
- **Desurvey:** minimum curvature. Survey stations with a direction change over 20° within 30 m are dropped (4 stations).
- **Log correction:** column-shifted rows are corrected as in Table S2.
- **MS code:** codes MS and SM count as massive sulphide (MS).
- **Composites:** 2 m. A composite is MS when MS makes up at least half its length.
- **Contacts:** the 3,842 contacts are the logged interval boundaries between MS and other codes, taken before compositing.
- **Top cuts:** grades are top-cut at the 99.5th percentile (Zn 36.9%, Pb 16.2%, Cu 22.9%).

## Frame (`unfold.py`)
- **Centreline:** the median northing of OH composites in 50 m easting bins, smoothed with a 5-bin moving mean and resampled to 5 m.
- **Forward mapping:**
  - s is the arc length of the closest centreline point;
  - n is the signed horizontal offset, positive north;
  - z = Z − z0, where z0 is the mean Z of the composites.
- **Inverse mapping:** (X, Y) = c(s) + n·N_h(s). The centreline is linearly interpolated, and its tangents are taken by central differences and linearly interpolated; N_h is turned to point north.
- **Physical volume (`physvol.py`):** each cell is weighted by |det ∂(X,Y)/∂(s,n)|, computed by central differences of the inverse mapping.

## Intercept thickness (`truewidth.py`)
- **Formula:** t = L·|u·N|.
- **u:** the hole direction in (s, z, n), the derivative of the composites' (s, z, n) with respect to downhole depth, normalised.
- **N:** the normal (−b, −c, 1)/‖·‖ of a plane n = a + b·s + c·z, least-squares fitted to the MS contacts within 60 m. The radius widens to 150 m when fewer than 8 contacts lie within 60 m.
- **Floor:** |u·N| is floored at 0.25.

## Interpolator
- **Engine:** DeepRBF 3.5.0 (doi:10.5281/zenodo.24268959).
- **Kernel:** generalised Cauchy, φ(r) = (1 + r²)^(−1). The distance r is taken after scaling the coordinates by ranges s 80 m, z 80 m and n 10 m.
- **Fit:** regularisation (nugget) 0.05; no polynomial drift.
- **Solver:** direct for Far West and Far East; GMRES for West and East, with an H-matrix product (ACA). Sampled-row residuals are checked against the exact product, with fallback to the exact product if the check fails.
- **Thinning:** coded composites more than 40 m from the horizon are thinned to one in four.
- **Grid:** 10 × 10 × 2.5 m in (s, z, n), for |n| ≤ 60 m, within 60 m of data.

## Constructions
| Construction | Data and rule |
|---|---|
| Indicator | Fitted to 0/1 MS (and OH); shell at the 0.5 level |
| Signed distance | Distance to the nearest MS contact in the same hole × \|u·N\|; + inside, − outside; capped at ±20 m; holes without MS take −20 m. The field is fitted to d + 20 and the contacts enter at 20; MS where the field ≥ 20 |
| Contact surfaces | Per hole: w = Σ intercept thickness, n_c = mean centre. Holes crossing the horizon (within 15 m) without MS give w = 0. w(s,z) and n_c(s,z) are interpolated in 2D within 60 m of a crossing hole. MS where \|n − n_c\| ≤ w/2 and w ≥ w_min (1 m) |
| Correlated lenses (`model_lenses.py`) | Intercepts: MS split at gaps > 2.5 m. Links: separation d ≤ 50 m in (s, z) and \|Δn_c\| ≤ 3 + 0.06d m, accepted in order of \|Δn_c\|/(3 + 0.06d) and merged by union–find; at most one intercept per hole per lens; a new link must agree with the lens's local centre. Each lens: w and n_c are interpolated from its intercepts, plus zeros at holes passing within 3 m of the lens position without MS; evaluated within 60 m of its intercepts; union over lenses |

## Scenarios
The scenarios are listed in Table S3. The ±1 m contact shift moves each MS contact outward or inward by 1 m along the hole, measured across the lens:
- for the surfaces and lenses: w ± 2 m;
- for the signed distance: the distances are shifted by ±1 m.

## Withheld sets (`model_withheld_draws.py`)
- **Selection:** collars are binned into 50 × 50 m blocks; whole blocks are taken in random order (numpy default_rng, seeds 1–5) until at least 274 holes are withheld.
- **Pool:** the 1,829 logged holes.
- **Held fixed:** the centreline, the grid and the evaluation mask.
- **Changed column:** any cell differs in that column.
- **Denominator:** columns in the union of the two footprints, binned by the distance to the nearest remaining hole that crosses the horizon.

## Density
- **Mineral masses (weight per cent):**
  - galena = Pb/0.866
  - sphalerite = Zn/0.671
  - chalcopyrite = Cu/0.346
  - barite = Ba/0.588
  - pyrite = max(0, S − S in the other minerals)/0.534
- **Gangue:** the remainder to 100%. If the minerals sum to more than 100%, they are rescaled.
- **Density:** ρ = 100/Σ(m_i/ρ_i), with ρ = 7.58, 4.05, 4.19, 4.48, 5.01 and 2.75 t/m³ for galena, sphalerite, chalcopyrite, barite, pyrite and gangue.
- **Composites without S:**
  - MS: ρ = 3.031 + 0.0155·(Zn+Pb+Cu) (n = 111, R² 0.32, RMSE 0.213)
  - Other rock: ρ = 2.769 + 0.0297·(Zn+Pb+Cu) (n = 11,123, R² 0.54, RMSE 0.061)

## Kriging (`grade_tonnage.py`)
- **Method:** ordinary point kriging at cell centres. Only MS composites are used (hard boundary).
- **Variogram:** one spherical model per variable (nugget plus one structure), fitted to directional experimental variograms in (s, z, n). The parameters are in `gt_variograms.json`.
- **Search:** up to 24 composites within 1.5 ranges, minimum 3.
- **Cells beyond the search:** they take the lens-group mean of the MS composites and are flagged as unsupported.
- **Tonnes and cut-offs:** tonnes = volume × kriged density. Zn cut-offs run from 0 to 10%, also repeated with density × 0.95.
