# Global OceanVal example (entirely synthetic data)

A complete [OceanVal](https://github.com/pmlmodelling/OceanVal) validation of a **global** model, built
from **synthetic data only**. No real model output and no real observations are used.
The data are made from formulae of latitude, longitude, depth and season (plus smooth random fields) on
real coastlines, so the maps look like the real ocean: warm tropics, subtropical salinity maxima,
nutrient-rich high latitudes and upwelling zones, empty gyres, western-boundary warm tongues.

The data provider is called **Zissou Oceanographic** throughout.

## The finished report

Open [`report/index.html`](report/index.html) (it forwards to the summary page). `report/` holds
**HTML only**: no matchups, results or data files.

## What is validated

Everything is surface-only, except temperature, which is also validated through the water column.

| Variable | Stand-in for | Layers |
|---|---|---|
| Temperature | COBE2 (monthly series) | surface |
| Temperature | WOA23 (climatology) | surface + water column (8 depths to 700 m) |
| Temperature | Argo/CTD-like profiles (point data, csv) | surface + water column |
| Salinity, nitrate, phosphate, silicate, oxygen | WOA23 | surface |
| Chlorophyll, KD490 | OCCCI (monthly series) | surface |
| Alkalinity, pH | GLODAPv2.2016b (climatology) | surface |

The "observations" use the built-in datasets' variable names, units and file layouts, but are read from
disk, so nothing is downloaded. The model is the synthetic truth plus smooth, variable-specific biases and
its own noise; each observational dataset is the truth plus independent noise and gaps (polar night for
ocean colour, sparse high latitudes), so the validation has real differences to find.

## Re-making it

Needs OceanVal and its dependencies (see its `oceanval.yml`) plus `global-land-mask` (only to rebuild the
land mask).

```bash
./run_all.sh /some/scratch/dir
```

This generates the data in `data/` (about 100 MB, ~20 s, deterministic and git-ignored), runs
`matchup.py` and `validate.py` in the scratch directory (about 15 minutes), and copies the HTML to `report/`.
Run it in a scratch directory, not here: OceanVal writes `oceanval_matchups/`, `oceanval_report/` and
`oceanval_results/` into the working directory and replaces them.

| File | What it does |
|---|---|
| `make_synthetic_data.py` | writes the model, the gridded and point observations and the land mask |
| `matchup.py` | registers the 11 gridded and 1 point comparisons, runs `oceanval.matchup` |
| `validate.py` | builds the HTML report |
| `collect_report.py` | keeps only the HTML in `report/` |
| `data/land_mask.nc` | 1° land mask from real coastlines (committed) |
