# Location breakpoints (PELT robustness)

Final figure for the southward shift of active administrative seats
in interior China, 221 BCE–1911 CE.

## Figure

`figures/pelt_location_robust.png` (also `.pdf`)

- **a** Mean latitude of seats active in that year
- **b** Number of active seats south of 33 N
- **c** Breakpoint probability (detection frequency): share of PELT
  specifications on (a) and (b) that place at least one break within
  ±15 years of *t*. It is a share of parameter settings, not a statistical
  probability; settings are not independent.

Dotted lines mark 283 CE and 623 CE.

## Reproduce

```text
python3 pelt_location_robust.py --plot-only   # redraw from saved detections
python3 pelt_location_robust.py               # rerun the full PELT grid (~5 min)
python3 pelt_null_baseline.py 100             # null baseline, 100 simulations (~25 min, 12 cores)
```

Requires: `numpy`, `pandas`, `matplotlib`, `ruptures`, `scikit-learn`.

Grid: L1 and L2, `jump=5`, penalty multipliers 0.75–12 (23 values),
minimum segment 30–200 years (9 values). 414 specifications per series;
series are `lat_mean`, `n_south`, `south_share`, and the joint
(`lat_mean`, `n_south`) after z-scoring.

## Data

| file | contents |
|------|----------|
| `data/annual_location_series.csv` | Yearly location series |
| `data/pelt_detections.csv` | Every PELT break (or empty spec) |
| `data/breakpoint_probability.csv` | Panel-c series |
| `data/pelt_hitrate.csv` | Hit rate within ±20 years of selected years |
| `data/pelt_detection_check.csv` | Per series and model: empty specs, breaks per spec |
| `data/null_peak_heights.csv` | Null baseline: top-two peak heights per simulation |
| `data/null_frequency_curves.csv` | Observed frequency and null mean / 95th / 99th percentile |
| `data/peak_table.csv` | Peaks of the pooled detection frequency with per-series rates and step sizes |

`annual_location_series.csv` columns: `year`, `n_active`, `lat_mean`,
`n_south`, `south_share` (`n_south / n_active`). A seat is active in
year *y* if `BEG_YR ≤ y ≤ END_YR`. Built from
`cities221bc2023ad.shp` (Beijing 1954). The raw shapefile is not redistributed here;
the yearly series derived from it is included, so every step from the series onward
reruns from this repository.

Penalty used by each specification:

`pen = log(n) * scale * multiplier`

with `scale = variance(series)` for L2 and `std(series)` for L1, because the
L2 cost is in squared units and the L1 cost in absolute units. (v1 used the
variance for both: L1 on `n_south` found no break in any of its 207
specifications, and L1 on `south_share` over-detected. Fixed 2026-10-08;
`pelt_location_robust.py` now prints a per-series empty-spec check after every run.)

Null baseline (`pelt_null_baseline.py`): block bootstrap (10-yr blocks) of the
yearly first differences of `lat_mean` and `n_south`, same block order for both,
cumulated from the 221 BCE value; each simulated pair goes through the same
grid and pooling as panel c.

## License

MIT (see `LICENSE`).
