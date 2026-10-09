"""Null baseline for the PELT detection frequency (Supplementary Fig. panel c).

Question: how high can the pooled detection frequency get when the series has no
dated break structure but the same step-size distribution and the same short-range
dependence as the observed series?

Null model (block bootstrap of first differences):
  1. Take the year-to-year differences of mean latitude and of n_south.
  2. Cut them into blocks of BLOCK years and reshuffle the blocks (the same block
     order for both series, so their cross-correlation is kept).
  3. Cumulate from the observed 221 BCE value.
The simulated series keep the overall drift, the step sizes, and dependence within
BLOCK years. The timing of large steps is random.

Each simulated pair goes through exactly the grid and the pooling used for the
observed series (L1/L2, reduced to 8 penalties x 4 minimum segment lengths, +/-15 yr;
the observed curve is recomputed on the same subset).
We record the height of the highest and second-highest peak (peaks >= 100 yr apart)
of the pooled detection frequency, and compare them with the observed values.

Usage:
    python3 pelt_null_baseline.py [N_SIM]        # default 100; a few minutes on 10 cores
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker; avoids CPU oversubscription

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import find_peaks

import pelt_location_robust as P

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BLOCK = 10
SEED = 20261008
# Reduced grid for the null (a subset of the main grid; the observed frequency is
# recomputed on the same subset). 8 x 4 x 2 models = 64 specifications per series.
NULL_PENS = [0.75, 1.5, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0]
NULL_MIN_SIZES = [30, 60, 100, 200]


def pooled_rate(years, lat, nsouth) -> np.ndarray:
    combos = list(itertools.product(NULL_PENS, NULL_MIN_SIZES, P.JUMPS, P.MODELS))
    rows = []
    for col, sig in (("lat_mean", lat), ("n_south", nsouth)):
        for pen, ms, jump, model in combos:
            bk = P.pelt_years(years, sig, pen, ms, jump, model)
            spec = dict(series=col, pen=pen, min_size=ms, jump=jump, model=model)
            if not bk:
                rows.append({**spec, "breakpoint": np.nan})
            rows.extend({**spec, "breakpoint": b} for b in bk)
    det = pd.DataFrame(rows)
    return P.breakpoint_probability(years, det, P.POOL_SERIES, half=P.PROB_HALF)


def top_peaks(rate: np.ndarray, k: int = 2, distance: int = 100) -> list[float]:
    idx, _ = find_peaks(rate, distance=distance)
    h = sorted(rate[idx], reverse=True)[:k]
    return h + [0.0] * (k - len(h))


def block_shuffle_diffs(d_lat, d_ns, rng) -> tuple[np.ndarray, np.ndarray]:
    n = len(d_lat)
    nb = int(np.ceil(n / BLOCK))
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(b * BLOCK, min((b + 1) * BLOCK, n)) for b in order])
    return d_lat[idx], d_ns[idx]


def one_sim(seed: int, years, lat0, ns0, d_lat, d_ns) -> dict:
    rng = np.random.default_rng(seed)
    sl, sn = block_shuffle_diffs(d_lat, d_ns, rng)
    lat = lat0 + np.concatenate([[0.0], np.cumsum(sl)])
    ns = ns0 + np.concatenate([[0.0], np.cumsum(sn)])
    rate = pooled_rate(years, lat, ns)
    p1, p2 = top_peaks(rate)
    return {"seed": seed, "peak1": p1, "peak2": p2, "rate": rate}


def main(n_sim: int) -> None:
    a = pd.read_csv(DATA / "annual_location_series.csv")
    years = a.year.to_numpy()
    lat = a.lat_mean.to_numpy(float)
    ns = a.n_south.to_numpy(float)
    d_lat, d_ns = np.diff(lat), np.diff(ns)

    det = pd.read_csv(DATA / "pelt_detections.csv")
    det = det[det.pen.isin(NULL_PENS) & det.min_size.isin(NULL_MIN_SIZES)]
    obs = P.breakpoint_probability(years, det, P.POOL_SERIES, half=P.PROB_HALF)
    o1, o2 = top_peaks(obs)

    seeds = [SEED + i for i in range(n_sim)]
    res = Parallel(n_jobs=10, verbose=5)(
        delayed(one_sim)(s, years, lat[0], ns[0], d_lat, d_ns) for s in seeds
    )
    summ = pd.DataFrame([{k: r[k] for k in ("seed", "peak1", "peak2")} for r in res])
    summ.to_csv(DATA / "null_peak_heights.csv", index=False)
    curves = np.vstack([r["rate"] for r in res])
    pd.DataFrame({
        "year": years,
        "observed": obs,
        "null_mean": curves.mean(0),
        "null_p95": np.quantile(curves, 0.95, axis=0),
        "null_p99": np.quantile(curves, 0.99, axis=0),
    }).to_csv(DATA / "null_frequency_curves.csv", index=False)

    print(f"\nobserved: peak1 = {o1:.3f}, peak2 = {o2:.3f}")
    for name, o in (("peak1", o1), ("peak2", o2)):
        v = summ[name].to_numpy()
        p = (np.sum(v >= o) + 1) / (len(v) + 1)
        print(f"null {name}: median {np.median(v):.3f}, p95 {np.quantile(v, .95):.3f}, "
              f"max {v.max():.3f};  one-sided p = {p:.3f}  (n = {len(v)})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
