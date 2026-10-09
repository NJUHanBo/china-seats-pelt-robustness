"""PELT robustness for the location of active administrative seats.

Yearly series: mean latitude and the count of seats south of 33 N,
221 BCE to 1911 CE. The grid is L1/L2, jump = 5, penalty multipliers
0.75-12, and minimum segment lengths 30-200 years.

Breakpoint probability in panel c is the share of latitude and
n-south specifications that place at least one break within +/-15 years.

Usage:
    python3 pelt_location_robust.py --plot-only
    python3 pelt_location_robust.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"

PENS = [round(x, 2) for x in list(np.arange(0.75, 5.01, 0.25)) + [6, 7, 8, 10, 12]]
MIN_SIZES = [30, 40, 50, 60, 80, 100, 120, 150, 200]
JUMPS = [5]
MODELS = ["l2", "l1"]
UNI_SERIES = ["lat_mean", "n_south", "south_share"]
POOL_SERIES = ["lat_mean", "n_south"]
PROB_HALF = 15
MARK_YEARS = (283, 623)
HIT_WINDOWS = [
    (28, 20),
    (283, 20),
    (448, 20),
    (623, 20),
    (1388, 20),
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def scale_for(model: str, x: np.ndarray) -> float:
    """Scale of the segment cost, so that `pen` means the same thing in L1 and L2.

    The L2 cost is a sum of squared deviations (units of variance), the L1 cost a
    sum of absolute deviations (units of the standard deviation). The penalty
    log(n) * scale * multiplier therefore uses the variance for L2 and the
    standard deviation for L1. An earlier version used the variance for both,
    which left L1 on n_south with no detections at all (variance ~1e5 against
    a cost of order 1e2-1e3) and over-detected on south_share (variance ~0.02).
    """
    x = np.asarray(x, float)
    return float(np.var(x)) if model == "l2" else float(np.std(x))


def pelt_years(years: np.ndarray, signal: np.ndarray, pen: float, min_size: int,
               jump: int, model: str) -> list[int]:
    sig = np.asarray(signal, float)
    penalty = np.log(len(sig)) * scale_for(model, sig) * pen
    bkps = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(sig).predict(pen=penalty)
    return [int(years[b - 1]) for b in bkps[:-1]]


def pelt_joint(years: np.ndarray, X: np.ndarray, pen: float, min_size: int,
               jump: int, model: str) -> list[int]:
    Z = StandardScaler().fit_transform(X)
    penalty = np.log(len(Z)) * scale_for(model, Z) * pen
    bkps = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(Z).predict(pen=penalty)
    return [int(years[b - 1]) for b in bkps[:-1]]


def spec_keys(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["series", "pen", "min_size", "jump", "model"], dropna=False)


def breakpoint_probability(years: np.ndarray, detections: pd.DataFrame,
                           series: list[str], half: int = PROB_HALF) -> np.ndarray:
    sub = detections[detections.series.isin(series) & detections.breakpoint.notna()]
    runs: dict[tuple, list[float]] = {}
    for row in sub.itertuples():
        key = (row.series, row.pen, row.min_size, row.jump, row.model)
        runs.setdefault(key, []).append(row.breakpoint)
    rate = np.zeros(len(years))
    if not runs:
        return rate
    for bkps in runs.values():
        covered = np.zeros(len(years), dtype=bool)
        for b in bkps:
            covered |= np.abs(years - b) <= half
        rate += covered.astype(float)
    return rate / len(runs)


def write_hitrate(detections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    series_list = UNI_SERIES + ["joint"]
    for centre, tol in HIT_WINDOWS:
        for series in series_list:
            sub = detections[detections.series == series]
            n_spec = spec_keys(sub).ngroups
            n_hit = 0
            for _, g in spec_keys(sub):
                vals = g.breakpoint.dropna()
                if any(abs(v - centre) <= tol for v in vals):
                    n_hit += 1
            rows.append({
                "window": centre,
                "tol_yr": tol,
                "series": series,
                "n_spec": n_spec,
                "n_hit": n_hit,
                "hit_rate": (n_hit / n_spec) if n_spec else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "pelt_hitrate.csv", index=False)
    return out


def plot_figure(annual: pd.DataFrame, detections: pd.DataFrame) -> np.ndarray:
    years = annual.year.to_numpy()
    rate = breakpoint_probability(years, detections, POOL_SERIES, half=PROB_HALF)
    pd.DataFrame({"year": years, "breakpoint_probability": rate}).to_csv(
        DATA / "breakpoint_probability.csv", index=False
    )

    fig, axes = plt.subplots(3, 1, figsize=(120 / 25.4, 118 / 25.4), sharex=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.93, bottom=0.09, hspace=0.22)

    ax = axes[0]
    ax.plot(annual.year, annual.lat_mean, color="#1A5276", lw=1.05)
    ax.set_ylabel("Mean latitude (°N)")
    ax.text(-0.14, 1.08, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[1]
    ax.plot(annual.year, annual.n_south, color="#8B1E1E", lw=1.05)
    ax.set_ylabel("Seats south of 33°N")
    ax.text(-0.14, 1.08, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[2]
    ax.fill_between(years, 0, rate, color="#4D5560", alpha=0.22, lw=0)
    ax.plot(years, rate, color="#2C3E50", lw=1.15)
    ax.set_ylabel("Breakpoint probability")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Year (CE)")
    ax.text(-0.14, 1.08, "c", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.set_xlim(-221, 1911)

    for ax in axes:
        for t in MARK_YEARS:
            ax.axvline(t, color="#888888", lw=0.6, ls=":")

    fig.suptitle(
        "PELT robustness, location  (L1/L2, jump 5; penalty 0.75–12 × min. 30–200 yr)",
        fontsize=9, x=0.56,
    )
    FIG.mkdir(exist_ok=True)
    out = FIG / "pelt_location_robust"
    fig.savefig(out.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(out.with_suffix(".png"))
    return rate


def load_annual() -> pd.DataFrame:
    df = pd.read_csv(DATA / "annual_location_series.csv")
    if "south_share" not in df.columns:
        df["south_share"] = df.n_south / df.n_active
    return df


def run_grid(annual: pd.DataFrame) -> pd.DataFrame:
    years = annual.year.to_numpy()
    combos = list(itertools.product(PENS, MIN_SIZES, JUMPS, MODELS))
    records = []
    total = len(combos) * (len(UNI_SERIES) + 1)
    k = 0
    print(
        f"grid: {len(PENS)} penalties x {len(MIN_SIZES)} min_size x "
        f"{len(MODELS)} models x jump={JUMPS[0]}"
    )
    print(f"{len(combos)} specs/series x {len(UNI_SERIES) + 1} series = {total}")

    for col in UNI_SERIES:
        sig = annual[col].to_numpy()
        for pen, ms, jump, model in combos:
            k += 1
            if k % 80 == 0:
                print(f"  {k}/{total} {col} p{pen} m{ms} {model}", flush=True)
            bkps = pelt_years(years, sig, pen, ms, jump, model)
            spec = dict(series=col, pen=pen, min_size=ms, jump=jump, model=model)
            if not bkps:
                records.append({**spec, "breakpoint": np.nan})
            for b in bkps:
                records.append({**spec, "breakpoint": b})

    X = annual[["lat_mean", "n_south"]].to_numpy()
    for pen, ms, jump, model in combos:
        k += 1
        if k % 80 == 0:
            print(f"  {k}/{total} joint p{pen} m{ms} {model}", flush=True)
        bkps = pelt_joint(years, X, pen, ms, jump, model)
        spec = dict(series="joint", pen=pen, min_size=ms, jump=jump, model=model)
        if not bkps:
            records.append({**spec, "breakpoint": np.nan})
        for b in bkps:
            records.append({**spec, "breakpoint": b})

    out = pd.DataFrame(records)
    out["bin50"] = np.floor(out.breakpoint / 50.0) * 50
    check_detections(out)
    out.to_csv(DATA / "pelt_detections.csv", index=False)
    return out


def check_detections(det: pd.DataFrame, warn_frac: float = 0.25) -> None:
    """Report, per series and model, how many specifications found no break at all
    and how many breaks the median specification returned. A series/model block that is
    mostly empty (or mostly very busy) usually means the penalty is on the wrong scale."""
    keys = ["series", "model", "pen", "min_size", "jump"]
    per_spec = det.groupby(keys, dropna=False).breakpoint.apply(lambda s: int(s.notna().sum()))
    summ = per_spec.groupby(["series", "model"]).agg(
        n_spec="size",
        n_empty=lambda s: int((s == 0).sum()),
        median_breaks="median",
        max_breaks="max",
    )
    summ["frac_empty"] = summ.n_empty / summ.n_spec
    print("\ndetection check (breaks per specification)")
    print(summ.to_string())
    summ.to_csv(DATA / "pelt_detection_check.csv")
    bad = summ[summ.frac_empty > warn_frac]
    if len(bad):
        print(f"WARNING: more than {warn_frac:.0%} of specifications are empty in:\n{bad}")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    annual = load_annual()
    detections = run_grid(annual)
    hits = detections.dropna(subset=["breakpoint"])
    print(f"detections: {len(hits)}")
    hit = write_hitrate(detections)
    print("\nhit rate (break within +/-20 yr)")
    print(hit.to_string(index=False))
    rate = plot_figure(annual, detections)
    years = annual.year.to_numpy()
    for y in [t for t, _ in HIT_WINDOWS]:
        i = int(np.argmin(np.abs(years - y)))
        print(f"  P({y}) = {rate[i]:.3f}")


def plot_only() -> None:
    annual = load_annual()
    detections = pd.read_csv(DATA / "pelt_detections.csv")
    plot_figure(annual, detections)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot_only()
    else:
        main()
