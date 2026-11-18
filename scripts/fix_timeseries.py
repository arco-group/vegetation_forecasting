"""Recompute the vegetation-index columns of the per-minicube time-series CSVs from the raw GreenEarthNet cubes,
with three corrections applied.

For every <split>/<season or tile>/<cube>.nc in --raw-root, the CSV with the same relative path in --csv-root is
read and its 100 vegetation-index columns ({avg,std,Q1,Q2,Q3}_{RVI,NDVI,GNDVI,EVI,SAVI}_{all,no_thick,no_clouds,
clear_sky}) are recomputed from the cube; all other columns (weather, cloud fractions, s2_avail) are copied
unchanged. The result is written to the same relative path in --out-root.

Index definitions: per-pixel index over cropland pixels (ESA WorldCover class 40), with masks all / no_thick
(dlmask != 1) / no_clouds (dlmask not in {1, 2}) / clear_sky (dlmask == 0); statistics over finite values: mean,
std (ddof 0), 25/50/75th percentiles. Reflectance = int16 x scale_factor + add_offset (per file).

Corrections:
  1. Sentinel-2 processing baseline 04.00 (acquisitions from 2022-01-25): the +0.1 radiometric offset is not
     removed in the GreenEarthNet cubes. It is subtracted from B02/B03/B04/B8A and reflectance is floored at
     1e-4. Each acquisition is also checked against the data (1st percentile of clear-sky cropland blue reflectance
     above 0.08 <=> offset present); disagreements are logged.
  2. EVI uses the blue band: 2.5 (NIR - R) / (NIR + 6 R - 7.5 B + 1), clipped per pixel to [-1, 1].
  3. Pixels with all four bands equal to 0 (Sentinel-2 NO_DATA) are treated as missing instead of producing VI = 0.

Requires numpy and h5py. Resumable (existing outputs are skipped); one JSON line of range statistics per cube is
appended to <out-root>/fix_log/<split>.jsonl.

    python scripts/fix_timeseries.py --raw-root <greenearthnet> --csv-root <timeSeries/global> \
        --out-root <timeSeries_fixed/global> --workers 8
"""
import argparse
import csv
import glob
import json
import os
import warnings
from datetime import date, timedelta
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np

SPLITS = ["train", "val_chopped", "ood-t_chopped", "ood-s_chopped", "ood-st_chopped"]
OFFSET_START = date(2022, 1, 25)
OFFSET = 0.1
MIN_REFLECTANCE = 1e-4
FLOOR_THRESHOLD = 0.08
FLOOR_MIN_PIXELS = 500
CROPLAND = 40
FILL_VALUE = -32767


def evi_raw(n, r, g, b):
    return 2.5 * (n - r) / (n + 6 * r - 7.5 * b + 1)


INDICES = {
    "RVI": lambda n, r, g, b: n / r,
    "NDVI": lambda n, r, g, b: (n - r) / (n + r),
    "GNDVI": lambda n, r, g, b: (n - g) / (n + g),
    "EVI": lambda n, r, g, b: np.clip(evi_raw(n, r, g, b), -1, 1),
    "SAVI": lambda n, r, g, b: 1.5 * (n - r) / (n + r + 0.5),
}
MASKS = {
    "all": lambda dl: np.ones_like(dl, dtype=bool),
    "no_thick": lambda dl: dl != 1,
    "no_clouds": lambda dl: (dl != 1) & (dl != 2),
    "clear_sky": lambda dl: dl == 0,
}
STATS = {
    "avg": np.mean,
    "std": np.std,
    "Q1": lambda x: np.percentile(x, 25),
    "Q2": lambda x: np.percentile(x, 50),
    "Q3": lambda x: np.percentile(x, 75),
}


def fmt(v):
    return "" if not np.isfinite(v) else repr(float(v))


def read_cube(path):
    with h5py.File(path, "r") as f:
        units = f["time"].attrs["units"]
        units = units.decode() if isinstance(units, bytes) else str(units)
        if not units.startswith("days since "):
            raise ValueError(f"{path}: unexpected time units {units!r}")
        t0 = date.fromisoformat(units[len("days since "):][:10])
        days = [t0 + timedelta(days=int(t)) for t in f["time"][:]]
        dlmask = f["s2_dlmask"][:]
        crop = f["esawc_lc"][:] == CROPLAND
        bands = {}
        for b in ("B02", "B03", "B04", "B8A"):
            d = f["s2_" + b]
            x = d[:].astype(float)
            x[x == FILL_VALUE] = np.nan
            bands[b] = x * float(d.attrs["scale_factor"][0]) + float(d.attrs["add_offset"][0])
    nodata = np.logical_and.reduce([np.abs(v) < 1e-6 for v in bands.values()])  # correction 3
    for v in bands.values():
        v[nodata] = np.nan
    return days, dlmask, crop, bands, int(nodata.sum())


def fix_cube(job, csv_root, out_root):
    nc, split, rel = job
    src = os.path.join(csv_root, split, rel)
    dst = os.path.join(out_root, split, rel)
    if os.path.exists(dst):
        return None
    if not os.path.exists(src):
        return {"split": split, "cube": rel, "error": "csv not found"}
    warnings.filterwarnings("ignore")
    with open(src, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    col = {h: i for i, h in enumerate(header)}
    row_of_day = {r[0][:10]: r for r in body}

    days, dlmask, crop, bands, n_nodata = read_cube(nc)
    stats = {"split": split, "cube": rel, "n_acq": 0, "n_offset_acq": 0, "n_acq_without_row": 0,
             "n_floor_checked": 0, "floor_mismatch": [], "pix_nodata": n_nodata}
    for i, day in enumerate(days):
        n, r, g, b = (bands[k][i] for k in ("B8A", "B04", "B03", "B02"))
        has_data = bool(np.isfinite(n).any())
        row = row_of_day.get(day.isoformat())
        if row is None:
            stats["n_acq_without_row"] += int(has_data)
            continue
        if has_data:
            stats["n_acq"] += 1
            offset = day >= OFFSET_START
            clear = crop & (dlmask[i] == 0) & np.isfinite(b)
            if clear.sum() >= FLOOR_MIN_PIXELS:
                stats["n_floor_checked"] += 1
                floor = float(np.percentile(b[clear], 1))
                if (floor > FLOOR_THRESHOLD) != offset:
                    stats["floor_mismatch"].append([day.isoformat(), round(floor, 4)])
            if offset:  # correction 1
                n, r, g, b = (np.maximum(x - OFFSET, MIN_REFLECTANCE) for x in (n, r, g, b))
                stats["n_offset_acq"] += 1
        values = {v: fn(n, r, g, b) for v, fn in INDICES.items()}  # correction 2 in INDICES["EVI"]
        for mname, mfn in MASKS.items():
            m = crop & mfn(dlmask[i])
            for v, arr in values.items():
                x = arr[m]
                x = x[np.isfinite(x)]
                for sname, sfn in STATS.items():
                    key = f"{sname}_{v}_{mname}"
                    if key in col:
                        row[col[key]] = fmt(sfn(x)) if len(x) else ""

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    with open(tmp, "w", newline="") as f:
        csv.writer(f).writerows([header] + body)
    os.replace(tmp, dst)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-root", required=True, help="GreenEarthNet root with <split>/<folder>/<cube>.nc")
    ap.add_argument("--csv-root", required=True, help="original CSVs with <split>/<folder>/<cube>.csv")
    ap.add_argument("--out-root", required=True, help="output root for the corrected CSVs")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cubes per split (quick test)")
    args = ap.parse_args(argv)

    log_dir = os.path.join(args.out_root, "fix_log")
    os.makedirs(log_dir, exist_ok=True)
    worker = partial(fix_cube, csv_root=args.csv_root, out_root=args.out_root)
    for split in args.splits:
        cubes = sorted(glob.glob(os.path.join(args.raw_root, split, "*", "*.nc")))[: args.limit]
        jobs = [(nc, split, os.path.join(os.path.basename(os.path.dirname(nc)),
                                         os.path.basename(nc)[:-3] + ".csv")) for nc in cubes]
        done = 0
        with Pool(args.workers) as pool, open(os.path.join(log_dir, f"{split}.jsonl"), "a") as log:
            for s in pool.imap_unordered(worker, jobs, chunksize=4):
                done += 1
                if s is not None:
                    log.write(json.dumps(s) + "\n")
                if done % 500 == 0:
                    log.flush()
                    print(f"{split}: {done}/{len(jobs)}", flush=True)
        print(f"{split}: {len(jobs)} cubes done", flush=True)


if __name__ == "__main__":
    main()
