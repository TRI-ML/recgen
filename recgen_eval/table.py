#!/usr/bin/env python3
"""
Print the paper-table metrics (CD_norm, ADD-SB, ADD-SB@0.1, ADD-SB@0.05,
DRE@0.05) for a finished evaluation run and, optionally, compare against the
published RecGen numbers (RecGen paper, main quantitative table).

Usage:
    python -m recgen_eval.table --run_dir <...>/dataset_HB_multiview1 [--expect hb1]
"""

import argparse
import os

import pandas as pd

# Published RecGen rows (mean over 538 HB / 500 ArtVIP instances):
# CD_norm ↓, ADD-SB ↓, ADD-SB@0.1 ↑ (%), ADD-SB@0.05 ↑ (%), DRE@0.05 ↑ (%)
PAPER_NUMBERS = {
    "hb1":      {"label": "HB, RecGen (1-view)",     "N": 538, "chamfer_normalized": 0.0321, "ADDSS": 0.0488, "ADDSS_10": 95.0, "ADDSS_05": 73.8, "DRE_005": 51.5},
    "hb2":      {"label": "HB, RecGen (2-view)",     "N": 538, "chamfer_normalized": 0.0289, "ADDSS": 0.0475, "ADDSS_10": 95.4, "ADDSS_05": 74.2, "DRE_005": 50.9},
    "artvip1":  {"label": "ArtVIP, RecGen (1-view)", "N": 500, "chamfer_normalized": 0.0261, "ADDSS": 0.0337, "ADDSS_10": 96.4, "ADDSS_05": 84.0, "DRE_005": 24.4},
    "artvip2":  {"label": "ArtVIP, RecGen (2-view)", "N": 500, "chamfer_normalized": 0.0239, "ADDSS": 0.0323, "ADDSS_10": 96.4, "ADDSS_05": 86.4, "DRE_005": 24.8},
}

METRIC_LABELS = [
    ("chamfer_normalized", "CD_norm (down)"),
    ("ADDSS", "ADD-SB (down)"),
    ("ADDSS_10", "ADD-SB@0.1 (up, %)"),
    ("ADDSS_05", "ADD-SB@0.05 (up, %)"),
    ("DRE_005", "DRE@0.05 (up, %)"),
]


def read_run_metrics(run_dir):
    """Read the paper-table metrics from a finished run directory.

    Sources:
      * predictions/final_results_from_json/0_all_frames_metrics_results_statistics.csv (MEAN row)
      * diameter_error.csv (written by ``python -m recgen_eval.diameter``)
    """
    stats_csv = os.path.join(run_dir, "predictions", "final_results_from_json",
                             "0_all_frames_metrics_results_statistics.csv")
    if not os.path.exists(stats_csv):
        raise FileNotFoundError(
            f"{stats_csv} not found — run the evaluation (or `python -m recgen_eval.aggregate`) first")
    df = pd.read_csv(stats_csv)
    mean_row = df[df["Statistic"] == "MEAN"].iloc[0]

    median_row = df[df["Statistic"] == "MEDIAN"].iloc[0]

    result = {
        "chamfer_normalized": float(mean_row["chamfer_normalized"]),
        "ADDSS": float(mean_row["ADDSS"]),
        # recalls stored as 0-1 fractions in the statistics CSV → percent
        "ADDSS_10": float(mean_row["ADDSS_10"]) * 100.0,
        "ADDSS_05": float(mean_row["ADDSS_05"]) * 100.0,
        # unnormalized chamfer distances (cm), mean and median — not part of the
        # published table (which is diameter-normalized) but useful in absolute terms
        "unnormalized_cm": {
            "Chamfer (ICP)": (float(mean_row["CHAMFER"]), float(median_row["CHAMFER"])),
            "Chamfer (no ICP)": (float(mean_row["CHAMFER_NO_ICP"]), float(median_row["CHAMFER_NO_ICP"])),
            "Chamfer (ICP+scale)": (float(mean_row["CHAMFER_ICP_WITH_SCALE"]), float(median_row["CHAMFER_ICP_WITH_SCALE"])),
        },
        "ADD-S@0.1d_pct": float(mean_row["ADD-S"]) * 100.0,
    }

    # Sample count + failures from the per-sample CSV (drop the trailing MEAN row)
    persample_csv = stats_csv.replace("_statistics.csv", ".csv")
    if os.path.exists(persample_csv):
        ps = pd.read_csv(persample_csv)
        ps = ps[ps["Frame_ID"] != "MEAN"]
        result["N"] = len(ps)
        result["N_failed"] = int(pd.to_numeric(ps["FAILED"], errors="coerce").fillna(0).sum())

    # DRE@0.05 from diameter_error.csv (optional — needs recgen_eval.diameter first)
    diam_csv = os.path.join(run_dir, "diameter_error.csv")
    if os.path.exists(diam_csv):
        diam = pd.read_csv(diam_csv)
        if not diam.empty and "rel_error" in diam.columns:
            result["DRE_005"] = float((diam["rel_error"] < 0.05).mean()) * 100.0
    else:
        result["DRE_005"] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize an eval run against the paper table")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory (e.g. .../dataset_HB_multiview1)")
    parser.add_argument("--expect", type=str, default=None, choices=sorted(PAPER_NUMBERS.keys()),
                        help="Compare against a published RecGen row (hb1, hb2, artvip1, artvip2)")
    args = parser.parse_args()

    got = read_run_metrics(args.run_dir)
    expect = PAPER_NUMBERS.get(args.expect) if args.expect else None

    print(f"\nRun: {args.run_dir}")
    n_failed = got.get("N_failed", 0)
    if "N" in got:
        print(f"Samples: {got['N']} (failed: {n_failed})"
              + (f"   [paper: {expect['N']}]" if expect else ""))
    if n_failed > 0:
        print("\n" + "!" * 72)
        print(f"WARNING: {n_failed} samples FAILED (penalty-scored). Failed samples carry")
        print("no CD_norm/ADD-SB values, so the means below cover only the survivors and")
        print("are NOT comparable to the paper numbers (the paper runs had 0 failures).")
        print("Fix the failures (usually GPU OOM — reduce --workers_per_gpu) and resume")
        print("with --overwrite 0 after deleting the failed samples' metrics.json.")
        print("!" * 72)
    if got.get("DRE_005") is None:
        print("NOTE: diameter_error.csv missing — run `python -m recgen_eval.diameter` for DRE@0.05")

    header = f"{'metric':<22s} {'this run':>10s}"
    if expect:
        header += f" {'paper':>10s} {'delta':>10s}"
    print("\n" + header)
    print("-" * len(header))
    for key, label in METRIC_LABELS:
        val = got.get(key)
        line = f"{label:<22s} " + (f"{val:>10.4f}" if val is not None else f"{'—':>10s}")
        if expect:
            ref = expect[key]
            line += f" {ref:>10.4f}"
            if val is not None:
                line += f" {val - ref:>+10.4f}"
        print(line)
    print()

    if "unnormalized_cm" in got:
        print("unnormalized (cm)         mean     median")
        for label, (mean_v, med_v) in got["unnormalized_cm"].items():
            print(f"{label:<22s} {mean_v:>9.4f} {med_v:>10.4f}")
        if "ADD-S@0.1d_pct" in got:
            print(f"{'ADD-S@0.1d (%)':<22s} {got['ADD-S@0.1d_pct']:>9.1f}")
        print()

    if expect and n_failed > 0:
        print("RESULT: INVALID comparison (failures present) — see warning above.")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
