"""Export thresholded occupancy PLY files from a saved occupancy grid .npz.

This is a post-processing helper for spad_npz_occupancy_mapping.py --grid-out.
It lets you adjust occupancy thresholds without rerunning the expensive mapping
stage.
"""

import argparse
import math
import os

import numpy as np


def logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help=".npz file produced by --grid-out")
    ap.add_argument("--ply-out", required=True, help="Output thresholded PLY path")
    ap.add_argument("--min-prob", type=float, default=0.5, help="Minimum occupancy probability to export")
    ap.add_argument("--max-prob", type=float, default=1.0, help="Maximum occupancy probability to export")
    ap.add_argument(
        "--active-eps",
        type=float,
        default=1e-6,
        help="Ignore near-prior voxels with abs(log-odds) <= active-eps",
    )
    ap.add_argument(
        "--mode",
        choices=["occupied", "active"],
        default="occupied",
        help="occupied exports min_prob..max_prob; active exports all updated voxels with occupancy scalar",
    )
    return ap.parse_args()


def write_ply(path: str, xyz: np.ndarray, prob: np.ndarray) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float occupancy\n")
        f.write("end_header\n")
        for i in range(xyz.shape[0]):
            p = float(prob[i])
            g = int(max(0, min(255, round(p * 255.0))))
            f.write(f"{xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} {g} {g} {g} {p:.8f}\n")


def main() -> None:
    args = parse_args()
    with np.load(args.grid) as data:
        Lgrid = np.asarray(data["Lgrid"], dtype=np.float32)
        mins = np.asarray(data["mins"], dtype=np.float64)
        voxel = float(np.asarray(data["voxel"]).reshape(-1)[0])

    active = np.abs(Lgrid) > float(args.active_eps)
    prob_grid = 1.0 / (1.0 + np.exp(-Lgrid.astype(np.float64)))

    if args.mode == "active":
        mask = active
    else:
        mask = active & (prob_grid >= float(args.min_prob)) & (prob_grid <= float(args.max_prob))

    xs, ys, zs = np.where(mask)
    xyz = np.empty((xs.size, 3), dtype=np.float32)
    xyz[:, 0] = mins[0] + voxel * (xs.astype(np.float64) + 0.5)
    xyz[:, 1] = mins[1] + voxel * (ys.astype(np.float64) + 0.5)
    xyz[:, 2] = mins[2] + voxel * (zs.astype(np.float64) + 0.5)
    prob = prob_grid[xs, ys, zs].astype(np.float32)

    write_ply(args.ply_out, xyz, prob)
    print("saved", args.ply_out, "N=", int(xs.size), "mode=", args.mode)
    print("active voxels:", int(np.count_nonzero(active)))
    if xs.size:
        print("prob range:", float(prob.min()), float(prob.max()))
        print("equivalent log-odds threshold:", logit(args.min_prob))


if __name__ == "__main__":
    main()
