"""
Convert SPAD histogram data in an .npz file to an ASCII PLY point cloud.

The depth is estimated by taking the maximum-count time bin in spad_hist for
each pixel, then back-projecting the resulting depth map with camera intrinsics.
"""

import argparse
import os
from typing import Tuple

import numpy as np


C_M_PER_S = 299_792_458.0

DEFAULT_NPZ = "./scene_0000.npz"
DEFAULT_OUT = "./scene_0000_from_spad.ply"

NYU_WIDTH = 640.0
NYU_HEIGHT = 480.0
NYU_FX = 518.8579
NYU_FY = 519.4696
NYU_CX = 325.5824
NYU_CY = 253.7362


def scaled_nyu_intrinsics(width: int, height: int) -> Tuple[float, float, float, float]:
    """Scale NYUv2/Kinect intrinsics from 640x480 to the current image size."""
    sx = width / NYU_WIDTH
    sy = height / NYU_HEIGHT
    return NYU_FX * sx, NYU_FY * sy, NYU_CX * sx, NYU_CY * sy


def estimate_depth_from_spad(
    spad_hist: np.ndarray, tmax_ns: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return depth, peak bin, peak count, and maximum measurable distance."""
    if spad_hist.ndim != 3:
        raise ValueError(f"spad_hist must have shape (H, W, T), got {spad_hist.shape}")

    n_tbins = spad_hist.shape[2]
    dmax_m = C_M_PER_S * (tmax_ns * 1e-9) / 2.0
    peak_bins = np.argmax(spad_hist, axis=2).astype(np.float32)
    peak_counts = np.max(spad_hist, axis=2)
    depth_m = peak_bins * dmax_m / float(n_tbins)
    return depth_m.astype(np.float32), peak_bins, peak_counts, dmax_m


def depth_to_points(
    depth_m: np.ndarray,
    peak_counts: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dmax_m: float,
    min_peak_count: float,
    flip_y: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map to XYZ points and return points plus valid mask."""
    height, width = depth_m.shape
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    valid = (
        np.isfinite(depth_m)
        & (depth_m > 0)
        & (depth_m <= dmax_m)
        & (peak_counts >= min_peak_count)
    )

    z = depth_m[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    if flip_y:
        y = -y

    points = np.column_stack((x, y, z)).astype(np.float32)
    return points, valid


def write_ascii_ply(path: str, points: np.ndarray) -> None:
    """Write XYZ points as an ASCII PLY file."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a SPAD histogram .npz file to an ASCII PLY point cloud."
    )
    parser.add_argument("--npz", default=DEFAULT_NPZ, help="Input .npz path.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output .ply path.")
    parser.add_argument("--tmax-ns", type=float, default=100.0, help="Laser period in nanoseconds.")
    parser.add_argument(
        "--min-peak-count",
        type=float,
        default=1.0,
        help="Discard pixels whose maximum histogram count is below this value.",
    )
    parser.add_argument("--fx", type=float, default=None, help="Camera focal length fx in pixels.")
    parser.add_argument("--fy", type=float, default=None, help="Camera focal length fy in pixels.")
    parser.add_argument("--cx", type=float, default=None, help="Camera principal point cx in pixels.")
    parser.add_argument("--cy", type=float, default=None, help="Camera principal point cy in pixels.")
    parser.add_argument("--flip-y", action="store_true", help="Flip the output Y axis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = np.load(args.npz)
    if "spad_hist" not in data:
        raise KeyError(f"{args.npz} does not contain a 'spad_hist' array")

    spad_hist = data["spad_hist"]
    if spad_hist.ndim != 3:
        raise ValueError(f"spad_hist must have shape (H, W, T), got {spad_hist.shape}")

    height, width, n_tbins = spad_hist.shape
    default_fx, default_fy, default_cx, default_cy = scaled_nyu_intrinsics(width, height)
    fx = default_fx if args.fx is None else args.fx
    fy = default_fy if args.fy is None else args.fy
    cx = default_cx if args.cx is None else args.cx
    cy = default_cy if args.cy is None else args.cy

    depth_m, peak_bins, peak_counts, dmax_m = estimate_depth_from_spad(spad_hist, args.tmax_ns)
    points, valid = depth_to_points(
        depth_m=depth_m,
        peak_counts=peak_counts,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        dmax_m=dmax_m,
        min_peak_count=args.min_peak_count,
        flip_y=args.flip_y,
    )
    write_ascii_ply(args.out, points)

    valid_depth = depth_m[valid]
    print(f"Input       : {args.npz}")
    print(f"spad_hist   : shape={spad_hist.shape}, dtype={spad_hist.dtype}")
    print(f"Time bins   : {n_tbins}, tmax={args.tmax_ns:g} ns, dmax={dmax_m:.4f} m")
    print(f"Intrinsics  : fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    print(f"Peak filter : min_peak_count={args.min_peak_count:g}")
    print(f"Valid points: {points.shape[0]} / {height * width}")
    if valid_depth.size:
        print(f"Depth range : {valid_depth.min():.4f} ~ {valid_depth.max():.4f} m")
        print(f"Peak bins   : {peak_bins[valid].min():.0f} ~ {peak_bins[valid].max():.0f}")
    print(f"Output      : {args.out}")


if __name__ == "__main__":
    main()
