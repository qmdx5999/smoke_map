import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _format_number(x):
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        return f"{float(x):.6g}"
    return str(x)


def describe_array(name, arr):
    arr = np.asarray(arr)
    print(f"{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  size: {arr.size}")
    if arr.size == 0:
        return

    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.number) else None
    if finite is not None:
        finite_count = int(np.count_nonzero(finite))
        nonzero_count = int(np.count_nonzero(arr))
        print(f"  finite: {finite_count}/{arr.size}")
        print(f"  nonzero: {nonzero_count}/{arr.size}")
        print(f"  min: {_format_number(np.nanmin(arr))}")
        print(f"  max: {_format_number(np.nanmax(arr))}")
        print(f"  mean: {_format_number(np.nanmean(arr))}")
        print(f"  sum: {_format_number(np.nansum(arr))}")
    else:
        sample = arr.reshape(-1)[:5]
        print(f"  sample: {sample}")


def print_npz_info(npz_path, npz):
    print("=" * 72)
    print(f"file: {npz_path}")
    print(f"keys: {', '.join(npz.files)}")
    print("=" * 72)

    for key in npz.files:
        describe_array(key, npz[key])

    if {"coords", "data", "shape"}.issubset(npz.files):
        coords = np.asarray(npz["coords"])
        values = np.asarray(npz["data"])
        shape = tuple(int(v) for v in np.asarray(npz["shape"]).reshape(-1))
        dense_size = int(np.prod(shape)) if len(shape) > 0 else 0
        nnz = int(values.size)
        sparsity = 1.0 - (nnz / dense_size) if dense_size > 0 else float("nan")
        print("sparse histogram:")
        print(f"  dense shape: {shape}")
        print(f"  stored entries: {nnz}")
        print(f"  dense entries: {dense_size}")
        print(f"  sparsity: {sparsity:.6%}")
        if coords.ndim == 2:
            for axis in range(coords.shape[0]):
                axis_values = coords[axis]
                print(
                    f"  coords axis {axis}: "
                    f"min={int(axis_values.min())}, max={int(axis_values.max())}"
                )


def safe_index(idx, upper):
    if upper <= 0:
        return 0
    return max(0, min(int(idx), upper - 1))


def summarize_histogram(hist):
    hist = np.asarray(hist)
    if hist.ndim != 3:
        return None
    total = hist.sum(axis=-1)
    peak = hist.max(axis=-1)
    argmax = hist.argmax(axis=-1)
    return total, peak, argmax


def build_lidar_pointcloud(z_sim, intensity_sim, v_fov_up, v_fov_down):
    ranges = np.asarray(z_sim, dtype=np.float64)
    intensity = np.asarray(intensity_sim, dtype=np.float64)
    if ranges.shape != intensity.shape:
        raise ValueError(
            f"z_sim shape {ranges.shape} does not match intensity_sim shape {intensity.shape}"
        )

    rows, cols = ranges.shape
    valid = np.isfinite(ranges) & (ranges > 0.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)

    row_idx, col_idx = np.indices((rows, cols))
    elevation = np.deg2rad(np.linspace(v_fov_up, v_fov_down, rows))[:, None]
    azimuth = np.linspace(-np.pi, np.pi, cols, endpoint=False)[None, :]

    cos_el = np.cos(elevation)
    x = ranges * cos_el * np.cos(azimuth)
    y = ranges * cos_el * np.sin(azimuth)
    z = ranges * np.sin(elevation)

    points = np.stack([x[valid], y[valid], z[valid]], axis=1)
    colors = intensity[valid]
    return points, colors


def sample_points(points, colors, max_points, seed=0):
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], colors[idx]


def set_axes_equal(ax, points):
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_pointcloud(points, colors, save_path, show):
    if points.shape[0] == 0:
        print("pointcloud: no valid points to plot")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=1.0,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_title("NPZ LiDAR pointcloud from z_sim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=22, azim=-55)
    set_axes_equal(ax, points)
    fig.colorbar(sc, ax=ax, shrink=0.65, label="intensity_sim")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"saved pointcloud image: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_array_image(arr, title, save_path, cmap="viridis", colorbar_label=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        im = ax.imshow(arr, cmap=cmap)
    except ValueError:
        im = ax.imshow(arr, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if colorbar_label is not None:
        fig.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)
    else:
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"saved image: {save_path}")


def plot_histogram_1d(hist_1d, save_path, title, show=False):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(hist_1d.shape[0]), hist_1d, lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("bin")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"saved histogram plot: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a SPAD .npz file and render useful previews."
    )
    parser.add_argument("--npz", required=True, help="Path to .npz file")
    parser.add_argument("--save", default="npz_overview.png", help="Output image path")
    parser.add_argument("--show", action="store_true", help="Show matplotlib window")
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum plotted points; <=0 means no limit",
    )
    parser.add_argument("--v-fov-up", type=float, default=10.67)
    parser.add_argument("--v-fov-down", type=float, default=-30.67)
    parser.add_argument("--row", type=int, default=-1, help="Row index for 1D histogram preview")
    parser.add_argument("--col", type=int, default=-1, help="Column index for 1D histogram preview")
    return parser.parse_args()


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=False) as npz:
        print_npz_info(npz_path, npz)
        files = set(npz.files)

        if {"spad_hist", "gt_depth"}.issubset(files):
            hist = np.asarray(npz["spad_hist"])
            depth = np.asarray(npz["gt_depth"])
            phi_bar = np.asarray(npz["phi_bar"]) if "phi_bar" in files else None

            print("spad_hist previews:")
            print(f"  total counts image shape: {hist.sum(axis=-1).shape}")
            print(f"  peak counts image shape: {hist.max(axis=-1).shape}")
            print(f"  argmax image shape: {hist.argmax(axis=-1).shape}")

            total_img, peak_img, argmax_img = summarize_histogram(hist)
            prefix = Path(args.save).with_suffix("")
            plot_array_image(total_img, "spad_hist sum over bins", prefix.as_posix() + "_sum.png", cmap="magma", colorbar_label="sum")
            plot_array_image(peak_img, "spad_hist max over bins", prefix.as_posix() + "_peak.png", cmap="viridis", colorbar_label="max")
            plot_array_image(argmax_img, "spad_hist argmax bin", prefix.as_posix() + "_argmax.png", cmap="viridis", colorbar_label="bin")
            plot_array_image(depth, "gt_depth", prefix.as_posix() + "_depth.png", cmap="viridis", colorbar_label="m")

            center_row = hist.shape[0] // 2
            center_col = hist.shape[1] // 2
            row = center_row if args.row < 0 else safe_index(args.row, hist.shape[0])
            col = center_col if args.col < 0 else safe_index(args.col, hist.shape[1])
            row_hist = hist[row, center_col]
            col_hist = hist[center_row, col]
            center_hist = hist[center_row, center_col]
            plot_histogram_1d(
                center_hist,
                prefix.as_posix() + f"_hist_center_r{center_row}_c{center_col}.png",
                f"spad_hist at center ({center_row}, {center_col})",
                show=bool(args.show),
            )
            plot_histogram_1d(
                row_hist,
                prefix.as_posix() + f"_hist_row{row}_centercol.png",
                f"spad_hist row {row}, center col {center_col}",
                show=False,
            )
            plot_histogram_1d(
                col_hist,
                prefix.as_posix() + f"_hist_centerrow_col{col}.png",
                f"spad_hist center row {center_row}, col {col}",
                show=False,
            )

            if phi_bar is not None:
                phi_total = phi_bar.sum(axis=-1)
                plot_array_image(phi_total, "phi_bar sum over bins", prefix.as_posix() + "_phi_sum.png", cmap="magma", colorbar_label="sum")

            overview_fig = plt.figure(figsize=(14, 10))
            gs = GridSpec(2, 2, figure=overview_fig)
            ax0 = overview_fig.add_subplot(gs[0, 0])
            im0 = ax0.imshow(total_img, cmap="magma")
            ax0.set_title("spad_hist sum")
            fig = overview_fig
            fig.colorbar(im0, ax=ax0, shrink=0.8)

            ax1 = overview_fig.add_subplot(gs[0, 1])
            im1 = ax1.imshow(peak_img, cmap="viridis")
            ax1.set_title("spad_hist max")
            fig.colorbar(im1, ax=ax1, shrink=0.8)

            ax2 = overview_fig.add_subplot(gs[1, 0])
            im2 = ax2.imshow(depth, cmap="viridis")
            ax2.set_title("gt_depth")
            fig.colorbar(im2, ax=ax2, shrink=0.8)

            ax3 = overview_fig.add_subplot(gs[1, 1])
            ax3.plot(center_hist, lw=1.0)
            ax3.set_title(f"center histogram ({center_row}, {center_col})")
            ax3.set_xlabel("bin")
            ax3.set_ylabel("count")

            overview_fig.tight_layout()
            overview_fig.savefig(Path(args.save), dpi=200)
            print(f"saved overview image: {args.save}")
            if args.show:
                plt.show()
            plt.close(overview_fig)
            return

        if {"z_sim", "intensity_sim"}.issubset(files):
            points, colors = build_lidar_pointcloud(
                npz["z_sim"],
                npz["intensity_sim"],
                args.v_fov_up,
                args.v_fov_down,
            )

            print("pointcloud:")
            print(f"  valid points: {points.shape[0]}")
            points_plot, colors_plot = sample_points(points, colors, int(args.max_points))
            print(f"  plotted points: {points_plot.shape[0]}")
            if points.shape[0] > 0:
                print(f"  xyz min: {np.array2string(points.min(axis=0), precision=4)}")
                print(f"  xyz max: {np.array2string(points.max(axis=0), precision=4)}")

            plot_pointcloud(points_plot, colors_plot, Path(args.save), bool(args.show))
            return

        raise KeyError(
            "Unsupported npz format. Expected either {'spad_hist', 'gt_depth'} or {'z_sim', 'intensity_sim'}."
        )


if __name__ == "__main__":
    main()
