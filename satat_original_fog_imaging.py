"""
Reproduce the pixel-wise fog separation pipeline from Satat et al. (2018).

Pipeline:
    photon histogram
    -> Gaussian KDE
    -> Gamma fog distribution fit
    -> non-negative fog subtraction
    -> Gaussian surface distribution fit
    -> range and reflectance images

Only ``spad_hist`` is used for inference. ``gt_range`` is optional and is
loaded only after reconstruction to compute evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import digamma, gammaln, polygamma


C_M_PER_S = 299_792_458.0
DEFAULT_NPZ = Path("out/scene_00_0000.npz")
DEFAULT_OUT_DIR = Path("satat_out")
GAUSSIAN_FIT_HALF_WIDTH_BINS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the original Satat et al. 2018 fog-imaging pipeline."
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="Input SPAD .npz file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root. Results are written to OUT_DIR/NPZ_STEM/.",
    )
    parser.add_argument(
        "--tmax-ns",
        type=float,
        default=100.0,
        help="Full histogram time span in nanoseconds.",
    )
    parser.add_argument(
        "--kde-bandwidth-ns",
        type=float,
        default=0.08,
        help="Gaussian KDE bandwidth in nanoseconds (paper default: 0.08 ns).",
    )
    parser.add_argument(
        "--range-max",
        type=float,
        default=7.0,
        help="Maximum accepted one-way range in meters.",
    )
    parser.add_argument(
        "--confidence-relative",
        type=float,
        default=0.1,
        help="Keep pixels whose reflectance*range reaches this fraction of the frame maximum.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=16,
        help="Number of image rows processed per memory-bounded chunk.",
    )
    parser.add_argument(
        "--min-photons",
        type=float,
        default=1.0,
        help="Minimum total photon count required for fitting a pixel.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.tmax_ns <= 0:
        raise ValueError("--tmax-ns must be positive")
    if args.kde_bandwidth_ns <= 0:
        raise ValueError("--kde-bandwidth-ns must be positive")
    if args.range_max <= 0:
        raise ValueError("--range-max must be positive")
    if not 0.0 <= args.confidence_relative <= 1.0:
        raise ValueError("--confidence-relative must be in [0, 1]")
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    if args.min_photons < 0:
        raise ValueError("--min-photons must be non-negative")


def load_spad_hist(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "spad_hist" not in data:
            raise KeyError(f"{path} does not contain 'spad_hist'")
        hist = np.asarray(data["spad_hist"], dtype=np.float64)
    if hist.ndim != 3:
        raise ValueError(f"spad_hist must have shape (H, W, T), got {hist.shape}")
    if not np.all(np.isfinite(hist)):
        raise ValueError("spad_hist contains non-finite values")
    if np.any(hist < 0):
        raise ValueError("spad_hist contains negative photon counts")
    return hist


def load_gt_range(path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "gt_range" not in data:
            raise KeyError(f"{path} does not contain 'gt_range'")
        gt_range = np.asarray(data["gt_range"], dtype=np.float64)
    if gt_range.shape != expected_shape:
        raise ValueError(
            f"gt_range shape {gt_range.shape} does not match image shape {expected_shape}"
        )
    if not np.all(np.isfinite(gt_range)):
        raise ValueError("gt_range contains non-finite values")
    return gt_range


def gamma_shape_mle(
    weighted_mean: np.ndarray,
    weighted_log_mean: np.ndarray,
    valid: np.ndarray,
    iterations: int = 12,
) -> np.ndarray:
    """Solve log(k)-digamma(k)=log(E[x])-E[log(x)] for Gamma shape k."""
    statistic = np.maximum(
        np.log(np.maximum(weighted_mean, 1e-12)) - weighted_log_mean,
        1e-10,
    )
    shape = np.ones_like(weighted_mean, dtype=np.float64)
    initial = (
        3.0
        - statistic
        + np.sqrt((statistic - 3.0) ** 2 + 24.0 * statistic)
    ) / (12.0 * statistic)
    shape[valid] = np.clip(initial[valid], 0.05, 1e5)

    for _ in range(iterations):
        current = shape[valid]
        numerator = np.log(current) - digamma(current) - statistic[valid]
        denominator = (1.0 / current) - polygamma(1, current)
        update = current - numerator / np.minimum(denominator, -1e-12)
        shape[valid] = np.clip(update, 0.05, 1e5)
    return shape


def discrete_gamma_mass(
    time_ns: np.ndarray,
    dt_ns: float,
    shape: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    x = time_ns[None, None, :]
    k = shape[:, :, None]
    theta = scale[:, :, None]
    log_pdf = (k - 1.0) * np.log(x) - (x / theta) - gammaln(k) - k * np.log(theta)
    mass = np.exp(np.clip(log_pdf, -745.0, 80.0)) * dt_ns
    mass_sum = mass.sum(axis=2, keepdims=True)
    return np.divide(mass, mass_sum, out=np.zeros_like(mass), where=mass_sum > 0)


def reconstruct_satat(
    hist: np.ndarray,
    tmax_ns: float,
    kde_bandwidth_ns: float,
    range_max_m: float,
    confidence_relative: float,
    chunk_rows: int,
    min_photons: float,
) -> Dict[str, np.ndarray]:
    height, width, n_bins = hist.shape
    dt_ns = tmax_ns / float(n_bins)
    sigma_bins = kde_bandwidth_ns / dt_ns
    time_ns = (np.arange(n_bins, dtype=np.float64) + 0.5) * dt_ns
    log_time_ns = np.log(time_ns)
    range_per_bin_m = C_M_PER_S * (dt_ns * 1e-9) / 2.0
    max_surface_bin = min(
        n_bins,
        max(1, int(math.floor(range_max_m / range_per_bin_m)) + 1),
    )

    raw_range = np.full((height, width), np.nan, dtype=np.float64)
    reflectance = np.zeros((height, width), dtype=np.float64)
    gamma_shape = np.full((height, width), np.nan, dtype=np.float64)
    gamma_scale_ns = np.full((height, width), np.nan, dtype=np.float64)
    gaussian_sigma_ns = np.full((height, width), np.nan, dtype=np.float64)
    fit_valid = np.zeros((height, width), dtype=bool)

    for row0 in range(0, height, chunk_rows):
        row1 = min(row0 + chunk_rows, height)
        counts = hist[row0:row1]
        totals = counts.sum(axis=2)
        valid = totals >= min_photons

        mean_time = np.divide(
            np.sum(counts * time_ns[None, None, :], axis=2),
            totals,
            out=np.zeros_like(totals),
            where=totals > 0,
        )
        mean_log_time = np.divide(
            np.sum(counts * log_time_ns[None, None, :], axis=2),
            totals,
            out=np.zeros_like(totals),
            where=totals > 0,
        )
        shape = gamma_shape_mle(mean_time, mean_log_time, valid)
        scale = np.divide(
            mean_time,
            shape,
            out=np.ones_like(mean_time),
            where=shape > 0,
        )
        scale = np.clip(scale, 1e-6, tmax_ns * 100.0)

        smoothed = gaussian_filter1d(
            counts,
            sigma=sigma_bins,
            axis=2,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )
        smoothed_sum = smoothed.sum(axis=2, keepdims=True)
        measured_mass = np.divide(
            smoothed,
            smoothed_sum,
            out=np.zeros_like(smoothed),
            where=smoothed_sum > 0,
        )
        fog_mass = discrete_gamma_mass(time_ns, dt_ns, shape, scale)
        residual = np.maximum(
            measured_mass[:, :, :max_surface_bin] - fog_mass[:, :, :max_surface_bin],
            0.0,
        )

        surface_time = time_ns[:max_surface_bin]
        peak_bin = np.argmax(residual, axis=2)
        surface_bin = np.arange(max_surface_bin, dtype=np.int64)
        local_support = (
            np.abs(surface_bin[None, None, :] - peak_bin[:, :, None])
            <= GAUSSIAN_FIT_HALF_WIDTH_BINS
        )
        local_residual = np.where(local_support, residual, 0.0)
        surface_mass = local_residual.sum(axis=2)
        mu_ns = np.divide(
            np.sum(local_residual * surface_time[None, None, :], axis=2),
            surface_mass,
            out=np.zeros_like(surface_mass),
            where=surface_mass > 0,
        )
        variance_ns2 = np.divide(
            np.sum(
                local_residual
                * (surface_time[None, None, :] - mu_ns[:, :, None]) ** 2,
                axis=2,
            ),
            surface_mass,
            out=np.zeros_like(surface_mass),
            where=surface_mass > 0,
        )
        sigma_ns = np.sqrt(np.maximum(variance_ns2, (0.5 * dt_ns) ** 2))

        gaussian_mass = np.exp(
            -0.5
            * (
                (surface_time[None, None, :] - mu_ns[:, :, None])
                / sigma_ns[:, :, None]
            )
            ** 2
        )
        gaussian_mass_sum = gaussian_mass.sum(axis=2, keepdims=True)
        gaussian_mass = np.divide(
            gaussian_mass,
            gaussian_mass_sum,
            out=np.zeros_like(gaussian_mass),
            where=gaussian_mass_sum > 0,
        )
        fitted_signal_mass = surface_mass[:, :, None] * gaussian_mass
        fitted_peak_counts = totals * fitted_signal_mass.max(axis=2)
        range_m = mu_ns * 1e-9 * C_M_PER_S / 2.0

        chunk_valid = (
            valid
            & np.isfinite(range_m)
            & (range_m > 0)
            & (range_m <= range_max_m)
            & np.isfinite(fitted_peak_counts)
            & (fitted_peak_counts > 0)
        )

        raw_range[row0:row1] = np.where(chunk_valid, range_m, np.nan)
        reflectance[row0:row1] = np.where(chunk_valid, fitted_peak_counts, 0.0)
        gamma_shape[row0:row1] = np.where(valid, shape, np.nan)
        gamma_scale_ns[row0:row1] = np.where(valid, scale, np.nan)
        gaussian_sigma_ns[row0:row1] = np.where(chunk_valid, sigma_ns, np.nan)
        fit_valid[row0:row1] = chunk_valid
        print(f"Processed rows {row0:3d}:{row1:3d} / {height}", flush=True)

    confidence = reflectance * np.nan_to_num(raw_range, nan=0.0)
    finite_confidence = confidence[fit_valid & np.isfinite(confidence)]
    confidence_peak = float(finite_confidence.max()) if finite_confidence.size else 0.0
    threshold = confidence_relative * confidence_peak
    valid_mask = fit_valid & (confidence >= threshold) & (confidence > 0)
    depth = np.where(valid_mask, raw_range, np.nan)

    reflectance_display = np.zeros_like(reflectance)
    reflectance_max = float(reflectance[valid_mask].max()) if np.any(valid_mask) else 0.0
    if reflectance_max > 0:
        reflectance_display[valid_mask] = reflectance[valid_mask] / reflectance_max

    return {
        "depth": depth.astype(np.float32),
        "raw_range": raw_range.astype(np.float32),
        "reflectance": reflectance.astype(np.float32),
        "reflectance_normalized": reflectance_display.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "valid_mask": valid_mask,
        "fit_valid": fit_valid,
        "gamma_shape": gamma_shape.astype(np.float32),
        "gamma_scale_ns": gamma_scale_ns.astype(np.float32),
        "gaussian_sigma_ns": gaussian_sigma_ns.astype(np.float32),
        "confidence_threshold": np.asarray(threshold, dtype=np.float32),
        "range_per_bin_m": np.asarray(range_per_bin_m, dtype=np.float32),
    }


def save_range_image(
    path: Path,
    ranges: np.ndarray,
    range_max_m: float,
    title: str,
) -> None:
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color=(0.15, 0.15, 0.15, 1.0))
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(ranges, cmap=cmap, vmin=0.0, vmax=range_max_m)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Range (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_reflectance_image(path: Path, reflectance: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(reflectance, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Satat 2018 Estimated Reflectance")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Normalized reflectance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_valid_mask(path: Path, valid_mask: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(valid_mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    ax.set_title("Satat 2018 Valid Surface Mask")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1])
    colorbar.set_label("Valid")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_metrics(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    gt_range: np.ndarray,
    range_max_m: float,
) -> Dict[str, float]:
    gt_valid = np.isfinite(gt_range) & (gt_range > 0) & (gt_range <= range_max_m)
    evaluation_mask = valid_mask & gt_valid & np.isfinite(depth)
    valid_rate = float(np.count_nonzero(evaluation_mask) / max(np.count_nonzero(gt_valid), 1))

    metrics: Dict[str, float] = {
        "valid_pixel_rate": valid_rate,
        "evaluated_pixels": float(np.count_nonzero(evaluation_mask)),
        "gt_valid_pixels": float(np.count_nonzero(gt_valid)),
    }
    if not np.any(evaluation_mask):
        metrics.update(
            {
                "mae_m": math.nan,
                "rmse_m": math.nan,
                "accuracy_5cm": math.nan,
                "accuracy_15cm": math.nan,
            }
        )
        return metrics

    error = np.abs(depth[evaluation_mask].astype(np.float64) - gt_range[evaluation_mask])
    metrics.update(
        {
            "mae_m": float(np.mean(error)),
            "rmse_m": float(np.sqrt(np.mean(error**2))),
            "accuracy_5cm": float(np.mean(error <= 0.05)),
            "accuracy_15cm": float(np.mean(error <= 0.15)),
        }
    )
    return metrics


def save_metrics(path: Path, metrics: Dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_path = args.npz.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    frame_out = args.out_dir / input_path.stem
    frame_out.mkdir(parents=True, exist_ok=True)

    print(f"Input       : {input_path}")
    print(f"Output      : {frame_out.resolve()}")
    hist = load_spad_hist(input_path)
    gt_range = load_gt_range(input_path, hist.shape[:2])
    print(f"spad_hist   : shape={hist.shape}, dtype={hist.dtype}")
    print(
        f"Timing      : tmax={args.tmax_ns:g} ns, "
        f"KDE bandwidth={args.kde_bandwidth_ns:g} ns"
    )

    start = time.perf_counter()
    result = reconstruct_satat(
        hist=hist,
        tmax_ns=args.tmax_ns,
        kde_bandwidth_ns=args.kde_bandwidth_ns,
        range_max_m=args.range_max,
        confidence_relative=args.confidence_relative,
        chunk_rows=args.chunk_rows,
        min_photons=args.min_photons,
    )
    runtime_s = time.perf_counter() - start

    np.savez_compressed(frame_out / "result.npz", **result)
    save_range_image(
        frame_out / "depth.png",
        result["depth"],
        args.range_max,
        "Satat 2018 Estimated Range",
    )
    save_range_image(
        frame_out / "gt_range.png",
        gt_range,
        args.range_max,
        "Ground-Truth Range",
    )
    save_reflectance_image(
        frame_out / "reflectance.png",
        result["reflectance_normalized"],
    )
    save_valid_mask(frame_out / "valid_mask.png", result["valid_mask"])

    metrics = compute_metrics(
        depth=result["depth"],
        valid_mask=result["valid_mask"],
        gt_range=gt_range,
        range_max_m=args.range_max,
    )
    metrics["runtime_s"] = runtime_s
    save_metrics(frame_out / "metrics.csv", metrics)
    print(
        "Metrics     : "
        f"MAE={metrics['mae_m']:.4f} m, "
        f"RMSE={metrics['rmse_m']:.4f} m, "
        f"acc@5cm={metrics['accuracy_5cm']:.2%}, "
        f"acc@15cm={metrics['accuracy_15cm']:.2%}, "
        f"valid={metrics['valid_pixel_rate']:.2%}"
    )

    print(f"Runtime     : {runtime_s:.2f} s")
    print(f"Valid pixels: {np.count_nonzero(result['valid_mask'])} / {hist.shape[0] * hist.shape[1]}")
    print(
        "Generated   : depth.png, gt_range.png, reflectance.png, "
        "valid_mask.png, result.npz, metrics.csv"
    )


if __name__ == "__main__":
    main()
