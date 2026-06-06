import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import spad_npz_occupancy_mapping as mapping


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Diagnose clear and smoke-aware range likelihoods for one SPAD pixel."
    )
    ap.add_argument("--npz", required=True)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--col", type=int, required=True)
    ap.add_argument("--n-pulses", type=int, default=None)
    ap.add_argument("--peak-bin", type=int, default=None)
    ap.add_argument("--tmax-ns", type=float, default=None)
    ap.add_argument("--Wr-bin", type=int, default=12)
    ap.add_argument("--M", type=int, default=81)
    ap.add_argument("--win-half", type=int, default=25)
    ap.add_argument("--sigma-bins", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--out", default="smoke_likelihood_diagnostic.png")
    return ap.parse_args()


def metadata_tmax_ns(metadata, cli_value: Optional[float]) -> float:
    if cli_value is not None:
        return float(cli_value)
    capture = metadata.get("capture_model", {})
    if isinstance(capture, dict) and "tmax_ns" in capture:
        return float(capture["tmax_ns"])
    return 100.0


def main() -> None:
    args = parse_args()
    spad_hist, metadata = mapping.load_spad_frame_npz(Path(args.npz))
    height, width, n_bins = spad_hist.shape
    if not (0 <= args.row < height and 0 <= args.col < width):
        raise IndexError(f"pixel ({args.row}, {args.col}) is outside image shape {(height, width)}")

    model_args = SimpleNamespace(likelihood_model="smoke", n_pulses=args.n_pulses)
    _, kappa, gamma, smoke_step_m = mapping.resolve_likelihood_model(model_args, metadata)
    n_pulses = mapping.resolve_n_pulses(model_args, metadata, "smoke")
    tmax_ns = metadata_tmax_ns(metadata, args.tmax_ns)
    bin_to_m = mapping.C_M_PER_S * (tmax_ns * 1e-9) / 2.0 / float(n_bins)

    h = spad_hist[args.row, args.col, :].astype(np.float64)
    peak_bin = int(np.argmax(h)) if args.peak_bin is None else int(args.peak_bin)
    if not 0 <= peak_bin < n_bins:
        raise ValueError(f"peak_bin must be in [0, {n_bins - 1}]")

    fog = metadata.get("fog_model", {})
    fog_range_max = float(fog.get("range_max_m", n_bins * bin_to_m))
    templates, smoke_step_used = mapping.build_smoke_integral_templates(
        n_bins=n_bins,
        bin_to_m=bin_to_m,
        sigma_bins=args.sigma_bins,
        kappa=kappa,
        fog_step_m=smoke_step_m,
        range_max_m=fog_range_max,
    )

    r_clear, _, p_clear = mapping.compute_ll_grid_numba(
        h,
        peak_bin,
        args.Wr_bin,
        args.M,
        args.win_half,
        args.sigma_bins,
        args.tau,
        bin_to_m,
    )
    r_smoke, _, p_smoke = mapping.compute_ll_grid_smoke_numba(
        h,
        peak_bin,
        args.Wr_bin,
        args.M,
        args.win_half,
        args.sigma_bins,
        args.tau,
        bin_to_m,
        gamma,
        float(n_pulses),
        templates,
        smoke_step_used,
    )

    clear_idx = int(np.argmax(p_clear))
    smoke_idx = int(np.argmax(p_smoke))
    clear_range = float(r_clear[clear_idx])
    smoke_range = float(r_smoke[smoke_idx])

    smoke_rb = smoke_range / bin_to_m
    S_full = mapping.build_S_gaussian(0, n_bins, smoke_rb, args.sigma_bins)
    row_idx = int(np.floor(max(smoke_range, 0.0) / smoke_step_used))
    row_idx = min(max(row_idx, 0), templates.shape[0] - 1)
    G_full = templates[row_idx, :].astype(np.float64)
    smoke_scale = float(n_pulses) * float(gamma)
    _, alpha, beta = mapping.fit_profile_one_r_smoke_count_numba(
        h,
        S_full,
        G_full,
        smoke_scale,
    )

    fitted_surface = float(alpha) * S_full
    fitted_smoke = smoke_scale * G_full
    fitted_background = np.full(n_bins, float(beta), dtype=np.float64)
    fitted_total = fitted_surface + fitted_smoke + fitted_background

    gt_range = None
    with np.load(args.npz) as data:
        if "gt_range" in data:
            gt_range = float(data["gt_range"][args.row, args.col])

    bins = np.arange(n_bins)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    axes[0].step(bins, h, where="mid", color="black", linewidth=0.8, label="observed spad_hist")
    axes[0].plot(bins, fitted_total, color="tab:red", linewidth=1.4, label="fitted total")
    axes[0].axvline(peak_bin, color="0.5", linestyle=":", label=f"proposal peak={peak_bin}")
    axes[0].set_ylabel("Photon count")
    axes[0].legend()

    axes[1].plot(bins, fitted_surface, label="fitted surface")
    axes[1].plot(bins, fitted_smoke, label="fixed smoke")
    axes[1].plot(bins, fitted_background, label="fitted background")
    axes[1].set_ylabel("Expected count")
    axes[1].legend()

    axes[2].plot(r_clear, p_clear, label="clear posterior")
    axes[2].plot(r_smoke, p_smoke, label="smoke posterior")
    axes[2].axvline(clear_range, color="tab:blue", linestyle=":")
    axes[2].axvline(smoke_range, color="tab:orange", linestyle=":")
    if gt_range is not None:
        axes[2].axvline(gt_range, color="tab:green", linestyle="--", label="gt_range")
    axes[2].set_xlabel("Range (m)")
    axes[2].set_ylabel("Posterior probability")
    axes[2].legend()

    fig.suptitle(
        f"Pixel ({args.row}, {args.col}) | n_pulses={n_pulses}, "
        f"kappa={kappa:g}, gamma={gamma:g}"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)

    print(f"pixel=({args.row}, {args.col}) peak_bin={peak_bin}")
    print(f"n_pulses={n_pulses} kappa={kappa:g} gamma={gamma:g} smoke_step={smoke_step_used:g}")
    print(f"clear_range={clear_range:.6f} m")
    print(f"smoke_range={smoke_range:.6f} m")
    if gt_range is not None:
        print(f"gt_range={gt_range:.6f} m")
        print(f"clear_abs_error={abs(clear_range - gt_range):.6f} m")
        print(f"smoke_abs_error={abs(smoke_range - gt_range):.6f} m")
    print(f"alpha={float(alpha):.6f} beta={float(beta):.6f}")
    print(f"fitted_surface_sum={float(fitted_surface.sum()):.6f}")
    print(f"fitted_smoke_sum={float(fitted_smoke.sum()):.6f}")
    print(f"fitted_background_sum={float(fitted_background.sum()):.6f}")
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
