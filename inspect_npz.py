import argparse
from pathlib import Path

import numpy as np


def format_number(value):
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.6g}"
    if isinstance(value, (np.bool_, bool)):
        return str(bool(value))
    return str(value)


def format_sample(values, sample_count):
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0 or sample_count <= 0:
        return "[]"
    count = min(int(sample_count), int(flat.size))
    items = [format_number(x) for x in flat[:count]]
    suffix = ", ..." if flat.size > count else ""
    return "[" + ", ".join(items) + suffix + "]"


def print_numeric_stats(arr):
    finite = np.isfinite(arr)
    finite_count = int(np.count_nonzero(finite))
    nonzero_count = int(np.count_nonzero(arr))
    print(f"  finite: {finite_count}/{arr.size}")
    print(f"  nonzero: {nonzero_count}/{arr.size}")

    if finite_count == 0:
        print("  min: n/a")
        print("  max: n/a")
        print("  mean: n/a")
        print("  sum: n/a")
        return

    finite_values = arr[finite]
    print(f"  min: {format_number(np.min(finite_values))}")
    print(f"  max: {format_number(np.max(finite_values))}")
    print(f"  mean: {format_number(np.mean(finite_values))}")
    print(f"  sum: {format_number(np.sum(finite_values))}")


def print_quick_diagnosis(arr):
    if arr.ndim == 2:
        print("  diagnosis: 2D array, likely image/depth/map-like data")
        if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
            finite = np.isfinite(arr)
            positive = int(np.count_nonzero(arr[finite] > 0)) if np.any(finite) else 0
            print(f"  positive finite pixels: {positive}/{arr.size}")
    elif arr.ndim == 3:
        h, w, bins = arr.shape
        print(f"  diagnosis: 3D array, possible H x W x B histogram ({h} x {w} x {bins})")
        if np.issubdtype(arr.dtype, np.number) and arr.size > 0:
            total_per_pixel = arr.sum(axis=-1)
            peak_per_pixel = arr.max(axis=-1)
            print(f"  per-pixel total min: {format_number(np.min(total_per_pixel))}")
            print(f"  per-pixel total max: {format_number(np.max(total_per_pixel))}")
            print(f"  per-pixel total mean: {format_number(np.mean(total_per_pixel))}")
            print(f"  per-pixel peak min: {format_number(np.min(peak_per_pixel))}")
            print(f"  per-pixel peak max: {format_number(np.max(peak_per_pixel))}")
            print(f"  per-pixel peak mean: {format_number(np.mean(peak_per_pixel))}")


def describe_array(name, arr, sample_count):
    arr = np.asarray(arr)
    print("-" * 72)
    print(f"key: {name}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  ndim: {arr.ndim}")
    print(f"  size: {arr.size}")

    if arr.size == 0:
        print("  sample: []")
        return

    if np.issubdtype(arr.dtype, np.number):
        print_numeric_stats(arr)
    else:
        print(f"  sample: {format_sample(arr, sample_count)}")
        return

    print_quick_diagnosis(arr)
    print(f"  sample: {format_sample(arr, sample_count)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print a lightweight text summary of arrays stored in a .npz file."
    )
    parser.add_argument("--npz", required=True, help="Path to the .npz file")
    parser.add_argument("--key", default=None, help="Only inspect this key")
    parser.add_argument(
        "--sample",
        type=int,
        default=8,
        help="Number of flattened sample values to print for each array",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow loading object arrays from the .npz file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=bool(args.allow_pickle)) as npz:
        keys = list(npz.files)
        print("=" * 72)
        print(f"file: {npz_path}")
        print(f"keys ({len(keys)}): {', '.join(keys)}")
        print("=" * 72)

        if args.key is not None:
            if args.key not in keys:
                raise KeyError(f"key {args.key!r} not found. Available keys: {keys}")
            keys = [args.key]

        for key in keys:
            try:
                describe_array(key, npz[key], int(args.sample))
            except ValueError as exc:
                print("-" * 72)
                print(f"key: {key}")
                print(f"  error: {exc}")
                if "Object arrays cannot be loaded" in str(exc):
                    print("  hint: rerun with --allow-pickle if you trust this file")


if __name__ == "__main__":
    main()
