"""Evaluate reconstructed surface point clouds against multi-frame GT ranges."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


def decode_npz_scalar(value: np.ndarray) -> str:
    item = value.item() if value.shape == () else value.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def discover_frames(npz_dir: str, max_frames: int) -> List[Path]:
    root = Path(npz_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)
    frames = sorted(root.glob("*.npz"), key=lambda path: path.name)
    if not frames:
        raise FileNotFoundError(f"No .npz files found in {root}")
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def load_poses(path: str) -> Dict[str, np.ndarray]:
    pose_path = Path(path)
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)

    poses: Dict[str, np.ndarray] = {}
    with pose_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 17:
                raise ValueError(
                    f"{pose_path}:{line_no}: expected frame key plus 16 values"
                )
            key = parts[0]
            if key in poses:
                raise ValueError(f"{pose_path}:{line_no}: duplicate pose key {key!r}")
            matrix = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
            matrix = matrix.reshape(4, 4)
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{pose_path}:{line_no}: pose contains non-finite values")
            poses[key] = matrix
    return poses


def require_metadata_float(metadata: Dict[str, object], key: str) -> float:
    if key not in metadata:
        raise KeyError(f"camera_model is missing {key!r}")
    value = float(metadata[key])
    if not math.isfinite(value):
        raise ValueError(f"camera_model.{key} must be finite")
    return value


def load_gt_frame(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    with np.load(path) as data:
        if "spad_hist" not in data:
            raise KeyError(f"{path}: missing spad_hist")
        if "gt_range" not in data:
            raise KeyError(f"{path}: missing gt_range")
        if "camera_model" not in data:
            raise KeyError(f"{path}: missing camera_model")
        spad_hist = np.asarray(data["spad_hist"])
        gt_range = np.asarray(data["gt_range"], dtype=np.float64)
        metadata = json.loads(decode_npz_scalar(np.asarray(data["camera_model"])))

    if spad_hist.ndim != 3:
        raise ValueError(f"{path}: spad_hist must be (H, W, T), got {spad_hist.shape}")
    if gt_range.shape != spad_hist.shape[:2]:
        raise ValueError(
            f"{path}: gt_range shape {gt_range.shape} does not match "
            f"spad_hist shape {spad_hist.shape[:2]}"
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: camera_model must decode to an object")
    return spad_hist, gt_range, metadata


def select_rays(spad_hist: np.ndarray, peak_thr: float, max_rays: int) -> np.ndarray:
    hist_data = spad_hist.reshape(-1, spad_hist.shape[-1])
    candidates = np.where(hist_data.max(axis=1) >= peak_thr)[0]
    if max_rays > 0 and candidates.size > max_rays:
        selection = np.linspace(0, candidates.size - 1, max_rays).astype(np.int64)
        return candidates[selection]
    return candidates


def project_gt_frame(
    spad_hist: np.ndarray,
    gt_range: np.ndarray,
    metadata: Dict[str, object],
    T_wc: np.ndarray,
    peak_thr: float,
    max_rays: int,
    range_max: float,
) -> np.ndarray:
    height, width = spad_hist.shape[:2]
    fx = require_metadata_float(metadata, "fx")
    fy = require_metadata_float(metadata, "fy")
    cx = require_metadata_float(metadata, "cx")
    cy = require_metadata_float(metadata, "cy")
    image_y_axis = str(metadata.get("image_y_axis", "down")).lower()
    if image_y_axis not in ("down", "up"):
        raise ValueError(f"unsupported camera_model.image_y_axis={image_y_axis!r}")

    rays = select_rays(spad_hist, peak_thr, max_rays)
    if rays.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    rows = rays // width
    cols = rays % width
    x_normalized = (cols.astype(np.float64) - cx) / fx
    y_normalized = (rows.astype(np.float64) - cy) / fy
    if image_y_axis == "up":
        y_normalized = -y_normalized

    directions_camera = np.column_stack(
        [x_normalized, y_normalized, np.ones(rays.size, dtype=np.float64)]
    )
    directions_camera /= np.linalg.norm(directions_camera, axis=1, keepdims=True)

    ranges = gt_range.reshape(-1)[rays]
    valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges <= range_max)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    rotation = np.asarray(T_wc[:3, :3], dtype=np.float64)
    origin = np.asarray(T_wc[:3, 3], dtype=np.float64)
    directions_world = directions_camera[valid] @ rotation.T
    directions_world /= np.linalg.norm(directions_world, axis=1, keepdims=True)
    return origin[None, :] + directions_world * ranges[valid, None]


def build_gt_cloud(
    frames: List[Path],
    poses: Dict[str, np.ndarray],
    peak_thr: float,
    max_rays: int,
    range_max: float,
) -> np.ndarray:
    point_chunks: List[np.ndarray] = []
    selected_total = 0
    for frame_index, frame_path in enumerate(frames, start=1):
        frame_key = frame_path.stem
        if frame_key not in poses:
            raise KeyError(f"poses file is missing frame {frame_key!r}")
        spad_hist, gt_range, metadata = load_gt_frame(frame_path)
        points = project_gt_frame(
            spad_hist,
            gt_range,
            metadata,
            poses[frame_key],
            peak_thr,
            max_rays,
            range_max,
        )
        point_chunks.append(points)
        selected_total += points.shape[0]
        if frame_index == 1 or frame_index % 10 == 0 or frame_index == len(frames):
            print(
                f"GT frame {frame_index}/{len(frames)}: {frame_key}, "
                f"valid sampled points={points.shape[0]}, cumulative={selected_total}"
            )

    if not point_chunks or selected_total == 0:
        raise ValueError("GT cloud is empty")
    return np.concatenate(point_chunks, axis=0)


def read_ascii_ply_xyz(path: str) -> np.ndarray:
    ply_path = Path(path)
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)

    vertex_count = None
    vertex_properties: List[str] = []
    in_vertex_element = False
    with ply_path.open("r", encoding="ascii") as handle:
        first_line = handle.readline().strip()
        if first_line != "ply":
            raise ValueError(f"{ply_path}: not a PLY file")

        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"{ply_path}: missing end_header")
            line = raw.strip()
            if line == "format binary_little_endian 1.0" or line == "format binary_big_endian 1.0":
                raise ValueError(f"{ply_path}: only ASCII PLY is supported")
            if line.startswith("element "):
                parts = line.split()
                in_vertex_element = len(parts) == 3 and parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
                    vertex_properties = []
                continue
            if line.startswith("property ") and in_vertex_element:
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"{ply_path}: list vertex properties are unsupported")
                vertex_properties.append(parts[2])
                continue
            if line == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"{ply_path}: missing vertex element")
        try:
            xyz_indices = [vertex_properties.index(axis) for axis in ("x", "y", "z")]
        except ValueError as exc:
            raise ValueError(f"{ply_path}: vertex properties must include x, y, z") from exc

        points = np.empty((vertex_count, 3), dtype=np.float64)
        for index in range(vertex_count):
            raw = handle.readline()
            if not raw:
                raise ValueError(
                    f"{ply_path}: expected {vertex_count} vertices, found {index}"
                )
            values = raw.split()
            points[index, 0] = float(values[xyz_indices[0]])
            points[index, 1] = float(values[xyz_indices[1]])
            points[index, 2] = float(values[xyz_indices[2]])

    if not np.all(np.isfinite(points)):
        raise ValueError(f"{ply_path}: point cloud contains non-finite coordinates")
    if points.shape[0] == 0:
        raise ValueError(f"{ply_path}: point cloud is empty")
    return points


def distance_summary(distances: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean_m": float(np.mean(distances)),
        f"{prefix}_median_m": float(np.median(distances)),
        f"{prefix}_p90_m": float(np.quantile(distances, 0.90)),
        f"{prefix}_p95_m": float(np.quantile(distances, 0.95)),
    }


def evaluate_surface(surface_path: str, gt_points: np.ndarray) -> Dict[str, object]:
    prediction = read_ascii_ply_xyz(surface_path)
    print(f"loaded prediction {surface_path}: N={prediction.shape[0]}")

    gt_tree = cKDTree(gt_points)
    prediction_tree = cKDTree(prediction)
    prediction_to_gt = gt_tree.query(prediction, k=1, workers=-1)[0]
    gt_to_prediction = prediction_tree.query(gt_points, k=1, workers=-1)[0]

    precision_5cm = float(np.mean(prediction_to_gt <= 0.05))
    recall_5cm = float(np.mean(gt_to_prediction <= 0.05))
    denominator = precision_5cm + recall_5cm
    f1_5cm = 0.0 if denominator <= 0.0 else 2.0 * precision_5cm * recall_5cm / denominator

    result: Dict[str, object] = {
        "surface": str(surface_path),
        "prediction_points": int(prediction.shape[0]),
        "gt_points": int(gt_points.shape[0]),
    }
    result.update(distance_summary(prediction_to_gt, "pred_to_gt"))
    result.update(
        {
            "pred_within_5cm": precision_5cm,
            "pred_within_10cm": float(np.mean(prediction_to_gt <= 0.10)),
            "phantom_rate_15cm": float(np.mean(prediction_to_gt > 0.15)),
        }
    )
    result.update(distance_summary(gt_to_prediction, "gt_to_pred"))
    result.update(
        {
            "coverage_5cm": recall_5cm,
            "coverage_10cm": float(np.mean(gt_to_prediction <= 0.10)),
            "missing_rate_15cm": float(np.mean(gt_to_prediction > 0.15)),
            "precision_5cm": precision_5cm,
            "recall_5cm": recall_5cm,
            "f1_5cm": f1_5cm,
        }
    )
    return result


def print_results(results: List[Dict[str, object]]) -> None:
    print("\nSurface-to-GT metrics")
    header = (
        "surface",
        "N_pred",
        "P->GT med",
        "P->GT P95",
        "precision@5cm",
        "phantom>15cm",
        "GT->P med",
        "GT->P P95",
        "recall@5cm",
        "missing>15cm",
        "F1@5cm",
    )
    print(" | ".join(header))
    for result in results:
        print(
            f"{Path(str(result['surface'])).name} | "
            f"{int(result['prediction_points'])} | "
            f"{float(result['pred_to_gt_median_m']):.6f} | "
            f"{float(result['pred_to_gt_p95_m']):.6f} | "
            f"{float(result['precision_5cm']):.4%} | "
            f"{float(result['phantom_rate_15cm']):.4%} | "
            f"{float(result['gt_to_pred_median_m']):.6f} | "
            f"{float(result['gt_to_pred_p95_m']):.6f} | "
            f"{float(result['recall_5cm']):.4%} | "
            f"{float(result['missing_rate_15cm']):.4%} | "
            f"{float(result['f1_5cm']):.4%}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-dir", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--max-rays", type=int, default=2000)
    parser.add_argument("--peak-thr", type=float, default=5.0)
    parser.add_argument("--range-max", type=float, default=7.0)
    parser.add_argument("--surface", nargs="+", required=True)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.max_rays < 0:
        raise ValueError("--max-rays must be non-negative")
    if not math.isfinite(args.peak_thr):
        raise ValueError("--peak-thr must be finite")
    if not math.isfinite(args.range_max) or args.range_max <= 0.0:
        raise ValueError("--range-max must be finite and positive")

    frames = discover_frames(args.npz_dir, args.max_frames)
    poses = load_poses(args.poses)
    print(
        f"building GT cloud from {len(frames)} frames, "
        f"peak_thr={args.peak_thr:g}, max_rays={args.max_rays}"
    )
    gt_points = build_gt_cloud(
        frames,
        poses,
        float(args.peak_thr),
        int(args.max_rays),
        float(args.range_max),
    )
    print(f"GT cloud complete: N={gt_points.shape[0]}")

    results = [evaluate_surface(path, gt_points) for path in args.surface]
    print_results(results)

    output_path = Path(args.out_csv)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"saved metrics CSV: {output_path}")


if __name__ == "__main__":
    main()
