import argparse
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = 299_792_458.0  # m/s
COMMON_SIZES = ((192, 256), (256, 192))
TARGET_PIXELS = 192 * 256

OUTPUT_COLUMNS = ["Timestamp", "X", "Y", "Z", "Reflectivity"]


def _str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _extract_timestamp_from_name(path):
    name = os.path.basename(path)
    m = re.search(r"(\d+)(?=\.txt$)", name)
    if m:
        return int(m.group(1))

    nums = re.findall(r"\d+", name)
    if nums:
        return int(nums[-1])
    return -1


def _infer_shape(num_pos):
    for h, w in COMMON_SIZES:
        if h * w == num_pos:
            return h, w, num_pos

    if num_pos >= TARGET_PIXELS:
        return 192, 256, TARGET_PIXELS

    h = int(np.sqrt(num_pos))
    while h > 1 and num_pos % h != 0:
        h -= 1
    w = num_pos // h
    return h, w, num_pos


def _build_intrinsics(fx, fy, cx, cy, k1, k2, p1, p2):
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    D = np.array([k1, k2, p1, p2], dtype=np.float64)
    return K, D


def _top2_counts_per_row(data):
    top2 = np.partition(data, -2, axis=1)[:, -2:]
    second = top2[:, 0].astype(np.float32)
    peak = top2[:, 1].astype(np.float32)
    return peak, second


def _load_hist_intensity_depth(txt_path, dt_ps):
    data = np.loadtxt(txt_path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    num_pos, _ = data.shape
    intensity_1d, _ = _top2_counts_per_row(data)
    peak_bin_1d = data.argmax(axis=1).astype(np.int32)

    dt = dt_ps * 1e-12
    bin_to_m = C * dt / 2.0
    # depth_m_1d = peak_bin_1d.astype(np.float32) * bin_to_m
    depth_m_1d = ( peak_bin_1d.astype(np.float32) - 17) * bin_to_m#25 18
    
    ###
    offset = np.loadtxt("./offset.txt", dtype=float)
    idx_offset = np.argwhere( offset < 10 )[:,0]
    depth_m_1d[idx_offset] -= offset[idx_offset]

    height, width, keep_n = _infer_shape(num_pos)
    if keep_n < num_pos:
        intensity_1d = intensity_1d[:keep_n]
        depth_m_1d = depth_m_1d[:keep_n]

    intensity = intensity_1d.reshape(height, width)
    depth_m = depth_m_1d.reshape(height, width)
    
    return intensity, depth_m, height, width


def _compute_points_undistort_points(
    intensity,
    depth_m,
    K,
    D,
    depth_is_range,
    min_range_m,
    max_range_m,
    intensity_min,
    intensity_max,
    undistort_intensity,
):
    if undistort_intensity:
        intensity_u = cv2.undistort(intensity, K, D)
    else:
        intensity_u = intensity

    depth_u = depth_m

    h, w = depth_u.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    valid = (
        np.isfinite(depth_u)
        & (depth_u > float(min_range_m))
        & (depth_u < float(max_range_m))
        & (intensity_u >= float(intensity_min))
    )
    if intensity_max is not None:
        valid &= intensity_u <= float(intensity_max)
    
    offset = np.loadtxt("offset.txt", dtype=float)
    offset_m = offset.reshape(h, w)
    valid_offset = offset_m < 10
    valid = valid & valid_offset

    u_valid = u[valid].astype(np.float32)
    v_valid = v[valid].astype(np.float32)
    d_valid = depth_u[valid].astype(np.float32)
    i_valid = intensity_u[valid].astype(np.float32)

    uv = np.stack([u_valid, v_valid], axis=1).reshape(-1, 1, 2).astype(np.float32)
    xy = cv2.undistortPoints(uv, K, D)
    x_n = xy[:, 0, 0]
    y_n = xy[:, 0, 1]

    ray = np.stack([x_n, y_n, np.ones_like(x_n)], axis=1)
    ray_norm = np.linalg.norm(ray, axis=1, keepdims=True)
    ray_unit = ray / np.maximum(ray_norm, 1e-12)

    if depth_is_range:
        pts = ray_unit * d_valid.reshape(-1, 1)
    else:
        pts = ray_unit * (d_valid.reshape(-1, 1) / np.maximum(ray_unit[:, 2:3], 1e-12))

    reflectivity = np.rint(i_valid).astype(np.int32)
    return pts.astype(np.float32), reflectivity, intensity_u, depth_u


def _normalize_to_u8(values):
    arr = values.astype(np.float32)
    if arr.size == 0:
        return np.array([], dtype=np.uint8)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) * (255.0 / (hi - lo))).clip(0, 255).astype(np.uint8)


def _write_ply_ascii(path, pts, reflectivity):
    i_255 = _normalize_to_u8(reflectivity.astype(np.float32))
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for idx in range(pts.shape[0]):
            x, y, z = pts[idx]
            ii = int(i_255[idx]) if idx < i_255.shape[0] else 0
            f.write(f"{x} {y} {z} {ii} {ii} {ii}\n")


def _save_maps_png(path, intensity_u, depth_u):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(intensity_u, origin="upper")
    plt.title("Intensity (max count)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(depth_u, origin="upper")
    plt.title("Depth (m) from peak bin")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _points_to_simple_csv(pts, reflectivity, timestamp):
    n = pts.shape[0]
    df = pd.DataFrame(
        {
            "Timestamp": np.full(n, int(timestamp), dtype=np.int64),
            "X": pts[:, 0],
            "Y": pts[:, 1],
            "Z": pts[:, 2],
            "Reflectivity": reflectivity.astype(np.int32),
        },
        columns=OUTPUT_COLUMNS,
    )
    return df


def _maybe_visualize_open3d(pts, reflectivity, point_size=1.5):
    try:
        import open3d as o3d
        import matplotlib.cm as cm
    except Exception:
        print("[warn] open3d/matplotlib colormap not available, skip visualization")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    if reflectivity.size > 0:
        refl = reflectivity.astype(np.float32)
        refl_norm = (refl - refl.min()) / (refl.max() - refl.min() + 1e-12)
        colors = cm.get_cmap("turbo")(refl_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SP Point Cloud (undistortPoints)", width=1200, height=800)
    vis.add_geometry(pcd)
    vis.add_geometry(axis)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = float(point_size)
        opt.background_color = np.array([0.02, 0.02, 0.02], dtype=np.float64)
    vis.run()
    vis.destroy_window()


def convert_one_txt(
    txt_path,
    output_base,
    dt_ps,
    depth_is_range,
    undistort_intensity,
    K,
    D,
    min_range_m,
    max_range_m,
    intensity_min,
    intensity_max,
    output_mode,
    save_maps,
    show,
    show_point_size=1.5,
):
    timestamp = _extract_timestamp_from_name(txt_path)
    if timestamp < 0:
        raise ValueError(f"No timestamp found in filename: {txt_path}")

    intensity, depth_m, _, _ = _load_hist_intensity_depth(txt_path=txt_path, dt_ps=dt_ps)
    pts, refl, intensity_u, depth_u = _compute_points_undistort_points(
        intensity=intensity,
        depth_m=depth_m,
        K=K,
        D=D,
        depth_is_range=depth_is_range,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        undistort_intensity=undistort_intensity,
    )

    csv_path = ""
    ply_path = ""
    map_path = ""

    if output_mode in ("csv", "both"):
        csv_path = output_base + ".csv"
        _points_to_simple_csv(pts, refl, timestamp).to_csv(csv_path, index=False)

    if output_mode in ("ply", "both"):
        ply_path = output_base + ".ply"
        _write_ply_ascii(ply_path, pts, refl)

    if save_maps:
        map_path = output_base + "_maps.png"
        _save_maps_png(map_path, intensity_u, depth_u)

    if show:
        _maybe_visualize_open3d(pts, refl, point_size=show_point_size)

    return {
        "txt": txt_path,
        "timestamp": timestamp,
        "points": int(pts.shape[0]),
        "csv": csv_path,
        "ply": ply_path,
        "maps": map_path,
    }


def batch_convert(
    input_dir,
    output_dir,
    pattern,
    prefix,
    start_index,
    dt_ps,
    depth_is_range,
    undistort_intensity,
    K,
    D,
    min_range_m,
    max_range_m,
    intensity_min,
    intensity_max,
    output_mode,
    save_maps,
):
    os.makedirs(output_dir, exist_ok=True)

    txt_files = glob.glob(os.path.join(input_dir, pattern))
    if not txt_files:
        raise FileNotFoundError(f"No files matched: {os.path.join(input_dir, pattern)}")

    txt_files.sort(key=lambda p: (_extract_timestamp_from_name(p), os.path.basename(p)))

    results = []
    out_idx = int(start_index)
    for txt_path in txt_files:
        out_base = os.path.join(output_dir, f"{prefix}{out_idx}")
        info = convert_one_txt(
            txt_path=txt_path,
            output_base=out_base,
            dt_ps=dt_ps,
            depth_is_range=depth_is_range,
            undistort_intensity=undistort_intensity,
            K=K,
            D=D,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            output_mode=output_mode,
            save_maps=save_maps,
            show=False,
        )
        results.append(info)

        outputs = []
        if info["csv"]:
            outputs.append(os.path.basename(info["csv"]))
        if info["ply"]:
            outputs.append(os.path.basename(info["ply"]))
        if info["maps"]:
            outputs.append(os.path.basename(info["maps"]))
        out_text = ", ".join(outputs) if outputs else "<none>"

        print(
            f"[{out_idx}] {os.path.basename(txt_path)} -> {out_text}, "
            f"ts={info['timestamp']}, points={info['points']}"
        )
        out_idx += 1

    return results


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert single-photon histogram txt to point cloud with cv2.undistortPoints."
    )

    parser.add_argument("--input-dir", type=str, default="./imaging")
    parser.add_argument("--output-dir", type=str, default="./imaging")
    parser.add_argument(
        "--pattern",
        type=str,
        default="RawDataHistogramMap_frame_0_*.txt",
        help="glob pattern under input-dir",
    )
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--start-index", type=int, default=1)

    parser.add_argument("--single-txt", type=str, default="", help="optional: convert one txt only")
    parser.add_argument(
        "--single-out-base",
        type=str,
        default="",
        help="optional: output base path without extension for --single-txt",
    )

    parser.add_argument("--dt-ps", type=float, default=750.0)
    parser.add_argument("--depth-is-range", type=_str2bool, default=True)
    parser.add_argument("--undistort-intensity", type=_str2bool, default=True)

    parser.add_argument("--fx", type=float, default=118.6514575329715)
    parser.add_argument("--fy", type=float, default=118.7964934010577)
    parser.add_argument("--cx", type=float, default=130.6802784645003)
    parser.add_argument("--cy", type=float, default=100.3605702468140)

    parser.add_argument("--k1", type=float, default=-0.257910069121181)
    parser.add_argument("--k2", type=float, default=0.053237073644331)
    parser.add_argument("--p1", type=float, default=0.0)
    parser.add_argument("--p2", type=float, default=0.0)

    parser.add_argument("--min-range-m", type=float, default=0.0)
    parser.add_argument("--max-range-m", type=float, default=20.0)
    parser.add_argument("--intensity-min", type=float, default=1.0)
    parser.add_argument(
        "--intensity-max",
        type=float,
        default=None,
        help="optional max photon-count threshold",
    )

    parser.add_argument(
        "--output-mode",
        type=str,
        choices=["ply", "csv", "both"],
        default="csv",
        help="export file mode",
    )
    parser.add_argument("--save-maps", type=_str2bool, default=True)
    parser.add_argument("--show", type=_str2bool, default=False)
    parser.add_argument("--show-point-size", type=float, default=1.5)

    return parser


def main():
    args = build_parser().parse_args()

    K, D = _build_intrinsics(
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        k1=args.k1,
        k2=args.k2,
        p1=args.p1,
        p2=args.p2,
    )

    if args.single_txt:
        if args.single_out_base:
            out_base = args.single_out_base
        else:
            base = os.path.splitext(os.path.basename(args.single_txt))[0]
            out_base = os.path.join(args.output_dir, base)

        os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
        info = convert_one_txt(
            txt_path=args.single_txt,
            output_base=out_base,
            dt_ps=args.dt_ps,
            depth_is_range=args.depth_is_range,
            undistort_intensity=args.undistort_intensity,
            K=K,
            D=D,
            min_range_m=args.min_range_m,
            max_range_m=args.max_range_m,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max,
            output_mode=args.output_mode,
            save_maps=args.save_maps,
            show=args.show,
            show_point_size=args.show_point_size,
        )
        print(
            f"done: {os.path.basename(info['txt'])}, points={info['points']}, "
            f"csv={info['csv'] or '-'}, ply={info['ply'] or '-'}, maps={info['maps'] or '-'}"
        )
        return

    results = batch_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        prefix=args.prefix,
        start_index=args.start_index,
        dt_ps=args.dt_ps,
        depth_is_range=args.depth_is_range,
        undistort_intensity=args.undistort_intensity,
        K=K,
        D=D,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
        output_mode=args.output_mode,
        save_maps=args.save_maps,
    )
    print(f"done: converted {len(results)} files")

    if args.show and results:
        target = results[0]["txt"]
        out_base = os.path.join(args.output_dir, "_preview_first")
        info = convert_one_txt(
            txt_path=target,
            output_base=out_base,
            dt_ps=args.dt_ps,
            depth_is_range=args.depth_is_range,
            undistort_intensity=args.undistort_intensity,
            K=K,
            D=D,
            min_range_m=args.min_range_m,
            max_range_m=args.max_range_m,
            intensity_min=args.intensity_min,
            intensity_max=args.intensity_max,
            output_mode="ply",
            save_maps=False,
            show=True,
            show_point_size=args.show_point_size,
        )
        print(f"preview shown for: {os.path.basename(info['txt'])}")


if __name__ == "__main__":
    main()
