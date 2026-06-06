"""
Generate simulated SPAD histograms from an ICL-NUIM RGB-D trajectory.

Input directory layout:
  scene_00_0000.png
  scene_00_0000.depth
  scene_00_0000.txt
  ...

Output:
  one .npz per frame containing:
    spad_hist : (Nr, Nc, N_tbins) float32
    gt_depth_z: (Nr, Nc)          float32
    gt_range  : (Nr, Nc)          float32
    camera_model: JSON metadata string
    phi_bar   : (Nr, Nc, N_tbins) float32   optional
  poses.txt compatible with spad_npz_occupancy_mapping.py
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from SPCSim.data_loaders.transient_loaders import TransientGenerator
from SPCSim.sensors.dtof import BaseEWHSPC


PDE = 0.03675
TMAX = 100
N_TBINS = 1000
FWHM = 0.5
N_PULSES = 5000
ALPHA_SIG = 0.5
ALPHA_BKG = 1.0

FOG_ENABLED = False

DCR_CPS = 100

GEN_FOG_CONFIG: Dict[str, object] = {}
LAST_EXTRA_SAVE: Dict[str, np.ndarray] = {}

NR = 256
NC = 256

ICL_WIDTH = 640
ICL_HEIGHT = 480
ICL_FX = 481.20
ICL_FY = 480.00
ICL_CX = 319.50
ICL_CY = 239.50

POSE_RE = re.compile(
    r"^(cam_pos|cam_dir|cam_up|cam_lookat|cam_sky|cam_right|cam_fpoint|cam_angle)\s*=\s*(.+?);$"
)


def build_gaussian_irf_1d(center_bin: float, sigma_bins: float, n_bins: int, device: str):
    x = torch.arange(n_bins, dtype=torch.float32, device=device)
    sigma = max(float(sigma_bins), 1e-6)
    irf = torch.exp(-0.5 * ((x - float(center_bin)) / sigma) ** 2)
    total = torch.sum(irf)
    if float(total.detach().cpu()) > 0.0:
        irf = irf / total
    return irf


def make_ray_integral_smoke_phi_bar(
    nr: int,
    nc: int,
    range_limit_m,
    sigma: float,
    density: float,
    fog_step: float,
    range_max: float,
    device: str,
):
    """Build per-pixel smoke returns, clipped at each pixel's surface range."""
    dmax = 3e8 * TMAX * 1e-9 / 2
    step = max(float(fog_step), 1e-6)
    rmax = min(max(float(range_max), 0.0), float(dmax))
    if rmax <= 0.0 or float(density) <= 0.0:
        return torch.zeros((nr, nc, N_TBINS), dtype=torch.float32, device=device)

    sigma_bins = FWHM / (2.355 * (TMAX / N_TBINS))
    step_profiles = []
    s_values = np.arange(0.0, rmax + 0.5 * step, step, dtype=np.float64)
    for s in s_values:
        center_bin = (s / dmax) * N_TBINS
        irf = build_gaussian_irf_1d(center_bin, sigma_bins, N_TBINS, device)
        weight = float(density) * math.exp(-2.0 * float(sigma) * float(s)) * step
        step_profiles.append(irf * float(weight))

    per_step = torch.stack(step_profiles, dim=0)
    
    # cum_profile[k] 表示：如果某个像素的物体距离大约是 s_k，那么这个像素前方烟雾产生的总时间回波
    cum_profile = torch.cumsum(per_step, dim=0)

    if torch.is_tensor(range_limit_m):
        range_t = range_limit_m.to(device=device, dtype=torch.float32)
    else:
        range_t = torch.tensor(range_limit_m, device=device, dtype=torch.float32)
    range_t = torch.clamp(range_t, min=0.0, max=float(rmax))
    
    # idx 的含义是：这个像素应该使用 cum_profile 的第几个累计结果
    idx = torch.floor(range_t / float(step)).to(dtype=torch.long)
    idx = torch.clamp(idx, min=0, max=cum_profile.shape[0] - 1)
    return cum_profile[idx.reshape(-1)].reshape(nr, nc, N_TBINS).contiguous()


def make_dark_phi_bar(nr: int, nc: int, device: str):
    bin_size = TMAX * 1e-9 / N_TBINS
    b_d_per_bin = DCR_CPS * bin_size
    return torch.full((nr, nc, N_TBINS), b_d_per_bin, dtype=torch.float32, device=device)


def parse_vec3(value: str) -> np.ndarray:
    if not value.startswith("[") or "]" not in value:
        raise ValueError(f"invalid vector literal: {value!r}")
    # 提取向量内容：去掉前后的方括号,切片提取中间的内容，例如"[0.0, 0.0, 0.0]" → "0.0, 0.0, 0.0"
    content = value[value.find("[") + 1 : value.find("]")]
    # 用逗号,将内容分割成多个部分,对每个部分调用strip()，去掉前后的空白字符,例如"0.0, 0.0, 0.0" → ["0.0", "0.0", "0.0"]
    parts = [p.strip() for p in content.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 3 vector values, got {len(parts)} in {value!r}")
    vec = np.asarray([float(p) for p in parts], dtype=np.float64)
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"non-finite vector values in {value!r}")
    return vec


def parse_scalar(value: str) -> float:
    scalar = float(value.strip().rstrip("'"))
    if not np.isfinite(scalar):
        raise ValueError(f"non-finite scalar value in {value!r}")
    return scalar


def normalize(vec: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        raise ValueError(f"{name} has near-zero norm")
    return vec / norm


def load_camera_metadata(txt_path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            match = POSE_RE.match(line)
            if match is None:
                raise ValueError(f"{txt_path}:{line_no}: unsupported line format: {line!r}")
            key, value = match.groups()
            if key == "cam_angle":
                data[key] = parse_scalar(value)
            else:
                data[key] = parse_vec3(value)
    return data


def metadata_to_t_wc(meta: Dict[str, object]) -> np.ndarray:
    """
    输入一个包含相机位置、前向、向上方向的元数据字典，输出一个标准的 4x4 T_wc 矩阵。
    这个矩阵完全符合建图脚本的相机坐标系约定（x 向右、y 向下、z 向前），可以直接写入 poses.txt 文件供建图使用
    """
    # 构建 T_wc 矩阵只需要三个最核心的参数：相机位置cam_pos、相机前向cam_dir、世界向上方向cam_up
    required = ["cam_pos", "cam_dir", "cam_up"]
    missing = [key for key in required if key not in meta]
    if missing:
        raise KeyError(f"missing camera fields: {missing}")

    # origin：相机光心在世界坐标系中的三维坐标，直接作为 T_wc 矩阵的平移向量
    origin = np.asarray(meta["cam_pos"], dtype=np.float64)
    # z_forward：相机的前向方向向量（镜头指向的方向），归一化后作为相机坐标系的 z 轴
    z_forward = normalize(np.asarray(meta["cam_dir"], dtype=np.float64), "cam_dir")
    # up_world：世界坐标系的向上方向向量（不是相机的向上方向）
    up_world = normalize(np.asarray(meta["cam_up"], dtype=np.float64), "cam_up")

    # 使用叉乘计算相机的右向向量,叉乘顺序绝对不能搞反,必须是up_world × z_forward,两个向量的叉乘结果，一定同时垂直于这两个向量
    # 无论up_world和z_forward是否正交，x_right都一定同时垂直于它们两个,保证了相机的右向向量，一定垂直于相机的前向向量
    # 结果：x_right就是相机坐标系的 x 轴正方向（图像向右）
    x_right = np.cross(up_world, z_forward)
    x_right = normalize(x_right, "camera right axis")
    # 这次用已经得到的x_right和原始的z_forward做叉乘,结果y_down一定同时垂直于z_forward和x_right
    y_down = np.cross(z_forward, x_right)
    y_down = normalize(y_down, "camera down axis")

    # Camera convention for the mapping code:
    #   x -> image right, y -> image down, z -> forward.
    # 把三个正交基向量组合成 3x3 的旋转矩阵 R_wc,np.column_stack：将三个向量作为矩阵的三列
    # 旋转矩阵的本质：每一列代表新坐标系（相机坐标系）的一个基向量在旧坐标系（世界坐标系）下的坐标
    R_wc = np.column_stack((x_right, y_down, z_forward))
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = origin
    return T_wc


def write_poses_txt(path: Path, poses: Iterable[Tuple[str, np.ndarray]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for frame_key, T_wc in poses:
            flat = " ".join(f"{v:.9g}" for v in T_wc.reshape(-1))
            f.write(f"{frame_key} {flat}\n")


def load_png_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def load_depth_txt(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32)
    if values.size != ICL_WIDTH * ICL_HEIGHT:
        raise ValueError(
            f"{path}: expected {ICL_WIDTH * ICL_HEIGHT} depth values, got {values.size}"
        )
    depth = values.reshape(ICL_HEIGHT, ICL_WIDTH)
    if not np.all(np.isfinite(depth)):
        raise ValueError(f"{path}: depth contains non-finite values")
    return depth


def resize_rgb_depth(rgb: np.ndarray, depth_m: np.ndarray, nr: int, nc: int) -> Tuple[np.ndarray, np.ndarray]:
    rgb_r = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize((nc, nr), resample=Image.BILINEAR),
        dtype=np.uint8,
    )
    depth_r = np.asarray(
        Image.fromarray(depth_m, mode="F").resize((nc, nr), resample=Image.NEAREST),
        dtype=np.float32,
    )
    return rgb_r, depth_r


def rgb_to_gray01(rgb: np.ndarray) -> np.ndarray:
    """把 RGB 彩色图像转换成 0 到 1 之间的灰度图"""
    rgb_f = rgb.astype(np.float32) / 255.0
    gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
    return gray.astype(np.float32)


def z_depth_to_range(depth_z_m: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    depth_z_m是输入深度图，单位是米。每个像素存的是 z-depth
    做相机几何转换 z-depth → 每个像素射线方向上的真实距离
    在 SPAD / ToF 模拟里，这一步很重要，因为 SPAD 测量的是光传播时间，对应的是 真实传播距离 range，不是单纯的 z 方向深度
    """
    """Convert optical-axis z-depth to Euclidean range along each camera ray."""
    h, w = depth_z_m.shape
    # 给深度图中的每个像素分配一个图像坐标 (u, v)
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # 把像素坐标转换成归一化相机坐标,归一化之后，每个像素对应的相机射线方向可以写成[x_n, y_n, 1],其中 1 表示 z 方向
    x_n = (u_grid - np.float32(cx)) / np.float32(fx)
    y_n = (v_grid - np.float32(cy)) / np.float32(fy)
    # 计算每条射线方向 [x_n, y_n, 1] 的长度
    ray_norm = np.sqrt(1.0 + x_n * x_n + y_n * y_n).astype(np.float32)
    # 最终把 z-depth 乘以射线长度因子，得到真实距离
    return (depth_z_m.astype(np.float32) * ray_norm).astype(np.float32)


def build_camera_model_metadata(fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> str:
    """把相机内参和一些数据说明打包成一个 JSON 字符串，方便后面一起保存到 .npz 文件里"""
    camera_model = {
        "dataset": "ICL-NUIM",
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "depth_model": "z",
        "tof_model": "range",
        "image_y_axis": "up",
        "pose_convention": "T_wc columns are camera x-right, image-up, z-forward",
    }
    return json.dumps(camera_model, sort_keys=True)


def build_fog_model_metadata(args: argparse.Namespace, dmax: float) -> str:
    range_max = float(args.fog_range_max) if args.fog_range_max is not None else float(dmax)
    enabled = str(args.fog_model).lower() != "none"
    if args.fog_model == "ray_integral":
        smoke_return = "per_pixel_uniform_density_ray_integral"
        reference = "Per-pixel uniform-density ray integral smoke model"
    else:
        smoke_return = "none"
        reference = "none"

    fog_model = {
        "enabled": bool(enabled),
        "model": str(args.fog_model),
        "extinction_1_per_m": float(args.fog_extinction),
        "density": float(args.fog_density),
        "step_m": float(args.fog_step),
        "range_max_m": range_max,
        "surface_attenuation": "exp(-2*sigma*range)" if enabled else "none",
        "smoke_return": smoke_return,
        "reference": reference,
        "pile_up_model": "not simulated by current BaseEWHSPC fast_sim capture",
    }
    return json.dumps(fog_model, sort_keys=True)


def build_capture_model_metadata() -> str:
    capture_model = {
        "sensor": "SwissSPAD2",
        "n_pulses": int(N_PULSES),
        "tmax_ns": float(TMAX),
        "n_tbins": int(N_TBINS),
        "fwhm_ns": float(FWHM),
        "fast_sim": True,
        "sampling": "Poisson(phi_bar * n_pulses)",
    }
    return json.dumps(capture_model, sort_keys=True)


def generate_one_scene(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    nr: int = NR,
    nc: int = NC,
    device: str = "cpu",
    save_phi_bar: bool = False,
    save_components: bool = False,
    fog_model: str = "none",
    fog_extinction: float = 0.15,
    fog_density: float = 0.03,
    fog_step: float = 0.05,
    fog_range_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    dmax = 3e8 * TMAX * 1e-9 / 2
    cfg = GEN_FOG_CONFIG
    if cfg:
        save_components = bool(cfg.get("save_components", save_components))
        fog_model = str(cfg.get("fog_model", fog_model))
        fog_extinction = float(cfg.get("fog_extinction", fog_extinction))
        fog_density = float(cfg.get("fog_density", fog_density))
        fog_step = float(cfg.get("fog_step", fog_step))
        fog_range_max = cfg.get("fog_range_max", fog_range_max)

    rgb_r, depth_r = resize_rgb_depth(rgb, depth_m, nr=nr, nc=nc)
    depth_z = np.clip(depth_r, 0.1, dmax).astype(np.float32)
    # z-depth 转成真实 ToF range
    gt_range = np.clip(z_depth_to_range(depth_z, fx=fx, fy=fy, cx=cx, cy=cy), 0.1, dmax).astype(np.float32)
    # 把 RGB 图转成 0 到 1 的灰度图,这个灰度图在代码里被当作粗略的表面反射率
    albedo = rgb_to_gray01(rgb_r)

    # 转成 Torch tensor,gt_dist_t: 每个像素的真实 ToF range;albedo_t: 每个像素的反射率近似值
    gt_dist_t = torch.tensor(gt_range, device=device)
    albedo_t = torch.tensor(albedo, device=device)

    # 创建瞬态响应生成器
    tr_gen = TransientGenerator(Nr=nr, Nc=nc, N_tbins=N_TBINS, tmax=TMAX, FWHM=FWHM, device=device)
    # 生成目标表面回波的形状,可以理解成：对于每个像素，根据 gt_range 把一个激光脉冲移动到对应的时间 bin 上,nr × nc × N_TBINS
    r_t = tr_gen.get_shifted_laser_pulse_mesh(gt_dist_t)
    albedo_mean = torch.mean(albedo_t)
    if float(albedo_mean.detach().cpu()) <= 0.0:
        albedo_norm = albedo_t + 1.0
    else:
        albedo_norm = albedo_t / albedo_mean
    # 计算目标信号衰减
    signal_attn = tr_gen.get_signal_attenuation(albedo_norm, gt_dist_t)
    # 目标信号缩放系数,k_signal 控制每个像素目标回波的幅度
    k_signal = signal_attn * float(ALPHA_SIG * PDE) / torch.mean(signal_attn)
    # 无雾目标表面回波,表示没有雾衰减时，目标表面反射造成的期望光子时间分布
    phi_surface_clear = torch.multiply(r_t, k_signal)

    bkg_attn = albedo_norm.reshape(nr, nc, 1)
    # 每个像素、每个时间 bin 上的背景光期望光子数
    phi_background = bkg_attn * float(ALPHA_BKG * PDE) / float(N_TBINS)

    fog_model = str(fog_model).lower()
    if fog_model == "none":
        phi_surface = phi_surface_clear
        phi_smoke = torch.zeros_like(phi_surface_clear)
    # 有雾：先计算目标表面的双程衰减
    else:
        # 衰减：exp(-2σr)
        attenuation = torch.exp(-2.0 * float(fog_extinction) * gt_dist_t).reshape(nr, nc, 1)
        # 雾中目标表面回波 phi_surface
        phi_surface = phi_surface_clear * attenuation
        if fog_model == "ray_integral":
            # 确定烟雾积分的最远距离
            rmax = float(fog_range_max) if fog_range_max is not None else float(dmax)
            phi_smoke = make_ray_integral_smoke_phi_bar(
                nr,
                nc,
                range_limit_m=gt_dist_t,
                sigma=fog_extinction,
                density=fog_density,
                fog_step=fog_step,
                range_max=rmax,
                device=device,
            )
        else:
            raise ValueError(f"unsupported fog_model: {fog_model!r}")

    phi_dark = make_dark_phi_bar(nr, nc, device)
    # phi_bar = 目标表面 + 烟雾散射 + 背景光 + 暗计数,是“每个像素、每个时间 bin 期望有多少光子”的理论均值
    phi_bar = phi_surface + phi_smoke + phi_background + phi_dark

    # 创建 SPAD 模拟器
    spc = BaseEWHSPC(
        nr,
        nc,
        N_pulses=N_PULSES,
        device=device,
        N_tbins=N_TBINS,
        N_ewhbins=N_TBINS,
        fast_sim=True,
    )
    # 根据phi_bar期望值模拟 SPAD 的实际观测
    captured = spc.capture(phi_bar)
    spad_hist = captured["ewh"].cpu().numpy().astype(np.float32)
    phi_bar_np = phi_bar.cpu().numpy().astype(np.float32) if save_phi_bar else None

    global LAST_EXTRA_SAVE
    LAST_EXTRA_SAVE = {}
    if save_components:
        LAST_EXTRA_SAVE.update({
            "phi_surface_clear": phi_surface_clear.cpu().numpy().astype(np.float32),
            "phi_surface": phi_surface.cpu().numpy().astype(np.float32),
            "phi_smoke": phi_smoke.cpu().numpy().astype(np.float32),
            "phi_background": phi_background.cpu().numpy().astype(np.float32),
            "phi_dark": phi_dark.cpu().numpy().astype(np.float32),
        })

    return spad_hist, depth_z, gt_range, phi_bar_np


def scaled_icl_intrinsics(width: int, height: int) -> Tuple[float, float, float, float]:
    sx = float(width) / float(ICL_WIDTH)
    sy = float(height) / float(ICL_HEIGHT)
    return ICL_FX * sx, ICL_FY * sy, ICL_CX * sx, ICL_CY * sy


def discover_frames(root: Path, start: int, count: Optional[int]) -> List[str]:
    """输入 ICL-NUIM 数据集的目录，返回一个有序的有效帧名列表，保证列表中的每一个帧名都同时存在对应的.png、.depth和.txt三个文件"""
    names = sorted({p.stem for p in root.glob("*.png")})
    if not names:
        raise FileNotFoundError(f"no .png frames found in {root}")

    valid: List[str] = []
    for stem in names:
        missing = [ext for ext in (".png", ".depth", ".txt") if not (root / f"{stem}{ext}").is_file()]
        if missing:
            continue
        valid.append(stem)

    if start < 0 or start > len(valid):
        raise ValueError(f"invalid --start {start}, available frames: {len(valid)}")

    if count is None:
        return valid[start:]
    return valid[start : start + max(count, 0)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SwissSPAD2 dToF SPAD histograms from an ICL-NUIM trajectory"
    )
    parser.add_argument("--in-dir", required=True, help="Path to living_room_traj0_loop directory")
    parser.add_argument("--out", required=True, help="Output directory for .npz files and poses.txt")
    parser.add_argument("--n", type=int, default=None, help="Number of frames to process (default: all)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index in sorted order")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device: cpu or cuda",
    )
    # phi_bar是SPAD 在 capture 之前的期望时间直方图，phi_bar = phi_surface + phi_smoke + phi_background + phi_dark
    parser.add_argument("--save-phi-bar", action="store_true", help="Also save phi_bar in each .npz")
    # 保存 forward model 的各个组成部分,如果打开它，脚本会额外保存
    # phi_surface_clear : 没有雾衰减前的目标表面回波;phi_surface: 加了雾双程衰减后的目标表面回波
    # phi_smoke: 雾本身产生的散射回波;phi_background: 背景光;phi_dark: 探测器暗计数
    parser.add_argument("--save-components", action="store_true", help="Save forward-model phi components in each .npz")
    # 雾模型开关。none:不加雾;ray_integral:沿射线积分模拟烟雾散射
    parser.add_argument(
        "--fog-model",
        choices=("none", "ray_integral"),
        default="none",
        help="Fog/smoke forward model",
    )
    # 雾的消光系数，记作 sigma 或 σ，单位是：1/m,它控制目标表面回波被雾削弱的程度
    parser.add_argument("--fog-extinction", type=float, default=0.15, help="Fog extinction sigma in 1/m")
    # 均匀烟雾的后向散射密度,fog_density 越大，烟雾自身产生的散射回波 phi_smoke 越强
    parser.add_argument("--fog-density", type=float, default=0.03, help="Uniform smoke backscatter density")
    parser.add_argument("--fog-step", type=float, default=0.05, help="Ray-integral fog sampling step in meters")
    parser.add_argument("--fog-range-max", type=float, default=None, help="Ray-integral fog max range in meters")
    parser.add_argument("--nr", type=int, default=NR, help="Output image height")
    parser.add_argument("--nc", type=int, default=NC, help="Output image width")
    parser.add_argument("--poses-name", default="poses.txt", help="Output poses filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global FOG_ENABLED
    FOG_ENABLED = args.fog_model != "none"
    global GEN_FOG_CONFIG
    GEN_FOG_CONFIG = {
        "save_components": bool(args.save_components),
        "fog_model": args.fog_model,
        "fog_extinction": float(args.fog_extinction),
        "fog_density": float(args.fog_density),
        "fog_step": float(args.fog_step),
        "fog_range_max": args.fog_range_max,
    }

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    if not in_dir.is_dir():
        raise NotADirectoryError(in_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 获得帧名列表
    frame_keys = discover_frames(in_dir, start=int(args.start), count=args.n)
    if not frame_keys:
        raise FileNotFoundError("no valid frame triplets found for the requested range")

    fx, fy, cx, cy = scaled_icl_intrinsics(width=int(args.nc), height=int(args.nr))
    print(
        f"Sensor : SwissSPAD2 (PDE={PDE}, tmax={TMAX}ns, FWHM={FWHM}ns, "
        f"N_pulses={N_PULSES}, N_tbins={N_TBINS})"
    )
    print(f"Input  : {in_dir} ({len(frame_keys)} frames)")
    print(f"Output : {args.nr}x{args.nc} pixels -> {out_dir}")
    print(f"ICL intrinsics scaled : fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    print(
        f"Fog    : model={args.fog_model}, extinction={args.fog_extinction}, "
        f"density={args.fog_density}, step={args.fog_step}"
    )

    poses: List[Tuple[str, np.ndarray]] = []
    for idx, frame_key in enumerate(frame_keys, start=1):
        print(f"  [{idx}/{len(frame_keys)}] {frame_key}", end="", flush=True)

        rgb = load_png_rgb(in_dir / f"{frame_key}.png")
        depth = load_depth_txt(in_dir / f"{frame_key}.depth")
        meta = load_camera_metadata(in_dir / f"{frame_key}.txt")
        T_wc = metadata_to_t_wc(meta)

        # 模拟得到的 SPAD 时间直方图、缩放并裁剪后的 z-depth 真值、由 z-depth 转换得到的真实 ToF range、理论瞬态响应
        spad_hist, gt_depth_z, gt_range, phi_bar = generate_one_scene(
            rgb,
            depth,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            nr=int(args.nr),
            nc=int(args.nc),
            device=args.device,
            save_phi_bar=bool(args.save_phi_bar),
        )

        camera_model = build_camera_model_metadata(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=int(args.nc),
            height=int(args.nr),
        )
        save_dict = {
            "spad_hist": spad_hist,
            "gt_depth_z": gt_depth_z,
            "gt_range": gt_range,
            "gt_depth": gt_depth_z,
            "camera_model": np.asarray(camera_model),
            "fog_model": np.asarray(build_fog_model_metadata(args, 3e8 * TMAX * 1e-9 / 2)),
            "capture_model": np.asarray(build_capture_model_metadata()),
        }
        if phi_bar is not None:
            save_dict["phi_bar"] = phi_bar
        save_dict.update(LAST_EXTRA_SAVE)
        np.savez_compressed(out_dir / f"{frame_key}.npz", **save_dict)
        poses.append((frame_key, T_wc))
        print(
            f" -> saved ({spad_hist.shape}, z {gt_depth_z.min():.2f}~{gt_depth_z.max():.2f} m, "
            f"range {gt_range.min():.2f}~{gt_range.max():.2f} m)"
        )

    poses_path = out_dir / args.poses_name
    write_poses_txt(poses_path, poses)
    print(f"Poses  : {poses_path}")
    print("Done.")


if __name__ == "__main__":
    main()
