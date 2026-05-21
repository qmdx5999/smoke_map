# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:26:28 2026

@author: wenzt
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ===================== 配置区 =====================

TXT_PATH = 'C:\\document\\code\\singlephotongs\\datanew\\nets_box\\sp\\RawDataHistogramMap_frame_0_1770463592301086.txt'

# bin时间分辨率
DT_PS = 750#297.0
C = 299_792_458.0  # m/s

# 输出目录
OUT_DIR = "output_from_txt"
os.makedirs(OUT_DIR, exist_ok=True)

# 自动推断/截断成常见尺寸
COMMON_SIZES = [(192, 256), (256, 192)]
TARGET_PIXELS = 192 * 256  # 49152

# 可选：去畸变（注意：深度不做 undistort）
UNDISTORT_INTENSITY = True

# 可选：深度含义
# True  = depth是range（沿像素射线的距离），点云用“单位射线*range”
# False = depth是Z（沿相机光轴的深度），点云用针孔/射线换算（不推荐ToF这样用）
DEPTH_IS_RANGE = True

# 相机内参（你给的）
fx, fy = 118.6514575329715, 118.7964934010577
cx, cy = 130.6802784645003, 100.3605702468140

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

# 畸变系数 [k1, k2, p1, p2]
k1, k2 = -0.257910069121181, 0.053237073644331
p1, p2 = 0.0, 0.0
D = np.array([k1, k2, p1, p2], dtype=np.float64)

# 有效点过滤（按需调）
MIN_RANGE_M = 0.5
MAX_RANGE_M = 20.0
INTENSITY_MIN = 10  # 强度小于该值的点丢掉（去噪）

# ===================== 读txt并生成强度/深度(1D) =====================

print("读取txt...")
data = np.loadtxt(TXT_PATH, dtype=np.int64)  # shape: (positions, bins)
num_pos, num_bins = data.shape
print(f"data shape = {data.shape} (positions x bins)")

# 强度：每行最大值
intensity_1d = data.max(axis=1).astype(np.float32)

# 深度：峰值bin -> 时间 -> 距离(米)
peak_bin_1d = data.argmax(axis=1).astype(np.int32)

dt = DT_PS * 1e-12
bin_to_m = C * dt / 2.0
depth_m_1d = ( peak_bin_1d.astype(np.float32) - 25 ) * bin_to_m

print(f"intensity range = {intensity_1d.min()} ~ {intensity_1d.max()}")
print(f"peak bin range  = {peak_bin_1d.min()} ~ {peak_bin_1d.max()}")
print(f"depth range (m) = {depth_m_1d.min():.4f} ~ {depth_m_1d.max():.4f}")
print(f"bin_to_m = {bin_to_m*1e3:.6f} mm/bin")

# ===================== reshape成2D =====================

height = width = None
for h, w in COMMON_SIZES:
    if h * w == num_pos:
        height, width = h, w
        break

if height is None:
    if num_pos >= TARGET_PIXELS:
        print(f"位置数 {num_pos} >= {TARGET_PIXELS}，截断到 {TARGET_PIXELS} 以适配 192x256")
        intensity_1d = intensity_1d[:TARGET_PIXELS]
        depth_m_1d = depth_m_1d[:TARGET_PIXELS]
        num_pos = TARGET_PIXELS
        height, width = 192, 256
    else:
        # 找因子最接近正方形
        h = int(np.sqrt(num_pos))
        while num_pos % h != 0:
            h -= 1
        height, width = h, num_pos // h

print(f"image shape = {height} x {width}")

intensity = intensity_1d.reshape(height, width)
depth_m = depth_m_1d.reshape(height, width)

# ===================== 可选：仅对强度去畸变（深度不去畸变） =====================

if UNDISTORT_INTENSITY:
    print("对强度图执行去畸变（深度不做undistort）...")
    intensity_u = cv2.undistort(intensity, K, D)
else:
    intensity_u = intensity

depth_m_u = depth_m  # 深度保持原样（不重采样）

# 保存可视化图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(intensity_u, origin="upper")
plt.title("Intensity (max count)")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(depth_m_u, origin="upper")
plt.title("Depth (m) from peak bin (NOT undistorted)")
plt.colorbar()
plt.tight_layout()
viz_path = os.path.join(OUT_DIR, "intensity_depth_maps.png")
plt.savefig(viz_path, dpi=150)
print("save:", viz_path)

# ===================== 深度+强度 -> 点云（用 undistortPoints 修正射线） =====================

H, W = depth_m_u.shape

# 构建像素网格
u, v = np.meshgrid(np.arange(W), np.arange(H))

# 有效mask：深度、强度阈值
valid = (
    np.isfinite(depth_m_u) &
    (depth_m_u > MIN_RANGE_M) &
    (depth_m_u < MAX_RANGE_M) &
    (intensity_u >= INTENSITY_MIN)
)

u_valid = u[valid].astype(np.float32)
v_valid = v[valid].astype(np.float32)
d_valid = depth_m_u[valid].astype(np.float32)
I_valid = intensity_u[valid].astype(np.float32)

print(f"valid points: {d_valid.size} / {H*W}")

# 强度归一化到 0~255（用于PLY灰度显示）
if I_valid.size > 0:
    I_min, I_max = float(I_valid.min()), float(I_valid.max())
    if I_max > I_min:
        I_255 = ((I_valid - I_min) / (I_max - I_min) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        I_255 = np.zeros_like(I_valid, dtype=np.uint8)
else:
    I_255 = np.array([], dtype=np.uint8)

# ---------- 用 undistortPoints 得到去畸变的归一化坐标 (x_n, y_n) ----------
# OpenCV 需要形状 (N,1,2)
uv = np.stack([u_valid, v_valid], axis=1).reshape(-1, 1, 2).astype(np.float32)

# 输出为归一化坐标：已经等效做了除以 fx/fy 并考虑畸变
xy = cv2.undistortPoints(uv, K, D)  # shape: (N,1,2)

x_n = xy[:, 0, 0]
y_n = xy[:, 0, 1]

# 射线方向并单位化
ray = np.stack([x_n, y_n, np.ones_like(x_n)], axis=1)  # (N,3)
ray_norm = np.linalg.norm(ray, axis=1, keepdims=True)
ray_unit = ray / np.maximum(ray_norm, 1e-12)

# ---------- 根据深度含义生成 3D 点 ----------
if DEPTH_IS_RANGE:
    # range：沿射线距离
    pts = ray_unit * d_valid.reshape(-1, 1)
else:
    # 若 depth 表示相机光轴 Z（一般 ToF 不推荐）
    z = d_valid
    pts = ray_unit * (z.reshape(-1, 1) / np.maximum(ray_unit[:, 2:3], 1e-12))

# ===================== 导出PLY（ASCII，灰度强度=RGB） =====================

ply_path = os.path.join(OUT_DIR, "intensity_pointcloud.ply")
with open(ply_path, "w") as f:
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
    for i in range(pts.shape[0]):
        x, y, z = pts[i]
        ii = int(I_255[i]) if I_255.size == pts.shape[0] else 0
        f.write(f"{x} {y} {z} {ii} {ii} {ii}\n")

print("save:", ply_path)
print("done.")

# ===================== 点云可视化（Open3D） =====================

import open3d as o3d
import matplotlib.cm as cm

print("Open3D 彩色强度可视化...")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

# 强度归一化到 0~1 映射 colormap
if I_valid.size > 0:
    I = I_valid.astype(np.float32)
    I_norm = (I - I.min()) / (I.max() - I.min() + 1e-12)
    cmap = cm.get_cmap("turbo")
    colors = cmap(I_norm)[:, :3]  # RGBA -> RGB
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

o3d.visualization.draw_geometries(
    [pcd, axis],
    window_name="Intensity Point Cloud (Color)",
    width=1200,
    height=800
)
