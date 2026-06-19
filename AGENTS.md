# 回答与操作偏好

后续回答中，如果必须使用英文术语、英文变量名或英文论文/代码概念，应尽量在英文后面跟中文括号释义，例如 `posterior（后验概率）`、`occupancy grid（占据栅格）`、`profile likelihood（剖面似然）`。代码原文、命令、文件名、函数名和不可翻译的标识符可以保持原样。

禁止批量删除文件或目录。不要使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。如果需要删除文件，只能一次删除一个明确路径的文件；如果需要批量删除，应停止操作并让用户手动处理。

# 当前项目状态

更新时间：2026-06-19。

当前目录 `D:\pythonProgram\smoke_map` 是一个围绕 Single-Photon LiDAR / SPAD / TCSPC photon histogram（光子到达直方图）开展三维占据建图和雾中单光子成像研究的原型项目。

占据建图主线的核心目标是避免先把 histogram（直方图）压缩成单一 depth（深度）或 point cloud（点云），而是把 histogram-level uncertainty（直方图级不确定性）直接传递到 occupancy mapping（占据建图）：

```text
raw photon histogram
  -> Poisson/profile likelihood range inference
  -> range posterior / multi-peak hypotheses
  -> CDF-marginalized occupancy update
  -> log-odds occupancy map
```

当前已经实现：

- 从 ICL-NUIM 连续 RGB-D 序列生成 SPAD histogram（光子直方图）。
- 保存并读取多帧 `T_wc` camera-to-world（相机到世界）位姿。
- per-pixel ray-integral smoke/fog forward model（逐像素射线积分烟雾/雾前向模型）。
- accumulated-count smoke likelihood（累计计数域烟雾似然）。
- smoke-aware local range posterior（烟雾感知局部距离后验）。
- lightweight likelihood-ratio surface evidence test（轻量似然比表面证据检验）。
- CDF-marginalized occupancy update（CDF 边缘化占据更新）。
- surface point cloud（表面点云）、occupied voxel cloud（占据体素点云）和 dense log-odds grid（稠密对数几率栅格）输出。
- visible surface benchmark（可见表面基准评测）。
- Satat et al. 2018 Gamma--Gaussian（伽马--高斯）单帧雾中成像复现。

当前 occupancy inverse sensor model（占据逆传感器模型）的语义是：

```text
表面前方：free（自由）
返回附近：occupied（占据）
表面后方：occluded/unknown（遮挡未知）
```

因此当前输出应表述为 voxel-grid probabilistic occupancy（体素栅格概率占据）和 surface-dominant occupied evidence（以表面为主的占据证据）。不要描述为 watertight reconstruction（封闭重建）、solid reconstruction（实体重建）或 dense filled occupancy（密实体素占据）。

# 当前主要文件与目录

```text
.
├── AGENTS.md
├── spad_npz_occupancy_mapping.py
├── generate_spad_data_icl.py
├── generate_spad_data.py
├── diagnose_smoke_likelihood.py
├── evaluate_surface_against_gt.py
├── analyze_occupancy_grid.py
├── export_occupancy_from_grid.py
├── spad_npz_to_ply.py
├── satat_original_fog_imaging.py
├── plot_pixel_histogram.py
├── view_npz.py
├── visualize_npz2pointcloud.py
├── living_room_traj0_loop/
├── out/
├── SPCSim/
├── mapping/
│   ├── bare_jrnl_new_sample4.tex
│   ├── report.bib
│   └── IEEEtran.cls
├── papers/
│   └── Satat 等 - 2018 - Towards photography through realistic fog.pdf
├── satat_out/
│   └── scene_00_0000/
├── max5k.npz
├── max5k.png
├── max5k_occ.ply
├── max5k_occ_scalar.ply
├── max5k_surface.ply
├── smoke_100f_unfiltered.npz
├── smoke_100f_unfiltered.png
├── smoke_100f_unfiltered_occ.ply
├── smoke_100f_unfiltered_surface.ply
├── smoke_100f_filtered.npz
├── smoke_100f_filtered.png
├── smoke_100f_filtered_occ.ply
├── smoke_100f_filtered_surface.ply
├── smoke_likelihood_diagnostic.png
└── Physics_Aware_Bayesian_Semantic_Geometric_Mapping_for_Single_Photon_LiDAR.pdf
```

# 论文文件

主稿是 `mapping/bare_jrnl_new_sample4.tex`，题目：

```text
Photon-Histogram to Occupancy Map:
Uncertainty-Aware Mapping for Single-Photon LiDAR
```

主稿已写入 histogram-level inference（直方图级推断）、multi-peak hypotheses（多峰假设）、CDF occupancy update（CDF 占据更新）和 log-odds mapping（对数几率建图）的方法主线。

当前仍未完成的章节包括：

- `BENCHMARK RESULTS`
- `Datasets`
- `Evaluation Setup`
- `Results`
- `REAL-WORLD APPLICATIONS`
- `Hardware System Setup`
- `conclusion`

`mapping/report.bib` 是参考文献库。主稿中仍有空引用 `\cite{}`，写论文时必须补充真实引用或删除空引用。

代码侧已有 visible surface benchmark（可见表面基准评测），但尚未回填到论文结果章节。真实硬件实验、CRLB / Fisher information（克拉美罗下界 / 费舍尔信息）和 measured IRF（实测系统响应）尚未完成，不能写成已完成结果。

# ICL-NUIM 数据与 NPZ 格式

`living_room_traj0_loop/` 是 ICL-NUIM `Living Room / traj0 loop` 连续室内序列。当前包含 1510 帧 RGB、depth（深度）和 camera metadata（相机元数据）。

当前 `out/` 包含 1510 个 `.npz` 和 `poses.txt`。这些数据带有 `camera_model` 和 `fog_model`，但没有 `capture_model`；使用 smoke likelihood（烟雾似然）建图时需要显式传入：

```text
--n-pulses 5000
```

每帧原始 ICL 文件包括：

```text
scene_00_xxxx.png    RGB 图像，640 x 480
scene_00_xxxx.depth  文本 z-depth（光轴深度），640 x 480
scene_00_xxxx.txt    cam_pos / cam_dir / cam_up 等相机信息
```

当前 SPAD `.npz` 主要字段：

```text
spad_hist    : (256, 256, 1000), float32
gt_depth_z   : (256, 256), float32  # ICL z-depth（光轴深度）
gt_range     : (256, 256), float32  # 沿像素射线的 ToF 欧氏距离
gt_depth     : (256, 256), float32  # 兼容字段，等于 gt_depth_z
camera_model : JSON metadata string
fog_model    : JSON metadata string
```

重新使用当前 `generate_spad_data_icl.py` 生成的数据还会包含：

```text
capture_model: JSON metadata string
```

其内容至少包括：

```json
{
  "sensor": "SwissSPAD2",
  "n_pulses": 5000,
  "tmax_ns": 100.0,
  "n_tbins": 1000,
  "fwhm_ns": 0.5,
  "fast_sim": true,
  "sampling": "Poisson(phi_bar * n_pulses)"
}
```

开启 `--save-phi-bar` 时还会保存：

```text
phi_bar : (256, 256, 1000), float32
```

额外开启 `--save-components` 时会保存调试用 forward-model components（前向模型分量）：

```text
phi_surface_clear
phi_surface
phi_smoke
phi_background
phi_dark
```

这些分量只用于 simulation diagnosis（仿真诊断），不能作为正式 occupancy mapping（占据建图）或 Satat 成像推断输入，否则会泄漏仿真真值。

## ICL 内参与缩放

ICL 原始分辨率和内参：

```text
width  = 640
height = 480
fx = 481.20
fy = 480.00
cx = 319.50
cy = 239.50
```

当前 SPAD histogram（光子直方图）分辨率是 `256 x 256`，因此内参必须随 resize（缩放）同步变化：

```text
fx = 481.20 * (256 / 640) = 192.48
fy = 480.00 * (256 / 480) = 256.0
cx = 319.50 * (256 / 640) = 127.8
cy = 239.50 * (256 / 480) = 127.733333...
```

横向参数 `fx/cx` 按宽度比例 `256/640` 缩放，纵向参数 `fy/cy` 按高度比例 `256/480` 缩放。两者比例不同是因为原始 `640x480` 图像被缩放成正方形 `256x256`。

这些 camera intrinsics（相机内参）用于像素到 ray（射线）的反投影：

```text
x_n = (u - cx) / fx
y_n = (v - cy) / fy
ray ≈ [x_n, y_n, 1]
```

内参必须与生成 `.npz` 时的输出分辨率一致，否则多帧建图会出现空间拉伸、压缩或错位。后续刷新 `AGENTS.md` 时，必须同时保留缩放公式、缩放原因和像素到射线反投影的用途说明。

当前 `camera_model` 示例：

```json
{
  "dataset": "ICL-NUIM",
  "width": 256,
  "height": 256,
  "fx": 192.48,
  "fy": 256.0,
  "cx": 127.8,
  "cy": 127.73333333333333,
  "depth_model": "z",
  "tof_model": "range",
  "image_y_axis": "up",
  "pose_convention": "T_wc columns are camera x-right, image-up, z-forward"
}
```

`gt_depth_z` 和 `gt_range` 不能混用：

- `gt_depth_z` 是沿相机光轴的 z-depth（光轴深度）。
- `gt_range` 是沿像素 ray（射线）的真实传播距离。
- SPAD transient（瞬态）按 `gt_range` 生成。
- 建图和 Satat 深度结果都应与 `gt_range` 比较。

# ICL 序列转 SPAD

`generate_spad_data_icl.py` 将 ICL RGB-D 序列转换为 SPAD `.npz` 序列，并生成 `poses.txt`。

主要流程：

- RGB 图像提供 albedo/intensity proxy（反照率/强度近似）。
- `.depth` 提供 z-depth（光轴深度）。
- 使用内参转换：

```text
gt_range = z * sqrt(1 + x_n^2 + y_n^2)
```

- 用 `gt_range` 生成表面 transient（瞬态）。
- 构建表面、烟雾、背景光和暗计数组件。
- 通过 `Poisson(phi_bar * n_pulses)` 生成累计 histogram（直方图）。
- 从 `cam_pos / cam_dir / cam_up` 构建 `T_wc`。

当前传感器参数：

```text
TMAX = 100 ns
N_TBINS = 1000
N_PULSES = 5000
FWHM = 0.5 ns
PDE = 0.03675
ALPHA_SIG = 0.5
ALPHA_BKG = 1.0
DCR_CPS = 100
```

当前 fog model（雾模型）支持：

```text
--fog-model none
--fog-model ray_integral
```

`ray_integral` 是 per-pixel uniform-density ray integral（逐像素均匀密度射线积分）模型。每个像素的烟雾积分上限是：

```text
min(gt_range[u,v], fog_range_max, dmax)
```

表面返回使用双程衰减：

```text
phi_surface = phi_surface_clear * exp(-2 * fog_extinction * gt_range)
```

烟雾散射权重包含：

```text
density * exp(-2 * extinction * s) * ds
```

当前 `BaseEWHSPC(fast_sim=True)` 不模拟 first-photon pile-up（首光子堆积）或 detector dead time（探测器死时间）。

生成完整序列：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\generate_spad_data_icl.py --in-dir .\living_room_traj0_loop --out out
```

生成 ray-integral fog（射线积分雾）数据：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\generate_spad_data_icl.py --in-dir .\living_room_traj0_loop --out out_fog_ray --fog-model ray_integral --fog-extinction 0.15 --fog-density 0.03 --fog-step 0.05
```

小规模调试：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\generate_spad_data_icl.py --in-dir .\living_room_traj0_loop --out out_fog_ray_debug --n 10 --fog-model ray_integral --fog-extinction 0.15 --fog-density 0.03 --fog-step 0.05 --save-phi-bar --save-components
```

# 位姿文件 `poses.txt`

每行格式：

```text
frame_key m00 m01 m02 m03 m10 m11 m12 m13 m20 m21 m22 m23 m30 m31 m32 m33
```

约定：

- `frame_key` 等于 `.npz` 文件 stem（主文件名）。
- 矩阵是 camera-to-world（相机到世界）变换 `T_wc`。
- 矩阵按行展开。
- `R_wc` 三列分别是 camera x-right（图像右）、image-up（图像上）、z-forward（相机前方）。

构造方式：

```text
z_forward = normalize(cam_dir)
x_right   = normalize(cross(cam_up, z_forward))
y_axis    = normalize(cross(z_forward, x_right))
R_wc      = column_stack(x_right, y_axis, z_forward)
t_wc      = cam_pos
```

第一帧：

```text
scene_00_0000 -0.999762369 0 -0.021799208 1.3705 0 1 0 1.51739 0.021799208 0 -0.999762369 1.44963 0 0 0 1
```

`image_y_axis=up` 是多帧清晰配准的重要约定。普通像素公式 `y_n=(v-cy)/fy` 是 image-down（图像向下），建图时需要根据 metadata（元数据）翻转：

```text
y_n_all = -y_n_all
```

# 核心建图脚本

`spad_npz_occupancy_mapping.py` 支持单帧和多帧 histogram-level occupancy mapping（直方图级占据建图）。

输入要求：

- 每个 `.npz` 必须包含 `(H,W,T)` 的 `spad_hist`。
- 多帧模式使用 `--npz-dir` 和 `--poses`。
- ICL 数据推荐 `--range_model range` 和 `--image-y-axis auto`。

主要实现：

| 概念 | 代码 |
|---|---|
| Poisson likelihood（泊松似然） | `poisson_ll_numba()` |
| clear profile likelihood（无雾剖面似然） | `profile_ll_one_r_numba()` |
| smoke profile likelihood（烟雾剖面似然） | `fit_profile_one_r_smoke_count_numba()` |
| Gaussian IRF（高斯系统响应） | `build_S_gaussian()` |
| smoke integral template（烟雾积分模板） | `build_smoke_integral_templates()` |
| peak detection（峰检测） | `mp_find_peaks()` |
| clear range posterior（无雾距离后验） | `compute_ll_grid_numba()` |
| smoke range posterior（烟雾距离后验） | `compute_ll_grid_smoke_numba()` |
| smoke posterior details（烟雾后验拟合细节） | `compute_ll_grid_smoke_details_numba()` |
| smoke-only H0（仅烟雾假设） | `fit_smoke_background_only_numba()` |
| H0 evaluation（H0 评估） | `evaluate_smoke_peak_h0_numba()` |
| posterior CDF（后验累积分布） | `cdf_lookup_numba()` |
| entropy weight（熵权重） | `posterior_entropy_weight()` |
| CDF occupancy update（CDF 占据更新） | `dda_update_dense_cdf()` |
| frame processing（帧处理） | `process_frame()` |

当前主要默认值：

```text
--max-frames 0
--max_rays 0
--peak_thr 5.0
--voxel 0.10
--range_max 7.0
--z_min -7.0
--z_max 7.0
--Wr_bin 12
--M 81
--p_occ 0.65
--p_free 0.45
--update-scale 0.01
--ray-norm-target 10000
--p0 0.50
--Lmin -10.0
--Lmax 10.0
--tau 1.0
--win_half 25
--sigma_bins 2.0
--likelihood-model auto
--n-pulses None
--smoke-peak-filter off
--surface-dll-min 10.0
--surface-alpha-min 0.5
--print-peak-filter-stats off
--peak-filter-details-csv None
--range_model range
--max_peaks 3
--mp_thr 2.0
--mp_support 0
--export-min-prob 0.51
--image-y-axis auto
--diagnostic-checkpoints all
```

`--max-frames 0` 和 `--max_rays 0` 表示不限制。`--mp_support 0` 表示自动使用 `ceil(4*sigma_bins)`。

单帧默认输入 `scene_0000.npz` 不在当前根目录，实际运行应显式指定 `--npz`，或使用多帧模式。

## clear 与 smoke likelihood

clear model（无雾模型）：

```text
lambda_b(r) = alpha * S_b(r) + beta
```

smoke model（烟雾模型）：

```text
lambda_b(r) = alpha * S_b(r)
            + N_pulses * gamma * G_b(r; kappa)
            + beta
```

其中：

- `S_b(r)` 是候选距离的 Gaussian surface template（高斯表面模板）。
- `G_b(r;kappa)` 是 unit-density ray-integral smoke template（单位密度射线积分烟雾模板）。
- `kappa` 来自 `fog_model.extinction_1_per_m`。
- `gamma` 来自 `fog_model.density`。
- `N_pulses` 来自 `--n-pulses` 或 `capture_model.n_pulses`。
- `alpha` 是包含表面衰减的 effective surface count amplitude（有效表面计数幅度）。
- 自由 `alpha` 不能单独分解出无雾反射率和消光衰减。

`--likelihood-model`：

- `auto`：检测到启用的 `ray_integral fog_model` 时使用 smoke，否则使用 clear。
- `clear`：强制使用表面加常数背景模型。
- `smoke`：强制使用烟雾模型；缺少有效 `fog_model` 或脉冲数时直接报错。

## lightweight surface peak filter

在每个 peak hypothesis（峰假设）的 MAP range（最大后验距离）处比较：

```text
H1: lambda = alpha * S(r_hat) + smoke(r_hat) + beta
H0: lambda = smoke(r_hat) + beta0
delta_ll = LL(H1) - LL(H0)
```

接受条件：

```text
delta_ll >= surface_dll_min
alpha >= surface_alpha_min
```

只有接受的峰才进入 posterior merge（后验合并）、surface cloud（表面点云）和 occupancy fusion（占据融合）。

该方法是 surface/non-surface（表面/非表面）轻量似然比检验，不是完整的 surface/smoke/background（表面/烟雾/背景）潜变量分类。

推荐强烟雾参数：

```text
--mp_thr 2
--max_peaks 3
--smoke-peak-filter
--surface-dll-min 10
--surface-alpha-min 0.5
```

`--peak-filter-details-csv` 可输出逐峰：

```text
peak_bin
r_hat
alpha
LL(H0)
LL(H1)
delta_ll
gt_range
abs_range_error
```

`gt_range` 只用于诊断，不参与推断和过滤。

# 推荐建图命令

当前 `out/` 没有 `capture_model`，使用 smoke likelihood 时必须传入 `--n-pulses 5000`。

100 帧过滤实验：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\spad_npz_occupancy_mapping.py --npz-dir out --poses out/poses.txt --max-frames 100 --max_rays 2000 --range_model range --image-y-axis auto --likelihood-model smoke --n-pulses 5000 --mp_thr 2 --max_peaks 3 --smoke-peak-filter --surface-dll-min 10 --surface-alpha-min 0.5 --print-peak-filter-stats --update-scale 0.005 --surface-out smoke_100f_filtered_surface.ply --ply-out smoke_100f_filtered_occ.ply --grid-out smoke_100f_filtered.npz --profile
```

未过滤对照使用相同参数，但移除：

```text
--smoke-peak-filter
--print-peak-filter-stats
```

全序列建图：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\spad_npz_occupancy_mapping.py --npz-dir out --poses out/poses.txt --fx 192.48 --fy 256.0 --cx 127.8 --cy 127.7333333 --range_model range --range_max 7.0 --image-y-axis auto --max_rays 10000 --update-scale 0.005 --ply-out full_occ.ply --surface-out full_surface.ply --grid-out full.npz --profile
```

`--grid-out` 很重要。保存 grid（栅格）后可以按不同阈值重新导出 PLY，无需重跑 histogram mapping（直方图建图）。

# 当前烟雾峰过滤实验

当前数据烟雾参数：

```text
kappa = 0.15 1/m
gamma = 0.03
```

100 帧实验共同参数：

```text
--max-frames 100
--max_rays 2000
--mp_thr 2
--max_peaks 3
--update-scale 0.005
```

未过滤：

```text
surface points: 576,461
active voxels: 40,569
occupied voxels @0.51: 460
```

过滤后：

```text
total peak proposals: 576,461
accepted surface peaks: 158,017
rejected peaks: 418,444
surface points: 158,017
active voxels: 40,601
occupied voxels @0.51: 1,788
```

过滤后表面点减少约 `72.6%`，主要删除烟雾产生的虚假片层。occupied voxels（占据体素）增加不等于虚假占据增加：过滤后 posterior mass（后验质量）更集中在真实表面，使表面附近占据证据增强。

## GT surface evaluation

`evaluate_surface_against_gt.py` 使用 `gt_range + camera_model + T_wc` 构建世界坐标 GT surface cloud（真值表面点云），并对预测表面点云计算双向 nearest-neighbor distance（最近邻距离）。

评测命令：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\evaluate_surface_against_gt.py --npz-dir out --poses out/poses.txt --max-frames 100 --max-rays 2000 --range-max 7.0 --surface smoke_100f_unfiltered_surface.ply smoke_100f_filtered_surface.ply --out-csv smoke_100f_surface_metrics.csv
```

结果：

```text
                                  unfiltered       filtered
prediction -> GT median          1.5605 m         0.00834 m
prediction -> GT P95             2.2019 m         0.02024 m
precision @5 cm                  27.33%           99.45%
phantom surface rate >15 cm      72.61%           0.53%
recall @5 cm                     90.10%           90.07%
F1 @5 cm                         41.94%           94.53%
```

该结果支持将当前过滤器称为 lightweight likelihood-ratio surface evidence test（轻量似然比表面证据检验）。不要称为完整 Bayesian latent classification（贝叶斯潜变量分类）。

该评测针对 visible surface reconstruction（可见表面重建），不是 watertight volume occupancy（封闭体积占据）或实体体积 IoU。

# Satat 2018 雾中成像复现

论文：

```text
papers/Satat 等 - 2018 - Towards photography through realistic fog.pdf
```

复现脚本：

```text
satat_original_fog_imaging.py
```

当前流程：

```text
spad_hist
  -> Gaussian KDE（高斯核密度估计）
  -> weighted Gamma MLE fog fit（加权伽马最大似然雾拟合）
  -> non-negative fog subtraction（非负雾响应相减）
  -> local Gaussian surface fit（局部高斯表面拟合）
  -> range + reflectance（距离图与反射率图）
```

实现细节：

- KDE bandwidth（核带宽）默认 `0.08 ns`，对应论文的 `80 ps`。
- Gamma 参数使用 histogram bin count（直方图时间箱计数）作为权重进行 MLE（最大似然估计）。
- 雾响应相减后的负值截断为零。
- 在主残差峰周围 `±8 bins` 做 Gaussian（高斯）局部拟合。
- Gaussian 均值转换为 ToF range（飞行时间距离）。
- Gaussian 拟合峰值作为 reflectance（反射率）。
- `confidence = reflectance * range`。
- 默认 `--confidence-relative 0.10`。
- 黑色 depth pixels（深度像素）是未通过置信度阈值的 invalid pixels（无效像素），不是算法保留的雾深度。

脚本推断只使用 `spad_hist`。当前脚本要求输入同时包含 `gt_range`，但 `gt_range` 只用于生成真值图和评测指标，不参与 KDE、Gamma 或 Gaussian 拟合。

输出路径：

```text
<out-dir>/<npz-stem>/
```

例如：

```text
satat_out/scene_00_0000/
```

输出文件：

```text
depth.png
gt_range.png
reflectance.png
valid_mask.png
result.npz
metrics.csv
```

`depth.png` 和 `gt_range.png` 使用相同的 `0～range_max` 色标。Satat 输出是 ray range（射线距离），正确真值是 `gt_range`，不是 `gt_depth_z`。

运行命令：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\satat_original_fog_imaging.py --npz .\out\scene_00_0000.npz --out-dir .\satat_out --kde-bandwidth-ns 0.08 --confidence-relative 0.10
```

当前 `scene_00_0000` 结果：

```text
valid pixel rate : 87.21%
MAE              : 0.01268 m
RMSE             : 0.04457 m
accuracy @5 cm   : 99.67%
accuracy @15 cm  : 99.85%
```

这些误差只在 valid pixels（有效像素）上计算，因此必须与有效率一起报告。

该实现是论文分阶段 Gamma--Gaussian 模型的工程复现，但当前输入是累计 Poisson histogram（泊松直方图），且仿真不含 first-photon pile-up（首光子堆积）。不要把它表述为论文硬件采样过程的完全复现。

Satat 方法当前定位为单帧 fog imaging baseline（雾中成像基线），尚未接入多帧 occupancy mapping（占据建图）。

# 网格分析与导出

`analyze_occupancy_grid.py`：

- 读取 `--grid-out` 生成的 `.npz`。
- 将 `Lgrid` 转成 occupancy probability（占据概率）。
- 输出全部体素和 active voxels（活跃体素）的统计和直方图。

示例：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\analyze_occupancy_grid.py --grid .\max5k.npz --hist-out max5k.png
```

`export_occupancy_from_grid.py`：

- 读取 `Lgrid/mins/voxel`。
- 使用：

```text
p = sigmoid(Lgrid) = 1 / (1 + exp(-Lgrid))
```

- 按 probability threshold（概率阈值）重新导出 PLY。
- 输出字段：

```text
x y z red green blue occupancy
```

CloudCompare 中应使用 `occupancy` scalar field（占据标量场）显示概率。

示例：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\max5k.npz --ply-out max5k_occ_scalar.ply --min-prob 0.5
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\max5k.npz --ply-out max5k_occ070.ply --min-prob 0.70
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\max5k.npz --ply-out max5k_active.ply --mode active
```

# 其它当前工具

`diagnose_smoke_likelihood.py`：

- 像素级比较 clear posterior（无雾后验）和 smoke posterior（烟雾后验）。
- 拆分 fitted surface（拟合表面）、fixed smoke（固定烟雾）、background（背景）和 total（总响应）。
- `gt_range` 只用于参考线和误差，不参与拟合。
- 强烟雾下全局最大峰可能是烟雾峰，必要时使用 `--peak-bin` 指定表面候选。

示例：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\diagnose_smoke_likelihood.py --npz out/scene_00_1500.npz --row 50 --col 12 --n-pulses 5000 --peak-bin 263
```

`spad_npz_to_ply.py`：

- 对每个像素取 histogram 最大 bin。
- 转换为确定性 depth/range（深度/距离）。
- 投影为 point cloud（点云）。
- 适合作为 peak-depth baseline（峰值深度基线），使用时必须确认 depth/range 语义。

`view_npz.py` 用于快速查看 `.npz` 字段、形状、统计值和字符串 metadata（元数据）。

`plot_pixel_histogram.py` 用于查看指定像素的 transient/histogram（瞬态/直方图）。

`visualize_npz2pointcloud.py` 是当前的点云可视化辅助脚本。

`generate_spad_data.py` 是 NYUv2 单帧 SPAD 仿真脚本，不用于当前 ICL 连续序列建图。

# 当前局限与下一步

当前仍存在以下限制：

- range posterior（距离后验）是 peak-anchored local approximation（峰锚定局部近似），不是全距离联合多返回 mixture fitting（混合拟合）。
- smoke likelihood 使用仿真 metadata 中已知、沿 ray 均匀的 `kappa/gamma`，尚未从观测估计未知烟雾参数或空间变化的烟雾场。
- 当前 peak filter（峰过滤器）只判别 surface/non-surface（表面/非表面），不能显式区分 smoke/background/multiple scattering（烟雾/背景/多次散射）。
- CDF occupancy update（CDF 占据更新）仍未加入 transmittance/visibility weighting（透过率/可见性加权）。
- 尚未构建 smoke density map（烟雾密度图）。
- 仿真是 uniform-density single-scattering approximation（均匀密度单次散射近似），不包含完整 phase function（相函数）或 multiple scattering（多次散射）。
- SPAD capture（采样）不模拟 first-photon pile-up（首光子堆积）和 dead time（死时间）。
- 当前 IRF（系统响应）是 Gaussian（高斯）模型，不支持 measured IRF（实测系统响应）。
- 尚未实现 CRLB / Fisher information（克拉美罗下界 / 费舍尔信息）。
- 当前只有 visible surface benchmark（可见表面基准），没有 occupancy benchmark（占据基准）。
- dense grid（稠密栅格）在更大场景中会占用较多内存。

建图主线下一步是最小化实现 transmittance/visibility-aware occupancy update（透过率/可见性感知占据更新）：

```text
expected_range = sum(ray_p * ray_r)
visibility_weight = max(w_min, exp(-2 * kappa * expected_range))
update_weight = entropy_weight * visibility_weight * effective_update_scale
```

第一版只调整整条 ray（射线）的 log-odds update strength（对数几率更新强度），不修改 posterior（后验）、CDF inverse sensor model（CDF 逆传感器模型）和地图结构。
