# 回答与操作偏好

后续回答中，如果必须使用英文术语、英文变量名或英文论文/代码概念，应尽量在英文后面跟中文括号释义，例如 `posterior（后验概率）`、`occupancy grid（占据栅格）`、`profile likelihood（剖面似然）`。代码原文、命令、文件名、函数名和不可翻译的标识符可以保持原样。

禁止批量删除文件或目录。不要使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。如果需要删除文件，只能一次删除一个明确路径的文件；如果需要批量删除，应停止操作并让用户手动处理。

# 当前项目状态

更新时间：2026-06-06。

当前目录 `D:\pythonProgram\smoke_map` 是一个围绕 Single-Photon LiDAR / SPAD / TCSPC photon histogram（光子到达直方图）做三维占据建图的研究原型。核心目标是避免先把 histogram（直方图）压成单一 depth（深度）或 point cloud（点云），而是把 histogram-level uncertainty（直方图级不确定性）直接传进 occupancy mapping（占据建图）。

当前主线已经完成从 ICL-NUIM 连续 RGB-D 序列生成 SPAD histogram（光子直方图）、读取多帧 `T_wc` 位姿、从 range posterior（距离后验）做 CDF-marginalized occupancy update（CDF 边缘化占据更新），并输出 surface point cloud（表面点云）、occupied voxel cloud（占据体素点云）和完整 dense log-odds grid（稠密对数几率栅格）。截至 2026-06-06，仿真端 `generate_spad_data_icl.py` 已支持 per-pixel ray-integral smoke/fog forward model（逐像素射线积分烟雾/雾前向模型），建图端 `spad_npz_occupancy_mapping.py` 也已加入第一版 smoke-aware likelihood（烟雾感知似然），并修正到与 `spad_hist` 一致的 accumulated-count domain（累计计数域）。当前改动只涉及 measurement likelihood（测量似然）和 range posterior（距离后验）；后续 CDF occupancy update（CDF 占据更新）仍保持原实现。

当前主要文件与目录：

```text
.
├── AGENTS.md
├── spad_npz_occupancy_mapping.py
├── spad_npz_to_ply.py
├── analyze_occupancy_grid.py
├── export_occupancy_from_grid.py
├── generate_spad_data.py
├── generate_spad_data_icl.py
├── diagnose_smoke_likelihood.py
├── plot_pixel_histogram.py
├── view_npz.py
├── visualize_npz2pointcloud.py
├── raw2pc_undistortPoints.py
├── raw2pc_undistortPoints_single.py
├── scene_0000.npz
├── living_room_traj0_loop/
├── out/
├── SPCSim/
├── f_all.npz / f_all.png / f_all_occ.ply / f_all_surface.ply
├── full.npz / full.png / full_occ.ply / full_surface.ply
├── Physics_Aware_Bayesian_Semantic_Geometric_Mapping_for_Single_Photon_LiDAR.pdf
└── mapping/
    ├── bare_jrnl_new_sample4.tex
    ├── report.bib
    └── IEEEtran.cls
```

# 论文文件

主稿是 `mapping/bare_jrnl_new_sample4.tex`，题目：

```text
Photon-Histogram to Occupancy Map:
Uncertainty-Aware Mapping for Single-Photon LiDAR
```

主稿已经写了方法主线：

```text
raw photon histogram
  -> Poisson/profile likelihood range inference
  -> range posterior / multi-peak hypotheses
  -> CDF-marginalized occupancy update
  -> log-odds occupancy map
```

主稿仍未完成的部分包括：

- `BENCHMARK RESULTS`
- `Datasets`
- `Evaluation Setup`
- `Results`
- `REAL-WORLD APPLICATIONS`
- `Hardware System Setup`
- `conclusion`

`mapping/report.bib` 是参考文献库。主稿里仍有不少空引用 `\cite{}`，后续写论文时需要补真实引用或删掉。

注意：论文 Method（方法）部分和当前代码主线基本对应，但 benchmark（基准评测）、真实硬件实验、CRLB / Fisher information（克拉美罗下界 / 费舍尔信息）、measured IRF（实测系统响应）和若干引用仍是空的或占位的。写论文时不要把尚未实现的 benchmark / CRLB / measured IRF 当作已完成结果来表述。

当前占据输出应表述为 voxel-grid probabilistic occupancy（体素栅格概率占据）和 surface-dominant occupied evidence（以表面为主的占据证据）。不要把当前结果描述为 watertight reconstruction（封闭重建）、solid reconstruction（实体重建）或 dense filled occupancy（密实体素占据）。当前 inverse sensor model（逆传感器模型）是：表面前方 free（自由空间）、返回附近 occupied（占据）、表面后方 occluded/unknown（遮挡未知），因此导出的 occupied voxels（占据体素）天然更像 surface shell（表面壳）。

# 核心建图脚本：`spad_npz_occupancy_mapping.py`

这是当前最重要的建图脚本，已经实现单帧和多帧 `spad_hist` occupancy mapping（占据建图）。

输入要求：

- 每个 `.npz` 必须包含 `spad_hist`。
- `spad_hist` 形状应为 `(H, W, T)`。
- 新版 ICL `.npz` 建议包含 `camera_model` metadata（元数据），用于自动读取 `fx/fy/cx/cy` 和 `image_y_axis`。

当前新版 ICL `.npz` 至少包含：

```text
spad_hist    : (256, 256, 1000), float32
gt_depth_z   : (256, 256), float32  # ICL z-depth（光轴深度）
gt_range     : (256, 256), float32  # ToF/ray range（沿射线欧氏距离）
gt_depth     : (256, 256), float32  # 兼容字段，当前等于 gt_depth_z
camera_model : JSON metadata string
fog_model    : JSON metadata string  # 有雾/无雾仿真模型元数据，新版生成脚本会写入
capture_model: JSON metadata string  # SPAD capture（采样）参数，新版生成脚本会写入
```

`capture_model` 当前至少包含：

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

旧版已经生成的 `.npz` 可能没有 `capture_model`。使用 smoke likelihood（烟雾似然）时，累计脉冲数是必需参数：建图端优先使用命令行 `--n-pulses`，其次读取 `capture_model.n_pulses`；两者都缺少时会直接报错。对当前旧有雾数据应显式传入：

```powershell
--n-pulses 5000
```

如果 `generate_spad_data_icl.py` 开启 `--save-phi-bar`，还会保存：

```text
phi_bar : (256, 256, 1000), float32  # SPAD capture（采样）前的总期望响应
```

如果额外开启 `--save-components`，还会保存调试用 forward-model components（前向模型分量）：

```text
phi_surface_clear : (256, 256, 1000), float32  # 无雾衰减前的表面返回
phi_surface       : (256, 256, 1000), float32  # 加雾双程衰减后的表面返回
phi_smoke         : (256, 256, 1000), float32  # 烟雾/雾散射返回
phi_background    : (256, 256, 1000), float32  # 背景光
phi_dark          : (256, 256, 1000), float32  # 暗计数
```

这些 component（分量）只用于 debug / diagnosis（调试/诊断）和验证仿真合理性，不应作为正式 occupancy mapping（占据建图）输入，否则会把仿真真值泄漏给建图算法。

`camera_model` 当前包含：

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

主要实现内容：

| 论文概念 | 当前代码 |
|---|---|
| Poisson likelihood（泊松似然） | `poisson_ll_numba()` |
| profile likelihood（剖面似然）消去 `a,beta` | `profile_ll_one_r_numba()` |
| accumulated-count smoke profile likelihood（累计计数域烟雾剖面似然） | `fit_profile_one_r_smoke_count_numba()` / `profile_ll_one_r_smoke_count_numba()` |
| Gaussian IRF（高斯系统响应） | `build_S_gaussian()` |
| smoke integral template（烟雾积分模板） | `build_smoke_integral_templates()` |
| peak detection（峰值检测） | `mp_find_peaks()` |
| range hypothesis grid（距离假设网格） | `compute_ll_grid_numba()` |
| smoke-aware range hypothesis grid（烟雾感知距离假设网格） | `compute_ll_grid_smoke_numba()` |
| posterior softmax（后验归一化） | `compute_ll_grid_numba()` |
| posterior CDF（后验累积分布） | `cdf_lookup_numba()` |
| CDF occupancy update（CDF 占据更新） | `dda_update_dense_cdf()` |
| entropy adaptive weight（熵自适应权重） | `posterior_entropy_weight()` |
| 多峰 posterior merge（后验合并） | 按 `mp_find_peaks()` 的 peak score（峰值分数）加权合并多个局部 posterior |
| ray density normalization（射线密度归一化） | `--ray-norm-target` 与 `--update-scale` 控制每帧更新强度 |
| range / z 模型切换 | `--range_model range|z`，ICL 多帧推荐 `range` |
| camera metadata（相机元数据）读取 | `load_spad_frame_npz()` 读取 `camera_model` |
| fog/capture metadata（烟雾/采样元数据）读取 | `load_spad_frame_npz()` 读取 `fog_model` 与 `capture_model` |
| likelihood mode（似然模式）选择 | `resolve_likelihood_model()` 与 `--likelihood-model auto|clear|smoke` |
| pulse-count resolution（脉冲数解析） | `resolve_n_pulses()`，CLI `--n-pulses` 优先于 `capture_model.n_pulses` |
| image y axis（图像 y 轴方向） | `--image-y-axis auto|down|up`，ICL 新数据自动取 `up` |
| full grid dump（完整栅格导出） | `--grid-out` 保存 `Lgrid/mins/voxel`，`--grid-ply-out` 导出带 `occupancy` scalar（占据概率标量）的 PLY |
| timing profile（耗时剖析） | `--profile` |
| 多帧 `T_wc` 位姿读取 | `load_poses_txt()` |
| 单帧/多帧统一处理 | `process_frame()` |

当前关键默认值：

```text
--voxel 0.10
--range_max 7.0
--z_min -7.0
--z_max 7.0
--p_occ 0.65
--p_free 0.45
--update-scale 0.01
--ray-norm-target 10000
--range_model range
--likelihood-model auto
--max_peaks 2
--mp_thr 10.0
--export-min-prob 0.51
--image-y-axis auto
--diagnostic-checkpoints all
```

`--image-y-axis auto` 对新 ICL `.npz` 会读取 `camera_model.image_y_axis=up`，并在建图时执行 `y_n_all = -y_n_all`。原因是 ICL `poses.txt` 的 `T_wc` 第二列来自 `cam_up`（图像上方向），而普通像素坐标 `y_n=(v-cy)/fy` 默认是 image-down（图像向下）约定。这个修正是多帧清晰配准的关键之一。

`--range_model z` 在当前 dense-grid updater（稠密栅格更新器）里使用 world z（世界 z）解释距离，多帧 ICL 不推荐使用。当前 ICL SPAD transient（瞬态）按 `gt_range` 生成，应使用 `--range_model range`。

## 当前 smoke-aware likelihood（烟雾感知似然）

当前 clear likelihood（无烟雾似然）仍保留为：

```text
lambda_b(r) = alpha * S_b(r) + beta
```

第一版 smoke likelihood 使用与 `spad_hist` 相同的 accumulated-count domain（累计计数域）：

```text
lambda_b(r) = alpha * S_b(r)
            + N_pulses * gamma * G_b(r; kappa)
            + beta
```

其中：

- `S_b(r)` 是候选距离 `r` 对应的 Gaussian surface template（高斯表面模板）。
- `G_b(r; kappa)` 是从传感器到候选表面距离的 unit-density ray-integral smoke template（单位密度射线积分烟雾模板），其内部包含双程透过率 `exp(-2*kappa*s)`。
- `kappa` 来自 `fog_model.extinction_1_per_m`，`gamma` 来自 `fog_model.density`。
- `N_pulses` 来自 `--n-pulses` 或 `capture_model.n_pulses`，用于把单脉冲烟雾期望响应换算成累计 photon count（光子计数）。
- `alpha` 是已经包含表面衰减的 effective surface count amplitude（有效表面计数幅度），profile likelihood（剖面似然）只优化非负 `alpha` 和 `beta`。
- 当前代码不宣称能从自由 `alpha` 中分别恢复无雾反射强度 `a` 和 `exp(-2*kappa*r)` 消光衰减；若要分别识别二者，需要额外反照率、幅度先验或联合模型。

smoke likelihood 对每个 peak proposal（峰候选）仍只在 `peak_bin ± Wr_bin` 的候选距离网格上计算 posterior（后验），但 likelihood window（似然窗口）使用从 bin 0 到 `peak_bin + win_half` 的 prefix window（前缀窗口），以利用表面前方的烟雾散射形状。

`--likelihood-model` 行为：

- `auto`：检测到 `fog_model.enabled=true` 且 `fog_model.model=ray_integral` 时使用 smoke likelihood，否则使用 clear likelihood。
- `clear`：强制使用旧的 surface + constant background（表面 + 常数背景）模型，可作为有雾数据上的 baseline（基线）。
- `smoke`：强制使用烟雾模型；缺少有效 `fog_model` 或 `n_pulses` 时直接报错。

# 位姿文件 `poses.txt`

`poses.txt` 每行格式：

```text
frame_key m00 m01 m02 m03 m10 m11 m12 m13 m20 m21 m22 m23 m30 m31 m32 m33
```

约定：

- `frame_key` 必须等于 `.npz` 文件名去掉扩展名后的 stem（主文件名）。
- 矩阵是 `T_wc`，即 camera-to-world（相机坐标系到世界坐标系）齐次变换矩阵。
- 矩阵按行展开保存。
- 当前 ICL `T_wc` 的列向量是：camera x-right（图像右）、image-up（图像上）、z-forward（相机前方）。
- 对第一帧，`cam_up=[0,1,0]`，所以 `T_wc` 第二列是 `[0,1,0]`，表示相机局部 `+y` 指向图像上方。

第一帧示例：

```text
scene_00_0000 -0.999762369 0 -0.021799208 1.3705 0 1 0 1.51739 0.021799208 0 -0.999762369 1.44963 0 0 0 1
```

它来自 `scene_00_0000.txt` 中的：

```text
cam_pos = [1.3705, 1.51739, 1.44963]
cam_dir = [-0.0217992, 0, -0.999762]
cam_up  = [0, 1, 0]
```

构造方式：

```text
z_forward = normalize(cam_dir)
x_right   = normalize(cross(cam_up, z_forward))
y_axis    = normalize(cross(z_forward, x_right))  # 当前等价于 image-up
R_wc      = column_stack(x_right, y_axis, z_forward)
t_wc      = cam_pos
```

注意：`generate_spad_data_icl.py` 中局部变量名 `y_down` 具有误导性。当前 ICL 第一帧算出的第二列实际是 image-up（图像上），不是 image-down（图像下）。

# 多帧仿真数据：`living_room_traj0_loop` 和 `out/`

`living_room_traj0_loop/` 是从 ICL-NUIM 的 `Living Room / traj0 loop` 解压得到的连续室内序列。它适合用于 `spad_npz_occupancy_mapping.py` 的多帧建图验证。

截至 2026-06-05：

- `living_room_traj0_loop/` 中有 1510 帧 RGB/depth/metadata（元数据）。

每帧原始 ICL 文件包含：

- `scene_00_xxxx.png`：RGB 图像，分辨率 `640 x 480`。
- `scene_00_xxxx.depth`：文本深度图，按行展开，共 `640*480` 个浮点数。
- `scene_00_xxxx.txt`：相机元数据，含 `cam_pos`、`cam_dir`、`cam_up`、`cam_angle` 等。

# ICL 序列转 SPAD：`generate_spad_data_icl.py`

`generate_spad_data_icl.py` 用于把 `living_room_traj0_loop` 批量转成 SPAD `.npz` 序列，并导出与多帧建图脚本兼容的 `poses.txt`。

实现要点：

- 读取 `.png` 作为 albedo / intensity（反照率 / 强度）来源。
- 读取 `.depth` 作为 z-depth（光轴深度）。
- 将 `z-depth` 转为 ToF/ray range（沿射线欧氏距离）：`gt_range = z * sqrt(1 + x_n^2 + y_n^2)`。
- 使用 `gt_range` 生成 SPAD transient（瞬态）和 histogram（直方图）。新版脚本显式拆分 forward model components（前向模型分量）：`phi_surface_clear`、`phi_surface`、`phi_smoke`、`phi_background`、`phi_dark`，并组合为 `phi_bar = phi_surface + phi_smoke + phi_background + phi_dark`。
- 解析每帧 `.txt` 中的 `cam_pos / cam_dir / cam_up`，重建 `T_wc`。
- 输出 `.npz` 与 `poses.txt`，命名保持 `scene_00_xxxx`。
- `camera_model` 写入 `depth_model=z`、`tof_model=range`、`image_y_axis=up`。
- 新增 `fog_model` JSON metadata（雾模型元数据），记录 `fog_model`、`fog_extinction`、`fog_density`、`fog_step`、`fog_range_max`、`surface_attenuation` 和 `smoke_return`。
- 新增 `capture_model` JSON metadata（采样模型元数据），记录 `n_pulses`、`tmax_ns`、`n_tbins`、`fwhm_ns`、`fast_sim` 和采样语义 `Poisson(phi_bar * n_pulses)`。
- 当前只保留 `--fog-model none|ray_integral`。旧 `--fog` 和 `gamma` fog（Gamma 雾模型）接口已经移除。
- `ray_integral` 是 per-pixel uniform-density ray integral（逐像素、均匀密度射线积分）模型：每个像素的烟雾积分上限是 `min(gt_range[u,v], fog_range_max, dmax)`，因此表面后方的 smoke/fog scattering（烟雾/雾散射）不会继续贡献到该像素。
- 有雾时表面返回使用双程衰减：`phi_surface = phi_surface_clear * exp(-2 * fog_extinction * gt_range)`。
- 当前 `SPCSim` 的 `BaseEWHSPC(fast_sim=True)` 仍不模拟 first-photon pile-up（首光子堆积）/ dead time（死时间）畸变。

生成命令：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\generate_spad_data_icl.py --in-dir .\living_room_traj0_loop --out out
```

如果要保存理论瞬态响应 `phi_bar`，额外加：

```powershell
--save-phi-bar
```

如果只做小规模 debug / diagnosis（调试/诊断），可以额外保存 forward-model components（前向模型分量）：

```powershell
--save-components
```

注意：`--save-components` 会显著增大 `.npz` 体积。正式多帧建图数据通常不需要开启它；后续建图代码应主要读取 `spad_hist`、`camera_model`、`fog_model`、`capture_model` 和 `poses.txt`，不应依赖 `phi_surface` / `phi_smoke` 等仿真真值分量。

生成逐像素 `ray_integral` 有雾数据的推荐命令：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\generate_spad_data_icl.py --in-dir .\living_room_traj0_loop --out out_fog_ray --fog-model ray_integral --fog-extinction 0.15 --fog-density 0.03 --fog-step 0.05
```

小规模有雾调试命令：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\generate_spad_data_icl.py --in-dir .\living_room_traj0_loop --out out_fog_ray_debug --n 10 --fog-model ray_integral --fog-extinction 0.15 --fog-density 0.03 --fog-step 0.05 --save-phi-bar --save-components
```

当前传感器/仿真参数：

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

ICL 内参从 `640x480` 缩放到 `256x256`：

```text
fx = 481.20 * (256 / 640) = 192.48
fy = 480.00 * (256 / 480) = 256.0
cx = 319.50 * (256 / 640) = 127.8
cy = 239.50 * (256 / 480) = 127.733333...
```

ICL 内参来源与缩放原因（刷新 AGENTS.md 时必须保留）：

- 后续刷新或重写 `AGENTS.md` 时，必须同时保留上面的 “ICL 内参从 `640x480` 缩放到 `256x256`” 公式块，以及本段 “ICL 内参来源与缩放原因” 说明。
- `fx/fy/cx/cy` 是 camera intrinsics（相机内参），用于把像素 `(u,v)` 反投影成归一化相机射线：`x_n = (u - cx) / fx`、`y_n = (v - cy) / fy`，再形成 ray direction（射线方向）近似 `[x_n, y_n, 1]`。
- 原始 ICL-NUIM RGB-D 图像分辨率是 `640x480`，原始内参是 `fx=481.20, fy=480.00, cx=319.50, cy=239.50`；当前生成的 SPAD histogram（光子直方图）分辨率是 `256x256`，所以内参必须随 resize（缩放）按宽高比例同步缩放。
- 横向参数 `fx/cx` 按宽度比例 `256 / 640` 缩放，纵向参数 `fy/cy` 按高度比例 `256 / 480` 缩放。`fx` 和 `fy` 的缩放比例不同，是因为原始宽高 `640x480` 被缩放成正方形 `256x256`。
- 这些值必须和生成 `.npz` 时的输出分辨率一致，否则每个 histogram pixel（直方图像素）对应的 ray direction（射线方向）会错，多帧建图会出现空间拉伸、压缩或错位。
- 当前新版 `out/*.npz` 已包含 `camera_model` metadata（相机模型元数据），建图脚本在不手动传 `--fx/--fy/--cx/--cy` 时可以自动读取这些值；命令中显式填写这些值主要是为了实验可复现，并兼容旧 `.npz` 没有 metadata 的情况。
- 后续刷新或重写 `AGENTS.md` 时，不要只保留内参数值，必须保留“为什么要这样缩放”和“这些内参用于像素到 ray（射线）反投影”的解释。

# 推荐建图命令

有雾数据第一版 smoke-aware likelihood 建图命令，新版 `.npz` 已有 `capture_model` 时可以自动读取 `n_pulses`：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\spad_npz_occupancy_mapping.py --npz-dir out_fog_ray --poses out_fog_ray/poses.txt --fx 192.48 --fy 256.0 --cx 127.8 --cy 127.7333333 --range_model range --range_max 7.0 --image-y-axis auto --likelihood-model smoke --max_rays 10000 --update-scale 0.005 --ply-out fog_occ.ply --surface-out fog_surface.ply --grid-out fog.npz --profile
```

当前旧有雾数据没有 `capture_model` 时必须显式添加：

```powershell
--likelihood-model smoke --n-pulses 5000
```

100 帧快速验证：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\spad_npz_occupancy_mapping.py --npz-dir out --poses out/poses.txt --fx 192.48 --fy 256.0 --cx 127.8 --cy 127.7333333 --range_model range --range_max 7.0 --image-y-axis auto --max-frames 100 --max_rays 10000 --update-scale 0.01 --ply-out f100_occ.ply --surface-out f100_surface.ply --grid-out f100.npz --profile
```

全帧推荐结果命令（较温和更新强度）：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\spad_npz_occupancy_mapping.py --npz-dir out --poses out/poses.txt --fx 192.48 --fy 256.0 --cx 127.8 --cy 127.7333333 --range_model range --range_max 7.0 --image-y-axis auto --max_rays 10000 --update-scale 0.005 --ply-out full_occ.ply --surface-out full_surface.ply --grid-out full.npz --profile
```

说明：

- `--max-frames` 不写或为 `0` 表示处理全部 1510 帧。
- `--max_rays 10000` 表示每帧最多均匀抽样 10000 条 candidate rays（候选射线）。
- 全帧运行大约需要 1.5 小时左右，取决于机器状态。
- `--grid-out` 很重要，后续可以不重跑昂贵的 histogram mapping（直方图建图），直接从 grid（栅格）按不同阈值导出 PLY。

# 当前实验记录

100 帧 `f100` 实验，命令中使用 `--max-frames 100 --max_rays 10000 --update-scale 0.01`：

```text
surface points: 650,597
active voxels: 31,689
occupied voxels @0.51: 6,950
grid shape: (140, 140, 140)
```

全帧 `f_all` 实验，使用 `--update-scale 0.01`：

```text
surface points: 8,803,959
active voxels: 60,548
occupied voxels @0.51: 15,790
active p25/p50/p75/p95/p99 ≈ 0.016 / 0.471 / 0.513 / 0.924 / 0.9999
```

全帧 `full` 实验，使用 `--update-scale 0.005`，更适合作为当前主结果：

```text
surface points: 8,803,959
active voxels: 60,514
occupied voxels @0.51: 14,061
active p25/p50/p75/p95/p99 ≈ 0.112 / 0.485 / 0.507 / 0.776 / 0.998
```

结论：

- 修复 `z-depth -> range` 和 `image_y_axis=up` 后，全帧表面点云已经从“帧越多越糊”变成清晰房间结构。
- `--update-scale 0.005` 比 `0.01` 更温和，active voxels（活跃体素）两端饱和更少，概率层次更适合展示。
- 仍会有少量体素接近 `0.000045` 或 `0.999955`，这是 `Lmin=-10` / `Lmax=10` log-odds clipping（对数几率截断）的结果。被许多帧重复扫到的 free/occupied 区域饱和是合理现象。
- 大量 unknown/prior voxels（未知/先验体素）是正常的，因为地图范围是 `[-7,7]^3`，真实可见房间区域只占一部分，且表面后方按 occluded/unknown（遮挡未知）处理。

# 后处理与阈值导出

`analyze_occupancy_grid.py` 用于读取 `--grid-out` 生成的 `.npz`，把 `Lgrid` 转成 occupancy probability（占据概率），打印 all voxels（全部体素）和 active voxels（活跃体素）的分位数、阈值计数，并输出概率直方图 PNG。

示例：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\analyze_occupancy_grid.py --grid .\full.npz --hist-out full.png
```

`export_occupancy_from_grid.py` 用于从保存好的 `--grid-out` 结果重新按阈值导出 PLY，不必重跑 histogram mapping（直方图建图）。它读取 `Lgrid/mins/voxel`，把 log-odds grid（对数几率栅格）转换成 occupancy probability（占据概率）：

```text
p = sigmoid(Lgrid) = 1 / (1 + exp(-Lgrid))
```

默认 `--mode occupied` 会导出 active voxels（活跃体素）中满足 `--min-prob <= p <= --max-prob` 的体素。`--mode active` 会导出所有 active voxels，并同样保留每个体素的 `occupancy` scalar field（占据概率标量场）。

输出 PLY 字段为：

```text
x y z red green blue occupancy
```

其中 `red/green/blue` 是灰度 fallback（兜底颜色），`occupancy` 是推荐在 CloudCompare 中使用的 scalar field（标量场）。CloudCompare 导入 PLY 时应把 `occupancy` 添加/识别为 scalar field；导入后把颜色显示切到 `Scalar field`，再用 color scale（色带）表达占据概率高低。蓝/绿/黄/红等颜色由 CloudCompare 当前 color scale 决定；代码决定的是每个点携带的 `occupancy` 数值。

推荐从 `max5k.npz` 或 `full.npz` 导出多个阈值观察：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\max5k.npz --ply-out max5k_occ_scalar.ply --min-prob 0.5
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\max5k.npz --ply-out max5k_occ055.ply --min-prob 0.55
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\max5k.npz --ply-out max5k_occ070.ply --min-prob 0.70
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\max5k.npz --ply-out max5k_active.ply --mode active
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\full.npz --ply-out full_occ055.ply --min-prob 0.55
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\full.npz --ply-out full_occ060.ply --min-prob 0.60
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\full.npz --ply-out full_occ070.ply --min-prob 0.70
D:/Anaconda3/envs/pytorch/python.exe .\export_occupancy_from_grid.py --grid .\full.npz --ply-out full_occ090.ply --min-prob 0.90
```

当前建图脚本里的 `--export-min-prob` 默认是 `0.51`，直接随建图导出的 occupied cloud（占据点云）会偏宽，适合保留弱占据证据；如果想看高置信结构，优先用 `export_occupancy_from_grid.py` 从 `--grid-out` 的 `.npz` 重新导出 `0.70` 或 `0.90` 阈值版本。

# 其它脚本

`generate_spad_data.py` 是旧的 NYUv2 单帧 SPAD 仿真脚本。它适合生成单张 SPAD histogram（光子直方图）样例，不适合直接做多帧建图序列，因为 NYUv2 labeled `.mat` 不提供连续轨迹和同一场景下的真实位姿。

`spad_npz_to_ply.py` 是传统 baseline（基线）：对 `spad_hist` 每个像素取最大 bin，转成确定性 depth/range（深度/距离）后投影成点云。适合做 `peak-depth point cloud` 对照，但需要注意 depth/range 语义。

`plot_pixel_histogram.py` 是像素级 histogram（直方图）快速查看脚本，会优先读取 `.npz` 中的 `phi_bar` 或 `spad_hist` 并保存 `histogram.png`。

`diagnose_smoke_likelihood.py` 是独立的像素级 smoke likelihood diagnosis（烟雾似然诊断）工具。它用于在给定 surface proposal peak（表面候选峰）后：

- 比较 clear posterior（无烟雾后验）和 smoke posterior（烟雾后验）。
- 输出 clear/smoke MAP range（最大后验距离）及其相对 `gt_range` 的误差。
- 将 smoke model 拆成 `fitted surface`、`fixed smoke`、`fitted background` 和 `fitted total`。
- 检查固定烟雾分量是否已经按 `N_pulses * gamma * G` 转到累计计数尺度。

该脚本默认使用 `np.argmax(spad_hist)` 作为 `peak_bin`，但强烟雾条件下全局最大 bin 可能是近距离 smoke return（烟雾返回），而不是真实表面。此时必须使用 `--peak-bin` 指定要诊断的表面候选，否则 `peak_bin ± Wr_bin` 的局部距离搜索不会覆盖真实表面。

旧数据诊断示例：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\diagnose_smoke_likelihood.py --npz out/scene_00_1500.npz --row 50 --col 12 --n-pulses 5000 --peak-bin 263
```

当前验证记录：

```text
frame       : scene_00_1500
pixel       : (50, 12)
proposal bin: 263
gt_range    : 3.938756 m
clear_range : 3.924283 m, absolute error ≈ 1.45 cm
smoke_range : 3.933277 m, absolute error ≈ 0.55 cm
surface sum : 19.23 counts
smoke sum   : 349.74 counts
background  : 197.66 counts
```

此单像素结果说明累计计数域 smoke term（烟雾项）已经实际参与 posterior（后验）计算，但它只是局部 sanity check（合理性检查），不能替代多像素、多帧 benchmark（基准评测）。

`gt_range` 在诊断脚本中只用于画参考线和计算误差，不参与 likelihood fit（似然拟合）。脚本也不会读取 `phi_surface`、`phi_smoke` 或 `phi_bar` 作为推断输入，因此不会把仿真真值泄漏给模型。

`view_npz.py` 是 `.npz` 快速检查脚本，当前已经适配字符串型 `camera_model` metadata（元数据），不会再对非数值数组调用 `min/max/mean`。

`raw2pc_undistortPoints.py` 和 `raw2pc_undistortPoints_single.py` 是面向 histogram txt 的传统点云转换脚本。当前目录没有对应 txt 数据和 `offset.txt`，所以它们主要作为旧 baseline 参考。

`visualize_npz2pointcloud.py` 目前属于本地辅助脚本。

# 当前代码与论文的差距

当前代码已经覆盖论文 Method（方法）部分的主要工程链路，但还不能支撑论文中所有声明。

仍未完整实现或未验证的点：

- 论文 forward model（前向模型）写的是多返回联合叠加 `lambda_b = sum_n a_n S(t_b - tau_n) + beta`；当前代码是 peak-anchored local posterior approximation（峰值锚定局部后验近似），不是联合 mixture model（混合模型）拟合。
- 建图端已经加入第一版 accumulated-count smoke likelihood（累计计数域烟雾似然），但当前只支持已知、沿 ray 均匀的 `kappa/gamma`，参数来自仿真 `fog_model` metadata（元数据），尚未从观测数据估计未知烟雾参数或空间变化的烟雾场。
- 当前 smoke likelihood 只修改 measurement likelihood（测量似然）和 range posterior（距离后验）；occupancy inverse sensor model（占据逆传感器模型）及 CDF occupancy update（CDF 占据更新）仍是原来的 surface/free/unknown 语义，尚未实现 smoke-aware occupancy update（烟雾感知占据更新）或 smoke density map（烟雾密度地图）。
- 当前 `--max_peaks` 默认固定为 `2`，且 posterior 仍依赖 peak proposal（峰候选）。强烟雾下全局最大峰可能是 smoke return（烟雾返回）而不是真实表面；自适应峰数量、surface peak filtering（表面峰筛选）和 coarse-to-fine full-range posterior（粗到细全距离后验）尚未实现。
- 尚未实现显式的 `surface / smoke / background` latent classification（潜变量分类）。当前诊断脚本需要给定正确的 surface proposal peak（表面候选峰），并不负责自动判断某个峰属于表面还是烟雾。
- 当前有雾仿真是 uniform-density single-scattering approximation（均匀密度单次散射近似），还不是完整 3D smoke density field（烟雾密度场）、phase function（相函数）或 multiple scattering（多次散射）模型。
- 当前 SPAD capture（采样）仍不模拟 first-photon pile-up（首光子堆积）/ detector dead time（探测器死时间）
- 当前只使用 Gaussian IRF（高斯系统响应），没有 measured IRF（实测系统响应）加载。
- 没有实际计算 CRLB / Fisher information（克拉美罗下界 / 费舍尔信息）。
- 目前有 `analyze_occupancy_grid.py` 和 `export_occupancy_from_grid.py` 这类后处理/诊断工具，但还没有真正的 benchmark evaluation（基准评测）脚本，例如 IoU、precision、recall、phantom occupied voxels（虚假占据体素）统计。
- 没有 OctoMap、peak-depth occupancy mapping、SPL reconstruction + mapping 等 baseline 自动对比。
- 当前地图是 dense grid（稠密栅格），大范围多帧场景会吃内存；后续可考虑 sparse voxel hash（稀疏体素哈希）或 octree（八叉树）。
- 当前 occupancy output（占据输出）应解释为 surface-dominant occupied evidence（以表面为主的占据证据），不是 watertight/solid volume reconstruction（封闭/实体体积重建）。
