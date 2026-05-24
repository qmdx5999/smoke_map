# 回答与操作偏好

后续回答中，如果必须使用英文术语、英文变量名或英文论文/代码概念，应尽量在英文后面跟中文括号释义，例如 `posterior（后验概率）`、`occupancy grid（占据栅格）`、`profile likelihood（剖面似然）`。代码原文、命令、文件名、函数名和不可翻译的标识符可以保持原样。

禁止批量删除文件或目录。不要使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。如果需要删除文件，只能一次删除一个明确路径的文件；如果需要批量删除，应停止操作并让用户手动处理。

# 当前项目状态

当前目录 `D:\pythonProgram\smoke_map` 是一个围绕 Single-Photon LiDAR / SPAD / TCSPC photon histogram（光子到达直方图）做三维占据建图的研究原型。核心目标是避免先把 histogram 压成单一 depth（深度）或 point cloud（点云），而是把 histogram-level uncertainty（直方图级不确定性）直接传进 occupancy mapping（占据建图）。

当前主要文件与目录：

```text
.
├── AGENTS.md
├── spad_npz_occupancy_mapping.py
├── spad_npz_to_ply.py
├── generate_spad_data.py
├── generate_spad_data_icl.py
├── raw2pc_undistortPoints.py
├── raw2pc_undistortPoints_single.py
├── scene_0000.npz
├── scene_0000_from_spad.ply
├── scene_0000_occupied_rgb.ply
├── scene_0000_profile_surface.ply
├── living_room_traj0_loop/
├── SPCSim/
├── Physics_Aware_Bayesian_Semantic_Geometric_Mapping_for_Single_Photon_LiDAR.pdf
└── mapping/
    ├── bare_jrnl_new_sample4.tex
    ├── report.bib
    ├── IEEEtran.cls
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

# 核心建图脚本：`spad_npz_occupancy_mapping.py`

这是当前最重要的建图脚本，已经实现单帧和多帧 `spad_hist` occupancy mapping（占据建图）。

输入要求：

- 每个 `.npz` 必须包含 `spad_hist`。
- `spad_hist` 形状应为 `(H, W, T)`。
- 当前单帧样例 `scene_0000.npz` 包含：
  - `spad_hist`: `(256, 256, 1000)`, `float32`
  - `gt_depth`: `(256, 256)`
  - `phi_bar`: `(256, 256, 1000)`

主要实现内容：

| 论文概念 | 当前代码 |
|---|---|
| Poisson likelihood（泊松似然） | `poisson_ll_numba()` |
| profile likelihood（剖面似然）消去 `a,beta` | `profile_ll_one_r_numba()` |
| Gaussian IRF（高斯系统响应） | `build_S_gaussian()` |
| peak detection（峰值检测） | `mp_find_peaks()` |
| range hypothesis grid（距离假设网格） | `compute_ll_grid_numba()` |
| posterior softmax（后验归一化） | `compute_ll_grid_numba()` |
| posterior CDF（后验累积分布） | `cdf_lookup_numba()` |
| CDF occupancy update（CDF 占据更新） | `dda_update_dense_cdf()` |
| entropy adaptive weight（熵自适应权重） | `posterior_entropy_weight()` |
| 多帧 `T_wc` 位姿读取 | `load_poses_txt()` |
| 单帧/多帧统一处理 | `process_frame()` |

单帧运行示例：

```powershell
python spad_npz_occupancy_mapping.py --npz scene_0000.npz --max_rays 100
```

多帧运行示例：

```powershell
python spad_npz_occupancy_mapping.py --npz-dir path\to\frames --poses path\to\poses.txt --map-min-x -5 --map-max-x 5 --map-min-y -5 --map-max-y 5 --map-min-z 0 --map-max-z 6
```

`poses.txt` 每行格式：

```text
frame_key m00 m01 m02 m03 m10 m11 m12 m13 m20 m21 m22 m23 m30 m31 m32 m33
```

约定：

- `frame_key` 必须等于 `.npz` 文件名去掉扩展名后的 stem（主文件名）。
- 矩阵是 `T_wc`，即 camera-to-world（相机坐标系到世界坐标系）齐次变换矩阵。
- 当前假设相机光轴为 `+z`，ray direction（射线方向）由 `[x_n, y_n, 1.0]` 构造。
- 多帧模式下所有帧共用同一个 dense `Lgrid`（稠密占据栅格）。

输出：

- `scene_0000_occupied_rgb.ply` 或 `--ply-out` 指定路径：占据体素点云。
- `scene_0000_profile_surface.ply` 或 `--surface-out` 指定路径：profile posterior（剖面后验）峰值表面点。

# 多帧仿真数据：`living_room_traj0_loop`

当前目录中的 `living_room_traj0_loop` 是从 ICL-NUIM 的 `Living Room / traj0 loop` 解压得到的连续室内序列。它适合用于 `spad_npz_occupancy_mapping.py` 的多帧建图验证。

每帧包含：

- `scene_00_xxxx.png`：RGB 图像，分辨率 `640 x 480`
- `scene_00_xxxx.depth`：文本深度图，按行展开，共 `640*480` 个浮点数
- `scene_00_xxxx.txt`：相机元数据，含 `cam_pos`、`cam_dir`、`cam_up`、`cam_angle` 等

这里的 `.txt` 不是现成的 `poses.txt`，但已经足够恢复 `T_wc`。

# ICL 序列转 SPAD：`generate_spad_data_icl.py`

`generate_spad_data_icl.py` 用于把 `living_room_traj0_loop` 批量转成 SPAD `.npz` 序列，并导出与多帧建图脚本兼容的 `poses.txt`。

实现要点：

- 读取 `.png` 作为 albedo / intensity（反照率 / 强度）来源。
- 读取 `.depth` 作为每像素 direct range（直线距离），直接输入 `TransientGenerator`。
- 解析每帧 `.txt` 中的 `cam_pos / cam_dir / cam_up`，重建 `T_wc`。
- 输出 `.npz` 与 `poses.txt`，命名保持 `scene_00_xxxx`。

输出 `.npz` 至少包含：

- `spad_hist`
- `gt_depth`
- `phi_bar`（仅当 `--save-phi-bar` 打开时）

生成命令示例：

```powershell
D:/Anaconda3/envs/pytorch/python.exe .\generate_spad_data_icl.py --in-dir .\living_room_traj0_loop\ --out out
```

如果这条命令成功完成，则 `out/` 中得到的 `.npz + poses.txt` 已经在文件格式上完全适配 `spad_npz_occupancy_mapping.py` 的多帧模式。

重要说明：

- `generate_spad_data_icl.py` 默认输出分辨率是 `256 x 256`。
- 它内部使用的是 ICL-NUIM 相机内参从 `640x480` 缩放到 `256x256` 后的值：
  - `fx = 192.48`
  - `fy = 256.0`
  - `cx = 127.8`
  - `cy = 127.733333...`
- 因此运行多帧建图时，不应继续使用 `spad_npz_occupancy_mapping.py` 的 NYU 默认内参，而应显式传入上面这组 ICL 缩放内参。

通用内参规则：

- `spad_npz_occupancy_mapping.py` 只有在输入 `.npz` 来自 NYU 默认相机模型时，才可以不传 `--fx --fy --cx --cy`。
- 如果 `.npz` 来自其他相机或其他数据集，应显式传入对应的内参。
- 最可靠的做法是：生成 `.npz` 时用什么相机模型，建图时就传什么相机模型在当前输出分辨率下的 `fx fy cx cy`。

如果已知原始相机内参和原始分辨率，且只是做了 resize（缩放）而没有 crop（裁剪）或 pad（补边），则可按下面公式计算新分辨率内参：

```text
sx = new_width  / old_width
sy = new_height / old_height

fx_new = fx_old * sx
fy_new = fy_old * sy
cx_new = cx_old * sx
cy_new = cy_old * sy
```

例如 ICL-NUIM 从 `640x480` 缩放到 `256x256`：

```text
sx = 256 / 640 = 0.4
sy = 256 / 480 = 0.533333...

fx = 481.20 * 0.4        = 192.48
fy = 480.00 * 0.533333   = 256.0
cx = 319.50 * 0.4        = 127.8
cy = 239.50 * 0.533333   = 127.733333...
```

如果中间做过 crop（裁剪）或 pad（补边），则不能只做线性缩放，还要把主点一起平移：

```text
cx_after_crop = cx_old - crop_left
cy_after_crop = cy_old - crop_top
```

然后再按 resize 公式继续缩放。

如果数据集根本没有提供相机内参，就不能可靠恢复真实射线方向；这时应优先去查数据集文档、原始标定文件或生成脚本，而不是随意猜一个 `fx fy cx cy`。

推荐多帧建图命令示例：

```powershell
python spad_npz_occupancy_mapping.py --npz-dir out --poses out\poses.txt --fx 192.48 --fy 256.0 --cx 127.8 --cy 127.7333333 --map-min-x -5 --map-max-x 5 --map-min-y -5 --map-max-y 5 --map-min-z 0 --map-max-z 6
```

如果只是先快速试跑，也可以额外加上：

```powershell
--max-frames 20 --max_rays 1000
```

# 其它脚本

`generate_spad_data.py` 是旧的 NYUv2 单帧 SPAD 仿真脚本。它适合生成单张 SPAD histogram（光子直方图）样例，不适合直接做多帧建图序列，因为 NYUv2 labeled `.mat` 不提供连续轨迹和同一场景下的真实位姿。

`spad_npz_to_ply.py` 是传统 baseline（基线）：对 `spad_hist` 每个像素取最大 bin，转成确定性 depth（深度），再投影成点云。适合做 `peak-depth point cloud` 对照。

`raw2pc_undistortPoints.py` 和 `raw2pc_undistortPoints_single.py` 是面向 histogram txt 的传统点云转换脚本。当前目录没有对应 txt 数据和 `offset.txt`，所以它们主要作为旧 baseline 参考。

`view_npz.py`、`visualize_npz2pointcloud.py` 目前属于本地辅助脚本。

# 当前代码与论文的差距

当前代码已经覆盖论文 Method（方法）部分的主要工程链路，但还不能支撑论文中所有声明。

仍未完整实现或未验证的点：

- 论文 forward model（前向模型）写的是多返回联合叠加 `lambda_b = sum_n a_n S(t_b - tau_n) + beta`；当前代码是 peak-anchored local posterior approximation（峰值锚定局部后验近似），不是联合 mixture model（混合模型）拟合。
- 当前只使用 Gaussian IRF（高斯系统响应），没有 measured IRF（实测系统响应）加载。
- 没有实际计算 CRLB / Fisher information（克拉美罗下界 / 费舍尔信息）。
- 没有 benchmark evaluation（基准评测）脚本，例如 IoU、precision、recall、phantom occupied voxels（虚假占据体素）统计。
- 没有 OctoMap、peak-depth occupancy mapping、SPL reconstruction + mapping 等 baseline 自动对比。
- 当前地图是 dense grid（稠密栅格），大范围多帧场景会吃内存；后续可考虑 sparse voxel hash（稀疏体素哈希）或 octree（八叉树）。
