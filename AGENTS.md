# 回答语言偏好

后续回答中，如果必须使用英文术语、英文变量名或英文论文/代码概念，应尽量在英文后面跟中文括号释义，例如 `posterior（后验概率）`、`occupancy grid（占据栅格）`、`profile likelihood（剖面似然）`。代码原文、命令、文件名、函数名和不可翻译的标识符可以保持原样，但解释性文字中应优先补充中文释义。

# 项目接手说明

本文档记录当前文件夹 `D:\pythonProgram\smoke_map` 中已经能判断出的项目信息，方便后续继续接手、整理和扩展。

## 项目整体目标

这个项目围绕 **Single-Photon LiDAR / SPAD / TCSPC photon histogram** 数据做三维建图。

核心研究问题是：单光子 LiDAR 每个像素拿到的不是一个确定深度，而是一条 photon arrival histogram。传统做法通常直接取最大峰值，把 histogram 转成 depth image 或 point cloud，再做建图；但这种做法会丢掉 histogram 中的噪声、不确定性和多峰信息。

当前论文和代码的主要思路是：

```text
raw photon histogram
  -> Poisson/profile likelihood range inference
  -> range posterior / 多峰距离假设
  -> uncertainty-aware occupancy grid update
  -> 3D occupancy map
```

也就是说，项目真正想做的不是普通点云重建，而是 **直接把 photon histogram 级别的不确定性传递到 occupancy mapping 中**。

## 当前目录结构

```text
.
├── AGENTS.md
├── test_mapping_profile_3d_pdfalign_fast_v3.py
├── raw2pc_undistortPoints.py
├── raw2pc_undistortPoints_single.py
└── mapping
    ├── bare_jrnl_new_sample4.tex
    ├── report.bib
    ├── IEEEtran.cls
    └── name.tex
```

## 论文文件说明

### `mapping/bare_jrnl_new_sample4.tex`

这是当前最重要的论文主稿，题目是：

```text
Photon-Histogram to Occupancy Map:
Uncertainty-Aware Mapping for Single-Photon LiDAR
```

主稿已经写好的部分：

- `Abstract`
- `Introduction`
- `Related work`
- `Probabilistic Modeling Preliminaries`
- `Single-Photon LiDAR Forward Model`
- `Probabilistic Map Representation`
- `Method`
- `Range Posterior from Photon Histograms`
- `Bayesian Occupancy Update with Range Uncertainty`
- 一个算法伪代码

主稿还没写完的部分：

- `BENCHMARK RESULTS`
- `Datasets`
- `Evaluation Setup`
- `Results`
- `REAL-WORLD APPLICATIONS`
- `Hardware System Setup`
- `conclusion`

主稿的技术主线：

1. 用 Poisson 模型描述 histogram bin count：

   ```text
   h_b ~ Poisson(lambda_b)
   ```

2. 用 IRF / Gaussian pulse 表示单光子 LiDAR 的系统响应：

   ```text
   lambda_b = sum_n a_n S(t_b - tau_n) + beta
   ```

   其中 `a_n` 是返回强度，`tau_n` 是 ToF，`beta` 是背景光/暗计数。

3. 不直接从 histogram 取单个峰值，而是在候选 range 上计算 likelihood。

4. 对 nuisance parameters `a` 和 `beta` 做 profile likelihood：

   ```text
   ell(r) = max_{a,beta >= 0} L(r, a, beta)
   ```

5. 对各个候选 range 的 score 做 softmax，得到离散 posterior：

   ```text
   p(r | histogram)
   ```

6. 把 range posterior 通过 inverse sensor model 融合进 occupancy grid。

### `mapping/report.bib`

参考文献库。主稿使用：

```latex
\bibliography{IEEEabrv,report}
```

但当前目录里没有 `IEEEabrv.bib`。如果 Overleaf 或本地 BibTeX 报错，可以临时改成：

```latex
\bibliography{report}
```

### `mapping/IEEEtran.cls`

IEEE LaTeX 模板类文件。

### `mapping/name.tex`

这是另一个较早版本或废稿，主题更偏：

```text
Physics-Aware Bayesian Semantic-Geometric Mapping for Single-Photon LiDAR
```

它包含 CRLB-guided uncertainty、semantic-geometric mapping 等想法，但文件有明显乱码，且引用 `references.bib`，当前目录没有这个文件。后续建议先以 `bare_jrnl_new_sample4.tex` 为主，`name.tex` 仅作为早期思路参考。

## Python 脚本说明

### `raw2pc_undistortPoints.py`

这是较完整的传统 baseline 脚本：把 histogram txt 文件转换成点云。

处理流程：

```text
histogram txt
  -> 每个像素取 argmax peak bin
  -> peak bin 转 depth/range
  -> 使用相机内参和畸变参数通过 cv2.undistortPoints 得到射线
  -> depth/range 沿射线投影成 3D point cloud
  -> 输出 PLY / CSV / intensity-depth 可视化图
```

关键点：

- 输入默认来自 `./imaging`。
- 默认匹配文件名类似 `RawDataHistogramMap_frame_0_*.txt`。
- 默认图像尺寸按 `192 x 256` 或 `256 x 192` 推断。
- 单个 histogram txt 通常形状是 `49152 x bins`，其中 `49152 = 192 * 256`。
- 默认时间 bin 宽度 `dt_ps = 750.0` ps。
- 距离换算：

  ```text
  bin_to_m = C * dt / 2
  ```

- 当前代码中有经验校正：

  ```python
  depth_m_1d = (peak_bin_1d.astype(np.float32) - 17) * bin_to_m
  ```

- 还会读取 `offset.txt` 做额外校正和过滤；当前仓库里没有 `offset.txt`，运行时需要提供。

常见运行方式示例：

```powershell
python raw2pc_undistortPoints.py --single-txt path\to\RawDataHistogramMap_frame_0_xxx.txt --output-dir output
```

主要依赖：

- `numpy`
- `opencv-python` / `cv2`
- `matplotlib`
- `pandas`
- 可选 `open3d`

这个脚本适合作为论文 baseline：`peak-depth point cloud`。

### `raw2pc_undistortPoints_single.py`

这是单文件、硬编码路径版本的点云转换脚本，更像早期实验脚本。

特点：

- `TXT_PATH` 是硬编码的绝对路径，需要手动改。
- 输出目录是 `output_from_txt`。
- 默认 `DT_PS = 750`。
- 默认用 `peak_bin - 25` 转换为距离。
- 也使用 `cv2.undistortPoints` 做射线去畸变。
- 会保存 `intensity_depth_maps.png` 和 `intensity_pointcloud.ply`。
- 会尝试使用 Open3D 打开可视化窗口。

这个脚本适合理解最简单的单帧处理流程，但后续开发应优先使用 `raw2pc_undistortPoints.py`。

### `test_mapping_profile_3d_pdfalign_fast_v3.py`

这是当前最接近论文方法的核心原型脚本。

目标：

```text
直接从 histogram txt 构建 occupancy grid，而不是先转确定性点云。
```

主要流程：

1. 读取 histogram txt：

   ```text
   shape: U x B
   U 通常为 49152
   B 是时间 bin 数，例如 672
   ```

2. 过滤低信号 ray：

   ```python
   mx = data.max(axis=1)
   cand = np.where(mx >= peak_thr)
   ```

3. 对每条 ray 找 peak：

   ```python
   mp_find_peaks(...)
   ```

   支持多峰，但默认参数 `--max_peaks 1`，实际默认仍然是单峰。

4. 对 peak 附近构建 range hypothesis grid：

   ```python
   compute_ll_grid_numba(...)
   ```

5. 对每个候选 range 构建 IRF/Gaussian 响应并计算 Poisson profile likelihood：

   ```python
   profile_ll_one_r_numba(...)
   ```

6. 对 likelihood 做 softmax，得到离散 posterior/pdf 和 cdf。

7. 从 posterior 中找 peak valley bounds，形成 occupied interval。

8. 沿每条 ray 做 DDA voxel traversal：

   ```python
   dda_update_dense(...)
   ```

9. 用 log-odds 更新 dense occupancy grid：

   ```text
   free: dL_free = logit(p_free) - logit(p0)
   occ:  dL_occ  = logit(p_occ)  - logit(p0)
   ```

10. 输出：

    ```text
    occ_slice.png
    occupied_rgb.ply
    ```

关键参数：

```text
--txt              必填，histogram txt 路径
--voxel            voxel 尺寸，默认 0.10 m
--range_max        局部地图 x/y 半径，也作为 ray marching 距离，默认 4.0 m
--z_min            地图 z 下界
--z_max            地图 z 上界
--max_rays         最多使用多少条 ray，默认 3000
--peak_thr         ray 最大 count 低于该值则跳过，默认 50
--Wr_bin           peak 附近 range 搜索窗口，单位 bin，默认 12
--M                range hypothesis 数量，默认 81
--p_occ            occupied 观测概率，默认 0.70
--p_free           free 观测概率，默认 0.35
--p0               occupancy prior，默认 0.50
--tau              softmax temperature，默认 1.0
--win_half         likelihood 计算窗口半宽，默认 25
--sigma_bins       Gaussian IRF 宽度，默认 2.0
--irf              可选，实测 IRF 的 .npy/.txt 文件；不传则使用 Gaussian
--range_model      range 或 z，默认 range
--max_peaks        最大峰数，默认 1
--occ_wmax_vox     occupied interval 最大宽度，单位 voxel，默认 2.5
--max_out          PLY 最大导出点数，0 表示不限制
```

运行示例：

```powershell
python test_mapping_profile_3d_pdfalign_fast_v3.py --txt path\to\RawDataHistogramMap_frame_0_xxx.txt --max_rays 3000
```

如果有实测 IRF：

```powershell
python test_mapping_profile_3d_pdfalign_fast_v3.py --txt path\to\data.txt --irf path\to\irf.npy
```

主要依赖：

- `numpy`
- `matplotlib`
- `opencv-python` / `cv2`
- 可选 `numba`，没有 numba 时能跑但会很慢

## 论文与代码的对应关系

| 论文概念 | 代码位置 |
|---|---|
| Poisson log-likelihood | `_poisson_ll_numba` |
| profile likelihood over `a,beta` | `profile_ll_one_r_numba` |
| Gaussian IRF | `build_S_gaussian` |
| measured IRF | `load_irf_1d`, `build_S_irf` |
| range hypothesis grid | `compute_ll_grid_numba` |
| posterior softmax | `compute_ll_grid_numba` |
| multi-peak detection | `mp_find_peaks` |
| occupied interval extraction | `peak_valley_bounds_numba` |
| occupancy grid / log-odds | `dda_update_dense` |
| peak-depth point cloud baseline | `raw2pc_undistortPoints.py` |

## 当前实现与论文宣称的差距

需要特别注意：论文文字中的一些点，目前代码还没有完全实现或没有完全对应。

1. 论文强调 multi-peak posterior，但核心脚本默认：

   ```text
   --max_peaks 1
   ```

   也就是说默认实验并没有真正使用多峰。

2. 论文强调 posterior dispersion 产生 adaptive update weights，但当前代码主要是：

   ```text
   posterior -> occupied interval -> 固定 dL_free / dL_occ
   ```

   还不是严格的 CDF marginalization 或基于 posterior 方差的自适应权重。

3. 论文公式中使用了完整的 marginal occupancy probability：

   ```text
   P(voxel occupied | histogram)
   ```

   但当前代码实现更像工程近似：用 posterior valley bounds 得到区间，然后做普通 free/occupied log-odds 更新。

4. 论文中还有很多空引用：

   ```latex
   \cite{}
   ```

   需要补真实文献或删除。

5. `mapping/name.tex` 中的 CRLB/semantic mapping 想法没有在当前 Python 代码中看到完整对应实现。

## 已知缺失文件和运行风险

当前目录里没有实际数据文件。运行 Python 脚本通常还需要：

- `RawDataHistogramMap_frame_0_*.txt` 形式的 histogram txt
- `offset.txt`，用于 `raw2pc_undistortPoints.py` 的距离校正
- 可选 IRF 文件，例如 `.npy` 或 `.txt`

本地没有检测到 `pdflatex` 时，无法直接在本机编译论文。可以使用 Overleaf 编译。

Overleaf 如果右侧 PDF 预览报 `PDF渲染错误`，不一定是 LaTeX 编译失败，可能是浏览器或网络无法访问 `compiles.overleafusercontent.com`。

## 建议的接手顺序

1. 先理解传统 baseline：

   ```text
   raw2pc_undistortPoints.py
   ```

   目标是明确 `histogram -> peak depth -> point cloud` 的流程。

2. 再理解论文方法原型：

   ```text
   test_mapping_profile_3d_pdfalign_fast_v3.py
   ```

   重点看 profile likelihood、posterior、occupancy update。

3. 对照主稿：

   ```text
   mapping/bare_jrnl_new_sample4.tex
   ```

   把公式和代码逐项对应。

4. 后续实验至少应包含：

   ```text
   peak-depth point cloud + mapping baseline
   vs
   profile likelihood histogram-to-occupancy method
   ```

5. 如果要增强创新性，应优先补齐：

   - 真正的 multi-peak 实验；
   - posterior CDF marginalization 的 occupancy update；
   - posterior dispersion / entropy / variance 控制 update strength；
   - 与已有 SPL reconstruction + OctoMap 的比较。

## 建议阅读的前置文献

为了完全理解这个项目，建议按顺序读：

1. Single-Photon LiDAR 基础与综述  
   Rapp et al., *Advances in Single-Photon Lidar for Autonomous Vehicles*, IEEE Signal Processing Magazine, 2020.

2. 多深度 / 多返回 single-photon imaging  
   Shin et al., *Computational multi-depth single-photon imaging*, Optics Express, 2016.

3. Bayesian SPL 多表面重建  
   Tachella et al., *Bayesian 3D Reconstruction of Complex Scenes from Single-Photon Lidar Data*, SIAM Imaging Sciences, 2019.

4. 实时 SPL 3D 重建  
   Tachella et al., *Real-time 3D reconstruction from single-photon lidar data using plug-and-play point cloud denoisers*, Nature Communications, 2019.

5. Occupancy mapping 基础  
   Hornung et al., *OctoMap: An Efficient Probabilistic 3D Mapping Framework Based on Octrees*, Autonomous Robots, 2013.

6. 与本项目动机接近的下游不确定性传播工作  
   Goyal et al., *Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs*, ICCV, 2025.

## 对创新点的当前判断

这个项目的潜在创新点不是单独的 Poisson 建模、profile likelihood 或 occupancy grid。这些都有前人基础。

更准确的创新点应表述为：

```text
将 single-photon LiDAR 的 raw histogram-level uncertainty 和 multi-peak range hypotheses
直接融合进 occupancy mapping，避免确定性 depth/point cloud 中间表示导致的错误累积。
```

如果后续实验能证明该方法在低 SNR、多峰、多返回、背景光强或烟雾/遮挡场景下明显减少 phantom occupied voxels，并优于 `peak-depth + occupancy mapping` 以及 `SPL reconstruction + occupancy mapping`，这篇文章才比较容易站住。
