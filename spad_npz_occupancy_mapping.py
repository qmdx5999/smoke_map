"""
Build a probabilistic occupancy grid from SPAD histograms stored in an .npz file.

This adapts the occupancy-mapping pipeline from
test_mapping_profile_3d_pdfalign_fast_v3.py to scene_0000.npz-style data:
spad_hist has shape (H, W, T), and the time-bin distance scale is derived from
the laser period and number of bins.
"""

import argparse
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco


C_M_PER_S = 299_792_458.0

DEFAULT_NPZ = "scene_0000.npz"
DEFAULT_SLICE_OUT = "scene_0000_occ_slice.png"
DEFAULT_PLY_OUT = "scene_0000_occupied_rgb.ply"
DEFAULT_SURFACE_OUT = "scene_0000_profile_surface.ply"

NYU_WIDTH = 640.0
NYU_HEIGHT = 480.0
NYU_FX = 518.8579
NYU_FY = 519.4696
NYU_CX = 325.5824
NYU_CY = 253.7362


def logit(p: float) -> float:
    """
    把0-1 之间的概率值，转换成无界的对数几率值
    """
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    
    return math.log(p / (1.0 - p))


def scaled_nyu_intrinsics(width: int, height: int):
    """
    根据输入图像的分辨率，自动计算NYU Depth 数据集标准内参的缩放版本
    """
    sx = width / NYU_WIDTH
    sy = height / NYU_HEIGHT
    
    return NYU_FX * sx, NYU_FY * sy, NYU_CX * sx, NYU_CY * sy


def gaussian_kernel_1d(sigma_bins: float, radius: int) -> np.ndarray:
    """
    生成一个和 SPAD 脉冲形状完全一样的高斯模板，和直方图做卷积后，真实的高斯形状峰值会被放大，而随机噪声会被抑制，
    这样就能从噪声中找到所有显著的反射峰值
    """
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(max(sigma_bins, 1e-6))) ** 2)
    k /= k.sum() + 1e-12
    
    return k


def mp_find_peaks(
    hist: np.ndarray,  # 输入的一维光子计数直方图
    max_peaks: int,  # 最多检测多少个峰值
    sigma_bins: float,  # 高斯核的标准差，单位：bin
    score_thr: float,  # 峰值的最小分数阈值，低于这个值的峰被认为是噪声
    support_radius: int,  # 检测到一个峰后，要擦除的周围区域的半径
    min_sep: int,  # 两个峰值之间的最小允许间距
):
    """Matching-pursuit style peak picking using a Gaussian correlation kernel."""
    """
    匹配追踪峰值检测算法，核心思想：面前有一堆沙子，里面混着几颗大小不同的石头。匹配追踪算法的做法是：
    先找到最大的那颗石头，把它捡出来，然后在剩下的沙子里，再找下一颗最大的石头，重复这个过程，直到捡够了指定数量的石头，或者剩下的都是沙子
    沙子 = 背景噪声，石头 = 真实的反射峰值，每次找到一个峰值后，就把它周围的区域 "擦除"，避免重复检测同一个峰

    先做一次卷积找当前最强峰，找到后把峰附近一段 support_radius（支持半径） 置零，再找下一个峰，重复直到找到 max_peaks 个，或者分数低于阈值
    """
    h_res = hist.copy()
    kernel = gaussian_kernel_1d(sigma_bins, support_radius)  # 生成一个一维的高斯核函数，用来匹配峰值的模板，它的形状和 SPAD 传感器的脉冲响应形状是一样的
    peaks = []
    scores = []
    n_bins = int(h_res.size)  # 获取直方图的长度（时间 bin 的数量）

    # 最多找 max_peaks 个峰
    for _ in range(int(max_peaks)):
        # 卷积运算，用高斯核这个 "模板"，在直方图上从左到右滑动，每个位置计算模板和直方图的匹配程度，匹配得越好，响应值越高，
        # mode="same"表示输出的卷积结果和输入的直方图长度相同
        corr = np.convolve(h_res.astype(np.float64), kernel, mode="same")
        t = int(np.argmax(corr))  # 找到卷积响应最大的位置，这个位置就是当前最显著的峰值的位置
        s = float(corr[t])  # 获取这个最大响应值，也就是这个峰的 "分数"
        
        # 如果当前最显著的峰的分数都低于阈值，说明剩下的都是噪声了，直接跳出循环，不再继续找峰
        if s < score_thr:
            break
        
        # 检查和已有的峰是否太近
        if any(abs(t - p) < min_sep for p in peaks):
            lo = max(0, t - support_radius)
            hi = min(n_bins, t + support_radius + 1)
            h_res[lo:hi] = 0
            continue

        # 将有效峰加入列表，并擦除周围区域
        peaks.append(t)
        scores.append(s)
        lo = max(0, t - support_radius)
        hi = min(n_bins, t + support_radius + 1)
        h_res[lo:hi] = 0

    # 对峰值按位置排序
    order = np.argsort(peaks)
    peaks = [peaks[i] for i in order]
    scores = [scores[i] for i in order]
    
    return peaks, scores


@njit(cache=True, fastmath=True)
def poisson_ll_numba(h, lam):
    """
    给定观测到的光子计数直方图h和期望光子数分布lam，计算它们之间的泊松对数似然值,这个值越大，说明lam和h越匹配，也就是模型越符合真实的观测数据
    ll 是 Log Likelihood 对数似然 的缩写
    h:观测到的光子计数直方图;lam:期望的光子数分布
    """
    s = 0.0  # 初始化总对数似然值为 0
    # 循环计算每个 bin 的对数似然
    for i in range(h.size):
        hi = h[i]  # 第 i 个 bin 中观测到的光子数（整数）
        li = lam[i]  # 第 i 个 bin 中期望的光子数（浮点数）
        s += hi * math.log(li) - li  # 泊松分布的概率公式取对数
    
    return s  # 返回所有 bin 的对数似然值的总和,这个值越大，说明lam和h越匹配


@njit(cache=True, fastmath=True)
def profile_ll_one_r_numba(h, S, eps=1e-6, max_iter=12):
    """
    给定一个观测到的光子直方图和一个假设的距离对应的高斯脉冲形状，自动找到最匹配的反射率和背景光强度，然后返回它们的匹配程度
    匹配程度越高，墙在这个距离的可能性就越大。one_r表示这个函数只计算一个候选距离的剖面似然值
    """
    beta = 0.0  # 初始化背景率beta为 0
    hmax = 0.0  # 初始化直方图的最大值hmax为 0
    for i in range(h.size):
        beta += h[i]  # 计算直方图的总和
        if h[i] > hmax:
            hmax = h[i]  # 计算直方图的最大值
    beta = max(beta / max(1, h.size), 0.0)  # beta / h.size：直方图的平均值，用平均值作为背景率的初始值
    
    # hmax为直方图的最大值，hmax - beta就是 峰值处的总光子数减去背景光子数 即 峰值处的信号光子数，为什么用峰值处的总光子数减去beta作为反射率a的初始值
    # 因为h = a * S + β,S的最大值是1，所以在峰值处hmax = a * 1 + β，所以a = hmax -β
    a = max(hmax - beta, 0.0)

    smax = 0.0
    for i in range(S.size):
        if S[i] > smax:
            smax = S[i]  # 遍历整个高斯模板S，找到窗口内的实际最大值smax
    if smax <= 0.0:  # 当高斯模板全为 0 时
        lam = np.empty_like(h)  # 创建一个和输入直方图h形状完全相同、数据类型完全相同的空数组
        for i in range(h.size):
            lam[i] = max(beta, eps)  # 候选距离太远了，没有任何反射光子到达探测器，我们看到的所有光子都是背景光，每个 bin 的期望光子数都等于背景率beta
        return poisson_ll_numba(h, lam)  # 计算并返回对数似然值
    a = a / max(smax, 1e-12)  # 之前的初始化公式a = hmax - beta是在smax=1.0的情况下适用，若S的最大值是smax则应该这样初始化a

    lam = np.empty_like(h)  # 初始化期望光子数数组
    for _ in range(max_iter):  # 牛顿法迭代循环
        for i in range(h.size):
            lam[i] = max(a * S[i] + beta, eps)  # 计算每个 bin 的期望光子数

        g_a = 0.0  # 对数似然对反射率a的梯度（一阶偏导数）
        g_b = 0.0  # 对数似然对背景率beta的梯度（一阶偏导数）
        for i in range(h.size):  # 循环计算每个 bin 对梯度的贡献，更新梯度g_a g_b
            inv = h[i] / lam[i] - 1.0
            g_a += S[i] * inv
            g_b += inv
        if abs(g_a) + abs(g_b) < 1e-6:  # 收敛判断
            break

        # 初始化海森矩阵元素
        H_aa = 0.0  # 对数似然对a的二阶偏导数
        H_ab = 0.0  # 对数似然对a和beta的混合二阶偏导数
        H_bb = 0.0  # 对数似然对beta的二阶偏导数
        for i in range(h.size):  # 循环计算每个 bin 对海森矩阵的贡献，更新二阶导数H_aa H_ab H_bb
            w = h[i] / (lam[i] * lam[i])
            H_aa += -(S[i] * S[i]) * w
            H_ab += -S[i] * w
            H_bb += -w

        det = H_aa * H_bb - H_ab * H_ab  # 计算海森矩阵的行列式
        if (not math.isfinite(det)) or abs(det) < 1e-18:
            break

        delta_a = (H_bb * g_a - H_ab * g_b) / det  # 计算反射率a的更新量
        delta_b = (-H_ab * g_a + H_aa * g_b) / det  # 计算背景率beta的更新量
        base_ll = poisson_ll_numba(h, lam)  # 计算当前参数下的基准似然值

        step = 1.0  # 初始化步长
        ok = False  # 初始化成功标志,初始化为 False，表示还没有找到合适的步长
        for __ in range(10):
            # 用当前的步长step乘以更新量delta_a和delta_b，得到新的参数值
            a_new = a - step * delta_a
            b_new = beta - step * delta_b
            if a_new < 0.0:
                a_new = 0.0
            if b_new < 0.0:
                b_new = 0.0

            # 计算新的期望光子数数组
            for i in range(h.size):
                lam[i] = max(a_new * S[i] + b_new, eps)
            ll_new = poisson_ll_numba(h, lam)  # 计算新的似然值
            if math.isfinite(ll_new) and ll_new >= base_ll - 1e-10:
                a = a_new
                beta = b_new
                ok = True  # 设置ok=True，表示这次更新成功
                break  # 跳出线搜索循环，进入下一次牛顿迭代
            step *= 0.5  # 如果不满足接受条件，就把步长减半,然后回到循环开头，用新的步长再试一次
        if not ok:  # 如果 10 次尝试都失败了，ok仍然是 False,这说明当前点的二次近似非常差，继续迭代也不会有好结果
            break  # 直接退出牛顿迭代循环，保持原来的参数值不变

    # 用牛顿法迭代收敛后得到的最优参数a和beta，计算最终的最大对数似然值并返回
    for i in range(h.size):
        lam[i] = max(a * S[i] + beta, eps)
    return poisson_ll_numba(h, lam)


@njit(cache=True, fastmath=True)
def build_S_gaussian(idx0, idx1, rbin, sigma_bins):
    """
    给定一个候选距离rbin，生成一个和直方图窗口一样长的高斯脉冲模板，模拟如果墙正好在rbin这个距离，我们应该看到的光子分布形状
    给我一个候选距离，我给你生成一个 "如果墙在这个距离，应该长什么样" 的光子分布模板。
    然后我们就可以用这个模板去和真实的直方图比一比，看看有多像，越像就说明墙在这个距离的概率越大
    idx0：直方图窗口的左边界；idx1：直方图窗口的右边界；rbin：候选距离的 bin 索引，要为这个距离生成模板；sigma_bins：高斯脉冲的标准差，单位：bin
    """
    n = idx1 - idx0  # 计算模板长度
    S = np.empty(n, dtype=np.float64)  # 创建一个空的浮点数组，用来存储高斯模板的值
    s = max(sigma_bins, 1e-6)  # 数值安全处理，防止sigma_bins为 0 导致后面除以 0 的错误
    for i in range(n):
        delta = (idx0 + i) - rbin  # idx0+i：当前计算的位置的 bin 索引：rbin：高斯的中心位置，也就是候选距离
        S[i] = math.exp(-0.5 * (delta / s) * (delta / s))
    
    return S  # 返回生成的高斯模板数组


@njit(cache=True, fastmath=True)
def compute_ll_grid_numba(h, peak_bin, Wr_bin, M, win_half, sigma_bins, tau, bin_to_m):
    """
    给定一个检测到的光子直方图峰值位置，在峰值周围的一个小范围内，以超分辨率的方式（就是用了M）计算每个可能距离的剖面似然值，
    然后将其转换为贝叶斯后验概率分布和累积分布函数，最终输出完整的距离估计结果
    h:完整的光子计数直方图;peak_bin:卷积后检测到的峰值位置（bin 索引）;Wr_bin:重建窗口半宽（候选距离范围半宽）;M:超分辨率采样点数
    win_half:直方图窗口半宽;sigma_bins:高斯脉冲的标准差;tau:时间 bin 宽度，单位是ns;bin_to_m:bin 到米的转换系数
    """
    B = h.size  # 完整直方图的总 bin 数
    
    # 计算直方图窗口的边界
    b0 = peak_bin - win_half
    if b0 < 0:
        b0 = 0
    b1 = peak_bin + win_half + 1  # Python 的切片是左闭右开的。如果要包含peak_bin + win_half这个 bin，那么右边界就必须是peak_bin + win_half + 1
    if b1 > B:
        b1 = B

    # 截取局部直方图
    nwin = b1 - b0
    h_w = np.empty(nwin, dtype=np.float64)  # 只截取峰值周围±win_half个 bin 的局部直方图h_w
    for i in range(nwin):
        h_w[i] = h[b0 + i]

    ll = np.empty(M, dtype=np.float64)  # 创建一个长度为M的空数组ll,用来存储每个候选距离的剖面似然值
    r0 = peak_bin - Wr_bin  # 候选距离的起点
    r1 = peak_bin + Wr_bin  # 候选距离的终点
    step = 0.0 if M == 1 else (r1 - r0) / (M - 1)  # 两个相邻候选距离之间的间隔

    # 计算每个候选距离的似然值
    for k in range(M):
        rb = r0 + step * k  # 计算第 k 个候选距离的 bin 索引rb
        S = build_S_gaussian(b0, b1, rb, sigma_bins)  # 为这个距离生成对应的高斯脉冲模板S
        ll[k] = profile_ll_one_r_numba(h_w, S)  # 计算这个距离的剖面似然值并把似然值存入ll数组的第 k 个位置，最终得到距离 - 似然曲线

    # 将似然曲线归一化到与时间 bin 宽度无关的尺度，然后找到最大的似然值。这个最大值用来判断这个峰值是否是真实的物体反射，还是噪声产生的虚假信号
    t = max(tau, 1e-6)
    m = ll[0] / t
    for i in range(1, M):
        v = ll[i] / t
        if v > m:
            m = v

    p = np.empty(M, dtype=np.float64)  # 创建一个长度为 M 的空数组p，p[i]表示物体在第 i 个候选距离的概率
    s = 0.0  # 用来存储所有指数化似然值的总和
    for i in range(M):  # 循环计算每个距离的指数化似然值，因为计算的是对数似然值，而概率和似然值成正比，不是和对数似然值成正比
        v = math.exp(ll[i] / t - m)
        p[i] = v
        s += v
    if s <= 0.0:  # 如果所有的指数化似然值都下溢变成了 0，那么总和s就会等于 0
        for i in range(M):
            p[i] = 1.0 / M  # 无法进行归一化，所以我们假设所有距离的概率相等，也就是均匀分布，通常发生在信噪比极低的时候，所有候选距离的似然值都非常小，几乎没有区别
    else:
        for i in range(M):
            p[i] /= s  # 将每个指数化似然值除以总和s，这样所有概率加起来就等于 1，符合概率的定义，得到归一化的后验概率分布

    # 计算累积分布函数 (CDF)
    cdf = np.empty(M, dtype=np.float64)
    acc = 0.0
    for i in range(M):
        acc += p[i]
        cdf[i] = acc

    r_grid_m = np.empty(M, dtype=np.float64)
    for i in range(M):
        r_grid_m[i] = (r0 + step * i) * bin_to_m  # 每个候选距离的 bin 索引rb = r0 + step * i，乘以转换系数bin_to_m，转换成了以米为单位的实际距离
    
    # r_grid_m:候选距离网格（米）,提供每个候选距离的实际位置;cdf:后验累积分布函数;p:后验概率分布
    return r_grid_m, cdf, p


@njit(cache=True, fastmath=True)
def peak_valley_bounds_numba(pdf, peak_idx):
    """
    从后验概率分布中，找到每个峰值的左右谷底，确定这个峰值的独立范围
    pdf:后验概率分布数组;peak_idx:峰值在 pdf 数组中的索引
    l:峰值的左边界索引（谷底位置）;r:峰值的右边界索引（谷底位置）
    """
    n = pdf.size  # 获取后验概率分布数组的长度，也就是候选距离的数量M
    l = peak_idx  # 将左边界初始化为峰值本身的位置
    
    # 向左寻找谷底,从峰值向左走，直到遇到第一个比当前点高的点,当循环结束时，l就是峰值左边的第一个谷底位置
    while l - 1 >= 0 and pdf[l - 1] <= pdf[l]:
        l -= 1
    r = peak_idx  # 将右边界初始化为峰值本身的位置

    # 向右寻找谷底,当循环结束时，r就是峰值右边的第一个谷底位置
    while r + 1 < n and pdf[r + 1] <= pdf[r]:
        r += 1
    
    return l, r  # 返回找到的左右谷底边界


@njit(cache=True, fastmath=True)
def cdf_lookup_numba(r_grid_m, cdf_grid, x):
    """Step-CDF lookup: return accumulated posterior mass for r <= x."""
    """
    给定严格升序排列的候选距离数组r_grid_m和对应的累积分布函数数组cdf_grid，返回距离小于等于查询值x的累积后验概率

    r_grid_m:升序排列的候选距离网格，单位：米（由compute_ll_grid_numba生成）
    cdf_grid:与r_grid_m一一对应的累积分布函数值，cdf_grid[i] = P(r ≤ r_grid_m[i])
    x:待查询的距离值
    """
    n = r_grid_m.size
    if n == 0:
        return 0.0
    if x < r_grid_m[0]:
        return 0.0
    if x >= r_grid_m[n - 1]:
        return 1.0

    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        
        # 如果r_grid_m[mid] ≤ x,说明最后一个≤x 的元素一定在mid的右侧（包括mid），因此将左边界移到mid + 1
        if r_grid_m[mid] <= x:
            lo = mid + 1
        
        # 如果r_grid_m[mid] > x,说明最后一个≤x 的元素一定在mid的左侧，因此将右边界移到mid
        else:
            hi = mid
        # 循环终止条件：lo == hi，此时lo指向第一个大于x的元素的索引

    idx = lo - 1  # 因为lo是第一个大于x的元素索引，所以lo-1就是最后一个≤x 的元素的索引
    if idx < 0:
        return 0.0
    v = cdf_grid[idx]
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


@njit(cache=True, fastmath=True)
def logit_numba(p):
    """Numerically stable logit for numba code."""
    if p < 1e-6:
        p = 1e-6
    elif p > 1.0 - 1e-6:
        p = 1.0 - 1e-6
    
    return math.log(p / (1.0 - p))


def posterior_entropy_weight(pdf: np.ndarray, w_min: float = 0.2) -> float:
    """
    Convert posterior dispersion into an adaptive update weight.
    Sharp posteriors get weights near 1; broad posteriors get weights near w_min.
    """
    """
    pdf:合并后的距离后验概率分布数组（就是main函数中的ray_p）
    w_min:最小更新权重，默认 0.2。即使是最不确定的测量，也至少保留 20% 的更新强度
    """
    p = np.asarray(pdf, dtype=np.float64)
    n = int(p.size)  # n是后验分布的采样点数量（默认单峰 81 个，双峰最多 162 个）
    if n <= 1:
        return 1.0

    s = float(p.sum())
    if s <= 0.0 or not np.isfinite(s):
        return float(w_min)

    p = p / s  # 将后验分布归一化，确保所有概率值的总和为 1
    valid = p > 0.0  # 找出所有大于 0 的概率值
    if not np.any(valid):
        return float(w_min)

    entropy = -float(np.sum(p[valid] * np.log(p[valid])))  # 信息熵的标准计算公式,计算后验分布的熵
    
    # 熵的归一化
    entropy_norm = entropy / max(math.log(float(n)), 1e-12)
    entropy_norm = max(0.0, min(1.0, entropy_norm))

    w_min = max(0.0, min(1.0, float(w_min)))
    
    # 当entropy_norm=0（完全确定）：权重 = w_min + (1-w_min)*1 = 1.0
    # 当entropy_norm=1（完全不确定）：权重 = w_min + (1-w_min)*0 = w_min
    # 中间值线性插值
    return w_min + (1.0 - w_min) * (1.0 - entropy_norm)


@njit(cache=True, fastmath=True)
def dda_update_dense(
    Lgrid, origin, d, end_dist, mins, voxel,
    nx, ny, nz, rm_arr, rp_arr, K,
    Lmin, Lmax, dL_free, dL_occ, range_model_is_range,
):
    """
    沿着一条相机/SPAD ray 穿过 3D voxel grid，把 histogram 推出的距离区间融合成 occupancy log-odds 地图
    Lgrid:三维对数几率栅格数组，形状为(nx, ny, nz)，存储每个体素的占用概率;origin:相机在世界坐标系中的原点坐标 (x, y, z);d:射线的单位方向向量
    end_dist:传感器的最大有效测量距离;mins:栅格地图的最小世界坐标 (min_x, min_y, min_z);voxel:体素的边长（单位：米）
    nx, ny, nz:栅格地图在 x、y、z 三个方向上的体素数量;rm_arr, rp_arr:所有峰值的左右边界数组;K:有效峰值的数量
    Lmin, Lmax:对数几率的最小值和最大值，防止数值溢出;dL_free:自由空间的对数几率更新量;dL_occ:占用空间的对数几率更新量
    range_model_is_range:距离模型标志，和之前生成三维点时的参数一致

    输入：
    一条 ray 的起点 origin
    一条 ray 的方向 d
    这条 ray 上可能有物体的距离区间 [rm, rp]
    当前 3D 地图 Lgrid

    1.把 ray 起点和终点转换成 voxel 坐标
    2.用 DDA 沿 ray 遍历 voxel
    3.计算每个 voxel 沿 ray 的距离 rho
    4.根据 rho 和 occupied intervals 更新 log-odds
        如果 voxel 在物体前面 rho < rm 则更新为 free：Lgrid[x, y, z] += dL_free
        如果 voxel 落在物体区间里 rm <= rho <= rp 则更新为 occupied：Lgrid[x, y, z] += dL_occ
        如果有多个返回峰，中间区域也会更新为 free：第一个物体后面、第二个物体前面 -> free
        最后一个 occupied interval 后面的空间不更新
    """
    # 将相机原点转换为体素坐标
    v0x = (origin[0] - mins[0]) / voxel
    v0y = (origin[1] - mins[1]) / voxel
    v0z = (origin[2] - mins[2]) / voxel

    # 将射线终点转换为体素坐标
    v1x = (origin[0] + d[0] * end_dist - mins[0]) / voxel
    v1y = (origin[1] + d[1] * end_dist - mins[1]) / voxel
    v1z = (origin[2] + d[2] * end_dist - mins[2]) / voxel

    # math.floor函数会向下取整，得到点所在的体素的整数索引
    x = int(math.floor(v0x))
    y = int(math.floor(v0y))
    z = int(math.floor(v0z))
    x1 = int(math.floor(v1x))
    y1 = int(math.floor(v1y))
    z1 = int(math.floor(v1z))

    # 计算射线在体素坐标系中的方向向量
    dx = v1x - v0x  # 这条 ray 在 grid 的 x 方向一共走了多少个 voxel 单位
    dy = v1y - v0y
    dz = v1z - v0z

    # 计算 DDA 步进方向
    # 如果dx > 0，说明射线在 x 方向上向右移动，每次步进+1;如果dx < 0，说明射线在 x 方向上向左移动，每次步进-1;如果dx = 0，说明射线在 x 方向上不移动，步进0
    sx = 1 if dx > 0.0 else (-1 if dx < 0.0 else 0)
    sy = 1 if dy > 0.0 else (-1 if dy < 0.0 else 0)
    sz = 1 if dz > 0.0 else (-1 if dz < 0.0 else 0)

    if dx == 0.0:  # 说明说明射线在 x 方向完全不动，不会穿过任何 x 边界，设置tMaxX和tDeltaX为无穷大，意思是永远不要优先选择跨 x 边界
        tMaxX = 1e30
        tDeltaX = 1e30
    else:
        # 如果射线往 x 正方向走，当前 voxel 是 x，下一个 x 边界是 x + 1；如果射线往 x 负方向走，当前 voxel 是 x，下一个 x 边界是 x
        next_boundary = (x + 1) if sx > 0 else x
        
        # 射线ray参数t是射线上的归一化位置参数，在 DDA 里射线可以写成pos(t) = start + t * (end - start)
        # t = 0表示在射线起点 start；t = 1表示在射线终点 end；t = 0.5表示在起点和终点的正中间；t = 0.25表示走完整条线段的 25%
        # 所以这里的 t 表示从起点到终点这段 ray，已经走了多少比例
        # 所以 tMaxX 表示从起点出发，走到第一个 x 边界时，t 等于多少，即沿整条 ray 走百分之多少的长度，就会碰到第一个 x 边界
        tMaxX = (next_boundary - v0x) / dx
        
        # tDeltaX 表示每跨过一个 x voxel 边界，t 要增加多少
        tDeltaX = 1.0 / abs(dx)

    if dy == 0.0:
        tMaxY = 1e30
        tDeltaY = 1e30
    else:
        next_boundary = (y + 1) if sy > 0 else y
        tMaxY = (next_boundary - v0y) / dy
        tDeltaY = 1.0 / abs(dy)

    if dz == 0.0:
        tMaxZ = 1e30
        tDeltaZ = 1e30
    else:
        next_boundary = (z + 1) if sz > 0 else z
        tMaxZ = (next_boundary - v0z) / dz
        tDeltaZ = 1.0 / abs(dz)

    max_steps = 20000
    for _ in range(max_steps):
        # x, y, z 是当前 voxel 的索引，x1, y1, z1 是 ray 终点所在 voxel 的索引，如果当前 voxel 已经是终点 voxel，就停止遍历
        if x == x1 and y == y1 and z == z1:
            break

        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            # 把 voxel 索引转回真实世界坐标，得到当前 voxel 的中心点坐标，单位是米
            cx = mins[0] + voxel * (x + 0.5)
            cy = mins[1] + voxel * (y + 0.5)
            cz = mins[2] + voxel * (z + 0.5)

            if range_model_is_range:
                # 点积：rho = d · (voxel_center - origin)，d 是单位方向向量，voxel_center - origin 是从传感器指向当前 voxel center 的向量
                # 点积的结果就是当前 voxel center 在 ray 方向上的距离，因为 d 已经归一化，所以 rho 的单位是米
                rho = d[0] * (cx - origin[0]) + d[1] * (cy - origin[1]) + d[2] * (cz - origin[2])
            else:
                rho = cz

            if rho > 0.0 and rho <= end_dist and K > 0:
                for i in range(K):
                    rm = rm_arr[i]
                    rp = rp_arr[i]
                    if rho < rm:
                        L = Lgrid[x, y, z] + dL_free
                        if L < Lmin:
                            L = Lmin
                        if L > Lmax:
                            L = Lmax
                        Lgrid[x, y, z] = L
                        break
                    if rho <= rp:
                        L = Lgrid[x, y, z] + dL_occ
                        if L < Lmin:
                            L = Lmin
                        if L > Lmax:
                            L = Lmax
                        Lgrid[x, y, z] = L
                        break
                    
                    # 处理多个 peak / 多个 occupied interval 的情况
                    if i + 1 < K:
                        nrm = rm_arr[i + 1]
                        if rho > rp and rho < nrm:
                            L = Lgrid[x, y, z] + dL_free
                            if L < Lmin:
                                L = Lmin
                            if L > Lmax:
                                L = Lmax
                            Lgrid[x, y, z] = L
                            break

        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                x += sx
                tMaxX += tDeltaX
            else:
                z += sz
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                y += sy
                tMaxY += tDeltaY
            else:
                z += sz
                tMaxZ += tDeltaZ


@njit(cache=True, fastmath=True)
def dda_update_dense_cdf(
    Lgrid, origin, d, end_dist, mins, voxel,
    nx, ny, nz, r_grid_m, cdf_grid, n_hyp,
    Lmin, Lmax, logit_p0, p_occ, p_free, update_weight, range_model_is_range,
):
    """DDA ray update using CDF-marginalized voxel occupancy probability."""
    """
    沿着一条相机/SPAD ray 穿过 3D voxel grid，把完整的距离后验CDF融合成 occupancy log-odds 地图
    Lgrid:三维对数几率栅格数组，形状为(nx, ny, nz)，存储每个体素的占用概率;origin:相机在世界坐标系中的原点坐标 (x, y, z);d:射线的单位方向向量
    end_dist:传感器的最大有效测量距离;mins:栅格地图的最小世界坐标 (min_x, min_y, min_z);voxel:体素的边长（单位：米）
    nx, ny, nz:栅格地图在 x、y、z 三个方向上的体素数量;r_grid_m:升序排列的候选距离网格，单位：米;cdf_grid:与r_grid_m一一对应的累积分布函数值
    n_hyp:距离假设点的总数量;Lmin, Lmax:对数几率的最小值和最大值，防止数值溢出;logit_p0:先验占用概率的对数几率值
    p_occ:物体在体素内部时的观测占用概率;p_free:物体在体素后面时的观测占用概率;update_weight:基于后验熵的自适应更新权重
    range_model_is_range:距离模型标志，和之前生成三维点时的参数一致

    输入：
    一条 ray 的起点 origin
    一条 ray 的方向 d
    这条 ray 上完整的距离后验累积分布函数 (r_grid_m, cdf_grid)
    当前 3D 地图 Lgrid

    1.把 ray 起点和终点转换成 voxel 坐标
    2.用 DDA 沿 ray 遍历 voxel
    3.计算每个 voxel 沿 ray 的距离 rho
    4.根据 rho 和完整后验CDF计算体素占用概率并更新 log-odds
        计算体素沿射线的前后边界 a_v 和 b_v
        查询CDF得到物体在体素前面的概率 Fa 和物体在体素前面或内部的概率 Fb
        推导物体在体素内部的概率 hit = Fb - Fa 和物体在体素后面的概率 behind = 1 - Fb
        概率加权得到体素的单次观测占用概率 P_v = 0.5*Fa + p_occ*hit + p_free*behind
        用贝叶斯规则更新对数几率：L += update_weight * (logit(P_v) - logit_p0)
        对更新后的对数几率进行数值钳位，防止溢出
    """
    # 将相机原点（射线起点）从世界坐标系转换为连续的体素坐标
    # origin：相机在世界坐标系中的位置，代码中固定为(0,0,0);mins：栅格地图在世界坐标系中的最小边界，默认是(-5, -5, 0);voxel：体素的边长，默认是0.1米
    v0x = (origin[0] - mins[0]) / voxel
    v0y = (origin[1] - mins[1]) / voxel
    v0z = (origin[2] - mins[2]) / voxel

    # 计算射线在最大有效距离处的终点，并转换为连续的体素坐标
    # d：射线的单位方向向量，表示这条射线在三维空间中的指向;end_dist：传感器的最大有效测量距离，默认是5.0米
    v1x = (origin[0] + d[0] * end_dist - mins[0]) / voxel
    v1y = (origin[1] + d[1] * end_dist - mins[1]) / voxel
    v1z = (origin[2] + d[2] * end_dist - mins[2]) / voxel

    # 将连续的体素坐标向下取整，得到射线起点和终点所在的体素整数索引
    x = int(math.floor(v0x))
    y = int(math.floor(v0y))
    z = int(math.floor(v0z))
    x1 = int(math.floor(v1x))
    y1 = int(math.floor(v1y))
    z1 = int(math.floor(v1z))

    # 计算射线在体素坐标系中的总位移向量
    dx = v1x - v0x
    dy = v1y - v0y
    dz = v1z - v0z

    # 确定射线在每个轴上的步进方向
    sx = 1 if dx > 0.0 else (-1 if dx < 0.0 else 0)
    sy = 1 if dy > 0.0 else (-1 if dy < 0.0 else 0)
    sz = 1 if dz > 0.0 else (-1 if dz < 0.0 else 0)

    # DDA的核心思想：一条射线穿过体素网格时，每次只会穿过一个体素的边界。我们只需要每次找到最近的那个边界，走过去，然后重复这个过程，直到到达射线终点
    # 把射线用参数t进行参数化，pos(t)=start+t·(end−start)，计算走到下一个 x 边界需要多少t？走到下一个 y 边界需要多少t？走到下一个 z 边界需要多少t？
    # 哪个需要的t最小，就先走到哪个边界
    if dx == 0.0:
        tMaxX = 1e30
        tDeltaX = 1e30
    else:
        next_boundary = (x + 1) if sx > 0 else x
        
        # 从射线起点到下一个 x 边界的距离 / 射线从起点到终点在 x 方向的总位移 = 从射线起点出发，走到下一个 x 边界所需要的t值
        tMaxX = (next_boundary - v0x) / dx
        tDeltaX = 1.0 / abs(dx)  # 每跨过一个 x 体素需要的t

    if dy == 0.0:
        tMaxY = 1e30
        tDeltaY = 1e30
    else:
        next_boundary = (y + 1) if sy > 0 else y
        tMaxY = (next_boundary - v0y) / dy
        tDeltaY = 1.0 / abs(dy)

    if dz == 0.0:
        tMaxZ = 1e30
        tDeltaZ = 1e30
    else:
        next_boundary = (z + 1) if sz > 0 else z
        tMaxZ = (next_boundary - v0z) / dz
        tDeltaZ = 1.0 / abs(dz)

    max_steps = 20000
    for _ in range(max_steps):
        if x == x1 and y == y1 and z == z1:
            break  # 当当前体素就是射线终点所在的体素时，退出循环

        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            # 计算体素中心的世界坐标，体素索引x对应的是体素的左边界坐标，体素 x 的 x 坐标范围是：[mins[0] + x*voxel, mins[0] + (x+1)*voxel)
            # 所以体素的中心坐标就是 mins[0] + voxel*(x+0.5)
            cx = mins[0] + voxel * (x + 0.5)
            cy = mins[1] + voxel * (y + 0.5)
            cz = mins[2] + voxel * (z + 0.5)

            if range_model_is_range:
                # 体素中心到相机原点的沿射线距离
                rho = d[0] * (cx - origin[0]) + d[1] * (cy - origin[1]) + d[2] * (cz - origin[2])
            else:
                rho = cz

            if rho > 0.0 and rho <= end_dist and n_hyp > 0:
                # 计算体素沿射线方向的前后边界距离
                a_v = rho - 0.5 * voxel
                b_v = rho + 0.5 * voxel
                
                # 查询 CDF 得到累积概率
                Fa = cdf_lookup_numba(r_grid_m, cdf_grid, a_v)  # Fa = P(r ≤ a_v)：物体在体素前面的概率，体素被遮挡，无法得知状态，保持先验概率0.5
                Fb = cdf_lookup_numba(r_grid_m, cdf_grid, b_v)  # Fb = P(r ≤ b_v)：物体在体素前面或内部的概率

                hit = Fb - Fa  # 物体正好在体素内部，概率为hit = Fb - Fa，此时体素被物体占据，所以占用概率为p_occ
                if hit < 0.0:
                    hit = 0.0
                behind = 1.0 - Fb  # 物体在体素后面，概率为behind = 1 - Fb，体素是自由的，占用概率为p_free
                if behind < 0.0:
                    behind = 0.0

                P_v = 0.5 * Fa + p_occ * hit + p_free * behind  # 计算体素的单次观测占用概率
                if P_v < 1e-6:
                    P_v = 1e-6
                if P_v > 1.0 - 1e-6:
                    P_v = 1.0 - 1e-6

                # 贝叶斯对数几率更新
                L = Lgrid[x, y, z] + update_weight * (logit_numba(P_v) - logit_p0)
                if L < Lmin:
                    L = Lmin
                if L > Lmax:
                    L = Lmax
                Lgrid[x, y, z] = L

        if tMaxX < tMaxY:
            if tMaxX < tMaxZ:
                x += sx
                tMaxX += tDeltaX
            else:
                z += sz
                tMaxZ += tDeltaZ
        else:
            if tMaxY < tMaxZ:
                y += sy
                tMaxY += tDeltaY
            else:
                z += sz
                tMaxZ += tDeltaZ


def write_occ_slice(path: str, Lgrid: np.ndarray, z_slice: float, z_min: float, voxel: float, p0: float) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    nz = Lgrid.shape[2]
    izs = int(math.floor((z_slice - z_min) / voxel))
    if not (0 <= izs < nz):
        print(f"[WARN] z_slice={z_slice} is outside grid; skip slice output")
        return

    Ls = Lgrid[:, :, izs].astype(np.float64)
    slice_img = 1.0 / (1.0 + np.exp(-Ls))
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_img.T, origin="lower", vmin=0.0, vmax=1.0)
    plt.title(f"Occupancy slice at z={z_slice:.2f} m")
    plt.colorbar()
    plt.savefig(path, dpi=200)
    plt.close()
    print("saved", path)


def write_occupied_ply(
    path: str,
    Lgrid: np.ndarray,
    mins: np.ndarray,
    voxel: float,
    max_out: int,
    min_prob: float,
) -> int:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    min_log_odds = logit(min_prob)
    xs, ys, zs = np.where(Lgrid >= min_log_odds)
    if max_out > 0 and xs.size > max_out:
        rng = np.random.default_rng(0)
        sel = rng.choice(xs.size, size=max_out, replace=False)
        xs = xs[sel]
        ys = ys[sel]
        zs = zs[sel]

    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xs.size}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(xs.size):
            ix = int(xs[i])
            iy = int(ys[i])
            iz = int(zs[i])
            L = float(Lgrid[ix, iy, iz])
            p = 1.0 / (1.0 + math.exp(-L))
            g = int(max(0, min(255, round(p * 255.0))))
            cx = mins[0] + voxel * (ix + 0.5)
            cy = mins[1] + voxel * (iy + 0.5)
            cz = mins[2] + voxel * (iz + 0.5)
            f.write(f"{cx:.6f} {cy:.6f} {cz:.6f} {g} {g} {g}\n")
    print("saved", path, "N=", xs.size, "min_prob=", min_prob)
    return int(xs.size)


def write_surface_ply(path: str, points: np.ndarray) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(points.shape[0]):
            x, y, z, conf = points[i]
            g = int(max(0, min(255, round(80.0 + 175.0 * conf))))
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {g} {g} {g}\n")
    print("saved", path, "N=", points.shape[0])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ, help="SPAD .npz path containing spad_hist")
    ap.add_argument("--slice-out", default=DEFAULT_SLICE_OUT)
    ap.add_argument("--ply-out", default=DEFAULT_PLY_OUT)
    ap.add_argument("--surface-out", default=DEFAULT_SURFACE_OUT)
    ap.add_argument("--tmax-ns", type=float, default=100.0)  # 激光的最大飞行时间，单位：纳秒 (ns)，默认值100ns，最大测距距离15米
    ap.add_argument("--voxel", type=float, default=0.10)  # 体素大小，单位：米
    ap.add_argument("--range_max", type=float, default=5.0)  # 局部地图在 x/y 方向的半长，单位：米，地图范围：x∈[-5,5], y∈[-5,5]
    ap.add_argument("--z_min", type=float, default=0.0)  # 地图 z 方向的最小值，单位：米，一般设为地面高度
    ap.add_argument("--z_max", type=float, default=5.0)  # 地图 z 方向的最大值，单位：米，一般设为天花板高度
    
    # 最大使用的射线数量，0 表示使用所有通过--peak_thr筛选的射线
    ap.add_argument("--max_rays", type=int, default=0, help="subsample rays, 0=use all candidate rays")
    ap.add_argument("--peak_thr", type=float, default=5.0)  # 射线筛选阈值，直方图最大值小于这个值的射线会被直接跳过
    ap.add_argument("--Wr_bin", type=int, default=12)  # 每个峰值周围的局部窗口半宽，单位：bin，只计算这个范围内的距离后验
    ap.add_argument("--M", type=int, default=81)  # 每个峰值周围采样的距离假设点数量
    ap.add_argument("--p_occ", type=float, default=0.70)
    ap.add_argument("--p_free", type=float, default=0.35)
    ap.add_argument("--p0", type=float, default=0.50)  # 体素初始先验概率
    ap.add_argument("--Lmin", type=float, default=-5.0)  # 对数几率的最小值，对应概率≈0.007
    ap.add_argument("--Lmax", type=float, default=5.0)  # 对数几率的最大值，对应概率≈0.993
    ap.add_argument("--z_slice", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--win_half", type=int, default=25)  # 直方图截取半宽，单位：bin，计算似然时只截取峰值周围这么大的窗口，必须大于--Wr_bin
    ap.add_argument("--sigma_bins", type=float, default=2.0)  # 高斯 IRF 的标准差，单位：bin
    
    # 距离计算模型："z"：仅用 z 坐标作为距离（2.5D 俯视模式）；"range"：标准 3D 射线距离
    # 俯视场景（比如无人机）用 "z"，普通 3D 场景（比如地面机器人）必须改成 "range"
    ap.add_argument("--range_model", choices=["range", "z"], default="range")
    ap.add_argument("--max_peaks", type=int, default=2)  # 每条射线最多检测多少个峰
    
    # 匹配追踪 (MP) 峰值检测阈值，卷积响应小于这个值的峰不会被检测到，控制峰值检测的灵敏度
    # 调大：只检测明显的峰，减少假阳性；调小：检测更多弱峰，可能引入噪声
    ap.add_argument("--mp_thr", type=float, default=5.0)
    
    # 峰值支持半径，单位：bin，0 表示自动设为 4×--sigma_bins，检测到一个峰后，会把这个半径内的直方图置零
    ap.add_argument("--mp_support", type=int, default=0, help="support radius in bins, 0=auto(4*sigma)")
    ap.add_argument("--occ_wmax_vox", type=float, default=2.5)  # 旧 interval update（区间更新）参数；CDF 更新不再使用
    
    # 导出占据体素的最小概率阈值，只有概率大于等于这个值的体素才会被导出
    ap.add_argument(
        "--export-min-prob",
        type=float,
        default=0.55,
        help="Only export voxels with occupancy probability at least this value.",
    )
    ap.add_argument("--max_out", type=int, default=0)  # 导出 PLY 点云的最大点数，0 表示不限制，当体素太多时，限制点数防止文件过大
    
    # 如果相机不是 NYU 标准内参，就手动指定
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--profile", action="store_true", help="Print timing breakdown for major pipeline stages.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print("loading npz ...")
    data = np.load(args.npz)
    if "spad_hist" not in data:
        raise KeyError(f"{args.npz} does not contain a 'spad_hist' array")
    spad_hist = np.asarray(data["spad_hist"])
    if spad_hist.ndim != 3:
        raise ValueError(f"spad_hist must have shape (H, W, T), got {spad_hist.shape}")

    H, W, B = spad_hist.shape
    hist_data = spad_hist.reshape(H * W, B).astype(np.float64)
    bin_to_m = C_M_PER_S * (args.tmax_ns * 1e-9) / 2.0 / float(B)
    print(f"spad_hist shape={spad_hist.shape}, dtype={spad_hist.dtype}")
    print(f"bin_to_m={bin_to_m:.8f} m/bin, tmax={args.tmax_ns:g} ns")

    default_fx, default_fy, default_cx, default_cy = scaled_nyu_intrinsics(W, H)
    fx = default_fx if args.fx is None else args.fx
    fy = default_fy if args.fy is None else args.fy
    cx = default_cx if args.cx is None else args.cx
    cy = default_cy if args.cy is None else args.cy
    print(f"intrinsics fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")

    # 生成两个和图像一样大的表格，一个表格里全是像素的列号，另一个表格里全是像素的行号
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    
    # 从像素坐标到归一化图像坐标的转换，是针孔相机模型的核心公式
    # 平移：将坐标原点从图像左上角（像素坐标的原点）移动到图像中心（主点）
    # 缩放：将像素单位转换为归一化单位
    x_n_all = ((u_grid.reshape(-1) - cx) / fx).astype(np.float64)
    y_n_all = ((v_grid.reshape(-1) - cy) / fy).astype(np.float64)

    voxel = float(args.voxel)
    
    # 定义地图的边界
    mins = np.array([-args.range_max, -args.range_max, args.z_min], dtype=np.float64)
    maxs = np.array([args.range_max, args.range_max, args.z_max], dtype=np.float64)
    
    # 计算每个方向的体素数量，maxs - mins计算每个方向的总长度，除以体素大小，得到每个方向需要多少个体素，np.ceil()向上取整，保证整个场景都被网格覆盖
    dims = np.ceil((maxs - mins) / voxel).astype(np.int64)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    print("grid dims =", (nx, ny, nz), "voxel =", voxel)
    
    # 初始化对数几率网格，这就是全局 3D 占据地图，是整个算法的最终输出，每个元素对应一个体素的对数几率值，初始化为 0，对应概率 p=0.5，
    # 表示所有体素初始状态都是完全不确定的
    Lgrid = np.zeros((nx, ny, nz), dtype=np.float32)

    mx = hist_data.max(axis=1)
    cand = np.where(mx >= args.peak_thr)[0]  # 候选射线列表，里面的每个数字都是原始数据中对应射线的索引
    print("candidate rays:", cand.size)
    if cand.size == 0:
        print("No rays pass threshold.")
        return
    if args.max_rays > 0 and cand.size > args.max_rays:
        sel = np.linspace(0, cand.size - 1, args.max_rays).astype(np.int64)
        rays = cand[sel]  # 均匀子采样
    else:
        rays = cand
    # 最终得到的rays列表，就是后面要进入核心循环处理的所有射线
    print("using rays:", rays.size)

    origin = np.zeros(3, dtype=np.float64)  # 激光雷达在局部地图坐标系中的位置，设为 (0,0,0)
    end_dist = float(args.range_max)  # 每条射线的最大遍历距离，单位：米
    range_model_is_range = args.range_model == "range"
    
    # CDF update（累积分布函数更新）使用完整后验概率计算每个 voxel（体素）的单次观测概率
    logit_p0 = float(logit(args.p0))
    
    # 对数几率的上下限
    Lmin = float(args.Lmin)
    Lmax = float(args.Lmax)
    
    # 峰值检测的支持半径，单位：bin，检测到一个峰后，会把这个峰周围support个 bin 范围内的直方图全部置零，避免重复检测同一个峰
    support = int(args.mp_support) if args.mp_support > 0 else int(np.ceil(4.0 * args.sigma_bins))
    
    # 两个峰值之间的最小距离，单位：bin，如果新检测到的峰和已经检测到的峰之间的距离小于min_sep，就认为是同一个峰，直接丢弃
    min_sep = support

    surface_points = []  # 初始化表面点云列表
    profile_enabled = bool(args.profile)
    t_peak = 0.0
    t_posterior = 0.0
    t_merge = 0.0
    t_surface = 0.0
    t_dda = 0.0
    t_output = 0.0
    rays_with_peaks = 0
    rays_fused = 0
    posterior_evals = 0
    t0 = time.time()
    for k, idx in enumerate(rays):  # 遍历每条射线
        h = hist_data[idx]  # 取出它的光子直方图
        
        # 检测直方图中的所有显著峰值
        # peaks：检测到的峰值位置列表，每个元素是峰值所在的时间 bin 索引
        # peak_scores：对应峰值的置信度分数列表，分数越高表示这个峰越可能是真实的物体反射
        _tp = time.perf_counter()
        peaks, peak_scores = mp_find_peaks(
            h,
            max_peaks=args.max_peaks,
            sigma_bins=args.sigma_bins,
            score_thr=args.mp_thr,
            support_radius=support,
            min_sep=min_sep,
        )
        t_peak += time.perf_counter() - _tp
        if len(peaks) == 0:
            continue
        rays_with_peaks += 1

        peak_posteriors_r = []  # 存储每个峰值对应的距离网格数组
        peak_posteriors_p = []  # 存储每个峰值对应的后验概率分布数组，与peak_posteriors_r一一对应
        profile_points = []  # 存储这条射线所有的表面点信息，每个元素是一个元组(peak_depth, peak_conf)，表示在peak_depth米处有一个表面点，置信度为peak_conf

        total_peak_score = float(sum(max(float(s), 0.0) for s in peak_scores))  # 计算总分数
        if total_peak_score <= 0.0:
            total_peak_score = float(len(peaks))  # 如果总分数为 0，就用峰值的数量代替总分数，这样每个峰值的权重就相等

        # pk：当前峰值所在的时间 bin 索引;pk_score：当前峰值的匹配追踪分数，分数越高表示这个峰越可能是真实的物体反射，而非噪声
        for pk, pk_score in zip(peaks, peak_scores):
            # r_grid_m：距离数组，每个元素对应一个候选距离 r，单位为米；pdf_grid：距离后验概率密度函数，pdf_grid[i]：表示墙在r_grid_m[i]米处的概率
            _tp = time.perf_counter()
            r_grid_m, _, pdf_grid = compute_ll_grid_numba(
                h,
                int(pk),
                int(args.Wr_bin),  # 峰值周围的局部窗口半宽，单位 bin，默认值为12
                int(args.M),  # 采样点数，默认值为81，在 ±12 个 bin 的范围内采样 81 个点，也就是每个 bin 采样约 3.4 个点
                int(args.win_half),  # 直方图截取半宽，单位 bin，默认值为25，计算似然时只截取峰值周围 ±25 个 bin 的直方图
                float(args.sigma_bins),
                float(args.tau),
                float(bin_to_m),
            )
            t_posterior += time.perf_counter() - _tp
            posterior_evals += 1
            peak_weight = max(float(pk_score), 0.0) / total_peak_score  # 计算当前峰值的权重
            pk_local = int(np.argmax(pdf_grid))  # 返回pdf_grid数组中最大值所在的索引
            peak_depth = float(r_grid_m[pk_local])  # 用刚才找到的概率峰值索引，去距离网格中取出对应的实际距离
            peak_conf = float(pdf_grid[pk_local])  # 取出概率峰值处的概率值
            if peak_depth <= 0.0 or peak_depth > end_dist:
                continue

            peak_posteriors_r.append(r_grid_m)  # 将当前峰值的距离网格存入列表
            peak_posteriors_p.append(pdf_grid * peak_weight)  # 将当前峰值的后验概率分布乘以权重后存入列表。这一步实现了按置信度加权
            profile_points.append((peak_depth, peak_conf))  # 将当前峰值对应的表面点存入列表，后续会导出为 PLY 文件

        if len(peak_posteriors_r) == 0:
            continue

        _tp = time.perf_counter()
        # 如果只检测到一个有效峰值，直接使用这个峰值的距离网格和加权后的概率分布
        if len(peak_posteriors_r) == 1:
            ray_r = np.asarray(peak_posteriors_r[0], dtype=np.float64)
            ray_p = np.asarray(peak_posteriors_p[0], dtype=np.float64)
        else:
            # 拼接所有峰值的后验数组,np.concatenate将所有峰值的距离网格和加权后的概率分布分别拼接成两个一维数组
            # 例如：如果检测到 2 个峰值，每个峰值有 81 个采样点，那么拼接后的ray_r和ray_p的长度都是 162
            ray_r = np.concatenate([np.asarray(arr, dtype=np.float64) for arr in peak_posteriors_r])
            ray_p = np.concatenate([np.asarray(arr, dtype=np.float64) for arr in peak_posteriors_p])
            
            # 按距离升序排序
            order = np.argsort(ray_r)
            ray_r = ray_r[order]
            ray_p = ray_p[order]

        total_prob = float(ray_p.sum())
        if total_prob <= 0.0:
            continue
        ray_p /= total_prob  #  概率归一化
        ray_cdf = np.cumsum(ray_p)  # 计算累积分布函数 (CDF)
        ray_cdf[-1] = 1.0
        update_weight = posterior_entropy_weight(ray_p, w_min=0.2)
        t_merge += time.perf_counter() - _tp

        d = np.array([float(x_n_all[idx]), float(y_n_all[idx]), 1.0], dtype=np.float64)  # 构建这条射线的方向向量
        nrm = np.linalg.norm(d)  # 计算d的模长
        if nrm <= 1e-12:
            continue
        d /= nrm  # 归一化得到单位方向向量,代表这条射线在三维空间中的方向向量

        _tp = time.perf_counter()
        for peak_depth, peak_conf in profile_points:
            if peak_depth <= 0.0 or peak_depth > end_dist:
                continue
            if range_model_is_range:
                p3 = d * peak_depth  # 三维点坐标 = 单位方向向量 × 直线距离
            else:
                scale = peak_depth / max(d[2], 1e-12)
                p3 = d * scale
            surface_points.append((float(p3[0]), float(p3[1]), float(p3[2]), float(peak_conf)))  # surface_points列表:带置信度的三维点云
        t_surface += time.perf_counter() - _tp

        # 把当前这一条 ray 的 CDF 后验结果写进 3D occupancy map
        _tp = time.perf_counter()
        dda_update_dense_cdf(
            Lgrid, origin, d, end_dist, mins, voxel,
            nx, ny, nz, ray_r, ray_cdf, int(ray_r.size),
            Lmin, Lmax, logit_p0, float(args.p_occ), float(args.p_free), float(update_weight),
            range_model_is_range,
        )
        t_dda += time.perf_counter() - _tp
        rays_fused += 1

        # 每处理 5000 条 ray 打印一次进度
        if (k + 1) % 5000 == 0:
            print("processed rays:", k + 1, "/", rays.size, "elapsed(s)=", round(time.time() - t0, 1))

    print("done. elapsed(s)=", round(time.time() - t0, 2))
    _tp = time.perf_counter()
    write_occ_slice(args.slice_out, Lgrid, args.z_slice, args.z_min, voxel, args.p0)
    if surface_points:
        write_surface_ply(args.surface_out, np.asarray(surface_points, dtype=np.float32))
    else:
        write_surface_ply(args.surface_out, np.empty((0, 4), dtype=np.float32))
    occupied_count = write_occupied_ply(
        args.ply_out,
        Lgrid,
        mins,
        voxel,
        int(args.max_out),
        float(args.export_min_prob),
    )
    t_output += time.perf_counter() - _tp
    active_count = int(np.count_nonzero(np.abs(Lgrid) > 1e-6))
    print("active voxels:", active_count)
    print("exported occupied voxels:", occupied_count)
    if profile_enabled:
        total_elapsed = max(time.time() - t0, 1e-12)
        measured = t_peak + t_posterior + t_merge + t_surface + t_dda + t_output
        other = max(0.0, total_elapsed - measured)
        print("profile timing breakdown:")
        print(f"  rays total: {rays.size}, with peaks: {rays_with_peaks}, fused: {rays_fused}, posterior evals: {posterior_evals}")
        print(f"  peak detection: {t_peak:.3f}s ({100.0 * t_peak / total_elapsed:.1f}%)")
        print(f"  posterior likelihood: {t_posterior:.3f}s ({100.0 * t_posterior / total_elapsed:.1f}%)")
        print(f"  posterior merge + entropy: {t_merge:.3f}s ({100.0 * t_merge / total_elapsed:.1f}%)")
        print(f"  surface points: {t_surface:.3f}s ({100.0 * t_surface / total_elapsed:.1f}%)")
        print(f"  DDA CDF update: {t_dda:.3f}s ({100.0 * t_dda / total_elapsed:.1f}%)")
        print(f"  output writing: {t_output:.3f}s ({100.0 * t_output / total_elapsed:.1f}%)")
        print(f"  other / overhead: {other:.3f}s ({100.0 * other / total_elapsed:.1f}%)")


if __name__ == "__main__":
    main()
