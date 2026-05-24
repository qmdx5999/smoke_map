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
from pathlib import Path
from typing import Dict, List, Tuple

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


def write_grid_npz(path: str, Lgrid: np.ndarray, mins: np.ndarray, voxel: float) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        path,
        Lgrid=np.asarray(Lgrid, dtype=np.float32),
        mins=np.asarray(mins, dtype=np.float32),
        voxel=np.asarray([voxel], dtype=np.float32),
    )
    print("saved", path, "shape=", tuple(int(v) for v in Lgrid.shape))


def write_grid_scalar_ply(path: str, Lgrid: np.ndarray, mins: np.ndarray, voxel: float) -> int:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    nx, ny, nz = Lgrid.shape
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {nx * ny * nz}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float occupancy\n")
        f.write("end_header\n")
        for ix in range(nx):
            cx = mins[0] + voxel * (ix + 0.5)
            for iy in range(ny):
                cy = mins[1] + voxel * (iy + 0.5)
                for iz in range(nz):
                    cz = mins[2] + voxel * (iz + 0.5)
                    L = float(Lgrid[ix, iy, iz])
                    p = 1.0 / (1.0 + math.exp(-L))
                    f.write(f"{cx:.6f} {cy:.6f} {cz:.6f} {p:.8f}\n")
    count = int(nx * ny * nz)
    print("saved", path, "N=", count, "scalar_field=occupancy")
    return count


def load_spad_hist_npz(path: Path) -> np.ndarray:
    """输入：单个.npz 文件的路径;输出：形状为(H, W, T)的三维 numpy 数组，存储该帧的 SPAD 直方图数据"""
    with np.load(path) as data:
        if "spad_hist" not in data:
            raise KeyError(f"{path} does not contain a 'spad_hist' array")
        spad_hist = np.asarray(data["spad_hist"])
    if spad_hist.ndim != 3:
        raise ValueError(f"{path}: spad_hist must have shape (H, W, T), got {spad_hist.shape}")
    return spad_hist


def discover_npz_frames(npz_dir: str, max_frames: int) -> List[Path]:
    """输出按文件名升序排列的.npz 文件路径列表"""
    root = Path(npz_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)
    frames = sorted(root.glob("*.npz"), key=lambda p: p.name)
    if not frames:
        raise FileNotFoundError(f"No .npz files found in {root}")
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def load_poses_txt(path: str) -> Dict[str, np.ndarray]:
    """输入一个位姿文本文件，输出一个{帧名: 4x4 T_wc矩阵}的字典"""
    pose_path = Path(path)
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)

    poses: Dict[str, np.ndarray] = {}
    with open(pose_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 17:
                raise ValueError(
                    f"{pose_path}:{line_no}: expected frame_key plus 16 matrix values, got {len(parts)} fields"
                )
            key = parts[0]
            if key in poses:
                raise ValueError(f"{pose_path}:{line_no}: duplicate pose key {key!r}")
            try:
                values = [float(v) for v in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"{pose_path}:{line_no}: non-numeric matrix value") from exc
            T_wc = np.asarray(values, dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(T_wc)):
                raise ValueError(f"{pose_path}:{line_no}: matrix contains non-finite values")
            poses[key] = T_wc

    if not poses:
        raise ValueError(f"{pose_path} does not contain any poses")
    return poses


def resolve_map_bounds(args: argparse.Namespace, multi_frame: bool) -> Tuple[np.ndarray, np.ndarray]:
    """确定整个三维占用栅格地图在世界坐标系中的空间边界"""
    # 收集手动边界参数
    explicit = [
        args.map_min_x, args.map_max_x,
        args.map_min_y, args.map_max_y,
        args.map_min_z, args.map_max_z,
    ]
    # 如果用户没有指定任何手动边界参数，进入自动边界模式
    if all(v is None for v in explicit):
        mins = np.array([-args.range_max, -args.range_max, args.z_min], dtype=np.float64)
        maxs = np.array([args.range_max, args.range_max, args.z_max], dtype=np.float64)
        return mins, maxs
    if any(v is None for v in explicit):
        mode = "multi-frame" if multi_frame else "single-frame"
        raise ValueError(f"{mode}: specify all six map bound arguments or none")

    # 如果用户指定了全部六个参数，直接使用这些值作为地图边界
    mins = np.array([args.map_min_x, args.map_min_y, args.map_min_z], dtype=np.float64)
    maxs = np.array([args.map_max_x, args.map_max_y, args.map_max_z], dtype=np.float64)
    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        raise ValueError("map bounds must be finite")
    if np.any(maxs <= mins):
        raise ValueError(f"invalid map bounds: mins={mins}, maxs={maxs}")
    return mins, maxs


def process_frame(
    frame_key: str,
    spad_hist: np.ndarray,
    T_wc: np.ndarray,
    Lgrid: np.ndarray,
    mins: np.ndarray,
    voxel: float,
    args: argparse.Namespace,
) -> Tuple[List[Tuple[float, float, float, float]], Dict[str, float]]:
    H, W, B = spad_hist.shape
    hist_data = spad_hist.reshape(H * W, B).astype(np.float64)
    bin_to_m = C_M_PER_S * (args.tmax_ns * 1e-9) / 2.0 / float(B)
    print(f"[{frame_key}] spad_hist shape={spad_hist.shape}, dtype={spad_hist.dtype}")
    print(f"[{frame_key}] bin_to_m={bin_to_m:.8f} m/bin, tmax={args.tmax_ns:g} ns")

    default_fx, default_fy, default_cx, default_cy = scaled_nyu_intrinsics(W, H)
    fx = default_fx if args.fx is None else args.fx
    fy = default_fy if args.fy is None else args.fy
    cx = default_cx if args.cx is None else args.cx
    cy = default_cy if args.cy is None else args.cy
    print(f"[{frame_key}] intrinsics fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")

    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x_n_all = ((u_grid.reshape(-1) - cx) / fx).astype(np.float64)
    y_n_all = ((v_grid.reshape(-1) - cy) / fy).astype(np.float64)

    # T_wc：相机坐标系 → 世界坐标系的齐次变换矩阵
    if T_wc.shape != (4, 4):
        raise ValueError(f"{frame_key}: T_wc must have shape (4, 4), got {T_wc.shape}")
    # R_wc：3x3 旋转矩阵，描述相机的朝向,正交矩阵，将相机坐标系下的方向向量，旋转到世界坐标系下
    R_wc = np.asarray(T_wc[:3, :3], dtype=np.float64)
    # origin_w：3x1 平移向量，描述相机在世界坐标系中的位置,也就是这一帧相机光心的三维坐标
    origin_w = np.asarray(T_wc[:3, 3], dtype=np.float64)

    mx = hist_data.max(axis=1)
    cand = np.where(mx >= args.peak_thr)[0]
    print(f"[{frame_key}] candidate rays: {cand.size}")
    if cand.size == 0:
        return [], {
            "rays_total": 0.0,
            "rays_with_peaks": 0.0,
            "rays_fused": 0.0,
            "posterior_evals": 0.0,
            "t_peak": 0.0,
            "t_posterior": 0.0,
            "t_merge": 0.0,
            "t_surface": 0.0,
            "t_dda": 0.0,
        }

    if args.max_rays > 0 and cand.size > args.max_rays:
        sel = np.linspace(0, cand.size - 1, args.max_rays).astype(np.int64)
        rays = cand[sel]
    else:
        rays = cand
    print(f"[{frame_key}] using rays: {rays.size}")
    ray_density_scale = min(1.0, float(args.ray_norm_target) / max(float(rays.size), 1.0))
    effective_update_scale = float(args.update_scale) * ray_density_scale
    print(
        f"[{frame_key}] ray_density_scale={ray_density_scale:.6f} "
        f"effective_update_scale={effective_update_scale:.6f}"
    )

    end_dist = float(args.range_max)
    range_model_is_range = args.range_model == "range"
    logit_p0 = float(logit(args.p0))
    Lmin = float(args.Lmin)
    Lmax = float(args.Lmax)
    nx, ny, nz = Lgrid.shape
    support = int(args.mp_support) if args.mp_support > 0 else int(np.ceil(4.0 * args.sigma_bins))
    min_sep = support

    surface_points: List[Tuple[float, float, float, float]] = []
    t_peak = 0.0
    t_posterior = 0.0
    t_merge = 0.0
    t_surface = 0.0
    t_dda = 0.0
    rays_with_peaks = 0
    rays_fused = 0
    posterior_evals = 0
    frame_t0 = time.time()

    for k, idx in enumerate(rays):
        h = hist_data[idx]
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

        peak_posteriors_r = []
        peak_posteriors_p = []
        profile_points = []
        total_peak_score = float(sum(max(float(s), 0.0) for s in peak_scores))
        if total_peak_score <= 0.0:
            total_peak_score = float(len(peaks))

        for pk, pk_score in zip(peaks, peak_scores):
            _tp = time.perf_counter()
            # 计算距离后验
            r_grid_m, _, pdf_grid = compute_ll_grid_numba(
                h,
                int(pk),
                int(args.Wr_bin),
                int(args.M),
                int(args.win_half),
                float(args.sigma_bins),
                float(args.tau),
                float(bin_to_m),
            )
            t_posterior += time.perf_counter() - _tp
            posterior_evals += 1
            peak_weight = max(float(pk_score), 0.0) / total_peak_score
            # 后验概率最大的位置
            pk_local = int(np.argmax(pdf_grid))
            # 后验概率最大的位置对应的深度值和置信度
            peak_depth = float(r_grid_m[pk_local])
            peak_conf = float(pdf_grid[pk_local])
            if peak_depth <= 0.0 or peak_depth > end_dist:
                continue
            peak_posteriors_r.append(r_grid_m)
            peak_posteriors_p.append(pdf_grid * peak_weight)
            profile_points.append((peak_depth, peak_conf))

        if len(peak_posteriors_r) == 0:
            continue

        _tp = time.perf_counter()
        if len(peak_posteriors_r) == 1:
            ray_r = np.asarray(peak_posteriors_r[0], dtype=np.float64)
            ray_p = np.asarray(peak_posteriors_p[0], dtype=np.float64)
        else:
            ray_r = np.concatenate([np.asarray(arr, dtype=np.float64) for arr in peak_posteriors_r])
            ray_p = np.concatenate([np.asarray(arr, dtype=np.float64) for arr in peak_posteriors_p])
            order = np.argsort(ray_r)
            ray_r = ray_r[order]
            ray_p = ray_p[order]

        total_prob = float(ray_p.sum())
        if total_prob <= 0.0:
            continue
        ray_p /= total_prob
        ray_cdf = np.cumsum(ray_p)
        ray_cdf[-1] = 1.0
        update_weight = posterior_entropy_weight(ray_p, w_min=0.2) * effective_update_scale
        t_merge += time.perf_counter() - _tp

        # 生成相机坐标系下的原始方向向量
        d_cam = np.array([float(x_n_all[idx]), float(y_n_all[idx]), 1.0], dtype=np.float64)
        # 归一化为单位向量:计算方向向量的模长,将方向向量除以模长，得到单位方向向量
        nrm = np.linalg.norm(d_cam)
        if nrm <= 1e-12:
            continue
        d_cam /= nrm
        # 旋转方向向量:@是 Python 中的矩阵乘法运算符,用旋转矩阵R_wc乘以相机坐标系下的方向向量d_cam，得到世界坐标系下的方向向量d_w
        d_w = R_wc @ d_cam
        # 再次归一化
        nrm_w = np.linalg.norm(d_w)
        if nrm_w <= 1e-12:
            continue
        d_w = np.asarray(d_w / nrm_w, dtype=np.float64)

        _tp = time.perf_counter()
        for peak_depth, peak_conf in profile_points:
            if peak_depth <= 0.0 or peak_depth > end_dist:
                continue
            if range_model_is_range:
                # p3:这个表面点在世界坐标系中的三维坐标;origin_w:相机光心在世界坐标系中的三维坐标
                # d_w:世界坐标系下的从相机指向这个点的单位方向向量;peak_depth:这个点到相机光心的直线距离
                # 相机光心 + 方向 × 深度
                p3 = origin_w + d_w * peak_depth
            else:
                scale = peak_depth / max(d_w[2], 1e-12)
                p3 = origin_w + d_w * scale
            # 这里把每条射线的最大后验概率对应的深度值，通过 origin_w + d_w * peak_depth 投影成世界坐标系下的三维点
            # peak_conf 作为灰度值编码到 PLY 中
            surface_points.append((float(p3[0]), float(p3[1]), float(p3[2]), float(peak_conf)))
        t_surface += time.perf_counter() - _tp

        _tp = time.perf_counter()
        dda_update_dense_cdf(
            Lgrid, origin_w, d_w, end_dist, mins, voxel,
            nx, ny, nz, ray_r, ray_cdf, int(ray_r.size),
            Lmin, Lmax, logit_p0, float(args.p_occ), float(args.p_free), float(update_weight),
            range_model_is_range,
        )
        t_dda += time.perf_counter() - _tp
        rays_fused += 1

        if (k + 1) % 5000 == 0:
            elapsed = round(time.time() - frame_t0, 1)
            print(f"[{frame_key}] processed rays: {k + 1} / {rays.size} elapsed(s)= {elapsed}")

    print(f"[{frame_key}] done. elapsed(s)= {round(time.time() - frame_t0, 2)}")
    
    return surface_points, {
        "rays_total": float(rays.size),
        "rays_with_peaks": float(rays_with_peaks),
        "rays_fused": float(rays_fused),
        "posterior_evals": float(posterior_evals),
        "t_peak": t_peak,
        "t_posterior": t_posterior,
        "t_merge": t_merge,
        "t_surface": t_surface,
        "t_dda": t_dda,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ, help="Single SPAD .npz path containing spad_hist")
    ap.add_argument("--npz-dir", "--npz_dir", default=None, help="Directory of per-frame .npz files containing spad_hist")
    ap.add_argument("--poses", default=None, help="Text file with frame_key followed by 16 camera-to-world matrix values")
    ap.add_argument("--max-frames", "--max_frames", type=int, default=0, help="Limit multi-frame processing, 0=all")
    ap.add_argument("--ply-out", default=DEFAULT_PLY_OUT)
    ap.add_argument("--surface-out", default=DEFAULT_SURFACE_OUT)
    ap.add_argument("--tmax-ns", type=float, default=100.0)
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--range_max", type=float, default=5.0)
    ap.add_argument("--z_min", type=float, default=-6.0)
    ap.add_argument("--z_max", type=float, default=6.0)
    ap.add_argument("--map-min-x", type=float, default=None)
    ap.add_argument("--map-max-x", type=float, default=None)
    ap.add_argument("--map-min-y", type=float, default=None)
    ap.add_argument("--map-max-y", type=float, default=None)
    ap.add_argument("--map-min-z", type=float, default=None)
    ap.add_argument("--map-max-z", type=float, default=None)
    ap.add_argument("--max_rays", type=int, default=0, help="subsample rays per frame, 0=all candidate rays")
    ap.add_argument("--peak_thr", type=float, default=5.0)
    ap.add_argument("--Wr_bin", type=int, default=12)
    ap.add_argument("--M", type=int, default=81)
    ap.add_argument("--p_occ", type=float, default=0.70)
    ap.add_argument("--p_free", type=float, default=0.35)
    ap.add_argument("--update-scale", type=float, default=1.0, help="Global multiplier for each ray log-odds update; values below 1 reduce saturation.")
    ap.add_argument(
        "--ray-norm-target",
        type=float,
        default=10000.0,
        help="Target ray count used to normalize per-frame update strength; scales down updates when a frame uses more rays.",
    )
    ap.add_argument("--p0", type=float, default=0.50)
    ap.add_argument("--Lmin", type=float, default=-10.0)
    ap.add_argument("--Lmax", type=float, default=10.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--win_half", type=int, default=25)
    ap.add_argument("--sigma_bins", type=float, default=2.0)
    ap.add_argument("--range_model", choices=["range", "z"], default="range")
    ap.add_argument("--max_peaks", type=int, default=2)
    ap.add_argument("--mp_thr", type=float, default=5.0)
    ap.add_argument("--mp_support", type=int, default=0, help="support radius in bins, 0=auto(4*sigma)")
    ap.add_argument("--export-min-prob", type=float, default=0.55)
    ap.add_argument("--max_out", type=int, default=0)
    ap.add_argument("--grid-out", default=None, help="Optional .npz dump of the full occupancy log-odds grid for later thresholding.")
    ap.add_argument("--grid-ply-out", default=None, help="Optional PLY dump of all voxels with an occupancy scalar field for CloudCompare thresholding.")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--profile", action="store_true", help="Print timing breakdown for major pipeline stages.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    multi_frame = args.npz_dir is not None
    if multi_frame and args.poses is None:
        raise ValueError("--poses is required when using --npz-dir")

    if multi_frame:
        frames = discover_npz_frames(args.npz_dir, int(args.max_frames))
        poses = load_poses_txt(args.poses)
        missing = [p.stem for p in frames if p.stem not in poses]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise KeyError(f"poses file is missing {len(missing)} frame keys: {preview}{suffix}")
        print(f"multi-frame mode: {len(frames)} frames from {args.npz_dir}")
    else:
        frames = [Path(args.npz)]
        poses = {frames[0].stem: np.eye(4, dtype=np.float64)}
        print("single-frame mode")

    voxel = float(args.voxel)
    # 得到地图边界
    mins, maxs = resolve_map_bounds(args, multi_frame)
    # maxs - mins：计算 x、y、z 三个方向的总长度,除以体素边长得到每个方向需要的体素数量
    dims = np.ceil((maxs - mins) / voxel).astype(np.int64)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    print("grid mins =", mins, "maxs =", maxs)
    print("grid dims =", (nx, ny, nz), "voxel =", voxel)
    # 创建一个三维数组Lgrid，形状为(nx, ny, nz)，每个元素对应一个体素,初始值全为 0，对应先验占用概率 0.5
    Lgrid = np.zeros((nx, ny, nz), dtype=np.float32)

    # 收集所有帧检测到的表面点云，每个元素是一个包含 4 个浮点数的元组：(x, y, z, conf)
    all_surface_points: List[Tuple[float, float, float, float]] = []
    # 用来累计统计整个多帧处理过程的各项性能指标
    # rays_total:所有帧总共处理的射线数量;rays_with_peaks:所有帧中检测到有效峰值的射线数量;rays_fused:所有帧中成功融合到地图中的射线数量
    # posterior_evals:所有帧中总共计算的距离后验次数;t_peak:所有帧中峰值检测阶段的总耗时;t_posterior:所有帧中后验似然计算阶段的总耗时
    # t_merge:所有帧中后验合并与熵计算阶段的总耗时;t_surface:所有帧中表面点生成阶段的总耗时;t_dda:所有帧中 DDA 地图更新阶段的总耗时
    # 所有帧处理完成后，如果开启了--profile参数，这些统计值会被打印出来，方便分析性能瓶颈
    totals = {
        "rays_total": 0.0,
        "rays_with_peaks": 0.0,
        "rays_fused": 0.0,
        "posterior_evals": 0.0,
        "t_peak": 0.0,
        "t_posterior": 0.0,
        "t_merge": 0.0,
        "t_surface": 0.0,
        "t_dda": 0.0,
    }
    t0 = time.time()

    for frame_i, frame_path in enumerate(frames, start=1):
        # 取出当前帧文件的文件名，就是去掉.npz后缀，也就是帧名，这个帧名会用来从poses字典中查找对应的相机位姿矩阵
        frame_key = frame_path.stem
        print(f"loading frame {frame_i}/{len(frames)}: {frame_path}")
        # 加载当前帧的 SPAD 直方图数据，返回值spad_hist是一个三维 numpy 数组，形状为(H, W, T)
        spad_hist = load_spad_hist_npz(frame_path)
        surface_points, stats = process_frame(
            frame_key,
            spad_hist,
            poses[frame_key],
            Lgrid,
            mins,
            voxel,
            args,
        )
        # 所有帧的表面点不做任何筛选，直接追加到同一个列表
        all_surface_points.extend(surface_points)
        for key in totals:
            totals[key] += float(stats.get(key, 0.0))

    print("all frames done. elapsed(s)=", round(time.time() - t0, 2))
    _tp = time.perf_counter()
    if all_surface_points:
        write_surface_ply(args.surface_out, np.asarray(all_surface_points, dtype=np.float32))
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
    if args.grid_out:
        write_grid_npz(args.grid_out, Lgrid, mins, voxel)
    if args.grid_ply_out:
        write_grid_scalar_ply(args.grid_ply_out, Lgrid, mins, voxel)
    t_output = time.perf_counter() - _tp
    active_count = int(np.count_nonzero(np.abs(Lgrid) > 1e-6))
    print("active voxels:", active_count)
    print("exported occupied voxels:", occupied_count)

    if bool(args.profile):
        total_elapsed = max(time.time() - t0, 1e-12)
        measured = (
            totals["t_peak"] + totals["t_posterior"] + totals["t_merge"]
            + totals["t_surface"] + totals["t_dda"] + t_output
        )
        other = max(0.0, total_elapsed - measured)
        print("profile timing breakdown:")
        print(
            f"  frames: {len(frames)}, rays total: {int(totals['rays_total'])}, "
            f"with peaks: {int(totals['rays_with_peaks'])}, fused: {int(totals['rays_fused'])}, "
            f"posterior evals: {int(totals['posterior_evals'])}"
        )
        print(f"  peak detection: {totals['t_peak']:.3f}s ({100.0 * totals['t_peak'] / total_elapsed:.1f}%)")
        print(f"  posterior likelihood: {totals['t_posterior']:.3f}s ({100.0 * totals['t_posterior'] / total_elapsed:.1f}%)")
        print(f"  posterior merge + entropy: {totals['t_merge']:.3f}s ({100.0 * totals['t_merge'] / total_elapsed:.1f}%)")
        print(f"  surface points: {totals['t_surface']:.3f}s ({100.0 * totals['t_surface'] / total_elapsed:.1f}%)")
        print(f"  DDA CDF update: {totals['t_dda']:.3f}s ({100.0 * totals['t_dda'] / total_elapsed:.1f}%)")
        print(f"  output writing: {t_output:.3f}s ({100.0 * t_output / total_elapsed:.1f}%)")
        print(f"  other / overhead: {other:.3f}s ({100.0 * other / total_elapsed:.1f}%)")


if __name__ == "__main__":
    main()
