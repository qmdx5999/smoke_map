"""
Build a probabilistic occupancy grid from SPAD histograms stored in an .npz file.

This adapts the occupancy-mapping pipeline from
test_mapping_profile_3d_pdfalign_fast_v3.py to scene_0000.npz-style data:
spad_hist has shape (H, W, T), and the time-bin distance scale is derived from
the laser period and number of bins.
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
def fit_profile_one_r_smoke_count_numba(h, S, G, smoke_scale, eps=1e-6, max_iter=12):
    """
    拟合累计计数域模型:
    lambda = alpha * S + smoke_scale * G + beta.

    alpha 是包含表面衰减的有效表面计数幅度
    smoke_scale 等于 n_pulses * gamma
    只对非负 alpha 和 beta 做 profile optimization（剖面优化）

    h[i]	第 i 个 time bin 观测到的 photon count
    S[i]	假设表面在当前候选距离 r 时，表面回波的高斯模板
    G[i]	假设表面在当前候选距离 r 时，0 到 r 之间烟雾散射形成的模板
    smoke_scale	烟雾强度尺度，通常是 n_pulses * gamma
    alpha	表面回波的有效幅度，已经包含烟雾衰减后的结果
    beta	每个 bin 的常数背景项
    λ[i]	模型预测第 i 个 bin 应该有多少 photon count
    """
    # beta:临时用来累加观测总 photon count
    beta = 0.0
    
    # smoke_sum:烟雾模板的总 photon count
    smoke_sum = 0.0
    
    # hmax:观测 histogram 的最大值
    hmax = 0.0
    
    # smoke_max:烟雾项在所有 bin 里的最大值
    smoke_max = 0.0
    
    # scale:非负烟雾尺度
    scale = max(smoke_scale, 0.0)
    for i in range(h.size):
        # 累加观测总光子数
        beta += h[i]
        
        # 记录观测峰值
        if h[i] > hmax:
            hmax = h[i]
        
        # smoke_i 是第 i 个 bin 里，模型认为烟雾会贡献多少 photon count
        smoke_i = scale * G[i]
        
        # 统计烟雾总量
        smoke_sum += smoke_i
        
        # 记录烟雾峰值
        if smoke_i > smoke_max:
            smoke_max = smoke_i
    
    # 观测总光子数 ≈ 表面总光子数 + 烟雾总光子数 + 背景总光子数,粗略估计beta ≈ (观测总光子数 - 烟雾总光子数) / bin数量
    beta = max((beta - smoke_sum) / max(1, h.size), 0.0)

    # 找 surface 模板的最大值
    smax = 0.0
    for i in range(S.size):
        if S[i] > smax:
            smax = S[i]

    # 如果 surface 模板无效,只用烟雾散射 + 背景
    if smax <= 0.0:
        lam_bg = np.empty_like(h)
        for i in range(h.size):
            lam_bg[i] = max(scale * G[i] + beta, eps)
        
        # ll:当前退化模型的 Poisson log-likelihood;alpha:0.0，因为没有 surface 项;beta:当前估计的背景
        return poisson_ll_numba(h, lam_bg), 0.0, beta

    # hmax ≈ alpha * smax + smoke_max + beta,alpha ≈ (hmax - smoke_max - beta) / smax
    alpha = max(hmax - beta - smoke_max, 0.0) / max(smax, 1e-12)
    
    # 创建预测 histogram 数组
    lam = np.empty_like(h)
    for _ in range(max_iter):
        for i in range(h.size):
            lam[i] = max(alpha * S[i] + scale * G[i] + beta, eps)

        g_alpha = 0.0
        g_beta = 0.0
        for i in range(h.size):
            inv = h[i] / lam[i] - 1.0
            g_alpha += S[i] * inv
            g_beta += inv
        if abs(g_alpha) + abs(g_beta) < 1e-6:
            break

        H_aa = 0.0
        H_ab = 0.0
        H_bb = 0.0
        for i in range(h.size):
            w = h[i] / (lam[i] * lam[i])
            H_aa += -(S[i] * S[i]) * w
            H_ab += -S[i] * w
            H_bb += -w

        det = H_aa * H_bb - H_ab * H_ab
        if (not math.isfinite(det)) or abs(det) < 1e-18:
            break

        delta_alpha = (H_bb * g_alpha - H_ab * g_beta) / det
        delta_beta = (-H_ab * g_alpha + H_aa * g_beta) / det
        base_ll = poisson_ll_numba(h, lam)

        step = 1.0
        ok = False
        for __ in range(10):
            alpha_new = max(alpha - step * delta_alpha, 0.0)
            beta_new = max(beta - step * delta_beta, 0.0)
            for i in range(h.size):
                lam[i] = max(alpha_new * S[i] + scale * G[i] + beta_new, eps)
            ll_new = poisson_ll_numba(h, lam)
            if math.isfinite(ll_new) and ll_new >= base_ll - 1e-10:
                alpha = alpha_new
                beta = beta_new
                ok = True
                break
            step *= 0.5
        if not ok:
            break

    for i in range(h.size):
        lam[i] = max(alpha * S[i] + scale * G[i] + beta, eps)
    
    # poisson_ll_numba(h, lam)就是在问如果模型预测是 lam，那么观测到 h 的可能性有多大
    # 返回的是 Poisson log-likelihood，数值越大，说明当前候选距离 r 越能解释这个像素的 SPAD histogram
    # alpha 是拟合出来的表面回波强度,不是原始反射率，而是已经包含烟雾衰减后的有效表面 photon count 幅度
    # beta 是拟合出来的常数背景强度
    return poisson_ll_numba(h, lam), alpha, beta


@njit(cache=True, fastmath=True)
def profile_ll_one_r_smoke_count_numba(h, S, G, smoke_scale, eps=1e-6, max_iter=12):
    ll, _, _ = fit_profile_one_r_smoke_count_numba(h, S, G, smoke_scale, eps, max_iter)
    return ll


@njit(cache=True, fastmath=True)
def fit_smoke_background_only_numba(h, G, smoke_scale, eps=1e-6, max_iter=12):
    """Fit H0: lambda = smoke_scale * G + beta with beta >= 0."""
    scale = max(smoke_scale, 0.0)
    beta = 0.0
    smoke_sum = 0.0
    for i in range(h.size):
        beta += h[i]
        smoke_sum += scale * G[i]
    beta = max((beta - smoke_sum) / max(1, h.size), 0.0)

    lam = np.empty_like(h)
    for _ in range(max_iter):
        gradient = 0.0
        hessian = 0.0
        for i in range(h.size):
            lam_i = max(scale * G[i] + beta, eps)
            lam[i] = lam_i
            gradient += h[i] / lam_i - 1.0
            hessian += -h[i] / (lam_i * lam_i)
        if abs(gradient) < 1e-6 or (not math.isfinite(hessian)) or abs(hessian) < 1e-18:
            break

        base_ll = poisson_ll_numba(h, lam)
        delta = gradient / hessian
        step = 1.0
        ok = False
        for __ in range(10):
            beta_new = max(beta - step * delta, 0.0)
            for i in range(h.size):
                lam[i] = max(scale * G[i] + beta_new, eps)
            ll_new = poisson_ll_numba(h, lam)
            if math.isfinite(ll_new) and ll_new >= base_ll - 1e-10:
                beta = beta_new
                ok = True
                break
            step *= 0.5
        if not ok:
            break

    for i in range(h.size):
        lam[i] = max(scale * G[i] + beta, eps)
    return poisson_ll_numba(h, lam), beta


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


def build_smoke_integral_templates(
    n_bins: int,
    bin_to_m: float,
    sigma_bins: float,
    kappa: float,
    fog_step_m: float,
    range_max_m: float,
) -> Tuple[np.ndarray, float]:
    """
    Precompute unit-density G(r; kappa) for the smoke-aware likelihood.

    The caller multiplies the selected row by gamma=density.
    预计算单位烟雾密度下,从传感器到不同距离为止，烟雾沿途散射会在 SPAD histogram 中形成什么形状的查表数组
    """
    step_m = max(float(fog_step_m), 1e-6)
    dmax = float(n_bins) * float(bin_to_m)
    
    # rmax = min(用户指定的最大距离, histogram 最大距离)
    rmax = min(max(float(range_max_m), 0.0), dmax)
    if rmax <= 0.0:
        return np.zeros((1, n_bins), dtype=np.float64), step_m

    # 计算需要多少个距离采样点,这里决定要构造多少个距离模板
    n_steps = int(math.floor(rmax / step_m)) + 1
    
    # 初始化模板数组,那么templates[j, :]会是一条完整的 histogram 模板
    templates = np.zeros((n_steps, n_bins), dtype=np.float64)
    
    # 构造 bin 坐标,后面用它来生成高斯脉冲
    x = np.arange(n_bins, dtype=np.float64)
    
    # impulse response function 的标准差，单位是 bin
    sigma = max(float(sigma_bins), 1e-6)
    kap = max(float(kappa), 0.0)
    
    # 每一个 j 对应一个烟雾散射位置
    for j in range(n_steps):
        # s_m 表示沿 ray 距离传感器 s 米的位置有一小段烟雾，它会产生一部分散射光
        s_m = float(j) * step_m
        
        # 把距离 s_m 转成 bin 位置
        center_bin = s_m / max(float(bin_to_m), 1e-12)
        
        # 构造这个距离处的 IRF 高斯脉冲,生成一个以 center_bin 为中心的高斯峰
        irf = np.exp(-0.5 * ((x - center_bin) / sigma) ** 2)
        total = float(irf.sum())
        if total > 0.0:
            irf /= total
        
        # 计算这一小段烟雾的散射权重
        weight = math.exp(-2.0 * kap * s_m) * step_m
        
        # 对于每个距离 j，不是只保存该距离一小段烟雾的贡献，而是保存从 0 到 j * step_m 之间所有烟雾散射贡献的累计结果
        if j == 0:
            templates[j, :] = irf * weight
        else:
            templates[j, :] = templates[j - 1, :] + irf * weight
    
    # templates[j, :] 表示从传感器到距离 j * step_m 之间所有烟雾散射贡献的累计 histogram
    # 给定一条 ray，如果从相机到距离 r 之间都有均匀烟雾，那么这些烟雾会在 SPAD histogram 上形成一个前向散射背景,templates[j, :] 就是这个背景的形状
    return templates, step_m


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
def compute_ll_grid_smoke_details_numba(
    h,
    peak_bin,
    Wr_bin,
    M,
    win_half,
    sigma_bins,
    tau,
    bin_to_m,
    gamma,
    n_pulses,
    smoke_templates,
    smoke_step_m,
):
    """
    Smoke-aware range posterior（烟雾感知距离后验） using a prefix window.
    Candidate r stays near peak_bin, while likelihood uses bins [0, peak_bin+win_half].
    """
    B = h.size
    b0 = 0
    b1 = peak_bin + win_half + 1
    if b1 > B:
        b1 = B
    if b1 <= b0:
        b1 = B

    # 复制窗口内的观测 histogram
    nwin = b1 - b0
    h_w = np.empty(nwin, dtype=np.float64)
    for i in range(nwin):
        h_w[i] = h[i]

    # 初始化 likelihood 数组,ll[k] 用来存第 k 个候选距离的 log-likelihood
    ll = np.empty(M, dtype=np.float64)
    alpha_grid = np.empty(M, dtype=np.float64)
    beta_grid = np.empty(M, dtype=np.float64)
    r0 = max(0, peak_bin - Wr_bin)
    r1 = min(B - 1, peak_bin + Wr_bin)
    step = 0.0 if M == 1 else (r1 - r0) / (M - 1)

    # 遍历每个候选距离
    for k in range(M):
        # rb 是当前候选表面位置，单位是 bin
        rb = r0 + step * k
        
        # 把 bin 位置转换成实际距离，单位是米
        r_m = rb * bin_to_m
        
        # 构造当前候选距离对应的 surface 模板,表示如果表面在 rb 这个 bin，那么表面反射在当前窗口 [b0, b1) 内应该长什么样
        S = build_S_gaussian(b0, b1, rb, sigma_bins)
        
        # 把当前候选表面距离 r_m 映射到 smoke template 的行号
        row = int(math.floor(max(r_m, 0.0) / max(smoke_step_m, 1e-12)))
        if row < 0:
            row = 0
        if row >= smoke_templates.shape[0]:
            row = smoke_templates.shape[0] - 1
        
        # 取出当前候选距离下的 smoke 模板,表示如果表面距离是 r_m，那么从 0 到 r_m 的烟雾散射，在当前观测窗口内形成的 histogram 形状
        G = np.empty(nwin, dtype=np.float64)
        for i in range(nwin):
            G[i] = smoke_templates[row, i]
        
        # 计算当前候选距离的 log-likelihood,在候选距离 rb 下，当前 histogram 被 smoke-aware model 解释得有多好,数值越大，这个候选距离越可能
        # ll[i] = log p(h | r_i),意思是如果真实表面距离是第 i 个候选距离 r_i，那么观测到当前 SPAD histogram h 的可能性有多大
        smoke_scale = max(n_pulses, 0.0) * max(gamma, 0.0)
        ll[k], alpha_grid[k], beta_grid[k] = fit_profile_one_r_smoke_count_numba(
            h_w, S, G, smoke_scale
        )

    t = max(tau, 1e-6)
    
    # 找到最大的 ll[i] / t
    m = ll[0] / t
    for i in range(1, M):
        v = ll[i] / t
        if v > m:
            m = v

    # 初始化 posterior 数组,p[i]表示第 i 个候选距离的概率
    p = np.empty(M, dtype=np.float64)
    s = 0.0
    
    # 把 log-likelihood 转换成未归一化概率
    for i in range(M):
        v = math.exp(ll[i] / t - m)
        p[i] = v
        s += v
    
    # 归一化成真正的概率分布
    if s <= 0.0:
        for i in range(M):
            p[i] = 1.0 / M
    else:
        for i in range(M):
            p[i] /= s

    # 把概率分布 p 转换成累计分布 cdf
    cdf = np.empty(M, dtype=np.float64)
    acc = 0.0
    for i in range(M):
        acc += p[i]
        cdf[i] = acc

    # 构造候选距离网格，单位是米
    r_grid_m = np.empty(M, dtype=np.float64)
    for i in range(M):
        # r_grid_m[i]表示第 i 个候选距离的实际 range
        r_grid_m[i] = (r0 + step * i) * bin_to_m
    
    # 候选距离网格，单位是米;后验概率的累计分布;每个候选距离的后验概率
    return r_grid_m, cdf, p, ll, alpha_grid, beta_grid


@njit(cache=True, fastmath=True)
def compute_ll_grid_smoke_numba(
    h,
    peak_bin,
    Wr_bin,
    M,
    win_half,
    sigma_bins,
    tau,
    bin_to_m,
    gamma,
    n_pulses,
    smoke_templates,
    smoke_step_m,
):
    r_grid_m, cdf, p, _, _, _ = compute_ll_grid_smoke_details_numba(
        h,
        peak_bin,
        Wr_bin,
        M,
        win_half,
        sigma_bins,
        tau,
        bin_to_m,
        gamma,
        n_pulses,
        smoke_templates,
        smoke_step_m,
    )
    return r_grid_m, cdf, p


@njit(cache=True, fastmath=True)
def evaluate_smoke_peak_h0_numba(
    h,
    peak_bin,
    Wr_bin,
    M,
    win_half,
    bin_to_m,
    gamma,
    n_pulses,
    smoke_templates,
    smoke_step_m,
    map_idx,
):
    """Evaluate H0 once at the MAP candidate of a smoke posterior."""
    B = h.size
    b1 = peak_bin + win_half + 1
    if b1 > B:
        b1 = B
    if b1 <= 0:
        b1 = B

    h_w = np.empty(b1, dtype=np.float64)
    for i in range(b1):
        h_w[i] = h[i]

    r0 = max(0, peak_bin - Wr_bin)
    r1 = min(B - 1, peak_bin + Wr_bin)
    step = 0.0 if M == 1 else (r1 - r0) / (M - 1)
    rb = r0 + step * map_idx
    r_m = rb * bin_to_m
    row = int(math.floor(max(r_m, 0.0) / max(smoke_step_m, 1e-12)))
    if row < 0:
        row = 0
    if row >= smoke_templates.shape[0]:
        row = smoke_templates.shape[0] - 1
    G = np.empty(b1, dtype=np.float64)
    for i in range(b1):
        G[i] = smoke_templates[row, i]

    smoke_scale = max(n_pulses, 0.0) * max(gamma, 0.0)
    ll_h0, beta_h0 = fit_smoke_background_only_numba(
        h_w, G, smoke_scale
    )
    return ll_h0, beta_h0


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


def _decode_npz_scalar(value: np.ndarray) -> str:
    item = value.item() if value.shape == () else value.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def load_spad_frame_npz(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load spad_hist plus optional camera/fog/capture metadata from one .npz frame."""
    with np.load(path) as data:
        if "spad_hist" not in data:
            raise KeyError(f"{path} does not contain a 'spad_hist' array")
        spad_hist = np.asarray(data["spad_hist"])
        metadata: Dict[str, object] = {}
        if "camera_model" in data:
            raw = _decode_npz_scalar(np.asarray(data["camera_model"]))
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid camera_model JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}: camera_model must decode to a JSON object")
            metadata = parsed
        if "fog_model" in data:
            raw = _decode_npz_scalar(np.asarray(data["fog_model"]))
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid fog_model JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}: fog_model must decode to a JSON object")
            metadata["fog_model"] = parsed
        if "capture_model" in data:
            raw = _decode_npz_scalar(np.asarray(data["capture_model"]))
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid capture_model JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}: capture_model must decode to a JSON object")
            metadata["capture_model"] = parsed
    if spad_hist.ndim != 3:
        raise ValueError(f"{path}: spad_hist must have shape (H, W, T), got {spad_hist.shape}")
    return spad_hist, metadata


def load_spad_hist_npz(path: Path) -> np.ndarray:
    """输入：单个.npz 文件的路径;输出：形状为(H, W, T)的三维 numpy 数组，存储该帧的 SPAD 直方图数据"""
    with np.load(path) as data:
        if "spad_hist" not in data:
            raise KeyError(f"{path} does not contain a 'spad_hist' array")
        spad_hist = np.asarray(data["spad_hist"])
    if spad_hist.ndim != 3:
        raise ValueError(f"{path}: spad_hist must have shape (H, W, T), got {spad_hist.shape}")
    return spad_hist


def load_optional_gt_range_npz(path: Path) -> Optional[np.ndarray]:
    """Load gt_range for diagnostics only; it is never used by inference."""
    with np.load(path) as data:
        if "gt_range" not in data:
            return None
        gt_range = np.asarray(data["gt_range"], dtype=np.float64)
    if gt_range.ndim != 2:
        raise ValueError(f"{path}: gt_range must have shape (H, W), got {gt_range.shape}")
    return gt_range


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


def metadata_float(metadata: Dict[str, object], key: str, default: Optional[float]) -> Optional[float]:
    """从 metadata 字典里读取某个字段，并把它转换成 float；如果字段不存在，就返回默认值 default"""
    if key not in metadata:
        return default
    try:
        value = float(metadata[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"camera_model.{key} must be numeric, got {metadata[key]!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"camera_model.{key} must be finite, got {value!r}")
    return value


def resolve_likelihood_model(args: argparse.Namespace, metadata: Dict[str, object]) -> Tuple[str, float, float, float]:
    """
    根据命令行参数 --likelihood-model 和当前帧 metadata 里的 fog_model，决定使用 clear 还是 smoke likelihood，并返回烟雾模型参数
    输出likelihood_model, kappa消光系数, gamma烟雾体散射项的强度系数, step_m积分步长
    """
    fog = metadata.get("fog_model")
    if fog is None:
        fog = {}
    if not isinstance(fog, dict):
        raise ValueError("fog_model metadata must be a JSON object when present")

    enabled = bool(fog.get("enabled", False))
    fog_kind = str(fog.get("model", "none")).lower()
    has_ray_integral = enabled and fog_kind == "ray_integral"
    choice = str(args.likelihood_model).lower()

    if choice == "clear":
        return "clear", 0.0, 0.0, 0.05
    if choice == "auto" and not has_ray_integral:
        return "clear", 0.0, 0.0, 0.05
    if choice not in ("auto", "smoke"):
        raise ValueError(f"unsupported likelihood_model: {choice!r}")
    if not has_ray_integral:
        raise ValueError(
            "--likelihood-model smoke requires fog_model.enabled=true and fog_model.model='ray_integral'"
        )

    try:
        kappa = float(fog["extinction_1_per_m"])
        gamma = float(fog["density"])
        step_m = float(fog.get("step_m", 0.05))
    except KeyError as exc:
        raise ValueError(f"fog_model is missing required field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("fog_model extinction_1_per_m, density, and step_m must be numeric") from exc

    if not (math.isfinite(kappa) and math.isfinite(gamma) and math.isfinite(step_m)):
        raise ValueError("fog_model smoke parameters must be finite")
    if kappa < 0.0 or gamma < 0.0 or step_m <= 0.0:
        raise ValueError("fog_model requires kappa>=0, density>=0, and step_m>0")
    return "smoke", kappa, gamma, step_m


def resolve_n_pulses(args: argparse.Namespace, metadata: Dict[str, object], likelihood_model: str) -> Optional[int]:
    """Resolve pulse count with CLI taking precedence over capture_model metadata."""
    if args.n_pulses is not None:
        n_pulses = int(args.n_pulses)
    else:
        capture = metadata.get("capture_model")
        if capture is None:
            capture = {}
        if not isinstance(capture, dict):
            raise ValueError("capture_model metadata must be a JSON object when present")
        raw = capture.get("n_pulses")
        n_pulses = None if raw is None else int(raw)

    if n_pulses is not None and n_pulses <= 0:
        raise ValueError("n_pulses must be a positive integer")
    if likelihood_model == "smoke" and n_pulses is None:
        raise ValueError(
            "smoke likelihood requires pulse count; pass --n-pulses or use data with capture_model.n_pulses"
        )
    return n_pulses


def resolve_image_y_axis(args: argparse.Namespace, metadata: Dict[str, object]) -> str:
    """决定当前帧图像 y 轴方向，优先使用命令行参数；如果命令行是 auto，就从 metadata 里读取；如果 metadata 也没有，就默认用 down"""
    choice = str(args.image_y_axis).lower()
    # 如果用户不选 auto，就直接返回用户指定值
    if choice != "auto":
        return choice
    # 如果是 auto，就从 metadata 里找,从 metadata 字典里读取 "image_y_axis"；如果没有这个字段，就默认用 "down"
    value = str(metadata.get("image_y_axis", "down")).lower()
    if value not in ("down", "up"):
        raise ValueError(f"camera_model.image_y_axis must be 'down' or 'up', got {value!r}")
    return value


def bbox_from_points(points: List[Tuple[float, float, float, float]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """根据当前帧生成的所有表面点 surface_points，计算这一帧点云的 3D bounding box"""
    if not points:
        return None
    arr = np.asarray(points, dtype=np.float64)
    # 只取前三列 x, y, z
    xyz = arr[:, :3]
    return xyz.min(axis=0), xyz.max(axis=0)


def bbox_overlap_ratio(
    a: Optional[Tuple[np.ndarray, np.ndarray]],
    b: Optional[Tuple[np.ndarray, np.ndarray]],
) -> float:
    if a is None or b is None:
        return 0.0
    # 交集的最小角要取两个最小角里更大的那个
    lo = np.maximum(a[0], b[0])
    # 交集的最大角要取两个最大角里更小的那个
    hi = np.minimum(a[1], b[1])
    # 计算交集盒子的长宽高
    extent = np.maximum(hi - lo, 0.0)
    # 计算交集体积,np.prod(extent) 是把三个方向的边长相乘,inter = intersection_volume
    inter = float(np.prod(extent))
    # 计算两个 bbox 各自的体积
    vol_a = float(np.prod(np.maximum(a[1] - a[0], 0.0)))
    vol_b = float(np.prod(np.maximum(b[1] - b[0], 0.0)))
    # 得到较小 bbox 的体积,denominator分母
    denom = max(min(vol_a, vol_b), 1e-12)
    # intersection / smaller_volume,交集体积占较小 bbox 体积的比例
    return inter / denom


def merge_bboxes(
    a: Optional[Tuple[np.ndarray, np.ndarray]],
    b: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """把两个 3D bbox 合并成一个更大的 bbox，使新的 bbox 能同时包住原来的两个 bbox"""
    if a is None:
        return b
    if b is None:
        return a
    return np.minimum(a[0], b[0]), np.maximum(a[1], b[1])


def format_bbox(bbox: Optional[Tuple[np.ndarray, np.ndarray]]) -> str:
    if bbox is None:
        return "empty"
    mn, mx = bbox
    return (
        f"min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) "
        f"max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})"
    )


def parse_diagnostic_checkpoints(value: str, total_frames: int) -> List[int]:
    """
    把用户输入的诊断检查点字符串，例如 "5,50,all"，解析成一个去重、升序排列的整数帧数列表，
    用来决定程序处理到哪些帧时打印 occupancy grid 的诊断信息
    """
    # set() 是 Python 里的集合类型,不允许重复元素,没有固定顺序,适合用来做去重
    out = set()
    # 把输入字符串按逗号拆开,例如"5,50,all"会拆成["5", "50", "all"]
    for raw in str(value).split(","):
        # 清理空格并转小写,.lower() 是字符串方法，作用是把字符串中的英文字母全部转成小写
        token = raw.strip().lower()
        # 如果用户写了多余的逗号,中间会出现空字符串 ""，这里直接跳过
        if not token:
            continue
        if token == "all":
            out.add(int(total_frames))
        # 如果 token 不是 "all"，就尝试把它转换成整数,例如"50" -> 50
        else:
            n = int(token)
            # 只接受正整数。0 和负数会被忽略
            if n > 0:
                out.add(min(n, int(total_frames)))
    # 返回排序后的结果,因为 set 本身没有顺序，所以最后用 sorted 排序
    return sorted(out)


def print_grid_diagnostics(label: str, Lgrid: np.ndarray, export_min_prob: float) -> None:
    # 把 Lgrid 里的 log-odds 值转成普通概率
    prob = 1.0 / (1.0 + np.exp(-Lgrid.astype(np.float64)))
    # 初始的 Lgrid 全是 0,如果某个 voxel 被 ray 更新过，它的 Lgrid 通常会偏离 0,active voxel = 被更新过的 voxel
    active = np.abs(Lgrid) > 1e-6
    # 占用概率大于等于 0.51 的 voxel，被认为是 occupied。
    occ = prob >= float(export_min_prob)
    # prob.reshape(-1)把三维概率网格拉平成一维数组,
    q_all = np.quantile(prob.reshape(-1), [0.0, 0.5, 0.9, 0.99, 1.0])
    if np.any(active):
        q_active = np.quantile(prob[active], [0.0, 0.1, 0.5, 0.9, 1.0])
        active_text = np.array2string(q_active, precision=4, separator=", ")
    else:
        active_text = "[]"
    print(
        f"[diagnostic {label}] active_voxels={int(active.sum())} "
        f"occupied_voxels@{export_min_prob:g}={int(occ.sum())} "
        f"prob_quantiles_all={np.array2string(q_all, precision=4, separator=', ')} "
        f"prob_quantiles_active={active_text}"
    )


def process_frame(
    frame_key: str,
    spad_hist: np.ndarray,
    frame_metadata: Dict[str, object],
    T_wc: np.ndarray,
    Lgrid: np.ndarray,
    mins: np.ndarray,
    voxel: float,
    args: argparse.Namespace,
    gt_range: Optional[np.ndarray] = None,
    peak_filter_csv_writer=None,
    peak_filter_warning_state: Optional[Dict[str, bool]] = None,
) -> Tuple[List[Tuple[float, float, float, float]], Dict[str, float]]:
    H, W, B = spad_hist.shape
    hist_data = spad_hist.reshape(H * W, B).astype(np.float64)
    bin_to_m = C_M_PER_S * (args.tmax_ns * 1e-9) / 2.0 / float(B)
    print(f"[{frame_key}] spad_hist shape={spad_hist.shape}, dtype={spad_hist.dtype}")
    print(f"[{frame_key}] bin_to_m={bin_to_m:.8f} m/bin, tmax={args.tmax_ns:g} ns")
    
    # 从当前帧的 metadata 中解析烟雾模型设置，并结合命令行参数决定这帧用 clear 还是 smoke；如果用 smoke，就取出烟雾消光系数、烟雾密度和积分步长
    likelihood_model, fog_kappa, fog_gamma, fog_step_m = resolve_likelihood_model(args, frame_metadata)
    n_pulses = resolve_n_pulses(args, frame_metadata, likelihood_model)
    peak_filter_active = bool(args.smoke_peak_filter) and likelihood_model == "smoke"
    if bool(args.smoke_peak_filter) and likelihood_model != "smoke":
        state = peak_filter_warning_state if peak_filter_warning_state is not None else {}
        if not bool(state.get("clear_skip_emitted", False)):
            print(
                "warning: --smoke-peak-filter is enabled, but the active likelihood is clear; "
                "peak filtering will be skipped. This warning is printed once per run."
            )
            state["clear_skip_emitted"] = True
    smoke_templates = np.zeros((1, B), dtype=np.float64)
    smoke_step_used = max(float(fog_step_m), 1e-6)
    if likelihood_model == "smoke":
        # smoke_templates 保存烟雾散射模板表；smoke_step_used 保存这个模板表对应的距离步长
        smoke_templates, smoke_step_used = build_smoke_integral_templates(
            n_bins=B,
            bin_to_m=float(bin_to_m),
            sigma_bins=float(args.sigma_bins),
            kappa=float(fog_kappa),
            fog_step_m=float(fog_step_m),
            range_max_m=float(args.range_max),
        )
    print(
        f"[{frame_key}] likelihood_model={likelihood_model} "
        f"n_pulses={n_pulses if n_pulses is not None else 'n/a'} "
        f"kappa={fog_kappa:.6g} gamma={fog_gamma:.6g} smoke_step={smoke_step_used:.6g}"
    )

    # 最终 fx, fy, cx, cy 的来源优先级是
    # 第 1 优先级：命令行参数
    # 例如 --fx、--fy、--cx、--cy

    # 第 2 优先级：.npz 文件里的 camera_model metadata
    # 例如 frame_metadata["fx"]

    # 第 3 优先级：根据图像尺寸缩放后的 NYU 默认内参
    # 例如 scaled_nyu_intrinsics(W, H)
    
    # 先根据当前 spad_hist 的宽高 W, H 算一套默认相机内参
    default_fx, default_fy, default_cx, default_cy = scaled_nyu_intrinsics(W, H)
    # 如果用户在命令行传了 --fx --fy --cx --cy，就用用户传的；否则先用默认值
    fx = default_fx if args.fx is None else args.fx
    fy = default_fy if args.fy is None else args.fy
    cx = default_cx if args.cx is None else args.cx
    cy = default_cy if args.cy is None else args.cy
    # 如果用户没有手动传入这些参数，那么程序再尝试从 .npz 文件的 frame_metadata 里读取更准确的内参
    # 如果 metadata 里没有，就继续保留刚才的默认值
    fx = metadata_float(frame_metadata, "fx", fx) if args.fx is None else fx
    fy = metadata_float(frame_metadata, "fy", fy) if args.fy is None else fy
    cx = metadata_float(frame_metadata, "cx", cx) if args.cx is None else cx
    cy = metadata_float(frame_metadata, "cy", cy) if args.cy is None else cy
    # 决定当前图像的 y 轴方向
    image_y_axis = resolve_image_y_axis(args, frame_metadata)
    print(f"[{frame_key}] intrinsics fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    print(f"[{frame_key}] image_y_axis={image_y_axis}")

    # 生成两个二维数组,u_grid从每一列上看都是横坐标 u,v_grid从每一行上看都是纵坐标 v
    # u_grid就是整张图每个像素的 u 坐标表，v_grid就是整张图每个像素的 v 坐标表
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    # reshape(-1)把二维数组拉平成一维数组，把像素横坐标 u 转成归一化相机坐标 x_n
    x_n_all = ((u_grid.reshape(-1) - cx) / fx).astype(np.float64)
    y_n_all = ((v_grid.reshape(-1) - cy) / fy).astype(np.float64)
    # 默认情况下y_n = (v - cy) / fy当 v 往下增大时，y_n 也增大,这对应的是图像 y 轴向下
    # 但如果 image_y_axis == "up"，说明当前相机模型约定相机/图像的 y 轴向上,这时需要把 y_n 取反
    if image_y_axis == "up":
        y_n_all = -y_n_all

    # T_wc：相机坐标系 → 世界坐标系的齐次变换矩阵
    if T_wc.shape != (4, 4):
        raise ValueError(f"{frame_key}: T_wc must have shape (4, 4), got {T_wc.shape}")
    # R_wc：3x3 旋转矩阵，描述相机的朝向,正交矩阵，将相机坐标系下的方向向量，旋转到世界坐标系下
    R_wc = np.asarray(T_wc[:3, :3], dtype=np.float64)
    # origin_w：3x1 平移向量，描述相机在世界坐标系中的位置,也就是这一帧相机光心的三维坐标
    origin_w = np.asarray(T_wc[:3, 3], dtype=np.float64)
    print(
        f"[{frame_key}] pose basis x={np.array2string(R_wc[:, 0], precision=4)} "
        f"y={np.array2string(R_wc[:, 1], precision=4)} "
        f"z={np.array2string(R_wc[:, 2], precision=4)}"
    )

    mx = hist_data.max(axis=1)
    # cand 是候选 ray 的索引数组
    cand = np.where(mx >= args.peak_thr)[0]
    print(f"[{frame_key}] candidate rays: {cand.size}")
    if cand.size == 0:
        if bool(args.print_peak_filter_stats) and peak_filter_active:
            print(
                f"[peak-filter] frame={frame_key} total_peaks=0 accepted_surface_peaks=0 "
                "rejected_peaks=0 rejected_by_delta_ll=0 rejected_by_alpha=0 "
                "mean_delta_ll=nan median_delta_ll=nan mean_alpha=nan median_alpha=nan"
            )
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

    # 需要下采样
    if args.max_rays > 0 and cand.size > args.max_rays:
        # 在 [0, cand.size - 1] 这个范围内，均匀生成 args.max_rays 个索引，.astype(np.int64)再转成整数
        sel = np.linspace(0, cand.size - 1, args.max_rays).astype(np.int64)
        # 从候选 ray 里均匀抽出 args.max_rays 条
        rays = cand[sel]
    # 不需要下采样
    else:
        rays = cand
    print(f"[{frame_key}] using rays: {rays.size}")
    # 一个缩放系数,0 < ray_density_scale <= 1.0,作用是根据 ray 数量调整每条 ray 的更新强度.ray 太多时可以缩小；ray 太少时不放大。
    ray_density_scale = min(1.0, float(args.ray_norm_target) / max(float(rays.size), 1.0))
    # 当前帧真正使用的更新强度系数,effective_update_scale = 用户设置的基础更新强度 × ray 数量归一化系数
    effective_update_scale = float(args.update_scale) * ray_density_scale
    # 目的都是当一帧有很多 ray 时，降低每条 ray 的更新强度，防止 occupancy grid 被过多 ray 更新得过快、过强、过早饱和
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
    filter_total_peaks = 0
    filter_accepted_peaks = 0
    filter_rejected_peaks = 0
    filter_rejected_by_delta_ll = 0
    filter_rejected_by_alpha = 0
    filter_delta_ll_values: List[float] = []
    filter_alpha_values: List[float] = []
    accepted_errors: List[float] = []
    rejected_errors: List[float] = []
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
            
            # 如果是烟雾模式似然
            if likelihood_model == "smoke":
                
                # 如果开启烟雾峰过滤器
                if peak_filter_active:
                    (
                        r_grid_m,
                        _,
                        pdf_grid,
                        ll_grid,
                        alpha_grid,
                        beta_grid,
                    ) = compute_ll_grid_smoke_details_numba(
                        h,
                        int(pk),
                        int(args.Wr_bin),
                        int(args.M),
                        int(args.win_half),
                        float(args.sigma_bins),
                        float(args.tau),
                        float(bin_to_m),
                        float(fog_gamma),
                        float(n_pulses),
                        smoke_templates,
                        float(smoke_step_used),
                    )
                
                # 关闭过滤器
                else:
                    r_grid_m, _, pdf_grid = compute_ll_grid_smoke_numba(
                        h,
                        int(pk),
                        int(args.Wr_bin),
                        int(args.M),
                        int(args.win_half),
                        float(args.sigma_bins),
                        float(args.tau),
                        float(bin_to_m),
                        float(fog_gamma),
                        float(n_pulses),
                        smoke_templates,
                        float(smoke_step_used),
                    )
            
            # 无烟似然
            else:
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

            # Reject out-of-range MAP estimates before H0 evaluation or filter statistics.
            if peak_depth <= 0.0 or peak_depth > end_dist:
                continue

            if peak_filter_active:
                # 取出 MAP 距离处的 H1 拟合结果
                # H1 模型在 MAP 距离处的 log-likelihood（对数似然）
                ll_h1 = float(ll_grid[pk_local])
                alpha = float(alpha_grid[pk_local])
                beta_h1 = float(beta_grid[pk_local])
                
                # 在同一个 MAP 距离计算 H0
                ll_h0, beta_h0 = evaluate_smoke_peak_h0_numba(
                    h,
                    int(pk),
                    int(args.Wr_bin),
                    int(args.M),
                    int(args.win_half),
                    float(bin_to_m),
                    float(fog_gamma),
                    float(n_pulses),
                    smoke_templates,
                    float(smoke_step_used),
                    int(pk_local),
                )
                delta_ll = float(ll_h1 - ll_h0)
                rejected_by_delta_ll = delta_ll < float(args.surface_dll_min)
                rejected_by_alpha = alpha < float(args.surface_alpha_min)
                accepted = not (rejected_by_delta_ll or rejected_by_alpha)

                filter_total_peaks += 1
                filter_delta_ll_values.append(delta_ll)
                filter_alpha_values.append(alpha)
                if accepted:
                    filter_accepted_peaks += 1
                else:
                    filter_rejected_peaks += 1
                if rejected_by_delta_ll:
                    filter_rejected_by_delta_ll += 1
                if rejected_by_alpha:
                    filter_rejected_by_alpha += 1

                row = int(idx // W)
                col = int(idx % W)
                gt_value = math.nan
                abs_range_error = math.nan
                if gt_range is not None:
                    gt_value = float(gt_range[row, col])
                    if math.isfinite(gt_value) and gt_value > 0.0:
                        abs_range_error = abs(peak_depth - gt_value)
                        if accepted:
                            accepted_errors.append(abs_range_error)
                        else:
                            rejected_errors.append(abs_range_error)
                if peak_filter_csv_writer is not None:
                    peak_filter_csv_writer.writerow({
                        "frame": frame_key,
                        "row": row,
                        "col": col,
                        "peak_bin": int(pk),
                        "peak_score": float(pk_score),
                        "r_hat": peak_depth,
                        "alpha": alpha,
                        "beta_h1": float(beta_h1),
                        "ll_h1": float(ll_h1),
                        "beta_h0": float(beta_h0),
                        "ll_h0": float(ll_h0),
                        "delta_ll": delta_ll,
                        "accepted": int(accepted),
                        "rejected_by_delta_ll": int(rejected_by_delta_ll),
                        "rejected_by_alpha": int(rejected_by_alpha),
                        "gt_range": gt_value if math.isfinite(gt_value) else "",
                        "abs_range_error": abs_range_error if math.isfinite(abs_range_error) else "",
                    })
                if not accepted:
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
        # 计算这条 ray 的最终更新权重，后验熵权重 × 全局更新强度 × ray 数量归一化系数
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

    if bool(args.print_peak_filter_stats) and peak_filter_active:
        delta_values = np.asarray(filter_delta_ll_values, dtype=np.float64)
        alpha_values = np.asarray(filter_alpha_values, dtype=np.float64)
        mean_delta = float(np.mean(delta_values)) if delta_values.size else math.nan
        median_delta = float(np.median(delta_values)) if delta_values.size else math.nan
        mean_alpha = float(np.mean(alpha_values)) if alpha_values.size else math.nan
        median_alpha = float(np.median(alpha_values)) if alpha_values.size else math.nan
        print(
            f"[peak-filter] frame={frame_key} total_peaks={filter_total_peaks} "
            f"accepted_surface_peaks={filter_accepted_peaks} rejected_peaks={filter_rejected_peaks} "
            f"rejected_by_delta_ll={filter_rejected_by_delta_ll} "
            f"rejected_by_alpha={filter_rejected_by_alpha} "
            f"mean_delta_ll={mean_delta:.6g} median_delta_ll={median_delta:.6g} "
            f"mean_alpha={mean_alpha:.6g} median_alpha={median_alpha:.6g}"
        )
        if accepted_errors or rejected_errors:
            accepted_error_mean = float(np.mean(accepted_errors)) if accepted_errors else math.nan
            accepted_error_median = float(np.median(accepted_errors)) if accepted_errors else math.nan
            rejected_error_mean = float(np.mean(rejected_errors)) if rejected_errors else math.nan
            rejected_error_median = float(np.median(rejected_errors)) if rejected_errors else math.nan
            print(
                f"[peak-filter-error] frame={frame_key} "
                f"accepted_n={len(accepted_errors)} accepted_mean={accepted_error_mean:.6g} "
                f"accepted_median={accepted_error_median:.6g} "
                f"rejected_n={len(rejected_errors)} rejected_mean={rejected_error_mean:.6g} "
                f"rejected_median={rejected_error_median:.6g}"
            )

    print(f"[{frame_key}] done. elapsed(s)= {round(time.time() - frame_t0, 2)}")
    
    return surface_points, {
        "rays_total": float(rays.size),
        "rays_with_peaks": float(rays_with_peaks),
        "rays_fused": float(rays_fused),
        "posterior_evals": float(posterior_evals),
        "filter_total_peaks": float(filter_total_peaks),
        "filter_accepted_peaks": float(filter_accepted_peaks),
        "filter_rejected_peaks": float(filter_rejected_peaks),
        "filter_rejected_by_delta_ll": float(filter_rejected_by_delta_ll),
        "filter_rejected_by_alpha": float(filter_rejected_by_alpha),
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
    ap.add_argument("--range_max", type=float, default=7.0)
    ap.add_argument("--z_min", type=float, default=-7.0)
    ap.add_argument("--z_max", type=float, default=7.0)
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
    ap.add_argument("--p_occ", type=float, default=0.65)
    ap.add_argument("--p_free", type=float, default=0.45)
    # 全局更新强度系数,每条 ray 更新 occupancy grid 时，不要用完整强度，而是乘一个较小系数
    ap.add_argument("--update-scale", type=float, default=0.01, help="Global multiplier for each ray log-odds update; values below 1 reduce saturation.")
    # 参考 ray 数量,如果当前帧 ray 数量不超过 10000，就不缩小;如果当前帧 ray 数量超过 10000，就按比例缩小每条 ray 的更新强度
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
    ap.add_argument(
        "--likelihood-model",
        choices=["auto", "clear", "smoke"],
        default="auto",
        help="Histogram likelihood model: auto uses fog_model ray_integral metadata when present.",
    )
    ap.add_argument(
        "--n-pulses",
        type=int,
        default=None,
        help="Laser pulse count for count-domain smoke likelihood; overrides capture_model metadata.",
    )
    
    # 启用 H1/H0 表面证据过滤
    ap.add_argument(
        "--smoke-peak-filter",
        action="store_true",
        help="Filter smoke-likelihood peaks using MAP H1-vs-H0 evidence before occupancy fusion.",
    )
    
    # delta_ll 的最低阈值
    ap.add_argument(
        "--surface-dll-min",
        type=float,
        default=10.0,
        help="Minimum LL(H1)-LL(H0) required to retain a smoke-likelihood peak.",
    )
    
    # 拟合表面幅度 alpha 的最低阈值
    ap.add_argument(
        "--surface-alpha-min",
        type=float,
        default=0.5,
        help="Minimum fitted effective surface amplitude required to retain a smoke-likelihood peak.",
    )
    
    # 加上它后程序会打印每帧过滤统计信息
    ap.add_argument(
        "--print-peak-filter-stats",
        action="store_true",
        help="Print per-frame smoke peak-filter counts and distribution summaries.",
    )
    
    # 每个被评估的 peak proposal（峰候选）写成一行
    ap.add_argument(
        "--peak-filter-details-csv",
        default=None,
        help="Optional CSV path for one diagnostic row per evaluated smoke peak.",
    )
    ap.add_argument("--range_model", choices=["range", "z"], default="range")
    ap.add_argument("--max_peaks", type=int, default=3)
    ap.add_argument("--mp_thr", type=float, default=2.0)
    ap.add_argument("--mp_support", type=int, default=0, help="support radius in bins, 0=auto(4*sigma)")
    ap.add_argument("--export-min-prob", type=float, default=0.51)
    ap.add_argument("--max_out", type=int, default=0)
    ap.add_argument("--grid-out", default=None, help="Optional .npz dump of the full occupancy log-odds grid for later thresholding.")
    ap.add_argument("--grid-ply-out", default=None, help="Optional PLY dump of all voxels with an occupancy scalar field for CloudCompare thresholding.")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--image-y-axis", choices=["auto", "down", "up"], default="auto")
    ap.add_argument(
        "--diagnostic-checkpoints",
        default="all",
        help="Comma-separated frame counts for grid diagnostics; use 'all' for the final frame.",
    )
    ap.add_argument("--profile", action="store_true", help="Print timing breakdown for major pipeline stages.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("surface_dll_min", "surface_alpha_min"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    multi_frame = args.npz_dir is not None
    if multi_frame and args.poses is None:
        raise ValueError("--poses is required when using --npz-dir")
    # 多帧模式下不建议使用 range_model == "z"
    if multi_frame and args.range_model == "z":
        print(
            "warning: --range_model z uses world z in the current dense-grid updater; "
            "ICL multi-frame data should use --range_model range."
        )

    if multi_frame:
        # 去 args.npz_dir 目录下找所有 .npz 文件，并按文件名排序,frames 是一个列表，里面放的是每一帧 .npz 文件的路径
        frames = discover_npz_frames(args.npz_dir, int(args.max_frames))
        # poses 是一个字典，保存每一帧对应的相机位姿矩阵,其中 key 是帧名，value 是这一帧的 4x4 相机位姿矩阵
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
    # 生成诊断检查点,diagnostic_checkpoints 是一个 list[int]整数列表,表示处理到哪些帧时打印 grid 诊断信息
    diagnostic_checkpoints = parse_diagnostic_checkpoints(args.diagnostic_checkpoints, len(frames))

    voxel = float(args.voxel)
    # 得到地图边界,mins 是一个长度为 3 的 NumPy 数组,mins = np.array([min_x, min_y, min_z])表示地图在世界坐标系中的最小边界坐标
    mins, maxs = resolve_map_bounds(args, multi_frame)
    # maxs - mins：计算 x、y、z 三个方向的总长度,除以体素边长得到每个方向需要的体素数量
    # 例如mins = [-7, -7, -7],maxs = [ 7,  7,  7],voxel = 0.1,那么maxs - mins = [14, 14, 14],dims = [140, 140, 140]
    # 也就是创建一个Lgrid.shape = (140, 140, 140)的三维占用栅格
    dims = np.ceil((maxs - mins) / voxel).astype(np.int64)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    print("grid mins =", mins, "maxs =", maxs)
    print("grid dims =", (nx, ny, nz), "voxel =", voxel)
    # 创建一个三维数组Lgrid，形状为(nx, ny, nz)，每个元素对应一个体素,初始值全为 0，对应先验占用概率 0.5
    Lgrid = np.zeros((nx, ny, nz), dtype=np.float32)

    # 收集所有帧检测到的表面点云，每个元素是一个包含 4 个浮点数的元组：(x, y, z, conf)
    all_surface_points: List[Tuple[float, float, float, float]] = []
    # 用来累计整个多帧建图过程中的统计量
    # rays_total:所有帧中实际处理过的 ray（射线）总数;rays_with_peaks:成功检测到至少一个 peak（峰）的 ray（射线）数量;
    # rays_fused:最终真正进入 occupancy fusion（占据融合）的 ray（射线）数量;posterior_evals:总共计算了多少次 peak posterior（峰距离后验概率）
    # filter_total_peaks:总共进入 smoke peak filter（烟雾峰过滤器）评估的峰数量
    # t_peak:所有帧中峰值检测阶段的总耗时;t_posterior:所有帧中后验似然计算阶段的总耗时
    # t_merge:所有帧中后验合并与熵计算阶段的总耗时;t_surface:所有帧中表面点生成阶段的总耗时;t_dda:所有帧中 DDA 地图更新阶段的总耗时
    # 所有帧处理完成后，如果开启了--profile参数，这些统计值会被打印出来，方便分析性能瓶颈
    totals = {
        "rays_total": 0.0,
        "rays_with_peaks": 0.0,
        "rays_fused": 0.0,
        "posterior_evals": 0.0,
        "filter_total_peaks": 0.0,
        "filter_accepted_peaks": 0.0,
        "filter_rejected_peaks": 0.0,
        "filter_rejected_by_delta_ll": 0.0,
        "filter_rejected_by_alpha": 0.0,
        "t_peak": 0.0,
        "t_posterior": 0.0,
        "t_merge": 0.0,
        "t_surface": 0.0,
        "t_dda": 0.0,
    }
    t0 = time.time()
    peak_filter_warning_state = {"clear_skip_emitted": False}
    peak_filter_csv_file = None
    peak_filter_csv_writer = None
    if args.peak_filter_details_csv:
        csv_path = Path(args.peak_filter_details_csv)
        if csv_path.parent != Path("."):
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        peak_filter_csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "frame", "row", "col", "peak_bin", "peak_score", "r_hat",
            "alpha", "beta_h1", "ll_h1", "beta_h0", "ll_h0", "delta_ll",
            "accepted", "rejected_by_delta_ll", "rejected_by_alpha",
            "gt_range", "abs_range_error",
        ]
        peak_filter_csv_writer = csv.DictWriter(peak_filter_csv_file, fieldnames=fieldnames)
        peak_filter_csv_writer.writeheader()
        print(f"peak-filter details CSV: {csv_path}")
    # 用来累计保存所有已处理帧的 3D bounding box
    accumulated_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None

    for frame_i, frame_path in enumerate(frames, start=1):
        # 取出当前帧文件的文件名，就是去掉.npz后缀，也就是帧名，这个帧名会用来从poses字典中查找对应的相机位姿矩阵
        frame_key = frame_path.stem
        print(f"loading frame {frame_i}/{len(frames)}: {frame_path}")
        # 从当前这一帧的 .npz 文件中读取 SPAD 直方图数据，以及可选的相机模型元数据
        spad_hist, frame_metadata = load_spad_frame_npz(frame_path)
        gt_range = None
        if bool(args.smoke_peak_filter) and (
            bool(args.print_peak_filter_stats) or peak_filter_csv_writer is not None
        ):
            gt_range = load_optional_gt_range_npz(frame_path)
            if gt_range is not None and gt_range.shape != spad_hist.shape[:2]:
                raise ValueError(
                    f"{frame_path}: gt_range shape {gt_range.shape} does not match "
                    f"spad_hist spatial shape {spad_hist.shape[:2]}"
                )
        # surface_points：当前帧估计出来的表面点列表，例如surface_points = [
        #     (0.2, -0.1, 1.5, 0.08),
        #     (0.5,  0.3, 2.0, 0.10),
        #     (-0.4, 0.2, 1.2, 0.06),
        # ]
        # stats：当前帧的统计信息字典,stats = {
        #     "rays_total": ...,    当前帧实际处理了多少条 ray
        #     "rays_with_peaks": ...,    有检测到峰值的 ray 数量
        #     "rays_fused": ...,    真正融合进 occupancy grid 的 ray 数量
        #     "posterior_evals": ...,    做了多少次距离后验计算
        #     "t_peak": ...,    各种耗时
        #     "t_posterior": ...,
        #     "t_merge": ...,
        #     "t_surface": ...,
        #     "t_dda": ...,
        # }
        surface_points, stats = process_frame(
            frame_key,
            spad_hist,
            frame_metadata,
            poses[frame_key],
            Lgrid,
            mins,
            voxel,
            args,
            gt_range=gt_range,
            peak_filter_csv_writer=peak_filter_csv_writer,
            peak_filter_warning_state=peak_filter_warning_state,
        )

        # 当前帧生成的所有表面点在世界坐标系中的 3D bounding box,例如frame_bbox = (
        #     np.array([-0.4, -0.1, 1.2]),    所有点在 x/y/z 三个方向上的最小坐标
        #     np.array([ 0.5,  0.3, 2.0])     所有点在 x/y/z 三个方向上的最大坐标
        # )
        frame_bbox = bbox_from_points(surface_points)
        # 当前帧的点云bounding box frame_bbox，和之前所有帧累计bounding box accumulated_bbox 的重叠程度
        # 用于检查当前帧生成的表面点范围，和之前累计的点云范围是否接近,如果 overlap 较高，说明当前帧点云空间范围和已有点云有较多重叠
        overlap = bbox_overlap_ratio(frame_bbox, accumulated_bbox)
        print(
            f"[{frame_key}] surface bbox {format_bbox(frame_bbox)} "
            f"overlap_with_previous={overlap:.4f}"
        )
        # 把“之前累计的包围盒” accumulated_bbox 和“当前帧的包围盒” frame_bbox 合并，得到新的累计包围盒
        accumulated_bbox = merge_bboxes(accumulated_bbox, frame_bbox)
        all_surface_points.extend(surface_points)
        for key in totals:
            totals[key] += float(stats.get(key, 0.0))
        # 如果当前处理到的帧编号 frame_i 在诊断检查点列表里，就打印一次当前 Lgrid 的诊断统计
        if frame_i in diagnostic_checkpoints:
            print_grid_diagnostics(f"frame {frame_i}", Lgrid, float(args.export_min_prob))

    if peak_filter_csv_file is not None:
        peak_filter_csv_file.close()

    print("all frames done. elapsed(s)=", round(time.time() - t0, 2))
    if bool(args.smoke_peak_filter):
        print(
            "peak-filter totals: "
            f"total_peaks={int(totals['filter_total_peaks'])} "
            f"accepted_surface_peaks={int(totals['filter_accepted_peaks'])} "
            f"rejected_peaks={int(totals['filter_rejected_peaks'])} "
            f"rejected_by_delta_ll={int(totals['filter_rejected_by_delta_ll'])} "
            f"rejected_by_alpha={int(totals['filter_rejected_by_alpha'])}"
        )
    print("accumulated surface bbox", format_bbox(accumulated_bbox))
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
