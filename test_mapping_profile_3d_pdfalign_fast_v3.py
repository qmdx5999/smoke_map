
import argparse
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

try:
    from numba import njit
except Exception:
    # fallback: run without numba (slow but works)
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco

DT_PS = 750.0
C = 299_792_458.0
dt = DT_PS * 1e-12
BIN_TO_M = C * dt / 2.0  # 44.519 mm/bin

# ========= Custom IRF (measured) =========
IRF_BINS = None   # 1D array, centered at peak (delta=0 at middle)
IRF_HALF = None   # half length in bins

def load_irf_1d(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        irf = np.load(path)
    else:
        irf = np.loadtxt(path)
    irf = np.asarray(irf, dtype=np.float64).reshape(-1)
    irf = np.maximum(irf, 0.0)
    s = irf.sum()
    if s > 0:
        irf = irf / s
    return irf

# ========================================

def logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def _gaussian_kernel_1d(sigma_bins: float, radius: int):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(max(sigma_bins, 1e-6))) ** 2)
    k /= (k.sum() + 1e-12)
    return k

def mp_find_peaks(hist: np.ndarray, max_peaks: int, sigma_bins: float,
                  score_thr: float, support_radius: int, min_sep: int):
    """
    MP peak picking with correlation. Uses numpy convolution (fast in C).
    """
    h_res = hist.copy()
    if IRF_BINS is not None and IRF_HALF is not None:
        rad = int(support_radius)
        lo = IRF_HALF - rad
        hi = IRF_HALF + rad + 1
        if lo >= 0 and hi <= len(IRF_BINS):
            kernel = IRF_BINS[lo:hi].copy()
            kernel = kernel / (kernel.sum() + 1e-12)
        else:
            kernel = _gaussian_kernel_1d(sigma_bins, support_radius)
    else:
        kernel = _gaussian_kernel_1d(sigma_bins, support_radius)

    peaks = []
    scores = []
    B = int(h_res.size)

    for _ in range(int(max_peaks)):
        corr = np.convolve(h_res.astype(np.float64), kernel, mode="same")
        t = int(np.argmax(corr))
        s = float(corr[t])
        if s < score_thr:
            break
        if any(abs(t - p) < min_sep for p in peaks):
            lo = max(0, t - support_radius)
            hi = min(B, t + support_radius + 1)
            h_res[lo:hi] = 0
            continue

        peaks.append(t)
        scores.append(s)

        lo = max(0, t - support_radius)
        hi = min(B, t + support_radius + 1)
        h_res[lo:hi] = 0

    order = np.argsort(peaks)
    peaks = [peaks[i] for i in order]
    scores = [scores[i] for i in order]
    return peaks, scores

@njit(cache=True, fastmath=True)
def _poisson_ll_numba(h, lam):
    s = 0.0
    for i in range(h.size):
        hi = h[i]
        li = lam[i]
        s += hi * math.log(li) - li
    return s

@njit(cache=True, fastmath=True)
def profile_ll_one_r_numba(h, S, eps=1e-6, max_iter=12):
    # h,S are 1D float64
    # init
    # median approx: use mean for speed (median is expensive in numba)
    beta = 0.0
    hmax = 0.0
    for i in range(h.size):
        beta += h[i]
        if h[i] > hmax:
            hmax = h[i]
    beta = max(beta / max(1, h.size), 0.0)
    a = max(hmax - beta, 0.0)

    smax = 0.0
    for i in range(S.size):
        if S[i] > smax:
            smax = S[i]
    if smax <= 0.0:
        lam = np.empty_like(h)
        for i in range(h.size):
            lam[i] = max(beta, eps)
        return _poisson_ll_numba(h, lam)
    a = a / max(smax, 1e-12)

    lam = np.empty_like(h)
    for _ in range(max_iter):
        for i in range(h.size):
            lam[i] = max(a * S[i] + beta, eps)

        g_a = 0.0
        g_b = 0.0
        for i in range(h.size):
            inv = h[i] / lam[i] - 1.0
            g_a += S[i] * inv
            g_b += inv
        if abs(g_a) + abs(g_b) < 1e-6:
            break

        H_aa = 0.0
        H_ab = 0.0
        H_bb = 0.0
        for i in range(h.size):
            w = h[i] / (lam[i] * lam[i])
            H_aa += -(S[i] * S[i]) * w
            H_ab += -(S[i]) * w
            H_bb += -(1.0) * w

        det = H_aa * H_bb - H_ab * H_ab
        if (not math.isfinite(det)) or abs(det) < 1e-18:
            break

        delta_a = ( H_bb * g_a - H_ab * g_b) / det
        delta_b = (-H_ab * g_a + H_aa * g_b) / det

        base_ll = _poisson_ll_numba(h, lam)

        step = 1.0
        ok = False
        for __ in range(10):
            a_new = a - step * delta_a
            b_new = beta - step * delta_b
            if a_new < 0.0: a_new = 0.0
            if b_new < 0.0: b_new = 0.0

            # ll_new
            for i in range(h.size):
                lam[i] = max(a_new * S[i] + b_new, eps)
            ll_new = _poisson_ll_numba(h, lam)
            if math.isfinite(ll_new) and ll_new >= base_ll - 1e-10:
                a = a_new
                beta = b_new
                ok = True
                break
            step *= 0.5
        if not ok:
            break

    for i in range(h.size):
        lam[i] = max(a * S[i] + beta, eps)
    return _poisson_ll_numba(h, lam)

@njit(cache=True, fastmath=True)
def build_S_gaussian(idx0, idx1, rbin, sigma_bins):
    n = idx1 - idx0
    S = np.empty(n, dtype=np.float64)
    s = max(sigma_bins, 1e-6)
    for i in range(n):
        delta = (idx0 + i) - rbin
        S[i] = math.exp(-0.5 * (delta / s) * (delta / s))
    return S

@njit(cache=True, fastmath=True)
def build_S_irf(idx0, idx1, rbin, irf_bins, irf_half):
    # linear interpolation on grid [-half..half]
    n = idx1 - idx0
    S = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = (idx0 + i) - rbin
        if x <= -irf_half or x >= irf_half:
            S[i] = 0.0
        else:
            # map x in (-half, half) to indices
            xf = x + irf_half
            j = int(xf)
            t = xf - j
            # j in [0..len-2]
            if j < 0:
                S[i] = irf_bins[0]
            elif j >= irf_bins.size - 1:
                S[i] = irf_bins[irf_bins.size - 1]
            else:
                S[i] = (1.0 - t) * irf_bins[j] + t * irf_bins[j + 1]
    return S

@njit(cache=True, fastmath=True)
def compute_ll_grid_numba(h, peak_bin, Wr_bin, M, win_half, sigma_bins, tau,
                         use_irf, irf_bins, irf_half, bin_to_m, bin_offset):
    B = h.size
    b0 = peak_bin - win_half
    if b0 < 0: b0 = 0
    b1 = peak_bin + win_half + 1
    if b1 > B: b1 = B

    nwin = b1 - b0
    h_w = np.empty(nwin, dtype=np.float64)
    for i in range(nwin):
        h_w[i] = h[b0 + i]

    ll = np.empty(M, dtype=np.float64)
    # r_bins linspace
    r0 = peak_bin - Wr_bin
    r1 = peak_bin + Wr_bin
    if M == 1:
        step = 0.0
    else:
        step = (r1 - r0) / (M - 1)

    for k in range(M):
        rb = r0 + step * k
        if use_irf:
            S = build_S_irf(b0, b1, rb, irf_bins, irf_half)
        else:
            S = build_S_gaussian(b0, b1, rb, sigma_bins)
        ll[k] = profile_ll_one_r_numba(h_w, S)

    # softmax(ll/tau) and cdf
    t = max(tau, 1e-6)
    # subtract max
    m = ll[0] / t
    for i in range(1, M):
        v = ll[i] / t
        if v > m: m = v
    p = np.empty(M, dtype=np.float64)
    s = 0.0
    for i in range(M):
        v = math.exp(ll[i] / t - m)
        p[i] = v
        s += v
    if s <= 0.0:
        for i in range(M):
            p[i] = 1.0 / M
    else:
        for i in range(M):
            p[i] /= s

    cdf = np.empty(M, dtype=np.float64)
    acc = 0.0
    for i in range(M):
        acc += p[i]
        cdf[i] = acc

    # r_grid_m
    r_grid_m = np.empty(M, dtype=np.float64)
    for i in range(M):
        r_grid_m[i] = ((r0 + step * i) - bin_offset) * bin_to_m

    return r_grid_m, cdf, p

@njit(cache=True, fastmath=True)
def peak_valley_bounds_numba(pdf, peak_idx):
    n = pdf.size
    l = peak_idx
    while l - 1 >= 0 and pdf[l - 1] <= pdf[l]:
        l -= 1
    r = peak_idx
    while r + 1 < n and pdf[r + 1] <= pdf[r]:
        r += 1
    return l, r

@njit(cache=True, fastmath=True)
def dda_update_dense(Lgrid, origin, d, end_dist, mins, voxel,
                     nx, ny, nz, rm_arr, rp_arr, K,
                     Lmin, Lmax, dL_free, dL_occ,
                     range_model_is_range):
    # DDA in voxel coords with mins offset
    # v0 is origin (assumed in same frame)
    v0x = (origin[0] - mins[0]) / voxel
    v0y = (origin[1] - mins[1]) / voxel
    v0z = (origin[2] - mins[2]) / voxel

    v1x = (origin[0] + d[0] * end_dist - mins[0]) / voxel
    v1y = (origin[1] + d[1] * end_dist - mins[1]) / voxel
    v1z = (origin[2] + d[2] * end_dist - mins[2]) / voxel

    x = int(math.floor(v0x))
    y = int(math.floor(v0y))
    z = int(math.floor(v0z))

    x1 = int(math.floor(v1x))
    y1 = int(math.floor(v1y))
    z1 = int(math.floor(v1z))

    dx = v1x - v0x
    dy = v1y - v0y
    dz = v1z - v0z

    sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
    sz = 1 if dz > 0 else (-1 if dz < 0 else 0)

    def tmax_tdelta(v0c, dvc, ic, step):
        if dvc == 0.0:
            return 1e30, 1e30
        next_boundary = (ic + 1) if step > 0 else ic
        tmax = (next_boundary - v0c) / dvc
        tdelta = 1.0 / abs(dvc)
        return tmax, tdelta

    tMaxX, tDeltaX = tmax_tdelta(v0x, dx, x, sx)
    tMaxY, tDeltaY = tmax_tdelta(v0y, dy, y, sy)
    tMaxZ, tDeltaZ = tmax_tdelta(v0z, dz, z, sz)

    # march
    max_steps = 20000
    for _ in range(max_steps):
        if x == x1 and y == y1 and z == z1:
            break

        if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
            # center in meters
            cx = mins[0] + voxel * (x + 0.5)
            cy = mins[1] + voxel * (y + 0.5)
            cz = mins[2] + voxel * (z + 0.5)

            if range_model_is_range:
                rho = d[0] * (cx - origin[0]) + d[1] * (cy - origin[1]) + d[2] * (cz - origin[2])
            else:
                rho = cz

            if rho > 0.0 and rho <= end_dist and K > 0:
                # piecewise Pv -> dL (unknown skip)
                updated = False

                # intervals assumed sorted, disjoint-ish
                for i in range(K):
                    rm = rm_arr[i]
                    rp = rp_arr[i]
                    if rho < rm:
                        # free
                        L = Lgrid[x, y, z] + dL_free
                        if L < Lmin: L = Lmin
                        if L > Lmax: L = Lmax
                        Lgrid[x, y, z] = L
                        updated = True
                        break
                    if rho <= rp:
                        # occ
                        L = Lgrid[x, y, z] + dL_occ
                        if L < Lmin: L = Lmin
                        if L > Lmax: L = Lmax
                        Lgrid[x, y, z] = L
                        updated = True
                        break
                    # between peaks: free
                    if i + 1 < K:
                        nrm = rm_arr[i + 1]
                        if rho > rp and rho < nrm:
                            L = Lgrid[x, y, z] + dL_free
                            if L < Lmin: L = Lmin
                            if L > Lmax: L = Lmax
                            Lgrid[x, y, z] = L
                            updated = True
                            break
                # after last peak -> unknown: no update

        # step
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default="", help="histogram txt path (U x B)")
    ap.add_argument("--npz", default="", help="SPAD .npz path containing a histogram array")
    ap.add_argument("--npz_key", default="spad_hist", help="histogram key inside --npz")
    ap.add_argument("--height", type=int, default=0, help="image height override for flat txt input")
    ap.add_argument("--width", type=int, default=0, help="image width override for flat txt input")
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--range_max", type=float, default=4.0, help="local map half-size in x/y, meters; marching end as well")
    ap.add_argument("--z_min", type=float, default=0.0)
    ap.add_argument("--z_max", type=float, default=4.0)
    ap.add_argument("--max_rays", type=int, default=3000, help="subsample rays")
    ap.add_argument("--peak_thr", type=float, default=50.0, help="skip rays with max count < thr")
    ap.add_argument("--Wr_bin", type=int, default=12)
    ap.add_argument("--M", type=int, default=81)
    ap.add_argument("--p_occ", type=float, default=0.70)
    ap.add_argument("--p_free", type=float, default=0.35)
    ap.add_argument("--p0", type=float, default=0.50)
    ap.add_argument("--Lmin", type=float, default=-5.0)
    ap.add_argument("--Lmax", type=float, default=5.0)
    ap.add_argument("--z_slice", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--win_half", type=int, default=25)
    ap.add_argument("--sigma_bins", type=float, default=2.0)
    ap.add_argument("--irf", type=str, default="", help="path to measured IRF (.npy/.txt). If empty, use Gaussian.")
    ap.add_argument("--range_model", choices=["range","z"], default="range")
    ap.add_argument("--max_peaks", type=int, default=1)
    ap.add_argument("--mp_thr", type=float, default=5.0)
    ap.add_argument("--mp_support", type=int, default=0, help="support radius in bins, 0=auto(4*sigma)")
    ap.add_argument("--occ_wmax_vox", type=float, default=2.5, help="cap each occupied interval width (voxels)")
    ap.add_argument("--max_out", type=int, default=0, help="max exported PLY points, 0=no limit")
    ap.add_argument("--dt_ps", type=float, default=DT_PS, help="time-bin width in ps, used if --bin_to_m and --tmax_ns are not set")
    ap.add_argument("--tmax_ns", type=float, default=0.0, help="total time span in ns; bin_to_m = c*tmax/(2*B)")
    ap.add_argument("--bin_to_m", type=float, default=0.0, help="direct meters-per-bin calibration")
    ap.add_argument("--bin_offset", type=float, default=0.0, help="range zero offset in bins")
    args = ap.parse_args()

    if bool(args.txt) == bool(args.npz):
        raise ValueError("Specify exactly one input: --txt or --npz")

    if args.npz:
        print("loading npz ...")
        with np.load(args.npz, allow_pickle=False) as npz:
            if args.npz_key not in npz.files:
                raise KeyError(f"key {args.npz_key!r} not found in {args.npz}; keys={npz.files}")
            hist = np.asarray(npz[args.npz_key])
        if hist.ndim != 3:
            raise ValueError(f"{args.npz_key} must have shape (H, W, B), got {hist.shape}")
        H, W, B = hist.shape
        data = hist.reshape(H * W, B).astype(np.int64, copy=False)
        print(f"loaded {args.npz}:{args.npz_key}, H={H}, W={W}, B={B}")
    else:
        print("loading txt ...")
        data = np.loadtxt(args.txt, dtype=np.int64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # U：总共有多少条激光射线（对应多少个像素），B：每条射线有多少个时间 bin
        U, B = data.shape
        if args.height > 0 and args.width > 0:
            H, W = int(args.height), int(args.width)
        else:
            H, W = 192, 256
        if U != H * W:
            print(f"[WARN] U != H*W, got U={U}, H={H}, W={W}, H*W={H*W}")

    if args.bin_to_m > 0.0:
        bin_to_m = float(args.bin_to_m)
        bin_source = "--bin_to_m"
    elif args.tmax_ns > 0.0:
        bin_to_m = C * (float(args.tmax_ns) * 1e-9) / (2.0 * float(B))
        bin_source = "--tmax_ns"
    else:
        bin_to_m = C * (float(args.dt_ps) * 1e-12) / 2.0
        bin_source = "--dt_ps"
    print(
        f"range calibration: bin_to_m={bin_to_m:.8g} m/bin "
        f"from {bin_source}, bin_offset={args.bin_offset:g}"
    )

    # IRF_BINS：一个数组，记录了你的传感器输出的脉冲形状，每个元素是对应时间 bin 的响应强度
    # IRF_HALF：一个整数，告诉你这个脉冲从中心峰值向左右两边各延伸了多少个 bin
    global IRF_BINS, IRF_HALF
    if args.irf:
        IRF_BINS = load_irf_1d(args.irf).astype(np.float64)
        IRF_HALF = (len(IRF_BINS) - 1) // 2
        print("[IRF] loaded:", args.irf, "len=", len(IRF_BINS), "half=", IRF_HALF)
    else:
        print("[IRF] Gaussian IRF, sigma_bins=", args.sigma_bins)

    # camera intrinsics (same as your slow script)
    fx, fy = 118.6514575329715, 118.7964934010577
    cx, cy = 130.6802784645003, 100.3605702468140
    # 内参矩阵 K 
    Kcam = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)
    k1, k2 = -0.257910069121181, 0.053237073644331
    p1, p2 = 0.0, 0.0
    # 畸变系数 D
    D = np.array([k1, k2, p1, p2], dtype=np.float64)

    # Precompute undistorted normalized coords for all pixels (BIG speedup)
    print("precomputing undistortPoints for all pixels ...")
    # u_grid：每个像素的 x 坐标（0~255），v_grid：每个像素的 y 坐标（0~191）
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    uv = np.stack([u_grid.reshape(-1), v_grid.reshape(-1)], axis=1).reshape(-1, 1, 2)
    # xy是一个(49152, 2)的数组，每个元素是对应像素的归一化坐标(x_n, y_n)
    xy = cv2.undistortPoints(uv, Kcam, D, P=None).reshape(-1, 2)
    x_n_all = xy[:, 0].astype(np.float64)
    y_n_all = xy[:, 1].astype(np.float64)

    voxel = float(args.voxel)  # 体素大小，单位：米
    # x 方向：从-range_max到range_max，y 方向：从-range_max到range_max，z 方向：从z_min到z_max
    mins = np.array([-args.range_max, -args.range_max, args.z_min], dtype=np.float64)  # 3D 网格的最小边界坐标
    maxs = np.array([ args.range_max,  args.range_max, args.z_max], dtype=np.float64)  # 3D 网格的最大边界坐标
    dims = np.ceil((maxs - mins) / voxel).astype(np.int64)  # 计算每个方向需要多少个体素
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])  # 三个方向的体素数量
    print("grid dims =", (nx, ny, nz), "voxel =", voxel)

    # Dense log-odds grid (float32 saves memory & is faster)
    Lgrid = np.zeros((nx, ny, nz), dtype=np.float32)  # 体素对数几率网格

    # Select rays
    mx = data.max(axis=1)  # 一个长度为 49152 的数组，每个元素是对应射线的最大光子计数
    cand = np.where(mx >= args.peak_thr)[0]  # 只保留那些最大光子计数超过阈值的射线
    print("candidate rays:", cand.size)
    if cand.size == 0:
        print("No rays pass threshold.")
        return
    if cand.size > args.max_rays:
        sel = np.linspace(0, cand.size - 1, args.max_rays).astype(np.int64)
        rays = cand[sel]
    else:
        rays = cand
    print("using rays:", rays.size)

    origin = np.zeros(3, dtype=np.float64)  # 激光雷达的位置，在局部地图坐标系中设为原点 (0,0,0)
    end_dist = float(args.range_max)  # 射线遍历的最大距离，单位：米
    range_model_is_range = (args.range_model == "range")

    # Precompute increments
    # 预计算对数几率更新增量，把公式 12 里固定不变的部分提前算好
    dL_free = float(logit(args.p_free) - logit(args.p0))
    dL_occ  = float(logit(args.p_occ)  - logit(args.p0))
    # 对数几率的上下限
    Lmin = float(args.Lmin)
    Lmax = float(args.Lmax)

    support = int(args.mp_support) if args.mp_support > 0 else int(np.ceil(4.0 * args.sigma_bins))  # 峰值检测的支持半径，单位是 bin
    min_sep = support  # 两个峰值之间的最小距离，单位是 bin

    use_irf = (IRF_BINS is not None and IRF_HALF is not None)
    irf_bins_nb = IRF_BINS if use_irf else np.empty(1, dtype=np.float64)
    irf_half_nb = int(IRF_HALF) if use_irf else 0

    t0 = time.time()
    # k：当前处理到第几条射线；idx：这条射线在原始数据中的索引；h：这条射线的原始光子直方图，长度为 672 的一维数组
    for k, idx in enumerate(rays):
        h = data[idx].astype(np.float64)

        # peaks：检测到的峰值位置列表，每个元素是一个 bin 索引
        peaks, _ = mp_find_peaks(
            h,
            max_peaks=args.max_peaks,  # 每条射线最多检测多少个峰值，默认值为1
            sigma_bins=args.sigma_bins,  # 高斯 IRF 的标准差，单位是 bin，默认值为2.0
            score_thr=args.mp_thr,
            support_radius=support,
            min_sep=min_sep
        )
        if len(peaks) == 0:
            continue

        # Build intervals (valley on posterior)
        # 存储这条射线所有的占用区间，每个区间是一个元组(rm, rp)，表示墙可能存在于[rm, rp]米之间
        intervals = []
        # 每个占用区间的最大允许宽度，单位是米
        wmax = float(args.occ_wmax_vox) * voxel
        # pk：当前处理的峰值在原始直方图中的 bin 索引
        for pk in peaks:
            # r_grid_m：距离网格，单位是米；pdf_grid：距离后验概率密度函数；cdf_grid：距离后验累积分布函数
            r_grid_m, cdf_grid, pdf_grid = compute_ll_grid_numba(
                h, int(pk),
                int(args.Wr_bin), int(args.M), int(args.win_half),
                float(args.sigma_bins), float(args.tau),
                use_irf, irf_bins_nb, irf_half_nb,
                float(bin_to_m), float(args.bin_offset)
            )
            pk_local = int(np.argmax(pdf_grid))
            l, r = peak_valley_bounds_numba(pdf_grid, pk_local)
            rm = float(r_grid_m[l])
            rp = float(r_grid_m[r])
            if rp - rm > wmax:
                c = 0.5 * (rm + rp)
                rm = c - 0.5 * wmax
                rp = c + 0.5 * wmax
            # clip
            if rp <= 0.0 or rm >= end_dist:
                continue
            rm = max(0.0, min(end_dist, rm))
            rp = max(0.0, min(end_dist, rp))
            if rp > rm:
                intervals.append((rm, rp))

        if len(intervals) == 0:
            continue
        intervals.sort(key=lambda t: t[0])

        # pack intervals to fixed-size arrays for numba
        Kint = min(len(intervals), 8)
        rm_arr = np.zeros(Kint, dtype=np.float64)
        rp_arr = np.zeros(Kint, dtype=np.float64)
        for i in range(Kint):
            rm_arr[i] = intervals[i][0]
            rp_arr[i] = intervals[i][1]

        # ray direction
        x_n = float(x_n_all[idx])
        y_n = float(y_n_all[idx])
        d = np.array([x_n, y_n, 1.0], dtype=np.float64)
        nrm = np.linalg.norm(d)
        if nrm <= 1e-12:
            continue
        d /= nrm

        dda_update_dense(
            Lgrid, origin, d, end_dist, mins, voxel,
            nx, ny, nz, rm_arr, rp_arr, Kint,
            Lmin, Lmax, dL_free, dL_occ,
            range_model_is_range
        )

        if (k + 1) % 5000 == 0:
            print("processed rays:", k + 1, "/", rays.size, "elapsed(s)=", round(time.time() - t0, 1))

    print("done. elapsed(s)=", round(time.time() - t0, 2))

    # slice plot
    izs = int(math.floor((args.z_slice - args.z_min) / voxel))
    if 0 <= izs < nz:
        slice_img = np.full((nx, ny), args.p0, dtype=np.float32)
        # vectorized sigmoid on slice
        Ls = Lgrid[:, :, izs].astype(np.float64)
        slice_img[:, :] = 1.0 / (1.0 + np.exp(-Ls))
        plt.figure(figsize=(8, 8))
        plt.imshow(slice_img.T, origin="lower", vmin=0.0, vmax=1.0)
        plt.title(f"Occupancy slice at z={args.z_slice:.2f} m")
        plt.colorbar()
        plt.savefig("occ_slice.png", dpi=200)
        print("saved occ_slice.png")

    # export ply (threshold on |L| to reduce size)
    thr_L = 1e-6
    xs, ys, zs = np.where(np.abs(Lgrid) > thr_L)
    print("active voxels:", xs.size)
    max_out = int(args.max_out)
    if max_out > 0 and xs.size > max_out:
        rng = np.random.default_rng(0)
        sel = rng.choice(xs.size, size=max_out, replace=False)
        xs = xs[sel]; ys = ys[sel]; zs = zs[sel]
    out_ply = "occupied_rgb.ply"
    with open(out_ply, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xs.size}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(xs.size):
            ix = int(xs[i]); iy = int(ys[i]); iz = int(zs[i])
            L = float(Lgrid[ix, iy, iz])
            p = 1.0 / (1.0 + math.exp(-L))
            g = int(max(0, min(255, round(p * 255.0))))
            cx = mins[0] + voxel * (ix + 0.5)
            cy = mins[1] + voxel * (iy + 0.5)
            cz = mins[2] + voxel * (iz + 0.5)
            f.write(f"{cx:.6f} {cy:.6f} {cz:.6f} {g} {g} {g}\n")
    print("saved", out_ply, "N=", xs.size)

if __name__ == "__main__":
    main()
