"""Analyze probability distribution of a saved occupancy log-odds grid."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help=".npz file produced by spad_npz_occupancy_mapping.py --grid-out")
    ap.add_argument("--hist-out", default="occupancy_prob_hist.png", help="Output histogram image path")
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--active-eps", type=float, default=1e-6, help="abs(log-odds) threshold for updated voxels")
    return ap.parse_args()


def summarize(name: str, p: np.ndarray) -> None:
    """
    量化分析占据概率的分布特征
    name: str：统计对象的名称（如 “all voxels”“active voxels”），用于控制台输出的标识
    p: np.ndarray：待统计的占据概率数组
    """
    if p.size == 0:
        print(f"{name}: empty")
        return
    # 计算概率数组的 9 个关键分位数:0%（最小值）,100%（最大值）,1%/5%/95%/99%（极端值，反映分布的尾部特征）
    # 25%/50%/75%（四分位数，反映分布的集中趋势，50% 是中位数）
    qs = np.percentile(p, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"{name}: count={p.size}")
    print(
        "  percentiles p0/p1/p5/p25/p50/p75/p95/p99/p100 = "
        + " ".join(f"{v:.6f}" for v in qs)
    )
    # 遍历一系列关键阈值（围绕 0.5—— 占据 / 未占据的分界点），统计每个阈值下，概率≥该值的体素数量
    for thr in [0.45, 0.48, 0.49, 0.50, 0.51, 0.52, 0.55, 0.60, 0.70]:
        print(f"  count p >= {thr:.2f}: {int(np.count_nonzero(p >= thr))}")


def main() -> None:
    args = parse_args()
    with np.load(args.grid) as data:
        Lgrid = np.asarray(data["Lgrid"], dtype=np.float64)

    # 逆运算，将对数几率转回 0~1 的占据概率,形状为(nx, ny, nz)，每个元素对应一个体素的占据概率（0~1）
    Pgrid = 1.0 / (1.0 + np.exp(-Lgrid))
    # Lgrid:三维对数几率网格，形状为(nx, ny, nz),对Lgrid中的每一个元素单独进行判断，生成一个形状和Lgrid完全相同的布尔数组
    # 如果它的log-odds 值的绝对值 > 1e-6 → 标记为True（活跃体素）;如果它的log-odds 值的绝对值 ≤ 1e-6 → 标记为False（先验体素）
    active = np.abs(Lgrid) > float(args.active_eps)
    # 经过reshape(-1)后变成一个一维数组，长度等于Pgrid所有维度的乘积
    P_all = Pgrid.reshape(-1)
    # 只包含被传感器更新过的活跃体素
    P_active = Pgrid[active]

    print("grid shape:", tuple(int(v) for v in Lgrid.shape))
    print("total voxels:", int(P_all.size))
    print("active voxels:", int(P_active.size))
    print("unknown/prior voxels:", int(P_all.size - P_active.size))
    summarize("all voxels", P_all)
    summarize("active voxels", P_active)

    out_dir = os.path.dirname(args.hist_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)
    axes[0].hist(P_all, bins=args.bins, range=(0.0, 1.0), color="#377eb8")
    axes[0].set_title("All voxels")
    axes[0].set_xlabel("occupancy probability")
    axes[0].set_ylabel("count")
    axes[0].set_yscale("log")

    axes[1].hist(P_active, bins=args.bins, range=(0.0, 1.0), color="#e41a1c")
    axes[1].set_title("Active voxels only")
    axes[1].set_xlabel("occupancy probability")
    axes[1].set_ylabel("count")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(0.51, color="gray", linestyle=":", linewidth=1.0)
        ax.axvline(0.55, color="gray", linestyle=":", linewidth=1.0)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.hist_out)
    print("saved", args.hist_out)


if __name__ == "__main__":
    main()
