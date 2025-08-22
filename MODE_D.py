import numpy as np
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
# 3D 绘图的 Axes3D 留着，以备将来跑 3 目标问题时用
from mpl_toolkits.mplot3d import Axes3D

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

##############################
#        配置区（和你原来几乎一样）
##############################
NUM_RUNS = 50              # 重复运行次数
N_GEN = 100               # 迭代代数（n_gen）
SEED_START = 1            # 随机种子起始值
PROBLEM_NAME = "zdt4"     # 这次我们关注 ZDT4
OBJ_NUM = 2               # ZDT4 是 2 个目标
PARTITIONS = 12           # 参考方向分区数
NEIGHBORS = 15            # MOEAD 算法中邻居个数
PROB_NEIGHBOR_MATING = 0.7  # 邻域交配概率
FIGURE_SIZE = (8, 6)      # 可视化图形尺寸

##############################
#       Problem Setup
##############################
problem = get_problem(PROBLEM_NAME)
# 如果你想试 WFG3，就把下面一行取消注释，注释上一行
# problem = WFG3(n_var=10, n_obj=3)

# 根据 OBJ_NUM (2) 和 PARTITIONS(12) 来生成一组均匀分布的参考方向
ref_dirs = get_reference_directions("uniform", OBJ_NUM, n_partitions=PARTITIONS)

##############################
#      辅助函数
##############################

def compute_spacing(F: np.ndarray) -> float:
    """
    计算解集在目标空间的均匀性指标 Spacing
    """
    F = np.array(F)
    N = F.shape[0]
    if N <= 1:
        return 0.0
    distances = []
    for i in range(N):
        diff = F - F[i]
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = np.inf  # 排除自身距离
        distances.append(np.min(dist))
    d_mean = np.mean(distances)
    return np.sqrt(np.sum((distances - d_mean) ** 2) / (N - 1))

def extract_pareto(F: np.ndarray) -> np.ndarray:
    """
    从给定的一堆点 F（形状为 (n_points, n_obj)）中提取非支配前沿
    """
    n_points = F.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if is_efficient[i]:
            for j in range(n_points):
                if i != j and is_efficient[j]:
                    # 如果 F[j] 在所有目标上都不劣于 F[i]，且至少一个目标严格优于 F[i]，则 j 支配 i
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_efficient[i] = False
                        break
    return F[is_efficient]

def plot_results_3d(results: List, pf_true: np.ndarray = None):
    """
    绘制三维目标空间（仅当 OBJ_NUM == 3 时使用）：
      - 蓝色散点：算法得到的整体 Pareto 前沿
      - 红色曲线：真实 Pareto 前沿（如果提供的话）
    """
    # 将所有 run 中得到的 F 叠在一起
    all_F = np.vstack([res.F for res in results])
    overall_pf = extract_pareto(all_F)

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(overall_pf[:, 0], overall_pf[:, 1], overall_pf[:, 2],
               color="blue", s=50, label="Overall Pareto front")
    if pf_true is not None:
        ax.plot(pf_true[:, 0], pf_true[:, 1], pf_true[:, 2],
                color="red", lw=2, label="True Pareto front")
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_zlabel("Objective 3")
    ax.set_title(PROBLEM_NAME)
    ax.legend()
    ax.grid(True)
    plt.show()

def plot_results_2d(results: List, pf_true: np.ndarray = None):
    """
    绘制二维目标空间（适用于 OBJ_NUM==2）：
      - 蓝色散点：算法得到的整体 Pareto 前沿
      - 红色曲线：真实 Pareto 前沿（如果提供的话）
    """
    # 合并所有 run 得到的 F，然后再筛一次非支配得到最终算法近似前沿
    all_F = np.vstack([res.F for res in results])
    overall_pf = extract_pareto(all_F)

    # 对 pf_true 按照第 1 个目标（f1）升序排序，保证画出的红线是连贯的
    if pf_true is not None:
        # pf_true 假设形状是 (n_points, 2)
        idx = np.argsort(pf_true[:, 0])
        pf_true_sorted = pf_true[idx]
    else:
        pf_true_sorted = None

    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111)

    # 画蓝点：算法近似的 Pareto 前沿
    ax.scatter(overall_pf[:, 0], overall_pf[:, 1],
               color="blue", s=50, label="Overall Pareto front")
    # 画红线：真实 Pareto 前沿
    if pf_true_sorted is not None:
        ax.plot(pf_true_sorted[:, 0], pf_true_sorted[:, 1],
                color="red", lw=2, label="True Pareto front")

    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_title(PROBLEM_NAME)
    ax.legend()
    ax.grid(True)
    plt.show()

##############################
#     MOEAD 算法运行函数
##############################

def moead_run(seed: int) -> 'Result':
    """
    单次运行 MOEAD 算法
    """
    algorithm = MOEAD(
        ref_dirs,
        n_neighbors=NEIGHBORS,
        prob_neighbor_mating=PROB_NEIGHBOR_MATING,
    )
    res = minimize(problem,
                   algorithm,
                   ('n_gen', N_GEN),
                   seed=seed,
                   verbose=False)
    return res

def run_algorithm_multiple_times(num_runs: int = 5) -> Tuple[List, dict]:
    """
    多次运行 MOEAD 算法，并计算 Hypervolume、IGD 与 Spacing 指标的均值和标准差
    """
    results = []
    hv_values, igd_values, spacing_values = [], [], []

    # 先从问题本身拿到真 Pareto 前沿（用于后面做 Hypervolume、IGD 的参考，以及最后画图）
    # 这里暂时不指定 n_pareto_points，让下面再重新调用一遍
    pf_tmp = problem.pareto_front(use_cache=False)
    if pf_tmp is not None:
        ref_point = np.max(pf_tmp, axis=0) * 1.1
    else:
        raise ValueError("未指定参考点且 problem.pareto_front() 返回空，请手动设置")

    for run in range(num_runs):
        print(f"运行 {run + 1}/{num_runs}...")
        res = moead_run(seed=SEED_START + run)
        results.append(res)

        # 计算 Hypervolume
        hv_indicator = Hypervolume(ref_point=ref_point)
        hv_values.append(hv_indicator.do(res.F))

        # 计算 IGD
        if pf_tmp is not None:
            igd_indicator = IGD(pf_tmp)
            igd_values.append(igd_indicator.do(res.F))

        # 计算 Spacing
        spacing_values.append(compute_spacing(res.F))

    metrics = {
        "HV_mean": np.mean(hv_values),
        "HV_std": np.std(hv_values),
        "IGD_mean": np.mean(igd_values) if igd_values else None,
        "IGD_std": np.std(igd_values) if igd_values else None,
        "Spacing_mean": np.mean(spacing_values),
        "Spacing_std": np.std(spacing_values)
    }
    return results, metrics

##############################
#           主程序
##############################
if __name__ == '__main__':
    results, metrics = run_algorithm_multiple_times(num_runs=NUM_RUNS)
    print(f"{metrics['HV_mean']:.6f} / {metrics['HV_std']:.6f}")
    print(f"{metrics['IGD_mean']:.6f} / {metrics['IGD_std']:.6f}")
    print(f"{metrics['Spacing_mean']:.6f} / {metrics['Spacing_std']:.6f}")

    # ——关键修改：显式让 Pymoo 采样更多点来生成“平滑”的 ZDT4 真实前沿”——
    # n_pareto_points=500 表示我们要在 x1∈[0,1] 上平均采样 500 个点
    # 这样 f2=1-sqrt(f1) 画出来就非常平滑了
    pf_true = problem.pareto_front(n_pareto_points=500, use_cache=False)

    # 由于 ZDT4 是二维目标问题，所以调用 2D 绘图
    plot_results_2d(results, pf_true)
