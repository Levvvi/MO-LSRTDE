import numpy as np
import matplotlib.pyplot as plt

# 修复 pymoo 编译警告：关闭未编译模块的提示
from pymoo.config import Config
Config.warnings['not_compiled'] = False

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.problems.many.wfg import WFG3


# 修复 matplotlib 显示中文乱码问题：指定支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体，例如 SimHei
plt.rcParams['axes.unicode_minus'] = False    # 保证负号正常显示

PROBLEM_NAME = "SYM-PART Simple"  # 可修改为其他问题名称

def extract_pareto(F: np.ndarray) -> np.ndarray:
    """
    从目标值矩阵 F 中提取非支配解（整体帕累托前沿）
    """
    n_points = F.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if is_efficient[i]:
            for j in range(n_points):
                if i != j and is_efficient[j]:
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_efficient[i] = False
                        break
    return F[is_efficient]

def compute_spacing(F: np.ndarray) -> float:
    """
    计算 Spacing 指标，用于反映解集在目标空间中的均匀性
    """
    F = np.array(F)
    N = F.shape[0]
    if N <= 1:
        return 0.0
    distances = []
    for i in range(N):
        diff = F - F[i]
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = np.inf  # 自身距离设为无穷大
        distances.append(dist.min())
    distances = np.array(distances)
    d_mean = distances.mean()
    return np.sqrt(np.sum((distances - d_mean) ** 2) / (N - 1))

def run_nsga2_multiple_times(num_runs: int = 50, pop_size: int = 100, n_gen: int = 200, seed: int = 1):
    """
    多次运行 NSGA2 算法，并计算各项指标
    :param num_runs: 运行次数
    :param pop_size: 种群大小
    :param n_gen: 迭代代数（注意：总评价次数 = pop_size * n_gen）
    :param seed: 随机种子
    :return: 运行结果列表, 指标字典, 真实 Pareto 前沿
    """
    results = []
    hv_values = []
    igd_values = []
    spacing_values = []

    # 获取测试问题，例如 dtlz1
    problem = WFG3(n_var=10, n_obj=3)

    # 获取真实 Pareto 前沿，并构造参考点（通常为真实前沿最大值的1.1倍）
    pf_true = problem.pareto_front(use_cache=False)
    if pf_true is None:
        raise ValueError("无法获取真实帕累托前沿，请手动指定参考点。")
    ref_point = np.max(pf_true, axis=0) * 1.1

    for run in range(num_runs):
        print(f"运行测试 {run + 1}/{num_runs} ...")
        algorithm = NSGA2(pop_size=pop_size)
        res = minimize(problem,
                       algorithm,
                       ('n_gen', n_gen),
                       seed=seed,
                       verbose=False)
        results.append(res)

        # 计算 Hypervolume 指标
        hv_indicator = Hypervolume(ref_point=ref_point)
        hv = hv_indicator.do(res.F)
        hv_values.append(hv)

        # 计算 IGD 指标
        igd_indicator = IGD(pf_true)
        igd = igd_indicator.do(res.F)
        igd_values.append(igd)

        # 计算 Spacing 指标
        spacing = compute_spacing(res.F)
        spacing_values.append(spacing)

    metrics = {
        "HV_mean": np.mean(hv_values),
        "HV_std": np.std(hv_values),
        "IGD_mean": np.mean(igd_values),
        "IGD_std": np.std(igd_values),
        "Spacing_mean": np.mean(spacing_values),
        "Spacing_std": np.std(spacing_values)
    }

    return results, metrics, pf_true

def plot_results_2d(all_F: np.ndarray, overall_pf: np.ndarray, pf_true: np.ndarray):
    """
    绘制 2D 目标空间图（适用于目标个数为2的情况）
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(all_F[:, 0], all_F[:, 1], color="gray", alpha=0.5, label="所有获得的点")
    plt.scatter(overall_pf[:, 0], overall_pf[:, 1], color="blue", s=50, label="整体帕累托前沿")
    plt.plot(pf_true[:, 0], pf_true[:, 1], color="red", lw=2, label="真实帕累托前沿")
    plt.xlabel("目标1")
    plt.ylabel("目标2")
    plt.title(PROBLEM_NAME + " 2D 可视化")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_results_3d(all_F: np.ndarray, overall_pf: np.ndarray, pf_true: np.ndarray):
    """
    绘制 3D 目标空间图（适用于目标个数为3的情况）
    """
    from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图模块
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_F[:, 0], all_F[:, 1], all_F[:, 2], color="gray", alpha=0.5, label="所有获得的点")
    ax.scatter(overall_pf[:, 0], overall_pf[:, 1], overall_pf[:, 2], color="blue", s=50, label="整体帕累托前沿")
    ax.plot(pf_true[:, 0], pf_true[:, 1], pf_true[:, 2], color="red", lw=2, label="真实帕累托前沿")
    ax.set_xlabel("目标1")
    ax.set_ylabel("目标2")
    ax.set_zlabel("目标3")
    plt.title(PROBLEM_NAME + " 3D 可视化")
    plt.legend()
    plt.show()

def main():
    # 参数设置
    num_runs = 50     # 运行次数
    pop_size = 100    # 种群大小
    n_gen = 100       # 迭代代数（注意：总评价次数 = pop_size * n_gen）
    seed = 1         # 随机种子

    # 多次运行 NSGA2 算法，并获取结果与指标
    results, metrics, pf_true = run_nsga2_multiple_times(num_runs=num_runs,
                                                          pop_size=pop_size,
                                                          n_gen=n_gen,
                                                          seed=seed)

    # 输出指标统计结果
    print("指标统计结果：")
    print("{:.4f} / {:.4f}".format(metrics["HV_mean"], metrics["HV_std"]))
    print("{:.4f} / {:.4f}".format(metrics["IGD_mean"], metrics["IGD_std"]))
    print("{:.4f} / {:.4f}".format(metrics["Spacing_mean"], metrics["Spacing_std"]))

    # 合并所有运行得到的目标值，并提取整体帕累托前沿
    all_F = np.vstack([res.F for res in results])
    overall_pf = extract_pareto(all_F)

    # 根据目标数选择可视化方式：2D 或 3D
    num_objectives = pf_true.shape[1]
    if num_objectives == 3:
        plot_results_3d(all_F, overall_pf, pf_true)
    elif num_objectives == 2:
        plot_results_2d(all_F, overall_pf, pf_true)
    else:
        print("当前问题的目标数为 {}，不支持可视化！".format(num_objectives))

if __name__ == '__main__':
    main()
