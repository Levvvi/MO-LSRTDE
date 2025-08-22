import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
# 导入RE问题集中的问题
from reproblem import RE21, RE25, RE31, RE36, RE37

# 配置输出目录
text_dir = r"D:\PyProject\retext"
img_dir = r"D:\PyProject\repjpeg"
os.makedirs(text_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# 基本函数：提取非支配解和计算Spacing指标
def extract_pareto(F):
    """从一组目标函数值中提取非支配解（Pareto前沿）"""
    F = np.array(F)
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

def compute_spacing(F):
    """计算Spacing指标，衡量目标空间内解集的均匀性"""
    F = np.array(F)
    N = F.shape[0]
    if N <= 1:
        return 0.0
    distances = []
    for i in range(N):
        diff = F - F[i]
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = np.inf
        distances.append(dist.min())
    distances = np.array(distances)
    d_mean = np.mean(distances)
    return np.sqrt(np.sum((distances - d_mean) ** 2) / (N - 1))

# 自定义问题类，用于pymoo算法评估
from pymoo.core.problem import ElementwiseProblem
class REProblem(ElementwiseProblem):
    def __init__(self, problem_instance):
        super().__init__(n_var=problem_instance.n_variables,
                         n_obj=problem_instance.n_objectives,
                         n_constr=0,
                         xl=np.array(problem_instance.lbound, dtype=float),
                         xu=np.array(problem_instance.ubound, dtype=float))
        self.problem = problem_instance
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.problem.evaluate(x)

# ---------------------------
# 运行每个问题50次，并保存结果与图像
problems = [RE21(), RE25(), RE31(), RE36(), RE37()]
for problem in problems:
    name = problem.problem_name
    print(f"Running MOEA/D on {name}...")
    hv_values, igd_values, spacing_values = [], [], []
    all_run_pf = []
    # 根据目标数生成参考方向
    if problem.n_objectives == 2:
        ref_dirs = get_reference_directions("uniform", problem.n_objectives, n_partitions=99)
    else:
        ref_dirs = get_reference_directions("energy", problem.n_objectives, 100, seed=1)
    for run in range(1, 51):
        # 运行MOEA/D算法
        algorithm = MOEAD(ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)
        res = minimize(REProblem(problem),
                       algorithm,
                       ('n_gen', 100),
                       seed=run,
                       verbose=False)
        # 提取该次运行的Pareto前沿解集
        pf = extract_pareto(res.F)
        all_run_pf.append(pf)
        print(f"  Run {run} completed, number of Pareto solutions: {pf.shape[0]}")
    # 合并所有运行的解集并提取整体Pareto前沿
    combined_pf = np.vstack(all_run_pf)
    overall_pf = extract_pareto(combined_pf)
    # 参考点：每个目标的最差值的1.1倍
    ref_point = np.max(overall_pf, axis=0) * 1.1
    # 计算每次运行的指标值
    for pf in all_run_pf:
        hv_values.append(Hypervolume(ref_point=ref_point).do(pf))
        igd_values.append(IGD(overall_pf).do(pf))
        spacing_values.append(compute_spacing(pf))
    hv_values = np.array(hv_values)
    igd_values = np.array(igd_values)
    spacing_values = np.array(spacing_values)
    # 保存指标至文本文件
    metrics_file = os.path.join(text_dir, f"{name}_metrics_MODE_D.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Problem: {name}\n")
        f.write("Run\tHV\tIGD\tSpacing\n")
        for i in range(50):
            f.write(f"{i+1}\t{hv_values[i]:.6f}\t{igd_values[i]:.6f}\t{spacing_values[i]:.6f}\n")
        f.write("Mean\t")
        f.write(f"{hv_values.mean():.6f}\t{igd_values.mean():.6f}\t{spacing_values.mean():.6f}\n")
        f.write("StdDev\t")
        f.write(f"{hv_values.std():.6f}\t{igd_values.std():.6f}\t{spacing_values.std():.6f}\n")
    print(f"Saved metrics for {name} to {metrics_file}")
    # 绘制并保存帕累托前沿图像
    if problem.n_objectives == 2:
        plt.figure()
        plt.scatter(overall_pf[:, 0], overall_pf[:, 1], color="blue", s=30, label="Pareto front")
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.title(name)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(img_dir, f"{name}_MODE_D.jpg")
        plt.savefig(plot_path, dpi=300)
        plt.close()
    elif problem.n_objectives == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(overall_pf[:, 0], overall_pf[:, 1], overall_pf[:, 2], color="blue", s=30, label="Pareto front")
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        ax.set_title(name)
        ax.legend()
        plot_path = os.path.join(img_dir, f"{name}_MODE_D.jpg")
        plt.savefig(plot_path, dpi=300)
        plt.close()
    print(f"Saved Pareto front plot for {name} to {plot_path}\n")
