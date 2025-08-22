import numpy as np
import random
from math import exp, tanh
from typing import Callable, List, Tuple
from pymoo.problems import get_problem
# 使用 pymoo 0.6.1.3 中的 Result 类
from pymoo.core.result import Result
# 直接从各指标模块导入
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.problems.multi.omnitest import OmniTest
from pymoo.visualization.scatter import Scatter
# -------------------- 配置部分 --------------------
# 使用 pymoo 的 get_problem 获取内置测试问题（例如 zdt4）
problem = get_problem("zdt3")


def fitness_function(x: np.ndarray) -> np.ndarray:
    """
    适应度计算函数：确保 x 为二维数组后调用问题的 evaluate 方法，
    返回第一行的目标函数值。
    """
    F = problem.evaluate(np.atleast_2d(x))
    return F[0]


# 根据 pymoo 问题设置决策变量维度和上下界
D = problem.n_var
lower_bound = problem.xl if isinstance(problem.xl, np.ndarray) else np.full(D, problem.xl)
upper_bound = problem.xu if isinstance(problem.xu, np.ndarray) else np.full(D, problem.xu)


# --------------------------------------------------

def non_dominated_sort(population: np.ndarray, fitness: np.ndarray) -> List[List[int]]:
    """
    非支配排序：将种群按 Pareto 等级划分
    """
    N = len(population)
    fronts = [[]]
    domination_count = np.zeros(N, dtype=int)
    dominated_solutions = [[] for _ in range(N)]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if all(fitness[p] <= fitness[q]) and any(fitness[p] < fitness[q]):
                dominated_solutions[p].append(q)
            elif all(fitness[q] <= fitness[p]) and any(fitness[q] < fitness[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def calculate_crowding_distance(front: List[int], fitness: np.ndarray) -> np.ndarray:
    """
    计算拥挤距离，用于环境选择中区分同一层个体
    """
    distance = np.zeros(len(front))
    num_objectives = fitness.shape[1]
    if len(front) == 0:
        return distance
    for m in range(num_objectives):
        front_fitness = [fitness[i][m] for i in front]
        sorted_indices = np.argsort(front_fitness)
        distance[sorted_indices[0]] = distance[sorted_indices[-1]] = float('inf')
        f_min = front_fitness[sorted_indices[0]]
        f_max = front_fitness[sorted_indices[-1]]
        if f_max - f_min == 0:
            continue
        for i in range(1, len(front) - 1):
            distance[sorted_indices[i]] += (front_fitness[sorted_indices[i + 1]] - front_fitness[
                sorted_indices[i - 1]]) / (f_max - f_min)
    return distance


def update_params(SR: float, MCr: float) -> Tuple[float, float]:
    """
    根据成功率 SR 更新变异因子 F 和交叉概率 Cr（无历史记忆机制）
    """
    # 计算当前成功率对应的变异因子均值 mF
    mF = 0.4 + 0.25 * tanh(5 * SR)
    # 从正态分布 N(mF, 0.02) 采样变异因子 F
    F = np.random.normal(mF, 0.02)
    # 将 F 限制在合理范围 [0.1, 0.9]（确保不取极端值 0 或 1）
    F = np.clip(F, 0.1, 0.9)
    # 以当前平均交叉率 MCr 为均值，从 N(MCr, 0.05) 采样交叉概率 Cr
    Cr = np.random.normal(MCr, 0.05)
    Cr = np.clip(Cr, 0, 1)
    return F, Cr



def update_memory(successful_F: List[float], successful_Cr: List[float], MF: np.ndarray, MCr: np.ndarray,
                  current_memory_index: int, H: int) -> int:
    """
    更新历史记忆
    """
    if successful_F:
        MF[current_memory_index] = np.mean(successful_F)
    if successful_Cr:
        MCr[current_memory_index] = np.mean(successful_Cr)
    current_memory_index = (current_memory_index + 1) % H
    return current_memory_index


def generate_population(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """
    生成随机种群
    """
    return np.random.rand(N, D) * (upper_bound - lower_bound) + lower_bound


def select_pbest(x_top: np.ndarray, fitness_top: np.ndarray, pb: float, N: int) -> np.ndarray:
    """
    选择 p-best 个体，基于目标函数值聚合排序
    """
    aggregate = np.sum(fitness_top, axis=1)
    sorted_indices = np.argsort(aggregate)
    pb_size = max(2, int(pb * len(x_top)))
    candidate_indices = sorted_indices[:pb_size]
    return np.random.choice(candidate_indices, size=N, replace=True)


def mutation_and_crossover(population: np.ndarray, F: float, Cr: float, pbest_population: np.ndarray,
                           lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """
    进行差分进化中的变异和交叉操作，生成新个体
    """
    N, D = population.shape
    u_new = np.copy(population)
    for i in range(N):
        indices = list(range(N))
        r1, r2, r3 = random.sample(indices, 3)
        pbest = pbest_population[i]
        v = population[r1] + F * (pbest - population[i]) + F * (population[r2] - population[r3])
        j_rand = random.randint(0, D - 1)
        for j in range(D):
            if random.random() <= Cr or j == j_rand:
                u_new[i, j] = v[j]
        u_new[i] = np.clip(u_new[i], lower_bound, upper_bound)
    return u_new


def environmental_selection(combined_population: np.ndarray, combined_fitness: np.ndarray, N: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    环境选择：根据非支配排序和拥挤距离选择下一代种群
    """
    fronts = non_dominated_sort(combined_population, combined_fitness)
    new_population_indices = []
    for front in fronts:
        if len(new_population_indices) + len(front) <= N:
            new_population_indices.extend(front)
        else:
            distances = calculate_crowding_distance(front, combined_fitness)
            sorted_front = sorted(front, key=lambda idx: distances[front.index(idx)], reverse=True)
            remaining = N - len(new_population_indices)
            new_population_indices.extend(sorted_front[:remaining])
            break
    new_population = combined_population[new_population_indices]
    new_fitness = combined_fitness[new_population_indices]
    return new_population, new_fitness


def multi_objective_LSRTDE(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                           max_evaluations: int, fitness_function: Callable) -> Result:
    """
    多目标 L-SRTDE 算法，实现基于成功率的参数自适应和双种群交互策略
    """
    NFE = 0  # 函数评价计数
    SR = 0.5  # 初始成功率
    MCr = 1.0  # 初始平均交叉率（无历史记忆，直接使用单一均值）

    # 初始化种群及适应度
    population = generate_population(N, D, lower_bound, upper_bound)
    fitness = np.array([fitness_function(ind) for ind in population])
    NFE += N

    # 初始化历史最优解集合（用于 p-best 选择）
    x_top = np.copy(population)
    fitness_top = np.copy(fitness)

    while NFE < max_evaluations:
        # 基于当前成功率计算贪婪参数 pb，并从历史最优集合 x_top 中选择 p-best 个体
        pb = 0.7 * exp(-7 * SR)
        pb_size = max(2, int(pb * len(x_top)))
        aggregate_top = np.sum(fitness_top, axis=1)
        sorted_indices_top = np.argsort(aggregate_top)
        candidate_indices = sorted_indices_top[:pb_size]
        pbest_population = x_top[np.random.choice(candidate_indices, size=N, replace=True)]

        # 更新当前代参数 F 和 Cr（基于 SR，自适应调整）
        F, Cr = update_params(SR, MCr)
        # 执行变异和交叉生成新子代
        u_new = mutation_and_crossover(population, F, Cr, pbest_population, lower_bound, upper_bound)
        u_fitness = np.array([fitness_function(ind) for ind in u_new])
        NFE += N

        # 计算当前代成功率：子代进入合并种群非支配层第一层的比例
        combined = np.vstack((population, u_new))
        combined_fit = np.vstack((fitness, u_fitness))
        new_fronts = non_dominated_sort(combined, combined_fit)
        if new_fronts[0]:
            num_success = sum(1 for idx in new_fronts[0] if idx >= len(population))
            SR = num_success / N
        else:
            SR = 0.0

        # 计算实际交叉率 Cr_a，并用其更新平均交叉率 MCr
        Cr_a = np.mean([np.any(u_new[i] != population[i]) for i in range(N)])
        MCr = Cr_a  # 更新交叉率均值为本代实际交叉比例

        # 环境选择：合并父代和子代，根据非支配排序和拥挤距离选择下一代种群
        combined_population = np.vstack((population, u_new))
        combined_fitness = np.vstack((fitness, u_fitness))
        population, fitness = environmental_selection(combined_population, combined_fitness, N)

        # 更新历史最优解集合 x_top：合并当前历史集与新子代，保留非支配解（若不足 N 则继续取下一层）
        combined_top = np.vstack((x_top, u_new))
        combined_top_fit = np.vstack((fitness_top, u_fitness))
        top_fronts = non_dominated_sort(combined_top, combined_top_fit)
        selected_indices = list(top_fronts[0])
        i = 1
        while len(selected_indices) < N and i < len(top_fronts):
            selected_indices.extend(top_fronts[i])
            i += 1
        if len(selected_indices) > N:
            selected_indices = selected_indices[:N]
        x_top = combined_top[selected_indices]
        fitness_top = combined_top_fit[selected_indices]

    # 返回结果（包含最后得到的帕累托解集）
    res = Result()
    res.X = x_top
    res.F = fitness_top
    return res


def compute_spacing(F):
    """
    计算 Spacing 指标，用于反映解集在目标空间的均匀性
    """
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
    d_mean = distances.mean()
    return np.sqrt(np.sum((distances - d_mean) ** 2) / (N - 1))


def run_algorithm_multiple_times(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                                 max_evaluations: int, fitness_function: Callable, num_runs: int = 5,
                                 ref_point=None) -> Tuple[List[Result], dict]:
    """
    多次运行算法，并计算 Hypervolume (HV)、IGD 和 Spacing 指标的均值与标准差
    返回:
        results: List[Result] 每次运行的结果
        metrics: dict 各指标的统计数据
    """
    if ref_point is None:
        # 自动获取参考点：取 Pareto 前沿的最大值乘以 1.1
        pareto_front = problem.pareto_front(use_cache=False)
        if pareto_front is not None:
            ref_point = np.max(pareto_front, axis=0) * 1.1
        else:
            raise ValueError("未提供 ref_point，且 problem.pareto_front() 为空，请手动指定参考点")

    results = []
    hv_values, igd_values, spacing_values = [], [], []

    for run in range(num_runs):
        print(f"运行测试 {run + 1}/{num_runs}...")
        res = multi_objective_LSRTDE(N, D, lower_bound, upper_bound, max_evaluations, fitness_function)
        results.append(res)

        # 计算 Hypervolume 指标
        hv_indicator = Hypervolume(ref_point=ref_point)
        hv_value = hv_indicator.do(res.F)
        hv_values.append(hv_value)

        # 计算 IGD 指标，自动获取 Pareto 前沿
        pf = problem.pareto_front(use_cache=False)
        if pf is not None:
            igd_indicator = IGD(pf)
            igd_value = igd_indicator.do(res.F)
            igd_values.append(igd_value)

        # 计算 Spacing 指标
        spacing_value = compute_spacing(res.F)
        spacing_values.append(spacing_value)

    metrics = {
        "HV_mean": np.mean(hv_values),
        "HV_std": np.std(hv_values),
        "IGD_mean": np.mean(igd_values) if igd_values else None,
        "IGD_std": np.std(igd_values) if igd_values else None,
        "Spacing_mean": np.mean(spacing_values),
        "Spacing_std": np.std(spacing_values)
    }

    return results, metrics


if __name__ == '__main__':
    N = 100  # 种群大小
    max_evaluations = 10000  # 总函数评估次数
    num_runs = 30  # 多次运行次数

    # 运行算法并计算各指标统计数据
    results, metrics = run_algorithm_multiple_times(N, D, lower_bound, upper_bound,
                                                    max_evaluations, fitness_function,
                                                    num_runs=num_runs)

    print("指标统计结果：")
    print("Hypervolume (HV) 均值：", metrics["HV_mean"], "标准差：", metrics["HV_std"])
    print("IGD 均值：", metrics["IGD_mean"], "标准差：", metrics["IGD_std"])
    print("Spacing 均值：", metrics["Spacing_mean"], "标准差：", metrics["Spacing_std"])

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图


def extract_pareto(F: np.ndarray) -> np.ndarray:
    """
    从目标值矩阵 F 中提取非支配解（帕累托前沿）
    这里采用简单的双重循环方法，适用于样本量不大的情况
    """
    n_points = F.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if is_efficient[i]:
            for j in range(n_points):
                if i != j and is_efficient[j]:
                    # 若 j 解支配 i 解，则标记 i 为 False
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_efficient[i] = False
                        break
    return F[is_efficient]


def plot_results_2d(results: list, pf_true: np.ndarray = None):
    """
    绘制2D目标空间图：
    - results: 多次运行得到的结果列表，每个结果中包含属性 F（目标值矩阵）
    - pf_true: 如果提供真实 Pareto 前沿，则以红色曲线绘制
    """
    # 合并所有运行的帕累托点
    all_F = np.vstack([res.F for res in results])
    # 提取整体帕累托前沿
    overall_pf = extract_pareto(all_F)

    plt.figure(figsize=(8, 6))
    # 绘制所有目标点（灰色，作为参考）
    plt.scatter(all_F[:, 0], all_F[:, 1], color="gray", alpha=0.5, label="All obtained points")
    # 绘制整体帕累托前沿（蓝色）
    plt.scatter(overall_pf[:, 0], overall_pf[:, 1], color="blue", label="Overall Pareto front", s=50)
    if pf_true is not None:
        # 如果提供真实前沿，以红色曲线显示
        plt.plot(pf_true[:, 0], pf_true[:, 1], color="red", lw=2, label="True Pareto front")
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("ZDT3")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_results_3d(results: list, pf_true: np.ndarray = None):
    """
    绘制3D目标空间图：
    - results: 多次运行得到的结果列表，每个结果中包含属性 F（目标值矩阵）
    - pf_true: 如果提供真实 Pareto 前沿，则以红色曲线绘制
    """
    all_F = np.vstack([res.F for res in results])
    overall_pf = extract_pareto(all_F)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制所有目标点
    ax.scatter(all_F[:, 0], all_F[:, 1], all_F[:, 2], color="gray", alpha=0.5, label="All obtained points")
    # 绘制整体帕累托前沿
    ax.scatter(overall_pf[:, 0], overall_pf[:, 1], overall_pf[:, 2], color="blue", s=50, label="Overall Pareto front")
    if pf_true is not None:
        ax.plot(pf_true[:, 0], pf_true[:, 1], pf_true[:, 2], color="red", lw=2, label="True Pareto front")
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_zlabel("Objective 3")
    ax.set_title("3D Objective Space and Pareto Front")
    ax.legend()
    plt.show()
# 假设 results 是 run_algorithm_multiple_times 返回的 List[Result]，
# 并且 problem.pareto_front(use_cache=False) 得到真实前沿 pf_true（如果有）
pf_true = problem.pareto_front(use_cache=False)
plot_results_2d(results, pf_true)
