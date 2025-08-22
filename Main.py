import numpy as np
import random
from math import exp
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.problems.multi.sympart import SYMPART, SYMPARTRotated
from pymoo.problems import get_problem
from pymoo.core.result import Result
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.problems.many.wfg import WFG1
# ==================== 配置区 ====================
# 算法及问题参数
POPULATION_SIZE = 100  # 种群大小
MAX_EVALUATIONS = 10000  # 总函数评价次数
NUM_RUNS = 50  # 重复运行次数

JADE_MEMORY_SIZE = 5  # JADE记忆大小
INITIAL_SR = 0.5  # 初始成功率
INITIAL_F = 0.5  # 初始变异因子记忆值
INITIAL_CR = 0.5  # 初始交叉概率记忆值

PROBLEM_NAME = "WFG3"  # 内置测试问题名称

# 可视化参数
FIGURE_SIZE = (8, 6)

# ==================== 问题设置 ====================
problem = WFG1(n_var=10, n_obj=3)
"""get_problem(PROBLEM_NAME)"""
D = problem.n_var
lower_bound = problem.xl if isinstance(problem.xl, np.ndarray) else np.full(D, problem.xl)
upper_bound = problem.xu if isinstance(problem.xu, np.ndarray) else np.full(D, problem.xu)


# ==================== 算法模块 ====================
def fitness_function(x: np.ndarray) -> np.ndarray:
    """计算个体适应度，返回目标函数值（单个样本）"""
    return problem.evaluate(np.atleast_2d(x))[0]


def non_dominated_sort(population: np.ndarray, fitness: np.ndarray) -> List[List[int]]:
    """对种群进行非支配排序，返回各 Pareto 层的个体索引列表"""
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
    """计算给定前沿中各个体的拥挤距离"""
    distance = np.zeros(len(front))
    num_objectives = fitness.shape[1]
    for m in range(num_objectives):
        front_fitness = [fitness[i][m] for i in front]
        sorted_indices = np.argsort(front_fitness)
        f_min = front_fitness[sorted_indices[0]]
        f_max = front_fitness[sorted_indices[-1]]
        if f_max - f_min == 0:
            continue
        distance[sorted_indices[0]] = distance[sorted_indices[-1]] = float('inf')
        for j in range(1, len(front) - 1):
            distance[sorted_indices[j]] += (front_fitness[sorted_indices[j + 1]] - front_fitness[
                sorted_indices[j - 1]]) / (f_max - f_min)
    return distance


def update_params(MF: np.ndarray, MCr: np.ndarray, H: int) -> Tuple[float, float]:
    """从记忆中采样，生成变异因子 F 与交叉概率 Cr"""
    idx = random.randrange(H)
    F = MF[idx] + 0.1 * np.random.standard_cauchy()
    while F <= 0:
        F = MF[idx] + 0.1 * np.random.standard_cauchy()
    F = min(F, 1.0)
    Cr = np.random.normal(MCr[idx], 0.1)
    Cr = 0.0 if Cr < 0.0 else (1.0 if Cr > 1.0 else Cr)
    return F, Cr


def update_memory(successful_F: List[float], successful_Cr: List[float],
                  MF: np.ndarray, MCr: np.ndarray,
                  current_memory_index: int, H: int) -> int:
    """更新 JADE 记忆库"""
    if successful_F:
        MF[current_memory_index] = np.mean(successful_F)
    if successful_Cr:
        MCr[current_memory_index] = np.mean(successful_Cr)
    return (current_memory_index + 1) % H


def generate_population(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """生成随机种群"""
    return np.random.rand(N, D) * (upper_bound - lower_bound) + lower_bound


def select_pbest(x_top: np.ndarray, fitness_top: np.ndarray, pb: float, N: int) -> np.ndarray:
    """
    根据非支配排序，从历史最优集合中选择 p-best 个体，
    返回选择的索引（采样 size=N, 可重复）
    """
    fronts = non_dominated_sort(x_top, fitness_top)
    pb_size = max(2, int(pb * len(x_top)))
    candidates = []
    for front in fronts:
        if len(candidates) + len(front) < pb_size:
            candidates.extend(front)
        else:
            remaining = pb_size - len(candidates)
            if remaining > 0:
                selected = random.sample(front, remaining) if remaining < len(front) else front[:remaining]
                candidates.extend(selected)
            break
    if not candidates:
        candidates = fronts[0][:2] if fronts else [0]
    return np.random.choice(candidates, size=N, replace=True)


def mutation_and_crossover(population: np.ndarray, F_list: np.ndarray, Cr_list: np.ndarray,
                           pbest_population: np.ndarray,
                           lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """差分进化：根据 DE/current-to-pbest/1 或 DE/rand/1 策略生成子代"""
    N, D = population.shape
    u_new = np.copy(population)
    for i in range(N):
        indices = list(range(N))
        indices.remove(i)
        r1, r2, r3 = random.sample(indices, 3)
        if random.random() < 0.5:
            v = population[i] + F_list[i] * (pbest_population[i] - population[i]) + F_list[i] * (
                        population[r1] - population[r2])
        else:
            v = population[r1] + F_list[i] * (population[r2] - population[r3])
        j_rand = random.randint(0, D - 1)
        for j in range(D):
            if random.random() <= Cr_list[i] or j == j_rand:
                u_new[i, j] = v[j]
        u_new[i] = np.clip(u_new[i], lower_bound, upper_bound)
    return u_new


def environmental_selection(combined_population: np.ndarray, combined_fitness: np.ndarray, N: int) -> Tuple[
    np.ndarray, np.ndarray, List[int]]:
    """
    合并父子代后，采用非支配排序与拥挤距离选择下一代种群，
    返回新种群、其适应度及对应索引
    """
    fronts = non_dominated_sort(combined_population, combined_fitness)
    new_population_indices = []
    if fronts and len(fronts[0]) >= N:
        front0 = fronts[0]
        front1 = fronts[1] if len(fronts) > 1 else []
        dist0 = calculate_crowding_distance(front0, combined_fitness)
        sorted_front0 = [front0[idx] for idx in np.argsort(dist0)[::-1]]
        if front1:
            dist1 = calculate_crowding_distance(front1, combined_fitness)
            sorted_front1 = [front1[idx] for idx in np.argsort(dist1)[::-1]]
        else:
            sorted_front1 = []
        k = min(len(sorted_front1), max(1, int(0.1 * N))) if len(front0) > N else min(len(sorted_front1),
                                                                                      max(1, int(0.05 * N)))
        new_population_indices = sorted_front0[:N - k] + (sorted_front1[:k] if k > 0 else [])
    else:
        for front in fronts:
            if len(new_population_indices) + len(front) <= N:
                new_population_indices.extend(front)
            else:
                dist = calculate_crowding_distance(front, combined_fitness)
                sorted_front = [front[idx] for idx in np.argsort(dist)[::-1]]
                remaining = N - len(new_population_indices)
                new_population_indices.extend(sorted_front[:remaining])
                break
    new_population = combined_population[new_population_indices]
    new_fitness = combined_fitness[new_population_indices]
    return new_population, new_fitness, new_population_indices


def multi_objective_LSRTDE(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                           max_evaluations: int, fitness_function: Callable) -> Result:
    """
    L-SRTDE 多目标优化算法，集成成功率自适应与双种群交互策略，
    返回最后的 Pareto 解集
    """
    NFE = 0
    SR = INITIAL_SR
    H = JADE_MEMORY_SIZE
    MF = np.full(H, INITIAL_F)
    MCr = np.full(H, INITIAL_CR)
    current_memory_index = 0

    population = generate_population(N, D, lower_bound, upper_bound)
    fitness = np.array([fitness_function(ind) for ind in population])
    NFE += N

    x_top = np.copy(population)
    fitness_top = np.copy(fitness)

    while NFE < max_evaluations:
        pb = 0.7 * exp(-7 * SR)
        pbest_indices = select_pbest(x_top, fitness_top, pb, N)
        pbest_population = x_top[pbest_indices]

        F_list = np.zeros(N)
        Cr_list = np.zeros(N)
        for i in range(N):
            F_list[i], Cr_list[i] = update_params(MF, MCr, H)
        u_new = mutation_and_crossover(population, F_list, Cr_list, pbest_population, lower_bound, upper_bound)
        u_fitness = np.array([fitness_function(ind) for ind in u_new])
        NFE += N

        combined = np.vstack((population, u_new))
        combined_fit = np.vstack((fitness, u_fitness))
        new_fronts = non_dominated_sort(combined, combined_fit)
        if new_fronts[0]:
            num_success = sum(1 for idx in new_fronts[0] if idx >= len(population))
            SR = num_success / N
        else:
            SR = 0.0

        population, fitness, selected_indices = environmental_selection(combined, combined_fit, N)

        successful_F = []
        successful_Cr = []
        for idx in selected_indices:
            if idx >= N:
                child_idx = idx - N
                successful_F.append(F_list[child_idx])
                successful_Cr.append(Cr_list[child_idx])
        current_memory_index = update_memory(successful_F, successful_Cr, MF, MCr, current_memory_index, H)

        combined_top = np.vstack((x_top, u_new))
        combined_top_fit = np.vstack((fitness_top, u_fitness))
        top_fronts = non_dominated_sort(combined_top, combined_top_fit)
        selected = list(top_fronts[0])
        i = 1
        while len(selected) < N and i < len(top_fronts):
            selected.extend(top_fronts[i])
            i += 1
        x_top = combined_top[selected[:N]]
        fitness_top = combined_top_fit[selected[:N]]

    res = Result()
    res.X = x_top
    res.F = fitness_top
    return res


def compute_spacing(F: np.ndarray) -> float:
    """计算解集在目标空间的均匀性指标 Spacing"""
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
    d_mean = np.mean(distances)
    return np.sqrt(np.sum((distances - d_mean) ** 2) / (N - 1))


def run_algorithm_multiple_times(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                                 max_evaluations: int, fitness_function: Callable, num_runs: int = 5,
                                 ref_point=None) -> Tuple[List[Result], dict]:
    """
    多次运行算法，计算 Hypervolume、IGD 与 Spacing 指标的均值和标准差，
    返回每次运行结果及统计指标
    """
    if ref_point is None:
        pareto_front = problem.pareto_front(use_cache=False)
        if pareto_front is not None:
            ref_point = np.max(pareto_front, axis=0) * 1.1
        else:
            raise ValueError("未指定参考点且 problem.pareto_front() 为空，请手动设置")

    results = []
    hv_values, igd_values, spacing_values = [], [], []

    for run in range(num_runs):
        print(f"运行 {run + 1}/{num_runs}...")
        res = multi_objective_LSRTDE(N, D, lower_bound, upper_bound, max_evaluations, fitness_function)
        results.append(res)

        hv_indicator = Hypervolume(ref_point=ref_point)
        hv_values.append(hv_indicator.do(res.F))

        pf = problem.pareto_front(use_cache=False)
        if pf is not None:
            igd_indicator = IGD(pf)
            igd_values.append(igd_indicator.do(res.F))

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


# ==================== 可视化模块 ====================
def extract_pareto(F: np.ndarray) -> np.ndarray:
    """提取非支配解，即 Pareto 前沿"""
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


def plot_results_2d(results: List[Result], pf_true: np.ndarray = None):
    """
    绘制二维目标空间：
      - 蓝色散点：整体 Pareto 前沿
      - 红色曲线：真实 Pareto 前沿（若提供）
    """
    all_F = np.vstack([res.F for res in results])
    overall_pf = extract_pareto(all_F)

    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(overall_pf[:, 0], overall_pf[:, 1], color="blue", label="Overall Pareto front", s=50)
    if pf_true is not None:
        plt.plot(pf_true[:, 0], pf_true[:, 1], color="red", lw=2, label="True Pareto front")
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(PROBLEM_NAME)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_results_3d(results: List[Result], pf_true: np.ndarray = None):
    """
    绘制三维目标空间：
      - 蓝色散点：整体 Pareto 前沿
      - 红色曲线：真实 Pareto 前沿（若提供）
    """
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
    plt.show()


# ==================== 主程序 ====================
if __name__ == '__main__':
    results, metrics = run_algorithm_multiple_times(POPULATION_SIZE, D, lower_bound, upper_bound,
                                                    MAX_EVALUATIONS, fitness_function, num_runs=NUM_RUNS)
    print("指标统计结果：")
    print(f"Hypervolume (HV) 均值：{metrics['HV_mean']}，标准差：{metrics['HV_std']}")
    print(f"IGD 均值：{metrics['IGD_mean']}，标准差：{metrics['IGD_std']}")
    print(f"Spacing 均值：{metrics['Spacing_mean']}，标准差：{metrics['Spacing_std']}")

    pf_true = problem.pareto_front(use_cache=False)
    plot_results_3d(results, pf_true)
    # 若问题为3目标，可调用 plot_results_3d(results, pf_true)
