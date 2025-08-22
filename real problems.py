import numpy as np
import random
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.core.result import Result
from pymoo.indicators.hv import Hypervolume


# IGD 部分已去除

# ==================== 定义 5 个真实世界问题 ====================
# 1. RE2-4-1: Four Bar Truss Design (2目标, 4决策变量)
def re24_1_objective(x):
    # 模拟目标函数：
    # f1: 假设为结构重量，f2: 假设为变形量
    f1 = np.sum(x)
    f2 = np.prod(x)  # 仅作为示例
    return np.array([f1, f2])


def re24_1_bounds():
    lower = np.array([1.0, 1.0, 1.0, 1.0])
    upper = np.array([10.0, 10.0, 10.0, 10.0])
    return lower, upper


# 2. RE2-3-2: Reinforced Concrete Beam Design (2目标, 3决策变量)
def re23_2_objective(x):
    # 参考文献中给出：
    # f1 = 29.4*x1 + 0.6*x2*x3
    # f2 = g1_violation + g2_violation，其中 g1, g2 为约束函数
    f1 = 29.4 * x[0] + 0.6 * x[1] * x[2]
    g1 = x[0] * x[2] - 7.735 * (x[0] ** 2) * x[1] - 180
    g2 = 4 - (x[2] / x[1]) if x[1] != 0 else -np.inf
    violation = max(-g1, 0) + max(-g2, 0)
    f2 = violation
    return np.array([f1, f2])


def re23_2_bounds():
    # x1 为预定义离散值 [0.2,15]（此处用连续近似），x2 ∈ [0,20], x3 ∈ [0,40]
    lower = np.array([0.2, 0.0, 0.0])
    upper = np.array([15.0, 20.0, 40.0])
    return lower, upper


# 3. RE2-4-3: Pressure Vessel Design (2目标, 4决策变量)
def re24_3_objective(x):
    # 标准公式（简化版）：
    # f1 = 0.6224*x0*x2*x3 + 1.7781*x1*x2**2 + 3.1661*x0**2*x3 + 19.84*x0**2*x2
    # f2: 此处模拟为约束违反总和（示例中令 f2 = 0 表示无违反）
    f1 = 0.6224 * x[0] * x[2] * x[3] + 1.7781 * x[1] * x[2] ** 2 + 3.1661 * x[0] ** 2 * x[3] + 19.84 * x[0] ** 2 * x[2]
    f2 = 0.0
    return np.array([f1, f2])


def re24_3_bounds():
    lower = np.array([1.0, 1.0, 10.0, 10.0])
    upper = np.array([99.0, 99.0, 200.0, 200.0])
    return lower, upper


# 4. RE2-2-4: Hatch Cover Design (2目标, 2决策变量)
def re22_4_objective(x):
    # 模拟：f1 为成本，f2 为约束违反（此处均假定 f2 = 0）
    f1 = 50 * x[0] + 30 * x[1]
    f2 = 0.0
    return np.array([f1, f2])


def re22_4_bounds():
    lower = np.array([1.0, 1.0])
    upper = np.array([100.0, 100.0])
    return lower, upper


# 5. RE2-3-5: Coil Compression Spring Design (2目标, 3决策变量)
def re23_5_objective(x):
    # 模拟弹簧设计：f1 为重量（示例公式），f2 为约束违反
    f1 = (x[2] + 2) * x[0] ** 2 * x[1]
    g1 = x[0] - 0.1
    g2 = 100 - x[1]
    violation = max(0, -g1) + max(0, -g2)
    f2 = violation
    return np.array([f1, f2])


def re23_5_bounds():
    lower = np.array([0.1, 10.0, 1.0])
    upper = np.array([2.0, 100.0, 5.0])
    return lower, upper


# ==================== 算法及实验设置 ====================
POPULATION_SIZE = 100
MAX_EVALUATIONS = 10000
NUM_RUNS = 50

# JADE 参数
JADE_MEMORY_SIZE = 5
INITIAL_SR = 0.5
INITIAL_F = 0.5
INITIAL_CR = 0.5

FIGURE_SIZE = (8, 6)


# ==================== 算法模块（与 Main.py 中基本一致） ====================
def fitness_function(x, obj_func):
    return obj_func(x)


def non_dominated_sort(population, fitness):
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


def calculate_crowding_distance(front, fitness):
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


def update_params(MF, MCr, H):
    idx = random.randrange(H)
    F = MF[idx] + 0.1 * np.random.standard_cauchy()
    while F <= 0:
        F = MF[idx] + 0.1 * np.random.standard_cauchy()
    F = min(F, 1.0)
    Cr = np.random.normal(MCr[idx], 0.1)
    Cr = 0.0 if Cr < 0.0 else (1.0 if Cr > 1.0 else Cr)
    return F, Cr


def update_memory(successful_F, successful_Cr, MF, MCr, current_memory_index, H):
    if successful_F:
        MF[current_memory_index] = np.mean(successful_F)
    if successful_Cr:
        MCr[current_memory_index] = np.mean(successful_Cr)
    return (current_memory_index + 1) % H


def generate_population(N, D, lower_bound, upper_bound):
    return np.random.rand(N, D) * (upper_bound - lower_bound) + lower_bound


def select_pbest(x_top, fitness_top, pb, N):
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


def mutation_and_crossover(population, F_list, Cr_list, pbest_population, lower_bound, upper_bound):
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


def environmental_selection(combined_population, combined_fitness, N):
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


def multi_objective_LSRTDE(N, D, lower_bound, upper_bound, max_evaluations, obj_func):
    NFE = 0
    SR = INITIAL_SR
    H = JADE_MEMORY_SIZE
    MF = np.full(H, INITIAL_F)
    MCr = np.full(H, INITIAL_CR)
    current_memory_index = 0

    population = generate_population(N, D, lower_bound, upper_bound)
    fitness = np.array([fitness_function(ind, obj_func) for ind in population])
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
        u_fitness = np.array([fitness_function(ind, obj_func) for ind in u_new])
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


def compute_spacing(F):
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


def extract_pareto(F):
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


def run_algorithm_multiple_times(N, D, lower_bound, upper_bound, max_evaluations, obj_func, num_runs=NUM_RUNS):
    results = []
    hv_values, spacing_values = [], []

    # 设置参考点：取首次运行目标向量最大值的 1.1 倍
    ref_point = None

    for run in range(num_runs):
        print(f"运行 {run + 1}/{num_runs} ...")
        res = multi_objective_LSRTDE(N, D, lower_bound, upper_bound, max_evaluations, obj_func)
        results.append(res)
        if ref_point is None:
            ref_point = np.max(res.F, axis=0) * 1.1
        hv_indicator = Hypervolume(ref_point=ref_point)
        hv_values.append(hv_indicator.do(res.F))
        spacing_values.append(compute_spacing(res.F))
    metrics = {
        "HV_mean": np.mean(hv_values),
        "HV_std": np.std(hv_values),
        "Spacing_mean": np.mean(spacing_values),
        "Spacing_std": np.std(spacing_values)
    }
    return results, metrics


def plot_results_2d(results, pf_true=None, title="Pareto Front"):
    all_F = np.vstack([res.F for res in results])
    overall_pf = extract_pareto(all_F)
    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(overall_pf[:, 0], overall_pf[:, 1], color="blue", label="Overall Pareto front", s=50)
    if pf_true is not None:
        plt.plot(pf_true[:, 0], pf_true[:, 1], color="red", lw=2, label="True Pareto front")
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# ==================== 主程序 ====================
if __name__ == '__main__':
    # 定义问题字典，每个问题包含目标函数和边界信息
    problems = {
        "RE2-4-1 Four Bar Truss Design": {
            "obj_func": re24_1_objective,
            "bounds": re24_1_bounds()
        },
        "RE2-3-2 Reinforced Concrete Beam Design": {
            "obj_func": re23_2_objective,
            "bounds": re23_2_bounds()
        },
        "RE2-4-3 Pressure Vessel Design": {
            "obj_func": re24_3_objective,
            "bounds": re24_3_bounds()
        },
        "RE2-2-4 Hatch Cover Design": {
            "obj_func": re22_4_objective,
            "bounds": re22_4_bounds()
        },
        "RE2-3-5 Coil Compression Spring Design": {
            "obj_func": re23_5_objective,
            "bounds": re23_5_bounds()
        }
    }

    all_pareto_fronts = {}  # 存储每个问题的合并 Pareto 前沿
    all_metrics = {}

    for problem_name, info in problems.items():
        print(f"\n====== {problem_name} ======")
        obj_func = info["obj_func"]
        lower_bound, upper_bound = info["bounds"]
        D = len(lower_bound)
        results, metrics = run_algorithm_multiple_times(POPULATION_SIZE, D, lower_bound, upper_bound, MAX_EVALUATIONS,
                                                        obj_func)
        all_metrics[problem_name] = metrics
        print("指标统计结果：")
        print(f"Hypervolume (HV) 均值：{metrics['HV_mean']}，标准差：{metrics['HV_std']}")
        print(f"Spacing 均值：{metrics['Spacing_mean']}，标准差：{metrics['Spacing_std']}")
        # 合并所有运行得到的 Pareto 前沿解集
        all_F = np.vstack([res.F for res in results])
        overall_pf = extract_pareto(all_F)
        all_pareto_fronts[problem_name] = overall_pf
        # 绘图展示
        plot_results_2d(results, title=problem_name)

    # 输出所有问题的 Pareto 前沿解集
    print("\n\n===== 所有问题的 Pareto 前沿解集 =====")
    for problem_name, pf in all_pareto_fronts.items():
        print(f"\n{problem_name} 的 Pareto 前沿解集：")
        print(pf)
