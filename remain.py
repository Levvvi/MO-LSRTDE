import os
import numpy as np
import random
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
# 导入RE问题集中的问题
from reproblem import RE21, RE25, RE31, RE36, RE37

# 配置输出目录
text_dir = r"D:\PyProject\retext"
img_dir = r"D:\PyProject\repjpeg"
os.makedirs(text_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# -------------------------
# 基本函数：非支配排序、拥挤度计算等
def non_dominated_sort(population, fitness):
    """对种群进行非支配排序，返回各Pareto层的个体索引列表"""
    N = len(population)
    fronts = [[]]
    domination_count = np.zeros(N, dtype=int)
    dominated_solutions = [[] for _ in range(N)]
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if np.all(fitness[p] <= fitness[q]) and np.any(fitness[p] < fitness[q]):
                dominated_solutions[p].append(q)
            elif np.all(fitness[q] <= fitness[p]) and np.any(fitness[q] < fitness[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts

def calculate_crowding_distance(front_indices, fitness):
    """计算给定前沿中各个体的拥挤距离。返回的数组长度与front_indices相同，
       数组中的第i个数对应front_indices中第i位置的解的拥挤距离。"""
    if len(front_indices) == 0:
        return np.array([])
    distance = np.zeros(len(front_indices))
    num_objectives = fitness.shape[1]
    for m in range(num_objectives):
        values = fitness[front_indices, m]
        sorted_idx = np.argsort(values)  # 返回相对于values数组的索引
        distance[sorted_idx[0]] = float('inf')
        distance[sorted_idx[-1]] = float('inf')
        if values[sorted_idx[-1]] - values[sorted_idx[0]] == 0:
            continue
        for j in range(1, len(front_indices) - 1):
            distance[sorted_idx[j]] += (values[sorted_idx[j + 1]] - values[sorted_idx[j - 1]]) / (values[sorted_idx[-1]] - values[sorted_idx[0]])
    return distance

def select_pbest(archive_pop, archive_fit, p_ratio, N):
    """从存档中按照非支配排序选择p-best个体，返回选择的索引（相对于archive_pop的下标）"""
    fronts = non_dominated_sort(archive_pop, archive_fit)
    p_size = max(2, int(p_ratio * len(archive_pop)))
    candidates = []
    for front in fronts:
        if len(candidates) + len(front) < p_size:
            candidates.extend(front)
        else:
            remaining = p_size - len(candidates)
            if remaining > 0:
                selected = random.sample(front, remaining) if remaining < len(front) else front[:remaining]
                candidates.extend(selected)
            break
    if not candidates:
        candidates = fronts[0][:2] if fronts else [0]
    return np.array([random.choice(candidates) for _ in range(N)])

def mutation_and_crossover(population, F_list, Cr_list, pbest_population, lower_bound, upper_bound):
    """利用DE/current-to-pbest/1或DE/rand/1策略生成子代"""
    N, D = population.shape
    offspring = np.copy(population)
    for i in range(N):
        indices = list(range(N))
        indices.remove(i)
        r1, r2, r3 = random.sample(indices, 3)
        if random.random() < 0.5:
            mutant = population[i] + F_list[i]*(pbest_population[i] - population[i]) + F_list[i]*(population[r1] - population[r2])
        else:
            mutant = population[r1] + F_list[i]*(population[r2] - population[r3])
        j_rand = random.randrange(D)
        for j in range(D):
            if random.random() <= Cr_list[i] or j == j_rand:
                offspring[i, j] = mutant[j]
        offspring[i] = np.clip(offspring[i], lower_bound, upper_bound)
    return offspring

def extract_pareto(F):
    """从一组目标函数值中提取非支配解，即Pareto前沿"""
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
    d_mean = np.mean(distances)
    return np.sqrt(np.sum((distances - d_mean)**2) / (N - 1))

# ---------------------------
# L-SRTDE算法主体
def multi_objective_LSRTDE(problem_instance, pop_size=100, max_evals=10000):
    D = problem_instance.n_variables
    lower_bound = np.array(problem_instance.lbound, dtype=float)
    upper_bound = np.array(problem_instance.ubound, dtype=float)
    def fitness(x):
        return problem_instance.evaluate(x)
    # 初始种群生成
    population = np.random.rand(pop_size, D)*(upper_bound - lower_bound) + lower_bound
    fitness_vals = np.array([fitness(ind) for ind in population])
    eval_count = pop_size

    # 初始化存档（非支配历史解）
    archive_pop = population.copy()
    archive_fit = fitness_vals.copy()

    # JADE记忆参数
    H = 5
    MF = np.full(H, 0.5)
    MCr = np.full(H, 0.5)
    mem_index = 0
    SR = 0.5

    while eval_count < max_evals:
        p_best_frac = 0.7 * exp(-7*SR)
        pbest_idx = select_pbest(archive_pop, archive_fit, p_best_frac, pop_size)
        pbest_pop = archive_pop[pbest_idx]
        F_list = np.zeros(pop_size)
        Cr_list = np.zeros(pop_size)
        for i in range(pop_size):
            idx = random.randrange(H)
            F = MF[idx] + 0.1*np.random.standard_cauchy()
            while F <= 0:
                F = MF[idx] + 0.1*np.random.standard_cauchy()
            F = min(F, 1.0)
            Cr = random.gauss(MCr[idx], 0.1)
            Cr = 0.0 if Cr < 0.0 else (1.0 if Cr > 1.0 else Cr)
            F_list[i] = F
            Cr_list[i] = Cr

        offspring = mutation_and_crossover(population, F_list, Cr_list, pbest_pop, lower_bound, upper_bound)
        offspring_fit = np.array([fitness(ind) for ind in offspring])
        eval_count += pop_size

        # 合并父子代
        combined = np.vstack((population, offspring))
        combined_fit = np.vstack((fitness_vals, offspring_fit))
        fronts = non_dominated_sort(combined, combined_fit)
        if fronts and len(fronts[0]) > 0:
            num_success = sum(1 for idx in fronts[0] if idx >= pop_size)
            SR = num_success / pop_size
        else:
            SR = 0.0

        new_pop_indices = []
        for front in fronts:
            if len(new_pop_indices) + len(front) <= pop_size:
                new_pop_indices.extend(front)
            else:
                remaining = pop_size - len(new_pop_indices)
                if remaining > 0:
                    # 修正：利用zip将front与拥挤度对应后排序
                    crowd_dist = calculate_crowding_distance(front, combined_fit)
                    sorted_front = [ind for ind, cd in sorted(zip(front, crowd_dist), key=lambda x: x[1], reverse=True)]
                    new_pop_indices.extend(sorted_front[:remaining])
            if len(new_pop_indices) >= pop_size:
                break
        new_pop_indices = new_pop_indices[:pop_size]
        population = combined[new_pop_indices]
        fitness_vals = combined_fit[new_pop_indices]

        # 更新存档（archive）——合并历史存档与新产生的子代
        all_archive = np.vstack((archive_pop, offspring))
        all_archive_fit = np.vstack((archive_fit, offspring_fit))
        archive_fronts = non_dominated_sort(all_archive, all_archive_fit)
        archive_idx = []
        for front in archive_fronts:
            if len(archive_idx) + len(front) <= pop_size:
                archive_idx.extend(front)
            else:
                remaining = pop_size - len(archive_idx)
                if remaining > 0:
                    crowd_dist = calculate_crowding_distance(front, all_archive_fit)
                    sorted_front = [ind for ind, cd in sorted(zip(front, crowd_dist), key=lambda x: x[1], reverse=True)]
                    archive_idx.extend(sorted_front[:remaining])
            if len(archive_idx) >= pop_size:
                break
        archive_idx = archive_idx[:pop_size]
        archive_pop = all_archive[archive_idx]
        archive_fit = all_archive_fit[archive_idx]

        # 更新JADE记忆
        successful_F = []
        successful_Cr = []
        for idx in new_pop_indices:
            if idx >= pop_size:
                off_idx = idx - pop_size
                successful_F.append(F_list[off_idx])
                successful_Cr.append(Cr_list[off_idx])
        if successful_F:
            MF[mem_index] = np.mean(successful_F)
        if successful_Cr:
            MCr[mem_index] = np.mean(successful_Cr)
        mem_index = (mem_index + 1) % H

    pareto_front = extract_pareto(archive_fit)
    return pareto_front

# ---------------------------
# 运行每个问题50次，并保存结果与图像
problems = [RE21(), RE25(), RE31(), RE36(), RE37()]
for problem in problems:
    name = problem.problem_name
    print(f"Running L-SRTDE on {name}...")
    hv_values, igd_values, spacing_values = [], [], []
    all_run_pf = []  # 存放每次运行得到的Pareto前沿目标值
    for run in range(1, 51):
        pf = multi_objective_LSRTDE(problem, pop_size=100, max_evals=10000)
        all_run_pf.append(pf)
        print(f"  Run {run} completed, number of Pareto solutions: {pf.shape[0]}")
    # 合并所有运行得到的Pareto前沿
    combined_pf = np.vstack(all_run_pf)
    overall_pf = extract_pareto(combined_pf)
    # 参考点：选取每个目标中最差值的1.1倍
    ref_point = np.max(overall_pf, axis=0) * 1.1

    # 对每次运行计算各指标
    for pf in all_run_pf:
        hv = Hypervolume(ref_point=ref_point).do(pf)
        hv_values.append(hv)
        igd = IGD(overall_pf).do(pf)
        igd_values.append(igd)
        sp = compute_spacing(pf)
        spacing_values.append(sp)
    hv_values = np.array(hv_values)
    igd_values = np.array(igd_values)
    spacing_values = np.array(spacing_values)

    # 保存指标至文本文件
    metrics_file = os.path.join(text_dir, f"{name}_metrics.txt")
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

    # 绘图并保存：判断目标数量选择2D或3D图
    if problem.n_objectives == 2:
        plt.figure()
        plt.scatter(overall_pf[:,0], overall_pf[:,1], color="blue", s=30, label="Pareto front")
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.title(name)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(img_dir, f"{name}.jpg")
        plt.savefig(plot_path, dpi=300)
        plt.close()
    elif problem.n_objectives == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(overall_pf[:,0], overall_pf[:,1], overall_pf[:,2], color="blue", s=30, label="Pareto front")
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        ax.set_title(name)
        ax.legend()
        plot_path = os.path.join(img_dir, f"{name}.jpg")
        plt.savefig(plot_path, dpi=300)
        plt.close()
    print(f"Saved Pareto front plot for {name} to {plot_path}\n")
