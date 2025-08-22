import numpy as np
import random
from math import exp, tanh
from typing import Callable, List, Tuple
import pandas as pd
from multiobj_test_functions import test_functions


def non_dominated_sort(population: np.ndarray, fitness: np.ndarray) -> List[List[int]]:
    """
    对种群进行非支配排序，返回各前沿的个体索引列表。
    """
    N = len(population)
    fronts = [[]]
    domination_count = np.zeros(N, dtype=int)
    dominated_solutions = [[] for _ in range(N)]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            # 判断 p 是否支配 q：所有目标 p<=q 且至少有一个目标 p<q
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

    # 去除最后一个空前沿
    if not fronts[-1]:
        fronts.pop()
    return fronts


def calculate_crowding_distance(front: List[int], fitness: np.ndarray) -> np.ndarray:
    """
    计算给定前沿中每个个体的拥挤度距离。
    """
    distance = np.zeros(len(front))
    num_objectives = fitness.shape[1]
    if len(front) == 0:
        return distance
    # 对每个目标分别计算
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


def update_params(SR: float, H: int, MF: np.ndarray, MCr: np.ndarray, current_memory_index: int) -> Tuple[
    float, float, float]:
    """
    根据成功率 SR 及历史记忆更新参数，返回当前 F, pb, Cr。
    """
    mF = 0.4 + 0.25 * tanh(5 * SR)
    # 从历史记忆中随机选择一个存储单元
    h = np.random.randint(0, H)
    F = np.random.normal(MF[h], 0.02)
    F = np.clip(F, 0.1, 0.9)  # 限制 F 在 [0.1, 0.9]
    Cr = np.random.normal(MCr[h], 0.05)
    Cr = np.clip(Cr, 0, 1)
    pb = 0.7 * exp(-7 * SR)
    return F, pb, Cr


def update_memory(successful_F: List[float], successful_Cr: List[float], MF: np.ndarray, MCr: np.ndarray,
                  current_memory_index: int, H: int) -> int:
    """
    根据当前代成功的参数更新历史记忆，采用简单平均更新。
    """
    if successful_F:
        MF[current_memory_index] = np.mean(successful_F)
    if successful_Cr:
        MCr[current_memory_index] = np.mean(successful_Cr)
    current_memory_index = (current_memory_index + 1) % H
    return current_memory_index


def generate_population(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """
    在给定上下界内随机生成种群。
    """
    return np.random.rand(N, D) * (upper_bound - lower_bound) + lower_bound


def select_pbest(x_top: np.ndarray, fitness_top: np.ndarray, pb: float, N: int) -> np.ndarray:
    """
    根据 pb 百分比从 x_top 中选择 pbest 个体。这里采用各目标值之和作为代理指标进行排序。
    """
    # 计算代理指标（目标值之和，越小越好）
    aggregate = np.sum(fitness_top, axis=1)
    sorted_indices = np.argsort(aggregate)
    pb_size = max(2, int(pb * len(x_top)))
    candidate_indices = sorted_indices[:pb_size]
    # 对每个个体在种群中随机选择一个pbest候选（允许重复选择）
    return np.random.choice(candidate_indices, size=N, replace=True)


def mutation_and_crossover(population: np.ndarray, F: float, Cr: float, pbest_population: np.ndarray,
                           lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """
    对种群执行变异与二项式交叉操作生成新的候选解。
    对于每个个体 i：
      v_i = x_r1 + F * (pbest - x_i) + F * (x_r2 - x_r3)
    注意：pbest 是从历史最优种群中选出的。
    """
    N, D = population.shape
    u_new = np.copy(population)

    for i in range(N):
        indices = list(range(N))
        # 随机选取三个不同个体用于变异
        r1, r2, r3 = random.sample(indices, 3)
        # pbest 从 pbest_population 选择（对应 i）
        pbest = pbest_population[i]
        v = population[r1] + F * (pbest - population[i]) + F * (population[r2] - population[r3])
        # 二项式交叉
        j_rand = random.randint(0, D - 1)
        for j in range(D):
            if random.random() <= Cr or j == j_rand:
                u_new[i, j] = v[j]
        # 边界处理
        u_new[i] = np.clip(u_new[i], lower_bound, upper_bound)
    return u_new


def environmental_selection(combined_population: np.ndarray, combined_fitness: np.ndarray, N: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    使用 NSGA-II 风格的环境选择：
      1. 对合并种群做非支配排序。
      2. 按前沿依次填充种群；若某一前沿个体数超过剩余名额，
         则根据拥挤度距离选择。
    """
    fronts = non_dominated_sort(combined_population, combined_fitness)
    new_population_indices = []
    for front in fronts:
        if len(new_population_indices) + len(front) <= N:
            new_population_indices.extend(front)
        else:
            # 计算当前前沿的拥挤度距离
            distances = calculate_crowding_distance(front, combined_fitness)
            # 将 front 按拥挤度距离降序排序
            sorted_front = sorted(front, key=lambda idx: distances[front.index(idx)], reverse=True)
            remaining = N - len(new_population_indices)
            new_population_indices.extend(sorted_front[:remaining])
            break
    new_population = combined_population[new_population_indices]
    new_fitness = combined_fitness[new_population_indices]
    return new_population, new_fitness


def multi_objective_LSRTDE(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                           max_evaluations: int, fitness_function: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    多目标 L-SRTDE 主函数：
      - 维护双种群：当前种群 population 和历史最优种群 x_top
      - 使用非支配排序与拥挤度保持多样性
      - 使用 SHA 机制更新历史记忆（MF, MCr）
      - SR 定义为当前种群第一前沿个体比例
    """
    NFE = 0
    Nmin = 4  # 最小种群规模（此处暂未动态调整）
    H = 5
    SR = 0.5  # 初始成功率
    current_memory_index = 0
    # 初始化历史记忆数组（初始值可以设为 0.5，1.0 分别）
    MF = np.ones(H) * 0.5
    MCr = np.ones(H)

    population = generate_population(N, D, lower_bound, upper_bound)
    fitness = np.array([fitness_function(ind) for ind in population])
    NFE += N

    # 初始化历史最优种群 x_top 为初始种群（后续将不断更新）
    x_top = np.copy(population)
    fitness_top = np.copy(fitness)

    # 主循环：直到函数评估次数达到上限
    while NFE < max_evaluations:
        # 对当前种群进行非支配排序
        fronts = non_dominated_sort(population, fitness)
        # 定义 SR 为第一前沿个体比例
        SR = len(fronts[0]) / N if fronts[0] else 0.0

        # 从 x_top 中按 pb 比例选择 pbest 候选
        # 先计算 x_top 的代理适应度（各目标之和）
        aggregate_top = np.sum(fitness_top, axis=1)
        sorted_indices_top = np.argsort(aggregate_top)
        pb_dummy = 0.7 * exp(-7 * SR)  # 当前 pb
        pb_size = max(2, int(pb_dummy * len(x_top)))
        candidate_indices = sorted_indices_top[:pb_size]
        # 对种群中每个个体，随机选一个 pbest 来自候选集合
        pbest_population = x_top[np.random.choice(candidate_indices, size=N, replace=True)]

        # 更新参数：F, pb, Cr（pb 此处已用于选择 pbest）
        F, pb, Cr = update_params(SR, H, MF, MCr, current_memory_index)

        # 产生子代：变异与交叉
        u_new = mutation_and_crossover(population, F, Cr, pbest_population, lower_bound, upper_bound)
        u_fitness = np.array([fitness_function(ind) for ind in u_new])
        NFE += N

        # 记录成功的个体（定义为：u_new 是否进入非支配第一前沿）
        combined = np.vstack((population, u_new))
        combined_fit = np.vstack((fitness, u_fitness))
        new_fronts = non_dominated_sort(combined, combined_fit)
        successful_F = []
        successful_Cr = []
        # 对 u_new 部分（后 N 个解）判断是否在第一前沿
        for idx in new_fronts[0]:
            if idx >= len(population):  # 属于 u_new
                successful_F.append(F)
                successful_Cr.append(Cr)

        # 更新历史记忆
        if successful_F or successful_Cr:
            current_memory_index = update_memory(successful_F, successful_Cr, MF, MCr, current_memory_index, H)

        # 环境选择：合并当前种群与子代，然后选择 N 个个体
        combined_population = np.vstack((population, u_new))
        combined_fitness = np.vstack((fitness, u_fitness))
        population, fitness = environmental_selection(combined_population, combined_fitness, N)

        # 更新历史最优种群 x_top：合并 x_top 和 u_new，然后选择非支配前沿
        combined_top = np.vstack((x_top, u_new))
        combined_top_fit = np.vstack((fitness_top, u_fitness))
        top_fronts = non_dominated_sort(combined_top, combined_top_fit)
        # 简单选择第一前沿作为 x_top，若数量不足，再补充后续前沿
        selected_indices = list(top_fronts[0])
        i = 1
        while len(selected_indices) < N and i < len(top_fronts):
            selected_indices.extend(top_fronts[i])
            i += 1
        if len(selected_indices) > N:
            selected_indices = selected_indices[:N]
        x_top = combined_top[selected_indices]
        fitness_top = combined_top_fit[selected_indices]

    # 返回最终的 Pareto 前沿（即 x_top 及其适应度）
    return x_top, fitness_top


"-------------------------------------------------------------------------------------------------------------------"


# 选择一个多目标测试函数，例如使用 'uf5'
fitness_function = test_functions['test1']


def save_to_csv(pareto_solutions, pareto_fitness, test_num, file_name="pareto_results.csv"):
    # Convert the solutions and fitness to pandas DataFrames
    solutions_df = pd.DataFrame(pareto_solutions)
    fitness_df = pd.DataFrame(pareto_fitness)

    # Add test number as a column to distinguish between different runs
    solutions_df['Test Number'] = test_num
    fitness_df['Test Number'] = test_num

    # Save both solutions and fitness to CSV (you can use pd.concat to merge them side by side)
    if test_num == 1:
        # For the first run, save the headers
        solutions_df.to_csv(file_name, mode='w', header=True, index=False)
        fitness_df.to_csv(file_name, mode='a', header=True, index=False)
    else:
        # For subsequent runs, append without writing headers again
        solutions_df.to_csv(file_name, mode='a', header=False, index=False)
        fitness_df.to_csv(file_name, mode='a', header=False, index=False)
    print(f"Results of test {test_num} saved to {file_name}")

"-----------------------------------------修改运行次数-------------------------------------------------------------"

def run_algorithm_multiple_times(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                                 max_evaluations: int, fitness_function: Callable, num_runs: int = 3):
    for run in range(num_runs):
        print(f"Running test {run + 1}/{num_runs}...")
        try:
            pareto_solutions, pareto_fitness = multi_objective_LSRTDE(N, D, lower_bound, upper_bound,
                                                                      max_evaluations, fitness_function)

            # Save the results of each run into CSV file
            save_to_csv(pareto_solutions, pareto_fitness, run + 1)

        except Exception as e:
            print(f"Error in test {run + 1}: {str(e)}")
            # You can optionally handle errors here, or log them to another file if needed

    print("All tests completed. Results are logged in 'pareto_results.csv'.")


if __name__ == '__main__':
    # 设置问题参数
    N = 100  # 种群大小
    D = 30  # 维度
    lower_bound = np.zeros(D)
    upper_bound = np.ones(D) * 10
    max_evaluations = 10000  # 总函数评估次数

    # 运行多目标 L-SRTDE 算法 int 次
    run_algorithm_multiple_times(N, D, lower_bound, upper_bound, max_evaluations, fitness_function)

