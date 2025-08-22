import numpy as np
import random
from math import exp, tanh
from typing import Callable, List, Tuple
import pandas as pd
from pymoo.problems import get_problem

# -------------------- 修改部分 --------------------
# 使用 pymoo 的 get_problem 获取内置测试问题（例如 zdt1）
problem = get_problem("zdt4")
def fitness_function(x: np.ndarray) -> np.ndarray:
    # 确保 x 为二维数组后调用问题的 evaluate 方法，返回第一行结果
    F = problem.evaluate(np.atleast_2d(x))
    return F[0]

# 根据 pymoo 问题设置决策变量维度和上下界（注意：对于 zdt1，通常 xl=0, xu=1）
D = problem.n_var
lower_bound = problem.xl if isinstance(problem.xl, np.ndarray) else np.full(D, problem.xl)
upper_bound = problem.xu if isinstance(problem.xu, np.ndarray) else np.full(D, problem.xu)
# --------------------------------------------------

def non_dominated_sort(population: np.ndarray, fitness: np.ndarray) -> List[List[int]]:
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
            distance[sorted_indices[i]] += (front_fitness[sorted_indices[i + 1]] - front_fitness[sorted_indices[i - 1]]) / (f_max - f_min)
    return distance

def update_params(SR: float, H: int, MF: np.ndarray, MCr: np.ndarray, current_memory_index: int) -> Tuple[float, float, float]:
    mF = 0.4 + 0.25 * tanh(5 * SR)
    h = np.random.randint(0, H)
    F = np.random.normal(MF[h], 0.02)
    F = np.clip(F, 0.1, 0.9)
    Cr = np.random.normal(MCr[h], 0.05)
    Cr = np.clip(Cr, 0, 1)
    pb = 0.7 * exp(-7 * SR)
    return F, pb, Cr

def update_memory(successful_F: List[float], successful_Cr: List[float], MF: np.ndarray, MCr: np.ndarray,
                  current_memory_index: int, H: int) -> int:
    if successful_F:
        MF[current_memory_index] = np.mean(successful_F)
    if successful_Cr:
        MCr[current_memory_index] = np.mean(successful_Cr)
    current_memory_index = (current_memory_index + 1) % H
    return current_memory_index

def generate_population(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    return np.random.rand(N, D) * (upper_bound - lower_bound) + lower_bound

def select_pbest(x_top: np.ndarray, fitness_top: np.ndarray, pb: float, N: int) -> np.ndarray:
    aggregate = np.sum(fitness_top, axis=1)
    sorted_indices = np.argsort(aggregate)
    pb_size = max(2, int(pb * len(x_top)))
    candidate_indices = sorted_indices[:pb_size]
    return np.random.choice(candidate_indices, size=N, replace=True)

def mutation_and_crossover(population: np.ndarray, F: float, Cr: float, pbest_population: np.ndarray,
                           lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
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

def environmental_selection(combined_population: np.ndarray, combined_fitness: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
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
                           max_evaluations: int, fitness_function: Callable) -> Tuple[np.ndarray, np.ndarray]:
    NFE = 0
    Nmin = 4
    H = 5
    SR = 0.5
    current_memory_index = 0
    MF = np.ones(H) * 0.5
    MCr = np.ones(H)
    population = generate_population(N, D, lower_bound, upper_bound)
    fitness = np.array([fitness_function(ind) for ind in population])
    NFE += N
    x_top = np.copy(population)
    fitness_top = np.copy(fitness)
    while NFE < max_evaluations:
        fronts = non_dominated_sort(population, fitness)
        SR = len(fronts[0]) / N if fronts[0] else 0.0
        aggregate_top = np.sum(fitness_top, axis=1)
        sorted_indices_top = np.argsort(aggregate_top)
        pb_dummy = 0.7 * exp(-7 * SR)
        pb_size = max(2, int(pb_dummy * len(x_top)))
        candidate_indices = sorted_indices_top[:pb_size]
        pbest_population = x_top[np.random.choice(candidate_indices, size=N, replace=True)]
        F, pb, Cr = update_params(SR, H, MF, MCr, current_memory_index)
        u_new = mutation_and_crossover(population, F, Cr, pbest_population, lower_bound, upper_bound)
        u_fitness = np.array([fitness_function(ind) for ind in u_new])
        NFE += N
        combined = np.vstack((population, u_new))
        combined_fit = np.vstack((fitness, u_fitness))
        new_fronts = non_dominated_sort(combined, combined_fit)
        successful_F = []
        successful_Cr = []
        for idx in new_fronts[0]:
            if idx >= len(population):
                successful_F.append(F)
                successful_Cr.append(Cr)
        if successful_F or successful_Cr:
            current_memory_index = update_memory(successful_F, successful_Cr, MF, MCr, current_memory_index, H)
        combined_population = np.vstack((population, u_new))
        combined_fitness = np.vstack((fitness, u_fitness))
        population, fitness = environmental_selection(combined_population, combined_fitness, N)
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
    return x_top, fitness_top

def save_to_csv(pareto_solutions, pareto_fitness, test_num, file_name="pareto_results.csv"):
    solutions_df = pd.DataFrame(pareto_solutions)
    fitness_df = pd.DataFrame(pareto_fitness)
    solutions_df['Test Number'] = test_num
    fitness_df['Test Number'] = test_num
    if test_num == 1:
        solutions_df.to_csv(file_name, mode='w', header=True, index=False)
        fitness_df.to_csv(file_name, mode='a', header=True, index=False)
    else:
        solutions_df.to_csv(file_name, mode='a', header=False, index=False)
        fitness_df.to_csv(file_name, mode='a', header=False, index=False)
    print(f"Results of test {test_num} saved to {file_name}")

def run_algorithm_multiple_times(N: int, D: int, lower_bound: np.ndarray, upper_bound: np.ndarray,
                                 max_evaluations: int, fitness_function: Callable, num_runs: int = 1):
    for run in range(num_runs):
        print(f"Running test {run + 1}/{num_runs}...")
        try:
            pareto_solutions, pareto_fitness = multi_objective_LSRTDE(N, D, lower_bound, upper_bound,
                                                                      max_evaluations, fitness_function)
            save_to_csv(pareto_solutions, pareto_fitness, run + 1)
        except Exception as e:
            print(f"Error in test {run + 1}: {str(e)}")
    print("All tests completed. Results are logged in 'pareto_results.csv'.")

if __name__ == '__main__':
    N = 100  # 种群大小
    max_evaluations = 10000  # 总函数评估次数
    run_algorithm_multiple_times(N, D, lower_bound, upper_bound, max_evaluations, fitness_function)
