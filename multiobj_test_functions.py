import numpy as np
import math

# 1. Test function 1: Sphere vs. shifted Sphere
def test1(x: np.ndarray) -> np.ndarray:
    # f1 = sum(x^2), f2 = sum((x-2)^2)
    return np.array([np.sum(x ** 2), np.sum((x - 2) ** 2)])

# 2. Test function 2: First component vs. quadratic sum of rest
def test2(x: np.ndarray) -> np.ndarray:
    # f1 = x[0], f2 = 1 - sqrt(x[0]) + sum(x[1:]^2)
    f1 = x[0]
    f2 = 1 - math.sqrt(x[0]) + np.sum(x[1:] ** 2)
    return np.array([f1, f2])

# 3. ZDT1
def zdt1(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - math.sqrt(f1 / g))
    return np.array([f1, f2])

# 4. ZDT2
def zdt2(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - (f1 / g) ** 2)
    return np.array([f1, f2])

# 5. ZDT3
def zdt3(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - math.sqrt(f1 / g) - (f1 / g) * math.sin(10 * math.pi * f1))
    return np.array([f1, f2])

# 6. ZDT4
def zdt4(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f1 = x[0]
    g = 1 + 10 * (n - 1) + np.sum(x[1:] ** 2 - 10 * np.cos(4 * math.pi * x[1:]))
    f2 = g * (1 - math.sqrt(f1 / g))
    return np.array([f1, f2])

# 7. ZDT6
def zdt6(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f1 = 1 - math.exp(-4 * x[0]) * (math.sin(6 * math.pi * x[0])) ** 6
    g = 1 + 9 * (np.sum(x[1:]) / (n - 1)) ** 0.25
    f2 = g * (1 - (f1 / g) ** 2)
    return np.array([f1, f2])

# 8. DTLZ1 for 2 objectives
def dtlz1(x: np.ndarray) -> np.ndarray:
    n = len(x)
    g = 9 * np.sum(x[1:]) / (n - 1)
    f1 = 0.5 * x[0] * (1 + g)
    f2 = 0.5 * (1 - x[0]) * (1 + g)
    return np.array([f1, f2])

# 9. DTLZ2 for 2 objectives
def dtlz2(x: np.ndarray) -> np.ndarray:
    # For 2 objectives: f1 = cos(x[0]*pi/2), f2 = sin(x[0]*pi/2)
    f1 = math.cos(x[0] * math.pi / 2)
    f2 = math.sin(x[0] * math.pi / 2)
    return np.array([f1, f2])

# 10. DTLZ3 for 2 objectives (带有多峰特性)
def dtlz3(x: np.ndarray) -> np.ndarray:
    n = len(x)
    g = 100 * (len(x[1:]) + np.sum(((x[1:] - 0.5) ** 2 - np.cos(20 * math.pi * (x[1:] - 0.5)))))
    f1 = math.cos(x[0] * math.pi / 2) * (1 + g)
    f2 = math.sin(x[0] * math.pi / 2) * (1 + g)
    return np.array([f1, f2])

# 11. DTLZ4 for 2 objectives (引入参数alpha)
def dtlz4(x: np.ndarray) -> np.ndarray:
    n = len(x)
    alpha = 100
    x_mod = np.copy(x)
    x_mod[0] = x_mod[0] ** alpha
    g = 1 + 9 * np.sum(x_mod[1:]) / (n - 1)
    f1 = x_mod[0]
    f2 = g * (1 - math.sqrt(f1 / g))
    return np.array([f1, f2])

# 12. Test function 8: Sphere vs. shifted Sphere (重复test1，可作为另一测试)
def test8(x: np.ndarray) -> np.ndarray:
    f1 = np.sum(x ** 2)
    f2 = np.sum((x - 1) ** 2)
    return np.array([f1, f2])

# 13. Test function 9: Rosenbrock vs. shifted Rosenbrock (简化版)
def test9(x: np.ndarray) -> np.ndarray:
    f1 = np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
    f2 = np.sum(100 * ((x[1:] - 1) - (x[:-1] - 1) ** 2) ** 2 + ((x[:-1] - 1) - 1) ** 2)
    return np.array([f1, f2])

# 14. Test function 10: Ackley vs. shifted Ackley
def test10(x: np.ndarray) -> np.ndarray:
    a = 20
    b = 0.2
    c = 2 * math.pi
    n = len(x)
    f1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / n)) - np.exp(np.sum(np.cos(c * x)) / n) + a + math.e
    f2 = -a * np.exp(-b * np.sqrt(np.sum((x - 1) ** 2) / n)) - np.exp(np.sum(np.cos(c * (x - 1))) / n) + a + math.e
    return np.array([f1, f2])

# 15. Test function 11: Rastrigin vs. shifted Rastrigin
def test11(x: np.ndarray) -> np.ndarray:
    A = 10
    n = len(x)
    f1 = A * n + np.sum(x ** 2 - A * np.cos(2 * math.pi * x))
    f2 = A * n + np.sum((x - 1) ** 2 - A * np.cos(2 * math.pi * (x - 1)))
    return np.array([f1, f2])

# 16. Test function 12: Schaffer function N.1
def schaffer1(x: np.ndarray) -> np.ndarray:
    # 取 x 的第一个元素计算
    x_val = x[0]
    f1 = 0.5 + ((math.sin(x_val ** 2) - 0.5) ** 2) / ((1 + 0.001 * x_val ** 2) ** 2)
    f2 = 1 - math.sqrt(x_val)
    return np.array([f1, f2])

# 17. Test function 13: Schaffer function N.2
def schaffer2(x: np.ndarray) -> np.ndarray:
    x_val = x[0]
    f1 = x_val ** 2
    f2 = (x_val - 2) ** 2
    return np.array([f1, f2])

# 18. Test function 14: Viennet function (简化2目标版本)
def viennet(x: np.ndarray) -> np.ndarray:
    # 典型取 x 的前两个分量
    x1, x2 = x[0], x[1]
    f1 = 0.5 * (x1 ** 2 + x2 ** 2) + np.sin(x1 ** 2 + x2 ** 2)
    f2 = 0.5 * (x1 ** 2 + x2 ** 2) - np.sin(x1 ** 2 + x2 ** 2)
    return np.array([f1, f2])

# 19. Test function 15: Kursawe function
def kursawe(x: np.ndarray) -> np.ndarray:
    n = len(x)
    sum1 = 0
    sum2 = 0
    for i in range(n - 1):
        sum1 += -10 * np.exp(-0.2 * np.sqrt(x[i] ** 2 + x[i + 1] ** 2))
    for i in range(n):
        sum2 += np.abs(x[i]) ** 0.8 + 5 * np.sin(x[i] ** 3)
    return np.array([sum1, sum2])

# 20. Test function 16: UF1 (简化版本)
def uf1(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - np.sqrt(x[0] / g))
    return np.array([f1, f2])

# 21. Test function 17: UF2 (简化版本)
def uf2(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - (x[0] / g) ** 2)
    return np.array([f1, f2])

# 22. Test function 18: UF3 (简化版本)
def uf3(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - np.sqrt(x[0] / g) - (x[0] / g) * np.sin(10 * math.pi * x[0]))
    return np.array([f1, f2])

# 23. Test function 19: UF4 (简化版本)
def uf4(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - np.sqrt(x[0] / g) - (x[0] / g) ** 2)
    return np.array([f1, f2])

# 24. Test function 20: UF5 (简化版本)
def uf5(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - (x[0] / g) ** 2)
    return np.array([f1, f2])

# 25. Test function 21: Custom function: sphere and shifted sphere
def custom21(x: np.ndarray) -> np.ndarray:
    f1 = np.sum(x ** 2)
    f2 = np.sum((x - 1) ** 2)
    return np.array([f1, f2])

# 26. Test function 22: Custom: maximum and minimum
def custom22(x: np.ndarray) -> np.ndarray:
    f1 = np.max(x)
    f2 = np.min(x)
    return np.array([f1, f2])

# 27. Test function 23: Custom: product and sum
def custom23(x: np.ndarray) -> np.ndarray:
    f1 = np.prod(x)
    f2 = np.sum(x)
    return np.array([f1, f2])

# 28. Test function 24: Custom: sin and cos of sum
def custom24(x: np.ndarray) -> np.ndarray:
    s = np.sum(x)
    f1 = np.sin(s)
    f2 = np.cos(s)
    return np.array([f1, f2])

# 29. Test function 25: Custom: mean and standard deviation
def custom25(x: np.ndarray) -> np.ndarray:
    f1 = np.mean(x)
    f2 = np.std(x)
    return np.array([f1, f2])

# 30. Test function 26: Custom: quadratic and linear combination
def custom26(x: np.ndarray) -> np.ndarray:
    f1 = np.sum(x ** 2)
    f2 = np.sum(x)
    return np.array([f1, f2])

# 31. Test function 27: Custom: difference and absolute deviation
def custom27(x: np.ndarray) -> np.ndarray:
    f1 = np.sum(x)
    f2 = np.abs(np.sum(x) - 1)
    return np.array([f1, f2])

# 32. Test function 28: Custom: exponential and logarithm (with overflow protection)
def custom28(x: np.ndarray) -> np.ndarray:
    s = np.sum(x)
    f1 = np.exp(s) if s < 700 else 1e300
    f2 = np.log(np.abs(s) + 1)
    return np.array([f1, f2])

# 33. Test function 29: Custom: maximum of squares and minimum of squares
def custom29(x: np.ndarray) -> np.ndarray:
    f1 = np.max(x ** 2)
    f2 = np.min(x ** 2)
    return np.array([f1, f2])

# 34. Test function 30: Custom: sum of sin^2 and cos^2
def custom30(x: np.ndarray) -> np.ndarray:
    f1 = np.sum(np.sin(x) ** 2)
    f2 = np.sum(np.cos(x) ** 2)
    return np.array([f1, f2])

# 将所有测试函数放入字典，便于调用
test_functions = {
    'test1': test1,
    'test2': test2,
    'zdt1': zdt1,
    'zdt2': zdt2,
    'zdt3': zdt3,
    'zdt4': zdt4,
    'zdt6': zdt6,
    'dtlz1': dtlz1,
    'dtlz2': dtlz2,
    'dtlz3': dtlz3,
    'dtlz4': dtlz4,
    'test8': test8,
    'test9': test9,
    'test10': test10,
    'test11': test11,
    'schaffer1': schaffer1,
    'schaffer2': schaffer2,
    'viennet': viennet,
    'kursawe': kursawe,
    'uf1': uf1,
    'uf2': uf2,
    'uf3': uf3,
    'uf4': uf4,
    'uf5': uf5,
    'custom21': custom21,
    'custom22': custom22,
    'custom23': custom23,
    'custom24': custom24,
    'custom25': custom25,
    'custom26': custom26,
    'custom27': custom27,
    'custom28': custom28,
    'custom29': custom29,
    'custom30': custom30
}

if __name__ == '__main__':
    # 示例：打印所有测试函数名称及其在一个随机样本 x 上的输出
    x_sample = np.random.rand(30)  # 生成30维随机样本
    for name, func in test_functions.items():
        result = func(x_sample)
        print(f"{name}: {result}")
