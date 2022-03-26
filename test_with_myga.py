# @Time : 2022/3/24 15:55 

# @Author : 赵曰艺

# @File : test_with_myga.py 

# @Software: PyCharm

# coding:utf-8
import numpy as np
from myGA import *


def fitness_sphere(solution):  # (-5.12, 5.12)
    result = 1 / (solution[0] ** 2 + solution[1] ** 2 + 1)
    return result


def dejong(solution):  # (-2.048, 2.048)
    s = solution
    result = 100 * ((s[1] - s[0] ** 2) ** 2) + (s[0] - 1) ** 2
    result = 1 / (result + 1)
    return result


def goldstein_price(solution):  # (-2.0, 2.0)
    s = solution
    result = (1 + (s[0] + s[1] + 1) ** 2 + (19 - 14 * s[0] + 3 * s[0] ** 2 - 14 * s[1] + 6 * s[0] + s[1])) \
             * ((30 + 2 * s[0] - 3 * s[1] ** 2) ** 2
                * (18 - 32 * s[0] + 12 * s[0] ** 2 + 48 * s[1] - 36 * s[0] * s[1] + 27 * s[1]))
    result = 1 / (result + 1)
    return result


def himmelbaut(solution):  # (-6.0, 6.0)
    s = solution
    result = (s[0] ** 2 + s[1] - 11) ** 2 + (s[0] + s[1] ** 2 - 7) ** 2
    result = 1 / (result + 1)
    return result


def six_hump_cameback(solution):  # (-5.0, 5.0)
    s = solution
    result = 4 * s[0] ** 2 + 2.1 * s[0] ** 4 + s[0] ** 6 / 3 + s[0] * s[1] - 4 * s[1] ** 2 + 4 * s[1] ** 4
    result = 1 / (result + 1)
    return result


def bohachevsky(solution):  # (-1, 1)
    s = solution
    result = s[0] ** 2 + 2 * s[1] ** 2 - 0.3 * np.cos(3 * np.pi * s[0]) - 0.4 * np.cos(4 * np.pi * s[1] + 0.7)
    result = 1 / (result + 1)
    return result


def HD_sphere(solution):
    result = 0.0
    for x in solution:
        result += x ** 2
    result = 1 / (result + 1)
    return result


def HD_step(solution):
    result = 0.0
    for x in solution:
        result += (abs(x + 0.5)) ** 2
    result = 1 / (result + 1)
    return result


def HD_schwefel1(solution):
    result = 0.0
    for i in range(len(solution)):
        tmp_result = 0.0
        for j in range(i):
            tmp_result += solution[j] ** 2
        result += tmp_result ** 2
    result = 1 / (result + 1)
    return result


def HD_schwefel2(solution):
    result = 0.0
    for x in solution:
        result += abs(x) + x
    result = 1 / (result + 1)
    return result


def HD_rastrigin(solution):
    result = 0.0
    A = 10
    result += len(solution) * A
    for x in solution:
        result += x ** 2 - A * np.cos(2 * np.pi * x)
    result = 1 / (result + 1)
    return result


ga_instance = MyGA(num_gene=2,
                   num_pop=100,
                   range_gene=(-5.12, 5.12),
                   fitness_func=fitness_sphere,
                   num_parents=40,
                   prob_crossover=0.7,
                   prob_mutate=0.1,
                   precision=0.0001,
                   num_generation=1000
                   )

# ga_instance.reload_fitness_func(range_gene=(-2.0, 2.0),
#                                 fitness_func=goldstein_price,
#                                 num_gene=2,
#                                 code_type=CodeType.Binary
#                                 )
#
# ga_instance.reload_fitness_func(range_gene=(-2.0, 2.0),
#                                 fitness_func=goldstein_price
#                                 )
# ga_instance.reload_fitness_func(range_gene=(-6.0, 6.0),
#                                 fitness_func=himmelbaut
#                                 )
# ga_instance.reload_fitness_func(range_gene=(-5.0, 5.0),
#                                 fitness_func=six_hump_cameback
#                                 )
# ga_instance.reload_fitness_func(range_gene=(-1.0, 1.0),
#                                 fitness_func=bohachevsky,
#                                 num_gene=2
#                                 )
ga_instance.reload_fitness_func(range_gene=(-100.0, 100.0),
                                fitness_func=HD_sphere,
                                num_gene=100,
                                code_type=CodeType.Real_Num)
# ga_instance.reload_fitness_func(range_gene=(-100.0, 100.0),
#                                 fitness_func=HD_step,
#                                 num_gene=100,
#                                 code_type=CodeType.Real_Num)
ga_instance.run()

