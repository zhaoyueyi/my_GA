# @Time : 2022/3/24 15:55 

# @Author : 赵曰艺

# @File : test_with_myga.py 

# @Software: PyCharm

# coding:utf-8
import numpy as np
import myGA


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


ga_instance = myGA.MyGA(num_gene=2,
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
#                                 fitness_func=goldstein_price
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
ga_instance.reload_fitness_func(range_gene=(-1.0, 1.0),
                                fitness_func=bohachevsky
                                )

ga_instance.run()
