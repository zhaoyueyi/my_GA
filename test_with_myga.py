# @Time : 2022/3/24 15:55 

# @Author : 赵曰艺

# @File : test_with_myga.py 

# @Software: PyCharm

# coding:utf-8
import myGA


def fitness_sphere(solution):
    result = 1 / (solution[0] ** 2 + solution[1] ** 2)
    return result


ga_instance = myGA.MyGA(num_gene=2,
                        num_pop=100,
                        range_gene=(-5.12, 5.12),
                        fitness_func=fitness_sphere,
                        num_parents=40,
                        prob_crossover=0.7,
                        prob_mutate=0.1,
                        precision=0.0001,
                        num_generation=100
                        )

ga_instance.run()
