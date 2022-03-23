# @Time : 2022/3/20 21:58 

# @Author : 赵曰艺

# @File : myGA.py 

# @Software: PyCharm

# coding:utf-8
import numpy as np

precision = 0.001
s_range = (0, 1)
s_dim = 2
s_len = s_range[1] - s_range[0]
total = np.floor(s_len / precision)
width = 8


def encoding(solution, dim=1):
    base = 0
    result = []
    while base < dim:
        tmp = np.floor((solution[base] - s_range[0]) / precision)
        tmp = np.binary_repr(tmp, width=width)
        result.append(tmp)
        base += 1
    return result


def decoding(code, dim=1):
    base = 0
    result = []
    while dim > 0:
        tmp = code[base: base + width]
        tmp = int(tmp, base=2)
        tmp = s_range[0] + (tmp * precision)
        result.append(tmp)
        base += width
        dim -= 1
    return result


def on_start(ga_instance, dim=1, count=20):
    solutions = []
    base, base2 = 0, 0
    size = dim * count
    random_source = np.random.randint(low=0, high=1, size=size)
    while size > base:
        solution = []
        while dim > base2:
            solution.append(random_source[base + base2])
            base2 += 1
        solutions.append(solution)
        base += dim
    return solutions
    # print("on_start()")


def on_fitness(ga_instance, population_fitness):
    fitness_list = []
    print("on_fitness()")
    return fitness_list


def on_parents(ga_instance, selected_parents):
    ga_instance.roulette_wheel_selection()
    print("on_parents()")


def on_crossover(ga_instance, offspring_crossover):
    ga_instance.single_point_crossover()
    print("on_crossover()")


def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")


def on_generation(ga_instance):
    print("on_generation()")


def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")


def fitness_sphere(solution, solution_idx):
    pass


def fitness_dejong(solution, solution_idx):
    pass


def fitness_goldsteinprice(solution, solution_idx):
    pass


def fitness_himmelbaut(solution, solution_idx):
    pass


def fitness_sixhumpcameback(solution, solution_idx):
    pass


def fitness_bohachevsky(solution, solution_idx):
    pass


def fitness_shubert(solution, solution_idx):
    pass


def fitness_HD_sphere(solution, solution_idx):
    pass


def fitness_HD_step(solution, solution_idx):
    pass


def fitness_HD_schwefel1(solution, solution_idx):
    pass


def fitness_HD_schwefel2(solution, solution_idx):
    pass


def fitness_HD_rastrigin(solution, solution_idx):
    pass


def fitness_HD_griewank(solution, solution_idx):
    pass


def fitness_HD_ackley(solution, solution_idx):
    pass


def fitness_HD_rosenbrock(solution, solution_idx):
    pass


fitness_function = fitness_sphere


class myGA:
    def __init__(self):
        # initial first population -> np.array[[x,x], [x,x], [x,x], ...]
        #   num_gene:2 [xx, xx]
        #   num_pop
        #   range_init_value
        pass

    def run(self):
        # calculate fitness -> [x, x, x, ...]
        # save the best solution -> [x, x]
        # select parents: roulette_wheel_selection -> [[index, x], [x, x], ...]
        #   num_parents
        # crossover: single_point en/decode -> [bin, x, x, ...]
        # mutate: single_point en/decode -> [x, x, x, ...]
        # cal fitness -> [x, x, x, ...]
        # maintain best: replace the worst in children -> back to row 2
        pass

    def display(self):
        pass
