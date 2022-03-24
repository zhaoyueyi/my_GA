import numpy as np
import pygad

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


# def fitness_func(solution, solution_idx):
#     output = np.sum(solution * function_inputs)
#     fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
#     return fitness


def fitness_sphere(solution, solution_idx):
    output = np.square(solution[0]) + np.square(solution[1])
    fitness = 1.0 / output
    return fitness


def fitness_dejong(solution, solution_idx):
    s = solution
    result = 100 * ((s[1] - s[0] ** 2) ** 2) + (s[0] - 1) ** 2
    result = 1 / (result + 1)
    return result


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


# fitness_function = fitness_func

ga_instance = pygad.GA(num_generations=1000,  # 进化代数
                       num_parents_mating=40,  # 作为父母的数量
                       fitness_func=fitness_dejong,  # 评估函数
                       initial_population=None,  # 初始种群
                       sol_per_pop=100,  # 种群中个体数量
                       num_genes=2,  # 个体中基因数量
                       gene_type=float,  # 基因数值类型
                       init_range_low=-2.048,  # 初始化基因数值下限
                       init_range_high=2.048,  # 初始化基因数值上限
                       parent_selection_type="rws",  # 轮盘赌选择
                       keep_parents=-1,  # 保持所有父母进入下一种群（0为不进入）
                       # K_tournament=3,  # 锦标赛选择的父母数
                       crossover_type="single_point",  # 单点交叉
                       crossover_probability=0.7,  # 交叉概率
                       mutation_type="random",  # 随机变异
                       mutation_by_replacement=False,  # 随机变异中，true为用随机值替换基因，false为向基因添加随机值
                       # 三选一
                       mutation_probability=0.1,  # 突变基因的概率
                       # mutation_percent_genes="default",  # 突变基因的百分比
                       # mutation_num_genes=None,  # 突变基因的数量
                       # random_mutation_min_val=-5.12,  # 随机变异中随机值下限
                       # random_mutation_max_val=5.12,  # 随机变异中随机值上限
                       gene_space=None,  # 指定基因的可能值，可以是单一值[xx, xx]，也可以是多个之一[[xx, xx], [xx, xx]]
                       # on_start=on_start,  # GA开始前执行一次
                       # on_fitness=on_fitness,  # 计算fitness后回调
                       # on_parents=on_parents,  # 选择双亲后回调
                       # on_crossover=on_crossover,  # 交叉操作后回调
                       # on_mutation=on_mutation,  # 变异操作后回调
                       # on_generation=on_generation,  # 生成后代后回调
                       # on_stop=on_stop,  # GA完成前执行一次
                       delay_after_gen=0.0,  # 进化之间的延迟
                       save_best_solutions=True,  # 保存最佳个体至best_solutions
                       save_solutions=False,  # 保存所有个体至solutions
                       suppress_warnings=False,  # 是否打印警告消息
                       allow_duplicate_genes=True,  # 允许含有重复基因
                       stop_criteria=None  # 阻止进化条件，reach_xx为fitness达到xx， saturate_xx为进化次数达到xx
                       )

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(solution)
print(solution_fitness)
ga_instance.plot_fitness()
