# @Time : 2022/3/20 21:58
# @Author : 赵曰艺
# @File : myGA.py
# @Software: PyCharm
# coding:utf-8
from enum import Enum
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

class CodeType(Enum):
    Binary = 0
    Real_Num = 1

class MyGA:
    class Solution:
        def __init__(self, index=None, fitness=None, solution=None):
            self.index = index
            self.fitness = fitness
            self.solution = solution

    def __init__(self,
                 num_gene,
                 num_pop,
                 range_gene: Tuple[float, float],
                 fitness_func,
                 num_parents,
                 prob_crossover,
                 prob_mutate,
                 precision,
                 num_generation=10):
        """
        initial first population -> np.array[[x,x], [x,x], [x,x], ...]
        :param num_gene: 2 [xx, xx]
        :param num_pop:
        :param range_gene:
        :param fitness_func:
        :param num_parents:
        :param prob_crossover:
        :param prob_mutate:
        :param precision:
        :param num_generation:
        """
        self.code_type = None
        self.num_gene = None
        self.parents = None
        self.children_worst_solution = None
        self.best_solution = None
        self.range_gene = None
        self.num_encode = None
        self.fitness_func = None
        self.fitness = None
        self.population = None
        self.num_pop = num_pop
        self.num_parents = num_parents
        self.num_generation = num_generation
        self.prob_mutate = prob_mutate
        self.prob_crossover = prob_crossover
        self.precision = precision
        self.reload_fitness_func(range_gene=range_gene, fitness_func=fitness_func, num_gene=num_gene)

    def reload_fitness_func(self, range_gene: Tuple[float, float], fitness_func, num_gene, code_type=CodeType.Real_Num):
        if not callable(fitness_func) or fitness_func.__code__.co_argcount < 1:
            raise ValueError('Error: fitness function')
        # reload
        self.num_gene = num_gene
        self.code_type = code_type
        self.fitness_func = fitness_func
        self.range_gene = range_gene
        self.population = np.random.uniform(low=range_gene[0], high=range_gene[1], size=(self.num_pop, num_gene))
        if code_type == CodeType.Binary:
            self.num_encode = int(np.ceil(np.log2(((range_gene[1] - range_gene[0]) / self.precision))))
        # clear
        self.fitness = None
        self.best_solution = None
        self.children_worst_solution = self.Solution()
        self.parents = np.empty((self.num_parents, num_gene))

    def __calculate_fitness(self, population, save_worst=False):
        fitness = []
        for solution in population:
            fitness.append(self.fitness_func(solution))
        self.fitness = np.asarray(fitness)
        if save_worst:
            self.children_worst_solution.index = np.argmin(self.fitness)
            self.children_worst_solution.fitness = self.fitness[self.children_worst_solution.index]

    def __select_parents(self, population, num_parents, fitness):
        random_prob = np.random.uniform(low=0, high=fitness.sum(), size=num_parents)
        for idx, prob in enumerate(random_prob):
            i = -1
            while prob > 0:
                i += 1
                prob -= self.fitness[i]
            # TODO: index
            self.parents[idx] = population[i]

    def __encoding(self, solution):  # [xx, xx] -> 'xxxxxx'
        result = ''
        for i in solution:
            tmp = int(np.floor((i - self.range_gene[0]) / self.precision))
            tmp = np.binary_repr(tmp, width=self.num_encode)
            result += tmp
        return result

    def __decoding(self, code):  # 'xxxxxx' -> [xx, xx]
        result = []
        for i in range(int(len(code) / self.num_encode)):
            tmp = code[i * self.num_encode: i * self.num_encode + self.num_encode]
            tmp = int(tmp, base=2)
            tmp = self.range_gene[0] + (tmp * self.precision)
            result.append(tmp)
        return result

    def __crossover(self, parents, num_parents, prob_crossover, num_encode=None, num_gene=None, code_type=CodeType.Real_Num):
        if code_type == CodeType.Real_Num:
            random_pos_high = num_gene
        else:
            random_pos_high = num_encode * num_gene
        random_prob = np.random.uniform(low=0, high=1, size=int(num_parents / 2))
        random_pos = np.random.randint(low=0, high=random_pos_high, size=int(num_parents / 2))
        pair_parents = parents.reshape(-1, 2, num_gene).tolist()  # [[[xx, xx], [xx, xx]], [[xx, xx], [xx, xx]], ...]
        children = []
        if code_type == CodeType.Real_Num:  # real-num code
            for idx, pair in enumerate(pair_parents):  # [[xx, xx], [xx, xx]]
                if random_prob[idx] < prob_crossover:
                    father = pair[0]  # 'xxxxxxxxxx'
                    mother = pair[1]
                    child = father[: random_pos[idx]] + mother[random_pos[idx]:]
                    children.append(child)  # [xx, xx]
                    child = mother[: random_pos[idx]] + father[random_pos[idx]:]
                    children.append(child)
                else:
                    children.append(pair[0])
                    children.append(pair[1])
        else:  # binary code
            for idx, pair in enumerate(pair_parents):  # [[xx, xx], [xx, xx]]
                if random_prob[idx] < prob_crossover:
                    father = self.__encoding(pair[0])  # 'xxxxxxxxxx'
                    mother = self.__encoding(pair[1])
                    child = father[: random_pos[idx]] + mother[random_pos[idx]:]
                    children.append(self.__decoding(child))  # [xx, xx]
                    child = mother[: random_pos[idx]] + father[random_pos[idx]:]
                    children.append(self.__decoding(child))
                else:
                    children.append(pair[0])
                    children.append(pair[1])
        self.population = np.asarray(children)

    def __mutate(self, population, prob_mutate, num_encode=None, num_gene=None, range_gene=None, code_type=CodeType.Real_Num):
        if code_type == CodeType.Real_Num:
            random_value = np.random.uniform(low=range_gene[0],
                                             high=range_gene[1],
                                             size=len(population))
            random_pos_high = num_gene
        else:
            random_value = None
            random_pos_high = num_encode * num_gene
        random_pos = np.random.randint(low=0, high=random_pos_high, size=len(population))
        random_prob = np.random.uniform(low=0, high=1, size=len(population))
        if code_type == CodeType.Real_Num:
            for idx, solution in enumerate(population):  # [xx, xx]
                if random_prob[idx] < prob_mutate:
                    solution[random_pos[idx]] = random_value[idx]
                    self.population[idx] = solution
        else:
            for idx, solution in enumerate(population):  # [xx, xx]
                if random_prob[idx] < prob_mutate:
                    solution = self.__encoding(solution)
                    mutate_pos = '0' if solution[random_pos[idx]] == '1' else '1'
                    result = solution[:random_pos[idx]] + mutate_pos + solution[random_pos[idx]+1:]
                    self.population[idx] = np.asarray(self.__decoding(result))

    def run(self):
        """
        # calculate fitness -> [x, x, x, ...]
        # save the best solution -> [x, x]
        # select parents: roulette_wheel_selection -> [[index, x], [x, x], ...]
        #   num_parents
        # crossover: single_point en/decode -> [bin, x, x, ...]
        # mutate: single_point en/decode -> [x, x, x, ...]
        # cal fitness -> [x, x, x, ...]
        # maintain best: replace the worst in children -> back to row 2
        :return:
        """
        self.__calculate_fitness(self.population)
        best_index = np.argmax(self.fitness)
        self.best_solution = self.Solution(best_index,
                                           self.fitness[best_index],
                                           self.population[best_index])
        max_fitness = []
        min_fitness = []
        avg_fitness = []
        for _ in range(self.num_generation):
            self.__select_parents(self.population, self.num_parents, self.fitness)
            self.__crossover(self.parents, self.num_parents, self.prob_crossover, self.num_encode, self.num_gene, self.code_type)
            self.__mutate(self.population, self.prob_mutate, self.num_encode, self.num_gene, self.range_gene, self.code_type)
            self.__calculate_fitness(self.population, save_worst=True)
            # 精英保留策略
            if self.children_worst_solution.fitness < self.best_solution.fitness:
                self.population[self.children_worst_solution.index] = self.best_solution.solution
                self.fitness[self.children_worst_solution.index] = self.best_solution.fitness
            self.best_solution.index = np.argmax(self.fitness)
            self.best_solution.solution = self.population[self.best_solution.index]
            self.best_solution.fitness = self.fitness[self.best_solution.index]
            max_fitness.append(np.max(self.fitness))
            min_fitness.append(np.min(self.fitness))
            avg_fitness.append(np.average(self.fitness))
        self.display(max_fitness, min_fitness, avg_fitness)
        print(self.best_solution.fitness)
        print(self.best_solution.solution)

    def display(self, data1, data2, data3):
        """
        # max
        # min
        # avg
        # std
        :return:
        """
        plt.figure()
        plt.plot(data1, label='test')
        # plt.plot(data2)
        # plt.plot(data3)
        plt.title("Generation vs. Fitness")
        plt.show()
