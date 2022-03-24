# @Time : 2022/3/20 21:58 

# @Author : 赵曰艺

# @File : myGA.py 

# @Software: PyCharm

# coding:utf-8
import numpy as np
from typing import Tuple


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
        # initial first population -> np.array[[x,x], [x,x], [x,x], ...]
        #   num_gene:2 [xx, xx]
        #   num_pop
        #   range_init_value

        if not callable(fitness_func) or fitness_func.__code__.co_argcount < 1:
            raise ValueError('Error: fitness function')

        self.population = np.random.uniform(low=range_gene[0], high=range_gene[1], size=(num_pop, num_gene))
        self.fitness = None
        self.fitness_func = fitness_func
        self.num_gene = num_gene
        self.num_parents = num_parents
        self.num_generation = num_generation
        self.prob_mutate = prob_mutate
        self.prob_crossover = prob_crossover
        self.precision = precision
        self.num_encode = int(np.ceil(np.log2(((range_gene[1] - range_gene[0]) / precision))))
        self.range_gene = range_gene
        self.best_solution = None
        self.children_worst_solution = self.Solution()
        self.parents = np.empty((num_parents, 2))

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
        roulette_wheel = np.empty(len(fitness))
        tmp = 0.0
        for idx, i in enumerate(fitness):
            tmp += i
            roulette_wheel[idx] = tmp
        for idx, prob in enumerate(random_prob):
            i = -1
            while prob > 0:
                i += 1
                prob -= roulette_wheel[i]
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
        for i in range(int(len(code)/self.num_encode)):
            tmp = code[i*self.num_encode: i*self.num_encode+self.num_encode]
            tmp = int(tmp, base=2)
            tmp = self.range_gene[0] + (tmp * self.precision)
            result.append(tmp)
        return result

    def __crossover(self, parents, num_parents, prob_crossover, num_encode):
        random_prob = np.random.uniform(low=0, high=1, size=int(num_parents / 2))
        random_pos = np.random.randint(low=0, high=num_encode*self.num_gene, size=int(num_parents / 2))
        pair_parents = parents.reshape(-1, 2, self.num_gene)  # [[[xx, xx], [xx, xx]], [[xx, xx], [xx, xx]], ...]
        children = []
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

    def __mutate(self, population, prob_mutate, num_encode):
        random_prob = np.random.uniform(low=0, high=1, size=len(population))
        random_pos = np.random.randint(low=0, high=num_encode*self.num_gene, size=len(population))
        for idx, solution in enumerate(population):  # [xx, xx]
            if random_prob[idx] < prob_mutate:
                solution = self.__encoding(solution)
                mutate_pos = '0' if solution[random_pos[idx]] == '1' else '1'
                result = solution[:random_pos[idx]] + mutate_pos + solution[random_pos[idx]+1:]
                self.population[idx] = np.asarray(self.__decoding(result))

    def run(self):
        self.__calculate_fitness(self.population)
        best_index = np.argmax(self.fitness)
        self.best_solution = self.Solution(best_index,
                                           self.fitness[best_index],
                                           self.population[best_index], )
        for _ in range(self.num_generation):
            self.__select_parents(self.population, self.num_parents, self.fitness)
            self.__crossover(self.parents, self.num_parents, self.prob_crossover, self.num_encode)
            self.__mutate(self.population, self.prob_mutate, self.num_encode)
            self.__calculate_fitness(self.population, save_worst=True)
            # 精英保留策略
            if self.children_worst_solution.fitness < self.best_solution.fitness:
                self.population[self.children_worst_solution.index] = self.best_solution.solution
                self.fitness[self.children_worst_solution.index] = self.best_solution.fitness
            self.best_solution.index = np.argmax(self.fitness)
            self.best_solution.solution = self.population[self.best_solution.index]
            self.best_solution.fitness = self.fitness[self.best_solution.index]
        print(self.best_solution.fitness)
        print(self.best_solution.solution)
        # calculate fitness -> [x, x, x, ...]
        # save the best solution -> [x, x]
        # select parents: roulette_wheel_selection -> [[index, x], [x, x], ...]
        #   num_parents
        # crossover: single_point en/decode -> [bin, x, x, ...]
        # mutate: single_point en/decode -> [x, x, x, ...]
        # cal fitness -> [x, x, x, ...]
        # maintain best: replace the worst in children -> back to row 2

    def display(self):
        # max
        # min
        # avg
        # std
        pass
