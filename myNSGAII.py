# @Time : 2022/4/23 13:19 

# @Author : Yüehi

# @File : myNSGAII.py 

# @Software: PyCharm

# coding:utf-8
import numpy as np


class MultiObjProblem:
    def __init__(self,
                 num_obj: int,
                 dim: int,
                 bound: (float, float)):
        self.num_obj = num_obj
        self.bounds = bound
        self.dim = dim
        self.func_list = []

    def f1(self, x):
        pass

    def cal_fitness(self, solution):
        pass


class TestProblem(MultiObjProblem):
    obj = 2
    bounds = (-55, 55)
    dim = 30

    def __init__(self):
        MultiObjProblem.__init__(self, self.obj, self.dim, self.bounds)
        self.func_list = [self.f1, self.f2]

    def f1(self, x):
        return -x ** 2

    def f2(self, x):
        return -(x - 2) ** 2

    def cal_fitness(self, solution):
        return [self.f1(solution), self.f2(solution)]


class MyNSGAII:
    def __init__(self,
                 max_gen: int,
                 pop_size: int,
                 eta_crossover: float,
                 eta_mutate: float):
        self.eta_mutate = eta_mutate
        self.eta_crossover = eta_crossover
        self.max_gen = max_gen
        self.population = None
        self.pop_size = pop_size
        self.pop_rank = None
        self.pop_crowd = None
        self.pop_fitness = None
        self.pop_parents = None
        self.pop_fronts = None
        self.problem = None
        self.pro_bounds = None
        self.pro_dim = None
        self.pro_obj = None

    def load_problem(self, problem: MultiObjProblem):
        self.problem = problem
        self.pro_obj = problem.num_obj
        self.pro_dim = problem.dim
        self.pro_bounds = problem.bounds

    def _init_pop(self, size, dim, bounds):
        self.population = np.random.uniform(bounds[0], bounds[1], (size, dim))

    def _compute_fitness(self, population):
        fitness_list = []
        for p in population:
            fitness_list.append(self.problem.cal_fitness(p))
        self.pop_fitness = np.asarray(fitness_list)

    def _is_domination(self, x_index, y_index):
        x = self.pop_fitness[x_index]
        y = self.pop_fitness[y_index]
        return (x <= y).all() and (x < y).any()

    def _compute_rank(self, population):
        fronts = []
        front_1 = []
        list_domination_solutions = []
        list_dominated_count = []
        list_rank = []
        for p_index in range(len(population)):
            dom_solution = []
            dom_count = 0
            rank = np.Inf
            for q_index in range(len(population)):
                if self._is_domination(p_index, q_index):
                    dom_solution.append(q_index)
                elif self._is_domination(q_index, p_index):
                    dom_count += 1
            if dom_count == 0:
                rank = 0
                front_1.append(p_index)
            list_dominated_count.append(dom_count)
            list_domination_solutions.append(dom_solution)
            list_rank.append(rank)
        self.pop_rank = np.asarray(list_rank)
        front_index = 0
        fronts.append(front_1)
        while fronts[front_index]:
            front_next = []
            for p_index in fronts[front_index]:
                for q_index in list_domination_solutions[p_index]:
                    # list_dominated_count[q_index] -= 1
                    if (list_dominated_count[q_index] - 1) == 0:
                        self.pop_rank[q_index] = front_index + 1
                        front_next.append(q_index)
            front_index += 1
            fronts.append(front_next)
        self.pop_fronts = np.asarray(fronts)

    def _compute_crowd(self, fronts):
        self.pop_crowd = [-1] * len(self.population)
        for front in fronts:
            col_fitness = self.pop_fitness[front]
            table = np.column_stack((front, col_fitness))
            index_fitness_table = table[np.argsort(table[:, 1])]
            for i in range(len(front)):
                if i == 0 or i == len(index_fitness_table) - 1:
                    self.pop_crowd[index_fitness_table[i][0]] = np.Inf
                else:
                    crowd = 0
                    for j in range(self.pro_obj):
                        crowd += abs(index_fitness_table[i + 1][j + 1] - index_fitness_table[i - 1][j + 1])
                    self.pop_crowd[index_fitness_table[i][0]] = crowd

    def _select_parent(self):
        parents = []
        for _ in range(self.pop_size):
            l = np.random.randint(0, high=self.pop_size)
            r = np.random.randint(0, high=self.pop_size)
            if self.pop_rank[l] < self.pop_rank[r]:
                parents.append(self.population[l])
            elif self.pop_rank[l] > self.pop_rank[r]:
                parents.append(self.population[r])
            elif self.pop_crowd[l] < self.pop_crowd[r]:
                parents.append(self.population[r])
            elif self.pop_crowd[l] > self.pop_crowd[r]:
                parents.append(self.population[l])
            else:
                parents.append(self.population[l])
        self.pop_parents = np.asarray(parents)

    def _crossover_mutate(self):
        children = []
        eta_cro = self.eta_crossover
        eta_mut = self.eta_mutate
        index_r = int(self.pop_size / 2)
        rand_cro = np.random.uniform(low=0, high=1, size=index_r)
        rand_mut = np.random.uniform(low=0, high=1, size=self.pop_size)

        for index, i in enumerate(rand_cro):
            l = self.pop_parents[index]
            r = self.pop_parents[index + index_r]
            if i <= 0.5:
                beta = (i * 2) ** (1 / (1 + eta_cro))
            else:
                beta = (1 / (2 - i * 2)) ** (1 / (1 + eta_cro))
            child_l = 0.5 * ((1 + beta) * l + (1 - beta) * r)
            child_r = 0.5 * ((1 - beta) * l + (1 + beta) * r)
            if rand_mut[index] < 0.5:
                delta = (2 * rand_mut[index]) ** (1/(eta_mut+1))-1
            else:
                delta = (1-(2*(1-rand_mut[index]))) ** (1/(eta_mut+1))
            child_l = child_l + delta
            if rand_mut[index+index_r] < 0.5:
                delta = (2 * rand_mut[index]) ** (1/(eta_mut+1))-1
            else:
                delta = (1-(2*(1-rand_mut[index]))) ** (1/(eta_mut+1))
            child_r = child_r + delta
            children.append(child_l)
            children.append(child_r)
        children = np.asarray(children)
        self.population = np.vstack((self.population, children))

    def _select_elitism(self):
        population = []
        size = 0
        for front in self.pop_fronts:
            size += len(front)
            if size < self.pop_size:
                population = np.vstack((population, self.population[front]))
            else:
                tmp = np.column_stack((front, self.pop_crowd[front]))
                tmp = self.population[tmp[np.argsort(-tmp[:, 1])][:, 0]]
                population = np.vstack((population, tmp[:(len(front) - size + self.pop_size)]))
        self.population = np.asarray(population)

    def run(self):
        # init
        self._init_pop(self.pop_size * 2, self.pro_dim, self.pro_bounds)
        # rank
        self._compute_fitness(self.population)
        self._compute_rank(self.population)
        # crowding-distance
        self._compute_crowd(self.pop_fronts)
        for gen in range(self.max_gen):
            # parent-select: 二元锦标赛
            self._select_parent()
            # crossover-mutate
            self._crossover_mutate()
            # rank & distance
            self._compute_fitness(self.population)
            self._compute_rank(self.population)
            # crowding-distance
            self._compute_crowd(self.pop_fronts)
            # elitism select
            self._select_elitism()
            # terminate
