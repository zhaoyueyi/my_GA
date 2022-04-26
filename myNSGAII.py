# @Time : 2022/4/23 13:19 

# @Author : Yüehi

# @File : myNSGAII.py 

# @Software: PyCharm

# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as sp


class MultiObjProblem:
    def __init__(self,
                 name: str,
                 obj: int,
                 dim: int,
                 bound: (float, float)):
        self.name = name
        self.obj = obj
        self.bounds = bound
        self.dim = dim

    def cal_fitness(self, solution):
        pass

    def pareto_front(self, solution):
        pass


class ZDT(MultiObjProblem):
    def __init__(self, name, bounds):
        MultiObjProblem.__init__(self, name, 2, 30, bounds)

    def f2(self, g, h):
        return g * h

    def f1(self, x): pass

    def g(self, x): pass

    def h(self, f1, g): pass

    def cal_fitness(self, solution):
        f1 = self.f1(solution)
        g = self.g(solution)
        h = self.h(f1, g)
        f2 = self.f2(g, h)
        return [f1, f2]


class ZDT1(ZDT):
    def __init__(self):
        ZDT.__init__(self, 'zdt1', (0.0, 1.0))

    def f1(self, x):
        return x[0]

    def g(self, x):
        return 1 + (9 / 29) * np.sum(x[1:])

    def h(self, f1, g):
        return 1 - np.sqrt(f1 / g)


class ZDT2(ZDT):
    def __init__(self):
        ZDT.__init__(self, 'zdt2', (0.0, 1.0))

    def f1(self, x):
        return x[0]

    def g(self, x):
        return 1 + (9 / 29) * np.sum(x[1:])

    def h(self, f1, g):
        return 1 - (f1 / g) ** 2


class ZDT3(ZDT):
    def __init__(self):
        ZDT.__init__(self, 'zdt3', (0.0, 1.0))

    def f1(self, x):
        return x[0]

    def g(self, x):
        return 1 + (9 / 29) * np.sum(x[1:])

    def h(self, f1, g):
        return 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)


class DTLZ(MultiObjProblem):
    def __init__(self, name, dim, bounds):
        super().__init__(name, 3, dim, bounds)

    def X_M(self, x):
        M = self.obj
        X_M = x[M - 1:]
        return X_M

    def f(self, x):
        pass

    def cal_fitness(self, solution):
        return self.f(solution)

    def pareto_front(self, solution):
        M = self.obj
        x = []
        for i in range(self.dim):
            if i < M - 1:
                x.append(solution[i])
            else:
                x.append(0.5)
        return self.f(x)


class DTLZ1(DTLZ):
    def __init__(self):
        super().__init__('dtlz1', 3, (0.0, 1.0))

    def g(self, X_M):
        k = self.dim - self.obj + 1
        return 100 * (k + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5))))

    def f(self, x):
        f = []
        M = self.obj
        X_M = self.X_M(x)
        g = self.g(X_M)
        s1 = 0.5 * (1 + g)
        for i in range(M):
            f1 = s1
            f1 *= np.prod(x[:M - 1 - i])
            if i > 0: f1 *= 1 - x[M - 1 - i]
            f.append(f1)
        return f


class DTLZ4(DTLZ):
    def __init__(self):
        super().__init__('dtlz4', 3, (0.0, 1.0))
        self.alpha = 100

    def g(self, X_M):
        return np.sum(np.square(X_M - .5))

    def f(self, x):
        f = []
        X_M = self.X_M(x)
        g = self.g(X_M)
        alpha = self.alpha
        M = self.obj
        for i in range(M):
            f1 = (1 + g)
            f1 *= np.prod(np.cos(np.power(x[:M - 1 - i], alpha) * np.pi / 2))
            if i > 0: f1 *= np.sin(np.power(x[M - 1 - i], alpha) * np.pi / 2)
            f.append(f1)
        return f


class DTLZ5(DTLZ):
    def __init__(self):
        super().__init__('dtlz5', 3, (0.0, 1.0))

    def g(self, X_M):
        return np.sum(np.square(X_M - 0.5))

    def theta(self, x, g):
        theta = (np.pi / (4 * (1 + g))) * (1 + 2 * g * x)
        theta[0] = x[0]
        return theta

    def f(self, x):
        f = []
        X_M = self.X_M(x)
        g = self.g(X_M)
        theta = self.theta(x, g)
        M = self.obj
        for i in range(M):
            f1 = (1+g)
            if i > 0:   f1 *= np.sin(theta[M-1-i]*np.pi/2)
            if i < M-1: f1 *= np.prod(np.cos(theta[:M-1-i]*np.pi/2))
            f.append(f1)
        return f


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
        self.pop_igd = []
        self.problem = None
        self.pro_bounds = None
        self.pro_dim = None
        self.pro_obj = None
        self.pro_pof = None

    def load_problem(self, problem: MultiObjProblem):
        self.__init__(self.max_gen, self.pop_size, self.eta_crossover, self.eta_mutate)
        self.problem = problem
        self.pro_name = problem.name
        self.pro_obj = problem.obj
        self.pro_dim = problem.dim
        self.pro_bounds = problem.bounds
        self.pro_pof = np.loadtxt('pof/' + self.pro_name + '.csv', delimiter=',')

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
        result = (x <= y).all() and (x < y).any()
        return result

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
                if q_index == p_index: continue
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
                    list_dominated_count[q_index] -= 1
                    if (list_dominated_count[q_index]) <= 0:
                        self.pop_rank[q_index] = front_index + 1
                        front_next.append(q_index)
            front_index += 1
            fronts.append(front_next)
        self.pop_fronts = np.asarray(fronts)

    def __find_duplicates(self, X, epsilon=1e-16):
        D = sp.distance.cdist(X.astype(float), X.astype(float))
        D[np.triu_indices(len(X))] = np.Inf
        is_duplicate = np.any(D <= epsilon, axis=1)
        return is_duplicate

    def _compute_crowd(self, fronts):
        self.pop_crowd = np.full(len(self.population), -1.0)
        for front in fronts:
            F = self.pop_fitness[front]
            n_points, n_obj = F.shape
            if n_points <= 2:
                crowding = np.full(n_points, np.Inf)
            else:
                is_unique = np.where(np.logical_not(self.__find_duplicates(F, epsilon=1e-32)))[0]
                _F = F[is_unique]
                I = np.argsort(_F, axis=0, kind='mergesort')
                _F = _F[I, np.arange(n_obj)]
                dist = np.row_stack([_F, np.full(n_obj, np.Inf)]) - np.row_stack([np.full(n_obj, -np.Inf), _F])
                norm = np.max(_F, axis=0) - np.min(_F, axis=0)
                norm[norm == 0] = np.nan
                dist_to_last, dist_to_next = dist, np.copy(dist)
                dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm
                dist_to_last[np.isnan(dist_to_last)] = 0.0
                dist_to_next[np.isnan(dist_to_next)] = 0.0
                J = np.argsort(I, axis=0)
                _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj
                crowding = np.zeros(n_points)
                crowding[is_unique] = _cd
            for i in range(len(front)):
                self.pop_crowd[int(front[i])] = crowding[i]

    def _select_parent(self):
        parents = []
        for i in range(self.pop_size):
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

    def _crossover_mutate(self, probability_crossover, probability_mutate):
        children = []
        prob_cro = probability_crossover
        prob_mut = probability_mutate
        eta_cro = self.eta_crossover
        eta_mut = self.eta_mutate
        index_r = int(self.pop_size / 2)
        rand_prob_cro = np.random.uniform(low=0, high=1, size=index_r)
        rand_prob_mut = np.random.uniform(low=0, high=1, size=self.pop_size)
        rand_cro = np.random.uniform(low=0, high=1, size=(index_r, self.pro_dim))
        rand_mut = np.random.uniform(low=0, high=1, size=(self.pop_size, self.pro_dim))
        prop_cro = 1 / (1 + eta_cro)
        prop_mut = 1 / (eta_mut + 1)
        for index, i in enumerate(rand_cro):
            if rand_prob_cro[index] < prob_cro:
                l = self.pop_parents[index]
                r = self.pop_parents[index + index_r]
                for j in range(self.pro_dim):
                    if i[j] <= 0.5:
                        i[j] = (i[j] * 2) ** prop_cro
                    else:
                        i[j] = (1 / (2 - i[j] * 2)) ** prop_cro
                    if rand_mut[index][j] < 0.5:
                        rand_mut[index][j] = (2 * rand_mut[index][j]) ** prop_mut - 1
                    else:
                        rand_mut[index][j] = 1 - (2 * (1 - rand_mut[index][j])) ** prop_mut
                    if rand_mut[index + index_r][j] < 0.5:
                        rand_mut[index + index_r][j] = (2 * rand_mut[index + index_r][j]) ** prop_mut - 1
                    else:
                        rand_mut[index + index_r][j] = 1 - (2 * (1 - rand_mut[index + index_r][j])) ** prop_mut
                child_l = 0.5 * ((1 + i) * l + (1 - i) * r)
                child_r = 0.5 * ((1 - i) * l + (1 + i) * r)
                if rand_prob_mut[index] < prob_mut:
                    child_l = child_l + rand_mut[index]
                child_l = np.clip(child_l, self.pro_bounds[0], self.pro_bounds[1])
                if rand_prob_mut[index + index_r] < prob_mut:
                    child_r = child_r + rand_mut[index + index_r]
                child_r = np.clip(child_r, self.pro_bounds[0], self.pro_bounds[1])
                children.append(child_l)
                children.append(child_r)
        children = np.asarray(children)
        self.population = np.vstack((self.population, children))

    def _compute_igd(self):
        igd = []
        data = self.pop_fitness[self.pop_fronts[0]]
        for p in data:
            pof = self.pro_pof
            pof = np.abs(pof - p)
            pof = pof.sum(1)
            igd.append(min(pof))
        igd = np.mean(igd)
        self.pop_igd.append(igd)

    def _select_elitism(self):
        population = np.empty((0, self.pro_dim))
        size = 0
        for front in self.pop_fronts:
            size += len(front)
            if size < self.pop_size:
                population = np.vstack((population, self.population[front]))
            else:
                tmp = np.column_stack((front, self.pop_crowd[front]))
                tmp = self.population[tmp[np.argsort(-tmp[:, 1])][:, 0].astype(int)]
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
            self._crossover_mutate(0.9, 0.1)
            # rank & distance
            self._compute_fitness(self.population)
            self._compute_rank(self.population)
            # crowding-distance
            self._compute_crowd(self.pop_fronts)
            self._compute_igd()
            # elitism select
            if gen != self.max_gen - 1:
                self._select_elitism()
            # terminate
        plt.boxplot(self.pop_igd, showfliers=False)
        plt.show()
        display_data = self.pop_fitness[self.pop_fronts[0]]
        if self.pro_obj == 2:
            plt.scatter(display_data[:, 0], display_data[:, 1])
            plt.xlabel('f1')
            plt.ylabel('f2')
            plt.show()
        elif self.pro_obj == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(display_data[:, 0], display_data[:, 1], display_data[:, 2])
            ax.set_xlabel('f1')
            ax.set_ylabel('f2')
            ax.set_zlabel('f3')
            plt.show()
