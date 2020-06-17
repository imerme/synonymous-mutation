import numpy as np
import copy
from sklearn.model_selection import KFold, cross_validate
nr = np.random.RandomState(1)


class GaSnp(object):
    def __init__(self, data, N, build_func=None, patience=10, **kwargs):
        self.data = data
        self.best_indivadual = None
        self.best_fitness = 0
        self.population = self.init_population(N)
        self.size = len(self.population)
        self.fitness = None
        self.stop_condition = False
        self.build_func = build_func
        self.model_params = kwargs
        self.epochs = 0
        self.stop_counter = 0  # 记录没有连续提高的子代的次数。
        self.patience = patience  # 超过patience的没有连续提高的子代，迭代就会中止。

    def build_model(self):
        return self.build_func(**self.model_params)

    @classmethod
    def mask(cls, data):
        return cls._mask(data, mask)

    def _mask(self, data, mask):
        data = copy.deepcopy(self.data)
        for key in ['x_train', 'x_test']:
            data.__dict__[key] = data.__dict__[key][:, mask.astype(bool)]
        return data

    def init_population(self, n):
        return np.random.randint(2, size=[n, 40])

    def cross_mate(self, male, female):
        # 交换片段
        index = nr.randint(2, size=male.shape).astype(np.bool)
        male[index] = female[index]
        children = self.mutation(male)
        children[0] = self.best_individual  # 保留父代中最优个体
        return children

    def get_fitness(self, population):
        def get_individual_fitness(individual):
            # 数据处理
            data = self._mask(self.data, individual)
            # 模型训练
            model = self.build_model()

            out = cross_validate(model, data.x_train, data.y_train, n_jobs=5, scoring=['roc_auc'], cv=5)
            # 结果打分-auc
            return out['test_roc_auc'].mean()

        fitness = []
        for individual in population:
            fitness.append(get_individual_fitness(individual))
        fitness = np.array(fitness)
        temp = self.best_fitness
        self.best_fitness = np.argmax(fitness)

        # 跟新stop_counter
        if self.best_fitness - temp < 1e-4:
            self.stop_counter += 1
        else:
            self.stop_counter = 0

        # 如果stop_counter达到patience,则stop_condition为真。
        if self.stop_counter >= self.patience:
            self.stop_condition = True

        self.best_individual = population[self.best_fitness]
        return fitness

    def mutation(self, children):
        if nr.rand() > 0.995:
            index = nr.randint(2, size=children.size, dtype='bool')
            child[nr.randint(len(child))] = 1 - child[nr.randint(len(child))]
        return children

    def selection(self, population):
        fitness = self.get_fitness(population)
        fitness_p = fitness / sum(fitness)
        muting_pool = population[nr.choice(len(population), size=len(population), p=fitness_p)]
        return muting_pool

    def _split(self, muting_pool):
        male = muting_pool
        female = muting_pool.copy()
        nr.shuffle(female)
        return male, female

    def run(self):
        population = self.population
        while not self.stop_condition:
            self.epochs += 1
            muting_pool = self.selection(population)
            male, female = self._split(muting_pool)
            population = self.cross_mate(male, female)  # 子代
        else:
            self.population = population
            self.fitness = self.get_fitness(population)