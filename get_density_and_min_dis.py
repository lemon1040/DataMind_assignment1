import os
import math
import logging
from sklearn import preprocessing
from Distance.get_distance import *
import plot
import numpy as np

path = os.getcwd() + r'\data' + '\\'
logger = logging.getLogger('data_pretreatment')


class Pretreatment(object):

    def __init__(self):
        self.distance = {}
        self.data_size = 1
        self.result = []  # element: [rho, delta]
        self.rho_des_index = np.zeros(0)
        self.master = []  # delta point
        self.max_pos = -1
        self.max_density = -1  # the position where density is the max
        self.gamma = []  # rho * delta
        self.gamma_des_index = []  # Index from big to small
        self.cluster_center = []  # cluster center

        self.cluster_temp = []  # mark the cluster_center for every point
        self.cluster = []  # the final cluster

    def __initCluster__(self):
        self.cluster_temp = np.zeros(self.data_size)
        for center in self.cluster_center:
            self.cluster_temp[center] = center

    def load_dis_data(self, filename):
        """
        load data to memory
        """
        logger.info('load data')
        self.distance = {}  # the dictionary that save all of the distance between two vectors
        self.data_size = 1  # the size of vectors
        for line in open(path + filename, 'r'):
            x1, x2, d = line.strip().split(' ')
            x1, x2, d = int(x1), int(x2), float(d)
            self.data_size = max(x2 + 1, self.data_size)
            self.distance[(x1, x2)] = d
        self.master = np.zeros(self.data_size, dtype=int)
        logger.info('load accomplish')

    def get_dc(self, auto=False):
        """
        select the distance ranked
        if not auto, we will choose the distance at 2.5% top position as dc
        """
        if not auto:
            position = int((self.data_size * (self.data_size + 1) / 2 - self.data_size) * 0.018)
            dc = sorted(self.distance.items(), key=lambda item: item[1])[position][1]
            logger.info("dc - " + str(dc))
            return dc

    def calculate_density(self, dc):
        """
        calculate the density of each vector
        and get the max_pos
        """
        logger.info('calculate density begin')
        # func = lambda dij, dc: 1 if dij < dc else 0
        func = lambda dij, dc: math.exp(- (dij / dc) ** 2)
        max_density = -1
        for index in range(self.data_size):
            density = 0
            for front in range(index):
                density += func(self.distance[(front, index)], dc)
            for later in range(index + 1, self.data_size):
                density += func(self.distance[(index, later)], dc)
            self.result.append([density, float("inf")])
            max_density = max(max_density, density)
            if max_density == density:
                self.max_pos = index
                self.max_density = max_density
        self.result = np.array(self.result)
        self.rho_des_index = np.argsort(-self.result[:, 0])
        logger.info('calculate density end')

    def calculate_delta(self):
        """
        calculate the delta of each vector
        save the delta point as master
        """
        rho_des_index = self.rho_des_index
        self.result[rho_des_index[0]][1] = -1
        for i in range(1, self.data_size):
            for j in range(0, i):
                old_i, old_j = rho_des_index[i], rho_des_index[j]
                min_pos, max_pos = min(old_j, old_i), max(old_j, old_i)
                if self.distance[(min_pos, max_pos)] < self.result[old_i][1]:
                    self.result[old_i][1] = self.distance[(min_pos, max_pos)]
                    self.master[old_i] = old_j
        self.result[rho_des_index[0]][1] = max(self.result[:, 1])

        # for index in range(len(self.result))
        #     dis = float("inf")
        #     if index == self.max_pos:  # the max density position
        #         dis = 0
        #         for front in range(index):
        #             dis = max(self.distance[(front, index)], dis)
        #         for later in range(index + 1, self.data_size):
        #             dis = max(self.distance[(index, later)], dis)
        #         self.master[index] = index
        #     else:
        #         m_density = self.result[index][0]
        #         master_index = 0
        #         for front in range(index):
        #             if m_density < self.result[front][0]:
        #                 this_dis = self.distance[(front, index)]
        #                 dis = min(this_dis, dis)
        #                 if dis == this_dis:
        #                     master_index = front
        #         for later in range(index + 1, self.data_size):
        #             if m_density < self.result[later][0]:
        #                 this_dis = self.distance[(index, later)]
        #                 dis = min(this_dis, dis)
        #                 if dis == this_dis:
        #                     master_index = later
        #         self.master[index] = master_index
        #     self.result[index][1] = dis

    def calculate_gamma(self):
        """
        use the multiplication of normalized rho and delta as gamma to determine cluster center
        """
        scaler = preprocessing.MinMaxScaler()
        train_minmax = scaler.fit_transform(self.result)
        st_rho, st_delta = train_minmax[:, 0], train_minmax[:, 1]
        self.gamma = st_delta * st_rho
        self.gamma_des_index = np.argsort(-self.gamma)

    def calculate_cluster_center(self):
        """
        Intercept a point with gamma greater than 0.2 as the cluster center
        """
        self.cluster_center = np.where(self.gamma >= 0.2)[0]

    def get_center(self, filename):
        self.load_dis_data(filename)
        dc = self.get_dc()
        self.calculate_density(dc)
        self.calculate_delta()
        self.calculate_gamma()
        self.calculate_cluster_center()

    def get_cluster(self):
        self.__initCluster__()
        rho_des_index = self.rho_des_index
        for index in rho_des_index:
            if index not in self.cluster_center:
                self.cluster_temp[index] = self.cluster_temp[self.master[index]]
        for index in range(len(self.cluster_center)):
            self.cluster.append(np.where(self.cluster_temp == self.cluster_center[index])[0])


if __name__ == '__main__':
    pre = Pretreatment()
    pre.get_center('output.txt')
    plot.plot_diagram(np.arange(pre.data_size), sorted(pre.gamma, reverse=True), 'x', 'gamma', 'gamma diagram')
    print(pre.cluster_center)
    pre.get_cluster()
    print(pre.cluster)
    builder = GetDistance()
    builder.load('Aggregation.txt')
    # plot.plot_cluster('x', 'y', 'head', pre.cluster_center, builder.vectors)
    plot.plot_cluster('x', 'y', 'cluster', pre.cluster, builder.vectors)

