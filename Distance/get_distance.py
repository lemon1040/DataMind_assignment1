import os
import numpy as np


class GetDistance(object):

    def __init__(self):
        self.vectors = []
        self.path = os.getcwd() + r'\data' + '\\'

    def load(self, filename):
        """
        load data from file
        """
        for line in open(self.path + filename, 'r'):
            self.vectors.append(np.array(line.split(','), dtype=float))
        self.vectors = np.array(self.vectors)

    def calculate(self, fun, filename):
        """
        calculate the distance with given function and save the result
        """
        out = open(self.path + filename, 'w')
        for i in range(len(self.vectors) - 1):
            for j in range(i + 1, len(self.vectors)):
                out.write(str(i) + ' ' + str(j) + ' ' + str(fun.distance(self.vectors[i], self.vectors[j])) + '\n')
        out.close()
