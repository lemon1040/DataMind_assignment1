from Distance.get_distance import *
from Distance.distance import *
from plot import plot_diagram
import time


if __name__ == '__main__':
    start = time.time()
    builder = GetDistance()
    builder.load('Aggregation.txt')
    builder.calculate(SqrtDistance(), 'output.txt')
    plot_diagram(builder.vectors[:, 0], builder.vectors[:, 1], 'x', 'y', 'vector')
    end = time.time() - start
    print("Times: ", end)
