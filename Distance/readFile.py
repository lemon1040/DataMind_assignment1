from Distance.get_distance import *
from Distance.distance import *
from plot import plot_diagram


if __name__ == '__main__':
    builder = GetDistance()
    builder.load('Aggregation.txt')
    # builder.calculate(SqrtDistance(), 'output.txt')
    plot_diagram(builder.vectors[:, 0], builder.vectors[:, 1], 'x', 'y', 'vector')
