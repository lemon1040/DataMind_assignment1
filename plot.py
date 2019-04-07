import matplotlib.pyplot as plt
from get_density_and_min_dis import *


def plot_diagram(x, y, x_label, y_label, title, show_index=False, index=None, above=0):
    styles = ['k', 'g', 'r', 'c', 'm', 'y', 'b', '#9400D3', '#C0FF3E']
    plt.figure(0)
    plt.clf()
    plt.scatter(x, y, s=10, marker='.', color=styles[0])
    if show_index:
        for p_x, p_y, i in zip(x, y, index):
            plt.text(p_x, p_y + above, str(i), ha='center', va='bottom', fontsize=5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(title + '.png')
    plt.show()


def plot_cluster(x_label, y_label, title, clusters, vectors):
    styles = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#9400D3', '#C0FF3E']
    plt.figure(0)
    plt.clf()
    for index, cluster in enumerate(clusters):
        for point in cluster:
            plt.plot(vectors[point, 0], vectors[point, 1], marker='.', color=styles[index])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(title + '.png')
    plt.show()


if __name__ == '__main__':
    pre = Pretreatment()
    pre.load_dis_data('output.txt')
    dc = pre.get_dc()
    pre.calculate_density(dc)
    pre.calculate_delta()
    plot_diagram(pre.result[:, 0], pre.result[:, 1], 'rho', 'delta', 'Decision Graph')