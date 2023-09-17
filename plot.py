import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'

def plot_line(x, y, label, marker, linestyle = '-'):
    markersize = 8 if marker == 'x' else 12
    plt.plot(x, y, label = label, marker = marker, markersize = markersize, linestyle = linestyle)

if __name__ == '__main__':
    
    nodes_range = np.load(f'results/nodes_range.npy')
    
    sinkhorn_vios = np.load(f'results/sinkhorn_0.01_vios.npy')
    sinkhorn_times = np.load(f'results/sinkhorn_0.01_times.npy')

    ctreeot_vios = np.load(f'results/ctreeot_1_0.001_vios.npy')
    ctreeot_times = np.load(f'results/ctreeot_1_0.001_times.npy')

    fig = plt.figure(figsize=(9, 3))
    plot_line(nodes_range, sinkhorn_times, 'Sinkhorn', '.')
    plot_line(nodes_range, ctreeot_times, 'CTreeOT', '*')
    plt.xticks(np.arange(nodes_range.min(), nodes_range.max()+1, 10))
    plt.ylim([0, 0.2])
    plt.ylabel('Run time (s)', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0 + box.height*0.2, box.width, box.height*0.8])
    plt.legend(fontsize=14, loc='upper center', bbox_to_anchor = (0.5, -0.15), fancybox=True, ncol=2)
    plt.savefig('plots/runtime.png', dpi = 150, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(5, 5))
    plot_line(nodes_range, sinkhorn_vios, 'Sinkhorn', '.')
    plot_line(nodes_range, ctreeot_vios, 'CTreeOT', '*')
    plt.xticks(np.arange(nodes_range.min(), nodes_range.max()+1, 10))
    plt.ylabel('Number of violations', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0 + box.height*0.2, box.width, box.height*0.8])
    plt.legend(fontsize=14, loc='upper center', bbox_to_anchor = (0.5, -0.1), fancybox=True, ncol=2)
    plt.savefig('plots/violations.png', dpi = 150, bbox_inches='tight')
    plt.close(fig)