import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def plot_fingertip_position(runs, ax, title, show_legend=True, show_y_label=True):
    # Extended color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
        '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
        '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    for i, run in enumerate(runs):
        data = run[:, 8:10]
        
        t = np.arange(len(data))
        t_smooth = np.linspace(0, len(data) - 1, 1000)
        f_x = interp1d(t, data[:, 0], kind='cubic')
        f_y = interp1d(t, data[:, 1], kind='cubic')
        x_smooth = f_x(t_smooth)
        y_smooth = f_y(t_smooth)
        
        ax.plot(x_smooth, y_smooth, color=colors[i], label=f'Run {i+1}', alpha=0.7)
        
        diffs = np.diff(np.column_stack((x_smooth, y_smooth)), axis=0)
        distances = np.sqrt((diffs**2).sum(axis=1))
        cumulative_distances = np.cumsum(distances)
        total_distance = cumulative_distances[-1]
        
        min_arrow_distance = 0.05
        num_arrows = min(20, max(5, int(total_distance / min_arrow_distance)))
        arrow_distances = np.linspace(0, total_distance, num_arrows + 2)[1:-1]
        
        for distance in arrow_distances:
            idx = np.searchsorted(cumulative_distances, distance)
            if idx < len(x_smooth) - 1:
                ax.annotate('', xy=(x_smooth[idx+1], y_smooth[idx+1]), 
                            xytext=(x_smooth[idx], y_smooth[idx]),
                            arrowprops=dict(arrowstyle='->', color=colors[i],
                                            lw=1, alpha=0.7, mutation_scale=10))
    
    ax.plot(0, 0, 'kx', markersize=10, markeredgewidth=2, label='Target', zorder=10)
    
    ax.set_xlabel('Relative X-position')
    if show_y_label:
        ax.set_ylabel('Relative Y-position')
    ax.set_title(title)
    if show_legend:
        ax.legend(loc='best', frameon=True, framealpha=0.7, fontsize='x-small')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.25, 0.25)
    
    # Hide ticks on both axes
    ax.set_xticks([])
    ax.set_yticks([])

if __name__ == "__main__":
    anchored = pickle.load(open('results/reacher/runs_anchored.pkl', 'rb'))[:10]
    naive = pickle.load(open('results/reacher/runs_naive.pkl', 'rb'))[:10]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.4))
    
    plot_fingertip_position(naive, ax1, 'Fine-tuned naively', show_legend=False, show_y_label=True)
    plot_fingertip_position(anchored, ax2, 'Fine-tuned with anchors', show_legend=True, show_y_label=False)
    
    plt.tight_layout()
    plt.savefig('plots/reacher/reacher_evolution.svg', bbox_inches='tight')
