import pickle
import argparse
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, TypeVar, Generic
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import os
from matplotlib.patches import FancyArrowPatch, ConnectionPatch

T = TypeVar('T')

@dataclass
class Sims(Generic[T]):
    Source: T
    Target: T
    def __str__(self):
        return f"Source: {self.Source}\nTarget: {self.Target}"

@dataclass
class Method:
    original:  Sims[List[float]]
    naive:     Sims[List[float]]
    anchored:  Sims[List[float]]
    def __str__(self):
        return f"Original:\n{self.original}\n\nNaive:\n{self.naive}\n\nAnchored:\n{self.anchored}"

@dataclass
class Result:
    naive:     Sims[Tuple[float, float]]
    anchored:  Sims[Tuple[float, float]]
    def __str__(self):
        return f"Naive:\n{self.naive}\n\nAnchored:\n{self.anchored}"

def load_pickle(file_path: Path) -> Dict:
    try:
        with file_path.open('rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def seed_from_path(path: Path) -> int:
    return int(path.parts[path.parts.index('seeds') + 1])

def find_seed(path: Path) -> int:
    hypers_path = (Path(*path.parts[:path.parts.index('seeds')]) / 'hypers.json')
    with hypers_path.open() as f:
        hypers = json.load(f)
        return seed_from_path(Path(hypers["prev_folder"])) if hypers["prev_folder"] else seed_from_path(path)

def process(data: Dict) -> List[float]:
    return [np.mean(data[key]) for key in sorted(data.keys(), key=find_seed)]

def load_sims(method_path: Path) -> Sims[List[float]]:
    return Sims(
        Source=process(load_pickle(method_path / 'Source.pkl')),
        Target=process(load_pickle(method_path / 'Target.pkl'))
    )

def load_pickles_from_folder(folder_path: Path) -> Method:
    return Method(
        original=load_sims(folder_path / 'original'),
        naive=load_sims(folder_path / 'naive'),
        anchored=load_sims(folder_path / 'anchored')
    )

def diff_mean_std(from_list: List[float], to_list: List[float]) -> Tuple[float, float]:
    diff = np.array(to_list) - np.array(from_list)
    return np.mean(diff), np.std(diff)

def calculate_results(method: Method) -> Result:
    return Result(
        naive=Sims(
            Source=diff_mean_std(method.original.Source, method.naive.Source),
            Target=diff_mean_std(method.original.Target, method.naive.Target)
        ),
        anchored=Sims(
            Source=diff_mean_std(method.original.Source, method.anchored.Source),
            Target=diff_mean_std(method.original.Target, method.anchored.Target)
        )
    )

def plot_fancy_violins(methods: Dict[str, Method], output_folder: str):
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    fig, axes = plt.subplots(3, 2, figsize=(3.5, 3.5), sharey='row')

    env_names = {
        'pendulum': 'Pendulum',
        'reacher': 'Reacher',
        'lander': 'Lunar Lander'
    }

    def plot_violin(ax, naive: List[float], original: List[float], anchored: List[float], show_x_labels: bool, show_y_label: bool, show_y_ticks: bool, env: str):
        positions = [0, 1, 2]
        data = [naive, original, anchored]
        
        parts = ax.violinplot(data, positions, points=100, widths=0.5, showmeans=False, showextrema=False, showmedians=False)
        
        for pc in parts['bodies']:
            pc.set_facecolor('white')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
        n_bins = 256
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
        
        all_changes = [n - o for n, o in zip(naive, original)] + [a - o for a, o in zip(anchored, original)]
        max_abs_change = max(abs(max(all_changes)), abs(min(all_changes)))
        
        def get_color(change):
            norm_change = change / max_abs_change
            norm_change = np.sign(norm_change) * np.power(abs(norm_change), 0.5)
            return cmap(0.5 * (norm_change + 1))
        
        def draw_arrow(start_x, start_y, end_x, end_y):
            change = end_y - start_y
            color = get_color(change)
            
            arrow = ConnectionPatch(
                xyA=(start_x, start_y), xyB=(end_x, end_y),
                coordsA="data", coordsB="data",
                axesA=ax, axesB=ax,
                arrowstyle="->, head_length=0.6, head_width=0.4", shrinkA=3.5, shrinkB=1.5,
                color=color, alpha=0.4, linewidth=1.3
            )
            ax.add_artist(arrow)
        
        for n, o, a in zip(naive, original, anchored):
            draw_arrow(1, o, 0, n)  # Original to Naive
            draw_arrow(1, o, 2, a)  # Original to Anchored
            
            ax.scatter(0, n, color='black', alpha=0.5, s=10, zorder=3, edgecolors='none')
            ax.scatter(1, o, color='black', alpha=0.5, s=10, zorder=3, edgecolors='none')
            ax.scatter(2, a, color='black', alpha=0.5, s=10, zorder=3, edgecolors='none')

        # Remove the title setting
        # ax.set_title(label, fontsize=10, pad=3)

        ax.set_xticks([0, 1, 2])
        if show_x_labels:
            ax.set_xticklabels(['T', 'S', 'T'],
                                rotation=0, ha='center')
            
            # Get the positions of the x-ticks
            tick_positions = ax.get_xticks()
            
            # Add arrows with labels
            arrow_props = dict(arrowstyle='->', color='gray', lw=1.5, shrinkA=10, shrinkB=7)
            
            # Arrow from Source to Naive Target
            ax.annotate('', xy=(tick_positions[0], -0.1), xytext=(tick_positions[1], -0.1),
                        xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                        arrowprops=arrow_props)
            ax.text((tick_positions[0] + tick_positions[1])/2, -0.21, 'naive',
                    ha='center', va='center', transform=ax.get_xaxis_transform())
            
            # Arrow from Source to Anchored Target
            ax.annotate('', xy=(tick_positions[2], -0.1), xytext=(tick_positions[1], -0.1),
                        xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                        arrowprops=arrow_props)
            ax.text((tick_positions[1] + tick_positions[2])/2, -0.21, '$\\mathbf{ours}$',
                    ha='center', va='center', transform=ax.get_xaxis_transform())
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis='x', which='major', pad=0)
        ax.set_xlim(-0.5, 2.5)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.xaxis.grid(False)

        # Move y-axis ticks to the right
        ax.yaxis.tick_right()

        if show_y_ticks:
            ax.tick_params(axis='y', which='major', labelsize=6, labelright=True)
        else:
            ax.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=False)

        if show_y_label:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel('Rewards', fontsize=8, rotation=270, labelpad=10)

        if env == 'pendulum':
            ax.set_ylim(top=400)

    for i, (env, method) in enumerate(methods.items()):
        plot_violin(axes[i, 0], method.naive.Source, method.original.Source, method.anchored.Source, 
                    i == 2, False, False, env)
        plot_violin(axes[i, 1], method.naive.Target, method.original.Target, method.anchored.Target, 
                    i == 2, True, True, env)
        
        # Set the environment label on the left, rotated 90 degrees
        axes[i, 0].set_ylabel(env_names[env], fontsize=10, rotation=90, ha='center', va='center')
        axes[i, 0].yaxis.set_label_position("left")

    # Add separation lines
    line_positions = [2/3, 1/3]  # Adjusted positions for two lines
    for y_position in line_positions:
        line = plt.Line2D([-0.03, 1.065], [y_position, y_position],  # Adjusted x-coordinates
                          transform=fig.transFigure, color='#cccccc', 
                          linestyle='-', linewidth=0.5)
        fig.add_artist(line)

    # Add titles only for the first row
    axes[0, 0].set_title("Policy evaluations on source", fontsize=10, pad=10)
    axes[0, 1].set_title("Policy evaluations on target", fontsize=10, pad=10)

    plt.tight_layout()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.1, hspace=0.1)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'sim2sim_violins.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Figure saved as {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load pickle files from multiple folders and visualize results.')
    parser.add_argument('folder', type=str, help='parent folder containing environment folders')
    parser.add_argument('--plot', action='store_true', help='generate a plot of the results')
    args = parser.parse_args()

    input_folder = Path(args.folder)
    methods = {}
    for env in ['pendulum', 'reacher', 'lander']:  # Changed the order here
        env_folder = input_folder / env
        if env_folder.exists():
            methods[env] = load_pickles_from_folder(env_folder)
            print(f"\nMethod for {env}:")
            print(methods[env])
            result = calculate_results(methods[env])
            print(f"\nResult for {env}:")
            print(result)

    if args.plot:
        plot_fancy_violins(methods, "plots")
