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

def plot_fancy_violins(method: Method, reward_upper_bound: float):
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2), sharey=True)

    def plot_violin(ax, naive: List[float], original: List[float], anchored: List[float], label: str):
        sns.violinplot(data=[naive, original, anchored], ax=ax, inner=None, cut=0, color="white", linewidth=1.0)
        
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
                arrowstyle="->", shrinkA=4, shrinkB=2,
                color=color, alpha=0.6, linewidth=1.5
            )
            ax.add_artist(arrow)
        
        for n, o, a in zip(naive, original, anchored):
            draw_arrow(1, o, 0, n)  # Original to Naive
            draw_arrow(1, o, 2, a)  # Original to Anchored
            
            ax.scatter(0, n, color='black', alpha=0.8, s=10, zorder=3, edgecolors='none')
            ax.scatter(1, o, color='black', alpha=0.8, s=10, zorder=3, edgecolors='none')
            ax.scatter(2, a, color='black', alpha=0.8, s=10, zorder=3, edgecolors='none')

        ax.set_title(label, fontsize=10, pad=3)
        if ax == ax1:
            ax.set_ylabel('Rewards', fontsize=8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Naively-tuned\non target', 'Trained\non source', 'Anchor-tuned\non target'], 
                           fontsize=8, rotation=0, ha='center')
        ax.tick_params(axis='x', which='major', pad=0)
        ax.set_xlim(-0.5, 2.5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.xaxis.grid(False)
        
        # Set y-axis limits based on the reward upper bound
        y_min = min(min(naive), min(original), min(anchored))
        y_max = max(max(naive), max(original), max(anchored), reward_upper_bound)
        padding = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - padding, y_max + padding)
    
    plot_violin(ax1, method.naive.Source, method.original.Source, method.anchored.Source, "Tested on Source")
    plot_violin(ax2, method.naive.Target, method.original.Target, method.anchored.Target, "Tested on Target")
    
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.9, wspace=0.1)
    fig.suptitle('')
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/fancy_violins.svg', bbox_inches='tight')
    print(f"Figure saved as {os.path.abspath('plots/fancy_violins.svg')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load pickle files from a folder structure and visualize results.')
    parser.add_argument('folder', type=str, help='folder containing the pickle files')
    parser.add_argument('--plot', action='store_true', help='generate a plot of the results')
    parser.add_argument('--reward_upper_bound', type=float, default=200, help='upper bound for the reward')
    args = parser.parse_args()

    method = load_pickles_from_folder(Path(args.folder))
    print("Method:")
    print(method)
    result = calculate_results(method)
    print("\nResult:")
    print(result)

    if args.plot:
        plot_fancy_violins(method, args.reward_upper_bound)
