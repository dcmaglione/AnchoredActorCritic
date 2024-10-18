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

def plot_fancy_violins(method: Method):
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.5, 3))

    def plot_violin(ax, naive: List[float], original: List[float], anchored: List[float], label: str):
        # Use white color for violin plots
        sns.violinplot(data=[naive, original, anchored], ax=ax, inner=None, cut=0, color="white", linewidth=1.0)
        
        # Create a custom colormap
        colors = ['#d7191c', '#404040', '#1a9641']  # Red to Dark Grey to Green
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
        
        def get_color(change):
            # Normalize change to [-1, 1] range
            norm_change = max(min(change / 0.5, 1), -1)
            return cmap(0.5 * (norm_change + 1))  # Map [-1, 1] to [0, 1]
        
        def draw_arrow(start, end, start_x, end_x):
            change = end - start
            color = get_color(change)
            ax.annotate('', xy=(end_x, end), xytext=(start_x, start),
                        arrowprops=dict(arrowstyle='->', color=color, alpha=0.3,
                                        linewidth=1.5, connectionstyle="arc3,rad=0"))
        
        for i, (n, o, a) in enumerate(zip(naive, original, anchored)):
            # Change dot color to black and keep size small
            ax.scatter(0, n, color='black', alpha=0.8, s=10, zorder=3)
            ax.scatter(1, o, color='black', alpha=0.8, s=10, zorder=3)
            ax.scatter(2, a, color='black', alpha=0.8, s=10, zorder=3)

            # Original to Naive arrow
            draw_arrow(o, n, 1, 0)
            # Original to Anchored arrow
            draw_arrow(o, a, 1, 2)
        
        # Adjust the title position
        ax.set_title(label, fontsize=10, pad=3)  # Reduced pad value
        
        ax.set_ylabel('Rewards', fontsize=8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Naively-tuned\non target', 'Trained\non source', 'Anchor-tuned\non target'], 
                           fontsize=8, rotation=0, ha='center')
        ax.tick_params(axis='x', which='major', pad=-8)  # Use negative padding to move labels higher
        ax.set_xlim(-0.5, 2.5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Make horizontal grid lines dashed and more transparent
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        
        # Remove vertical grid lines
        ax.xaxis.grid(False)
        
        # Adjust y-axis to ensure all points are visible
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    
    plot_violin(ax1, method.naive.Source, method.original.Source, method.anchored.Source, "Tested on Source")
    plot_violin(ax2, method.naive.Target, method.original.Target, method.anchored.Target, "Tested on Target")
    
    # Adjust subplot parameters
    plt.subplots_adjust(left=0.205,    # left margin
                        right=0.967,   # right margin
                        bottom=0.1,    # bottom margin
                        top=0.95,      # Increased top margin to lower titles
                        hspace=0.5)    # height space between subplots
    
    # Remove overall figure title if it exists
    fig.suptitle('')
    
    # Remove tight_layout() as it might interfere with manual adjustments
    # plt.tight_layout()

    plt.savefig('fancy_violins.svg', bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load pickle files from a folder structure and visualize results.')
    parser.add_argument('folder', type=str, help='folder containing the pickle files')
    parser.add_argument('--plot', action='store_true', help='generate a plot of the results')
    args = parser.parse_args()

    method = load_pickles_from_folder(Path(args.folder))
    print("Method:")
    print(method)
    result = calculate_results(method)
    print("\nResult:")
    print(result)

    if args.plot:
        plot_fancy_violins(method)
