'''
"""
This script creates a visualization consisting of two violin plots with the following features:

1. Two categories of data represented as violin plots side by side.
2. Individual data points plotted within each violin plot ala a beeswarm plot.
3. Arrows connecting corresponding data points between the two violin plots, 
   showing how each point "moved" from one category to the other.

The resulting plot provides a clear visual representation of:
- The distribution of data in each category (through the violin plots)
- The individual data points (as scatter plots within the violins)
- The change or movement of each data point between categories (via connecting arrows)

This visualization is particularly useful for comparing paired data or 
showing before-and-after scenarios where you want to highlight both the 
overall distribution changes and individual data point movements.
"""
'''


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set the style and color palette
sns.set_style("whitegrid")
sns.set_palette("Set2")

np.random.seed(42)  # For reproducibility
category1 = np.random.normal(0, 1, 6)
category2 = category1 + np.random.normal(0.5, 0.5, 6)  # Simulating movement

fig, ax = plt.subplots(figsize=(10, 6))

# Create violin plots using Seaborn
sns.violinplot(data=[category1, category2], ax=ax, inner=None, cut=0)

# Function to get color based on change
def get_color(change):
    if change > 0:
        return plt.cm.Greens(min(change / 1.5, 1))  # Cap at 1
    else:
        return plt.cm.Reds(min(-change / 1.5, 1))  # Cap at 1

# Add individual points and arrows
for i, (c1, c2) in enumerate(zip(category1, category2)):
    ax.scatter(0, c1, color='#1f77b4', alpha=0.8, s=100, zorder=3)
    ax.scatter(1, c2, color='#ff7f0e', alpha=0.8, s=100, zorder=3)
    
    # Straight arrows with color based on change
    change = c2 - c1
    color = get_color(change)
    ax.annotate('', xy=(1, c2), xytext=(0, c1),
                arrowprops=dict(arrowstyle='->', color=color, alpha=0.8,
                                linewidth=2, connectionstyle="arc3,rad=0"))

# Customize the plot
ax.set_title('Violin Plot with Data Points and Movement Arrows', fontsize=16)
ax.set_ylabel('Values', fontsize=12)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Category 1', 'Category 2'], fontsize=12)
ax.set_xlim(-0.5, 1.5)

# Remove top and right spines
sns.despine()

plt.tight_layout()
plt.show()
