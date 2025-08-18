"""
Visualization utilities for federated learning experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def plot_training_curves(
    history_dict, 
    title='Training Progression', 
    filename=None
):
    """Plot accuracy and loss curves for multiple algorithms."""
    plt.figure(figsize=(15, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    for algo, metrics in history_dict.items():
        rounds = range(1, len(metrics['accuracies']) + 1)
        plt.plot(rounds, metrics['accuracies'], marker='o', label=f"{algo}")
    
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for algo, metrics in history_dict.items():
        rounds = range(1, len(metrics['losses']) + 1)
        plt.plot(rounds, metrics['losses'], marker='o', label=f"{algo}")
    
    plt.title(f"{title} - Loss")
    plt.xlabel("Round")
    plt.ylabel("Test Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_comparison_barplot(
    summary, 
    title='Comparison of Aggregation Methods',
    xlabel='Alpha',
    ylabel='Accuracy', 
    filename=None
):
    """Create a bar plot comparing different aggregators across alpha values."""
    plt.figure(figsize=(12, 8))
    
    # Extract data
    alphas = sorted([float(alpha) for alpha in summary.keys()], reverse=True)  # Higher alpha (more IID) first
    aggregators = list(summary[alphas[0]].keys())
    
    # Set width of bars
    bar_width = 0.8 / len(aggregators)
    
    # Set positions for bars on X axis
    positions = np.arange(len(alphas))
    
    # Plot bars for each aggregator
    for i, agg in enumerate(aggregators):
        means = [summary[str(alpha)][agg]['accuracy_mean'] for alpha in alphas]
        stds = [summary[str(alpha)][agg]['accuracy_std'] for alpha in alphas]
        
        plt.bar(
            positions + i*bar_width - (len(aggregators)-1)*bar_width/2, 
            means,
            width=bar_width,
            yerr=stds,
            label=agg.upper(),
            capsize=5
        )
    
    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(positions, [f"{alpha}" for alpha in alphas])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_temperature_heatmap(
    summary,
    title='Impact of Temperature on Performance',
    filename=None
):
    """Create a heatmap showing the relationship between temperature, accuracy, and convergence."""
    plt.figure(figsize=(12, 8))
    
    # Extract data
    temps = sorted([float(temp) for temp in summary.keys()])
    accuracies = [summary[str(temp)]['accuracy_mean'] for temp in temps]
    conv_rounds = [summary[str(temp)]['convergence_round_mean'] for temp in temps]
    
    # Create a 2D grid for the heatmap
    grid_data = np.column_stack([accuracies, conv_rounds])
    
    # Create heatmap
    sns.heatmap(
        grid_data.reshape(1, -1, order='F').reshape(len(temps), 2),
        annot=True,
        fmt=".3f",
        xticklabels=["Accuracy", "Convergence Round"],
        yticklabels=[f"T={t}" for t in temps],
        cmap="viridis"
    )
    
    plt.title(title)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    
    # Also create a line plot showing the trade-off
    plt.figure(figsize=(10, 6))
    
    plt.plot(temps, accuracies, marker='o', label="Final Accuracy", color='blue')
    
    plt.xlabel("Temperature Parameter")
    plt.ylabel("Final Test Accuracy")
    plt.title("Temperature Parameter Impact on FedDive Performance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add a second y-axis for convergence speed
    ax2 = plt.gca().twinx()
    ax2.plot(temps, conv_rounds, marker='s', label="Convergence Round", color='red')
    ax2.set_ylabel("Convergence Round")
    
    # Add the second legend
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    
    if filename:
        plt.savefig(filename.replace('.png', '_tradeoff.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_adversarial_comparison(
    summary,
    title='Robustness Against Adversarial Clients',
    filename=None
):
    """Create a bar plot comparing performance of aggregators under adversarial conditions."""
    plt.figure(figsize=(12, 8))
    
    # Extract data
    aggregators = list(summary.keys())
    accuracies = [summary[agg]['accuracy_mean'] for agg in aggregators]
    stds = [summary[agg]['accuracy_std'] for agg in aggregators]
    
    # Create bar plot
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.bar(
        range(len(aggregators)),
        accuracies,
        yerr=stds,
        capsize=5,
        color=bar_colors[:len(aggregators)]
    )
    
    # Add labels and title
    plt.xlabel("Aggregation Strategy")
    plt.ylabel("Test Accuracy")
    plt.title(title)
    plt.xticks(range(len(aggregators)), [agg.upper() for agg in aggregators])
    plt.grid(axis='y', alpha=0.3)
    
    # Add text labels on bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()