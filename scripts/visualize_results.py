#!/usr/bin/env python3
"""
Visualization Script for Apriori Benchmark Results

Generates comprehensive graphs comparing Traditional, Naive, and WDPA algorithms.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(results_file='results/benchmark/benchmark_results.json'):
    """Load benchmark results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_execution_time_comparison(results, output_dir='results/benchmark/plots'):
    """Plot execution time comparison across algorithms and support thresholds."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract data
    support_levels = []
    algorithms = {
        'Traditional': [],
        'Naive Parallel': [],
        'WDPA-BL': [],
        'WDPA-CL': [],
        'WDPA-BWT': [],
        'WDPA-CWT': []
    }

    for support_key, support_data in results['results_by_support'].items():
        # Extract support value
        min_support = support_data.get('traditional', {}).get('min_support', 0)
        if not min_support:
            min_support = list(support_data.values())[0].get('min_support', 0)

        support_levels.append(f"{min_support*100:.1f}%")

        # Extract times
        algorithms['Traditional'].append(support_data.get('traditional', {}).get('total_time', 0))
        algorithms['Naive Parallel'].append(support_data.get('naive_parallel', {}).get('total_time', 0))
        algorithms['WDPA-BL'].append(support_data.get('wdpa_BL', {}).get('total_time', 0))
        algorithms['WDPA-CL'].append(support_data.get('wdpa_CL', {}).get('total_time', 0))
        algorithms['WDPA-BWT'].append(support_data.get('wdpa_BWT', {}).get('total_time', 0))
        algorithms['WDPA-CWT'].append(support_data.get('wdpa_CWT', {}).get('total_time', 0))

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(support_levels))
    width = 0.12

    colors = {
        'Traditional': '#2C3E50',
        'Naive Parallel': '#E74C3C',
        'WDPA-BL': '#3498DB',
        'WDPA-CL': '#2ECC71',
        'WDPA-BWT': '#F39C12',
        'WDPA-CWT': '#9B59B6'
    }

    for i, (algo, times) in enumerate(algorithms.items()):
        offset = (i - 2.5) * width
        bars = ax.bar(x + offset, times, width, label=algo, color=colors[algo], alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}s',
                       ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_xlabel('Support Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Execution Time Comparison Across Algorithms', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/execution_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/execution_time_comparison.png")
    plt.close()


def plot_speedup_comparison(results, output_dir='results/benchmark/plots'):
    """Plot speedup comparison for parallel algorithms."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract data
    support_levels = []
    speedups = {
        'Naive Parallel': [],
        'WDPA-BL': [],
        'WDPA-CL': [],
        'WDPA-BWT': [],
        'WDPA-CWT': []
    }

    for support_key, support_data in results['results_by_support'].items():
        min_support = support_data.get('traditional', {}).get('min_support', 0)
        if not min_support:
            min_support = list(support_data.values())[0].get('min_support', 0)

        support_levels.append(f"{min_support*100:.1f}%")

        # Extract speedups
        speedups['Naive Parallel'].append(
            support_data.get('naive_parallel', {}).get('speedup_metrics', {}).get('speedup', 0)
        )
        speedups['WDPA-BL'].append(
            support_data.get('wdpa_BL', {}).get('speedup_metrics', {}).get('speedup', 0)
        )
        speedups['WDPA-CL'].append(
            support_data.get('wdpa_CL', {}).get('speedup_metrics', {}).get('speedup', 0)
        )
        speedups['WDPA-BWT'].append(
            support_data.get('wdpa_BWT', {}).get('speedup_metrics', {}).get('speedup', 0)
        )
        speedups['WDPA-CWT'].append(
            support_data.get('wdpa_CWT', {}).get('speedup_metrics', {}).get('speedup', 0)
        )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(support_levels))

    colors = {
        'Naive Parallel': '#E74C3C',
        'WDPA-BL': '#3498DB',
        'WDPA-CL': '#2ECC71',
        'WDPA-BWT': '#F39C12',
        'WDPA-CWT': '#9B59B6'
    }

    markers = {
        'Naive Parallel': 'o',
        'WDPA-BL': 's',
        'WDPA-CL': '^',
        'WDPA-BWT': 'D',
        'WDPA-CWT': 'v'
    }

    for algo, values in speedups.items():
        ax.plot(x, values, marker=markers[algo], linewidth=2.5, markersize=10,
                label=algo, color=colors[algo], alpha=0.8)

        # Add value labels
        for i, v in enumerate(values):
            if v > 0:
                ax.text(i, v + 0.05, f'{v:.2f}x', ha='center', va='bottom', fontsize=9)

    # Add baseline line at y=1
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (1x)')

    ax.set_xlabel('Support Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup vs Traditional Apriori', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Comparison: Parallel Algorithms vs Traditional', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/speedup_comparison.png")
    plt.close()


def plot_efficiency_comparison(results, output_dir='results/benchmark/plots'):
    """Plot efficiency comparison for parallel algorithms."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract data
    support_levels = []
    efficiencies = {
        'Naive Parallel': [],
        'WDPA-BL': [],
        'WDPA-CL': [],
        'WDPA-BWT': [],
        'WDPA-CWT': []
    }

    for support_key, support_data in results['results_by_support'].items():
        min_support = support_data.get('traditional', {}).get('min_support', 0)
        if not min_support:
            min_support = list(support_data.values())[0].get('min_support', 0)

        support_levels.append(f"{min_support*100:.1f}%")

        # Extract efficiencies (convert to percentage)
        efficiencies['Naive Parallel'].append(
            support_data.get('naive_parallel', {}).get('speedup_metrics', {}).get('efficiency', 0) * 100
        )
        efficiencies['WDPA-BL'].append(
            support_data.get('wdpa_BL', {}).get('speedup_metrics', {}).get('efficiency', 0) * 100
        )
        efficiencies['WDPA-CL'].append(
            support_data.get('wdpa_CL', {}).get('speedup_metrics', {}).get('efficiency', 0) * 100
        )
        efficiencies['WDPA-BWT'].append(
            support_data.get('wdpa_BWT', {}).get('speedup_metrics', {}).get('efficiency', 0) * 100
        )
        efficiencies['WDPA-CWT'].append(
            support_data.get('wdpa_CWT', {}).get('speedup_metrics', {}).get('efficiency', 0) * 100
        )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(support_levels))
    width = 0.15

    colors = {
        'Naive Parallel': '#E74C3C',
        'WDPA-BL': '#3498DB',
        'WDPA-CL': '#2ECC71',
        'WDPA-BWT': '#F39C12',
        'WDPA-CWT': '#9B59B6'
    }

    for i, (algo, values) in enumerate(efficiencies.items()):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=algo, color=colors[algo], alpha=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)

    # Add ideal efficiency line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (100%)')

    ax.set_xlabel('Support Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parallel Efficiency: Speedup / Number of Processors', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/efficiency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/efficiency_comparison.png")
    plt.close()


def plot_wdpa_strategies_detailed(results, output_dir='results/benchmark/plots'):
    """Detailed comparison of WDPA strategies only."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract data for lowest support (best parallelization case)
    first_support_key = list(results['results_by_support'].keys())[0]
    support_data = results['results_by_support'][first_support_key]
    min_support = support_data.get('traditional', {}).get('min_support', 0)

    strategies = ['BL', 'CL', 'BWT', 'CWT']
    metrics = {
        'Time (s)': [],
        'Speedup': [],
        'Efficiency (%)': []
    }

    for strategy in strategies:
        key = f'wdpa_{strategy}'
        data = support_data.get(key, {})
        metrics['Time (s)'].append(data.get('total_time', 0))
        metrics['Speedup'].append(data.get('speedup_metrics', {}).get('speedup', 0))
        metrics['Efficiency (%)'].append(data.get('speedup_metrics', {}).get('efficiency', 0) * 100)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    # Plot 1: Execution Time
    bars1 = axes[0].bar(strategies, metrics['Time (s)'], color=colors, alpha=0.8)
    axes[0].set_ylabel('Execution Time (seconds)', fontweight='bold')
    axes[0].set_title(f'Execution Time\n(Support: {min_support*100:.1f}%)', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Speedup
    bars2 = axes[1].bar(strategies, metrics['Speedup'], color=colors, alpha=0.8)
    axes[1].set_ylabel('Speedup vs Traditional', fontweight='bold')
    axes[1].set_title(f'Speedup\n(Support: {min_support*100:.1f}%)', fontweight='bold')
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}x', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Efficiency
    bars3 = axes[2].bar(strategies, metrics['Efficiency (%)'], color=colors, alpha=0.8)
    axes[2].set_ylabel('Efficiency (%)', fontweight='bold')
    axes[2].set_title(f'Parallel Efficiency\n(Support: {min_support*100:.1f}%)', fontweight='bold')
    axes[2].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_ylim(0, 100)
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('WDPA Lattice Distribution Strategies Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wdpa_strategies_detailed.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/wdpa_strategies_detailed.png")
    plt.close()


def plot_best_algorithm_summary(results, output_dir='results/benchmark/plots'):
    """Create a summary showing the best algorithm for each support level."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    support_levels = []
    best_algo = []
    best_time = []
    baseline_time = []

    for support_key, support_data in results['results_by_support'].items():
        min_support = support_data.get('traditional', {}).get('min_support', 0)
        if not min_support:
            min_support = list(support_data.values())[0].get('min_support', 0)

        support_levels.append(f"{min_support*100:.1f}%")

        # Find best algorithm
        times = {}
        times['Traditional'] = support_data.get('traditional', {}).get('total_time', float('inf'))
        times['Naive'] = support_data.get('naive_parallel', {}).get('total_time', float('inf'))
        times['WDPA-BL'] = support_data.get('wdpa_BL', {}).get('total_time', float('inf'))
        times['WDPA-CL'] = support_data.get('wdpa_CL', {}).get('total_time', float('inf'))
        times['WDPA-BWT'] = support_data.get('wdpa_BWT', {}).get('total_time', float('inf'))
        times['WDPA-CWT'] = support_data.get('wdpa_CWT', {}).get('total_time', float('inf'))

        best = min(times.items(), key=lambda x: x[1])
        best_algo.append(best[0])
        best_time.append(best[1])
        baseline_time.append(times['Traditional'])

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(support_levels))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_time, width, label='Traditional Apriori',
                   color='#2C3E50', alpha=0.8)
    bars2 = ax.bar(x + width/2, best_time, width, label='Best Parallel Algorithm',
                   color='#27AE60', alpha=0.8)

    # Add labels
    for i, (bar1, bar2, algo) in enumerate(zip(bars1, bars2, best_algo)):
        # Baseline label
        height1 = bar1.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1,
               f'{height1:.2f}s', ha='center', va='bottom', fontsize=9)

        # Best algorithm label
        height2 = bar2.get_height()
        speedup = height1 / height2 if height2 > 0 else 0
        ax.text(bar2.get_x() + bar2.get_width()/2., height2,
               f'{height2:.2f}s\n({algo})\n{speedup:.2f}x',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Support Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Best Algorithm Performance vs Traditional Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_algorithm_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/best_algorithm_summary.png")
    plt.close()


def generate_all_plots(results_file='results/benchmark/benchmark_results.json',
                       output_dir='results/benchmark/plots'):
    """Generate all visualization plots."""
    print("="*80)
    print("BENCHMARK RESULTS VISUALIZATION")
    print("="*80)
    print(f"Loading results from: {results_file}")

    results = load_results(results_file)

    print(f"\nGenerating plots in: {output_dir}")
    print("-"*80)

    plot_execution_time_comparison(results, output_dir)
    plot_speedup_comparison(results, output_dir)
    plot_efficiency_comparison(results, output_dir)
    plot_wdpa_strategies_detailed(results, output_dir)
    plot_best_algorithm_summary(results, output_dir)

    print("-"*80)
    print("\nAll plots generated successfully!")
    print("="*80)
    print(f"\nView your plots in: {output_dir}/")
    print("  - execution_time_comparison.png")
    print("  - speedup_comparison.png")
    print("  - efficiency_comparison.png")
    print("  - wdpa_strategies_detailed.png")
    print("  - best_algorithm_summary.png")
    print("="*80)


if __name__ == "__main__":
    generate_all_plots()
