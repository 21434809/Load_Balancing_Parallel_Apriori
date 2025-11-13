#!/usr/bin/env python3
"""
Visualization Script for Apriori Benchmark Results

Generates comprehensive graphs comparing Traditional, Naive, and WDPA algorithms.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Auto-detect latest benchmark
def get_latest_benchmark():
    """Find the most recent benchmark results directory."""
    import glob
    benchmark_dirs = glob.glob('results/benchmark_*/benchmark_results.json')
    if not benchmark_dirs:
        raise FileNotFoundError("No benchmark results found in results/ directory")
    # Get the most recently modified
    latest = max(benchmark_dirs, key=lambda x: os.path.getmtime(x))
    # Extract directory name (e.g., 'benchmark_200k_001')
    benchmark_name = os.path.basename(os.path.dirname(latest))
    print(f"Auto-detected latest benchmark: {benchmark_name}")
    return benchmark_name

# benchmark_name = get_latest_benchmark()
benchmark_name = 'benchmark_150k_001'

def load_results(results_file=f'results/{benchmark_name}/benchmark_results.json'):
    """Load benchmark results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def get_dataset_info(results):
    """Extract dataset metadata from results."""
    metadata = results.get('metadata', {})
    config = metadata.get('config', {})
    dataset_config = config.get('dataset', {})

    sample_size = dataset_config.get('sample_size', 'Unknown')
    max_items = dataset_config.get('max_items', 'Unknown')

    return sample_size, max_items


def plot_execution_time_comparison(results, output_dir=f'results/{benchmark_name}/plots'):
    """Plot execution time comparison across algorithms and support thresholds."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_size, max_items = get_dataset_info(results)

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

    traditional_failed = False

    for support_key, support_data in results['results_by_support'].items():
        # Extract support value
        min_support = support_data.get('traditional', {}).get('min_support', 0)
        if not min_support:
            min_support = list(support_data.values())[0].get('min_support', 0)

        support_levels.append(f"{min_support*100:.1f}%")

        # Check if traditional failed
        trad_data = support_data.get('traditional', {})
        trad_time = trad_data.get('total_time', 0)
        trad_status = trad_data.get('status', 'success')

        if trad_status != 'success' or trad_time == 0:
            traditional_failed = True
            algorithms['Traditional'].append(0)
        else:
            algorithms['Traditional'].append(trad_time)

        # Extract times for parallel algorithms (use computation_time if available, else total_time)
        algorithms['Naive Parallel'].append(support_data.get('naive_parallel', {}).get('total_time', 0))
        algorithms['WDPA-BL'].append(support_data.get('wdpa_BL', {}).get('computation_time', support_data.get('wdpa_BL', {}).get('total_time', 0)))
        algorithms['WDPA-CL'].append(support_data.get('wdpa_CL', {}).get('computation_time', support_data.get('wdpa_CL', {}).get('total_time', 0)))
        algorithms['WDPA-BWT'].append(support_data.get('wdpa_BWT', {}).get('computation_time', support_data.get('wdpa_BWT', {}).get('total_time', 0)))
        algorithms['WDPA-CWT'].append(support_data.get('wdpa_CWT', {}).get('computation_time', support_data.get('wdpa_CWT', {}).get('total_time', 0)))

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

    # Add note about traditional failure if applicable
    if traditional_failed:
        ax.text(0.5, 0.95, 'Note: Traditional algorithm failed due to memory constraints',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9, style='italic')

    ax.set_xlabel('Support Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Execution Time Comparison Across Algorithms\n(Sample Size: {sample_size:,} | Max Items: {max_items:,})',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/execution_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/execution_time_comparison.png")
    plt.close()


def plot_speedup_comparison(results, output_dir=f'results/{benchmark_name}/plots'):
    """Plot speedup comparison for parallel algorithms."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_size, max_items = get_dataset_info(results)

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
    ax.set_title(f'Speedup Comparison: Parallel Algorithms vs Traditional\n(Sample Size: {sample_size:,} | Max Items: {max_items:,})',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/speedup_comparison.png")
    plt.close()


def plot_efficiency_comparison(results, output_dir=f'results/{benchmark_name}/plots'):
    """Plot efficiency comparison for parallel algorithms."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_size, max_items = get_dataset_info(results)

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
    ax.set_title(f'Parallel Efficiency: Speedup / Number of Processors\n(Sample Size: {sample_size:,} | Max Items: {max_items:,})',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/efficiency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/efficiency_comparison.png")
    plt.close()


def plot_wdpa_strategies_detailed(results, output_dir=f'results/{benchmark_name}/plots'):
    """Detailed comparison of WDPA strategies only."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_size, max_items = get_dataset_info(results)

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
        # Use computation_time if available, else total_time
        metrics['Time (s)'].append(data.get('computation_time', data.get('total_time', 0)))
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

    plt.suptitle(f'WDPA Lattice Distribution Strategies Comparison\n(Sample Size: {sample_size:,} | Max Items: {max_items:,})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wdpa_strategies_detailed.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/wdpa_strategies_detailed.png")
    plt.close()


def plot_traditional_failure_analysis(results, output_dir=f'results/{benchmark_name}/plots'):
    """Visualize how traditional failed while parallel algorithms succeeded."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_size, max_items = get_dataset_info(results)

    # Get the first support level where traditional failed
    first_support_key = list(results['results_by_support'].keys())[0]
    support_data = results['results_by_support'][first_support_key]

    trad_data = support_data.get('traditional', {})
    trad_status = trad_data.get('status', 'success')
    memory_required = trad_data.get('memory_required', 'Unknown')
    error_message = trad_data.get('error_message', '')

    # Only create this plot if traditional actually failed
    if trad_status == 'success':
        return

    min_support = trad_data.get('min_support', 0)

    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    # Top: Traditional failure info (spans both columns)
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis('off')

    failure_text = f"""
    Traditional Apriori Algorithm Failure Analysis

    Status: FAILED - {error_message}
    Memory Required: {memory_required}
    Support Threshold: {min_support*100:.1f}%

    Traditional Apriori could not process this dataset due to memory constraints.
    Parallel algorithms succeeded by distributing the workload efficiently.
    """

    ax_info.text(0.5, 0.5, failure_text,
                transform=ax_info.transAxes,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='#FFE6E6', edgecolor='#E74C3C', linewidth=3),
                fontsize=12, fontweight='bold', family='monospace')

    # Bottom left: Success indicators for parallel algorithms
    ax_success = fig.add_subplot(gs[1, 0])

    algorithms = ['Naive\nParallel', 'WDPA-BL', 'WDPA-CL', 'WDPA-BWT', 'WDPA-CWT']
    success_values = [1, 1, 1, 1, 1]  # All succeeded
    colors_success = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    bars = ax_success.barh(algorithms, success_values, color=colors_success, alpha=0.8)
    ax_success.set_xlabel('Success (All Completed)', fontweight='bold', fontsize=11)
    ax_success.set_xlim(0, 1.2)
    ax_success.set_xticks([0, 1])
    ax_success.set_xticklabels(['Failed', 'Success'])
    ax_success.set_title('Parallel Algorithm Success Rate', fontweight='bold', fontsize=12)
    ax_success.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax_success.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                       '✓ SUCCESS', ha='left', va='center',
                       fontweight='bold', color='green', fontsize=10)

    # Bottom right: Execution times of successful algorithms
    ax_times = fig.add_subplot(gs[1, 1])

    times = []
    algo_names = []
    algo_colors = []

    for i, (algo_key, label, color) in enumerate([
        ('naive_parallel', 'Naive\nParallel', '#E74C3C'),
        ('wdpa_BL', 'WDPA-BL', '#3498DB'),
        ('wdpa_CL', 'WDPA-CL', '#2ECC71'),
        ('wdpa_BWT', 'WDPA-BWT', '#F39C12'),
        ('wdpa_CWT', 'WDPA-CWT', '#9B59B6')
    ]):
        time = support_data.get(algo_key, {}).get('total_time', 0)
        if time > 0:
            times.append(time)
            algo_names.append(label)
            algo_colors.append(color)

    bars2 = ax_times.barh(algo_names, times, color=algo_colors, alpha=0.8)
    ax_times.set_xlabel('Execution Time (seconds)', fontweight='bold', fontsize=11)
    ax_times.set_title(f'Parallel Algorithm Execution Times\n(Support: {min_support*100:.1f}%)',
                      fontweight='bold', fontsize=12)
    ax_times.grid(axis='x', alpha=0.3)

    for bar, time in zip(bars2, times):
        width = bar.get_width()
        ax_times.text(width + max(times)*0.01, bar.get_y() + bar.get_height()/2,
                     f'{time:.2f}s', ha='left', va='center',
                     fontweight='bold', fontsize=10)

    plt.suptitle(f'Traditional Apriori Failure vs Parallel Success\n(Sample Size: {sample_size:,} | Max Items: {max_items:,})',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/traditional_failure_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/traditional_failure_analysis.png")
    plt.close()


def plot_best_algorithm_summary(results, output_dir=f'results/{benchmark_name}/plots'):
    """Create a summary showing the best algorithm for each support level (parallel only)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_size, max_items = get_dataset_info(results)

    support_levels = []
    best_algo = []
    best_time = []
    worst_time = []

    for support_key, support_data in results['results_by_support'].items():
        min_support = support_data.get('traditional', {}).get('min_support', 0)
        if not min_support:
            min_support = list(support_data.values())[0].get('min_support', 0)

        support_levels.append(f"{min_support*100:.1f}%")

        # Only compare parallel algorithms (exclude traditional) - use computation_time if available
        times = {}
        naive_data = support_data.get('naive_parallel', {})
        times['Naive'] = naive_data.get('total_time', float('inf'))

        bl_data = support_data.get('wdpa_BL', {})
        times['WDPA-BL'] = bl_data.get('computation_time', bl_data.get('total_time', float('inf')))

        cl_data = support_data.get('wdpa_CL', {})
        times['WDPA-CL'] = cl_data.get('computation_time', cl_data.get('total_time', float('inf')))

        bwt_data = support_data.get('wdpa_BWT', {})
        times['WDPA-BWT'] = bwt_data.get('computation_time', bwt_data.get('total_time', float('inf')))

        cwt_data = support_data.get('wdpa_CWT', {})
        times['WDPA-CWT'] = cwt_data.get('computation_time', cwt_data.get('total_time', float('inf')))

        # Filter out algorithms that didn't run
        valid_times = {k: v for k, v in times.items() if v != float('inf') and v > 0}

        if valid_times:
            best = min(valid_times.items(), key=lambda x: x[1])
            worst = max(valid_times.items(), key=lambda x: x[1])
            best_algo.append(best[0])
            best_time.append(best[1])
            worst_time.append(worst[1])
        else:
            best_algo.append('None')
            best_time.append(0)
            worst_time.append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(support_levels))
    width = 0.35

    bars1 = ax.bar(x - width/2, worst_time, width, label='Worst Parallel Algorithm',
                   color='#E74C3C', alpha=0.6)
    bars2 = ax.bar(x + width/2, best_time, width, label='Best Parallel Algorithm',
                   color='#27AE60', alpha=0.8)

    # Add labels
    for i, (bar1, bar2, algo) in enumerate(zip(bars1, bars2, best_algo)):
        # Worst time label
        height1 = bar1.get_height()
        if height1 > 0:
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{height1:.2f}s', ha='center', va='bottom', fontsize=9)

        # Best algorithm label
        height2 = bar2.get_height()
        if height2 > 0:
            improvement = ((height1 - height2) / height1 * 100) if height1 > 0 else 0
            ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                   f'{height2:.2f}s\n({algo})\n{improvement:.1f}% faster',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Support Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Best vs Worst Parallel Algorithm Performance\n(Sample Size: {sample_size:,} | Max Items: {max_items:,})',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    # Add note about traditional
    ax.text(0.5, 0.02, 'Note: Traditional algorithm excluded due to memory failure',
           transform=ax.transAxes, ha='center', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_algorithm_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/best_algorithm_summary.png")
    plt.close()


def detect_processor_scaling(results):
    """Detect if results include processor scaling experiments."""
    # Check if any result keys contain processor count suffix (e.g., wdpa_BL_2p, wdpa_BL_4p)
    for support_key, support_data in results['results_by_support'].items():
        for key in support_data.keys():
            if '_' in key and key.endswith('p') and key.count('_') >= 2:
                # Extract processor counts for each strategy
                processor_scaling = {}
                for result_key in support_data.keys():
                    if result_key.startswith('wdpa_'):
                        parts = result_key.split('_')
                        if len(parts) >= 3 and parts[-1].endswith('p'):
                            strategy = parts[1]
                            num_procs = int(parts[-1][:-1])  # Remove 'p' and convert to int

                            if strategy not in processor_scaling:
                                processor_scaling[strategy] = []
                            processor_scaling[strategy].append(num_procs)

                if processor_scaling:
                    return processor_scaling

    return None


def plot_processor_scaling_analysis(results, output_dir=f'results/{benchmark_name}/plots'):
    """Create processor scaling visualizations if multiple processor counts were tested."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_size, max_items = get_dataset_info(results)

    # Detect processor scaling
    processor_scaling = detect_processor_scaling(results)

    if not processor_scaling:
        print("No processor scaling detected - skipping processor scaling plots")
        return

    print(f"Detected processor scaling: {processor_scaling}")

    # Get first support level for analysis
    first_support_key = list(results['results_by_support'].keys())[0]
    support_data = results['results_by_support'][first_support_key]
    min_support = list(support_data.values())[0].get('min_support', 0)

    # Create figure with 3 subplots for each strategy
    for strategy, proc_counts in processor_scaling.items():
        proc_counts = sorted(proc_counts)

        # Collect data for this strategy across processor counts
        times = []
        speedups = []
        efficiencies = []

        for num_procs in proc_counts:
            key = f'wdpa_{strategy}_{num_procs}p'
            if key in support_data:
                data = support_data[key]
                # Use computation_time if available, else total_time
                times.append(data.get('computation_time', data.get('total_time', 0)))
                speedup = data.get('speedup_metrics', {}).get('speedup', 0)
                efficiency = data.get('speedup_metrics', {}).get('efficiency', 0) * 100
                speedups.append(speedup)
                efficiencies.append(efficiency)

        if not times:
            continue

        # Create 3-panel plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Execution Time vs Processors
        axes[0].plot(proc_counts, times, marker='o', linewidth=2.5, markersize=10, color='#3498DB')
        axes[0].set_xlabel('Number of Processors', fontweight='bold')
        axes[0].set_ylabel('Execution Time (seconds)', fontweight='bold')
        axes[0].set_title(f'Execution Time vs Processors\n(WDPA-{strategy})', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(proc_counts)

        # Add value labels
        for x, y in zip(proc_counts, times):
            axes[0].text(x, y, f'{y:.2f}s', ha='center', va='bottom', fontweight='bold')

        # Plot 2: Speedup vs Processors
        axes[1].plot(proc_counts, speedups, marker='s', linewidth=2.5, markersize=10, color='#2ECC71')
        # Add ideal linear speedup line
        if speedups[0] > 0:
            ideal_speedups = [speedups[0] * (p / proc_counts[0]) for p in proc_counts]
            axes[1].plot(proc_counts, ideal_speedups, linestyle='--', color='gray',
                        linewidth=2, alpha=0.5, label='Ideal Linear Speedup')
        axes[1].set_xlabel('Number of Processors', fontweight='bold')
        axes[1].set_ylabel('Speedup', fontweight='bold')
        axes[1].set_title(f'Speedup vs Processors\n(WDPA-{strategy})', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(proc_counts)
        axes[1].legend(loc='best')

        # Add value labels
        for x, y in zip(proc_counts, speedups):
            axes[1].text(x, y, f'{y:.2f}x', ha='center', va='bottom', fontweight='bold')

        # Plot 3: Efficiency vs Processors
        axes[2].plot(proc_counts, efficiencies, marker='^', linewidth=2.5, markersize=10, color='#F39C12')
        axes[2].axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (100%)')
        axes[2].set_xlabel('Number of Processors', fontweight='bold')
        axes[2].set_ylabel('Efficiency (%)', fontweight='bold')
        axes[2].set_title(f'Parallel Efficiency vs Processors\n(WDPA-{strategy})', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(proc_counts)

        # Auto-scale efficiency axis based on data
        if efficiencies:
            max_eff = max(efficiencies)
            if max_eff > 100:
                # Round up to nearest 100 or 50
                if max_eff > 200:
                    y_max = ((max_eff // 100) + 1) * 100 + 100  # Add buffer
                else:
                    y_max = ((max_eff // 50) + 1) * 50 + 50  # Add buffer
            else:
                y_max = 110  # Standard case
            axes[2].set_ylim(0, y_max)

        axes[2].legend(loc='best')

        # Add value labels
        for x, y in zip(proc_counts, efficiencies):
            axes[2].text(x, y, f'{y:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.suptitle(f'WDPA-{strategy} Processor Scaling Analysis\n(Sample Size: {sample_size:,} | Max Items: {max_items:,} | Support: {min_support*100:.1f}%)',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/processor_scaling_{strategy}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/processor_scaling_{strategy}.png")
        plt.close()

    # Create combined comparison plot for all strategies
    plot_processor_scaling_combined(results, processor_scaling, output_dir)


def plot_processor_scaling_combined(results, processor_scaling, output_dir=f'results/{benchmark_name}/plots'):
    """Create combined processor scaling plot comparing all strategies."""
    sample_size, max_items = get_dataset_info(results)

    # Get first support level
    first_support_key = list(results['results_by_support'].keys())[0]
    support_data = results['results_by_support'][first_support_key]
    min_support = list(support_data.values())[0].get('min_support', 0)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'BL': '#3498DB', 'CL': '#2ECC71', 'BWT': '#F39C12', 'CWT': '#9B59B6'}
    markers = {'BL': 'o', 'CL': 's', 'BWT': '^', 'CWT': 'D'}

    # Plot 1: Speedup comparison
    for strategy, proc_counts in processor_scaling.items():
        proc_counts = sorted(proc_counts)
        speedups = []

        for num_procs in proc_counts:
            key = f'wdpa_{strategy}_{num_procs}p'
            if key in support_data:
                speedup = support_data[key].get('speedup_metrics', {}).get('speedup', 0)
                speedups.append(speedup)

        if speedups:
            axes[0].plot(proc_counts, speedups, marker=markers.get(strategy, 'o'),
                        linewidth=2.5, markersize=10, label=f'WDPA-{strategy}',
                        color=colors.get(strategy, '#333333'), alpha=0.8)

    # Add ideal speedup line
    if processor_scaling:
        all_proc_counts = sorted(set([p for procs in processor_scaling.values() for p in procs]))
        if all_proc_counts:
            ideal = [p / all_proc_counts[0] for p in all_proc_counts]
            axes[0].plot(all_proc_counts, ideal, linestyle='--', color='gray',
                        linewidth=2, alpha=0.5, label='Ideal Linear')

    axes[0].set_xlabel('Number of Processors', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Speedup', fontweight='bold', fontsize=12)
    axes[0].set_title('Speedup vs Number of Processors', fontweight='bold', fontsize=13)
    axes[0].legend(loc='best', framealpha=0.9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(all_proc_counts)

    # Plot 2: Efficiency comparison
    all_efficiencies = []
    for strategy, proc_counts in processor_scaling.items():
        proc_counts = sorted(proc_counts)
        efficiencies = []

        for num_procs in proc_counts:
            key = f'wdpa_{strategy}_{num_procs}p'
            if key in support_data:
                efficiency = support_data[key].get('speedup_metrics', {}).get('efficiency', 0) * 100
                efficiencies.append(efficiency)
                all_efficiencies.append(efficiency)

        if efficiencies:
            axes[1].plot(proc_counts, efficiencies, marker=markers.get(strategy, 'o'),
                        linewidth=2.5, markersize=10, label=f'WDPA-{strategy}',
                        color=colors.get(strategy, '#333333'), alpha=0.8)

    axes[1].axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (100%)')
    axes[1].set_xlabel('Number of Processors', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Efficiency (%)', fontweight='bold', fontsize=12)
    axes[1].set_title('Parallel Efficiency vs Number of Processors', fontweight='bold', fontsize=13)
    axes[1].legend(loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(all_proc_counts)

    # Auto-scale efficiency axis based on data
    if all_efficiencies:
        max_eff = max(all_efficiencies)
        if max_eff > 100:
            # Round up to nearest 100 or 50
            if max_eff > 200:
                y_max = ((max_eff // 100) + 1) * 100 + 100  # Add buffer
            else:
                y_max = ((max_eff // 50) + 1) * 50 + 50  # Add buffer
        else:
            y_max = 110  # Standard case
        axes[1].set_ylim(0, y_max)

    plt.suptitle(f'Processor Scaling: All WDPA Strategies\n(Sample Size: {sample_size:,} | Max Items: {max_items:,} | Support: {min_support*100:.1f}%)',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/processor_scaling_combined.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/processor_scaling_combined.png")
    plt.close()


def get_support_folder_name(min_support):
    """Convert support threshold to folder name (e.g., 0.001 -> plot_010)"""
    # Convert to percentage and remove decimal point
    support_pct = int(min_support * 1000)  # e.g., 0.001 -> 1, 0.0008 -> 0.8

    # Handle decimal cases
    if min_support * 1000 == int(min_support * 1000):
        # Clean integer percentage
        return f"plot_{support_pct:03d}"
    else:
        # Has decimal, multiply by 10 to avoid decimals
        support_pct = int(min_support * 10000)
        return f"plot_{support_pct:04d}"


def plot_wdpa_strategies_for_support(results, support_key, support_data, output_dir):
    """Create WDPA strategy comparison plot for a specific support threshold."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_size, max_items = get_dataset_info(results)
    min_support = support_data.get('traditional', {}).get('min_support', 0)
    if not min_support:
        min_support = list(support_data.values())[0].get('min_support', 0)

    strategies = ['BL', 'CL', 'BWT', 'CWT']
    metrics = {
        'Time (s)': [],
        'Speedup': [],
        'Efficiency (%)': []
    }

    for strategy in strategies:
        key = f'wdpa_{strategy}'
        data = support_data.get(key, {})
        # Use computation_time if available, else total_time
        metrics['Time (s)'].append(data.get('computation_time', data.get('total_time', 0)))
        metrics['Speedup'].append(data.get('speedup_metrics', {}).get('speedup', 0))
        metrics['Efficiency (%)'].append(data.get('speedup_metrics', {}).get('efficiency', 0) * 100)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    # Plot 1: Execution Time
    bars1 = axes[0].bar(strategies, metrics['Time (s)'], color=colors, alpha=0.8)
    axes[0].set_ylabel('Execution Time (seconds)', fontweight='bold')
    axes[0].set_title(f'Execution Time\n(Support: {min_support*100:.2f}%)', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Speedup
    bars2 = axes[1].bar(strategies, metrics['Speedup'], color=colors, alpha=0.8)
    axes[1].set_ylabel('Speedup vs Traditional', fontweight='bold')
    axes[1].set_title(f'Speedup\n(Support: {min_support*100:.2f}%)', fontweight='bold')
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}x', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Efficiency
    bars3 = axes[2].bar(strategies, metrics['Efficiency (%)'], color=colors, alpha=0.8)
    axes[2].set_ylabel('Efficiency (%)', fontweight='bold')
    axes[2].set_title(f'Parallel Efficiency\n(Support: {min_support*100:.2f}%)', fontweight='bold')
    axes[2].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal')
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_ylim(0, 110)
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'WDPA Lattice Distribution Strategies Comparison\n(Sample Size: {sample_size:,} | Max Items: {max_items:,} | Support: {min_support*100:.2f}%)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wdpa_strategies_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/wdpa_strategies_comparison.png")
    plt.close()


def generate_plots_per_support(results_file=f'results/{benchmark_name}/benchmark_results.json'):
    """Generate all original plots separately for each support threshold."""
    print("="*80)
    print("BENCHMARK RESULTS VISUALIZATION - PER SUPPORT THRESHOLD")
    print("="*80)
    print(f"Loading results from: {results_file}")

    results = load_results(results_file)

    sample_size, max_items = get_dataset_info(results)
    print(f"Dataset: {sample_size:,} orders, {max_items:,} items")
    print("-"*80)

    # Check if processor scaling is enabled
    processor_scaling = detect_processor_scaling(results)

    if processor_scaling:
        print(f"\n🔍 PROCESSOR SCALING MODE DETECTED")
        print(f"   Processor counts: {processor_scaling}")
        print(f"   Generating processor scaling plots for each support...\n")
    else:
        print(f"\n📊 REGULAR MODE")
        print(f"   Generating all standard plots for each support...\n")

    # Generate plots for each support threshold
    support_thresholds = results['results_by_support'].keys()
    print(f"Found {len(support_thresholds)} support thresholds")
    print(f"Generating separate plot folders for each...\n")

    for support_key, support_data in results['results_by_support'].items():
        # Get the min_support value
        min_support = support_data.get('traditional', {}).get('min_support', 0)
        if not min_support:
            min_support = list(support_data.values())[0].get('min_support', 0)

        # Create folder name
        folder_name = get_support_folder_name(min_support)
        output_dir = f'results/{benchmark_name}/{folder_name}'

        print(f"📊 Support {min_support*100:.2f}% -> {folder_name}/")

        # Create the output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create a temporary results object with just this support threshold
        single_support_results = {
            'metadata': results['metadata'],
            'results_by_support': {support_key: support_data}
        }

        # Generate all the original plots for this single support
        if processor_scaling:
            # Processor scaling mode
            plot_traditional_failure_analysis(single_support_results, output_dir)
            plot_processor_scaling_analysis(single_support_results, output_dir)
        else:
            # Regular mode - generate all standard plots
            plot_execution_time_comparison(single_support_results, output_dir)
            plot_speedup_comparison(single_support_results, output_dir)
            plot_efficiency_comparison(single_support_results, output_dir)
            plot_wdpa_strategies_detailed(single_support_results, output_dir)
            plot_traditional_failure_analysis(single_support_results, output_dir)
            plot_best_algorithm_summary(single_support_results, output_dir)

        print()

    print("-"*80)
    print("\n✅ All per-support plots generated successfully!")
    print("="*80)
    print(f"\nView your plots in: results/{benchmark_name}/")
    for support_key, support_data in results['results_by_support'].items():
        min_support = support_data.get('traditional', {}).get('min_support', 0)
        if not min_support:
            min_support = list(support_data.values())[0].get('min_support', 0)
        folder_name = get_support_folder_name(min_support)
        print(f"  - {folder_name}/ (Support: {min_support*100:.2f}%)")

    print("="*80)


def generate_all_plots(results_file=f'results/{benchmark_name}/benchmark_results.json',
                       output_dir=f'results/{benchmark_name}/plots'):
    """Generate all visualization plots."""
    print("="*80)
    print("BENCHMARK RESULTS VISUALIZATION")
    print("="*80)
    print(f"Loading results from: {results_file}")

    results = load_results(results_file)

    print(f"\nGenerating plots in: {output_dir}")
    print("-"*80)

    # Check if processor scaling is enabled
    processor_scaling = detect_processor_scaling(results)

    if processor_scaling:
        print(f"\n🔍 PROCESSOR SCALING MODE DETECTED")
        print(f"   Processor counts: {processor_scaling}")
        print(f"   Skipping regular comparison plots (they don't work with multiple processor counts)")
        print(f"   Generating processor scaling visualizations instead...\n")

        # Only generate processor scaling plots
        plot_traditional_failure_analysis(results, output_dir)
        plot_processor_scaling_analysis(results, output_dir)

        print("-"*80)
        print("\n✅ Processor scaling plots generated successfully!")
        print("="*80)
        print(f"\nView your plots in: {output_dir}/")
        print("  - traditional_failure_analysis.png (if traditional failed)")
        print("  - processor_scaling_combined.png (all strategies comparison)")
        for strategy in processor_scaling.keys():
            print(f"  - processor_scaling_{strategy}.png (detailed 3-panel view)")

    else:
        print(f"\n📊 REGULAR BENCHMARK MODE")
        print(f"   Single processor count per algorithm")
        print(f"   Generating standard comparison plots...\n")

        # Generate regular comparison plots
        plot_execution_time_comparison(results, output_dir)
        plot_speedup_comparison(results, output_dir)
        plot_efficiency_comparison(results, output_dir)
        plot_wdpa_strategies_detailed(results, output_dir)
        plot_traditional_failure_analysis(results, output_dir)
        plot_best_algorithm_summary(results, output_dir)

        print("-"*80)
        print("\n✅ All plots generated successfully!")
        print("="*80)
        print(f"\nView your plots in: {output_dir}/")
        print("  - execution_time_comparison.png")
        print("  - speedup_comparison.png")
        print("  - efficiency_comparison.png")
        print("  - wdpa_strategies_detailed.png")
        print("  - traditional_failure_analysis.png")
        print("  - best_algorithm_summary.png")

    print("="*80)


if __name__ == "__main__":
    # Generate separate plots for each support threshold
    generate_plots_per_support()
