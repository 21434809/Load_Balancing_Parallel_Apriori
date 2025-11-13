#!/usr/bin/env python3
"""
Enhanced Visualization Script for Apriori Benchmark Results - Publication Quality

Generates publication-ready graphs with configurable filtering and advanced analytics.

Configuration:
- PROCESSORS_TO_SHOW: Filter which processor counts to visualize
- SUPPORT_THRESHOLDS_TO_SHOW: Filter which support thresholds to visualize
- PUBLICATION_MODE: Enable high-quality publication settings
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import List, Dict, Optional
from scipy import stats

# ============================================================================
# CONFIGURATION SECTION - CUSTOMIZE YOUR VISUALIZATIONS HERE
# ============================================================================

# Filter processor counts to show (None = show all available)
# Example: [1, 4, 8] to only show single, 4, and 8 processor results
PROCESSORS_TO_SHOW: Optional[List[int]] = None  # None means show all
# PROCESSORS_TO_SHOW = [1, 4, 8]  # Uncomment to filter

# Filter support thresholds to show (None = show all available)
# Example: [0.0008, 0.001, 0.002] to only show specific thresholds
SUPPORT_THRESHOLDS_TO_SHOW: Optional[List[float]] = None  # None means show all
# SUPPORT_THRESHOLDS_TO_SHOW = [0.0008, 0.001, 0.002]  # Uncomment to filter

# Publication mode settings
PUBLICATION_MODE = True  # Set to True for high-quality publication graphs
DPI = 600 if PUBLICATION_MODE else 300  # 600 DPI for publication, 300 for preview
FONT_FAMILY = 'serif'  # 'serif' for academic papers, 'sans-serif' for modern look

# Auto-detect latest benchmark or specify manually
AUTO_DETECT_BENCHMARK = False
MANUAL_BENCHMARK_NAME = 'benchmark_150k_001'

# ============================================================================

# Publication-quality style settings
if PUBLICATION_MODE:
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.4)
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
    })
else:
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

# Enhanced color palette (colorblind-friendly)
COLORS = {
    'Traditional': '#2C3E50',      # Dark blue-gray
    'Naive Parallel': '#E74C3C',   # Red
    'WDPA-BL': '#3498DB',          # Blue
    'WDPA-CL': '#2ECC71',          # Green
    'WDPA-BWT': '#F39C12',         # Orange
    'WDPA-CWT': '#9B59B6',         # Purple
    'Ideal': '#95A5A6',            # Gray for reference lines
}

MARKERS = {
    'Traditional': 'o',
    'Naive Parallel': 's',
    'WDPA-BL': '^',
    'WDPA-CL': 'D',
    'WDPA-BWT': 'v',
    'WDPA-CWT': '*',
}


def get_latest_benchmark():
    """Find the most recent benchmark results directory."""
    import glob
    benchmark_dirs = glob.glob('results/benchmark_*/benchmark_results.json')
    if not benchmark_dirs:
        raise FileNotFoundError("No benchmark results found in results/ directory")
    latest = max(benchmark_dirs, key=lambda x: os.path.getmtime(x))
    benchmark_name = os.path.basename(os.path.dirname(latest))
    print(f"Auto-detected latest benchmark: {benchmark_name}")
    return benchmark_name


# Determine benchmark name
if AUTO_DETECT_BENCHMARK:
    benchmark_name = get_latest_benchmark()
else:
    benchmark_name = MANUAL_BENCHMARK_NAME
    print(f"Using manually specified benchmark: {benchmark_name}")


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


def filter_results_by_config(results):
    """Filter results based on PROCESSORS_TO_SHOW and SUPPORT_THRESHOLDS_TO_SHOW."""
    if PROCESSORS_TO_SHOW is None and SUPPORT_THRESHOLDS_TO_SHOW is None:
        return results  # No filtering needed

    filtered_results = {
        'metadata': results['metadata'],
        'results_by_support': {}
    }

    for support_key, support_data in results['results_by_support'].items():
        # Check support threshold filter
        min_support = support_data.get('traditional', {}).get('min_support', 0)
        if not min_support:
            min_support = list(support_data.values())[0].get('min_support', 0)

        if SUPPORT_THRESHOLDS_TO_SHOW is not None:
            if not any(abs(min_support - threshold) < 0.00001 for threshold in SUPPORT_THRESHOLDS_TO_SHOW):
                continue  # Skip this support threshold

        # Filter processor counts within this support threshold
        filtered_support_data = {}
        for algo_key, algo_data in support_data.items():
            # Check if this is a processor-specific result (e.g., wdpa_BL_4p)
            if '_' in algo_key and algo_key.split('_')[-1].endswith('p'):
                parts = algo_key.split('_')
                num_procs = int(parts[-1][:-1])  # Extract processor count

                if PROCESSORS_TO_SHOW is not None and num_procs not in PROCESSORS_TO_SHOW:
                    continue  # Skip this processor count

            filtered_support_data[algo_key] = algo_data

        if filtered_support_data:
            filtered_results['results_by_support'][support_key] = filtered_support_data

    print(f"\n📊 Filtering Configuration:")
    print(f"   Processors: {PROCESSORS_TO_SHOW if PROCESSORS_TO_SHOW else 'All available'}")
    print(f"   Support Thresholds: {SUPPORT_THRESHOLDS_TO_SHOW if SUPPORT_THRESHOLDS_TO_SHOW else 'All available'}")
    print(f"   Result: {len(filtered_results['results_by_support'])} support levels after filtering")

    return filtered_results


def calculate_load_imbalance_metrics(results, output_dir):
    """
    NEW ANALYSIS: Calculate and visualize load imbalance across strategies.
    This shows how evenly work is distributed across processors.
    """
    print("\n📊 Calculating load imbalance metrics...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sample_size, max_items = get_dataset_info(results)

    # Get first support level for detailed analysis
    first_support_key = list(results['results_by_support'].keys())[0]
    support_data = results['results_by_support'][first_support_key]
    min_support = list(support_data.values())[0].get('min_support', 0)

    strategies = ['BL', 'CL', 'BWT', 'CWT']
    imbalance_data = {
        'Max/Min Ratio': [],
        'Std Deviation': [],
        'Coefficient of Variation': []
    }

    for strategy in strategies:
        key = f'wdpa_{strategy}'
        data = support_data.get(key, {})

        # Get per-processor timing breakdown if available
        level_times = data.get('level_computation_times', {})
        if level_times:
            times = list(level_times.values())
            if len(times) > 1:
                max_time = max(times)
                min_time = min(times) if min(times) > 0 else max_time
                std_dev = np.std(times)
                mean_time = np.mean(times)
                cv = (std_dev / mean_time * 100) if mean_time > 0 else 0

                imbalance_data['Max/Min Ratio'].append(max_time / min_time if min_time > 0 else 1)
                imbalance_data['Std Deviation'].append(std_dev)
                imbalance_data['Coefficient of Variation'].append(cv)
            else:
                imbalance_data['Max/Min Ratio'].append(1.0)
                imbalance_data['Std Deviation'].append(0.0)
                imbalance_data['Coefficient of Variation'].append(0.0)
        else:
            # No detailed timing data available
            imbalance_data['Max/Min Ratio'].append(1.0)
            imbalance_data['Std Deviation'].append(0.0)
            imbalance_data['Coefficient of Variation'].append(0.0)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    x = np.arange(len(strategies))

    # Plot 1: Max/Min Ratio (ideal = 1.0)
    bars1 = axes[0].bar(x, imbalance_data['Max/Min Ratio'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Balance')
    axes[0].set_ylabel('Max/Min Load Ratio', fontweight='bold')
    axes[0].set_xlabel('Strategy', fontweight='bold')
    axes[0].set_title('Load Imbalance: Max/Min Ratio\n(Lower is Better)', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, imbalance_data['Max/Min Ratio']):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Coefficient of Variation (ideal = 0%)
    bars2 = axes[1].bar(x, imbalance_data['Coefficient of Variation'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Balance')
    axes[1].set_ylabel('Coefficient of Variation (%)', fontweight='bold')
    axes[1].set_xlabel('Strategy', fontweight='bold')
    axes[1].set_title('Load Variability: CV\n(Lower is Better)', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, imbalance_data['Coefficient of Variation']):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'Load Imbalance Analysis Across WDPA Strategies\nDataset: {sample_size:,} transactions, {max_items:,} items | Support: {min_support*100:.2f}%',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/load_imbalance_analysis.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/load_imbalance_analysis.png")
    plt.close()


def plot_overhead_analysis(results, output_dir):
    """
    NEW ANALYSIS: Visualize overhead (total time - computation time) across algorithms.
    Shows how much time is wasted on process management, data transfer, etc.
    """
    print("\n📊 Analyzing parallel overhead...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sample_size, max_items = get_dataset_info(results)

    support_levels = []
    overhead_data = {
        'Naive Parallel': [],
        'WDPA-BL': [],
        'WDPA-CL': [],
        'WDPA-BWT': [],
        'WDPA-CWT': []
    }

    for support_key, support_data in results['results_by_support'].items():
        min_support = list(support_data.values())[0].get('min_support', 0)
        support_levels.append(f"{min_support*100:.2f}%")

        for algo, key in [('Naive Parallel', 'naive_parallel'),
                          ('WDPA-BL', 'wdpa_BL'),
                          ('WDPA-CL', 'wdpa_CL'),
                          ('WDPA-BWT', 'wdpa_BWT'),
                          ('WDPA-CWT', 'wdpa_CWT')]:
            data = support_data.get(key, {})
            total_time = data.get('total_time', 0)
            comp_time = data.get('computation_time', total_time)
            overhead = total_time - comp_time
            overhead_pct = (overhead / total_time * 100) if total_time > 0 else 0
            overhead_data[algo].append(overhead_pct)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(support_levels))
    width = 0.15

    for i, (algo, overheads) in enumerate(overhead_data.items()):
        offset = (i - 2) * width
        color = COLORS.get(algo, '#333333')
        bars = ax.bar(x + offset, overheads, width, label=algo, color=color, alpha=0.8, edgecolor='black', linewidth=0.8)

        # Add value labels for significant overhead
        for bar, val in zip(bars, overheads):
            if val > 5:  # Only label if overhead > 5%
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Support Threshold', fontweight='bold', fontsize=12)
    ax.set_ylabel('Overhead (% of Total Time)', fontweight='bold', fontsize=12)
    ax.set_title('Parallel Overhead Analysis: Time Lost to Process Management\nDataset: {:,} transactions, {:,} items'.format(sample_size, max_items),
                fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max([max(v) for v in overhead_data.values()]) * 1.15)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/overhead_analysis.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/overhead_analysis.png")
    plt.close()


def plot_strong_scaling_efficiency(results, output_dir):
    """
    NEW ANALYSIS: Strong scaling efficiency - how well performance improves with more processors.
    Includes Amdahl's Law theoretical limit for comparison.
    """
    print("\n📊 Analyzing strong scaling efficiency...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sample_size, max_items = get_dataset_info(results)

    # Detect processor scaling
    first_support_key = list(results['results_by_support'].keys())[0]
    support_data = results['results_by_support'][first_support_key]

    processor_scaling = {}
    for result_key in support_data.keys():
        if result_key.startswith('wdpa_'):
            parts = result_key.split('_')
            if len(parts) >= 3 and parts[-1].endswith('p'):
                strategy = parts[1]
                num_procs = int(parts[-1][:-1])

                if strategy not in processor_scaling:
                    processor_scaling[strategy] = []
                processor_scaling[strategy].append(num_procs)

    if not processor_scaling:
        print("   ⚠ No processor scaling detected - skipping strong scaling analysis")
        return

    min_support = list(support_data.values())[0].get('min_support', 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors_map = {'BL': '#3498DB', 'CL': '#2ECC71', 'BWT': '#F39C12', 'CWT': '#9B59B6'}
    markers_map = {'BL': 'o', 'CL': 's', 'BWT': '^', 'CWT': '*'}

    # Plot 1: Speedup with Amdahl's Law
    for strategy, proc_counts in processor_scaling.items():
        proc_counts = sorted(proc_counts)
        speedups = []

        for num_procs in proc_counts:
            key = f'wdpa_{strategy}_{num_procs}p'
            if key in support_data:
                speedup = support_data[key].get('speedup_metrics', {}).get('speedup', 0)
                speedups.append(speedup)

        if speedups:
            axes[0].plot(proc_counts, speedups, marker=markers_map.get(strategy, 'o'),
                        linewidth=2.5, markersize=10, label=f'WDPA-{strategy}',
                        color=colors_map.get(strategy, '#333333'), alpha=0.85)

    # Add ideal linear speedup
    all_procs = sorted(set([p for procs in processor_scaling.values() for p in procs]))
    if all_procs:
        ideal = list(range(1, max(all_procs) + 1))
        axes[0].plot(ideal, ideal, linestyle='--', color='black', linewidth=2.5,
                    alpha=0.6, label='Ideal Linear Speedup')

        # Add Amdahl's Law curves for different parallel fractions
        for f, label in [(0.9, "Amdahl's Law (90% parallel)"), (0.95, "Amdahl's Law (95% parallel)")]:
            amdahl = [1 / ((1 - f) + f/p) for p in ideal]
            axes[0].plot(ideal, amdahl, linestyle=':', linewidth=2, alpha=0.6, label=label)

    axes[0].set_xlabel('Number of Processors', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Speedup', fontweight='bold', fontsize=12)
    axes[0].set_title('Strong Scaling: Speedup vs Processors\n(with Amdahl\'s Law Limits)', fontweight='bold')
    axes[0].legend(loc='best', framealpha=0.95, edgecolor='black', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(all_procs)

    # Plot 2: Parallel Efficiency
    for strategy, proc_counts in processor_scaling.items():
        proc_counts = sorted(proc_counts)
        efficiencies = []

        for num_procs in proc_counts:
            key = f'wdpa_{strategy}_{num_procs}p'
            if key in support_data:
                efficiency = support_data[key].get('speedup_metrics', {}).get('efficiency', 0) * 100
                efficiencies.append(efficiency)

        if efficiencies:
            axes[1].plot(proc_counts, efficiencies, marker=markers_map.get(strategy, 'o'),
                        linewidth=2.5, markersize=10, label=f'WDPA-{strategy}',
                        color=colors_map.get(strategy, '#333333'), alpha=0.85)

    axes[1].axhline(y=100, color='black', linestyle='--', linewidth=2.5, alpha=0.6, label='Ideal (100%)')
    axes[1].set_xlabel('Number of Processors', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Parallel Efficiency (%)', fontweight='bold', fontsize=12)
    axes[1].set_title('Strong Scaling: Efficiency vs Processors\n(Processor Utilization)', fontweight='bold')
    axes[1].legend(loc='best', framealpha=0.95, edgecolor='black', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(all_procs)

    # Auto-scale y-axis
    all_effs = []
    for strategy, proc_counts in processor_scaling.items():
        for num_procs in sorted(proc_counts):
            key = f'wdpa_{strategy}_{num_procs}p'
            if key in support_data:
                all_effs.append(support_data[key].get('speedup_metrics', {}).get('efficiency', 0) * 100)
    if all_effs:
        max_eff = max(all_effs)
        axes[1].set_ylim(0, max(110, max_eff * 1.1))

    plt.suptitle(f'Strong Scaling Analysis\nDataset: {sample_size:,} transactions, {max_items:,} items | Support: {min_support*100:.2f}%',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/strong_scaling_analysis.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/strong_scaling_analysis.png")
    plt.close()


def plot_cost_benefit_analysis(results, output_dir):
    """
    NEW ANALYSIS: Cost-benefit analysis showing speedup per processor.
    Helps determine the optimal number of processors.
    """
    print("\n📊 Performing cost-benefit analysis...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sample_size, max_items = get_dataset_info(results)

    # Detect processor scaling
    first_support_key = list(results['results_by_support'].keys())[0]
    support_data = results['results_by_support'][first_support_key]

    processor_scaling = {}
    for result_key in support_data.keys():
        if result_key.startswith('wdpa_'):
            parts = result_key.split('_')
            if len(parts) >= 3 and parts[-1].endswith('p'):
                strategy = parts[1]
                num_procs = int(parts[-1][:-1])

                if strategy not in processor_scaling:
                    processor_scaling[strategy] = []
                processor_scaling[strategy].append(num_procs)

    if not processor_scaling:
        print("   ⚠ No processor scaling detected - skipping cost-benefit analysis")
        return

    min_support = list(support_data.values())[0].get('min_support', 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors_map = {'BL': '#3498DB', 'CL': '#2ECC71', 'BWT': '#F39C12', 'CWT': '#9B59B6'}
    markers_map = {'BL': 'o', 'CL': 's', 'BWT': '^', 'CWT': '*'}

    # Plot 1: Speedup per Processor (benefit per resource unit)
    for strategy, proc_counts in processor_scaling.items():
        proc_counts = sorted(proc_counts)
        speedup_per_proc = []

        for num_procs in proc_counts:
            key = f'wdpa_{strategy}_{num_procs}p'
            if key in support_data:
                speedup = support_data[key].get('speedup_metrics', {}).get('speedup', 0)
                spp = speedup / num_procs if num_procs > 0 else 0
                speedup_per_proc.append(spp)

        if speedup_per_proc:
            axes[0].plot(proc_counts, speedup_per_proc, marker=markers_map.get(strategy, 'o'),
                        linewidth=2.5, markersize=10, label=f'WDPA-{strategy}',
                        color=colors_map.get(strategy, '#333333'), alpha=0.85)

            # Annotate values
            for x, y in zip(proc_counts, speedup_per_proc):
                axes[0].text(x, y * 1.02, f'{y:.2f}', ha='center', va='bottom', fontsize=8)

    axes[0].axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.6, label='Break-even (1.0)')
    axes[0].set_xlabel('Number of Processors', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Speedup / Processor Count', fontweight='bold', fontsize=12)
    axes[0].set_title('Cost-Benefit: Speedup per Processor\n(Higher = Better ROI)', fontweight='bold')
    axes[0].legend(loc='best', framealpha=0.95, edgecolor='black')
    axes[0].grid(True, alpha=0.3)
    all_procs = sorted(set([p for procs in processor_scaling.values() for p in procs]))
    axes[0].set_xticks(all_procs)

    # Plot 2: Marginal Speedup (incremental benefit)
    for strategy, proc_counts in processor_scaling.items():
        proc_counts = sorted(proc_counts)
        marginal_speedups = []
        prev_speedup = 0

        for i, num_procs in enumerate(proc_counts):
            key = f'wdpa_{strategy}_{num_procs}p'
            if key in support_data:
                speedup = support_data[key].get('speedup_metrics', {}).get('speedup', 0)
                marginal = speedup - prev_speedup if i > 0 else speedup
                marginal_speedups.append(marginal)
                prev_speedup = speedup

        if marginal_speedups:
            axes[1].plot(proc_counts, marginal_speedups, marker=markers_map.get(strategy, 'o'),
                        linewidth=2.5, markersize=10, label=f'WDPA-{strategy}',
                        color=colors_map.get(strategy, '#333333'), alpha=0.85)

            # Annotate values
            for x, y in zip(proc_counts, marginal_speedups):
                axes[1].text(x, y * 1.02, f'{y:.2f}', ha='center', va='bottom', fontsize=8)

    axes[1].set_xlabel('Number of Processors', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Marginal Speedup Gain', fontweight='bold', fontsize=12)
    axes[1].set_title('Marginal Returns: Speedup Gained per Added Processor\n(Diminishing Returns Indicator)', fontweight='bold')
    axes[1].legend(loc='best', framealpha=0.95, edgecolor='black')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(all_procs)

    plt.suptitle(f'Cost-Benefit Analysis: Optimal Processor Count\nDataset: {sample_size:,} transactions, {max_items:,} items | Support: {min_support*100:.2f}%',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cost_benefit_analysis.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/cost_benefit_analysis.png")
    plt.close()


def plot_execution_time_comparison(results, output_dir):
    """Enhanced execution time comparison with publication-quality styling."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sample_size, max_items = get_dataset_info(results)

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
        min_support = support_data.get('traditional', {}).get('min_support', 0)
        if not min_support:
            min_support = list(support_data.values())[0].get('min_support', 0)

        support_levels.append(f"{min_support*100:.2f}%")

        trad_data = support_data.get('traditional', {})
        trad_time = trad_data.get('total_time', 0)
        trad_status = trad_data.get('status', 'success')

        if trad_status != 'success' or trad_time == 0:
            traditional_failed = True
            algorithms['Traditional'].append(0)
        else:
            algorithms['Traditional'].append(trad_time)

        algorithms['Naive Parallel'].append(support_data.get('naive_parallel', {}).get('total_time', 0))
        algorithms['WDPA-BL'].append(support_data.get('wdpa_BL', {}).get('computation_time', support_data.get('wdpa_BL', {}).get('total_time', 0)))
        algorithms['WDPA-CL'].append(support_data.get('wdpa_CL', {}).get('computation_time', support_data.get('wdpa_CL', {}).get('total_time', 0)))
        algorithms['WDPA-BWT'].append(support_data.get('wdpa_BWT', {}).get('computation_time', support_data.get('wdpa_BWT', {}).get('total_time', 0)))
        algorithms['WDPA-CWT'].append(support_data.get('wdpa_CWT', {}).get('computation_time', support_data.get('wdpa_CWT', {}).get('total_time', 0)))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(support_levels))
    width = 0.12

    for i, (algo, times) in enumerate(algorithms.items()):
        offset = (i - 2.5) * width
        color = COLORS.get(algo, '#333333')
        bars = ax.bar(x + offset, times, width, label=algo, color=color, alpha=0.85,
                     edgecolor='black', linewidth=0.8)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=7, rotation=0)

    if traditional_failed:
        ax.text(0.5, 0.98, 'Note: Traditional algorithm failed due to memory constraints',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='black'),
               fontsize=9, style='italic')

    ax.set_xlabel('Support Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Execution Time Comparison Across Algorithms\nDataset: {sample_size:,} transactions | {max_items:,} items',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/execution_time_comparison.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/execution_time_comparison.png")
    plt.close()


def plot_speedup_comparison(results, output_dir):
    """Enhanced speedup comparison with better styling."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sample_size, max_items = get_dataset_info(results)

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

        support_levels.append(f"{min_support*100:.2f}%")

        speedups['Naive Parallel'].append(support_data.get('naive_parallel', {}).get('speedup_metrics', {}).get('speedup', 0))
        speedups['WDPA-BL'].append(support_data.get('wdpa_BL', {}).get('speedup_metrics', {}).get('speedup', 0))
        speedups['WDPA-CL'].append(support_data.get('wdpa_CL', {}).get('speedup_metrics', {}).get('speedup', 0))
        speedups['WDPA-BWT'].append(support_data.get('wdpa_BWT', {}).get('speedup_metrics', {}).get('speedup', 0))
        speedups['WDPA-CWT'].append(support_data.get('wdpa_CWT', {}).get('speedup_metrics', {}).get('speedup', 0))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(support_levels))

    for algo, values in speedups.items():
        marker = MARKERS.get(algo, 'o')
        color = COLORS.get(algo, '#333333')
        ax.plot(x, values, marker=marker, linewidth=2.5, markersize=9,
                label=algo, color=color, alpha=0.85)

        for i, v in enumerate(values):
            if v > 0:
                ax.text(i, v + 0.08, f'{v:.2f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(y=1.0, color=COLORS['Ideal'], linestyle='--', linewidth=2.5, alpha=0.7, label='Baseline (1×)')

    ax.set_xlabel('Support Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup vs Traditional Apriori', fontsize=12, fontweight='bold')
    ax.set_title(f'Speedup Comparison: Parallel Algorithms vs Traditional\nDataset: {sample_size:,} transactions | {max_items:,} items',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_comparison.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/speedup_comparison.png")
    plt.close()


def plot_efficiency_comparison(results, output_dir):
    """Enhanced efficiency comparison."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sample_size, max_items = get_dataset_info(results)

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

        support_levels.append(f"{min_support*100:.2f}%")

        efficiencies['Naive Parallel'].append(support_data.get('naive_parallel', {}).get('speedup_metrics', {}).get('efficiency', 0) * 100)
        efficiencies['WDPA-BL'].append(support_data.get('wdpa_BL', {}).get('speedup_metrics', {}).get('efficiency', 0) * 100)
        efficiencies['WDPA-CL'].append(support_data.get('wdpa_CL', {}).get('speedup_metrics', {}).get('efficiency', 0) * 100)
        efficiencies['WDPA-BWT'].append(support_data.get('wdpa_BWT', {}).get('speedup_metrics', {}).get('efficiency', 0) * 100)
        efficiencies['WDPA-CWT'].append(support_data.get('wdpa_CWT', {}).get('speedup_metrics', {}).get('efficiency', 0) * 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(support_levels))
    width = 0.15

    for i, (algo, values) in enumerate(efficiencies.items()):
        offset = (i - 2) * width
        color = COLORS.get(algo, '#333333')
        bars = ax.bar(x + offset, values, width, label=algo, color=color, alpha=0.85,
                     edgecolor='black', linewidth=0.8)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}%', ha='center', va='bottom', fontsize=7)

    ax.axhline(y=100, color=COLORS['Ideal'], linestyle='--', linewidth=2.5, alpha=0.7, label='Ideal (100%)')

    ax.set_xlabel('Support Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Parallel Efficiency: Speedup / Number of Processors\nDataset: {sample_size:,} transactions | {max_items:,} items',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(support_levels)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)

    # Auto-scale y-axis
    all_vals = [v for vals in efficiencies.values() for v in vals]
    if all_vals:
        max_eff = max(all_vals)
        ax.set_ylim(0, max(110, max_eff * 1.1))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/efficiency_comparison.png', dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/efficiency_comparison.png")
    plt.close()


def generate_publication_plots(results_file=f'results/{benchmark_name}/benchmark_results.json',
                                output_dir=f'results/{benchmark_name}/publication_plots'):
    """Generate all enhanced publication-quality plots."""
    print("="*80)
    print("ENHANCED PUBLICATION-QUALITY VISUALIZATION")
    print("="*80)
    print(f"Loading results from: {results_file}")
    print(f"Publication Mode: {'ENABLED' if PUBLICATION_MODE else 'DISABLED'}")
    print(f"DPI: {DPI}")

    results = load_results(results_file)

    # Apply filtering
    results = filter_results_by_config(results)

    print(f"\nGenerating plots in: {output_dir}")
    print("-"*80)

    # Generate all visualizations
    print("\n📊 Standard Comparisons:")
    plot_execution_time_comparison(results, output_dir)
    plot_speedup_comparison(results, output_dir)
    plot_efficiency_comparison(results, output_dir)

    print("\n📊 Advanced Analytics:")
    plot_overhead_analysis(results, output_dir)
    calculate_load_imbalance_metrics(results, output_dir)
    plot_strong_scaling_efficiency(results, output_dir)
    plot_cost_benefit_analysis(results, output_dir)

    print("\n" + "="*80)
    print("✅ All publication-quality plots generated successfully!")
    print("="*80)
    print(f"\nView your plots in: {output_dir}/")
    print("\nStandard Comparisons:")
    print("  - execution_time_comparison.png")
    print("  - speedup_comparison.png")
    print("  - efficiency_comparison.png")
    print("\nAdvanced Analytics:")
    print("  - overhead_analysis.png (NEW)")
    print("  - load_imbalance_analysis.png (NEW)")
    print("  - strong_scaling_analysis.png (NEW)")
    print("  - cost_benefit_analysis.png (NEW)")
    print("="*80)


if __name__ == "__main__":
    generate_publication_plots()
