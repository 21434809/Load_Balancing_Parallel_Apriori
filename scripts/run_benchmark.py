#!/usr/bin/env python3
"""
Unified Benchmark Script

Compares Traditional, Naive Parallel, and WDPA Apriori algorithms on the SAME dataset.
Generates comprehensive comparison results showing speedup and performance metrics.
"""

import sys
import os
# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

from src.utils.unified_data_loader import load_data_for_benchmark
from src.core.naive_parallel_apriori import run_naive_parallel_apriori
from wdpa_parallel import WDPAParallelMiner
from src.core.ye_parallel_apriori import run_ye_parallel_apriori


def load_config(config_file: str = 'configs/benchmark_config.json') -> Dict:
    """Load benchmark configuration."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def run_traditional_apriori(basket_encoded: pd.DataFrame, min_support: float, verbose: bool = True) -> Dict:
    """Run traditional single-threaded Apriori using mlxtend."""
    if verbose:
        print(f"\n{'='*80}")
        print("TRADITIONAL SINGLE-THREADED APRIORI")
        print(f"{'='*80}")
        print(f"Min support: {min_support*100:.2f}%")

    # Pre-check: Estimate memory usage for mlxtend
    # mlxtend creates arrays of size (n_items_choose_k) x n_transactions
    # For level 2: n_items * (n_items-1) / 2 combinations
    n_transactions = basket_encoded.shape[0]
    n_items = basket_encoded.shape[1]

    # Estimate level-2 memory (worst case)
    n_combinations_level2 = (n_items * (n_items - 1)) // 2
    estimated_memory_gb = (n_combinations_level2 * n_transactions * 1) / (1024**3)  # 1 byte per element

    # Skip if estimated memory > 8GB (conservative threshold)
    if estimated_memory_gb > 8.0:
        if verbose:
            print(f"SKIPPED: Dataset too large for traditional Apriori")
            print(f"   Estimated memory required: {estimated_memory_gb:.1f} GiB")
            print(f"   Dataset: {n_transactions:,} transactions x {n_items:,} items")
            print(f"   Traditional Apriori cannot handle this dataset size")
            print(f"{'='*80}")

        return {
            'method': 'Traditional Apriori',
            'min_support': min_support,
            'total_itemsets': 0,
            'total_time': 0,
            'status': 'skipped_memory_error',
            'memory_required': f"{estimated_memory_gb:.1f} GiB",
            'error_message': f"Could not compute - requires ~{estimated_memory_gb:.1f} GiB of RAM"
        }

    start_time = time.time()

    try:
        # Run Apriori
        frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)

        total_time = time.time() - start_time

        if verbose:
            print(f"Found {len(frequent_itemsets)} frequent itemsets")
            print(f"Total time: {total_time:.4f} seconds")
            print(f"{'='*80}")

        return {
            'method': 'Traditional Apriori',
            'min_support': min_support,
            'total_itemsets': len(frequent_itemsets),
            'total_time': total_time,
            'frequent_itemsets': frequent_itemsets,
            'status': 'success'
        }

    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        # Memory error - dataset too large for traditional Apriori
        error_msg = str(e)

        # Extract memory size from error message
        import re
        memory_match = re.search(r'(\d+\.?\d*)\s*(GiB|MiB|GB|MB)', error_msg)
        if memory_match:
            memory_needed = f"{memory_match.group(1)} {memory_match.group(2)}"
        else:
            memory_needed = "Unknown (very large)"

        if verbose:
            print(f"SKIPPED: Dataset too large for traditional Apriori")
            print(f"   Required memory: {memory_needed}")
            print(f"   Traditional Apriori cannot handle this dataset size")
            print(f"{'='*80}")

        return {
            'method': 'Traditional Apriori',
            'min_support': min_support,
            'total_itemsets': 0,
            'total_time': 0,
            'status': 'skipped_memory_error',
            'memory_required': memory_needed,
            'error_message': f"Could not compute - requires {memory_needed} of RAM"
        }


def run_naive_apriori(basket_encoded: pd.DataFrame, min_support: float, num_workers: int, verbose: bool = True) -> Dict:
    """Run naive parallel Apriori."""
    if verbose:
        print(f"\n{'='*80}")
        print("NAIVE PARALLEL APRIORI")
        print(f"{'='*80}")
        print(f"Min support: {min_support*100:.2f}%")
        print(f"Workers: {num_workers}")

    start_time = time.time()

    # Run Naive Parallel Apriori
    frequent_itemsets = run_naive_parallel_apriori(basket_encoded, min_support=min_support, num_workers=num_workers)

    total_time = time.time() - start_time

    if verbose:
        print(f"Found {len(frequent_itemsets)} frequent itemsets")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"{'='*80}")

    return {
        'method': 'Naive Parallel Apriori',
        'min_support': min_support,
        'num_workers': num_workers,
        'total_itemsets': len(frequent_itemsets),
        'total_time': total_time,
        'frequent_itemsets': frequent_itemsets
    }


def run_wdpa_strategy(tid, min_support: float, strategy: str, num_processors: int, max_k: int, verbose: bool = True) -> Dict:
    """Run WDPA with a specific distribution strategy."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"WDPA PARALLEL APRIORI - Strategy: {strategy}")
        print(f"{'='*80}")
        print(f"Min support: {min_support*100:.2f}%")
        print(f"Processors: {num_processors}")

    # Create miner
    miner = WDPAParallelMiner(
        tid=tid,
        min_support=min_support,
        num_processors=num_processors,
        strategy=strategy,
        max_k=max_k,
        verbose=verbose
    )

    # Run mining
    frequent_by_level = miner.mine()
    metrics = miner.get_metrics()

    # Count total itemsets
    total_itemsets = sum(len(itemsets) for itemsets in frequent_by_level.values())

    if verbose:
        print(f"\n{'='*80}")
        print(f"WDPA-{strategy} COMPLETE")
        print(f"Total itemsets: {total_itemsets}")
        print(f"Total time: {metrics['total_time']:.4f} seconds")
        print(f"{'='*80}")

    return {
        'method': f'WDPA-{strategy}',
        'strategy': strategy,
        'min_support': min_support,
        'num_processors': num_processors,
        'total_itemsets': total_itemsets,
        'total_time': metrics['total_time'],
        'level_times': metrics['level_times'],
        'itemsets_per_level': metrics['itemsets_per_level'],
        'frequent_by_level': frequent_by_level,
        'metrics': metrics
    }


def calculate_speedup_metrics(baseline_time: float, parallel_time: float, num_processors: int) -> Dict:
    """Calculate speedup metrics."""
    speedup = baseline_time / parallel_time if parallel_time > 0 else 0
    efficiency = speedup / num_processors if num_processors > 0 else 0
    time_saved = baseline_time - parallel_time
    percent_faster = ((baseline_time - parallel_time) / baseline_time * 100) if baseline_time > 0 else 0

    return {
        'speedup': speedup,
        'efficiency': efficiency,
        'time_saved': time_saved,
        'percent_faster': percent_faster,
        'baseline_time': baseline_time,
        'parallel_time': parallel_time,
        'num_processors': num_processors
    }


def run_full_benchmark(config: Dict, verbose: bool = True):
    """Run complete benchmark comparing all algorithms."""
    if verbose:
        print("\n" + "="*80)
        print("UNIFIED APRIORI BENCHMARK")
        print("="*80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Dataset: {config['dataset']['sample_size']:,} orders, {config['dataset']['max_items']:,} items")

    # Load data ONCE for all algorithms
    if verbose:
        print("\nLoading data for all algorithms...")
    data = load_data_for_benchmark(config['dataset'], algorithms_config=config['algorithms'], verbose=verbose)

    # Extract data
    basket_encoded = data['basket_encoded']
    tid = data['tid']
    transactions = data.get('transactions')
    metadata = data['metadata']

    # Results storage
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'dataset_info': metadata
        },
        'results_by_support': {}
    }

    # Run for each support threshold
    for min_support in config['mining_parameters']['support_thresholds']:
        if verbose:
            print(f"\n\n{'='*80}")
            print(f"TESTING WITH SUPPORT = {min_support*100:.2f}%")
            print(f"{'='*80}")

        support_results = {}

        # 1. Traditional Apriori (baseline)
        baseline_time = None
        if config['algorithms']['traditional']['enabled']:
            trad_results = run_traditional_apriori(basket_encoded, min_support, verbose=verbose)
            support_results['traditional'] = trad_results

            # Only use as baseline if it succeeded
            if trad_results.get('status') == 'success':
                baseline_time = trad_results['total_time']
            else:
                if verbose:
                    print(f"\nWARNING: Traditional Apriori skipped - parallel speedup will be relative to Naive Parallel")

        # 2. Naive Parallel Apriori
        if config['algorithms']['naive_parallel']['enabled']:
            num_workers = config['algorithms']['naive_parallel']['num_workers']
            naive_results = run_naive_apriori(basket_encoded, min_support, num_workers, verbose=verbose)
            support_results['naive_parallel'] = naive_results

            # Calculate speedup vs traditional (if available)
            if baseline_time:
                naive_results['speedup_metrics'] = calculate_speedup_metrics(
                    baseline_time, naive_results['total_time'], num_workers
                )

            # If no baseline, naive becomes the baseline for WDPA comparison
            if not baseline_time:
                baseline_time = naive_results['total_time']
                baseline_name = 'Naive Parallel'
            else:
                baseline_name = 'Traditional'

        # 3. WDPA with all strategies
        if config['algorithms']['wdpa']['enabled']:
            wdpa_config = config['algorithms']['wdpa']
            strategies = wdpa_config['strategies']
            num_processors = wdpa_config['num_processors']
            max_k = wdpa_config['max_k']

            for strategy in strategies:
                wdpa_results = run_wdpa_strategy(
                    tid, min_support, strategy, num_processors, max_k, verbose=verbose
                )
                support_results[f'wdpa_{strategy}'] = wdpa_results

                # Calculate speedup vs baseline (traditional or naive)
                if baseline_time:
                    wdpa_results['speedup_metrics'] = calculate_speedup_metrics(
                        baseline_time, wdpa_results['total_time'], num_processors
                    )
                    wdpa_results['speedup_baseline'] = baseline_name

        # 4. Ye 2006 parallel Apriori (Trie + rescans)
        if config['algorithms'].get('ye_parallel', {}).get('enabled', False):
            ye_conf = config['algorithms']['ye_parallel']
            ye_workers = ye_conf.get('num_workers', [1, 2, 4, 8, 16])
            ye_max_k = ye_conf.get('max_k', 5)

            # Ensure transactions available or derive from basket
            if transactions is None and basket_encoded is not None:
                arr_bool = basket_encoded.to_numpy(dtype=bool, copy=False)
                transactions = [list(row.nonzero()[0]) for row in arr_bool]
                for tx in transactions:
                    tx.sort()

            for workers in (ye_workers if isinstance(ye_workers, list) else [ye_workers]):
                ye_key = f'ye_parallel_{workers}p' if isinstance(ye_workers, list) else 'ye_parallel'
                start_t = time.time()
                ye_df = run_ye_parallel_apriori(transactions, min_support, num_workers=workers, max_k=ye_max_k)
                elapsed = time.time() - start_t
                support_results[ye_key] = {
                    'method': 'Ye Parallel Apriori',
                    'min_support': min_support,
                    'num_workers': workers,
                    'total_itemsets': len(ye_df),
                    'total_time': elapsed,
                    'frequent_itemsets': ye_df
                }

                if baseline_time:
                    support_results[ye_key]['speedup_metrics'] = calculate_speedup_metrics(
                        baseline_time, elapsed, workers
                    )
                    support_results[ye_key]['speedup_baseline'] = baseline_name

        results['results_by_support'][f'support_{min_support}'] = support_results

    # Save results in sample-size-specific folder
    sample_size = config['dataset']['sample_size']
    sample_size_k = sample_size // 1000  # Convert to K (e.g., 50000 -> 50k)
    base_output_dir = config['output']['results_directory']
    output_dir = f"{base_output_dir}_{sample_size_k}k"
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Saving results to: {output_dir}")
        print(f"{'='*80}")

    # Remove non-serializable objects for JSON
    results_to_save = json.loads(json.dumps(results, default=str))

    # Save complete results
    results_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    if verbose:
        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETE!")
        print(f"Results saved to: {results_file}")
        print(f"{'='*80}")

    # Generate summary
    generate_summary(results, config, verbose=verbose)

    return results


def generate_summary(results: Dict, config: Dict, verbose: bool = True):
    """Generate human-readable summary of benchmark results."""
    output_dir = config['output']['results_directory']
    summary_file = os.path.join(output_dir, 'benchmark_summary.txt')

    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("APRIORI BENCHMARK SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
        f.write(f"Dataset: {config['dataset']['sample_size']:,} orders, {config['dataset']['max_items']:,} items\n")
        f.write("\n")

        for support_key, support_results in results['results_by_support'].items():
            min_support = support_results.get('traditional', {}).get('min_support') or \
                         support_results.get('naive_parallel', {}).get('min_support') or \
                         list(support_results.values())[0].get('min_support', 0)

            f.write(f"\n{'='*80}\n")
            f.write(f"SUPPORT THRESHOLD: {min_support*100:.2f}%\n")
            f.write(f"{'='*80}\n\n")

            # Get baseline time
            baseline_time = support_results.get('traditional', {}).get('total_time', 0)

            f.write(f"{'Algorithm':<30} {'Time (s)':>12} {'Itemsets':>10} {'Speedup':>10} {'Efficiency':>10}\n")
            f.write("-"*80 + "\n")

            # Traditional
            if 'traditional' in support_results:
                trad = support_results['traditional']
                if trad.get('status') == 'skipped_memory_error':
                    memory_req = trad.get('memory_required', 'N/A')
                    f.write(f"{'Traditional Apriori':<30} {'SKIPPED':>12} {'-':>10} {'-':>10} {'-':>10}\n")
                    f.write(f"{'  (Memory required: ' + memory_req + ')':<30}\n")
                else:
                    f.write(f"{'Traditional Apriori':<30} {trad['total_time']:>12.4f} {trad['total_itemsets']:>10} {'-':>10} {'-':>10}\n")

            # Naive
            if 'naive_parallel' in support_results:
                naive = support_results['naive_parallel']
                speedup = naive.get('speedup_metrics', {}).get('speedup', 0)
                efficiency = naive.get('speedup_metrics', {}).get('efficiency', 0)
                f.write(f"{'Naive Parallel':<30} {naive['total_time']:>12.4f} {naive['total_itemsets']:>10} {speedup:>10.2f} {efficiency:>10.2%}\n")

            # WDPA strategies
            for strategy in ['BL', 'CL', 'BWT', 'CWT']:
                key = f'wdpa_{strategy}'
                if key in support_results:
                    wdpa = support_results[key]
                    speedup = wdpa.get('speedup_metrics', {}).get('speedup', 0)
                    efficiency = wdpa.get('speedup_metrics', {}).get('efficiency', 0)
                    name = f'WDPA-{strategy}'
                    f.write(f"{name:<30} {wdpa['total_time']:>12.4f} {wdpa['total_itemsets']:>10} {speedup:>10.2f} {efficiency:>10.2%}\n")

            f.write("\n")

        f.write(f"\n{'='*80}\n")
        f.write("Notes:\n")
        f.write("- Speedup = Baseline Time / Algorithm Time\n")
        f.write("- Efficiency = Speedup / Number of Processors\n")
        f.write("- All algorithms use the EXACT SAME dataset\n")
        f.write(f"{'='*80}\n")

    if verbose:
        print(f"Summary saved to: {summary_file}")


def main():
    """Main entry point."""
    # Load configuration
    config = load_config('configs/benchmark_config.json')

    # Run benchmark
    results = run_full_benchmark(config, verbose=True)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results directory: {config['output']['results_directory']}")
    print(f"- benchmark_results.json (complete results)")
    print(f"- benchmark_summary.txt (human-readable summary)")
    print("="*80)


if __name__ == "__main__":
    main()
