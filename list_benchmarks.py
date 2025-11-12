#!/usr/bin/env python3
"""
List all benchmark runs with details
"""

import os
import glob
import json
from datetime import datetime

def list_benchmarks():
    """List all available benchmark results."""
    benchmark_files = sorted(glob.glob('results/benchmark_*/benchmark_results.json'))

    if not benchmark_files:
        print("No benchmark results found in results/ directory")
        return

    print("="*100)
    print("AVAILABLE BENCHMARK RUNS")
    print("="*100)
    print(f"{'#':<4} {'Folder':<30} {'Date':<20} {'Sample':<10} {'Support':<20} {'Strategies'}")
    print("-"*100)

    for idx, filepath in enumerate(benchmark_files, 1):
        folder = os.path.basename(os.path.dirname(filepath))

        # Get modification time
        mtime = os.path.getmtime(filepath)
        date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

        # Load metadata
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            config = metadata.get('config', {})
            dataset = config.get('dataset', {})
            wdpa = config.get('algorithms', {}).get('wdpa', {})
            mining = config.get('mining_parameters', {})

            sample_size = dataset.get('sample_size', 'N/A')
            support_thresholds = mining.get('support_thresholds', [])
            support_str = ', '.join([f"{s*100:.3f}%" for s in support_thresholds])
            strategies = wdpa.get('strategies', [])
            strategies_str = ', '.join(strategies)
            num_procs = wdpa.get('num_processors', [])

            # Count results
            num_supports = len(data.get('results_by_support', {}))

            print(f"{idx:<4} {folder:<30} {date_str:<20} {sample_size:<10} {support_str:<20} {strategies_str}")

            # Show processor counts
            if isinstance(num_procs, list):
                print(f"     └─ Processors: {num_procs}")

        except Exception as e:
            print(f"{idx:<4} {folder:<30} {date_str:<20} ERROR: {str(e)}")

    print("-"*100)
    print(f"Total: {len(benchmark_files)} benchmark run(s)")
    print("="*100)
    print("\nTo visualize a specific benchmark:")
    print("  python visualize_results.py  (uses latest)")
    print("  # Or edit visualize_results.py and set: benchmark_name = 'benchmark_200k_001'")

if __name__ == '__main__':
    list_benchmarks()
