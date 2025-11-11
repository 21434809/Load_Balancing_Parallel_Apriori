#!/usr/bin/env python3
"""Quick test to verify pool reuse fix works"""

import sys
import json
from src.utils.unified_data_loader import load_data_for_benchmark
from src.core.wdpa_parallel import WDPAParallelMiner

# Small test configuration
config = {
    "sample_size": 5000,  # Small dataset
    "max_items": 1000,
    "chunk_size": 10000,
    "random_seed": 42
}

print("Loading small test dataset...")
data = load_data_for_benchmark(config, verbose=False)
tid = data['tid']

print(f"Dataset: {tid.total_transactions} transactions, {len(tid.item_to_tids)} items")

min_support = 0.01  # 1%
strategies = ['CWT']
processor_counts = [1, 2, 4]

results = {}

for strategy in strategies:
    for num_procs in processor_counts:
        print(f"\n{'='*60}")
        print(f"Testing WDPA-{strategy} with {num_procs} processor(s)")
        print(f"{'='*60}")

        miner = WDPAParallelMiner(
            tid=tid,
            min_support=min_support,
            num_processors=num_procs,
            strategy=strategy,
            max_k=3,  # Only up to 3-itemsets for quick test
            verbose=True
        )

        frequent = miner.mine()
        metrics = miner.get_metrics()

        key = f"{strategy}_{num_procs}p"
        results[key] = {
            'total_time': metrics['total_time'],
            'pure_computation_time': metrics['pure_computation_time'],
            'total_itemsets': sum(len(items) for items in frequent.values())
        }

        print(f"\nResults:")
        print(f"  Total time: {metrics['total_time']:.3f}s")
        print(f"  Pure computation: {metrics['pure_computation_time']:.3f}s")
        print(f"  Total itemsets: {results[key]['total_itemsets']}")

print(f"\n{'='*60}")
print("SPEEDUP ANALYSIS")
print(f"{'='*60}")

baseline_key = f"{strategies[0]}_1p"
baseline_time = results[baseline_key]['pure_computation_time']

print(f"\nBaseline (1p): {baseline_time:.3f}s\n")

for strategy in strategies:
    for num_procs in processor_counts:
        if num_procs == 1:
            continue

        key = f"{strategy}_{num_procs}p"
        comp_time = results[key]['pure_computation_time']
        speedup = baseline_time / comp_time if comp_time > 0 else 0
        efficiency = speedup / num_procs * 100

        print(f"{num_procs}p: {comp_time:.3f}s → Speedup: {speedup:.2f}x, Efficiency: {efficiency:.1f}%")

print(f"\n{'='*60}")
print("TEST COMPLETE!")
print(f"{'='*60}")

# Verify speedup is reasonable
cwt_4p_time = results['CWT_4p']['pure_computation_time']
cwt_1p_time = results['CWT_1p']['pure_computation_time']

if cwt_4p_time > cwt_1p_time:
    print(f"\n⚠️  WARNING: 4p is SLOWER than 1p!")
    print(f"   1p: {cwt_1p_time:.3f}s")
    print(f"   4p: {cwt_4p_time:.3f}s")
    print(f"   This suggests the pool reuse fix didn't work properly.")
else:
    speedup = cwt_1p_time / cwt_4p_time
    print(f"\n✅ SUCCESS: 4p is faster than 1p!")
    print(f"   1p: {cwt_1p_time:.3f}s")
    print(f"   4p: {cwt_4p_time:.3f}s")
    print(f"   Speedup: {speedup:.2f}x")
