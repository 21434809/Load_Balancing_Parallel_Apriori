"""
WDPA Performance Diagnostic Tool

This script will help you identify WHY your WDPA gets slower with more processors.
"""

import time
import pandas as pd
import numpy as np
from multiprocessing import Pool
import json


def diagnose_overhead_sources(config_file='configs/benchmark_config.json'):
    """Diagnose what's causing the slowdown."""
    
    print("="*80)
    print("WDPA PERFORMANCE DIAGNOSTIC")
    print("="*80)
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    sample_size = config['dataset']['sample_size']
    max_items = config['dataset']['max_items']
    
    print(f"\nDataset: {sample_size:,} orders, {max_items:,} items")
    
    # Load TID (you'll need to have this built first)
    from src.tid import TID
    try:
        tid = TID.load('results/tid_structure.pkl')
        print(f"\n✓ Loaded TID: {tid.total_transactions:,} transactions, {len(tid.item_to_tids):,} items")
    except:
        print("\n❌ Could not load TID structure!")
        print("   Please build TID first by running the data loader.")
        return
    
    # Test 1: Process spawning overhead
    print("\n" + "="*80)
    print("TEST 1: Process Spawning Overhead")
    print("="*80)
    
    def dummy_work(x):
        return x * 2
    
    for num_procs in [1, 2, 4, 8, 16]:
        start = time.time()
        with Pool(processes=num_procs) as pool:
            results = pool.map(dummy_work, range(100))
        spawn_time = time.time() - start
        
        print(f"  {num_procs:2d} processors: {spawn_time*1000:.2f}ms to spawn and complete trivial work")
    
    print("\n💡 If spawn time increases significantly, process overhead is your problem!")
    
    # Test 2: TID intersection speed
    print("\n" + "="*80)
    print("TEST 2: TID Intersection Speed")
    print("="*80)
    
    # Get some frequent items
    frequent_1 = tid.get_frequent_items(min_support=0.002)
    if len(frequent_1) < 10:
        print("  Not enough frequent items for testing")
        return
    
    # Test different itemset sizes
    print("\n  Testing TID intersection performance:")
    
    for k in [2, 3, 4]:
        if k == 2:
            test_itemset = frozenset([frequent_1[0][0], frequent_1[1][0]])
        elif k == 3 and len(frequent_1) >= 3:
            test_itemset = frozenset([frequent_1[0][0], frequent_1[1][0], frequent_1[2][0]])
        elif k == 4 and len(frequent_1) >= 4:
            test_itemset = frozenset([frequent_1[0][0], frequent_1[1][0], frequent_1[2][0], frequent_1[3][0]])
        else:
            continue
        
        # Time 1000 intersections
        start = time.time()
        for _ in range(1000):
            count = tid.get_itemset_support_count(test_itemset)
        intersect_time = (time.time() - start) / 1000
        
        print(f"    {k}-itemset: {intersect_time*1000:.3f}ms per intersection")
    
    # Test 3: Data transfer overhead
    print("\n" + "="*80)
    print("TEST 3: Data Transfer Overhead (Pickling TID)")
    print("="*80)
    
    tid_dict = {item: tids for item, tids in tid.item_to_tids.items()}
    
    import pickle
    start = time.time()
    pickled = pickle.dumps(tid_dict)
    pickle_time = time.time() - start
    pickle_size = len(pickled) / (1024**2)  # MB
    
    print(f"  TID dictionary size: {pickle_size:.2f} MB")
    print(f"  Pickle time: {pickle_time*1000:.2f}ms")
    print(f"  Unpickle time: {pickle.loads(pickled) and 'loaded' or ''}")
    
    print(f"\n💡 Each process needs to unpickle {pickle_size:.2f} MB of TID data!")
    print(f"   With {16} processes, that's {16 * pickle_time:.2f}s of overhead just for data transfer!")
    
    # Test 4: Actual workload distribution
    print("\n" + "="*80)
    print("TEST 4: Workload Distribution Analysis")
    print("="*80)
    
    # Generate level-2 candidates
    frequent_items = [item for item, count in frequent_1[:50]]  # Use top 50
    candidates = tid.generate_candidate_2_itemsets(frequent_items)
    
    print(f"\n  Generated {len(candidates):,} 2-itemset candidates from {len(frequent_items)} frequent items")
    
    # Calculate work per processor
    for num_procs in [2, 4, 8, 16]:
        candidates_per_proc = len(candidates) // num_procs
        
        # Estimate time per candidate (use measured intersection time)
        est_time_per_candidate = 0.001  # 1ms (adjust based on Test 2)
        est_total_time = candidates_per_proc * est_time_per_candidate
        
        # Process overhead (conservative)
        spawn_overhead = 0.05 * num_procs  # 50ms per process
        data_transfer = pickle_time * num_procs
        
        useful_work_time = est_total_time
        overhead_time = spawn_overhead + data_transfer
        total_time = useful_work_time + overhead_time
        
        efficiency = useful_work_time / total_time * 100
        
        print(f"\n  {num_procs} processors:")
        print(f"    Candidates per processor: {candidates_per_proc:,}")
        print(f"    Useful work time: {useful_work_time:.3f}s")
        print(f"    Overhead time: {overhead_time:.3f}s")
        print(f"    Total time: {total_time:.3f}s")
        print(f"    Efficiency: {efficiency:.1f}%")
        
        if efficiency < 50:
            print(f"    ⚠️  WARNING: Overhead > useful work! Parallelization hurts performance!")
    
    # Final recommendation
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    # Calculate critical thresholds
    work_per_proc = len(candidates) / 16  # Worst case: 16 processors
    overhead_per_proc = pickle_time + 0.05  # Pickle + spawn
    
    if work_per_proc * 0.001 < overhead_per_proc:
        print("\n❌ PROBLEM IDENTIFIED: Dataset too small for 8-16 processors!")
        print(f"   With {len(candidates):,} candidates and 16 processors:")
        print(f"   - Each processor gets {work_per_proc:.0f} candidates")
        print(f"   - Work time: {work_per_proc * 0.001:.3f}s")
        print(f"   - Overhead: {overhead_per_proc:.3f}s")
        print(f"   - Overhead is {overhead_per_proc / (work_per_proc * 0.001):.1f}x larger than useful work!")
        
        print("\n💡 SOLUTIONS:")
        print("   1. Use LARGER dataset (increase sample_size in config)")
        print("   2. Use FEWER processors (2-4 optimal for this dataset)")
        print("   3. Optimize data transfer (see fixes below)")
        
        # Calculate optimal processor count
        optimal_procs = int(len(candidates) * 0.001 / overhead_per_proc)
        optimal_procs = max(1, min(optimal_procs, 4))  # Clamp to 1-4
        print(f"   4. Optimal processor count for this dataset: {optimal_procs}")
    
    else:
        print("\n✓ Dataset size is appropriate for parallelization")
        print("  Looking for other bottlenecks...")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    diagnose_overhead_sources()