"""
Processor Scaling Comparison for WDPA-CWT

Tests WDPA-CWT with different processor counts:
- 2 processors
- 4 processors  
- 8 processors
- 16 processors

Logs are APPENDED to file (not overwritten).
"""

import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime

from tid import TID
# from wdpa_parallel import WDPADistributorImproved
from improved_wdpa_parallel import WDPAParallelMiner


class ProcessorScalingTest:
    """
    Test WDPA-CWT scaling across different processor counts.
    """
    
    def __init__(self, tid_filepath='tid_full.pkl', verbose=True):
        self.tid_filepath = tid_filepath
        self.verbose = verbose
        self.all_results = []
        
        print("Loading TID structure...")
        self.tid = TID.load(tid_filepath)
        print(f"Loaded: {self.tid.total_transactions:,} transactions, {len(self.tid.item_to_tids):,} items\n")
    
    def run_wdpa_with_processors(self, num_processors, min_support=0.002):
        """
        Run WDPA-CWT with specified number of processors.
        """
        if self.verbose:
            print("\n" + "█"*80)
            print(f"TESTING: WDPA-CWT with {num_processors} PROCESSORS")
            print("█"*80)
            print(f"Dataset: {self.tid.total_transactions:,} orders")
            print(f"Min support: {min_support*100:.2f}%")
            print(f"Strategy: CWT (Cyclic WeightTid)")
        
        start_time = time.time()
        
        try:
            # Run WDPA
            miner = WDPAParallelMiner(
                tid=self.tid,
                min_support=min_support,
                num_processors=num_processors,
                strategy='CWT',
                max_k=5,
                verbose=self.verbose
            )
            
            frequent_itemsets = miner.mine()
            metrics = miner.get_metrics()
            
            total_itemsets = sum(len(itemsets) for itemsets in frequent_itemsets.values())
            total_time = time.time() - start_time
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'num_processors': num_processors,
                'strategy': 'CWT',
                'min_support': min_support,
                'dataset_size': self.tid.total_transactions,
                'total_time': metrics['total_time'],
                'total_itemsets': total_itemsets,
                'itemsets_by_level': {k: len(v) for k, v in frequent_itemsets.items()},
                'level_times': metrics['level_times'],
                'success': True
            }
            
            if self.verbose:
                print(f"\nSUCCESS!")
                print(f"  Total time: {results['total_time']:.2f}s ({results['total_time']/60:.1f} min)")
                print(f"  Itemsets found: {total_itemsets:,}")
                print(f"  Levels: {list(frequent_itemsets.keys())}")
            
        except Exception as e:
            if self.verbose:
                print(f"\nERROR: {e}")
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'num_processors': num_processors,
                'strategy': 'CWT',
                'error': str(e),
                'success': False
            }
        
        return results
    
    def save_results_append(self, result):
        """
        APPEND results to JSON file (not overwrite).
        """
        results_file = 'results/processor_scaling_results.json'
        os.makedirs('results', exist_ok=True)
        
        # Load existing results if file exists
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    existing_data = json.load(f)
                    if 'experiments' in existing_data:
                        all_results = existing_data['experiments']
                    else:
                        all_results = []
            except:
                all_results = []
        else:
            all_results = []
        
        # Append new result
        all_results.append(result)
        
        # Save updated results
        data = {
            'metadata': {
                'test_type': 'processor_scaling',
                'strategy': 'CWT',
                'last_updated': datetime.now().isoformat(),
                'total_experiments': len(all_results)
            },
            'experiments': all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        if self.verbose:
            print(f"\nResults APPENDED to: {results_file}")
            print(f"  Total experiments in file: {len(all_results)}")
    
    def save_log_append(self, result):
        """
        APPEND human-readable log to text file.
        """
        log_file = 'results/processor_scaling_log.txt'
        os.makedirs('results', exist_ok=True)
        
        # Append to log
        with open(log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Experiment: {result['timestamp']}\n")
            f.write("="*80 + "\n")
            f.write(f"Processors: {result['num_processors']}\n")
            f.write(f"Strategy: {result.get('strategy', 'CWT')}\n")
            f.write(f"Min Support: {result.get('min_support', 0.002)*100:.2f}%\n")
            
            if result['success']:
                f.write(f"\nSTATUS: SUCCESS\n")
                f.write(f"Total Time: {result['total_time']:.2f}s ({result['total_time']/60:.1f} min)\n")
                f.write(f"Itemsets Found: {result['total_itemsets']:,}\n")
                f.write(f"Itemsets by Level: {result.get('itemsets_by_level', {})}\n")
                
                if 'level_times' in result:
                    f.write(f"\nTime per Level:\n")
                    for level, time_val in result['level_times'].items():
                        f.write(f"  Level {level}: {time_val:.2f}s\n")
            else:
                f.write(f"\nSTATUS: FAILED\n")
                f.write(f"Error: {result.get('error', 'Unknown')}\n")
            
            f.write("\n")
        
        if self.verbose:
            print(f"✓ Log APPENDED to: {log_file}")
    
    def run_scaling_test(self, processor_counts=[2, 4, 8, 16], min_support=0.002):
        """
        Run WDPA-CWT with different processor counts.
        """
        print("\n" + "█"*80)
        print("PROCESSOR SCALING TEST - WDPA-CWT")
        print("█"*80)
        print(f"Testing with: {processor_counts} processors")
        print(f"Min support: {min_support*100:.2f}%")
        print("█"*80)
        
        for num_procs in processor_counts:
            # Run test
            result = self.run_wdpa_with_processors(num_procs, min_support)
            
            # Save results (append)
            self.save_results_append(result)
            self.save_log_append(result)
            
            # Store for summary
            self.all_results.append(result)
            
            if self.verbose:
                print(f"\n{'─'*80}")
                print(f"Completed {num_procs} processors")
                print(f"{'─'*80}")
        
        # Print final summary
        self.print_summary()
    
    def print_summary(self):
        """Print comparison summary of all runs."""
        print("\n" + "█"*80)
        print("PROCESSOR SCALING SUMMARY")
        print("█"*80)
        
        if not self.all_results:
            print("No results to summarize")
            return
        
        # Filter successful runs
        successful = [r for r in self.all_results if r['success']]
        
        if not successful:
            print("No successful runs")
            return
        
        # Sort by processor count
        successful.sort(key=lambda x: x['num_processors'])
        
        # Find baseline (lowest processor count)
        baseline = successful[0]
        baseline_time = baseline['total_time']
        
        print("\n📊 RESULTS TABLE:")
        print("-"*80)
        print(f"{'Processors':<12} {'Time (min)':<15} {'Speedup':<12} {'Efficiency':<12}")
        print("-"*80)
        
        for result in successful:
            num_procs = result['num_processors']
            time_min = result['total_time'] / 60
            speedup = baseline_time / result['total_time']
            efficiency = (speedup / num_procs) * 100
            
            print(f"{num_procs:<12} {time_min:<15.1f} {speedup:<12.2f}x {efficiency:<12.1f}%")
        
        print("-"*80)
        
        # Best result
        fastest = min(successful, key=lambda x: x['total_time'])
        print(f"\nFASTEST: {fastest['num_processors']} processors")
        print(f"   Time: {fastest['total_time']/60:.1f} minutes")
        print(f"   Speedup vs {baseline['num_processors']}p: {baseline_time/fastest['total_time']:.2f}x")
        
        print("\n" + "█"*80)
    
    def generate_summary_file(self):
        """Generate a summary text file."""
        summary_file = 'results/processor_scaling_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PROCESSOR SCALING TEST SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Strategy: CWT (Cyclic WeightTid)\n")
            f.write(f"Dataset: {self.tid.total_transactions:,} orders\n")
            f.write("\n")
            
            successful = [r for r in self.all_results if r['success']]
            successful.sort(key=lambda x: x['num_processors'])
            
            if successful:
                baseline = successful[0]
                baseline_time = baseline['total_time']
                
                f.write("RESULTS:\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Processors':<12} {'Time (min)':<15} {'Speedup':<12} {'Efficiency':<12}\n")
                f.write("-"*80 + "\n")
                
                for result in successful:
                    num_procs = result['num_processors']
                    time_min = result['total_time'] / 60
                    speedup = baseline_time / result['total_time']
                    efficiency = (speedup / num_procs) * 100
                    
                    f.write(f"{num_procs:<12} {time_min:<15.1f} {speedup:<12.2f}x {efficiency:<12.1f}%\n")
                
                f.write("-"*80 + "\n")
                
                fastest = min(successful, key=lambda x: x['total_time'])
                f.write(f"\nFASTEST: {fastest['num_processors']} processors\n")
                f.write(f"Time: {fastest['total_time']/60:.1f} minutes\n")
                f.write(f"Speedup: {baseline_time/fastest['total_time']:.2f}x vs {baseline['num_processors']}p\n")
        
        print(f"\nSummary saved to: {summary_file}")


def main():
    """Main execution."""
    print("="*80)
    print("PROCESSOR SCALING TEST - WDPA-CWT")
    print("="*80)
    print("\nThis tests WDPA-CWT with different processor counts.")
    print("Results will be APPENDED to existing files (not overwritten).")
    print("="*80)
    
    if not os.path.exists('tid_full.pkl'):
        print("\n✗ ERROR: tid_full.pkl not found!")
        return
    
    # Create tester
    tester = ProcessorScalingTest('tid_full.pkl', verbose=True)
    
    # Test with different processor counts
    # Start with 2, then 4, then 8, then 16
    tester.run_scaling_test(
        processor_counts=[2, 4, 8, 16],  # Change this to test specific counts
        min_support=0.002  # 0.2% support
    )
    
    # Generate summary
    tester.generate_summary_file()
    
    print("\n" + "="*80)
    print("PROCESSOR SCALING TEST COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  - results/processor_scaling_results.json (appended)")
    print("  - results/processor_scaling_log.txt (appended)")
    print("  - results/processor_scaling_summary.txt (summary)")


if __name__ == "__main__":
    main()