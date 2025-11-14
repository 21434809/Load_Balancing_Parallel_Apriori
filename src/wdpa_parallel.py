import multiprocessing as mp
from multiprocessing import Pool, Manager
import numpy as np
from typing import Dict, List, Set, Tuple, FrozenSet
import time
from collections import defaultdict
import pandas as pd

from src.tid import TID  # Your existing TID implementation


class WDPADistributor:
    """
    Handles the 4 distribution strategies from the paper.
    
    This is the CORE INNOVATION of WDPA:
    How to split work among processors to balance load.
    """
    
    def __init__(self, tid: TID, verbose: bool = True):
        """
        Args:
            tid: Built TID structure (already has all TID lists)
            verbose: Print distribution info
        """
        self.tid = tid
        self.verbose = verbose
    
    def calculate_lattice(self, frequent_k: List[FrozenSet[int]]) -> int:
        """
        Calculate lattice count: number of candidate (k+1)-itemsets
        that can be formed from frequent k-itemsets.
        
        From paper Equation (1):
        Cnt_Lattice(Ii) = [(len(freq_k-1) - 1) - i] - 1
        
        Args:
            frequent_k: List of frequent k-itemsets
            
        Returns:
            Total number of combinations
        """
        n = len(frequent_k)
        # Total combinations = n choose 2
        total_lattice = (n * (n - 1)) // 2
        return total_lattice
    
    def calculate_weight(self, itemset: FrozenSet[int]) -> int:
        """
        Calculate weight of an itemset based on TID list lengths.
        
        From paper Equation (3):
        Value_WeightTid(Ii) = Σ len(TID_i) × len(TID_j) for all pairs in itemset
        
        WHY THIS MATTERS:
        - Larger TID lists = more work for intersection
        - Weight estimates computational cost
        - Better load balancing by distributing heavy items
        
        Example:
            Itemset: {Banana, Milk}
            TID(Banana): 472,565 orders
            TID(Milk): 245,000 orders
            Weight: 472,565 × 245,000 = 115,778,425,000
            
            This is MUCH heavier than rare items!
        
        Args:
            itemset: Frozen set of item IDs
            
        Returns:
            Weight value (higher = more work)
        """
        items = list(itemset)
        weight = 0
        
        # For each pair of items in the itemset
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                tid_i_len = len(self.tid.item_to_tids.get(items[i], set()))
                tid_j_len = len(self.tid.item_to_tids.get(items[j], set()))
                weight += tid_i_len * tid_j_len
        
        return weight
    
    def distribute_block_lattice(self, 
                                 frequent_k: List[FrozenSet[int]], 
                                 num_processors: int) -> Dict[int, List[FrozenSet[int]]]:
        """
        Strategy 1: Block_Lattice (BL)
        
        Simplest strategy: Divide itemsets into equal blocks.
        
        PROBLEM: Uneven load balance!
        - Early blocks have small TID lists (fast)
        - Later blocks have large TID lists (slow)
        
        Example with 8 itemsets, 4 processors:
        P0: [itemset_0, itemset_1]  ← Fast items
        P1: [itemset_2, itemset_3]
        P2: [itemset_4, itemset_5]
        P3: [itemset_6, itemset_7]  ← Slow items (popular products)
        
        Result: P3 takes much longer than P0!
        
        Args:
            frequent_k: Frequent k-itemsets to distribute
            num_processors: Number of processors
            
        Returns:
            Dictionary mapping processor_id -> list of itemsets
        """
        if self.verbose:
            print(f"\n[BL] Distributing {len(frequent_k)} itemsets across {num_processors} processors (Block)")
        
        distribution = defaultdict(list)
        block_size = len(frequent_k) // num_processors
        
        for proc_id in range(num_processors):
            start_idx = proc_id * block_size
            end_idx = start_idx + block_size if proc_id < num_processors - 1 else len(frequent_k)
            
            distribution[proc_id] = frequent_k[start_idx:end_idx]
            
            if self.verbose:
                print(f"  P{proc_id}: {len(distribution[proc_id])} itemsets")
        
        return dict(distribution)
    
    def distribute_cyclic_lattice(self, 
                                  frequent_k: List[FrozenSet[int]], 
                                  num_processors: int) -> Dict[int, List[FrozenSet[int]]]:
        """
        Strategy 2: Cyclic_Lattice (CL)
        
        Round-robin distribution: Better load balance than block.
        
        WHY BETTER:
        - Mixes heavy and light itemsets across processors
        - Avoids all heavy items going to one processor
        
        Example with 8 itemsets, 4 processors:
        P0: [itemset_0, itemset_4]  ← Heavy + Light
        P1: [itemset_1, itemset_5]  ← Heavy + Light
        P2: [itemset_2, itemset_6]  ← Heavy + Light
        P3: [itemset_3, itemset_7]  ← Heavy + Light
        
        Result: More balanced than block!
        
        Args:
            frequent_k: Frequent k-itemsets to distribute
            num_processors: Number of processors
            
        Returns:
            Dictionary mapping processor_id -> list of itemsets
        """
        if self.verbose:
            print(f"\n[CL] Distributing {len(frequent_k)} itemsets across {num_processors} processors (Cyclic)")
        
        distribution = defaultdict(list)
        
        # Round-robin assignment
        for idx, itemset in enumerate(frequent_k):
            proc_id = idx % num_processors
            distribution[proc_id].append(itemset)
        
        if self.verbose:
            for proc_id in range(num_processors):
                print(f"  P{proc_id}: {len(distribution[proc_id])} itemsets")
        
        return dict(distribution)
    
    def distribute_block_weighttid(self, 
                                   frequent_k: List[FrozenSet[int]], 
                                   num_processors: int) -> Dict[int, List[FrozenSet[int]]]:
        """
        Strategy 3: Block_WeightTid (BWT)
        
        Block distribution but considers TID weights!
        
        IMPROVEMENT over BL:
        - Calculate weight for each itemset
        - Sort by weight
        - Distribute blocks of equal WEIGHT (not equal count)
        
        Example:
        Instead of:
          P0: 1000 itemsets (light) = 1M weight
          P1: 1000 itemsets (heavy) = 100M weight  ← BAD!
        
        Do:
          P0: 1500 itemsets (light) = 50M weight
          P1: 500 itemsets (heavy) = 50M weight   ← GOOD!
        
        Args:
            frequent_k: Frequent k-itemsets to distribute
            num_processors: Number of processors
            
        Returns:
            Dictionary mapping processor_id -> list of itemsets
        """
        if self.verbose:
            print(f"\n[BWT] Distributing {len(frequent_k)} itemsets with TID weighting (Block)")
        
        # Calculate weights for all itemsets
        weighted_itemsets = []
        for itemset in frequent_k:
            weight = self.calculate_weight(itemset)
            weighted_itemsets.append((itemset, weight))
        
        # Sort by weight (descending)
        weighted_itemsets.sort(key=lambda x: x[1], reverse=True)
        
        total_weight = sum(w for _, w in weighted_itemsets)
        target_weight_per_proc = total_weight / num_processors
        
        if self.verbose:
            print(f"  Total weight: {total_weight:,}")
            print(f"  Target weight per processor: {target_weight_per_proc:,.0f}")
        
        # Distribute into blocks with equal weight
        distribution = defaultdict(list)
        current_proc = 0
        current_weight = 0
        
        for itemset, weight in weighted_itemsets:
            distribution[current_proc].append(itemset)
            current_weight += weight
            
            # Move to next processor if we've reached target weight
            if current_weight >= target_weight_per_proc and current_proc < num_processors - 1:
                if self.verbose:
                    print(f"  P{current_proc}: {len(distribution[current_proc])} itemsets, weight={current_weight:,}")
                current_proc += 1
                current_weight = 0
        
        if self.verbose:
            print(f"  P{current_proc}: {len(distribution[current_proc])} itemsets, weight={current_weight:,}")
        
        return dict(distribution)
    
    def distribute_cyclic_weighttid(self, 
                                   frequent_k: List[FrozenSet[int]], 
                                   num_processors: int) -> Dict[int, List[FrozenSet[int]]]:
        """
        Strategy 4: Cyclic_WeightTid (CWT) ⭐
        
        BEST STRATEGY from the paper!
        
        Combines cyclic distribution with weight sorting:
        1. Calculate weight for each itemset
        2. Sort by weight (heaviest first)
        3. Distribute cyclically
        
        WHY BEST:
        - Heaviest items distributed first (round-robin)
        - Each processor gets mix of heavy and light
        - Most even load balance
        
        Example with weights:
        Sorted: [Heavy1(100M), Heavy2(90M), Medium1(10M), Light1(1M), ...]
        
        P0: [Heavy1, Medium1, ...]  = Total: 55M
        P1: [Heavy2, Light1, ...]   = Total: 54M
        P2: [Heavy3, Light2, ...]   = Total: 55M
        P3: [Heavy4, Light3, ...]   = Total: 54M
        
        Result: All processors finish at nearly same time!
        
        Paper results: 50-120x speedup vs traditional Apriori
        
        Args:
            frequent_k: Frequent k-itemsets to distribute
            num_processors: Number of processors
            
        Returns:
            Dictionary mapping processor_id -> list of itemsets
        """
        if self.verbose:
            print(f"\n[CWT] Distributing {len(frequent_k)} itemsets with TID weighting (Cyclic) [BEST]")
        
        # Calculate weights for all itemsets
        weighted_itemsets = []
        for itemset in frequent_k:
            weight = self.calculate_weight(itemset)
            weighted_itemsets.append((itemset, weight))
        
        # Sort by weight (descending) - heaviest first
        weighted_itemsets.sort(key=lambda x: x[1], reverse=True)
        
        total_weight = sum(w for _, w in weighted_itemsets)
        
        if self.verbose:
            print(f"  Total weight: {total_weight:,}")
            print(f"  Target weight per processor: {total_weight / num_processors:,.0f}")
        
        # Cyclic distribution
        distribution = defaultdict(list)
        proc_weights = defaultdict(int)
        
        for idx, (itemset, weight) in enumerate(weighted_itemsets):
            proc_id = idx % num_processors
            distribution[proc_id].append(itemset)
            proc_weights[proc_id] += weight
        
        if self.verbose:
            for proc_id in range(num_processors):
                print(f"  P{proc_id}: {len(distribution[proc_id])} itemsets, weight={proc_weights[proc_id]:,}")
        
        return dict(distribution)


def worker_count_itemsets(args):
    """
    Worker function for parallel TID intersection.
    
    This runs on each slave processor (P1, P2, P3, ...).
    
    WHAT IT DOES:
    1. Receives a list of candidate itemsets
    2. For each itemset, performs TID intersection
    3. Counts support and filters by min_support
    4. Returns frequent itemsets back to master
    
    Args:
        args: Tuple of (itemsets, tid_dict, min_count, proc_id)
        
    Returns:
        List of (itemset, count) tuples for frequent itemsets
    """
    itemsets, tid_dict, min_count, proc_id = args
    
    frequent_local = []
    intersections_performed = 0
    
    for itemset in itemsets:
        # Perform TID intersection
        items = list(itemset)
        
        # Start with first item's TID
        if items[0] not in tid_dict:
            continue
        
        result_tid = tid_dict[items[0]].copy()
        
        # Intersect with remaining items
        for item in items[1:]:
            if item not in tid_dict:
                result_tid = set()
                break
            result_tid &= tid_dict[item]
            
            # Early termination if empty
            if not result_tid:
                break
        
        intersections_performed += 1
        count = len(result_tid)
        
        # Check if frequent
        if count >= min_count:
            frequent_local.append((itemset, count))
    
    return frequent_local, intersections_performed


class WDPAParallelMiner:
    """
    Main WDPA parallel mining engine.
    
    Implements the Master-Slave architecture from the paper.
    """
    
    def __init__(self, 
                 tid: TID, 
                 min_support: float, 
                 num_processors: int = 4,
                 strategy: str = 'CWT',
                 max_k: int = 5,
                 verbose: bool = True):
        """
        Args:
            tid: Built TID structure
            min_support: Minimum support threshold (e.g., 0.002 for 0.2%)
            num_processors: Number of processors to use
            strategy: Distribution strategy: 'BL', 'CL', 'BWT', 'CWT'
            max_k: Maximum itemset size
            verbose: Print progress
        """
        self.tid = tid
        self.min_support = min_support
        self.min_count = int(min_support * tid.total_transactions)
        self.num_processors = num_processors
        self.strategy = strategy
        self.max_k = max_k
        self.verbose = verbose
        
        self.distributor = WDPADistributor(tid, verbose=verbose)
        
        # Metrics
        self.metrics = {
            'total_time': 0,
            'level_times': {},
            'itemsets_per_level': {},
            'intersections_per_level': {},
            'distribution_time': {}
        }
    
    def mine(self) -> Dict[int, List[Tuple[FrozenSet[int], int]]]:
        """
        Main mining algorithm following the paper's steps.
        
        ALGORITHM (from paper):
        Step 1: Master reads DB (DONE - TID built)
        Step 2: Each processor scans DB (DONE - TID)
        Step 3: Calculate candidate k-itemsets
        Step 4: Master distributes itemsets using strategy
        Step 5: Each processor receives itemsets
        Step 6: Each processor counts by TID intersection
        Step 7: Filter by min_support
        Step 8: Slaves send results to Master
        Step 9: Repeat until no more frequent itemsets
        
        Returns:
            Dictionary mapping k -> list of (itemset, count) tuples
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"WDPA PARALLEL MINING - Strategy: {self.strategy}")
            print("="*80)
            print(f"Processors: {self.num_processors}")
            print(f"Min support: {self.min_support*100:.2f}% ({self.min_count:,} orders)")
            print(f"Total transactions: {self.tid.total_transactions:,}")
            print("="*80)
        
        start_time = time.time()
        frequent_by_level = {}
        
        # Step 3: Find frequent 1-itemsets (no parallelization needed)
        if self.verbose:
            print("\n[Level 1] Finding frequent items...")
        
        level_start = time.time()
        frequent_1 = self.tid.get_frequent_items(self.min_support)
        frequent_by_level[1] = [(frozenset([item]), count) for item, count in frequent_1]
        self.metrics['level_times'][1] = time.time() - level_start
        self.metrics['itemsets_per_level'][1] = len(frequent_1)
        
        if self.verbose:
            print(f"  Found {len(frequent_1)} frequent items")
            print(f"  Time: {self.metrics['level_times'][1]:.2f}s")
        
        if not frequent_1:
            print("No frequent items found!")
            return frequent_by_level
        
        # Prepare for levels 2+
        frequent_items = [item for item, count in frequent_1]
        frequent_k_minus_1 = [frozenset([item]) for item in frequent_items]
        
        # Convert TID to dictionary for pickling (multiprocessing requirement)
        tid_dict = {item: tids for item, tids in self.tid.item_to_tids.items()}
        
        # Levels 2 and beyond
        for k in range(2, self.max_k + 1):
            if self.verbose:
                print(f"\n[Level {k}] Generating and testing candidates...")
            
            level_start = time.time()
            
            # Generate candidates
            if k == 2:
                candidates = self.tid.generate_candidate_2_itemsets(frequent_items)
            else:
                candidates = self.tid.generate_candidate_k_itemsets(frequent_k_minus_1)
            
            if not candidates:
                if self.verbose:
                    print(f"  No candidates generated. Mining complete.")
                break
            
            if self.verbose:
                print(f"  Generated {len(candidates):,} candidates")
            
            # Step 4: Distribute candidates using chosen strategy
            dist_start = time.time()
            
            if self.strategy == 'BL':
                distribution = self.distributor.distribute_block_lattice(candidates, self.num_processors)
            elif self.strategy == 'CL':
                distribution = self.distributor.distribute_cyclic_lattice(candidates, self.num_processors)
            elif self.strategy == 'BWT':
                distribution = self.distributor.distribute_block_weighttid(candidates, self.num_processors)
            elif self.strategy == 'CWT':
                distribution = self.distributor.distribute_cyclic_weighttid(candidates, self.num_processors)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            self.metrics['distribution_time'][k] = time.time() - dist_start
            
            # Step 5-6: Parallel counting using multiprocessing
            if self.verbose:
                print(f"  Distributing work across {self.num_processors} processors...")
            
            # Prepare arguments for workers (only for processors with work)
            worker_args = [
                (distribution.get(proc_id, []), tid_dict, self.min_count, proc_id)
                for proc_id in range(self.num_processors)
            ]
            
            # Execute in parallel
            with Pool(processes=self.num_processors) as pool:
                results = pool.map(worker_count_itemsets, worker_args)
            
            # Step 8: Collect results from all processors
            frequent_k = []
            total_intersections = 0
            
            for frequent_local, intersections in results:
                frequent_k.extend(frequent_local)
                total_intersections += intersections
            
            if not frequent_k:
                if self.verbose:
                    print(f"  No frequent {k}-itemsets found. Mining complete.")
                break
            
            # Sort by count
            frequent_k.sort(key=lambda x: -x[1])
            frequent_by_level[k] = frequent_k
            
            level_time = time.time() - level_start
            self.metrics['level_times'][k] = level_time
            self.metrics['itemsets_per_level'][k] = len(frequent_k)
            self.metrics['intersections_per_level'][k] = total_intersections
            
            if self.verbose:
                print(f"  Found {len(frequent_k):,} frequent {k}-itemsets")
                print(f"  Performed {total_intersections:,} TID intersections")
                print(f"  Time: {level_time:.2f}s")
            
            # Prepare for next level
            frequent_k_minus_1 = [itemset for itemset, count in frequent_k]
        
        self.metrics['total_time'] = time.time() - start_time
        
        if self.verbose:
            print("\n" + "="*80)
            print("MINING COMPLETE!")
            print("="*80)
            total = sum(len(itemsets) for itemsets in frequent_by_level.values())
            print(f"Total frequent itemsets: {total:,}")
            print(f"Maximum itemset size: {max(frequent_by_level.keys())}")
            print(f"Total time: {self.metrics['total_time']:.2f}s")
            print("\nSummary by level:")
            for k in sorted(frequent_by_level.keys()):
                print(f"  Level {k}: {len(frequent_by_level[k]):,} itemsets " 
                      f"({self.metrics['level_times'].get(k, 0):.2f}s)")
            print("="*80)
        
        return frequent_by_level
    
    def get_metrics(self) -> Dict:
        """Return performance metrics"""
        return self.metrics


# Example usage
if __name__ == "__main__":
    print("WDPA Implementation Test")
    print("This module should be imported and used with a built TID structure.")
    print("\nExample usage:")
    print("""
    from tid import TID
    from wdpa_parallel import WDPAParallelMiner
    
    # Build TID structure
    tid = TID.load('tid_full.pkl')
    
    # Run WDPA with CWT strategy
    miner = WDPAParallelMiner(
        tid=tid,
        min_support=0.002,
        num_processors=8,
        strategy='CWT',
        verbose=True
    )
    
    results = miner.mine()
    """)