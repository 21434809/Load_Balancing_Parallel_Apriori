import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, FrozenSet
import time
import pickle


class TID: 
    def __init__(self):
        """Initialize empty TID structure"""
        # Maps item -> set of transaction IDs containing that item
        self.item_to_tids: Dict[int, Set[int]] = {}
        
        # Store itemset TIDs (for k-itemsets where k >= 2)
        self.itemset_to_tids: Dict[FrozenSet[int], Set[int]] = {}
        
        # Metadata
        self.total_transactions = 0
        self.build_time_seconds = 0.0
        
    def build_from_orders(self, orders_df: pd.DataFrame, order_col='order_id', product_col='product_id', verbose=True):
        """
        Args:
            orders_df: DataFrame with columns [order_id, product_id]
            order_col: Column name for transaction/order IDs
            product_col: Column name for item/product IDs
            verbose: Print progress info
        """
        if verbose:
            print("\n" + "="*60)
            print("Building TID Structure")
            print("="*60)
        
        start_time = time.time()
        
        # Clear any existing data
        self.item_to_tids.clear()
        self.itemset_to_tids.clear()
        
        # Count unique transactions
        self.total_transactions = orders_df[order_col].nunique()
        
        if verbose:
            print(f"Total transactions: {self.total_transactions:,}")
            print(f"Total records: {len(orders_df):,}")
            print("\nScanning database and building TID lists...")
        
        # SINGLE SCAN: Group by product and collect all order IDs
        for product_id, group in orders_df.groupby(product_col):
            self.item_to_tids[product_id] = set(group[order_col].unique())
        
        self.build_time_seconds = time.time() - start_time
        
        if verbose:
            print(f"Built TID lists for {len(self.item_to_tids):,} unique items")
            print(f"Build time: {self.build_time_seconds:.3f} seconds")
            avg_items = len(orders_df) / self.total_transactions
            print(f"Average items per transaction: {avg_items:.2f}")
            print("="*60)
    
    def get_support_count(self, item: int) -> int:
        """
        Get support count for a single item.
        
        Support count = number of transactions containing the item = length of TID list
        
        Args:
            item: Product/item ID
            
        Returns:
            Number of transactions containing this item
        """
        return len(self.item_to_tids.get(item, set()))
    
    def get_support(self, item: int) -> float:
        """
        Get support (as percentage) for a single item.
        
        Support = count / total_transactions
        
        Args:
            item: Product/item ID
            
        Returns:
            Support as decimal (e.g., 0.05 for 5%)
        """
        count = self.get_support_count(item)
        return count / self.total_transactions if self.total_transactions > 0 else 0.0
    
    def get_itemset_support_count(self, itemset: FrozenSet[int]) -> int:
        """
        Get support count for an itemset using TID INTERSECTION.
        
        Example:
            TID(Banana) = {1, 2, 5, 7, 9}
            TID(Milk)   = {1, 3, 5, 9, 10}
            TID({Banana, Milk}) = {1, 5, 9}  <- Intersection!
            Support count = 3
        
        Args:
            itemset: Frozenset of item IDs
            
        Returns:
            Number of transactions containing ALL items in the itemset
        """
        if not itemset:
            return 0
        
        # Check cache first
        if itemset in self.itemset_to_tids:
            return len(self.itemset_to_tids[itemset])
        
        # Get TID lists and intersect them
        items = list(itemset)
        
        # Start with first item's TID
        if items[0] not in self.item_to_tids:
            return 0
        
        result_tid = self.item_to_tids[items[0]].copy()
        
        # Intersect with remaining items
        for item in items[1:]:
            if item not in self.item_to_tids:
                return 0
            result_tid &= self.item_to_tids[item]
            
            # Early termination if empty
            if not result_tid:
                return 0
        
        # Cache the result
        self.itemset_to_tids[itemset] = result_tid
        
        return len(result_tid)
    
    def get_itemset_support(self, itemset: FrozenSet[int]) -> float:
        """
        Get support (as percentage) for an itemset.
        
        Args:
            itemset: Frozenset of item IDs
            
        Returns:
            Support as decimal
        """
        count = self.get_itemset_support_count(itemset)
        return count / self.total_transactions if self.total_transactions > 0 else 0.0
    
    def get_frequent_items(self, min_support: float) -> List[Tuple[int, int]]:
        """
        Get frequent 1-itemsets (single items) above minimum support.
        
        Args:
            min_support: Minimum support threshold (e.g., 0.002 for 0.2%)
            
        Returns:
            List of (item_id, count) tuples, sorted by count descending
        """
        min_count = int(min_support * self.total_transactions)
        
        frequent = [
            (item, len(tids))
            for item, tids in self.item_to_tids.items()
            if len(tids) >= min_count
        ]
        
        # Sort by count descending, then by item ID
        frequent.sort(key=lambda x: (-x[1], x[0]))
        
        return frequent
    
    def generate_candidate_2_itemsets(self, frequent_1_items: List[int]) -> List[FrozenSet[int]]:
        """
        Generate candidate 2-itemsets from frequent 1-itemsets.
        
        For 2-itemsets, we just combine all pairs of frequent items.
        
        Args:
            frequent_1_items: List of frequent item IDs
            
        Returns:
            List of candidate 2-itemsets (as frozensets)
        """
        candidates = []
        n = len(frequent_1_items)
        
        for i in range(n):
            for j in range(i + 1, n):
                candidate = frozenset([frequent_1_items[i], frequent_1_items[j]])
                candidates.append(candidate)
        
        return candidates
    
    def generate_candidate_k_itemsets(self, frequent_k_minus_1: List[FrozenSet[int]]) -> List[FrozenSet[int]]:
        """
        Generate candidate k-itemsets from frequent (k-1)-itemsets.
        
        This uses the Apriori principle: join itemsets that differ by one item.
        
        Args:
            frequent_k_minus_1: List of frequent (k-1)-itemsets
            
        Returns:
            List of candidate k-itemsets
        """
        candidates = []
        n = len(frequent_k_minus_1)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Union the two itemsets
                union = frequent_k_minus_1[i] | frequent_k_minus_1[j]
                
                # Only keep if the union has exactly k items
                # (i.e., they differ by exactly one item)
                if len(union) == len(frequent_k_minus_1[i]) + 1:
                    # Pruning: check if all (k-1) subsets are frequent
                    if self._has_all_frequent_subsets(union, frequent_k_minus_1):
                        candidates.append(union)
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        return candidates
    
    def _has_all_frequent_subsets(self, itemset: FrozenSet[int], frequent_k_minus_1: List[FrozenSet[int]]) -> bool:
        """
        Check if all (k-1) subsets of itemset are frequent.
        
        This is the pruning step in Apriori: if any subset is not frequent, the itemset cannot be frequent.
        
        Args:
            itemset: Candidate k-itemset to check
            frequent_k_minus_1: List of frequent (k-1)-itemsets
            
        Returns:
            True if all subsets are frequent
        """
        frequent_set = set(frequent_k_minus_1)
        
        # Generate all (k-1) subsets
        items = list(itemset)
        for i in range(len(items)):
            subset = frozenset(items[:i] + items[i+1:])
            if subset not in frequent_set:
                return False
        
        return True
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the TID structure.
        
        Returns:
            Dictionary with stats
        """
        if not self.item_to_tids:
            return {}
        
        tid_lengths = [len(tids) for tids in self.item_to_tids.values()]
        
        return {
            'total_transactions': self.total_transactions,
            'unique_items': len(self.item_to_tids),
            'build_time_seconds': self.build_time_seconds,
            'avg_tid_length': np.mean(tid_lengths),
            'median_tid_length': np.median(tid_lengths),
            'min_tid_length': np.min(tid_lengths),
            'max_tid_length': np.max(tid_lengths),
            'std_tid_length': np.std(tid_lengths),
        }
    
    def save(self, filepath: str):
        """Save TID structure to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"TID structure saved to: {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'TID':
        """Load TID structure from disk"""
        with open(filepath, 'rb') as f:
            tid = pickle.load(f)
        print(f"TID structure loaded from: {filepath}")
        return tid


def mine_frequent_itemsets(tid: TID, min_support: float, max_k: int = 10, 
                          verbose: bool = True) -> Dict[int, List[Tuple[FrozenSet[int], int]]]:
    """
    Mine frequent itemsets using the TID structure.
    
    Args:
        tid: TID structure (already built)
        min_support: Minimum support threshold
        max_k: Maximum itemset size to mine
        verbose: Print progress
        
    Returns:
        Dictionary mapping k -> list of (itemset, count) tuples
    """
    if verbose:
        print("\n" + "="*60)
        print("Mining Frequent Itemsets with TID Structure")
        print("="*60)
        print(f"Minimum support: {min_support*100:.2f}%")
        print(f"Minimum count: {int(min_support * tid.total_transactions)}")
    
    min_count = int(min_support * tid.total_transactions)
    frequent_by_level = {}
    
    # Level 1: Frequent items
    if verbose:
        print("\n[Level 1] Finding frequent items...")
    
    frequent_1 = tid.get_frequent_items(min_support)
    frequent_by_level[1] = [(frozenset([item]), count) for item, count in frequent_1]
    
    if verbose:
        print(f"Found {len(frequent_1)} frequent items")
    
    if not frequent_1:
        print("No frequent items found!")
        return frequent_by_level
    
    # Level 2 and beyond
    frequent_items = [item for item, count in frequent_1]
    frequent_k_minus_1 = [frozenset([item]) for item in frequent_items]
    
    for k in range(2, max_k + 1):
        if verbose:
            print(f"\n[Level {k}] Generating and testing candidates...")
        
        # Generate candidates
        if k == 2:
            candidates = tid.generate_candidate_2_itemsets(frequent_items)
        else:
            candidates = tid.generate_candidate_k_itemsets(frequent_k_minus_1)
        
        if not candidates:
            if verbose:
                print(f"No candidates generated. Mining complete.")
            break
        
        if verbose:
            print(f"  Generated {len(candidates)} candidates")
        
        # Test candidates and keep frequent ones
        frequent_k = []
        for candidate in candidates:
            count = tid.get_itemset_support_count(candidate)
            if count >= min_count:
                frequent_k.append((candidate, count))
        
        if not frequent_k:
            if verbose:
                print(f"No frequent {k}-itemsets found. Mining complete.")
            break
        
        # Sort by count descending
        frequent_k.sort(key=lambda x: -x[1])
        
        frequent_by_level[k] = frequent_k
        
        if verbose:
            print(f"Found {len(frequent_k)} frequent {k}-itemsets")
        
        # Prepare for next level
        frequent_k_minus_1 = [itemset for itemset, count in frequent_k]
    
    if verbose:
        print("\n" + "="*60)
        print("Mining Complete!")
        print("="*60)
        total = sum(len(itemsets) for itemsets in frequent_by_level.values())
        print(f"Total frequent itemsets: {total}")
        print(f"Maximum itemset size: {max(frequent_by_level.keys())}")
        print("\nSummary by level:")
        for k in sorted(frequent_by_level.keys()):
            print(f"  Level {k}: {len(frequent_by_level[k])} itemsets")
        print("="*60)
    
    return frequent_by_level


# Example usage
if __name__ == "__main__":
    print("\nTID Structure - Example with Small Dataset")
    print("="*60)
    
    # Create small example from the paper
    data = {
        'order_id': [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5],
        'product_id': [1, 3, 4, 7, 1, 6, 1, 2, 3, 4, 1, 3, 3, 4, 7]
    }
    df = pd.DataFrame(data)
    
    print("\nExample transactions:")
    for order_id in df['order_id'].unique():
        products = df[df['order_id'] == order_id]['product_id'].tolist()
        print(f"Order {order_id}: {products}")
    
    # Build TID
    tid = TID()
    tid.build_from_orders(df)
    
    # Show TID lists
    print("\nTID Lists:")
    for item in sorted(tid.item_to_tids.keys()):
        tids = sorted(tid.item_to_tids[item])
        print(f"Product {item}: Orders {tids} (count={len(tids)})")
    
    # Mine frequent itemsets
    frequent = mine_frequent_itemsets(tid, min_support=0.4, verbose=True)
    
    # Show some results
    print("\nExample frequent 2-itemsets:")
    if 2 in frequent:
        for itemset, count in frequent[2][:5]:
            items = sorted(list(itemset))
            support = count / tid.total_transactions
            print(f"  {items}: count={count}, support={support*100:.1f}%")