"""
Unified Data Loader for Benchmark Comparison

This module loads the Instacart dataset ONCE and prepares it in both formats:
1. Binary basket matrix (for Traditional and Naive Parallel Apriori)
2. TID structure (for WDPA)

This ensures ALL algorithms work on the EXACT SAME data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import time

from src.tid import TID


class UnifiedDataLoader:
    """
    Loads data once and prepares it for all algorithm types.
    """

    def __init__(self, config: Dict, algorithms_config: Dict = None):
        """
        Args:
            config: Dataset configuration with keys:
                - sample_size: Number of orders to sample
                - max_items: Maximum number of unique products to keep
                - random_seed: Random seed for reproducibility
            algorithms_config: Optional algorithms configuration to determine what to load
        """
        self.sample_size = config.get('sample_size', 10000)
        self.max_items = config.get('max_items', 500)
        self.random_seed = config.get('random_seed', 42)

        # Determine what data structures we need to create
        self.need_basket_matrix = True  # Default to True for backwards compatibility
        self.need_tid = True  # Default to True for backwards compatibility
        self.need_transactions = False

        if algorithms_config:
            # Only create basket matrix if traditional or naive_parallel is enabled
            self.need_basket_matrix = (
                algorithms_config.get('traditional', {}).get('enabled', False) or
                algorithms_config.get('naive_parallel', {}).get('enabled', False)
            )
            # Only create TID if WDPA is enabled
            self.need_tid = algorithms_config.get('wdpa', {}).get('enabled', False)
            # Create transactions if Ye parallel is enabled
            self.need_transactions = algorithms_config.get('ye_parallel', {}).get('enabled', False)

        # Will be populated by load()
        self.market_basket = None  # Raw transaction data
        self.basket_encoded = None  # Binary basket matrix
        self.tid = None  # TID structure
        self.transactions = None  # Horizontal transactions (List[List[int]])

        self.load_time = 0.0
        self.transform_time = 0.0
        self.tid_build_time = 0.0

    def load(self, verbose: bool = True) -> Dict:
        """
        Load and prepare data in both formats.

        Returns:
            Dictionary with prepared data and metadata
        """
        if verbose:
            print("="*80)
            print("UNIFIED DATA LOADER")
            print("="*80)
            print(f"Sample size: {self.sample_size:,} orders")
            print(f"Max items: {self.max_items:,} products")
            print(f"Random seed: {self.random_seed}")
            print(f"Will create basket matrix: {self.need_basket_matrix}")
            print(f"Will create TID structure: {self.need_tid}")
            print(f"Will create transactions (Ye): {self.need_transactions}")

        # Step 1: Load raw data
        if verbose:
            print("\n[1/3] Loading raw transaction data...")
        load_start = time.time()
        self.market_basket = self._load_raw_data(verbose)
        self.load_time = time.time() - load_start

        if verbose:
            print(f"  Loaded {len(self.market_basket):,} transactions")
            print(f"  Unique orders: {self.market_basket['order_id'].nunique():,}")
            print(f"  Unique products: {self.market_basket['product_id'].nunique():,}")
            print(f"  Time: {self.load_time:.2f}s")

        # Step 2: Create binary basket matrix (for Traditional & Naive) - CONDITIONAL
        if self.need_basket_matrix:
            if verbose:
                print("\n[2/3] Creating binary basket matrix...")
            transform_start = time.time()
            self.basket_encoded = self._create_basket_matrix(verbose)
            self.transform_time = time.time() - transform_start

            if verbose:
                print(f"  Shape: {self.basket_encoded.shape}")
                print(f"  Time: {self.transform_time:.2f}s")
        else:
            if verbose:
                print("\n[2/3] Skipping binary basket matrix (not needed for enabled algorithms)")
            self.transform_time = 0.0

        # Step 3: Build TID structure (for WDPA) - CONDITIONAL
        if self.need_tid:
            if verbose:
                print("\n[3/3] Building TID structure...")
            tid_start = time.time()
            self.tid = self._build_tid_structure(verbose)
            self.tid_build_time = time.time() - tid_start

            if verbose:
                print(f"  Total transactions: {self.tid.total_transactions:,}")
                print(f"  Unique items: {len(self.tid.item_to_tids):,}")
                print(f"  Time: {self.tid_build_time:.2f}s")
        else:
            if verbose:
                print("\n[3/3] Skipping TID structure (not needed for enabled algorithms)")
            self.tid_build_time = 0.0

        # Step 4: Build horizontal transactions for Ye - CONDITIONAL
        if self.need_transactions:
            if verbose:
                print("\n[EXTRA] Creating horizontal transactions for Ye (2006) parallel Apriori...")
            self.transactions = self._create_transactions(verbose)
            if verbose:
                print(f"  Transactions list created: {len(self.transactions):,} orders")
        else:
            self.transactions = None

        if verbose:
            print("\n" + "="*80)
            print("DATA LOADING COMPLETE")
            print(f"Total time: {self.load_time + self.transform_time + self.tid_build_time:.2f}s")
            print("="*80)

        return {
            'market_basket': self.market_basket,
            'basket_encoded': self.basket_encoded,
            'tid': self.tid,
            'transactions': self.transactions,
            'metadata': {
                'total_transactions': len(self.market_basket),
                'unique_orders': self.market_basket['order_id'].nunique(),
                'unique_products': self.market_basket['product_id'].nunique(),
                'basket_shape': self.basket_encoded.shape if self.basket_encoded is not None else None,
                'tid_total_transactions': self.tid.total_transactions if self.tid is not None else None,
                'tid_unique_items': len(self.tid.item_to_tids) if self.tid is not None else None,
                'has_transactions': self.transactions is not None,
                'load_time': self.load_time,
                'transform_time': self.transform_time,
                'tid_build_time': self.tid_build_time
            }
        }

    def _load_raw_data(self, verbose: bool = True) -> pd.DataFrame:
        """Load and sample the Instacart dataset."""
        # Load the main data files
        orders = pd.read_csv('data/orders.csv')
        order_products = pd.read_csv('data/order_products__prior.csv')
        products = pd.read_csv('data/products.csv')

        # Sample orders
        if verbose:
            print(f"  Sampling {self.sample_size:,} from {len(orders):,} total orders...")

        np.random.seed(self.random_seed)
        sample_orders = orders.sample(n=min(self.sample_size, len(orders)), random_state=self.random_seed)
        sample_order_ids = set(sample_orders['order_id'])

        # Filter order_products to only include sampled orders
        order_products_sample = order_products[order_products['order_id'].isin(sample_order_ids)]

        # Merge the datasets
        market_basket = order_products_sample.merge(products, on='product_id').merge(orders, on='order_id')
        market_basket = market_basket[['order_id', 'user_id', 'product_id', 'product_name']]

        # Filter to most frequent items
        if self.max_items and len(market_basket['product_name'].unique()) > self.max_items:
            if verbose:
                print(f"  Filtering to top {self.max_items:,} most frequent items...")
            item_counts = market_basket['product_name'].value_counts()
            top_items = set(item_counts.head(self.max_items).index)
            market_basket = market_basket[market_basket['product_name'].isin(top_items)]

        return market_basket

    def _create_basket_matrix(self, verbose: bool = True) -> pd.DataFrame:
        """Create binary basket matrix for mlxtend Apriori."""
        # Create pivot table with orders as rows and products as columns
        basket = self.market_basket.groupby(['order_id', 'product_name']).size().unstack(fill_value=0)

        # Convert to binary (0/1) format
        basket_encoded = (basket > 0).astype('int8')

        return basket_encoded

    def _build_tid_structure(self, verbose: bool = True) -> TID:
        """Build TID structure for WDPA."""
        # Create mapping from product_name to product_id for consistency
        product_mapping = self.market_basket[['product_name', 'product_id']].drop_duplicates()
        product_to_id = dict(zip(product_mapping['product_name'], product_mapping['product_id']))

        # Prepare data for TID: order_id and product_id
        tid_data = self.market_basket[['order_id', 'product_id']].copy()

        # Build TID structure
        tid = TID()
        tid.build_from_orders(tid_data, order_col='order_id', product_col='product_id', verbose=False)

        return tid

    def _create_transactions(self, verbose: bool = True) -> list[list[int]]:
        """
        Create horizontal transactions (sorted unique product_id per order).
        """
        grouped = self.market_basket.groupby('order_id')['product_id'].apply(lambda s: sorted(set(map(int, s.values)))).reset_index()
        return grouped['product_id'].tolist()

    def get_statistics(self) -> Dict:
        """Get detailed statistics about the loaded data."""
        if self.market_basket is None:
            return {}

        return {
            'raw_data': {
                'total_records': len(self.market_basket),
                'unique_orders': self.market_basket['order_id'].nunique(),
                'unique_users': self.market_basket['user_id'].nunique(),
                'unique_products': self.market_basket['product_id'].nunique(),
                'avg_items_per_order': len(self.market_basket) / self.market_basket['order_id'].nunique()
            },
            'basket_matrix': {
                'shape': self.basket_encoded.shape if self.basket_encoded is not None else None,
                'density': (self.basket_encoded.sum().sum() / (self.basket_encoded.shape[0] * self.basket_encoded.shape[1])) if self.basket_encoded is not None else None
            },
            'tid_structure': self.tid.get_statistics() if self.tid is not None else {},
            'load_times': {
                'load_time': self.load_time,
                'transform_time': self.transform_time,
                'tid_build_time': self.tid_build_time,
                'total_time': self.load_time + self.transform_time + self.tid_build_time
            }
        }


def load_data_for_benchmark(config: Dict, algorithms_config: Dict = None, verbose: bool = True) -> Dict:
    """
    Convenience function to load data for benchmark.

    Args:
        config: Dataset configuration
        algorithms_config: Optional algorithms configuration to determine what to load
        verbose: Print progress

    Returns:
        Dictionary with all prepared data
    """
    loader = UnifiedDataLoader(config, algorithms_config=algorithms_config)
    return loader.load(verbose=verbose)


if __name__ == "__main__":
    # Test the loader
    import json

    print("\nTesting Unified Data Loader")
    print("="*80)

    # Load config
    with open('configs/benchmark_config.json', 'r') as f:
        config = json.load(f)

    # Load data
    data = load_data_for_benchmark(config['dataset'], verbose=True)

    print("\nData Summary:")
    print(f"  Basket matrix shape: {data['basket_encoded'].shape}")
    print(f"  TID transactions: {data['tid'].total_transactions:,}")
    print(f"  TID unique items: {len(data['tid'].item_to_tids):,}")

    print("\nVerifying consistency:")
    print(f"  Orders in basket: {data['basket_encoded'].shape[0]:,}")
    print(f"  Orders in TID: {data['tid'].total_transactions:,}")
    print(f"  Match: {data['basket_encoded'].shape[0] == data['tid'].total_transactions}")
