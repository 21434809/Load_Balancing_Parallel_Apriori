import pandas as pd
import numpy as np
import os
from tid import TID, mine_frequent_itemsets
import time

def load_instacart_data(data_dir="../data", verbose=True):
    """
    Load the Instacart dataset from your downloaded data directory.
    
    Args:
        data_dir: Directory where you downloaded the data
        verbose: Print loading info
        
    Returns:
        Dictionary with all dataframes
    """
    if verbose:
        print("\n" + "="*60)
        print("Loading Instacart Dataset")
        print("="*60)
        print(f"Data directory: {data_dir}")
    
    data = {}
    
    # Load main files
    files_to_load = {
        'orders': 'orders.csv',
        'order_products_prior': 'order_products__prior.csv',
        'order_products_train': 'order_products__train.csv',
        'products': 'products.csv',
        'aisles': 'aisles.csv',
        'departments': 'departments.csv'
    }
    
    for name, filename in files_to_load.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            if verbose:
                print(f"\nLoading {filename}...")
            data[name] = pd.read_csv(filepath)
            if verbose:
                print(f"Loaded {len(data[name]):,} rows")
        else:
            if verbose:
                print(f"File not found: {filename}")
    
    if verbose:
        print("\n" + "="*60)
        print("Dataset loaded successfully!")
        print("="*60)
    
    return data


def get_transaction_dataframe(data, dataset='prior', max_orders=None, verbose=True):
    """
    Extract transaction data in the format needed for TID.
    
    Args:
        data: Dictionary with loaded dataframes
        dataset: 'prior' or 'train' or 'both'
        max_orders: Limit to N orders (for testing)
        verbose: Print info
        
    Returns:
        DataFrame with columns [order_id, product_id]
    """
    if verbose:
        print("\n" + "="*60)
        print("Preparing Transaction Data for TID")
        print("="*60)
    
    # Select dataset
    if dataset == 'prior':
        df = data['order_products_prior'].copy()
    elif dataset == 'train':
        df = data['order_products_train'].copy()
    elif dataset == 'both':
        df = pd.concat([
            data['order_products_prior'],
            data['order_products_train']
        ], ignore_index=True)
    else:
        raise ValueError("dataset must be 'prior', 'train', or 'both'")
    
    if verbose:
        print(f"\nDataset: {dataset}")
        print(f"Initial records: {len(df):,}")
    
    # Limit orders if requested
    if max_orders is not None:
        unique_orders = df['order_id'].unique()[:max_orders]
        df = df[df['order_id'].isin(unique_orders)]
        if verbose:
            print(f"Limited to first {max_orders:,} orders")
            print(f"Records after limiting: {len(df):,}")
    
    # Keep only needed columns
    transaction_df = df[['order_id', 'product_id']].copy()
    
    # Get statistics
    n_orders = transaction_df['order_id'].nunique()
    n_products = transaction_df['product_id'].nunique()
    avg_items = len(transaction_df) / n_orders
    
    if verbose:
        print(f"\nTransaction Data Ready:")
        print(f"  Orders: {n_orders:,}")
        print(f"  Unique products: {n_products:,}")
        print(f"  Total records: {len(transaction_df):,}")
        print(f"  Avg items per order: {avg_items:.2f}")
        print(f"  Data sparsity: {(1 - len(transaction_df)/(n_orders*n_products))*100:.2f}%")
        print("="*60)
    
    return transaction_df


def get_product_names(data, product_ids):
    """
    Get product names for given product IDs.
    
    Args:
        data: Dictionary with loaded dataframes
        product_ids: List of product IDs
        
    Returns:
        Dictionary mapping product_id -> product_name
    """
    products_df = data['products']
    result = {}
    
    for pid in product_ids:
        row = products_df[products_df['product_id'] == pid]
        if not row.empty:
            result[pid] = row.iloc[0]['product_name']
        else:
            result[pid] = f"Product_{pid}"
    
    return result


def run_tid_analysis(transaction_df, min_support=0.002, max_k=5, data=None, verbose=True):
    """
    Run complete TID analysis on transaction data.
    
    Args:
        transaction_df: DataFrame with [order_id, product_id]
        min_support: Minimum support threshold
        max_k: Maximum itemset size
        data: Original data dict (for product names)
        verbose: Print progress
        
    Returns:
        Tuple of (tid, frequent_itemsets)
    """
    # Build TID structure
    tid = TID()
    tid.build_from_orders(transaction_df, verbose=verbose)
    
    # Print TID statistics
    if verbose:
        stats = tid.get_statistics()
        print("\nTID Structure Statistics:")
        print("-"*60)
        print(f"  Total transactions: {stats['total_transactions']:,}")
        print(f"  Unique items: {stats['unique_items']:,}")
        print(f"  Build time: {stats['build_time_seconds']:.3f} seconds")
        print(f"  Avg TID length: {stats['avg_tid_length']:.1f}")
        print(f"  Median TID length: {stats['median_tid_length']:.1f}")
        print(f"  Max TID length: {stats['max_tid_length']:.0f}")
        print("-"*60)
    
    # Mine frequent itemsets
    frequent = mine_frequent_itemsets(tid, min_support=min_support, max_k=max_k, verbose=verbose)
    
    # Show examples with product names if available
    if verbose and data is not None and frequent:
        print("\n" + "="*60)
        print("Example Frequent Itemsets with Product Names")
        print("="*60)
        
        # Show some 1-itemsets
        if 1 in frequent:
            print("\nTop 10 Frequent Products:")
            for i, (itemset, count) in enumerate(frequent[1][:10], 1):
                product_id = list(itemset)[0]
                name = get_product_names(data, [product_id])[product_id]
                support = count / tid.total_transactions
                print(f"{i:2d}. {name:40s} (ID:{product_id:5d}) " f"count={count:6,} ({support*100:5.2f}%)")
        
        # Show some 2-itemsets
        if 2 in frequent and len(frequent[2]) > 0:
            print("\nTop 10 Frequent Product Pairs:")
            for i, (itemset, count) in enumerate(frequent[2][:10], 1):
                items = sorted(list(itemset))
                names = get_product_names(data, items)
                support = count / tid.total_transactions
                print(f"{i:2d}. {names[items[0]]:30s} + {names[items[1]]:30s}")
                print(f"    count={count:6,} ({support*100:5.2f}%)")
        
        # Show some 3-itemsets if they exist
        if 3 in frequent and len(frequent[3]) > 0:
            print("\nTop 5 Frequent Product Triplets:")
            for i, (itemset, count) in enumerate(frequent[3][:5], 1):
                items = sorted(list(itemset))
                names = get_product_names(data, items)
                support = count / tid.total_transactions
                print(f"{i}. {names[items[0]]} + {names[items[1]]} + {names[items[2]]}")
                print(f"   count={count:,} ({support*100:.2f}%)")
    
    return tid, frequent


def main():
    """
    Main function to run YOUR TID implementation on Instacart data.
    """
    # Check if data directory exists
    data_dir = "../data"
    if not os.path.exists(data_dir):
        print(f"\nData directory not found: {data_dir}")
        return
    
    # Load data
    data = load_instacart_data(data_dir)
    
    if not data:
        print("\nNo data loaded. Please check your data directory.")
        return
    
    # Print dataset overview
    print("\n" + "="*60)
    print("Dataset Overview")
    print("="*60)
    print(f"Total orders: {len(data['orders']):,}")
    print(f"Unique users: {data['orders']['user_id'].nunique():,}")
    print(f"Total products: {len(data['products']):,}")
    print(f"Prior order products: {len(data['order_products_prior']):,}")
    if 'order_products_train' in data:
        print(f"Train order products: {len(data['order_products_train']):,}")
    
    transaction_df_full = get_transaction_dataframe(
        data, 
        dataset='prior',
        max_orders=None,  # Use ALL orders
        verbose=True
    )
    
    tid_full, frequent_full = run_tid_analysis(
        transaction_df_full,
        min_support=0.002,  # 0.2% support (from paper)
        max_k=5,
        data=data,
        verbose=True
    )
    
    # Save for later use
    tid_full.save('tid_full.pkl')


if __name__ == "__main__":
    main()