import pandas as pd
import numpy as np
import json
import os
from warnings import filterwarnings
from mlxtend.frequent_patterns import apriori, association_rules
filterwarnings("ignore")


def load_instacart_data(sample_size=5000, max_items=1000, chunk_size=10000):
    """
    Load and prepare a sample of the Instacart Market Basket Analysis dataset with memory optimization.
    
    Args:
        sample_size (int): Number of transactions to sample (default: 5,000)
        max_items (int): Maximum number of unique items to keep (default: 1,000)
        chunk_size (int): Chunk size for processing large files (default: 10,000)
    
    Returns:
        pd.DataFrame: Prepared market basket data
    """
    print("Loading Instacart dataset with memory optimization...")
    
    # Load the main data files
    orders = pd.read_csv('data/orders.csv')
    order_products = pd.read_csv('data/order_products__prior.csv')
    products = pd.read_csv('data/products.csv')
    
    # Handle sampling vs full dataset
    if sample_size is not None:
        print(f"Sampling {sample_size} orders from {len(orders)} total orders...")
        sample_orders = orders.sample(n=min(sample_size, len(orders)), random_state=42)
        sample_order_ids = set(sample_orders['order_id'])
        
        # Filter order_products to only include sampled orders
        order_products_sample = order_products[order_products['order_id'].isin(sample_order_ids)]
        
        # Merge the datasets to create a complete transaction dataset
        market_basket = order_products_sample.merge(products, on='product_id').merge(orders, on='order_id')
        total_orders = len(sample_orders)
    else:
        print("Using FULL dataset (all orders)...")
        print("Warning: Full dataset is very large, using chunked processing...")
        # Use chunked processing for full dataset to avoid memory issues
        market_basket = load_large_dataset_chunked(
            sample_size=None, 
            max_items=max_items, 
            chunk_size=chunk_size
        )
        total_orders = len(orders)
    
    # Select relevant columns for market basket analysis
    market_basket = market_basket[['order_id', 'user_id', 'product_id', 'product_name']]
    
    # Memory optimization: Keep only most frequent items
    if max_items and len(market_basket['product_name'].unique()) > max_items:
        print(f"Filtering to top {max_items} most frequent items...")
        item_counts = market_basket['product_name'].value_counts()
        top_items = set(item_counts.head(max_items).index)
        market_basket = market_basket[market_basket['product_name'].isin(top_items)]
        print(f"Reduced to {len(market_basket)} transactions with {len(top_items)} unique items")
    
    print(f"Loaded {len(market_basket)} transactions from {total_orders} orders")
    return market_basket


def transform_basket(market_basket):
    """
    Transform the Instacart data into a format suitable for Apriori algorithm.
    
    Args:
        market_basket (pd.DataFrame): Transaction data with columns:
                                    ['order_id', 'user_id', 'product_id', 'product_name']
    
    Returns:
        tuple: (basket_encoded, basket) - encoded binary basket and original grouped basket
    """
    print("Starting basket transformation...")
    
    # Step 1: Create a pivot table with orders as rows and products as columns
    basket = market_basket.groupby(['order_id', 'product_name']).size().unstack(fill_value=0)
    
    print(f"Basket shape after grouping: {basket.shape}")
    
    # Step 2: Convert counts to binary (0/1) format
    # Any count > 0 becomes 1, 0 stays 0
    basket_encoded = (basket > 0).astype('int8')  # Use int8 to save memory
    
    print(f"Basket shape after encoding: {basket_encoded.shape}")
    
    return basket_encoded, basket


def run_apriori_algorithm(basket_encoded, min_support=0.01, min_threshold=1.0):
    """
    Run the traditional single-threaded Apriori algorithm to find frequent itemsets and association rules.
    
    Args:
        basket_encoded (pd.DataFrame): Binary encoded basket data
        min_support (float): Minimum support threshold (default: 0.01)
        min_threshold (float): Minimum threshold for association rules (default: 1.0)
    
    Returns:
        tuple: (frequent_itemsets, rules, sorted_rules)
    """
    print(f"Running Apriori algorithm with min_support={min_support}...")
    
    # Step 1: Find frequent itemsets using traditional Apriori
    frequent_itemsets = apriori(basket_encoded,
                               min_support=min_support, 
                               use_colnames=True)
    
    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    
    # Step 2: Generate association rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, 
                                metric="lift",
                                min_threshold=min_threshold)
        
        print(f"Generated {len(rules)} association rules")
        
        # Step 3: Sort rules by lift and support
        sorted_rules = rules.sort_values(['lift', 'support'], ascending=[False, False])
    
        if len(sorted_rules) > 0:
            print("Top 5 rules by lift:")
            print(sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    else:
        print("No frequent itemsets found with current support threshold")
        rules = pd.DataFrame()
        sorted_rules = pd.DataFrame()
    
    return frequent_itemsets, rules, sorted_rules


def save_results_to_json(results_metadata, tag: str = None, results_dir: str = 'results', suffix: str = ""):
    """
    Save analysis results to JSON files.
    
    Args:
        results_metadata (dict): Complete results metadata to save
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Optional tag prefix for filenames
    prefix = f"{tag}_" if tag else ""
    suffix = suffix or ""

    # Save complete results
    complete_path = os.path.join(results_dir, f"{prefix}complete_analysis{suffix}.json")
    with open(complete_path, 'w') as f:
        json.dump(results_metadata, f, indent=2, default=str)
    
    # Save individual result files for each support threshold
    for support_key, support_data in results_metadata['results'].items():
        indiv_path = os.path.join(results_dir, f"{prefix}apriori_results_{support_key}{suffix}.json")
        with open(indiv_path, 'w') as f:
            json.dump(support_data, f, indent=2, default=str)
    
    # Save summary results
    summary = {
        "experiment_info": results_metadata["experiment_info"],
        "dataset_info": results_metadata["dataset_info"],
        "summary": {
            support_key: {
                "min_support": data["min_support"],
                "frequent_itemsets_count": data["frequent_itemsets_count"],
                "rules_count": data["rules_count"]
            }
            for support_key, data in results_metadata["results"].items()
        }
    }
    
    summary_path = os.path.join(results_dir, f"{prefix}analysis_summary{suffix}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("Results saved to:")
    print(f"- {complete_path} (complete results)")
    print(f"- {summary_path} (summary)")
    print(f"- {results_dir}/{prefix}apriori_results_support_*{suffix}.json (individual support thresholds)")


def load_large_dataset_chunked(sample_size=20000, max_items=2000, chunk_size=5000):
    """
    Load large dataset using chunked processing to handle memory constraints.
    
    Args:
        sample_size (int): Number of transactions to sample (None for full dataset)
        max_items (int): Maximum number of unique items to keep (default: 2,000)
        chunk_size (int): Chunk size for processing (default: 5,000)
    
    Returns:
        pd.DataFrame: Prepared market basket data
    """
    print("Loading large dataset with chunked processing...")
    
    # Load orders and products (these are smaller)
    orders = pd.read_csv('data/orders.csv')
    products = pd.read_csv('data/products.csv')
    
    # Handle sampling vs full dataset
    if sample_size is not None:
        print(f"Sampling {sample_size} orders from {len(orders)} total orders...")
        sample_orders = orders.sample(n=min(sample_size, len(orders)), random_state=42)
        sample_order_ids = set(sample_orders['order_id'])
        total_orders = len(sample_orders)
    else:
        print("Processing FULL dataset (all orders)...")
        sample_order_ids = None  # Process all orders
        total_orders = len(orders)
    
    # Process order_products in chunks
    chunks = []
    chunk_count = 0
    total_processed = 0
    
    for chunk in pd.read_csv('data/order_products__prior.csv', chunksize=chunk_size):
        # Filter chunk based on sampling
        if sample_order_ids is not None:
            chunk_filtered = chunk[chunk['order_id'].isin(sample_order_ids)]
        else:
            chunk_filtered = chunk  # Use all data for full dataset
        
        if len(chunk_filtered) > 0:
            chunks.append(chunk_filtered)
            chunk_count += 1
            total_processed += len(chunk_filtered)
            print(f"Processed chunk {chunk_count}, found {len(chunk_filtered)} transactions (total: {total_processed:,})")
            
            # Stop if we have enough data (only for sampled dataset)
            if sample_size is not None and total_processed >= sample_size * 2:  # 2x buffer
                print(f"Stopping early - collected {total_processed:,} transactions (target: {sample_size:,})")
                break
    
    print(f"Combining {len(chunks)} chunks...")
    # Combine chunks
    order_products_sample = pd.concat(chunks, ignore_index=True)
    
    print("Merging with products and orders...")
    # Merge with products and orders
    market_basket = order_products_sample.merge(products, on='product_id').merge(orders, on='order_id')
    market_basket = market_basket[['order_id', 'user_id', 'product_id', 'product_name']]
    
    # Filter to most frequent items
    if max_items and len(market_basket['product_name'].unique()) > max_items:
        print(f"Filtering to top {max_items} most frequent items...")
        item_counts = market_basket['product_name'].value_counts()
        top_items = set(item_counts.head(max_items).index)
        market_basket = market_basket[market_basket['product_name'].isin(top_items)]
    
    print(f"Loaded {len(market_basket)} transactions with {market_basket['product_name'].nunique()} unique items")
    return market_basket


def run_scalability_test():
    """
    Test different dataset sizes to find optimal parameters.
    """
    print("=" * 60)
    print("SCALABILITY TEST - Finding Optimal Dataset Size")
    print("=" * 60)
    
    test_configs = [
        {"sample_size": 5000, "max_items": 500, "name": "Small"},
        {"sample_size": 10000, "max_items": 1000, "name": "Medium"},
        {"sample_size": 20000, "max_items": 2000, "name": "Large"},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} Configuration ---")
        print(f"Sample size: {config['sample_size']}, Max items: {config['max_items']}")
        
        try:
            # Load data
            market_basket = load_instacart_data(
                sample_size=config['sample_size'], 
                max_items=config['max_items']
            )
            
            # Transform data
            basket_encoded, basket = transform_basket(market_basket)
            
            # Test Apriori with different support thresholds
            support_thresholds = [0.1, 0.15, 0.2]
            config_results = {}
            
            for min_support in support_thresholds:
                try:
                    frequent_itemsets, rules, sorted_rules = run_apriori_algorithm(
                        basket_encoded, min_support=min_support, min_threshold=1.0
                    )
                    config_results[min_support] = {
                        'frequent_itemsets': len(frequent_itemsets),
                        'rules': len(rules),
                        'success': True
                    }
                except Exception as e:
                    config_results[min_support] = {
                        'frequent_itemsets': 0,
                        'rules': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            results[config['name']] = {
                'dataset_shape': market_basket.shape,
                'basket_shape': basket_encoded.shape,
                'results': config_results
            }
            
            print(f"[SUCCESS] {config['name']} configuration completed successfully")
            
        except Exception as e:
            print(f"[FAILED] {config['name']} configuration failed: {e}")
            results[config['name']] = {'error': str(e)}
    
    # Save scalability results
    with open('results/scalability_test.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nScalability test results saved to results/scalability_test.json")
    return results

