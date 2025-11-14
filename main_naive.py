#!/usr/bin/env python3
"""
NaIve Parallel Apriori Algorithm (ProcessPoolExecutor)

Runs a simple parallel Apriori baseline using static partitioning and boolean
intersections. Outputs JSON in the same structure as the traditional baseline.
"""

import json
import time
from datetime import datetime

import pandas as pd

from src.apriori import (
    load_instacart_data,
    transform_basket,
    save_results_to_json
)
from src.naive_parallel_apriori import run_naive_parallel_apriori
from mlxtend.frequent_patterns import association_rules


def load_config():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("Configuration loaded successfully!")
        return config
    except FileNotFoundError:
        print("Warning: config.json not found, using default configuration")
        return {
            "dataset_config": {"sample_size": 5000, "max_items": 1000, "chunk_size": 5000},
            "apriori_config": {"support_thresholds": [0.0015, 0.002, 0.0025, 0.003], "min_threshold": 1.0},
            "parallel_config": {"num_workers": 4},
            "output_config": {"save_individual_results": True, "save_summary": True, "results_directory": "results"}
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing config.json: {e}")
        print("Using default configuration")
        return {
            "dataset_config": {"sample_size": 5000, "max_items": 1000, "chunk_size": 5000},
            "apriori_config": {"support_thresholds": [0.0015, 0.002, 0.0025, 0.003], "min_threshold": 1.0},
            "parallel_config": {"num_workers": 4},
            "output_config": {"save_individual_results": True, "save_summary": True, "results_directory": "results"}
        }


def main():
    total_start_time = time.time()

    print("="*80)
    print("NAIVE PARALLEL APRIORI (ProcessPoolExecutor)")
    print("="*80)

    config_start_time = time.time()
    config = load_config()
    config_time = time.time() - config_start_time

    dataset_config = config.get("dataset_config", {})
    apriori_config = config.get("apriori_config", {})
    output_config = config.get("output_config", {})
    import os
    parallel_config = config.get("parallel_config", {})
    auto = bool(parallel_config.get("auto", False))
    if auto:
        num_workers = max(1, os.cpu_count() or 1)
    else:
        num_workers = int(parallel_config.get("num_workers", 4))

    print(f"\nConfiguration:")
    if dataset_config.get('use_full_dataset', False):
        print(f"- Dataset: FULL dataset (no sampling)")
    else:
        print(f"- Dataset: Sampled ({dataset_config.get('sample_size', 5000):,} orders)")
    print(f"- Max items: {dataset_config.get('max_items', 1000):,} items")
    print(f"- Support thresholds: {apriori_config.get('support_thresholds', [])}")
    print(f"- Num workers: {num_workers}{' (auto)' if auto else ''}")
    print(f"- Configuration load time: {config_time:.4f} seconds")

    print("\n1. Loading Instacart Market Basket Analysis dataset...")
    data_load_start_time = time.time()
    if dataset_config.get('use_full_dataset', False):
        print("Using FULL dataset (no sampling)...")
        market_basket = load_instacart_data(
            sample_size=None,
            max_items=dataset_config.get('max_items', 1000)
        )
    else:
        print(f"Using sampled dataset ({dataset_config.get('sample_size', 5000):,} orders)...")
        market_basket = load_instacart_data(
            sample_size=dataset_config.get('sample_size', 5000),
            max_items=dataset_config.get('max_items', 1000)
        )

    data_load_time = time.time() - data_load_start_time

    print(f"Dataset loaded successfully!")
    print(f"Shape: {market_basket.shape}")
    print(f"Columns: {market_basket.columns.tolist()}")
    print(f"Data loading time: {data_load_time:.4f} seconds")

    print(f"\nDataset Statistics:")
    print(f"- Total transactions: {len(market_basket)}")
    print(f"- Unique users: {market_basket['user_id'].nunique()}")
    print(f"- Unique products: {market_basket['product_id'].nunique()}")
    print(f"- Unique orders: {market_basket['order_id'].nunique()}")

    print("\n2. Transforming data for Apriori algorithm...")
    transform_start_time = time.time()
    basket_encoded, basket = transform_basket(market_basket)
    transform_time = time.time() - transform_start_time
    print(f"Basket transformation completed!")
    print(f"Encoded basket shape: {basket_encoded.shape}")
    print(f"Data transformation time: {transform_time:.4f} seconds")

    print("\n3. Running NaIve Parallel Apriori...")
    support_thresholds = apriori_config.get('support_thresholds', [])
    min_threshold = apriori_config.get('min_threshold', 1.0)
    all_results = {}
    apriori_total_time = 0

    for min_support in support_thresholds:
        print(f"\nRunning NaIve Parallel Apriori with min_support = {min_support}")
        print("-" * 50)

        apriori_start_time = time.time()
        frequent_itemsets = run_naive_parallel_apriori(basket_encoded, min_support=min_support, num_workers=num_workers)

        # Association rules (reuse mlxtend for consistency)
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
            sorted_rules = rules.sort_values(['lift', 'support'], ascending=[False, False])
        else:
            rules = pd.DataFrame()
            sorted_rules = pd.DataFrame()

        apriori_time = time.time() - apriori_start_time
        apriori_total_time += apriori_time

        all_results[f"support_{min_support}"] = {
            "min_support": min_support,
            "frequent_itemsets_count": len(frequent_itemsets),
            "rules_count": len(rules),
            "frequent_itemsets": frequent_itemsets.to_dict('records') if len(frequent_itemsets) > 0 else [],
            "top_rules": sorted_rules.head(20).to_dict('records') if len(sorted_rules) > 0 else [],
            "execution_time_seconds": apriori_time
        }

        print(f"Found {len(frequent_itemsets)} frequent itemsets")
        print(f"Generated {len(rules)} association rules")
        print(f"Execution time: {apriori_time:.4f} seconds")

    total_time = time.time() - total_start_time

    results_metadata = {
        "experiment_info": {
            "algorithm": "NaIve Parallel Apriori (ProcessPoolExecutor)",
            "dataset": "Instacart Market Basket Analysis",
            "timestamp": datetime.now().isoformat(),
            "configuration": config
        },
        "dataset_info": {
            "total_transactions": len(market_basket),
            "unique_users": market_basket['user_id'].nunique(),
            "unique_products": market_basket['product_id'].nunique(),
            "unique_orders": market_basket['order_id'].nunique(),
            "basket_shape": basket_encoded.shape
        },
        "performance_metrics": {
            "total_execution_time_seconds": total_time,
            "configuration_load_time_seconds": config_time,
            "data_loading_time_seconds": data_load_time,
            "data_transformation_time_seconds": transform_time,
            "apriori_total_time_seconds": apriori_total_time,
            "results_saving_time_seconds": 0,
            "throughput_transactions_per_second": len(market_basket) / total_time,
            "throughput_orders_per_second": dataset_config.get('sample_size', 0) / total_time if total_time > 0 else 0
        },
        "results": all_results
    }

    print("\n4. Saving results to JSON files...")
    save_start_time = time.time()
    suffix = "_full" if dataset_config.get('use_full_dataset', False) else ""
    save_results_to_json(
        results_metadata,
        tag='naive',
        results_dir=output_config.get('results_directory', 'results'),
        suffix=suffix
    )
    save_time = time.time() - save_start_time

    results_metadata["performance_metrics"]["results_saving_time_seconds"] = save_time

    print(f"Results saving time: {save_time:.4f} seconds")
    print("\n" + "="*80)
    print("NAIVE PARALLEL ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Results saved to '{output_config.get('results_directory', 'results')}/' directory")
    print("="*80)


if __name__ == "__main__":
    main()


