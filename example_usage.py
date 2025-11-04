#!/usr/bin/env python3
"""
Example usage of the Traditional Single-Threaded Apriori Algorithm
COS 781 Data Mining Project - Group 16

This script demonstrates how to use the traditional Apriori implementation
for association rule mining on the Instacart dataset.
"""

import pandas as pd
from src.apriori import (
    load_instacart_data,
    transform_basket, 
    run_apriori_algorithm
)

def example_usage():
    """
    Example demonstrating how to use the traditional Apriori functions.
    """
    print("Traditional Single-Threaded Apriori - Example Usage")
    print("=" * 60)
    
    try:
        # Load the Instacart dataset
        print("1. Loading Instacart dataset...")
        market_basket = load_instacart_data()
        print(f"Loaded dataset with shape: {market_basket.shape}")
        
        # Transform data for Apriori
        print("\n2. Transforming data...")
        basket_encoded, basket = transform_basket(market_basket)
        print(f"Transformed basket shape: {basket_encoded.shape}")
        
        # Run Apriori algorithm with different support thresholds
        print("\n3. Running Apriori algorithm...")
        support_thresholds = [0.15, 0.2, 0.25]
        
        for min_support in support_thresholds:
            print(f"\n--- Support Threshold: {min_support} ---")
            frequent_itemsets, rules, sorted_rules = run_apriori_algorithm(
                basket_encoded, 
                min_support=min_support, 
                min_threshold=1.0
            )
            
            print(f"Results: {len(frequent_itemsets)} frequent itemsets, {len(rules)} rules")
            
            if len(sorted_rules) > 0:
                print("Top 3 association rules:")
                top_rules = sorted_rules.head(3)
                for idx, rule in top_rules.iterrows():
                    print(f"  {list(rule['antecedents'])} -> {list(rule['consequents'])} "
                          f"(support: {rule['support']:.3f}, confidence: {rule['confidence']:.3f}, lift: {rule['lift']:.3f})")
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("Run 'python main.py' for the complete analysis with JSON output.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install -r requirements.txt")
        print("\nAnd ensure the Instacart dataset is in the 'data/' directory.")

if __name__ == "__main__":
    example_usage()
