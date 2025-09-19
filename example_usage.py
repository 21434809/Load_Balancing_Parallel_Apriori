#!/usr/bin/env python3
"""
Example usage of the Apriori functions for market basket analysis.

This script demonstrates how to use the functions created for the notebook solutions.
"""

import pandas as pd
from src.apriori import (
    transform_basket, 
    run_apriori_algorithm, 
    frequently_bought_together,
    calculate_memory_requirement,
    run_complete_analysis
)

def example_usage():
    """
    Example demonstrating how to use the Apriori functions.
    """
    print("Market Basket Analysis - Example Usage")
    print("=" * 50)
    
    # Load sample data (you would replace this with your actual data)
    HYPERMARKET_URL = "https://media.githubusercontent.com/media/dsfsi/dsfsi-datasets/refs/heads/master/data/cos781/hypermarket_dataset.csv"
    
    try:
        market_basket = pd.read_csv(HYPERMARKET_URL)
        print(f"Loaded dataset with shape: {market_basket.shape}")
        
        # Example 1: Individual function usage
        print("\n1. Individual Function Usage:")
        print("-" * 30)
        
        # Transform basket
        basket_encoded, basket = transform_basket(market_basket)
        
        # Run Apriori algorithm
        frequent_itemsets, rules, sorted_rules = run_apriori_algorithm(basket_encoded)
        
        # Get recommendations for a specific item
        if 'whole milk' in basket_encoded.columns:
            recommendations = frequently_bought_together(basket_encoded, 'whole milk')
            print(f"\nRecommendations for 'whole milk': {recommendations}")
        
        # Calculate memory requirements
        memory_req = calculate_memory_requirement(100000, 50000000)
        print(f"\nMemory requirement for N=100,000, M=50,000,000: {memory_req:,} bytes")
        
        print("\n2. Complete Analysis Pipeline:")
        print("-" * 30)
        
        # Run complete analysis
        results = run_complete_analysis(market_basket)
        
        print(f"\nAnalysis completed! Results contain:")
        print(f"- Basket encoded shape: {results['basket_encoded'].shape}")
        print(f"- Frequent itemsets: {len(results['frequent_itemsets'])}")
        print(f"- Association rules: {len(results['rules'])}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure you have internet connection and required packages installed.")
        print("Install packages with: pip install -r requirements.txt")

def demonstrate_notebook_solutions():
    """
    Demonstrate the exact solutions for the notebook cells.
    """
    print("\n" + "=" * 60)
    print("NOTEBOOK SOLUTION DEMONSTRATIONS")
    print("=" * 60)
    
    # Solution 1: Basket Transformation
    print("\n🔧 SOLUTION 1: Basket Transformation")
    print("-" * 40)
    print("""
    # Complete the basket transformation
    basket_encoded, basket = transform_basket(market_basket)
    
    # The function handles:
    # - Grouping by member and item
    # - Counting transactions
    # - Converting to binary format (0/1)
    """)
    
    # Solution 2: Apriori Algorithm
    print("\n🔧 SOLUTION 2: Apriori Algorithm")
    print("-" * 40)
    print("""
    # Complete the Apriori algorithm implementation
    frequent_itemsets, rules, sorted_rules = run_apriori_algorithm(
        basket_encoded, 
        min_support=0.01, 
        min_threshold=1.0
    )
    
    # The function handles:
    # - Finding frequent itemsets
    # - Generating association rules
    # - Sorting by lift and support
    """)
    
    # Solution 3: Recommendation Function
    print("\n🔧 SOLUTION 3: Recommendation Function")
    print("-" * 40)
    print("""
    # Complete the recommendation function
    recommendations = frequently_bought_together(
        basket_encoded, 
        'whole milk', 
        min_support=0.15, 
        min_threshold=1
    )
    
    # The function handles:
    # - Filtering customers who bought the target item
    # - Running Apriori on filtered data
    # - Extracting top recommendations
    """)
    
    # Solution 4: Memory Calculation
    print("\n🔧 SOLUTION 4: Memory Calculation")
    print("-" * 40)
    print("""
    # Complete the memory calculation
    memory_req = calculate_memory_requirement(N=100000, M=50000000)
    
    # The function handles:
    # - Triangular matrix calculation
    # - Hash table calculation
    # - Returning minimum of both methods
    """)

if __name__ == "__main__":
    example_usage()
    demonstrate_notebook_solutions()
