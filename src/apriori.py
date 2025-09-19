import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
from mlxtend.frequent_patterns import apriori, association_rules
filterwarnings("ignore")


def transform_basket(market_basket):
    """
    Complete the basket transformation by grouping transactions and encoding to binary format.
    
    Args:
        market_basket (pd.DataFrame): Original market basket data with columns:
                                    ['Member_number', 'itemDescription', 'Date']
    
    Returns:
        tuple: (basket_encoded, basket) - encoded binary basket and original grouped basket
    """
    print("Starting basket transformation...")
    
    # Step 1: Group by member and item, count transactions
    basket = (market_basket.groupby(['Member_number', 'itemDescription'])['Date']
              .count()
              .unstack()
              .reset_index()
              .fillna(0)
              .set_index('Member_number'))
    
    print("Basket shape after grouping:", basket.shape)
    print("Sample basket before encoding:")
    print(basket.iloc[:3, :5])  # Show first 3 rows, 5 columns
    
    # Step 2: Convert counts to binary (0/1) format
    def encode_units(x):
        if x <= 0:      # No purchase
            return 0
        if x >= 1:      # Purchase made
            return 1
    
    # Apply the encoding function
    basket_encoded = basket.map(encode_units)
    
    print("\nBasket shape after encoding:", basket_encoded.shape)
    print("Sample basket after encoding:")
    print(basket_encoded.iloc[:3, :5])
    
    return basket_encoded, basket


def run_apriori_algorithm(basket_encoded, min_support=0.01, min_threshold=1.0):
    """
    Complete the Apriori algorithm implementation to find frequent itemsets and association rules.
    
    Args:
        basket_encoded (pd.DataFrame): Binary encoded basket data
        min_support (float): Minimum support threshold (default: 0.01)
        min_threshold (float): Minimum threshold for association rules (default: 1.0)
    
    Returns:
        tuple: (frequent_itemsets, rules, sorted_rules)
    """
    print("Running Apriori algorithm...")
    
    # Step 1: Find frequent itemsets
    # Support threshold of 0.01 means item(s) must appear in at least 1% of transactions
    frequent_itemsets = apriori(basket_encoded,
                               min_support=min_support, 
                               use_colnames=True)
    
    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    print("Sample frequent itemsets:")
    print(frequent_itemsets.head())
    
    # Step 2: Generate association rules
    # Rules help us understand: "If someone buys X, they also buy Y"
    rules = association_rules(frequent_itemsets, 
                            metric="lift",      # Using lift metric
                            min_threshold=min_threshold)
    
    print(f"\nGenerated {len(rules)} association rules")
    
    # Step 3: Sort rules by lift and support to find the most interesting rules
    sorted_rules = rules.sort_values(['lift', 'support'], ascending=[False, False])
    
    print("\nTop 5 rules by lift:")
    print(sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    
    return frequent_itemsets, rules, sorted_rules


def frequently_bought_together(basket_encoded, item, min_support=0.15, min_threshold=1):
    """
    Find items frequently bought together with the given item.
    
    Args:
        basket_encoded (pd.DataFrame): Binary encoded basket data
        item (str): The target item to find recommendations for
        min_support (float): Minimum support threshold (default: 0.15)
        min_threshold (float): Minimum threshold for rules (default: 1)
    
    Returns:
        list: Top 6 unique consequents (recommendations)
    """
    print(f'Items frequently bought together with {item}')
    
    # Step 1: Filter the basket to only include customers who bought the target item
    item_df = basket_encoded.loc[basket_encoded[item] == 1]
    
    # Step 2: Apply Apriori algorithm on the filtered data
    frequent_itemsets = apriori(item_df, 
                               min_support=min_support, 
                               use_colnames=True)
    
    # Step 3: Generate association rules
    item_rules = association_rules(frequent_itemsets, 
                                  metric="lift", 
                                  min_threshold=min_threshold)
    
    # Step 4: Sort rules and extract recommendations
    item_rules_sorted = item_rules.sort_values(['lift', 'confidence'], 
                                             ascending=[False, False])
    
    # Return top 6 unique consequents
    return item_rules_sorted['consequents'].unique()[:6]


def calculate_memory_requirement(N, M):
    """
    Calculate minimum memory for Apriori second pass using two methods.
    
    Args:
        N (int): Number of frequent items
        M (int): Number of pairs between frequent items
    
    Returns:
        int: Minimum memory requirement in bytes
    """
    # Method 1: Triangular matrix
    # For N frequent items, we need to store N*(N-1)/2 pairs
    # Each pair needs 4 bytes for the count
    triangular_memory = N * (N - 1) // 2 * 4
    
    # Method 2: Hash table of triples
    # We have M pairs between frequent items + 1,000,000 other pairs
    # Each triple (item1, item2, count) needs 3 * 4 = 12 bytes
    hash_memory = (M + 1_000_000) * 12
    
    # Return the minimum of the two approaches
    return min(triangular_memory, hash_memory)


def test_memory_calculations():
    """
    Test the memory calculation function with given options.
    """
    options = [
        {"label": "Option 1", "N": 100_000, "M": 50_000_000, "S": 5_000_000_000},
        {"label": "Option 2", "N": 40_000, "M": 60_000_000, "S": 3_200_000_000},
        {"label": "Option 3", "N": 50_000, "M": 80_000_000, "S": 1_500_000_000},
        {"label": "Option 4", "N": 100_000, "M": 100_000_000, "S": 1_200_000_000},
    ]
    
    print("Memory calculations:")
    for opt in options:
        calculated = calculate_memory_requirement(opt["N"], opt["M"])
        percentage_diff = abs(calculated - opt["S"]) / opt["S"] * 100
        print(f"{opt['label']}: N={opt['N']:,}, M={opt['M']:,}")
        print(f"  Calculated: {calculated:,} bytes")
        print(f"  Given: {opt['S']:,} bytes") 
        print(f"  Difference: {percentage_diff:.1f}%")
        print()


def test_recommendation_function(basket_encoded, test_item='whole milk'):
    """
    Test the recommendation function with a given item.
    
    Args:
        basket_encoded (pd.DataFrame): Binary encoded basket data
        test_item (str): Item to test recommendations for
    """
    if test_item in basket_encoded.columns:
        recommendations = frequently_bought_together(basket_encoded, test_item)
        print(f"\nRecommendations for '{test_item}':")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {list(rec)}")
    else:
        print(f"'{test_item}' not found in dataset")


def run_complete_analysis(market_basket):
    """
    Run the complete market basket analysis pipeline.
    
    Args:
        market_basket (pd.DataFrame): Original market basket data
    
    Returns:
        dict: Results containing all analysis outputs
    """
    print("=" * 60)
    print("COMPLETE MARKET BASKET ANALYSIS")
    print("=" * 60)
    
    # Step 1: Transform basket
    basket_encoded, basket = transform_basket(market_basket)
    
    print("\n" + "=" * 60)
    
    # Step 2: Run Apriori algorithm
    frequent_itemsets, rules, sorted_rules = run_apriori_algorithm(basket_encoded)
    
    print("\n" + "=" * 60)
    
    # Step 3: Test recommendation function
    test_recommendation_function(basket_encoded)
    
    print("\n" + "=" * 60)
    
    # Step 4: Test memory calculations
    test_memory_calculations()
    
    return {
        'basket_encoded': basket_encoded,
        'basket': basket,
        'frequent_itemsets': frequent_itemsets,
        'rules': rules,
        'sorted_rules': sorted_rules
    }

