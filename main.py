# Main file for running the project 
# Imports 
import pandas as pd
from src.apriori import (
    transform_basket, 
    run_apriori_algorithm, 
    frequently_bought_together,
    calculate_memory_requirement,
    run_complete_analysis
)


# Load data from public GitHub repository
HYPERMARKET_URL = "https://media.githubusercontent.com/media/dsfsi/dsfsi-datasets/refs/heads/master/data/cos781/hypermarket_dataset.csv"


def run_analysis():
    print("Starting Load Balancing Parallel Apriori...")
    
    # Lets load the data
    # Load the hypermarket dataset into a pandas dataframe named 'market_basket'
    
    # YOUR CODE HERE
    market_basket = pd.read_csv(HYPERMARKET_URL)
    
    # Display basic information about the dataset
    print("Dataset shape:", market_basket.shape)
    print("Columns:", market_basket.columns.tolist())
    
    # Check the first few rows (first 5 rows)
    print("\nFirst 5 rows of the dataset:")
    print(market_basket.head(5))

    # Complete the exploratory data analysis
    
    # Find unique counts for each column
    unique_members = market_basket['Member_number'].nunique()
    unique_items = market_basket['itemDescription'].nunique()
    unique_dates = market_basket['Date'].nunique()

    print(f"Unique members: {unique_members}")
    print(f"Unique items: {unique_items}")
    print(f"Unique dates: {unique_dates}")

    # Check for missing values
    missing_values = market_basket.isnull().sum()

    print("\nMissing values:")
    print(missing_values)

    # Find the most popular items
    top_items = market_basket['itemDescription'].value_counts().head(10)

    print("\nTop 10 most frequent items:")
    print(top_items)
    
    # Run complete analysis using the new functions
    print("\n" + "="*60)
    print("RUNNING COMPLETE ANALYSIS WITH NEW FUNCTIONS")
    print("="*60)
    
    results = run_complete_analysis(market_basket)
    
    return results


def main():
    import os
    import glob
    
    print("Loading CSV data from data directory...")
    
    # Look for CSV files in the data directory
    data_dir = "data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir} directory")
        return
    
    # Load and display information for each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"File: {csv_file}")
        print(f"{'='*50}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"Shape: {df.shape}")
            print(f"Column headers: {df.columns.tolist()}")
            print(f"First 3 rows:")
            print(df.head(3))
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")


if __name__ == "__main__":
    main()  