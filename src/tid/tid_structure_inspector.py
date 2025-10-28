import pickle
import sys

def load_and_inspect_tid(filepath):
    """Load and display TID structure contents"""
    
    print("\n" + "="*60)
    print("TID Structure Inspector")
    print("="*60)
    print(f"Loading: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            tid = pickle.load(f)
        print("Successfully loaded!")
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return
    except Exception as e:
        print(f"ERROR: Could not load file: {e}")
        return
    
    # Display basic info
    print("\n" + "="*60)
    print("TID Structure Contents")
    print("="*60)
    
    print(f"\nTotal transactions: {tid.total_transactions:,}")
    print(f"Unique items: {len(tid.item_to_tids):,}")
    print(f"Build time: {tid.build_time_seconds:.3f} seconds")
    
    # Get statistics
    if tid.item_to_tids:
        tid_lengths = [len(tids) for tids in tid.item_to_tids.values()]
        
        print(f"\nTID List Statistics:")
        print(f"  Average TID length: {sum(tid_lengths) / len(tid_lengths):.1f}")
        print(f"  Shortest TID length: {min(tid_lengths)}")
        print(f"  Longest TID length: {max(tid_lengths)}")
        
        # Find item with longest TID
        max_item = max(tid.item_to_tids.items(), key=lambda x: len(x[1]))
        print(f"  Most frequent item: Product {max_item[0]} (appears in {len(max_item[1]):,} orders)")
    
    # Show sample of items
    print(f"\n" + "-"*60)
    print("Sample Items (first 10):")
    print("-"*60)
    
    for i, (item, tids) in enumerate(sorted(tid.item_to_tids.items())[:10]):
        sample_tids = sorted(list(tids))[:5]  # First 5 transaction IDs
        more = f"... +{len(tids)-5} more" if len(tids) > 5 else ""
        print(f"Product {item:5d}: {len(tids):6,} orders - TIDs: {sample_tids} {more}")
    
    # Check if itemsets were computed
    if tid.itemset_to_tids:
        print(f"\n" + "-"*60)
        print(f"Cached itemsets: {len(tid.itemset_to_tids):,}")
        print("-"*60)
        
        # Show a few examples
        print("\nSample cached itemsets (first 5):")
        for i, (itemset, tids) in enumerate(list(tid.itemset_to_tids.items())[:5]):
            items = sorted(list(itemset))
            print(f"  {items}: {len(tids)} orders")
    else:
        print(f"\nNo cached itemsets (not yet mined)")


if __name__ == "__main__":
    # Default file
    filepath = "tid_full.pkl"
    
    # Check if file specified on command line
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    load_and_inspect_tid(filepath)