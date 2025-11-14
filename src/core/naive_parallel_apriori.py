import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd


def _chunk_indices(indices: List[int], num_chunks: int) -> List[List[int]]:
    if num_chunks <= 1:
        return [indices]
    avg = max(1, len(indices) // num_chunks)
    return [indices[i:i + avg] for i in range(0, len(indices), avg)]


def _support_1_worker(args: Tuple[np.ndarray, List[str], float]) -> List[Tuple[frozenset, float]]:
    cols_block, names_block, min_support = args
    # Boolean matrix; support is mean across rows
    supports = cols_block.mean(axis=0)
    results = []
    for j, s in enumerate(supports.tolist()):
        if s >= min_support:
            results.append((frozenset([names_block[j]]), float(s)))
    return results


def _support_k_worker(args: Tuple[np.ndarray, List[Tuple[int, ...]], List[str], float]) -> List[Tuple[frozenset, float]]:
    arr, cand_index_tuples, col_names, min_support = args
    results = []
    for idx_tuple in cand_index_tuples:
        # AND across selected columns, then mean
        block = arr[:, idx_tuple]
        # .all(axis=1) returns bool vector; mean of bool is fraction of True
        s = block.all(axis=1).mean()
        if s >= min_support:
            itemset = frozenset(col_names[i] for i in idx_tuple)
            results.append((itemset, float(s)))
    return results


def _generate_candidates(prev_itemsets: List[Tuple[str, ...]], k: int, prev_itemset_set: set) -> List[Tuple[str, ...]]:
    """
    Generate candidate (k)-itemsets from frequent (k-1)-itemsets using Apriori join + prune.
    prev_itemsets: sorted tuples of item names of length k-1
    Returns sorted tuples of length k
    """
    candidates = []
    n = len(prev_itemsets)
    for i in range(n):
        for j in range(i + 1, n):
            a = prev_itemsets[i]
            b = prev_itemsets[j]
            # Join step: first k-2 items must match
            if a[:-1] == b[:-1]:
                c = tuple(sorted(set(a) | set(b)))
                if len(c) != k:
                    continue
                # Prune step: all (k-1)-subsets must be frequent
                all_subsets_frequent = True
                for subset in itertools.combinations(c, k - 1):
                    if frozenset(subset) not in prev_itemset_set:
                        all_subsets_frequent = False
                        break
                if all_subsets_frequent:
                    candidates.append(c)
            else:
                break  # because prev_itemsets are sorted lexicographically
    return candidates


def run_naive_parallel_apriori(basket_encoded: pd.DataFrame, min_support: float, num_workers: int = None) -> pd.DataFrame:
    """
    Naïve parallel Apriori:
    - Level 1: split columns evenly across processes and filter by support
    - Levels k>=2: generate candidates, split candidate list across processes, compute support by boolean AND

    Returns a DataFrame with columns: ['support', 'itemsets'] compatible with mlxtend.association_rules.
    """
    if num_workers is None or num_workers < 1:
        try:
            import os
            num_workers = max(1, (os.cpu_count() or 2) - 0)
        except Exception:
            num_workers = 1

    # Ensure boolean numpy matrix for efficient ops
    arr = basket_encoded.values.astype(bool, copy=False)
    col_names = list(basket_encoded.columns)
    num_rows, num_cols = arr.shape

    # Level 1 supports in parallel
    col_indices = list(range(num_cols))
    blocks = _chunk_indices(col_indices, num_workers)
    tasks = []
    results_1: List[Tuple[frozenset, float]] = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for block in blocks:
            if not block:
                continue
            cols_block = arr[:, block]
            names_block = [col_names[i] for i in block]
            tasks.append(ex.submit(_support_1_worker, (cols_block, names_block, min_support)))
        for fut in as_completed(tasks):
            results_1.extend(fut.result())

    # Prepare level 1 frequent itemsets
    frequent_itemsets_records = [
        {"support": s, "itemsets": iset}
        for iset, s in results_1
    ]

    # Iteratively build higher-level itemsets
    k = 2
    # Sorted tuples for join; maintain set for prune
    # For singletons, produce (item_name,) not tuple of characters
    prev_itemsets_sorted = sorted([tuple(sorted(iset)) for iset, _ in results_1])
    prev_set = set(iset for iset, _ in results_1)

    # Map column name to index for quick lookup
    name_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(col_names)}

    while prev_itemsets_sorted:
        candidates = _generate_candidates(prev_itemsets_sorted, k, prev_set)
        if not candidates:
            break

        # Convert candidates to index tuples
        cand_index_tuples: List[Tuple[int, ...]] = [tuple(name_to_idx[name] for name in cand) for cand in candidates]

        # Split candidates across workers
        cand_blocks = _chunk_indices(list(range(len(cand_index_tuples))), num_workers)
        tasks = []
        level_results: List[Tuple[frozenset, float]] = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for block in cand_blocks:
                if not block:
                    continue
                block_cands = [cand_index_tuples[i] for i in block]
                tasks.append(ex.submit(_support_k_worker, (arr, block_cands, col_names, min_support)))
            for fut in as_completed(tasks):
                level_results.extend(fut.result())

        if not level_results:
            break

        frequent_itemsets_records.extend(
            {"support": s, "itemsets": iset} for iset, s in level_results
        )

        # Prepare for next level
        prev_itemsets_sorted = sorted([tuple(sorted(iset)) for iset, _ in level_results])
        prev_set = set(iset for iset, _ in level_results)
        k += 1

    # Build DataFrame in mlxtend format (support first, then itemsets)
    if frequent_itemsets_records:
        df = pd.DataFrame(frequent_itemsets_records)
        # Ensure correct column order
        df = df[["support", "itemsets"]]
    else:
        df = pd.DataFrame(columns=["support", "itemsets"])

    return df


