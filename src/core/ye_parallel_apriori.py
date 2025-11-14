from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, FrozenSet
import itertools
import pandas as pd


class TrieNode:
    __slots__ = ("item", "children", "is_end")

    def __init__(self, item: int | None = None):
        self.item = item
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end = False


def build_trie_from_candidates(candidates: List[Tuple[int, ...]]) -> TrieNode:
    root = TrieNode()
    for candidate in candidates:
        node = root
        for item in candidate:
            if item not in node.children:
                node.children[item] = TrieNode(item)
            node = node.children[item]
        node.is_end = True
    return root


def apriori_join_prune(prev_frequents: List[Tuple[int, ...]], k: int, prev_set: set[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """
    Generate candidate k-itemsets from frequent (k-1)-itemsets using Apriori join + prune.
    """
    prev_frequents_sorted = sorted(prev_frequents)
    candidates: List[Tuple[int, ...]] = []
    n_prev = len(prev_frequents_sorted)
    for i in range(n_prev):
        for j in range(i + 1, n_prev):
            a = prev_frequents_sorted[i]
            b = prev_frequents_sorted[j]
            # Join step: first k-2 items must match
            if a[:-1] == b[:-1]:
                merged = tuple(sorted(set(a) | set(b)))
                if len(merged) != k:
                    continue
                # Prune step: all (k-1)-subsets must be frequent
                all_subsets_frequent = True
                for subset in itertools.combinations(merged, k - 1):
                    if tuple(subset) not in prev_set:
                        all_subsets_frequent = False
                        break
                if all_subsets_frequent:
                    candidates.append(merged)
            else:
                break
    return candidates


def _count_chunk(transactions: List[List[int]], trie: TrieNode) -> Dict[Tuple[int, ...], int]:
    counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    for tx in transactions:
        # Ensure transaction is sorted for ordered traversal
        # (Caller should provide sorted transactions; this is a safeguard.)
        if len(tx) > 1 and any(tx[i] > tx[i + 1] for i in range(len(tx) - 1)):
            tx = sorted(tx)

        def dfs(node: TrieNode, start_idx: int, prefix: List[int]):
            i = start_idx
            for item, child in node.children.items():
                # advance i until we match or exhaust
                j = i
                while j < len(tx) and tx[j] < item:
                    j += 1
                if j == len(tx) or tx[j] != item:
                    continue
                new_prefix = prefix + [item]
                if child.is_end:
                    counts[tuple(new_prefix)] += 1
                dfs(child, j + 1, new_prefix)

        dfs(trie, 0, [])
    return counts


def run_ye_parallel_apriori(
    transactions: List[List[int]],
    min_support: float,
    num_workers: int = 4,
    max_k: int = 5
) -> pd.DataFrame:
    """
    Ye (2006) style parallel Apriori:
    - Horizontal transactions
    - Per level: generate candidates, build Trie, parallel rescan of transactions to count supports
    Returns DataFrame with columns ['support', 'itemsets'] compatible with mlxtend.association_rules.
    """
    num_transactions = len(transactions)
    if num_transactions == 0:
        return pd.DataFrame(columns=["support", "itemsets"])

    absolute_min_support = max(1, int(min_support * num_transactions))

    # Level 1: count singletons (ensure uniqueness per transaction)
    singleton_counts = Counter(it for tx in transactions for it in set(tx))
    L1 = sorted([(item,) for item, cnt in singleton_counts.items() if cnt >= absolute_min_support])

    results_by_level: Dict[int, List[Tuple[FrozenSet[int], float]]] = {}
    if L1:
        results_by_level[1] = [(frozenset((item,)), singleton_counts[item] / num_transactions) for (item,) in L1]
    else:
        return pd.DataFrame(columns=["support", "itemsets"])

    # Iterate levels k >= 2
    k = 2
    prev_frequents = L1
    prev_set = set(prev_frequents)

    while prev_frequents and k <= max_k:
        candidates_k = apriori_join_prune(prev_frequents, k, prev_set)
        if not candidates_k:
            break

        trie = build_trie_from_candidates(candidates_k)

        # Parallel count by splitting transactions
        if num_workers and num_workers > 1:
            chunk_size = (num_transactions + num_workers - 1) // num_workers
            chunks = [transactions[i:i + chunk_size] for i in range(0, num_transactions, chunk_size)]
            partial_counts_list: List[Dict[Tuple[int, ...], int]] = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_count_chunk, chunk, trie) for chunk in chunks if chunk]
                for future in as_completed(futures):
                    partial_counts_list.append(future.result())
            counts_k: Dict[Tuple[int, ...], int] = defaultdict(int)
            for local_counts in partial_counts_list:
                for itemset_tuple, cnt in local_counts.items():
                    counts_k[itemset_tuple] += cnt
        else:
            counts_k = _count_chunk(transactions, trie)

        Lk = sorted([tup for tup, cnt in counts_k.items() if cnt >= absolute_min_support])
        if not Lk:
            break

        results_by_level[k] = [(frozenset(t), counts_k[t] / num_transactions) for t in Lk]
        prev_frequents = Lk
        prev_set = set(Lk)
        k += 1

    # Flatten to DataFrame
    flat_records = []
    for level in sorted(results_by_level.keys()):
        flat_records.extend([{"support": support, "itemsets": itemset} for itemset, support in results_by_level[level]])

    if flat_records:
        df = pd.DataFrame(flat_records, columns=["support", "itemsets"])
    else:
        df = pd.DataFrame(columns=["support", "itemsets"])
    return df


