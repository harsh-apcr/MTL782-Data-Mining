from llist import DLinkedList, Node
from hashtree import HashTree


def _private_generate_freq_one_itemsets(transactions, n, min_sup=0.6):
    """Get frequent 1-itemsets from a transaction basket
            Parameters
            -----------
            transactions : python dictionary with TID as key and transactions as value
              Set of all transactions in the data-set
            n : int
              Number of transactions
            min_sup : float (default: 0.5)
              A float between 0 and 1 for minimum support of the item-sets returned.
              The support is computed as the fraction
              `transactions_where_item(s)_occur / total_transactions`.
            Returns
            -----------
            A dictionary with
                keys as frequent 1-itemset which is a python frozenset
                values as their support count
            """
    freq_one_itemsets = dict()
    for _, t in transactions.items():
        for item in t:
            s = frozenset([item])
            if s in freq_one_itemsets:
                freq_one_itemsets[s] += 1
            else:
                freq_one_itemsets[s] = 1
    list_items = [(s, supp_count) for s, supp_count in freq_one_itemsets.items()]
    for s, sup_count in list_items:
        if sup_count < n * min_sup:
            del freq_one_itemsets[s]
    return freq_one_itemsets


# list of itemsets
def _private_can_merge(itemset1, itemset2):
    """Merges two frequent itemsets of size k to construct a candidate k+1 itemset via F_k x F_k method
        Parameters
        -----------
        itemset1 : frequent itemset 1
        itemset2 : frequent itemset 2
        Returns
        -----------
        True if they could be merged to produce a candidate itemset
        False otherwise
    """
    n = len(itemset1)
    if n != len(itemset2):
        return False
    else:
        for i in range(n):
            if i < n - 1 and itemset1[i] == itemset2[i]:
                continue
            elif i == n - 1 and itemset1[i] != itemset2[i]:
                return True
            else:
                return False


def _private_insert_all_except(ls, idx):
    output_list = []
    n = len(ls)
    for i in range(n):
        if i != idx:
            output_list.append(ls[i])
    return output_list


# generate candidate k-itemset from freq (k-1)-itemsets
# freq_itemsets is a dictionary with keys as frozenset and values as their support count
# k is size of frequent itemsets
def apriori_gen(freq_itemsets, k):
    """
    Generate candidate (k+1)-itemsets from frquent k-1 itemsets
        Parameters
        -----------
        freq_itemsets : python dictionary
            keys as python frozenset corresponding to frequent k-itemsets
            values as their support count
        k : int
            size of each frequent itemset
        Returns
        -----------
        DLinkedList
        List of candidate k+1-itemsets lists
    """
    candidate_itemsets = DLinkedList()
    # Candidate Generation Step F_{k} x F_{k} method
    # Generate candidate item sets of size k+1
    list_freq_itemsets = list(freq_itemsets.keys())
    n = len(list_freq_itemsets)
    for i in range(n):
        for j in range(i+1, n):
            l_itemset1 = list(list_freq_itemsets[i])
            l_itemset2 = list(list_freq_itemsets[j])
            if _private_can_merge(l_itemset1, l_itemset2):
                if l_itemset1[k - 1] <= l_itemset2[k - 1]:
                    candidate_itemset = [item for item in l_itemset1]
                    candidate_itemset.append(l_itemset2[k - 1])
                    candidate_itemsets.append(candidate_itemset)
                else:
                    candidate_itemset = [item for item in l_itemset2]
                    candidate_itemset.append(l_itemset1[k - 1])
                    candidate_itemsets.append(candidate_itemset)
    # Candidate Pruning Step
    m = len(candidate_itemsets)
    to_prune = []
    for j in range(m):
        candidate_itemset = candidate_itemsets[j]
        for i in range(0, k-1):
            aux_itemset = _private_insert_all_except(candidate_itemset, i)
            if frozenset(aux_itemset) not in freq_itemsets:
                # candidate_itemset is infrequent
                # we'll prune it immediately
                to_prune.append(j)
    for idx in reversed(to_prune):
        candidate_itemsets.remove(idx)

    return candidate_itemsets


def generate_freq_itemsets(transactions, min_sup=0.6, max_len=None, max_leaf_size=3, max_children=3):
    """Get frequent itemsets from a transaction basket
        Parameters
        -----------
        transactions : python dictionary with TID as key and transactions as value
          Set of all transactions in the data-set

        min_sup : float (default: 0.5)
          A float between 0 and 1 for minimum support of the item-sets returned.
          The support is computed as the fraction
          `transactions_where_item(s)_occur / total_transactions`.

        max_len : int (default: None)
          Maximum length of the item-sets generated. If `None` (default) all
          possible item-sets lengths (under the apriori condition) are evaluated.

        max_leaf_size : int (default : 3)

        max_children : int (default : 3)

        Returns
        -----------
        pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
          that are >= `min_support` and < than `max_len`
          (if `max_len` is not None).
          Each item-set in the 'item-sets' column is of type `frozenset`,
          which is a Python built-in type that behaves similarly to
          sets except that it is immutable
        """

    # list of all freq_itemset
    # freq_itemset[k] gives all frequent k-itemsets
    n = len(transactions)
    k = 0
    freq_itemsets = [_private_generate_freq_one_itemsets(transactions, n, min_sup)]
    while True:
        k += 1
        if max_len is not None:
            if k > max_len:
                break
        candidate_itemsets = apriori_gen(freq_itemsets[k - 1], k)
        # returns a DLinkedList of (candidate-itemset, supp_count)

        # support-count using hash tree
        hash_tree = HashTree(max_leaf_size=max_leaf_size,
                             max_children=max_children)

        for itemset, supp_count in candidate_itemsets:
            hash_tree.insert(itemset, supp_count)

        for _, t in transactions.items():
            # (k+1) is the size of candidate (k+1)-itemsets
            hash_tree.update_support(t, k+1)
        freq_kitemsets = dict()
        hash_tree.all_fitemsets(freq_kitemsets, n*min_sup)
        if len(freq_kitemsets) == 0:
            break
        freq_itemsets.append(freq_kitemsets)

    return freq_itemsets


