"""
Created on Sun March 24 23:40:21 2021
@author: Harsh Sharma (Student ID: 2019MT60628)
Title: Apriori Algorithm Implementation In Python 3
"""

from llist import DLinkedList
from hashtree import HashTree


class Rule:
    """
    Class for Rules
    """
    def __init__(self, antecedent, consequent, confidence):
        """
        Initialize a rule :- antecedent ---> consequent, conf = confidence
        :param antecedent: frozenset
        :param consequent: frozenset
        :param confidence: float
        """
        self.antecedent = antecedent
        self.consequent = consequent
        self.confidence = confidence

    def __str__(self):
        antecedent = list(self.antecedent)
        consequent = list(self.consequent)
        return "{Rule : " + str(antecedent) + " ---> " + str(consequent) + ", Confidence : " + str(self.confidence) + "}"


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
                keys as frequent 1-itemset which is a python tuple
                values as their support count
            """
    freq_one_itemsets = dict()
    for _, t in transactions.items():
        for item in t:
            s = (item,)
            if s in freq_one_itemsets:
                freq_one_itemsets[s] += 1
            else:
                freq_one_itemsets[s] = 1
    infrequent_items = []
    for item, sup_count in freq_one_itemsets.items():
        if sup_count < n * min_sup:
            infrequent_items.append(item)
    for item in infrequent_items:
        del freq_one_itemsets[item]
    return freq_one_itemsets


# list of itemsets
def _private_can_merge(itemset1, itemset2):
    """Merges two frequent itemsets of size k to construct a candidate k+1 itemset via F_k x F_k method
        Parameters
        -----------
        itemset1 : python tuple, frequent itemset 1
        itemset2 : python tuple, frequent itemset 2
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
    """
    Returns a python list, with elements from ls (in-order), except at ls[idx]
    :param ls: python list
    :param idx: int
    :return: python list
    """
    output_list = []
    n = len(ls)
    for i in range(n):
        if i != idx:
            output_list.append(ls[i])
    return output_list


# generate candidate (k+1)-itemset from freq k-itemsets
# freq_itemsets is a dictionary with keys as tuples and values as their support count
# k is size of freq_itemsets
def apriori_gen(freq_itemsets, k):
    """
    Generate candidate (k+1)-itemsets from frequent k-itemsets
        Parameters
        -----------
        freq_itemsets : python dictionary
            keys as python tuple corresponding to frequent k-itemsets
            values as their support count
        k : int
            size of each frequent itemset
        Returns
        -----------
        DLinkedList
        List of candidate (k+1)-itemsets lists
    """
    candidate_itemsets = DLinkedList()
    # Candidate Generation Step F_{k} x F_{k} method
    # Generate candidate item sets of size k+1
    list_freq_itemsets = list(freq_itemsets.keys())
    n = len(list_freq_itemsets)
    for i in range(n):
        for j in range(i + 1, n):
            l_itemset1 = list_freq_itemsets[i]
            l_itemset2 = list_freq_itemsets[j]
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
        candidate_itemset = candidate_itemsets[j][0]
        for i in range(k - 1):
            aux_itemset = _private_insert_all_except(candidate_itemset, i)
            if tuple(aux_itemset) not in freq_itemsets:
                # candidate_itemset is infrequent
                # we'll prune it immediately
                to_prune.append(j)
                break
    for idx in reversed(to_prune):
        candidate_itemsets.remove(idx)

    return candidate_itemsets


def gen_freq_itemsets(transactions, min_sup=0.5, max_len=None, max_leaf_size=15, max_children=50):
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

        max_leaf_size : int (default : 50)

        max_children : int (default : 100)

        Returns
        -----------
        List of Python Dictionary , denoted as freq_itemsets
            freq_itemsets[k] is a python dictionary :
                key: python tuple, denoting frequent itemset
                value: int, supp_count of the itemset
        """

    # list of all freq_itemset
    # freq_itemset[k] gives all frequent k-itemsets
    n = len(transactions)
    k = 0
    # List of all frequent k-itemsets, for each k >= 1, stored in a python dictionary along with support count
    freq_itemsets = [_private_generate_freq_one_itemsets(transactions, n, min_sup)]
    while True:
        k += 1
        if max_len is not None:
            if k > max_len:
                break
        candidate_itemsets = apriori_gen(freq_itemsets[k - 1], k)
        # returns a DLinkedList of (candidate (k+1)-itemset, supp_count)

        # support-count using hash tree
        hash_tree = HashTree(max_leaf_size=max_leaf_size,
                             max_children=max_children)

        for itemset, supp_count in candidate_itemsets:
            hash_tree.insert(itemset, supp_count, k+1)

        for _, t in transactions.items():
            # (k+1) is the size of candidate (k+1)-itemsets
            w = len(t)
            if w >= k + 1:
                hash_tree.update_support(t, w, k + 1)
        freq_kitemsets = dict()
        hash_tree.all_fitemsets(freq_kitemsets, n * min_sup)
        if len(freq_kitemsets) == 0:
            break
        freq_itemsets.append(freq_kitemsets)

    return freq_itemsets


def ap_genrules(freq_itemsets, k_itemset, k, min_conf, H, m, rules):
    # k = len(freq_itemsets_k)
    # m = len(H)
    if k > m + 1:
        H = apriori_gen(H, m)
        # H is a Doubly linked-list of all candidate m+1-itemset consequent, and supp_count : 0
        # convert H into a dictionary with keys as frozenset, denoting itemsets as lists in H
        # --------------------------- with values as int, denoting support count of corresponding frequent itemset
        _H = dict()
        for itemset, _ in H:
            _itemset = tuple(itemset)
            # _itemset is subset of k_itemset and hence must be in freq_itemsets[m]
            _H[_itemset] = freq_itemsets[m][_itemset]

        to_remove = []
        for h in _H:
            # h is subset of k_itemset
            antecedent = tuple([item for item in k_itemset if item not in h])
            l = len(antecedent)
            conf = freq_itemsets[k - 1][k_itemset] / freq_itemsets[l - 1][antecedent]
            if conf >= min_conf:
                rules.append(Rule(antecedent, h, conf))
            else:
                to_remove.append(h)
        for h in to_remove:
            del _H[h]

        ap_genrules(freq_itemsets, k_itemset, k, min_conf, _H, m+1, rules)


def gen_rules(freq_itemsets, min_conf=0.6):
    """
    Generate all rules from given frequent_itemsets list with conf >= min_conf
    Parameters
    -----------------------
    freq_itemsets: python list of dictionaries
                    freq_itemsets[k] is set of all frequent k-itemsets
                        key : python tuple denoting frequent k-itemsets
                        value : supp_count
    min_conf: python float
        min_conf threshold for the rules
    """
    k = 1
    rules = []
    for freq_itemsets_k in freq_itemsets:
        # freq_itemsets_k set of all frequent k-itemsets
        if k == 1:
            k += 1
            continue
        H = dict()  # set of all rule consequent of size 1 from each frequent-itemset from freq_itemsets_k
        for k_itemset in freq_itemsets_k:
            # k_itemset is a frequent k-itemset
            for item in k_itemset:
                H[(item,)] = freq_itemsets[0][(item,)]
            ap_genrules(freq_itemsets, k_itemset, k, min_conf, H, 1, rules)
            H.clear()
        k += 1
    return rules
