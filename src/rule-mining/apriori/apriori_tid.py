import itertools

from llist import DLinkedList


class Itemset:
    """
    Class for Itemset
    """
    def __init__(self, generators=None, extensions=None, itemset=None):
        """
        Initialize an itemset for Apriori-TID algorithm
        :param generators: ID, stores ID of later of the two frequent itemsets whose join generated self.itemset
        :param extensions: [ID], stores IDs of all the immediate candidate itemsets that are extension of self.itemset
        :param itemset: python tuple, itemset
        """
        self.generators = generators
        self.extensions = [] if extensions is None else extensions
        self.itemset = () if itemset is None else itemset

    def __eq__(self, other):
        return self.itemset == other.itemset

    def __hash__(self):
        return hash(self.itemset)

    def __len__(self):
        return len(self.itemset)

    def __getitem__(self, idx):
        return self.itemset[idx]

    def __str__(self):
        return str(list(self.itemset))


def _private_generate_freq_one_itemsets(candidate_itemsets_, n, min_sup=0.5):
    """
    Get frequent 1-itemsets from a transaction basket
        Parameters
        -----------
        candidate_itemsets_ : python dictionary with TID as key and set of Itemsets as value
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
            keys as frequent 1-itemset which is an Itemset
            values as their support count
            """
    freq_one_itemsets = dict()
    for _, items in candidate_itemsets_.items():
        for item in items:
            if item in freq_one_itemsets:
                freq_one_itemsets[item] += 1
            else:
                freq_one_itemsets[item] = 1
    infrequent_items = []
    for item, sup_count in freq_one_itemsets.items():
        if sup_count < n * min_sup:
            infrequent_items.append(item)
    for item in infrequent_items:
        del freq_one_itemsets[item]
    return freq_one_itemsets


def _private_can_merge(itemset1, itemset2):
    """Merges two frequent itemsets of size k to construct a candidate k+1 itemset via F_k x F_k method
        Parameters
        -----------
        itemset1 : Itemset , frequent itemset 1
        itemset2 : Itemset , frequent itemset 2
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
    :param ls: any python sequence
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
# freq_itemsets is a dictionary with keys as Itemset and values as their support count (int)
# k is size of freq_itemsets
def apriori_gen(freq_itemsets, k):
    """
    Generate candidate (k+1)-itemsets from frequent k-itemsets
        Parameters
        -----------
        freq_itemsets : python dictionary
            keys as Itemset corresponding to frequent k-itemsets
            values as their support count (int)
        k : int
            size of each frequent itemset
        Returns
        -----------
        DLinkedList
        List of candidate (k+1)-Itemset DLinkedList, with supp_count = 0
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
                    candidate_itemset = Itemset(generators=l_itemset2,
                                                extensions=None,
                                                itemset=tuple(candidate_itemset))
                    candidate_itemsets.append(candidate_itemset)
                    l_itemset1.extensions.append(candidate_itemset)
                else:
                    candidate_itemset = [item for item in l_itemset2]
                    candidate_itemset.append(l_itemset1[k - 1])
                    candidate_itemset = Itemset(generators=l_itemset1,
                                                extensions=None,
                                                itemset=tuple(candidate_itemset))
                    candidate_itemsets.append(candidate_itemset)
                    l_itemset2.extensions.append(candidate_itemset)
    # Candidate Pruning Step
    m = len(candidate_itemsets)
    to_prune = []
    for j in range(m):
        candidate_itemset = candidate_itemsets[j][0]   # An Itemset
        for i in range(k - 1):
            aux_itemset = _private_insert_all_except(candidate_itemset.itemset, i)
            if Itemset(itemset=tuple(aux_itemset)) not in freq_itemsets:
                # candidate_itemset is infrequent
                # we'll prune it immediately
                to_prune.append(j)
                break
    for idx in reversed(to_prune):
        candidate_itemsets.remove(idx)
    return candidate_itemsets


def _private_set_itemsets(transactions):
    dict_items = dict()
    candidate_itemsets = {TID: set() for TID in transactions}
    for TID, itemset in transactions.items():
        for item in itemset:
            if item not in dict_items:
                item_ = Itemset(itemset=(item,))
                dict_items[item] = item_
                candidate_itemsets[TID].add(item_)
            else:
                candidate_itemsets[TID].add(dict_items[item])
    return candidate_itemsets


def gen_freq_itemsets_tid(transactions, min_sup=0.5, max_len=None):
    """
    Get frequent itemsets from a transaction basket
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

        Returns
        -----------
        List of Python Dictionary , denoted as freq_itemsets
            freq_itemsets[k] is a python dictionary :
                key: Itemset , denoting frequent itemset
                value: int, supp_count of the itemset
            """
    # list of all freq_itemset
    # freq_itemset[k] gives all frequent k-itemsets
    n = len(transactions)
    k = 0
    candidate_itemsets_ = _private_set_itemsets(transactions)
    # List of all frequent k-itemsets, for each k >= 1, stored in a python dictionary along with support count
    freq_itemsets = [_private_generate_freq_one_itemsets(candidate_itemsets_, n, min_sup)]
    while True:
        k += 1
        if max_len is not None:
            if k > max_len:
                break
        # candidate k+1-itemsets
        candidate_itemsets = apriori_gen(freq_itemsets[k - 1], k)
        candidate_itemsets__ = dict()
        supp_counts_candidate = dict()

        for TID in candidate_itemsets_:
            candidate_itemsets_t = set()
            set_of_itemsets = candidate_itemsets_[TID]
            for candidate_k_1 in set_of_itemsets:
                if candidate_k_1.extensions is None:
                    continue
                # these extensions point to candidate itemsets in candidate_itemsets DLinkedList
                # candidate_k_1.extensions is not None
                for candidate_k in candidate_k_1.extensions:
                    # candidate_k.generators is not None as it is extension of candidate_k_1
                    id = candidate_k.generators
                    if id in set_of_itemsets:
                        candidate_itemsets_t.add(candidate_k)

            for candidate in candidate_itemsets_t:
                if candidate not in supp_counts_candidate:
                    supp_counts_candidate[candidate] = 1
                else:
                    supp_counts_candidate[candidate] += 1
            candidate_itemsets__[TID] = candidate_itemsets_t

        freq_kitemsets = {itemset: supp_counts_candidate[itemset] for itemset, _ in candidate_itemsets
                          if itemset in supp_counts_candidate and supp_counts_candidate[itemset] >= min_sup * n}
        if len(freq_kitemsets) == 0:
            break
        freq_itemsets.append(freq_kitemsets)
        candidate_itemsets_.clear()
        supp_counts_candidate.clear()
        candidate_itemsets_ = candidate_itemsets__

    return freq_itemsets
