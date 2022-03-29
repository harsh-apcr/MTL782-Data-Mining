"""
Created on Sun March 29 16:67:11 2021
@author: Harsh Sharma (Student ID: 2019MT60628)
Title: FP-Growth Implementation using Python.
"""
from llist import LinkedList


class FPTreeNode:
    """
    class for FP-Tree Node
    """

    def __init__(self, label, supp_count, parent=None, children=None):
        """
        Initialize an FPTree Node, with label, supp_count, parent/children pointers
        :param label: label of the node (any immutable literal)
        :param supp_count: int, support of that item in the transaction set
        :param parent: FPTreeNode, parent pointer in FP-tree
        :param children: List[FPTreeNode], all the children of self
        """
        self.label = label
        self.supp_count = supp_count
        self.parent = parent
        self.children = [] if children is None else children

    def __eq__(self, other):
        return self.label == other.label

    # inserts fp_node as a child of self
    def insert_node(self, fp_node):
        fp_node.parent = self
        self.children.append(fp_node)

    # label is 'label' of the node to check for
    def is_child(self, label):
        for child in self.children:
            if child.label == label:
                return child

    # just for debugging purpose
    def node_inorder_print(self):
        for child in self.children:
            child.node_inorder_print()
        print(self)

    def __str__(self):
        return f'[label : {self.label}, supp_count : {self.supp_count}]'


class FPTree:
    """
    class for FP-Tree
    """

    def __init__(self, null_label=None):
        # Root node is labelled null_label (-1 for our data-set)
        # it is supposed to be unique, and not assumed by any other node in the fp-tree
        self.null_label = null_label
        self.root = FPTreeNode(label=null_label, supp_count=0)
        # list containing additional pointers for each new item added to the FP-tree
        self.header_table = dict()

    def insert(self, transaction, default_count=1):
        """
        Inserts transaction into the FP-tree
        :param transaction: python list, containing all the transactions
        :param default_count: default count of items to be inserted into the tree from transaction
        :return: None
        """
        curr_node = self.root
        for item in transaction:
            child = curr_node.is_child(item)
            if child is None:
                # item is not a child of curr_node
                # we need to insert the remaining items in a path
                fp_node = FPTreeNode(label=item, supp_count=default_count)
                curr_node.insert_node(fp_node)
                if item not in self.header_table:
                    self.header_table[item] = LinkedList()
                self.header_table[item].append(fp_node)
                # inserted node is appended to curr_node.children list
                curr_node = curr_node.children[-1]
                # subsequently, we will get curr_node.is_child(item) as None because curr_node is a single node with
                # empty children list
            else:
                child.supp_count += default_count
                curr_node = child

    # just for debugging purpose
    def inorder_print(self):
        """
        Print Inorder traversal of the FP-tree
        """
        self.root.node_inorder_print()

    @staticmethod
    def remove(fp_node):
        """
        Removes fp_node from the FP-tree that it belongs
        :param fp_node: FPTreeNode, fp_tree_node to be removed
        :return: None
        """
        par_node = fp_node.parent
        par_node.children.remove(fp_node)
        for child in fp_node.children:
            par_node.insert_node(child)

    def conditional_fptree_paths(self, item):
        """
        Returns prefix-path subtree containing item as a suffix
        :param item: int, str (any label)
        :return: (List[List[label]],List[int]), denotes list of paths and their respective support counts
        """
        list = self.header_table[item]
        # LinkedList of same labelled-items
        paths = []
        supp_counts = []
        for node in list:
            path = []
            curr_node = node.val
            isFirst = True
            while curr_node != self.root:
                if isFirst:
                    isFirst = False
                else:
                    path.append(curr_node.label)
                curr_node = curr_node.parent
            paths.append(path)
            supp_counts.append(node.val.supp_count)
        return paths, supp_counts

    def _private_gen_freq_itemsets(self, suffix_itemset, freq_item, min_supp_count, freq_itemsets):
        """
        Generates frequent itemsets with suffix of the itemset : suffix_itemset, from FP-tree : self
        :param suffix_itemset: list of item labels, suffix of the itemset considered thus far
        :param freq_item: item label, we construct conditional fp-tree for [freq_item, *suffix_itemset]
        :param min_supp_count: int, minimum support count threshold
        :return: None
        """
        # declare [freq_item, *suffix_itemset] as frequent
        freq_itemsets.append([freq_item, *suffix_itemset])

        # construct conditional fp-tree of [freq_item, *suffix_itemset]
        paths, supp_counts = self.conditional_fptree_paths(freq_item)
        conditional_fptree = FPTree(self.null_label)
        n = len(paths)
        for i in range(n):
            FPTree.insert(conditional_fptree, reversed(paths[i]), supp_counts[i])
        # delete all those items that are now infrequent
        to_delete = []
        for item, head_list in conditional_fptree.header_table.items():
            supp_count = 0
            for list_node in head_list:
                supp_count += list_node.val.supp_count
            if supp_count < min_supp_count:
                for list_node in head_list:
                    FPTree.remove(list_node.val)
                to_delete.append(item)
        for item in to_delete:
            del conditional_fptree.header_table[item]
        # conditional_fptree is conditional fp-tree for [freq_items[k], *suffix_itemset]

        freq_items = []
        for item in conditional_fptree.header_table:
            freq_items.append(item)
        m_ = len(freq_items)
        # freq_items contains all the frequent items, given conditional_fptree of suffix_itemset as
        # [freq_items, *suffix_itemset]
        for j in range(m_):
            conditional_fptree._private_gen_freq_itemsets([freq_item, *suffix_itemset],
                                                          freq_items[j],
                                                          min_supp_count,
                                                          freq_itemsets)

    # read and test
    def gen_freq_itemsets_fp(self, freq_items, min_supp_count):
        """
        Generates frequent itemsets from FP-tree
        :param freq_items: list of item labels, in increasing order of supp_count
        :param min_supp_count: int, minimum support count threshold
        :return: List of all frequent itemsets
        """
        freq_itemsets = []
        for item in freq_items:
            self._private_gen_freq_itemsets([], item, min_supp_count, freq_itemsets)
        return freq_itemsets


def gen_freq_itemsets(transactions, null_label=None, min_sup=0.5):
    """
    Generates frequent itemsets, along with their support counts
    :param transactions: python dictionary, transaction IDs (keys), contains list of transactions(value)
    :param null_label: label of root node (generally should be something i.e. a label of any item)
    :param min_sup: minimum support threshold
    :return: python dictionary of frequent itemsets (key), along with their support counts(value)
    """
    n = len(transactions)
    freq_one_itemsets = dict()
    # First scan of the data-base to get the support counts of individual items
    for _, t in transactions.items():
        for item in t:
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

    # freq_one_itemsets contains (key=item, value=supp_count) pairs, with only frequent items

    fp_tree = FPTree(null_label=null_label)
    # Second scan of the data-base to construct FP-tree
    for _, transaction in transactions.items():
        _transaction = [item for item in transaction if item in freq_one_itemsets]
        # _transaction must be in decreasing order of support count
        _transaction = sorted(_transaction, key=lambda x: freq_one_itemsets[x], reverse=True)
        fp_tree.insert(_transaction)

    # freq_items are inserted in increasing order of support counts
    return fp_tree.gen_freq_itemsets_fp(list(freq_one_itemsets.keys()),
                                        min_supp_count=min_sup * n)
