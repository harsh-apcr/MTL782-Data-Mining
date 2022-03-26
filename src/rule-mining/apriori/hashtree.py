import numpy as np
from llist import DLinkedList
import itertools


class IllegalOperationException(Exception):
    pass


class HashTreeNode:
    """
    Class for Hash Tree Node
    """

    def __init__(self, max_children, is_leaf=True):
        """
        Creates an empty hash tree node with specified number of max_leaf_size, max_children
        :param max_children: int , max number of children associated with each hash tree node
        """
        self.is_leaf = is_leaf
        self.children = np.empty((max_children,), dtype='object') if not self.is_leaf else None
        self.values = DLinkedList() if self.is_leaf else None

    def insert_itemset(self, itemset, supp_count):
        if not self.is_leaf:
            raise IllegalOperationException('Cannot insert into an internal node')
        else:
            self.values.append(itemset, supp_count)

    def split_leaf(self, max_children, level):
        if not self.is_leaf:
            raise IllegalOperationException('Cannot split an internal node in a Hash Tree')
        else:
            # now self is an internal node
            self.is_leaf = False
            self.children = np.empty((max_children,), dtype='object')
            for itemset, supp_count in self.values:
                idx = itemset[level] % max_children
                if self.children[idx] is None:
                    self.children[idx] = HashTreeNode(max_children)
                self.children[idx].values.append(itemset, supp_count)

    def update_support(self, prefix_transaction, transaction, w, max_children, k):
        # w = len(transaction)
        if w < k:
            raise IllegalOperationException(f'Cannot update support count of candidate {k}-itemsets if size of '
                                            f'transaction-set : {w} is less than {k}')
        if self.is_leaf:
            if k > 0:
                for rem_items in list(itertools.combinations(transaction, k)):
                    self.values.increment_support_count([*prefix_transaction, *rem_items])
            else:
                self.values.increment_support_count(prefix_transaction)
        else:
            # self is not a leaf node, hence self.children is not None
            for i in range(w - k + 1):
                h = transaction[i] % max_children
                curr_node = self.children[h]
                if curr_node is not None:
                    curr_node.update_support([*prefix_transaction, transaction[i]],
                                             transaction[i + 1:],
                                             w-i-1,
                                             max_children,
                                             k - 1)

    def __str__(self):
        if not self.is_leaf:
            return 'TreeInternalNode'
        else:
            return str(self.values)

    # Stores all the itemsets encountered in the hash-tree into store_itemsets dictionary
    def inorder(self, store_itemsets, minsup_count):
        if self.is_leaf:
            for itemset, supp_count in self.values:
                if supp_count >= minsup_count:
                    store_itemsets[tuple(itemset)] = supp_count
        else:
            for child in self.children:
                if child is not None:
                    child.inorder(store_itemsets, minsup_count)

    # prints inorder traversal of tree (just for debugging purposes)
    def inorder_print(self):
        if self.is_leaf:
            print(self.values)
        else:
            for child in self.children:
                if child is not None:
                    child.inorder_print()
            print('Internal-Node')


class HashTree:

    def __init__(self, max_leaf_size, max_children):
        """
        Creates an empty hash tree with specified number of max_leaf_size, max_children
        :param max_leaf_size: int , max number of values stored in each leaf node
        :param max_children: int , max number of children associated with each hash tree internal node
        """
        self.max_leaf_size = max_leaf_size
        self.max_children = max_children
        # Root node is never a leaf
        self.root = HashTreeNode(max_children, is_leaf=False)

    # itemset is a python list, and its supp_count
    def insert(self, itemset, supp_count, max_depth):
        k = len(itemset)
        curr_node = self.root
        # self.root.children is not None
        for i in range(k):
            h = itemset[i] % self.max_children
            # take care of None pointer exception here curr_node.children could be None

            if curr_node.children is None:
                # curr_node is a leaf node
                # this if-block will never get executed
                pass

            elif curr_node.children[h] is None:
                # whenever a new node is constructed it is always considered to be a leaf node
                curr_node.children[h] = HashTreeNode(max_children=self.max_children)
                curr_node.children[h].insert_itemset(itemset, supp_count)
                break

            elif curr_node.children[h].is_leaf:
                # curr_node is at leaf node
                curr_node = curr_node.children[h]
                if curr_node.values.size < self.max_leaf_size:
                    curr_node.insert_itemset(itemset, supp_count)
                    break
                else:
                    if i + 1 >= max_depth:
                        # if next level is more than or equal to size of itemsets (max_depth of hash-tree)
                        curr_node.insert_itemset(itemset, supp_count)
                        break
                    else:
                        curr_node.split_leaf(self.max_children, i + 1)
                    # now curr_node is no longer a leaf node and instead is an internal node
            else:
                # curr_node is an internal node and curr_node.children[h] is not none and curr_node.children[h] is not leaf
                # curr_node.children[h] is an internal node as well
                curr_node = curr_node.children[h]

    # Update support count of candidate k-itemsets
    # transaction is a python list containing all the transactions
    def update_support(self, transaction, w, k):
        # complete the procedure
        self.root.update_support([], transaction, w, self.max_children, k)

    def all_fitemsets(self, store_itemsets, minsup_count):
        """
        Stores all the itemsets present at the leaves of the hash-tree, into store itemsets
        :param store_itemsets: empty python-dictionary
        :param minsup_count: int
        :return: None
        """
        curr_node = self.root
        curr_node.inorder(store_itemsets, minsup_count)

    def _print_tree(self):
        self.root.inorder_print()
