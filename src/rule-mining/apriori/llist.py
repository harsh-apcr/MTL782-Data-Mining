"""
DoublyLinkedList as an auxiliary data-structure for efficiency in Apriori-Algorithm
"""


class ListFullException(Exception):
    pass


class Node:
    # val is supposed to be a python sequence (item-set) (list by default)
    def __init__(self, val=None, supp_count=0, next=None, prev=None):
        if val is None:
            self.val = []
        else:
            self.val = val  # any python sequence
        self.next = next
        self.prev = prev
        self.supp_count = supp_count


class DListIterator:
    def __init__(self, head, tail):
        self.curr = head.next
        self.stop = tail

    def __next__(self):
        if self.curr == self.stop:
            raise StopIteration
        else:
            item = (self.curr.val, self.curr.supp_count)
            self.curr = self.curr.next
            return item


class DLinkedList:

    def __init__(self, max_size=None):
        head = Node()
        tail = Node()
        head.next = tail
        tail.prev = head
        self.head = head
        self.tail = tail
        self.size = 0
        self.max_size = max_size

    def append(self, val, supp_count=0):
        if self.max_size is not None:
            if self.size == self.max_size:
                raise ListFullException()
        node = Node(val=val, supp_count=supp_count, next=None, prev=None)
        node.next = self.tail
        node.prev = self.tail.prev
        self.tail.prev.next = node
        self.tail.prev = node
        self.size += 1

    # *vals are lists
    def append_bulk(self, *vals, supp_count=0):
        if self.max_size is not None:
            if self.size + len(vals) > self.max_size:
                raise ListFullException()
        else:
            for val in vals:
                node = Node(val=val, supp_count=supp_count, next=None, prev=None)
                node.next = self.tail
                node.prev = self.tail.prev
                self.tail.prev.next = node
                self.tail.prev = node
                self.size += 1

    def remove(self, idx):
        if idx < 0 or idx >= self.size:
            raise ValueError('Index out of bound error')

        curr_node = self.head.next
        for i in range(idx):
            curr_node = curr_node.next
        prev_node = curr_node.prev
        next_node = curr_node.next
        prev_node.next = next_node
        next_node.prev = prev_node
        curr_node.next = None
        curr_node.prev = None
        self.size -= 1

    def increment_support_count(self, itemset):
        curr_node = self.head.next
        while curr_node != self.tail:
            if curr_node.val == itemset:
                curr_node.supp_count += 1
                break
            curr_node = curr_node.next

    def __contains__(self, itemset):
        curr_node = self.head.next
        while curr_node != self.tail:
            if curr_node.val == itemset:
                return True
        return False

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < 0:
            raise ValueError('Index out of bound error')
        curr_node = self.head
        for i in range(idx + 1):
            curr_node = curr_node.next
            if curr_node == self.tail:
                raise ValueError('Index out of bound error')
        return curr_node.val, curr_node.supp_count

    def __iter__(self):
        return DListIterator(self.head, self.tail)

    def __str__(self):
        curr_node = self.head.next
        str_output = ''
        first = True
        while curr_node != self.tail:
            if first:
                str_output += '['
                first = False
            str_output += f'(val:{curr_node.val},supp_count:{curr_node.supp_count})'
            curr_node = curr_node.next
            if curr_node != self.tail:
                str_output += ',\n'
        str_output += ']'
        return str_output
