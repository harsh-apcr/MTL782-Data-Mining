"""
DoublyLinkedList as an auxiliary data-structure for efficiency in FP-tree implementation
"""


class Node:
    # val is supposed to be a fp-tree-node
    def __init__(self, val=None, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev


class ListIterator:
    def __init__(self, head, tail):
        self.curr = head.next
        self.stop = tail

    def __next__(self):
        if self.curr == self.stop:
            raise StopIteration
        else:
            curr_node = self.curr
            self.curr = self.curr.next
            return curr_node


class LinkedList:

    def __init__(self):
        head = Node()
        tail = Node()
        head.next = tail
        tail.prev = head
        self.head = head
        self.tail = tail
        self.size = 0

    def append(self, fp_node):
        node = Node(val=fp_node)
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

    def remove_node(self, curr_node):
        prev_node = curr_node.prev
        next_node = curr_node.next
        prev_node.next = next_node
        next_node.prev = prev_node
        curr_node.next = None
        curr_node.prev = None
        self.size -= 1

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
        return curr_node.val

    def __iter__(self):
        return ListIterator(self.head, self.tail)

    def __str__(self):
        curr_node = self.head.next
        str_output = ''
        first = True
        while curr_node != self.tail:
            if first:
                str_output += '['
                first = False
            str_output += str(curr_node.val)
            curr_node = curr_node.next
            if curr_node != self.tail:
                str_output += ','
        str_output += ']'
        return str_output
