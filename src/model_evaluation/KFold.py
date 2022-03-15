import numpy as np


class KFold:

    def __init__(self, n_split=5, shuffle=False, random_state=None):
        """
        Creates an instance of KFold iterator for pandas data-frame
        :param n_split: number of splits , must be at least 2
        :param shuffle: Whether to shuffle the data before splitting into batches.
                        Note that the samples within each split will not be shuffled.
        :param random_state: When shuffle is True, random_state affects the ordering of the indices,
                             which controls the randomness of each fold.
        """
        self.n_split = n_split
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data_frame):
        """
        Splits the pandas data_frame into self.n_splits for n_splits fold cross validation
        :param data_frame:
        :return: list of pairs of indices of test_idx and train_idx
        """

        n = len(data_frame)
        perm = range(0, n)
        if self.shuffle:
            if self.random_state is not None:
                perm = np.random.RandomState(seed=self.random_state).permutation(perm)
            else:
                perm = np.random.permutation(perm)

        i = 0
        idx_list = []
        split_size = n // self.n_split
        for k in range(0, self.n_split):
            if k == self.n_split - 1:
                split_size = n-i

            test_idx = [j for j in perm[i:i+split_size]]
            train_idx = []
            for j in range(0, n):
                if i <= j < i + split_size:
                    continue
                else:
                    train_idx.append(perm[j])

            idx_list.append((train_idx, test_idx))
            i += split_size
        return idx_list

