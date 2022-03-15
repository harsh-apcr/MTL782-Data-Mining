import numpy as np

"""
Statistical Scores
"""
from sklearn.metrics import roc_curve


# actual, predicted and class_variables contains categorical variables
# returns a matrix whose (i,j) entry denotes number of records of class i predicted to be of class j
def confusion_matrix(y_true, y_pred, class_vars):
    n = len(class_vars)
    class_idx = {_class: i for _class, i in zip(class_vars, range(0, n))}

    matrix = np.zeros((n, n), dtype='int32')
    for act, pred in zip(y_true, y_pred):
        matrix[class_idx[act], class_idx[pred]] += 1
    return matrix


# need to compute confusion matrix for these statistical scores
def true_positive(matrix, _class, class_vars):
    i = np.where(class_vars == _class)[0][0]
    return matrix[i, i]


def false_positive(matrix, _class, class_vars):
    i = np.where(class_vars == _class)[0][0]
    return matrix[:, i].sum() - matrix[i, i]


def true_negative(matrix, _class, class_vars):
    i = np.where(class_vars == _class)[0][0]
    return matrix.sum() - matrix[:, i] - matrix[i, :] + matrix[i, i]


def false_negative(matrix, _class, class_vars):
    i = np.where(class_vars == _class)[0][0]
    return matrix[i, :].sum() - matrix[i, i]


def accuracy(matrix):
    return np.trace(matrix) / matrix.sum()


def precision(matrix, class_vars):
    tp = 0
    fp = 0
    n = len(class_vars)
    for i in range(n):
        tp += true_positive(matrix, class_vars[i], class_vars)
        fp += false_positive(matrix, class_vars[i], class_vars)
    return tp / (tp + fp)


def recall(matrix, class_vars):
    tp = 0
    fn = 0
    n = len(class_vars)
    for i in range(n):
        tp += true_positive(matrix, class_vars[i], class_vars)
        fn += false_negative(matrix, class_vars[i], class_vars)
    return tp / (tp + fn)


def true_positive_rate(matrix, _class, class_vars):
    tp = true_positive(matrix, _class, class_vars)
    fn = false_negative(matrix, _class, class_vars)
    return tp / (tp + fn)


def false_positive_rate(matrix, _class, class_vars):
    fp = false_positive(matrix, _class, class_vars)
    tn = true_negative(matrix, _class, class_vars)
    return fp / (fp + tn)
