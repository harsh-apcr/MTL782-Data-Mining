from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append('../../model_evaluation')
from eval_metrics import confusion_matrix, accuracy


def tree_training(max_leaf_nodes, class_vars, X_train, y_train, X_test, y_test):
    model_tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                                        class_weight='balanced')
    model_tree.fit(X_train, y_train)

    y_train_pred = model_tree.predict(X_train)
    y_test_pred = model_tree.predict(X_test)

    # calculate the accuracy
    m_train = confusion_matrix(y_train, y_train_pred,
                               class_vars)

    m_test = confusion_matrix(y_test, y_test_pred,
                              class_vars)

    train_acc = accuracy(m_train)
    test_acc = accuracy(m_test)

    return train_acc, test_acc
