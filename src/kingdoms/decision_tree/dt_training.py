from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

def tree_training(max_leaf_nodes, X_train, y_train, X_test, y_test):

    model_tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,
                                        class_weight='balanced')
    model_tree.fit(X_train, y_train)

    y_train_pred = model_tree.predict(X_train)
    y_test_pred = model_tree.predict(X_test)

    # calculate the accuracy

    train_acc = accuracy_score(y_train, y_train_pred, normalize=True)
    test_acc = accuracy_score(y_test, y_test_pred, normalize=True)

    return train_acc, test_acc
