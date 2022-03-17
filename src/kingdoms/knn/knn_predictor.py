from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append('../../model_evaluation')
from eval_metrics import confusion_matrix, accuracy


# vary n_neighbors only from 1 to 5
def tree_training(class_vars, X_train, y_train, X_test, y_test, n_neighbors):
    """
    Hyper-Parameters of KNN
    ----------
    n_neighbors : int
        Number of neighbors to use
    """
    knn_predictor = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_predictor.fit(X_train, y_train)

    y_train_pred = knn_predictor.predict(X_train)
    y_test_pred = knn_predictor.predict(X_test)

    # calculate the accuracy
    m_train = confusion_matrix(y_train, y_train_pred,
                               class_vars)

    m_test = confusion_matrix(y_test, y_test_pred,
                              class_vars)

    train_acc = accuracy(m_train)
    test_acc = accuracy(m_test)

    return train_acc, test_acc