from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('../../model_evaluation')
from eval_metrics import confusion_matrix, accuracy


# You could choose any number of hyper parameters for tuning
def rf_training(class_vars, X_train, y_train, X_test, y_test, max_features, n_estimators, random_state=None):
    """
    Hyper-Parameters of Random Forest
    ----------
    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best split:
    """

    model_forest = RandomForestClassifier(max_features=max_features,
                                          random_state=random_state,
                                          n_estimators=n_estimators,
                                          bootstrap=True,
                                          class_weight='balanced')
    model_forest.fit(X_train, y_train)

    y_train_pred = model_forest.predict(X_train)
    y_test_pred = model_forest.predict(X_test)

    # calculate the accuracy
    m_train = confusion_matrix(y_train, y_train_pred,
                               class_vars)

    m_test = confusion_matrix(y_test, y_test_pred,
                              class_vars)

    train_acc = accuracy(m_train)
    test_acc = accuracy(m_test)

    return train_acc, test_acc
