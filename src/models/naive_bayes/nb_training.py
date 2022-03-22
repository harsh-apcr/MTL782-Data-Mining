from sklearn.naive_bayes import GaussianNB
import sys
sys.path.append('../../model_evaluation')
from eval_metrics import confusion_matrix, accuracy


# You could choose any number of hyper parameters for tuning
def nb_training(class_vars, X_train, y_train, X_test, y_test, priors=None, var_smoothing=1e-9):
    """
    Hyper-Parameters of GaussianNB
    ----------
    priors : array-like of shape (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.
    """
    model = GaussianNB(priors=priors, var_smoothing=var_smoothing)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # calculate the accuracy
    m_train = confusion_matrix(y_train, y_train_pred,
                               class_vars)

    m_test = confusion_matrix(y_test, y_test_pred,
                              class_vars)

    train_acc = accuracy(m_train)
    test_acc = accuracy(m_test)

    return train_acc, test_acc
