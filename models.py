# import classifiers from sklearn
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


## methods for classification models
def naive_bayes(x_train, y_train, x_test):
    """
    Trains Naive bayes and generates prediciton on training and testing data.

    Trains Multinomial Naive bayes  over training data and generates prediciton over training and testing data.
    
    Parameters
    ----------
    x_train : scipy.sparse.csr.csr_matrix
        Scipy sparse matrix containing training data
    y_train : pandas.core.series.Series
        pandas series containing true label of data
    x_test : scipy.sparse.csr.csr_matrix
        Scipy sparse matrix containing testing data

    Returns
    -------
    y_pred_class_train : numpy.ndarray
        numpy array containing predictions over training data 
    y_pred_class : numpy.ndarray
        numpy array containing predictions over testing data
    """
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    y_pred_class = nb.predict(x_test)
    y_pred_class_train = nb.predict(x_train)
    return y_pred_class_train, y_pred_class

def logistic_regression(x_train, y_train, x_test):
    """
    Trains logistic regression and generates prediciton on training and testing data.

    Trains a logistic regression over training data and generates prediciton over training and testing data.
    
    Parameters
    ----------
    x_train : scipy.sparse.csr.csr_matrix
        Scipy sparse matrix containing training data
    y_train : pandas.core.series.Series
        pandas series containing true label of data
    x_test : scipy.sparse.csr.csr_matrix
        Scipy sparse matrix containing testing data

    Returns
    -------
    y_pred_class_train : numpy.ndarray
        numpy array containing predictions over training data 
    y_pred_class : numpy.ndarray
        numpy array containing predictions over testing data
    """
    logreg = LogisticRegression(class_weight="balanced")
    logreg.fit(x_train, y_train)
    y_pred_class = logreg.predict(x_test)
    y_pred_class_train = logreg.predict(x_train)
    return y_pred_class_train, y_pred_class

def svm(x_train, y_train, x_test):
    """
    Trains SVM and generates prediciton on training and testing data.

    Trains a SVM model over training data and generates prediciton over training and testing data.
    
    Parameters
    ----------
    x_train : scipy.sparse.csr.csr_matrix
        Scipy sparse matrix containing training data
    y_train : pandas.core.series.Series
        pandas series containing true label of data
    x_test : scipy.sparse.csr.csr_matrix
        Scipy sparse matrix containing testing data

    Returns
    -------
    y_pred_class_train : numpy.ndarray
        numpy array containing predictions over training data 
    y_pred_class : numpy.ndarray
        numpy array containing predictions over testing data
    """
    classifier = LinearSVC(class_weight='balanced')
    classifier.fit(x_train, y_train)
    y_pred_class = classifier.predict(x_test)
    y_pred_class_train = classifier.predict(x_train)
    return y_pred_class_train, y_pred_class