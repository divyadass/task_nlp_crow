import numpy as np
import matplotlib.pyplot as plt 

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

import itertools

import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean(doc): # doc is a string of text
    """
    Performs text preprocessing.

    Text-cleaning including punctuation, digit, and stopwords removal. This method is
    used as a callable by other text-representations methods.
    
    Parameters
    ----------
    doc : str
        Raw text document.
        
    Returns
    -------
    doc : str
        Cleaned text document.
    """
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = " ".join([token for token in doc.split() if token not in ENGLISH_STOP_WORDS])
    # remove punctuation and numbers
    return doc

# Function to plot confusion matrix. 
# Ref:http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label',fontsize=15)


def get_confusion_matrix(true_class, pred_class):
    """
    Plots confusion matrix.
    
    Parameters
    ----------
    true_class : pandas.core.series.Series or numpy.ndarray
        Iterable containing true class labels. 
    pred_class : pandas.core.series.Series or numpy.ndarray
        Iterable containing predicted class labels.
    """
    cnf_matrix = confusion_matrix(true_class, pred_class)
    plt.figure(figsize=(8,6))
    plot_confusion_matrix(cnf_matrix, classes=['Not Sarcastic','Sarcastic'],normalize=True,
                          title='Confusion matrix with all features')

                          
## method to print metrics
def print_metrics(true_class, pred_class):
    """
    Prints F1-score, recall, and Precision
    
    Parameters
    ----------
    true_class : pandas.core.series.Series or numpy.ndarray
        Iterable containing true class labels 
    pred_class : pandas.core.series.Series or numpy.ndarray
        Iterable containing predicted class labels
    """
    print("F1 Score: ", f1_score(true_class, pred_class))
    print("Recall: ", recall_score(true_class, pred_class))
    print("Precision: ", precision_score(true_class, pred_class))
  
def take(n, iterable):
    """
    Return first n items of the iterable as a dict.
    
    Parameters
    ----------
    n : int
        Number of element to select from the iterable.
    iterable : dict_items
        Iterable obtained from tokenizer.word_index.items(). Contains list of Tuple containing word and Id. 
    
    Returns
    -------
    dict
        Filtered dictonary items as dictionary.
    """
    return dict(itertools.islice(iterable, n))

  
def get_class(y_pred_prob, threshold=0.5):
    """
    Generates a list of class labels for the given data.
    
    Parameters
    ----------
    y_pred_prob : numpy.ndarray
        Array containing predicted probabilities.
    threshold : float, default : 0.5
        Threshold to be used for classfication.
        
    Returns
    -------
    list    
        List containinig class label the input data.
    """
    y_pred_class = [1 if p > threshold else 0 for p in y_pred_prob]
    return y_pred_class