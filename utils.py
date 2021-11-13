import matplotlib.pyplot as plt 

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

import itertools

def clean(doc): # doc is a string of text
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
    Normalization can be applied by setting `normalize=True`.
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


def get_confusion_matrix(y_test, y_pred_class):
    # print the confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(8,6))
    plot_confusion_matrix(cnf_matrix, classes=['Not Sarcastic','Sarcastic'],normalize=True,
                          title='Confusion matrix with all features')

                          
## method to print metrics
def print_metrics(y_test, y_pred_class):
  print("Accuracy: ", accuracy_score(y_test, y_pred_class))
  print("F1 Score: ", f1_score(y_test, y_pred_class))
  # print("AUC: ", roc_auc_score(y_test, y_pred_prob))