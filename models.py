# import classifiers from sklearn
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


## methods for classification models
def naive_bayes(x_train, y_train, x_test):
  # Step 3: Train the classifier and predict for test data
  nb = MultinomialNB() # instantiate a Multinomial Naive Bayes model
  nb.fit(x_train, y_train) 
  y_pred_class = nb.predict(x_test) # make class predictions for test data
  x_pred_class = nb.predict(x_train)
  return x_pred_class, y_pred_class

def logistic_regression(x_train, y_train, x_test):
  logreg = LogisticRegression(class_weight="balanced") # instantiate a logistic regression model
  logreg.fit(x_train, y_train) # fit the model with training data
  # Make predictions on test data
  y_pred_class = logreg.predict(x_test)
  x_pred_class = logreg.predict(x_train)
  return x_pred_class, y_pred_class

def svm(x_train, y_train, x_test):
  classifier = LinearSVC(class_weight='balanced') # instantiate a logistic regression model
  classifier.fit(x_train, y_train) # fit the model with training data
  # Make predictions on test data
  y_pred_class = classifier.predict(x_test)
  x_pred_class = classifier.predict(x_train)
  return x_pred_class, y_pred_class