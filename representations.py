# import feature extraction methods from sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import numpy as np

## methods for WE representation of text data

# Creating a feature vector by averaging all embeddings for all sentences
def embedding_feats(list_of_lists, embeddings):
    DIMENSION = 100
    zero_vector = np.zeros(DIMENSION)
    feats = []
    embeddings_set  = set(embeddings.keys())
    for tokens in list_of_lists:
        feat_for_this =  np.zeros(DIMENSION)
        count_for_this = 0 + 1e-5 # to avoid divide-by-zero 
        for token in tokens:
            if token in embeddings_set:
                feat_for_this += embeddings[token]
                count_for_this +=1
        if(count_for_this!=0):
            feats.append(feat_for_this/count_for_this) 
        else:
            feats.append(zero_vector)
    return feats

# def extract_from_zip():
#   glove_file = base_path + "glove.6B.zip"
#   #Extract Glove embedding zip file    ###### needed only once
#   from zipfile import ZipFile
#   with ZipFile(glove_file, 'r') as z:
#     z.extractall()

def make_embedding_dict():
  EMBEDDING_FILE = 'data/glove.6B.100d.txt'
  embeddings = {}
  for o in open(EMBEDDING_FILE, encoding="utf8"):
      word = o.split(" ")[0]
      # print(word)
      embd = o.split(" ")[1:]
      embd = np.asarray(embd, dtype='float32')
      # print(embd)
      embeddings[word] = embd
  return embeddings
  
  
## methods for text-representations

def BoW_representation(max_features=None):
  vect = CountVectorizer(preprocessor=clean, max_features=max_features) # instantiate a vectoriezer
  X_train_dtm = vect.fit_transform(X_train) # use it to extract features from training data
  X_test_dtm = vect.transform(X_test) # transform testing data (using training data's features)
  print("train data:", X_train_dtm.shape, "test data:", X_test_dtm.shape)
  return X_train_dtm, X_test_dtm 

def tf_idf_representation(max_features=None):
  vect = TfidfVectorizer(preprocessor=clean, max_features=max_features) # instantiate a vectoriezer
  X_train_tfidf = vect.fit_transform(X_train) # use it to extract features from training data
  X_test_tfidf = vect.transform(X_test) # transform testing data (using training data's features)
  print("train data:", X_train_tfidf.shape, "test data:",X_test_tfidf.shape)
  return X_train_tfidf, X_test_tfidf

def word2Vec():
  embeddings = make_embedding_dict()
  # embeddings_set  = set(embeddings.keys())

  X_train_clean = X_train.reset_index(drop=True).apply(lambda x: clean(x))
  X_test_clean = X_test.reset_index(drop=True).apply(lambda x: clean(x))
  X_train_list=list(X_train_clean.apply(lambda x: x.split()))
  X_test_list=list(X_test_clean.apply(lambda x: x.split()))

  X_train_we=embedding_feats(X_train_list, embeddings)
  X_test_we=embedding_feats(X_test_list, embeddings)

  return X_train_we, X_test_we