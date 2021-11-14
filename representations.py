# import feature extraction methods from sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

from utils import clean

## methods for WE representation of text data

# Creating a feature vector by averaging all embeddings for all sentences
def embedding_feats(list_of_lists, embeddings):
    """
    Generates word embedding based vector representation of input documents.
    
    Parameters
    ----------
    list_of_lists : list of list containing str
        List of List containing input data.
    embeddings : dict
        Dictionary containing word and corresponding WE vector.
        
    Returns
    -------
    feats : list of array
        Input data represented as a List of array. Array shape being equal to size of embedding chosen.
    """
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
    """
    Generates embedding dictionary.
    
    Uses the Embeddings text file to create a dictionary containing WORD as key and associated
    word embedding vector as value. 
    
    Returns
    -------
    dict:
        Dictionary of word and embedding vector. 
    """
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
def BoW_representation(train, test, max_features=None):
    """
    Generates BoW representation
    
    Uses sklearn CountVectorizer to preprocess and generate BoW representation of text.
    
    Parameters
    ----------
    train : pandas.core.series.Series
        pandas series containing training text data
    test : pandas.core.series.Series
        pandas series containing testing text data
    max_features : int, default 'None'
        Number indicating count of vocab to be used
    
    Returns
    -------
    train_dtm : scipy.sparse.csr.csr_matrix
        train data with BoW representation 
    test_dtm : scipy.sparse.csr.csr_matrix
        test data with BoW representation 
    """  
    vect = CountVectorizer(preprocessor=clean, max_features=max_features) # instantiate a vectoriezer
    train_dtm = vect.fit_transform(train) # use it to extract features from training data
    test_dtm = vect.transform(test) # transform testing data (using training data's features)
    print("train data:", train_dtm.shape, "test data:", test_dtm.shape)
    return train_dtm, test_dtm 

def tf_idf_representation(train, test, max_features=None):
    """
    Generates TF-IDF representation
    
    Uses sklearn TfidfVectorizer to preprocess and generate BoW representation of text.
    
    Parameters
    ----------
    train : pandas.core.series.Series
        pandas series containing training text data
    test : pandas.core.series.Series
        pandas series containing testing text data
    max_features : int, default 'None'
        Number indicating count of vocab to be used
    
    Returns
    -------
    train_tfidf : scipy.sparse.csr.csr_matrix
        Train data with TF-IDF representation. 
    test_tfidf : scipy.sparse.csr.csr_matrix
        Test data with TF-IDF representation.
    """
    vect = TfidfVectorizer(preprocessor=clean, max_features=max_features) # instantiate a vectoriezer
    train_tfidf = vect.fit_transform(train) # use it to extract features from training data
    test_tfidf = vect.transform(test) # transform testing data (using training data's features)
    print("train data:", train_tfidf.shape, "test data:", test_tfidf.shape)
    return train_tfidf, test_tfidf

def word2Vec(train, test):
    """
    Generates word embedding vector for each document in the training and testing corpus.
    
    Parameters
    ----------
    train : pandas.core.series.Series
        Pandas series containing training text data.
    test : pandas.core.series.Series
        Pandas series containing testing text data.
    
    Returns
    -------
    train_we : list of array
        Train data represented as a List of array. Array shape being equal to size of embedding chosen.
    test_we : list of array
        Test data represented as a List of array. Array shape being equal to size of embedding chosen.
    """
    embeddings = make_embedding_dict()
    # embeddings_set  = set(embeddings.keys())

    train_clean = train.reset_index(drop=True).apply(lambda x: clean(x))
    test_clean = test.reset_index(drop=True).apply(lambda x: clean(x))
    train_list=list(train_clean.apply(lambda x: x.split()))
    test_list=list(test_clean.apply(lambda x: x.split()))

    train_we=embedding_feats(train_list, embeddings)
    test_we=embedding_feats(test_list, embeddings)

    return train_we, test_we