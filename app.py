from flask import Flask, jsonify, request
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import flask
app = Flask(__name__)

maxlen = 15 ## no of words to use from each headline

## loading the tokenizer object to make a sequence of number from string text and applying transformations done on training data
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

## loading the trained keras model - model architecture and weights
json_file = open('model_v1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
clf = model_from_json(loaded_model_json)
# load weights into new model
clf.load_weights("model_v1.h5")

## specifying threshold for classification - obtained/decided during training
threshold=0.35

def get_class(y_pred_prob):
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
    list of str    
        List containinig class label the input data.
    """
    y_pred_class = ["This Headline is Sarcastic." if p > threshold else "This Headline is Not Sarcastic." for p in y_pred_prob]
    return y_pred_class


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Classifies the incoming Text documents using the model trained.

    Returns
    -------
    json of list
        JSON containing list of output.
    """
    to_predict_json = request.get_json(force=True)    
    text_list_list = [sen.split() for sen in to_predict_json['review_text']]

    tokenized = tokenizer.texts_to_sequences(text_list_list)
    tokenized_pad = pad_sequences(tokenized, maxlen = maxlen, value=0.0)
    prob = clf.predict(tokenized_pad).flatten()
    pred_class = get_class(prob)       

    opt = dict()
    opt['output'] = pred_class

    return jsonify(opt)
    
if __name__ == '__main__':
    app.run(debug=True)