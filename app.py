from flask import Flask, jsonify, request
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import flask
app = Flask(__name__)

max_features = 10000  ## no of unique words in the vocabulary
maxlen = 15 ## no of words to use from each headline
embedding_size = 100 ## length of word embedding

######################### loading the tokenizer object to make a sequence of number from string text
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

######################### loading the trained keras model ############################

json_file = open('model_v1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
clf = model_from_json(loaded_model_json)
# load weights into new model
clf.load_weights("model_v1.h5")

@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()

    tokenized = tokenizer.texts_to_sequences([to_predict_list['review_text']])
    tokenized_pad = pad_sequences(tokenized, maxlen = maxlen, value=0.0)
    
    prob = clf.predict(tokenized_pad)
    
    if prob[0][0]>=0.5:
        prediction = "This text is Sarcastic"
    else:
        prediction = "This text is not Sarcastic"        
    
    return prediction
    
if __name__ == '__main__':
    app.run(debug=True)