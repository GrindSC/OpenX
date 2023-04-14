from flask import Flask, jsonify, request
import tensorflow as tf
import pickle
import numpy as np
from NearestMean import NearestMean

nm_model = pickle.load(open('nm_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
lr_model = pickle.load(open('lr_model.pkl', 'rb'))
nn_model = tf.keras.models.load_model('nn_model.h5', compile=False)

## 6.Create a very simple REST API that will serve your models

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    ## 6.1.Allow users to choose a model
    model_type = request.json['model']
    if model_type == 'heuristic':
        model = nm_model
    elif model_type == 'baseline1':
        model = rf_model
    elif model_type == 'baseline2':
        model = lr_model
    elif model_type == 'nn':
        model = nn_model
    else:
        return jsonify({'error': 'Wrong model type! Choose one of the following: heuristic, baseline1, baseline2, nn'})
    
    ## 6.2.Take all necessary input features and return a prediction
    inputs = request.json['inputs']

    # Load scaler and adjust data for prediction
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    inputs=scaler.transform(inputs)
    if model_type == 'nn':
        prediction = np.argmax(nn_model.predict(inputs,verbose=0),axis=1) + 1
    else:
        prediction = model.predict(inputs)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)