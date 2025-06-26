from flask import Flask, jsonify, Response, render_template
import torch
from gnn_fraud_detection import GCN
import streamlit as st
from scripts import function_to_liveData, function_to_cryptoData

app = Flask(__name__)

model = GCN(in_channels=166, out_channels=2)
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({"fraud_probability": 0.94})

@app.route('/')
def home():
    result = function_to_liveData()
    return render_template('index.html', result=result)

@app.route('/live', methods=['POST'])
def live():
    result = function_to_cryptoData()
    # return render_template('streamlit_viz.html', result=result)
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)