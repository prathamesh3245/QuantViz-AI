# from flask import Flask, jsonify
# import torch
# from gnn_fraud_detection import GCN

# app = Flask(__name__)

# model = GCN(in_channels=166, out_channels=2)
# model.load_state_dict(torch.load('model.pth'))

# @app.route('/predict', methods=['POST'])
# def predict():
#     return jsonify({"fraud_probability": 0.94})

# @app.route('/')
# def home():
#     return "QuantViz GNN API - Send POST requests to /predict"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
