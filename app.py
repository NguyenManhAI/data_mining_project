from flask import Flask, render_template, request, jsonify
import joblib
from flask_cors import CORS
import torch
from model.LSTMModel import LSTMModel, modelEmbedding, predict

path_best_parameter_model = 'model/best_params_model.pkl'
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load(path_best_parameter_model))

embedding_model = modelEmbedding()
text = 'I like it!'
print(predict(lstm_model, embedding_model, text, language_path='model/english.pickle')[0])

import nltk
nltk.download('stopwords')

# print(model.predictFromComment('Almost good, but the wheel is so bad, it stopped just 2 months after buying.'))
# Khởi tạo ứng dụng Flask
app = Flask(__name__)
CORS(app)  # Tạo đối tượng CORS cho ứng dụng Flask

svc = joblib.load('model/tfidf_svc_model.pkl')
print('hello==================================================')
print(svc.predictFromComment('i love it', type='label'))
@app.route('/',methods = ['POST', 'GET'])
def main():
    return 'hello chào bạn'

# Định nghĩa route cho form
@app.route('/svc_model', methods=['POST', 'GET'])
def svc_model():
    data = request.get_json()
    
    if data:
        message = data['message']
        
        label =  (svc.predictFromComment(message, type = 'label')) 
        proba = round(svc.predictFromComment(message, type = 'proba'),2)*100
        proba = round(proba,2) if label == 'positive' else round(100 - proba,2)
        return jsonify({'message': f'{label}','proba': proba, 'label' : f'{label}'})  
    
@app.route('/lstm_model', methods= ['POST', 'GET'])
def lstm():
    data = request.get_json()

    if data:
        message = data['message']
        
        pred = predict(lstm_model, embedding_model, message, language_path='model/english.pickle')
        label = pred[1]
        proba = pred[0]*100
        proba =  round(proba,2) if label == 'positive' else round(100 - proba,2)
        return jsonify({'message' : f'{label}', 'proba' : proba}) 


if __name__ == '__main__':
    # Chạy ứng dụng trên cổng 5000
    app.run(debug=True)