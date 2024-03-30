# Model LSTM
import torch
from model.LSTMModel import LSTMModel, modelEmbedding, predict

path_best_parameter_model = 'model/best_params_model.pkl'
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load(path_best_parameter_model))

embedding_model = modelEmbedding()

text = 'I like it!'
print(predict(lstm_model, embedding_model, text, language_path='model/english.pickle'))

# Model TFIDF-SVC
import joblib

import sys
sys.path.append('model')

model = joblib.load('model/tfidf_svc_model.pkl')

print(model.predictFromComment('i don\'t like it', type='label'))
print(model.predictFromComment('i don\'t like it', type='proba'))