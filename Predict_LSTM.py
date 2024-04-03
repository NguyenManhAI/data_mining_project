import torch
from torch.nn.utils.rnn import pack_padded_sequence
import random
import math
import nltk.data
import nltk
from sentence_transformers import SentenceTransformer # embedding câu

from model.LSTMModel import LSTMModel
from Preparedata_LSTM import Data

def text2embedding(model_embedding, sentence):
    model = model_embedding
    # Sentences are encoded by calling model.encode()
    embedding = model.encode(sentence)

    return torch.Tensor(embedding) if len(embedding.shape) >= 2 else torch.Tensor([embedding])

model_embedding = SentenceTransformer('paraphrase-MiniLM-L6-v2')
path_best_parameter_model = 'model\\best_params_model.pkl'
model = LSTMModel()
model.load_state_dict(torch.load(path_best_parameter_model))
tokenizer = nltk.data.load('model\\english.pickle')

def predict(text,model = model,model_embedding = model_embedding, speed = 1):
    tokens = tokenizer.tokenize(text)

    max_sequence_length = math.ceil(len(tokens) / speed)

    indices = random.sample(range(len(tokens)), max_sequence_length)
    indices.sort()
    tokens = [tokens[i] for i in indices]

    new_data = text2embedding(model_embedding,tokens)
    new_data = new_data.unsqueeze(0)

    sequence_lengths = [len(seq) for seq in new_data]

    new_data = pack_padded_sequence(new_data, sequence_lengths, batch_first=True)

    model.eval()
    with torch.no_grad():
        inputs = new_data
        outputs = model(inputs)#gọi hàm forward

        proba_label = torch.sigmoid(outputs)
        predict_label = (torch.sigmoid(outputs) > 0.5).float()


    return proba_label.item(), 'positive' if predict_label == 1 else 'negative'
if __name__ == "__main__":
    text = "I love this"
    print(predict(text))