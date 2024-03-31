import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import math
import nltk.data
import nltk
from sentence_transformers import SentenceTransformer # embedding câu

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim = 384, hidden_dim = 128, output_dim = 1, dropout = 0.2, numlayers = 1, bidirectional = False):
        super(LSTMModel, self).__init__()
        self.num_layers = numlayers
        self.D = 2 if bidirectional else 1

        self.hidden_dim = hidden_dim # có giá trị tự do
        self.embedding_dim = embedding_dim # chiều của embedding, vd: [1,2,3,...300]: 1 embedding có kích thước là 300
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first=True, num_layers = numlayers, dropout = dropout, bidirectional=bidirectional) # đầu vào của LSTM là có kích thước embedding và đầu ra có kích thước hidden, xây dựng 1 mô hình LSTM
        self.fc = nn.Linear(self.D*hidden_dim, output_dim) # 1 fully conected để làm đầu ra

    def forward(self, inputs):
        '''
        đầu vào input lần lượt là: batch_size, sequence_length, embedding_dim
        '''
        batch_size = inputs.batch_sizes[0].item() #input ở đây chính là 1 batch mà chúng ta cho vào, và tập huấn luyện của chúng ta chứa những inputs này

        hidden = self.init_hidden(batch_size)
        lstm_out, _ = self.lstm(inputs, hidden)

        # Giải nén lstm_out
        padded_outputs, _ = pad_packed_sequence(lstm_out, batch_first=True)


        padded_outputs = padded_outputs[:, -1, :]

        output = self.fc(padded_outputs)
        return output
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.D * self.num_layers, batch_size, self.hidden_dim), # gồm 1 phần chứa batch_size 'phần', 'phần' chứa hidden_dim units
                torch.zeros(self.D * self.num_layers, batch_size, self.hidden_dim))

def text2embedding(model_embedding, sentence):
    model = model_embedding
    # Sentences are encoded by calling model.encode()
    embedding = model.encode(sentence)

    return torch.Tensor(embedding) if len(embedding.shape) >= 2 else torch.Tensor([embedding])

def modelEmbedding():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def predict(model, model_embedding, text, speed = 1, language_path='english.pickle'):
    tokenizer = nltk.data.load(language_path)
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


if __name__ == '__main__':
    path_best_parameter_model = 'best_params_model.pkl'
    lstm_model = LSTMModel()
    lstm_model.load_state_dict(torch.load(path_best_parameter_model))

    embedding_model = modelEmbedding()

    text = 'I like it!'

    print(predict(lstm_model, embedding_model, text))

