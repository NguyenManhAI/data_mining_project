import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim = 384, hidden_dim = 128, output_dim = 1, dropout = 0.2, numlayers = 1, bidirectional = False):
        super(LSTMModel, self).__init__()
        self.num_layers = numlayers
        self.D = 2 if bidirectional else 1

        self.hidden_dim = hidden_dim # có giá trị tự do
        self.embedding_dim = embedding_dim # chiều của embedding, vd: [1,2,3,...300]: 1 embedding có kích thước là 300
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first=True, num_layers = numlayers, dropout = dropout, bidirectional=bidirectional) # đầu vào của LSTM là có kích thước embedding và đầu ra có kích thước hidden, xây dựng 1 mô hình LSTM
        # input_size = embedding_dim; hidden_size, num_layer
        # tuning ở ngay trên
        self.fc = nn.Linear(self.D*hidden_dim, output_dim) # 1 fully conected để làm đầu ra



    def forward(self, inputs):

        '''
        đầu vào input lần lượt là: batch_size, sequence_length, embedding_dim
        '''
        # inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, )
        batch_size = inputs.batch_sizes[0].item()#input ở đây chính là 1 batch mà chúng ta cho vào, và tập huấn luyện của chúng ta chứa những inputs này

        hidden = self.init_hidden(batch_size)
        lstm_out, _ = self.lstm(inputs, hidden)

        # Giải nén lstm_out
        padded_outputs, _ = pad_packed_sequence(lstm_out, batch_first=True)


        padded_outputs = padded_outputs[:, -1, :]

        output = self.fc(padded_outputs)
        return output

    def init_hidden(self, batch_size):
        return (torch.zeros(self.D * self.num_layers, batch_size, self.hidden_dim),# gồm 1 phần chứa batch_size 'phần', 'phần' chứa hidden_dim units
                torch.zeros(self.D * self.num_layers, batch_size, self.hidden_dim))