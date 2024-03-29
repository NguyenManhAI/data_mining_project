# Mô hình LSTM, trước hết biến đổi 1 câu thành embedding

# Xây dựng lớp mô hình LSTM

import torch
import torch.nn as nn


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import torch.optim.lr_scheduler as lr_scheduler
import math

from PrepareData import PrepareData
from Panthera import Panthera

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout = 0.2, numlayers = 1, bidirectional = False):
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


class Tiger:#huấn luyện và predict
  def __init__(self, data, target, model = None,hidden_dim = 128, dropout = 0.0, numlayers = 1, bidirectional = False, batch_size = 4, lr = 0.01, gammar = 0.1):
    if model == None:
      embedding_dim = 384

      output_dim = 1
      model = LSTMModel(embedding_dim, hidden_dim, output_dim, dropout=dropout, numlayers = numlayers, bidirectional = bidirectional)
      self.model = model
    else:
      self.model = model


    preparedata = PrepareData(data, target)


    self.train_loader ,self.validation_loader,self.test_loader = preparedata.getData(batch_size)

    self.max_sequence_length = 128

    self.lr = lr

    self.gammar = gammar

    pass


  def __init__(self, model = None,hidden_dim = 128, dropout = 0.0, numlayers = 1, bidirectional = False, batch_size = 4, lr = 0.01, gammar = 0.1):
    if model == None:
      embedding_dim = 384

      output_dim = 1
      model = LSTMModel(embedding_dim, hidden_dim, output_dim, dropout=dropout, numlayers = numlayers, bidirectional = bidirectional)
      self.model = model
    else:
      self.model = model

    preparedata = PrepareData()


    self.train_loader ,self.validation_loader,self.test_loader = preparedata.getData(batch_size)

    self.max_sequence_length = 128

    self.lr = lr

    self.gammar = gammar

    pass

  def get_model(self):

    return self.model

  def get_train_valid_test(self):
    return self.train_loader ,self.validation_loader, self.test_loader

  def train(self, earlystop_train = False):
    # Khởi tạo mô hình

    # Định nghĩa hàm mất mát và bộ tối ưu hóa
    criterion = nn.BCEWithLogitsLoss()# kết hợp giữa sigmoid và BCE
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gammar)
    # Huấn luyện mô hình
    all_train_loss = []
    all_validation_loss = []
    all_accuracy = []

    best_loss = float('inf')
    early_stop_counter = 0
    patience = 5

    for epoch in range(50):
      self.model.train()
      # self.train_loader.shuffle()

      train_loss = 0

      for inputs, labels in self.train_loader:
          optimizer.zero_grad()
          # print(self.model(inputs).size())
          outputs = self.model(inputs).squeeze()
          # print(outputs.size())
          labels = labels.squeeze()

          loss = criterion(outputs, labels)
          train_loss += loss.item() * inputs.batch_sizes[0]

          loss.backward()
          optimizer.step()

      train_loss /= len(self.train_loader.dataset)
      all_train_loss.append(train_loss)
      # Bước giảm learning rate
      scheduler.step()

      self.model.eval()
      with torch.no_grad():
        correct = 0
        total = 0
        validation_loss = 0
        for inputs, labels in self.validation_loader:
            outputs = self.model(inputs).squeeze()
            labels = labels.squeeze()

            loss = criterion(outputs, labels)
            validation_loss += loss.item() * inputs.batch_sizes[0]
            # print(inputs.batch_sizes[0])

            predicted_labels = (torch.sigmoid(outputs) > 0.5).float()

            try:
              total += labels.size(0)
            except:
              total += 1

            correct += (predicted_labels == labels.float()).sum().item()

        accuracy = correct / total
        validation_loss /= len(self.validation_loader.dataset)

        all_validation_loss.append(validation_loss)
        all_accuracy.append(accuracy)

        if earlystop_train:
          if train_loss < best_loss:
              best_loss = train_loss
              early_stop_counter = 0
          else:
              early_stop_counter += 1
              if early_stop_counter >= patience:
                  break
        else:
          if validation_loss < best_loss:
              best_loss = validation_loss
              early_stop_counter = 0
          else:
              early_stop_counter += 1
              if early_stop_counter >= patience:
                  break

    return all_train_loss, all_validation_loss, all_accuracy



  def test(self):
    # Đánh giá mô hình trên dữ liệu kiểm tra
    self.model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        y_true = []
        y_score = []
        y_pred = []

        for inputs, labels in self.test_loader:

            y_true += labels.tolist()

            outputs = self.model(inputs).squeeze()
            labels = labels.squeeze()

            proba_labels = torch.sigmoid(outputs)
            y_score += proba_labels.tolist()

            predicted_labels = (proba_labels > 0.5).float()
            y_pred += predicted_labels.tolist()

            try:
              total += labels.size(0)
            except:
              total += 1

            correct += (predicted_labels == labels.float()).sum().item()

        accuracy = correct / total


    return accuracy, y_true, y_score, y_pred

  def predict(self,text, check_spelling = False, speed = 1):# một đoạn hoàn chỉnh
    # chuyển về tokenizton sentence
    panthera = Panthera()
    tokens = panthera.execute(text, check_spelling)
    max_sequence_length = math.ceil(len(tokens) / speed)

    indices = random.sample(range(len(tokens)), max_sequence_length)
    indices.sort()
    tokens = [tokens[i] for i in indices]

    new_data = Panthera.text2embedding(tokens)
    new_data = new_data.unsqueeze(0)

    sequence_lengths = [len(seq) for seq in new_data]

    new_data = pack_padded_sequence(new_data, sequence_lengths, batch_first=True)

    # print(new_data.size())

    self.model.eval()
    with torch.no_grad():
        inputs = new_data
        outputs = self.model(inputs)#gọi hàm forward

        proba_label = torch.sigmoid(outputs)
        predict_label = (torch.sigmoid(outputs) > 0.5).float()


    return proba_label.item(), 'positive' if predict_label == 1 else 'negative'
  

