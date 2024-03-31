from model.LSTMModel import LSTMModel
import torch
import torch.nn as nn
from Preparedata_LSTM import PrepareData
import torch.optim.lr_scheduler as lr_scheduler

from Preparedata_LSTM import Data

class TrainTest:#huấn luyện

  def __init__(self,hidden_dim = 128, dropout = 0.0, numlayers = 1, bidirectional = False, batch_size = 4, lr = 0.01, gammar = 0.1):
    embedding_dim = 384

    output_dim = 1
    
    self.model = LSTMModel(embedding_dim, hidden_dim, output_dim, dropout=dropout, numlayers = numlayers, bidirectional = bidirectional)

    preparedata = PrepareData()


    self.train_loader ,self.validation_loader,self.test_loader = preparedata.getData(batch_size)

    self.max_sequence_length = 128

    self.lr = lr

    self.gammar = gammar

    pass

  def set_model(self, model):
     self.model = model

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
  
if __name__ == "__main__":
   traintest = TrainTest()
   print(traintest.train())
   print(traintest.test())