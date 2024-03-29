import pandas as pd
import numpy as np

import nltk.data
import nltk
import json
import csv
from transformers import pipeline# sửa lỗi chính tả
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer # embedding câu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import torch.optim.lr_scheduler as lr_scheduler
import math

class Data(Dataset):#dữ liệu phải có dạng: một cột là các comment, mỗi comment đã được phân tách thành list các câu, cột còn lại là sentiment positive, negative
  def __init__(self, data, target):
        self.data = data
        self.target = target

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    # print(index)
    x = self.data[index]
    y = self.target[index]
    return x, y

class PrepareData():
  '''
  Đầu vào vẫn là các list thông thường

  có 2 __init__:
  1, Nếu data, target làm đầu vào thì mất thời gian tải lại
  2, Nếu không, đã tải sẵn vào các file trainset.pt validationset.pt testset.pt -> nên dùng
  '''
  def __init__(self, data, target, download = True) -> None:

    dataset = Data(data, target)

    self.trainset, self.validationset, self.testset = random_split(dataset, [0.8, 0.1, 0.1])
  def __init__(self):

    path_trainset = '.\Data\\trainset.pt'
    path_validationset = '.\Data\\validationset.pt'
    path_testset = '.\Data\\testset.pt'

    # tải dữ liệu từ file
    self.trainset = torch.load(path_trainset)
    self.validationset = torch.load(path_validationset)
    self.testset = torch.load(path_testset)


  @staticmethod
  def collate_fn(batch):
    # Sắp xếp các mẫu trong batch theo độ dài chuỗi giảm dần
    batch.sort(key=lambda x: len(x[0]), reverse=True)


    # batch = [(data1, label1), (data2, label2), ...].Bằng cách sử dụng zip(*batch), chúng ta thực hiện unpack các thành phần của batch theo chiều dọc

    # Tách dữ liệu và nhãn từ các mẫu
    data, labels = zip(*batch)
    labels = torch.Tensor(labels)

    # Tạo batch tensor bằng cách padding các sequence trong list
    padded_sequences = nn.utils.rnn.pad_sequence(data, batch_first=True)

    # Đếm độ dài thực tế của mỗi sequence trong batch
    sequence_lengths = [len(seq) for seq in data]


    # Chuyển đổi batch tensor thành PackedSequence
    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(padded_sequences, sequence_lengths, batch_first=True)

    return packed_sequence, labels

  def getData(self, batch_size):

    train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                                shuffle=True,collate_fn=PrepareData.collate_fn, num_workers=0, pin_memory=False)
    validation_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                                shuffle=False,collate_fn=PrepareData.collate_fn, num_workers=0, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(self.validationset, batch_size=batch_size,
                                              shuffle=False,collate_fn=PrepareData.collate_fn, num_workers=0, pin_memory=False)

    return train_loader, validation_loader, test_loader

  