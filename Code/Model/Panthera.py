
import nltk.data
import nltk

from transformers import pipeline# sửa lỗi chính tả
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer # embedding câu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import torch.optim.lr_scheduler as lr_scheduler
import math


# Mô hình tokennizer sentence và sửa lỗi chính tả, embedding câu.
# nltk.download('punkt')
class Panthera:
  def __init__(self):
    # nltk.download('punkt')
    tokenizer = nltk.data.load('Data\english.pickle')
    fix_spelling = pipeline("text2text-generation",model="oliverguhr/spelling-correction-english-base")
    # fix_spelling= pipeline(
    #           'text2text-generation',
    #           'pszemraj/flan-t5-large-grammar-synthesis',
    #           )
    self.tokenize = tokenizer.tokenize
    self.fix_spelling = fix_spelling

  @staticmethod
  def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(Panthera.flatten(item))
        else:
            flat_list.append(item)
    return flat_list

  def execute(self,text, max_depth = 5, check_spelling = True):#1 đoạn văn bản
    # lặp lại cho đến khi không còn sự thay đổi nào
    list_tokens_sent = self.tokenize(text)

    if check_spelling:
      deep = 0
      while(True):
        for i in range(len(list_tokens_sent)):#duyệt list các token sent đã được chia tách, kiểm tra chính tả và sửa
          list_tokens_sent[i] = self.fix_spelling(list_tokens_sent[i], max_length = 2048)[0]['generated_text'] #, max_length = 2048


        # tiếp tục phân tách cho đến khi không thay đổi
        pre_len = len(list_tokens_sent)
        for i in range(len(list_tokens_sent)):
          list_tokens_sent[i] = self.tokenize(list_tokens_sent[i])

        # dãn list_tokens_sent thành 1 mảng chứa các token duy nhất
        list_tokens_sent = Panthera.flatten(list_tokens_sent)
        # kiểm tra thay đổi
        deep += 1
        if pre_len == len(list_tokens_sent) or deep == max_depth:
          break

    return list_tokens_sent
  @staticmethod
  def text2embedding(sentence):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # Sentences are encoded by calling model.encode()
    embedding = model.encode(sentence)

    return torch.Tensor(embedding) if len(embedding.shape) >= 2 else torch.Tensor([embedding])