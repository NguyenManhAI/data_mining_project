# ghép nối Review Title, Review Content
# lấy 2 cột Review và Labels
# Tokenization + check spelling + embedding
# chia thành train, val, test và lưu dưới dạng Dataset

import nltk.data
import nltk
import pandas as pd
from transformers import pipeline# sửa lỗi chính tả
import torch
from torch.utils.data import Dataset, random_split
from sentence_transformers import SentenceTransformer # embedding câu


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
# Mô hình tokennizer sentence và sửa lỗi chính tả, embedding câu.
# nltk.download('punkt')
class Preprocessing:
  def __init__(self, link):
    '''
    link: đường dẫn đến tập dữ liệu 
    '''
    tokenizer = nltk.data.load('model\english.pickle')
    fix_spelling = pipeline("text2text-generation",model="oliverguhr/spelling-correction-english-base")
    self.link = link
    self.tokenize = tokenizer.tokenize
    self.fix_spelling = fix_spelling

  @staticmethod
  def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(Preprocessing.flatten(item))
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
        list_tokens_sent = Preprocessing.flatten(list_tokens_sent)
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
  def preprocess(self):
    df = pd.read_csv(self.link)
    # df = df.head(20)
    df['Review'] = df['Review Title'] + '. ' + df['Review Content']
    df = df[['Review', 'Label']]
    df['Tokens'] = df['Review'].apply(lambda x: self.execute(x))
    df['Embeddings'] = df['Tokens'].apply(lambda x: Preprocessing.text2embedding(x))

    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Negative' else 1)

    dataset = Data(df['Embeddings'], df['Label'])
    trainset, validationset, testset = random_split(dataset, [0.8, 0.1, 0.1])

    return trainset, validationset, testset
    # lưu lại, đã lưu 
if __name__ == "__main__":
   p = Preprocessing('Data-for-Data-Mining-Project - Source.csv')
   trainset, validationset, testset = p.preprocess()

   print(trainset, validationset, testset)
   print(len(trainset), len(testset), len(validationset))
