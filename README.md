Đây là project của nhóm 1 - Môn Khai phá và phân tích dữ liệu.\
\
Đề tài: Phân loại một câu bình luận trên Amazon là khen hay chê.\
Mô tả sơ lược:
- Đầu vào : Một câu bình luận.\
  VD: "Good product! You should buy!"
- Đầu ra  : Một trong số 4 nhãn: Khen, Chê, Bình thường, Vô nghĩa.\
  VD: Nhãn tương ứng với ví dụ trên: "Khen"


## Hướng dẫn sử dụng model LSTM
- Tạo một file .py ở vị trí bất kì.
- Thêm đoạn code dưới đây vào file.
```
import torch
from model.LSTMModel import LSTMModel, modelEmbedding, predict

path_best_parameter_model = 'model/best_params_model.pkl'
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load(path_best_parameter_model))

embedding_model = modelEmbedding()

text = 'I hate it!'
print(predict(lstm_model, embedding_model, text, language_path='model/english.pickle'))

>>> (0.05263785645365715, 'negative')
```