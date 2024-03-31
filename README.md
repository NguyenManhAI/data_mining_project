# Đây là project của nhóm 1 - Môn Khai phá và phân tích dữ liệu.

## Đề tài: Phân loại một câu bình luận trên Amazon là khen hay chê.

### Mô tả sơ lược:
- Đầu vào : Một câu bình luận.\
  VD: "Good product! You should buy!"
- Đầu ra  : Một trong số 2 nhãn: Khen, Chê.\
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

## Hướng dẫn sử dụng model SVC
- Trước hết, tạo một file .py ở vị trí bất kì.
- Tìm đường dẫn đến folder chứa file tfidf_svc_model.py; giả sử là bên trong folder model.
- Sau đó, bên trong file đó, viết đoạn code bên dưới:
```
import joblib

import sys
sys.path.append('model')

model = joblib.load('model/tfidf_svc_model.pkl')

comment = 'I think i like it'
model.predictFromComment(comment)

>>> positive
```


## Hướng dẫn sử dụng model User Interface

### 1. Cài đặt Node.js

### 2. Cài đặt môi trường và thư viện cho backend
Chạy lệnh : `pip install -r requirements.txt`
(Các thư viện được liệt kê trong requirements.txt có thể không đủ.)

### 3. Cài đặt yarn:
Chạy câu lệnh npm install -g yarn trong cmd.

### 4. Cài đặt thư viện cho ui
- Mở cmd tại folder ui.
- Nhập lệnh sau và nhấn Enter: `yarn install`

### 5. Sử dụng UI
Mở 2 cmd:
- Một cmd ở bên ngoài ui, tại folder chứa app.py, chạy lệnh `flask run`.
- Một cmd ở bên trong ui, chạy lệnh `npm start`.
