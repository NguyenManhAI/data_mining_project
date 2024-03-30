# Đây là project của nhóm 1 - Môn Khai phá và phân tích dữ liệu.

## Đề tài: Phân loại một câu bình luận trên Amazon là khen hay chê.

### Mô tả sơ lược:
- Đầu vào : Một câu bình luận.\
  VD: "Good product! You should buy!"
- Đầu ra  : Một trong số 2 nhãn: Khen, Chê.\
  VD: Nhãn tương ứng với ví dụ trên: "Khen"\

### Hướng dẫn sử dụng model SVC
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
