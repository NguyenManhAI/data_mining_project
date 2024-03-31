import joblib
import numpy as np
from ..model import tfidf_svc_model as tsm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import optuna, os

if __name__ == '__main__':
    np.random.seed(42)

    # Load data
    data_train = pd.read_csv('data/data_train.csv')
    data_test = pd.read_csv('data/data_test.csv')
    data_val = pd.read_csv('data/data_val.csv')

    data_train['Label'] = (data_train['Label'] == 'Positive').astype(int)
    data_test['Label'] = (data_test['Label'] == 'Positive').astype(int)
    data_val['Label'] = (data_val['Label'] == 'Positive').astype(int)
    
    # Tunning parameters
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: tsm.objective(trial, data_train, data_test), n_trials=24)

    
    print(f'\nKết quả tốt nhất : {study.best_value}')
    print(f'Bộ tham số của kết quả tốt nhất : {study.best_params}')

    # Chạy lại mô hình với bộ test để thấy các thông số khác của kết quả tốt nhất
    X_train, X_test = tsm.create_X(data_train, data_test, study.best_params['data-type'])
    y_train, y_test = data_train['Label'], data_test['Label']

    model = tsm.TFIDF_SVC_Model(study.best_params['model-type'], study.best_params['kernel'])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('\nChạy lại mô hình với bộ test để thấy các thông số khác của kết quả tốt nhất:')
    print(f'Accuracy : {accuracy_score(y_test, y_pred):.3f}')
    print(f'Recall : {recall_score(y_test, y_pred):.3f}')
    print(f'Precision : {precision_score(y_test, y_pred):.3f}')
    print(f'F1 Score : {f1_score(y_test, y_pred):.3f}')

    # Đánh giá mô hình bằng bộ val
    X_train, X_val = tsm.create_X(data_train, data_val, study.best_params['data-type'])
    y_train, y_val = data_train['Label'], data_val['Label']

    y_pred = model.predict(X_val)

    print('\nĐánh giá mô hình bằng bộ val:')
    print(f'Accuracy : {accuracy_score(y_val, y_pred):.3f}')
    print(f'Recall : {recall_score(y_val, y_pred):.3f}')
    print(f'Precision : {precision_score(y_val, y_pred):.3f}')
    print(f'F1 Score : {f1_score(y_val, y_pred):.3f}')

    # Lưu mô hình lại, để sau dùng
    if not os.path.exists('model'):
        os.makedirs('model')
    joblib.dump(model, "model/tfidf_svc_model.pkl")
    print('\nSave model to model/tfidf_svc_model.pkl')
    print('Success!')