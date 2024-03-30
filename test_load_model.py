import joblib

import sys
sys.path.append('model')

model = joblib.load('model/tfidf_svc_model.pkl')

print(model.predictFromComment('i don\'t like it', type='label'))
print(model.predictFromComment('i don\'t like it', type='proba'))

