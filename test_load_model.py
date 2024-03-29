import joblib

model = joblib.load('E:/Subject/University/Data_Mining_And_Analysis/data_mining_project/model/sam_model.pkl')

print(model.predictFromComment('i hade it', type='label'))
print(model.predictFromComment('i hade it', type='proba'))